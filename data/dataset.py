import os
import glob
from typing import Sequence, Optional, Tuple, List

import numpy as np
import torch
from torch.utils.data import Dataset
import xarray as xr


class PairedDataset(Dataset):
    """
    Conditional dataset for arrays shaped like:
        (year, member_id, lat, lon)
    Returns one sample per (year, member) pair:
        cond: (C_cond, H, W)
        x0:   (C_tgt,  H, W)
        years: tensor([year_value, member_index])  # configurable

    Supports:
      • same-folder (data_root) OR separate roots (target_root, condition_root)
      • arbitrary variable sets for targets / conditions
      • configurable dim names
    """

    def __init__(
        self,
        data_root: Optional[str] = None,
        target_root: Optional[str] = None,
        condition_root: Optional[str] = None,
        file_pattern: str = "*.nc",

        # variables
        target_variables: Sequence[str] = ("tas",),
        condition_variables: Sequence[str] = ("ghg", "aerosol", "landuse", "year_index"),

        # dims
        year_dim: str = "year",
        member_dim: str = "member_id",
        lat_dim: str = "lat",
        lon_dim: str = "lon",

        # options
        normalize_targets: bool = True,
        return_member_in_years: bool = True,   # years -> [year_value, member_idx]
        chunks: Optional[dict] = None,         # e.g., {"year": 1, "member_id": 1, "lat": 192, "lon": 288}
        engine: Optional[str] = None,          # "netcdf4" or "h5netcdf"
        split: str = "train",
    ):
        super().__init__()

        # Resolve roots
        if data_root is not None:
            self.target_root = self.condition_root = data_root
        else:
            if target_root is None or condition_root is None:
                raise ValueError("Provide either data_root OR (target_root and condition_root).")
            self.target_root = target_root
            self.condition_root = condition_root

        self.file_pattern = file_pattern
        self.target_variables = list(target_variables)
        self.condition_variables = list(condition_variables)

        # dims
        self.year_dim = year_dim
        self.member_dim = member_dim
        self.lat_dim = lat_dim
        self.lon_dim = lon_dim

        self.normalize_targets = normalize_targets
        self.return_member_in_years = return_member_in_years
        self.split = split

        # defaults for chunks
        if chunks is None:
            chunks = {year_dim: 1, member_dim: 1, lat_dim: 192, lon_dim: 288}
        self.chunks = chunks
        self.engine = engine

        # Collect files
        t_files = sorted(glob.glob(os.path.join(self.target_root, file_pattern)))
        if not t_files:
            raise FileNotFoundError(f"No target files matching {file_pattern} under {self.target_root}")
        if self.condition_root == self.target_root:
            c_files = t_files
        else:
            c_files = sorted(glob.glob(os.path.join(self.condition_root, file_pattern)))
            if not c_files:
                raise FileNotFoundError(f"No condition files matching {file_pattern} under {self.condition_root}")
            # align by basename intersection
            t_map = {os.path.basename(p): p for p in t_files}
            c_map = {os.path.basename(p): p for p in c_files}
            common = sorted(set(t_map) & set(c_map))
            if not common:
                raise RuntimeError("No matching target/condition basenames between folders.")
            t_files = [t_map[b] for b in common]
            c_files = [c_map[b] for b in common]

        # Open multi-file datasets lazily with chunks
        self.ds_t = xr.open_mfdataset(
            t_files, combine="by_coords", chunks=self.chunks, engine=self.engine
        )
        self.ds_c = self.ds_t if self.condition_root == self.target_root else xr.open_mfdataset(
            c_files, combine="by_coords", chunks=self.chunks, engine=self.engine
        )

        # Validate required dims/vars exist
        for d in (self.year_dim, self.member_dim, self.lat_dim, self.lon_dim):
            if d not in self.ds_t.dims:
                raise KeyError(f"Target dataset missing required dimension '{d}'")
        for v in self.target_variables:
            if v not in self.ds_t:
                raise KeyError(f"Target variable '{v}' not found in target dataset")
        for v in self.condition_variables:
            if v not in self.ds_c:
                raise KeyError(f"Condition variable '{v}' not found in condition dataset")

        # Build index over all (year, member) pairs
        self.year_vals = self.ds_t[self.year_dim].values
        self.member_vals = self.ds_t[self.member_dim].values
        self._pairs: List[Tuple[int, int]] = [
            (iy, im) for iy in range(self.year_vals.shape[0]) for im in range(self.member_vals.shape[0])
        ]

        # Precompute normalization stats for targets (per-variable global mean/std)
        self._norm = None
        if self.normalize_targets:
            self._norm = self._compute_target_norm()

        # Spatial sizes (for info)
        self.H = int(self.ds_t.dims[self.lat_dim])
        self.W = int(self.ds_t.dims[self.lon_dim])

        print(
            f"[PairedDataset:{self.split}] pairs={len(self._pairs)}  "
            f"shape=({self.year_dim}:{len(self.year_vals)}, {self.member_dim}:{len(self.member_vals)}, "
            f"{self.lat_dim}:{self.H}, {self.lon_dim}:{self.W})  "
            f"targets={self.target_variables} cond={self.condition_variables}"
        )

    def _compute_target_norm(self):
        # Compute mean/std over (year, member, lat, lon) per target var (lazy-friendly)
        means, stds = [], []
        for v in self.target_variables:
            arr = self.ds_t[v]
            m = float(arr.mean().compute().item())
            s = float(arr.std().compute().item())
            if s == 0.0:
                s = 1e-6
            means.append(m)
            stds.append(s)
        return {
            "mean": np.array(means, dtype="float32"),
            "std": np.array(stds, dtype="float32"),
        }

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, idx: int):
        iy, im = self._pairs[idx]
        # Select a single (year, member) slice
        sel_t = self.ds_t.isel({self.year_dim: iy, self.member_dim: im})
        sel_c = self.ds_c.isel({self.year_dim: iy, self.member_dim: im})

        # Stack variables -> channel-first (C, H, W)
        x0_list = [sel_t[v].astype("float32").values for v in self.target_variables]
        cond_list = [sel_c[v].astype("float32").values for v in self.condition_variables]

        x0 = np.stack(x0_list, axis=0)    # (C_tgt, H, W)
        cond = np.stack(cond_list, axis=0)  # (C_cond, H, W)

        # Normalize targets per-variable (broadcast over H,W)
        if self._norm is not None:
            mean = self._norm["mean"][:, None, None]
            std = self._norm["std"][:, None, None]
            x0 = (x0 - mean) / std

        x0 = torch.from_numpy(x0)
        cond = torch.from_numpy(cond)

        # years tensor: [year_value, member_index] or just [year_value]
        if self.return_member_in_years:
            year_val = float(np.array(self.year_vals[iy]).item())
            years = torch.tensor([year_val, float(im)], dtype=torch.float32)
        else:
            year_val = float(np.array(self.year_vals[iy]).item())
            years = torch.tensor([year_val], dtype=torch.float32)

        return cond, x0, years

