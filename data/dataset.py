import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import WeightedRandomSampler
import math
import torch.distributed as dist
from torch.utils.data import Sampler

class DistributedWeightedRandomSampler(Sampler[int]):
    """
    Per-rank weighted sampling with replacement.
    Each rank independently samples `num_samples` indices from the same weight vector,
    using rank-specific RNG seeded by (base_seed, epoch, rank).

    Duplicates across ranks are possible (as with any with-replacement draw), which is fine
    for diffusion training. Call set_epoch(e) each epoch.
    """
    def __init__(self, weights: torch.Tensor, num_samples: int | None = None, replacement: bool = True, base_seed: int = 0):
        assert weights.dim() == 1
        self.weights = weights.double().clone()
        self.replacement = bool(replacement)
        self.world_size = dist.get_world_size() if (dist.is_available() and dist.is_initialized()) else 1
        self.rank = dist.get_rank() if (dist.is_available() and dist.is_initialized()) else 0
        self.num_samples = int(num_samples) if num_samples is not None else math.ceil(len(weights) / self.world_size)
        self.base_seed = int(base_seed)
        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __iter__(self):
        # rank-specific generator (keeps streams different across (epoch, rank))
        gen = torch.Generator(device="cpu")
        gen.manual_seed(self.base_seed + 12345 * self.epoch + self.rank)
        # multinomial supports replacement draws with weights
        idx = torch.multinomial(self.weights, self.num_samples, replacement=self.replacement, generator=gen)
        return iter(idx.tolist())

    def __len__(self) -> int:
        return self.num_samples

# ---------- BSP / KD-style leaves (1D or multi-D) ----------
def _reduce_condition_maps_to_scalar(cond_tm_1hw, reduce="mean"):
    """
    cond_tm_1hw: (T, M, 1, H, W) -> (T*M,) scalar per (t,m)
    """
    return _reduce_condition_maps(cond_tm_1hw, reduce=reduce)  # already defined above

def _cond_single_to_scalar(cond_n1hw, reduce="mean"):
    """
    cond_n1hw: (N, 1, H, W) -> (N,)
    """
    assert cond_n1hw.ndim == 4 and cond_n1hw.shape[1] == 1, f"Expected (N,1,H,W), got {cond_n1hw.shape}"
    if reduce == "mean":
        scalars = cond_n1hw.mean(axis=(1, 2, 3))
    elif reduce == "median":
        scalars = np.median(cond_n1hw, axis=(1, 2, 3))
    else:
        raise ValueError(f"Unknown reduce='{reduce}'")
    return scalars.astype(np.float64)

def _to_features(x, extra_features=None):
    """
    x: (N,) or (N,1) base feature (e.g., cumulative CO2)
    extra_features: optional list of (N,) arrays for multi-D BSP (e.g., GW level, month)
    returns X shape (N, D)
    """
    x = np.asarray(x).reshape(-1, 1).astype(np.float64)
    if extra_features is not None and len(extra_features):
        cols = [np.asarray(z).reshape(-1, 1).astype(np.float64) for z in extra_features]
        x = np.concatenate([x] + cols, axis=1)
    return x

def build_bsp_leaves(X, target_leaves=150, min_leaf=5):
    """
    Generic KD-style BSP:
      - Repeatedly split the largest leaf on the feature (column) with largest variance
      - Split at the median along that feature
    X: (N, D) numpy float64
    Returns: list of index arrays (leaves).
    """
    N, D = X.shape
    leaves = [np.arange(N)]
    while len(leaves) < target_leaves:
        # choose largest leaf to split
        sizes = np.array([len(ix) for ix in leaves])
        j = int(np.argmax(sizes))
        idx = leaves[j]
        if len(idx) < 2 * min_leaf:
            # try next largest that can be split
            candidates = np.argsort(-sizes)
            split_done = False
            for j2 in candidates:
                idx2 = leaves[j2]
                if len(idx2) < 2 * min_leaf:
                    continue
                # pick dim with largest variance in this leaf
                d = int(np.argmax(np.var(X[idx2, :], axis=0)))
                med = np.median(X[idx2, d])
                left = idx2[X[idx2, d] <= med]
                right = idx2[X[idx2, d] >  med]
                if len(left) >= min_leaf and len(right) >= min_leaf:
                    leaves.pop(j2)
                    leaves.extend([left, right])
                    split_done = True
                    break
            if not split_done:
                break
        else:
            # pick dim with largest variance in this leaf
            d = int(np.argmax(np.var(X[idx, :], axis=0)))
            med = np.median(X[idx, d])
            left = idx[X[idx, d] <= med]
            right = idx[X[idx, d] >  med]
            if len(left) >= min_leaf and len(right) >= min_leaf:
                leaves.pop(j)
                leaves.extend([left, right])
            else:
                break
    return leaves

def _weights_from_leaves(N, leaves):
    """Return per-sample weights w[i] = 1 / size(leaf(i))."""
    leaf_sizes = np.zeros(N, dtype=np.int64)
    for L in leaves:
        leaf_sizes[L] = len(L)
    w = (1.0 / np.maximum(leaf_sizes, 1)).astype(np.float64)
    return torch.from_numpy(w).double()

def _make_sampler_from_weights(weights, epoch_len=None, replace=True, distributed=False, base_seed=0):
    N = len(weights)
    if epoch_len is None:
        epoch_len = N
    if distributed:
        return DistributedWeightedRandomSampler(weights=weights, num_samples=epoch_len,
                                               replacement=replace, base_seed=base_seed)
    else:
        return WeightedRandomSampler(weights=weights, num_samples=epoch_len, replacement=replace)

def make_bsp_sampler_all_members(cond_tm_1hw, target_leaves=150, min_leaf=5,
                                 epoch_len=None, replace=True, reduce="mean",
                                 distributed=False, extra_features=None, base_seed=0):
    """
    Adaptive BSP sampler for AllMembersDataset.
    cond_tm_1hw: (T, M, 1, H, W)
    extra_features: list of length-T*M arrays (or None) to form multi-D splits.
    """
    x = _reduce_condition_maps_to_scalar(cond_tm_1hw, reduce=reduce)  # (T*M,)
    X = _to_features(x, extra_features=extra_features)                # (T*M, D)
    leaves = build_bsp_leaves(X, target_leaves=target_leaves, min_leaf=min_leaf)
    weights = _weights_from_leaves(len(x), leaves)
    sampler = _make_sampler_from_weights(weights, epoch_len=epoch_len, replace=replace,
                                         distributed=distributed, base_seed=base_seed)
    return sampler, dict(leaves=leaves, features="scalar+" + ("extra" if extra_features else "none"))

def make_bsp_sampler_single_member(cond_n1hw, target_leaves=150, min_leaf=5,
                                   epoch_len=None, replace=True, reduce="mean",
                                   distributed=False, extra_features=None, base_seed=0):
    """
    Adaptive BSP sampler for SingleMemberDataset.
    cond_n1hw: (N, 1, H, W)
    """
    x = _cond_single_to_scalar(cond_n1hw, reduce=reduce)  # (N,)
    X = _to_features(x, extra_features=extra_features)    # (N, D)
    leaves = build_bsp_leaves(X, target_leaves=target_leaves, min_leaf=min_leaf)
    weights = _weights_from_leaves(len(x), leaves)
    sampler = _make_sampler_from_weights(weights, epoch_len=epoch_len, replace=replace,
                                         distributed=distributed, base_seed=base_seed)
    return sampler, dict(leaves=leaves, features="scalar+" + ("extra" if extra_features else "none"))


# ---------- utilities ----------
def _flatten_tm(T, M):
    # Returns array of (t,m) -> flat idx, and inverse maps
    flat = np.arange(T * M, dtype=np.int64)
    t = flat // M
    m = flat % M
    return flat, t, m

def _reduce_condition_maps(cond_tm_1hw, reduce="mean"):
    """
    cond_tm_1hw: np.ndarray shape (T, M, 1, H, W)
    Returns: 1D array of length T*M with a scalar per (t,m).
    """
    assert cond_tm_1hw.ndim == 5 and cond_tm_1hw.shape[2] == 1, f"Expected (T,M,1,H,W), got {cond_tm_1hw.shape}"
    if reduce == "mean":
        scalars = cond_tm_1hw.mean(axis=(2, 3, 4))  # -> (T,M)
    elif reduce == "median":
        scalars = np.median(cond_tm_1hw, axis=(2, 3, 4))  # -> (T,M)
    else:
        raise ValueError(f"Unknown reduce='{reduce}'")
    return scalars.reshape(-1).astype(np.float64)  # -> (T*M,)

# ---------- Equal-width bins (simple) ----------
def make_equalwidth_bin_sampler_all_members(cond_tm_1hw, n_bins=150, epoch_len=None, replace=True, reduce="mean"):
    """
    Balanced sampling over fixed edges between min..max.
    """
    x = _reduce_condition_maps(cond_tm_1hw, reduce=reduce)   # (T*M,)
    edges = np.linspace(x.min(), x.max(), n_bins + 1, dtype=np.float64)
    # digitize to 0..n_bins-1
    bins = np.clip(np.digitize(x, edges[:-1], right=False) - 1, 0, n_bins - 1)
    counts = np.bincount(bins, minlength=n_bins)
    inv_counts = np.zeros_like(counts, dtype=np.float64)
    nonzero = counts > 0
    inv_counts[nonzero] = 1.0 / counts[nonzero]
    weights = inv_counts[bins]                                         # (T*M,)
    weights = torch.from_numpy(weights).double()
    if epoch_len is None:
        epoch_len = len(x)
    sampler = WeightedRandomSampler(weights=weights, num_samples=epoch_len, replacement=replace)
    return sampler, dict(edges=edges, counts=counts, bins=bins)

# ---------- BSP / KD-style leaves (adaptive) ----------
def _to_2d_features_1d(x1d):
    return np.asarray(x1d)[:, None].astype(np.float64)

def build_bsp_leaves_from_1d(x1d, target_leaves=150, min_leaf=5):
    """
    KD/BSP-like, but for 1D feature: repeatedly split the largest leaf at its median.
    Returns: list of index arrays (leaves).
    """
    X = _to_2d_features_1d(x1d)  # (N,1)
    N = X.shape[0]
    leaves = [np.arange(N)]
    while len(leaves) < target_leaves:
        sizes = np.array([len(ix) for ix in leaves])
        order = np.argsort(-sizes)  # largest first
        split = False
        for j in order:
            idx = leaves[j]
            if len(idx) < 2 * min_leaf:
                continue
            med = np.median(X[idx, 0])
            left = idx[X[idx, 0] <= med]
            right = idx[X[idx, 0] >  med]
            if len(left) >= min_leaf and len(right) >= min_leaf:
                leaves.pop(j)
                leaves.append(left)
                leaves.append(right)
                split = True
                break
        if not split:
            break
    return leaves



class AllMembersDataset(torch.utils.data.Dataset):
    """
    cond_np: (T, M, 1, H, W)
    tgt_np:  (T, M, 1, H, W)
    time_ids: (T,)
    """
    def __init__(self, cond_np, tgt_np, time_ids=None):
        assert cond_np.shape[:2] == tgt_np.shape[:2], "T and M must match for cond and target"
        self.cond = cond_np
        self.tgt = tgt_np
        self.time_ids = time_ids
        self.T, self.M = cond_np.shape[:2]

    def __len__(self):
        return self.T * self.M

    def __getitem__(self, idx):
        import torch
        t = idx // self.M
        m = idx % self.M
        cond = torch.from_numpy(self.cond[t, m])   # (1,H,W)
        x0   = torch.from_numpy(self.tgt[t, m])    # (1,H,W)
        if self.time_ids is not None:
            year = torch.tensor(self.time_ids[t], dtype=torch.long)
            return cond, x0, year
        return cond, x0
        
class SingleMemberDataset(Dataset):
    """
    Returns:
      cond:  (1,H,W)
      x0:    (1,H,W)  # one randomly chosen member from (M,H,W)
    """
    def __init__(self, cond_arr: np.ndarray, target_arr: np.ndarray,
                 member_mode: str = "random", fixed_member: int = 0):
        assert cond_arr.ndim == 4 and cond_arr.shape[1] == 1, f"cond_arr shape {cond_arr.shape} expected (N,1,H,W)"
        assert target_arr.ndim == 4, f"target_arr shape {target_arr.shape} expected (N,M,H,W)"
        self.cond = cond_arr.astype(np.float32)
        self.tgt  = target_arr.astype(np.float32)
        self.member_mode = member_mode
        self.fixed_member = fixed_member

    def __len__(self):
        return self.cond.shape[0]

    def __getitem__(self, idx):
        cond = torch.from_numpy(self.cond[idx])           # (1,H,W)
        members = torch.from_numpy(self.tgt[idx])         # (M,H,W)
        if self.member_mode == "fixed":
            k = int(self.fixed_member)
        else:
            k = torch.randint(0, members.shape[0], (1,)).item()
        x0 = members[k:k+1, ...]                          # (1,H,W)
        return cond, x0
