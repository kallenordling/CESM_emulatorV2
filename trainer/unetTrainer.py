# trainers/unet_trainer.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any

import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.logging import get_logger
from diffusers import DDPMScheduler

try:
    import torch._dynamo  # optional; can help disable graph breaks on debug
except Exception:
    torch = torch

LOG = get_logger(__name__, log_level="INFO")


def _unwrap(model: nn.Module) -> nn.Module:
    """Get the underlying model (unwrap DDP / accelerate wrappers)."""
    base = model
    while hasattr(base, "module"):
        base = base.module
    return base


@dataclass
class TrainerConfig:
    # dataloader
    batch_size: int = 8
    num_workers: int = 4
    pin_memory: bool = True

    # optimization
    max_epochs: int = 100
    lr: float = 2e-4
    betas: tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    use_amp: bool = True

    # sampling/preview
    preview_every: int = 10
    preview_batch: int = 2

    # checkpointing
    ckpt_every: int = 10
    keep_last: int = 5
    ckpt_dir: str = "checkpoints"

    # loss knobs passed through to the model (if supported)
    mean_anchor_beta: float = 0.1
    cond_loss_scaling: float = 0.1


class UNetTrainer:
    """
    Trainer that:
      • builds loaders
      • computes loss the SAME WAY as your train_new_loss.py:
          - uses model.loss_components(...) if available
          - else falls back to model.loss(...)
        and logs 'mse_raw', 'mse_lat', 'cond_loss', and 'total'.
      • supports mixed precision, grad clipping, ckpt save, and simple previews.
    """

    def __init__(
        self,
        train_set,
        val_set,
        *,
        model: nn.Module,
        accelerator: Accelerator,
        scheduler: Optional[DDPMScheduler] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        cfg: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.accelerator = accelerator
        self.model = model
        self.scheduler = scheduler  # (DDPM noise scheduler, if you use it during sampling)
        self.cfg = self._cfg_from_dict(cfg)

        # Forward loss knobs to the underlying diffusion module if it exposes them
        m = _unwrap(self.model)
        # These attributes mirror what you set in train_new_loss.py
        if hasattr(m, "mean_anchor_beta"):
            m.mean_anchor_beta = float(self.cfg.mean_anchor_beta)
        else:
            try:
                setattr(m, "mean_anchor_beta", float(self.cfg.mean_anchor_beta))
            except Exception:
                pass
        try:
            m._cfg_cond_loss_scaling = float(self.cfg.cond_loss_scaling)
        except Exception:
            pass

        self.train_loader = DataLoader(
            train_set,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
            drop_last=False,
        )

        self.val_loader = DataLoader(
            val_set,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
            drop_last=False,
        )

        self.optimizer = optimizer or torch.optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.lr,
            betas=self.cfg.betas,
            weight_decay=self.cfg.weight_decay,
        )

        # prepare with accelerate
        (
            self.model,
            self.optimizer,
            self.train_loader,
            self.val_loader,
        ) = self.accelerator.prepare(
            self.model, self.optimizer, self.train_loader, self.val_loader
        )

        self.scaler = torch.cuda.amp.GradScaler(enabled=self.cfg.use_amp)

        self._global_step = 0
        LOG.info("UNetTrainer initialized.")

    # ------------------------ public API ------------------------

    def train(self) -> None:
        for epoch in range(1, self.cfg.max_epochs + 1):
            train_logs = self._run_epoch(epoch, train=True)
            val_logs = self._run_epoch(epoch, train=False)

            # log to wandb/tensorboard via accelerator
            to_log = {f"train/{k}": v for k, v in train_logs.items()}
            to_log.update({f"val/{k}": v for k, v in val_logs.items()})
            to_log["epoch"] = epoch
            self.accelerator.log(to_log, step=self._global_step)

            # checkpointing (only main process)
            if self.accelerator.is_main_process and (epoch % self.cfg.ckpt_every == 0):
                self._save_ckpt(epoch)

            # preview samples (optional)
            if self.accelerator.is_main_process and (epoch % self.cfg.preview_every == 0):
                self._preview(epoch)

    # ---------------------- internals ---------------------------

    def _run_epoch(self, epoch: int, *, train: bool) -> Dict[str, float]:
        self.model.train(mode=train)

        loader = self.train_loader if train else self.val_loader
        meter: Dict[str, float] = {"loss": 0.0, "mse_raw": 0.0, "mse_lat": 0.0, "cond_loss": 0.0}
        steps = 0

        for batch in loader:
            # Support both (cond, x0) or (cond, x0, years) tuples as in your script
            if len(batch) == 3:
                cond, x0, years = batch
            else:
                cond, x0 = batch
                years = None

            with self.accelerator.accumulate(self.model):
                if train:
                    self.optimizer.zero_grad(set_to_none=True)

                # === LOSS (mirrors train_new_loss.py) ===
                if self.cfg.use_amp:
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        loss, comps = self._compute_loss(x0, cond, years=years)
                else:
                    loss, comps = self._compute_loss(x0, cond, years=years)

                if train:
                    self.accelerator.backward(loss)

                    if self.cfg.grad_clip and self.cfg.grad_clip > 0:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)

                    self.optimizer.step()

                # meters
                meter["loss"] += float(loss.detach().item())
                meter["mse_raw"] += float(comps.get("mse_raw", loss).detach().item())
                meter["mse_lat"] += float(comps.get("mse_lat", loss).detach().item())
                meter["cond_loss"] += float(comps.get("cond_loss", torch.zeros([], device=loss.device)).detach().item())
                steps += 1
                if train:
                    self._global_step += 1

        # average
        for k in meter:
            meter[k] = meter[k] / max(1, steps)

        split = "train" if train else "val"
        LOG.info(f"[{split}] epoch {epoch} :: "
                 f"loss={meter['loss']:.6f} mse_raw={meter['mse_raw']:.6f} "
                 f"mse_lat={meter['mse_lat']:.6f} cond_loss={meter['cond_loss']:.6f}")
        return meter

    @torch.no_grad()
    def _preview(self, epoch: int) -> None:
        """Lightweight preview: run a forward sample on a tiny batch and log images if tracker supports it."""
        try:
            import torchvision
            from torchvision.utils import make_grid
        except Exception:
            LOG.info("torchvision not available; skipping preview.")
            return

        self.model.eval()
        for batch in self.val_loader:
            if len(batch) == 3:
                cond, x0, years = batch
            else:
                cond, x0 = batch
                years = None

            cond = cond[: self.cfg.preview_batch]
            x0 = x0[: self.cfg.preview_batch]

            # Unwrap to access diffusion.sample(...)
            diff = _unwrap(self.model)
            B, _, H, W = cond.shape
            try:
                pred = diff.sample(cond, shape=(B, 1, H, W), device=cond.device)
            except TypeError:
                pred = diff.sample(cond, shape=(B, 1, H, W), device=cond.device)

            # normalize to [0,1]
            def _mm01(t):
                t = t.detach()
                t = (t - t.amin(dim=(2, 3), keepdim=True)) / (t.amax(dim=(2, 3), keepdim=True) - t.amin(dim=(2, 3), keepdim=True) + 1e-8)
                return t

            grid = make_grid(torch.cat([_mm01(cond), _mm01(x0), _mm01(pred)], dim=0), nrow=cond.size(0))
            self.accelerator.log({"preview/triptych": [self.accelerator.prepare_model(self.model), grid]}, step=self._global_step)
            break  # one preview is enough

    def _compute_loss(self, x0: torch.Tensor, cond: torch.Tensor, years: Optional[torch.Tensor] = None):
        """
        EXACT selection policy used in your train_new_loss.py:
          - if the underlying diffusion exposes `loss_components(x0, cond, years=...)`,
            expect a dict with keys: 'mse_raw', 'mse_lat', 'cond_loss', 'total'.
          - otherwise, use `loss(x0, cond)` and mirror to fields.
        """
        m = _unwrap(self.model)

        if hasattr(m, "loss_components"):
            # signature variants allowed: with/without years kwarg
            try:
                comps: Dict[str, torch.Tensor] = m.loss_components(x0, cond, years=years)
            except TypeError:
                comps = m.loss_components(x0, cond)
            loss = comps["total"]
            return loss, comps

        # Fallback to simple loss()
        if hasattr(m, "loss"):
            loss = m.loss(x0, cond) if years is None else m.loss(x0, cond, years=years)
        else:
            # very old API: forward returns loss
            out = m(x0, cond) if years is None else m(x0, cond, years=years)
            loss = out if torch.is_tensor(out) else out["loss"]

        comps = {
            "mse_raw": loss.detach(),
            "mse_lat": loss.detach(),
            "cond_loss": torch.zeros((), device=loss.device),
            "total": loss,
        }
        return loss, comps

    def _save_ckpt(self, epoch: int) -> None:
        import os, glob, re
        os.makedirs(self.cfg.ckpt_dir, exist_ok=True)
        path = os.path.join(self.cfg.ckpt_dir, f"ckpt_epoch_{epoch:04d}.pt")
        self.accelerator.save_state(path)
        LOG.info(f"Saved checkpoint -> {path}")

        # prune older ones
        def _key(p):
            m = re.search(r"epoch[_-](\d+)", os.path.basename(p))
            return (int(m.group(1)) if m else -1, os.path.getmtime(p))
        ckpts = sorted(glob.glob(os.path.join(self.cfg.ckpt_dir, "ckpt_epoch_*.pt")), key=_key, reverse=True)
        for p in ckpts[self.cfg.keep_last:]:
            try:
                os.remove(p)
            except OSError:
                pass

    @staticmethod
    def _cfg_from_dict(raw: Optional[Dict[str, Any]]) -> TrainerConfig:
        if raw is None:
            return TrainerConfig()
        # allow hydra DictConfig or plain dict, coerce types safely
        cfg = TrainerConfig(
            batch_size=int(raw.get("batch_size", 8)),
            num_workers=int(raw.get("num_workers", 4)),
            pin_memory=bool(raw.get("pin_memory", True)),
            max_epochs=int(raw.get("max_epochs", raw.get("num_epochs", 100))),
            lr=float(raw.get("optimizer", {}).get("lr", raw.get("lr", 2e-4))),
            betas=tuple(raw.get("optimizer", {}).get("betas", raw.get("betas", (0.9, 0.999)))),
            weight_decay=float(raw.get("optimizer", {}).get("weight_decay", raw.get("weight_decay", 1e-4))),
            grad_clip=float(raw.get("grad_clip", raw.get("max_grad_norm", 1.0))),
            use_amp=bool(raw.get("use_amp", True)),
            preview_every=int(raw.get("preview_every", 10)),
            preview_batch=int(raw.get("preview_batch", 2)),
            ckpt_every=int(raw.get("ckpt_every", raw.get("save_every", 10))),
            keep_last=int(raw.get("keep_last", 5)),
            ckpt_dir=str(raw.get("ckpt_dir", "checkpoints")),
            mean_anchor_beta=float(raw.get("mean_anchor_beta", 0.1)),
            cond_loss_scaling=float(raw.get("cond_loss_scaling", raw.get("mean_anchor_beta", 0.1))),
        )
        return cfg
