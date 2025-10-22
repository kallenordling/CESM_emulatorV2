#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import os
import random
from typing import Any, Dict

import numpy as np
import torch
from accelerate import Accelerator
from omegaconf import OmegaConf


def set_seed(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # may reduce speed
    torch.backends.cudnn.benchmark = False


def _locate(target: str):
    """Import a dotted path like 'pkg.module:Class' or 'pkg.module.Class'."""
    if ":" in target:
        module_path, name = target.split(":", 1)
    else:
        module_path, name = target.rsplit(".", 1)
    mod = importlib.import_module(module_path)
    return getattr(mod, name)


def instantiate(spec: Dict[str, Any]) -> Any:
    """
    Minimal Hydra-like instantiate:
      spec = {"_target_": "pkg.module:Class", "arg1": 1, "arg2": "x"}
    """
    spec = dict(spec)  # shallow copy
    target = spec.pop("_target_")
    cls = _locate(target)
    return cls(**spec)


def maybe_instantiate(spec_or_none):
    return None if spec_or_none is None else instantiate(spec_or_none)


def main():
    parser = argparse.ArgumentParser(description="Train diffusion UNet with external YAML config.")
    parser.add_argument("--config", "-c", type=str, required=True, help="Path to config YAML.")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)

    # ---- output dir & seed ---------------------------------------------------
    outdir = cfg.get("output_dir", "outputs/run")
    os.makedirs(outdir, exist_ok=True)

    set_seed(cfg.get("seed"))

    # Save the resolved config for reproducibility
    OmegaConf.save(config=cfg, f=os.path.join(outdir, "config_resolved.yaml"))

    # ---- accelerator ---------------------------------------------------------
    acc_kwargs = dict(cfg.get("accelerate", {}))
    accelerator = Accelerator(**acc_kwargs)

    if accelerator.is_main_process:
        print(f"[INFO] Using output dir: {outdir}")
        print(f"[INFO] Accelerator state: {accelerator.state}")

    # ---- datasets ------------------------------------------------------------
    train_set = instantiate(cfg.datasets.train)
    val_set = instantiate(cfg.datasets.val)

    # ---- model / scheduler / optimizer (optimizer is optional; trainer can make one) ---
    model = instantiate(cfg.model)

    scheduler = maybe_instantiate(cfg.get("scheduler"))
    optimizer = None
    if "optimizer" in cfg and cfg.optimizer is not None:
        # If user wants to define optimizer here and pass to trainer
        opt_spec = dict(cfg.optimizer)
        opt_target = opt_spec.pop("_target_", "torch.optim:AdamW")
        Opt = _locate(opt_target)
        optimizer = Opt(model.parameters(), **opt_spec)

    # ---- trainer -------------------------------------------------------------
    # The trainer in trainers/unet_trainer.py expects:
    #   UNetTrainer(train_set, val_set, model=model, accelerator=accelerator,
    #               scheduler=scheduler, optimizer=optimizer, cfg=cfg.trainer.cfg)
    # NOTE: cfg.trainer must contain {"_target_": "trainers.unet_trainer:UNetTrainer", "cfg": {...}}
    trainer_target = cfg.trainer.get("_target_", "trainers.unet_trainer:UNetTrainer")
    Trainer = _locate(trainer_target)

    trainer = Trainer(
        train_set,
        val_set,
        model=model,
        accelerator=accelerator,
        scheduler=scheduler,
        optimizer=optimizer,
        cfg=cfg.trainer.get("cfg", {}),
    )

    # ---- run -----------------------------------------------------------------
    trainer.train()


if __name__ == "__main__":
    main()
