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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _locate(target: str):
    """Import 'module:Class' or 'module.Class' dynamically."""
    if ":" in target:
        module_path, name = target.split(":", 1)
    else:
        module_path, name = target.rsplit(".", 1)
    mod = importlib.import_module(module_path)
    return getattr(mod, name)


def instantiate(spec: Dict[str, Any]) -> Any:
    spec = dict(spec)
    target = spec.pop("_target_")
    cls = _locate(target)
    return cls(**spec)


def maybe_instantiate(spec_or_none):
    return None if spec_or_none is None else instantiate(spec_or_none)


def main():
    parser = argparse.ArgumentParser(description="Train diffusion UNet with YAML config.")
    parser.add_argument("--config", "-c", type=str, required=True)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)

    outdir = cfg.get("output_dir", "outputs/run")
    os.makedirs(outdir, exist_ok=True)
    OmegaConf.save(config=cfg, f=os.path.join(outdir, "config_resolved.yaml"))

    set_seed(cfg.get("seed"))

    accelerator = Accelerator(**dict(cfg.get("accelerate", {})))

    # --- instantiate dataset, model, trainer ---
    train_set = instantiate(cfg.datasets.train)
    val_set = instantiate(cfg.datasets.val)
    model = instantiate(cfg.model)

    scheduler = maybe_instantiate(cfg.get("scheduler"))
    optimizer = None
    if "optimizer" in cfg and cfg.optimizer is not None:
        opt_spec = dict(cfg.optimizer)
        opt_target = opt_spec.pop("_target_", "torch.optim:AdamW")
        Opt = _locate(opt_target)
        optimizer = Opt(model.parameters(), **opt_spec)

    # trainer
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

    trainer.train()


if __name__ == "__main__":
    main()
