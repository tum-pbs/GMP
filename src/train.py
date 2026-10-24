import logging
from typing import Any

import torch
torch.set_float32_matmul_precision('high')
import hydra
import os
from pathlib import Path
import lightning as L
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from lightning.pytorch.utilities.rank_zero import rank_zero_only, rank_zero_info

from src.utils.utils import load_run_id, generate_id, save_config, setup_training_mode

OmegaConf.register_new_resolver("eval", eval)

log = logging.getLogger(__name__)


@hydra.main(config_path="../config", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> Any:

    L.seed_everything(cfg.seed, workers=True)

    name = getattr(cfg, "name", "")
    if not name:
        run_id = generate_id()
        cfg.name = run_id
        project_path = Path(cfg.trainer.default_root_dir)
        project_path_for_ckpt = project_path
        rank_zero_info(f"Starting new run with ID {run_id}.")

    else:
        project_path = Path(cfg.trainer.default_root_dir)
        project_path_for_ckpt = project_path

        if os.path.exists(project_path):
            rank_zero_info(f"Directory {project_path} exists. Trying to continue run.")
            run_id = load_run_id(project_path)
        else:
            run_id = generate_id()
            rank_zero_info(f"Starting new run with name {cfg.name} and ID {run_id}.")

    cfg.run_id = run_id

    # Resolve any interpolations in the config
    for _ in range(2):
        cfg = OmegaConf.to_container(cfg, resolve=True)
        cfg = OmegaConf.create(cfg)

    # Log training mode settings
    setup_training_mode(cfg)

    datamodule = instantiate(cfg['data'])
    model = instantiate(cfg['module'])
    trainer: L.Trainer = instantiate(cfg['trainer'])

    # Attempt to resume from the latest checkpoint in the project path.
    ckpt_path = None
    last_ckpt = project_path_for_ckpt / "checkpoints" / "last.ckpt"
    if last_ckpt.exists():
        rank_zero_info(f"Found checkpoint at {last_ckpt}, resuming training.")
        ckpt_path = last_ckpt.as_posix()
        
        checkpoint = torch.load(last_ckpt)
        current_epoch = checkpoint['epoch']
        max_epochs = cfg.trainer.max_epochs
        if max_epochs > 0 and current_epoch >= max_epochs:
            rank_zero_info(f"Checkpoint epoch ({current_epoch}) >= max_epochs ({max_epochs}). Running inference.")
            model.set_stage("best_val")
            rank_zero_info("Running evaluation on test set with best weights...")
            best_ckpt = project_path_for_ckpt / "checkpoints" / "best.ckpt"
            trainer.test(model=model, datamodule=datamodule, ckpt_path=best_ckpt)
            return  # Exit after inference

    save_config(cfg, Path(project_path))

    # Log config to wandb if available
    if trainer.is_global_zero and trainer.loggers:
        for logger in trainer.loggers:
            # Check for WandbLogger specifically
            if hasattr(logger, "experiment"):
                from lightning.pytorch.loggers.wandb import WandbLogger
                if isinstance(logger, WandbLogger):
                    # In DDP, the experiment object might be a dummy function on some ranks,
                    # so we check if .config exists and is not a callable function.
                    exp = logger.experiment
                    if hasattr(exp, "config") and not callable(exp):
                        #exp.config.update(cfg, allow_val_change=True)
                        resolved_cfg = OmegaConf.to_container(cfg, resolve=True)
                        exp.config.update(resolved_cfg, allow_val_change=True)

    rank_zero_info("Configuration:\n%s", cfg)

    model.set_stage("fit")
    rank_zero_info("Starting training")
    trainer.fit(model=model, datamodule=datamodule, ckpt_path=ckpt_path)

    # set stage to inform callbacks about behavior
    model.set_stage("last_ema")
    rank_zero_info("Running evaluation on test set with last weights...")
    trainer.test(model=model, datamodule=datamodule)

    model.set_stage("best_val")
    rank_zero_info("Running evaluation on test set with best validation weights (no EMA)...")
    best_ckpt = project_path_for_ckpt / "checkpoints" / "best.ckpt"
    if best_ckpt.exists():
        trainer.test(model=model, datamodule=datamodule, ckpt_path=best_ckpt)
    else:
        rank_zero_info(f"Checkpoint {best_ckpt} not found, skipping evaluation on best validation weights.")



if __name__ == "__main__":
    main()
