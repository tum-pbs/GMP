import logging
from typing import Any, Dict, Optional
import torch
import pytorch_lightning as pl

log = logging.getLogger(__name__)


class ProgressLoggingCallback(pl.Callback):
    """Callback to log epoch, iteration, and loss metrics cleanly to console/log files."""

    def __init__(self, log_every_n_steps: int = 1) -> None:
        super().__init__()
        self.log_every_n_steps = log_every_n_steps

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if (batch_idx + 1) % self.log_every_n_steps == 0:
            epoch = trainer.current_epoch
            loss = None

            # Extract loss from outputs
            if outputs is not None:
                if isinstance(outputs, dict) and "loss" in outputs:
                    loss = outputs["loss"]
                elif torch.is_tensor(outputs):
                    loss = outputs

            # Fallback: check callback_metrics
            if loss is None:
                for k, v in trainer.callback_metrics.items():
                    if "loss" in k:
                        loss = v
                        break

            if torch.is_tensor(loss):
                loss = loss.item()

            try:
                lr = pl_module.optimizers().param_groups[0]["lr"]
                lr_str = f" | LR: {lr:.2e}"
            except Exception:
                lr_str = ""

            if loss is not None:
                print(f"Epoch {epoch:03d} | Batch {batch_idx + 1:04d} | Train Loss: {loss:.6f}{lr_str}", flush=True)
            else:
                print(f"Epoch {epoch:03d} | Batch {batch_idx + 1:04d}{lr_str}", flush=True)

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        epoch = trainer.current_epoch
        metrics = trainer.callback_metrics
        loss_val = metrics.get("train/loss", metrics.get("train/loss_epoch", None))
        loss_str = f" | Epoch Loss: {loss_val.item():.6f}" if (loss_val is not None and torch.is_tensor(loss_val)) else ""
        print(f"=== Epoch {epoch:03d} Finished{loss_str} ===", flush=True)

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        epoch = trainer.current_epoch
        val_loss = None

        for k, v in trainer.callback_metrics.items():
            if "val" in k and "loss" in k:
                val_loss = v
                break

        if val_loss is None:
            val_loss = trainer.callback_metrics.get("val_loss")

        if torch.is_tensor(val_loss):
            val_loss = val_loss.item()

        if val_loss is not None:
            print(f"Epoch {epoch:03d} Validation | Val Loss: {val_loss:.6f}", flush=True)

