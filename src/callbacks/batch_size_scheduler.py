from __future__ import annotations

import lightning as L
from lightning.pytorch import Callback
from lightning.pytorch.utilities.rank_zero import rank_zero_info
from typing import Optional, Dict, Any

class BatchSizeSchedulerCallback(Callback):
    """Callback to dynamically schedule/scale the batch size (via gradient accumulation steps) during training.
    
    Supports:
    1. Interval schedule: multiply gradient accumulation steps by increase_factor every interval_epochs epochs.
    2. Epoch dict schedule: lookup absolute value or multiplier of accumulation steps from a dictionary.
    """

    def __init__(
        self,
        enabled: bool = False,
        schedule_type: str = "interval",  # "interval" or "epoch_dict"
        # For "interval" schedule type:
        interval_epochs: int = 50,
        increase_factor: float = 2.0,
        # For "epoch_dict" schedule type:
        epoch_dict: Optional[Dict[int, float]] = None,
        epoch_dict_type: str = "multiplier",  # "multiplier" or "absolute"
        # Max cap to prevent memory issues / extremely slow epochs:
        max_accumulation: int = 128,
    ) -> None:
        super().__init__()
        self.enabled = enabled
        self.schedule_type = schedule_type
        self.interval_epochs = interval_epochs
        self.increase_factor = increase_factor
        self.epoch_dict = epoch_dict
        self.epoch_dict_type = epoch_dict_type
        self.max_accumulation = max_accumulation
        
        # State to store the initial baseline accumulate_grad_batches
        self.initial_accumulate: Optional[int] = None

    def on_train_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if not self.enabled:
            return

        # Ensure we have the initial value captured
        if self.initial_accumulate is None:
            self.initial_accumulate = trainer.accumulate_grad_batches
            rank_zero_info(
                f"[BatchSizeSchedulerCallback] Captured initial accumulate_grad_batches = {self.initial_accumulate}"
            )

        current_epoch = trainer.current_epoch
        new_accumulate = self.initial_accumulate

        if self.schedule_type == "interval":
            if self.interval_epochs and self.interval_epochs > 0:
                num_increases = current_epoch // self.interval_epochs
                new_accumulate = int(self.initial_accumulate * (self.increase_factor ** num_increases))
        elif self.schedule_type == "epoch_dict":
            if self.epoch_dict:
                # Find all epochs in the dictionary that are <= current_epoch
                # Need to handle keys that might be parsed as strings from YAML
                typed_epoch_dict = {int(k): v for k, v in self.epoch_dict.items()}
                applicable_epochs = [epoch for epoch in typed_epoch_dict if epoch <= current_epoch]
                if applicable_epochs:
                    best_epoch = max(applicable_epochs)
                    value = typed_epoch_dict[best_epoch]
                    if self.epoch_dict_type == "multiplier":
                        new_accumulate = int(self.initial_accumulate * value)
                    else:
                        new_accumulate = int(value)

        # Enforce bounds
        new_accumulate = max(1, min(new_accumulate, self.max_accumulation))

        # Dynamically set on the trainer if changed
        if trainer.accumulate_grad_batches != new_accumulate:
            rank_zero_info(
                f"[BatchSizeSchedulerCallback] Epoch {current_epoch}: Changing accumulate_grad_batches "
                f"from {trainer.accumulate_grad_batches} to {new_accumulate}."
            )
            trainer.accumulate_grad_batches = new_accumulate

    def on_save_checkpoint(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        checkpoint: Dict[str, Any],
    ) -> Dict[str, Any]:
        if self.enabled:
            # Safely get/create callbacks state dict
            if 'callbacks' not in checkpoint:
                checkpoint['callbacks'] = {}
            checkpoint['callbacks']['batch_size_scheduler'] = {
                "initial_accumulate": self.initial_accumulate
            }
        return checkpoint

    def on_load_checkpoint(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        checkpoint: Dict[str, Any],
    ) -> None:
        callback_state = checkpoint.get('callbacks', {}).get('batch_size_scheduler', {})
        if callback_state:
            self.initial_accumulate = callback_state.get("initial_accumulate", self.initial_accumulate)
            rank_zero_info(
                f"[BatchSizeSchedulerCallback] Restored initial accumulate_grad_batches = {self.initial_accumulate}"
            )
