from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import lightning as L
import os
import torch
from lightning.pytorch import Callback
from lightning.pytorch.utilities.types import STEP_OUTPUT
from lightning.pytorch.callbacks import ModelCheckpoint
from torchmetrics.utilities import rank_zero_info, rank_zero_warn

def save_last_checkpoint(trainer: "L.Trainer", dirpath: str) -> None:

    try:
        filepath = dirpath + '/last.ckpt'
    except Exception as e: # noqa
        filepath = '/tmp/checkpoint-ema/last.ckpt'
        raise e

    ckpt_path = Path(filepath)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    trainer.save_checkpoint(ckpt_path.as_posix(), weights_only=True)

class EMACallback(Callback):
    """Exponential moving average weights for evaluation.

    Keeps a shadow copy of trainable parameters that is updated after every
    training batch. For validation, test and predict phases the EMA weights are
    temporarily swapped in and the original weights are restored afterwards.
    """

    def __init__(self, 
                 dirpath: str,
                 decay: float = 0.999, 
                 ema_device: Optional[str] = None,
                 use_ema_for_test: bool = False,
                 ) -> None:
        super().__init__()
        if decay <= 0.0 or decay >= 1.0:
            raise ValueError("decay must be in (0, 1)")
        self.decay = decay
        self.ema_device = ema_device
        self.ema_state: Dict[str, torch.Tensor] = {}
        self._backup: Dict[str, torch.Tensor] = {}
        self.use_ema_for_test = use_ema_for_test
        self.dirpath = dirpath
        print('use_ema_for_test: ', self.use_ema_for_test)

    @torch.no_grad()
    def on_fit_start(self, trainer: "L.Trainer", pl_module: "L.LightningModule") -> None:  # type: ignore[name-defined]
        """Initialize EMA tensors to match the current model parameters."""
        self._build_ema_state(pl_module)

    @torch.no_grad()
    def on_train_batch_end(
        self,
        trainer: "L.Trainer",  # type: ignore[name-defined]
        pl_module: "L.LightningModule",  # type: ignore[name-defined]
        outputs: STEP_OUTPUT,
        batch,
        batch_idx: int,
    ) -> None:
        """Update EMA weights after each optimizer step."""
        if not self.ema_state:
            self._build_ema_state(pl_module)
        for name, param in pl_module.named_parameters():
            if not param.requires_grad:
                continue
            ema_param = self.ema_state.get(name)
            if ema_param is None:
                device = torch.device(self.ema_device) if self.ema_device else None
                ema_tensor = param.detach().clone()
                if device:
                    ema_tensor = ema_tensor.to(device)
                self.ema_state[name] = ema_tensor
                continue
            ema_param.mul_(self.decay).add_(param.detach(), alpha=1 - self.decay)

    @torch.no_grad()
    def on_validation_start(
        self, trainer: "L.Trainer", pl_module: "L.LightningModule"  # type: ignore[name-defined]
    ) -> None:
        if self.check_if_swapping(pl_module):
            rank_zero_info('on_validation_start: swap in ema weights')
            self._swap_in_ema_weights(pl_module)
        else:
            rank_zero_info('on_validation_start: do not swap in ema weights')


    @torch.no_grad()
    def on_validation_end(
        self, trainer: "L.Trainer", pl_module: "L.LightningModule"  # type: ignore[name-defined]
    ) -> None:
        if self.check_if_swapping(pl_module):
            self._restore_original_weights(pl_module)

    @torch.no_grad()
    def on_test_start(
        self, trainer: "L.Trainer", pl_module: "L.LightningModule"  # type: ignore[name-defined]
    ) -> None:

        try:
            ckpt = torch.load(trainer.checkpoint_callback.dirpath + '/last.ckpt')
            self.on_load_checkpoint(trainer, pl_module, ckpt['callbacks']['ema'])
            
            if self.check_if_swapping(pl_module):
                rank_zero_info('on_test_start: swap in ema weights')
                self._swap_in_ema_weights(pl_module)
            else:
                rank_zero_info('on_test_start: do not swap in ema weights')

        except Exception as e:
            print(e)
            rank_zero_warn('No checkpoint found, using original weights.')

    def check_if_swapping(self, pl_module: "L.LightningModule"):

        return self.use_ema_for_test and ("ema" in getattr(pl_module, "stage", "ema"))

    @torch.no_grad()
    def on_test_end(
        self, trainer: "L.Trainer", pl_module: "L.LightningModule"  # type: ignore[name-defined]
    ) -> None:
        if self.check_if_swapping(pl_module):
            rank_zero_info('on_test_end: restore original weights')
            self._restore_original_weights(pl_module)

    def on_train_epoch_end(self, trainer: "L.Trainer", pl_module: "L.LightningModule") -> None:  # type: ignore[name-defined]

        checkpoint_callback: ModelCheckpoint = trainer.checkpoint_callback # noqa
        if checkpoint_callback is not None:
            self._swap_in_ema_weights(pl_module)
            save_last_checkpoint(trainer, self.dirpath)
            self._restore_original_weights(pl_module)


    @torch.no_grad()
    def on_predict_start(
        self, trainer: "L.Trainer", pl_module: "L.LightningModule"  # type: ignore[name-defined]
    ) -> None:
        if self.check_if_swapping(pl_module):
            self._swap_in_ema_weights(pl_module)

    @torch.no_grad()
    def on_predict_end(
        self, trainer: "L.Trainer", pl_module: "L.LightningModule"  # type: ignore[name-defined]
    ) -> None:
        if self.check_if_swapping(pl_module):
            self._restore_original_weights(pl_module)

    def _build_ema_state(self, pl_module: "L.LightningModule") -> None:  # type: ignore[name-defined]
        """Create the EMA shadow copy for all trainable parameters."""
        self.ema_state.clear()
        device = torch.device(self.ema_device) if self.ema_device else None
        for name, param in pl_module.named_parameters():
            if not param.requires_grad:
                continue
            ema_tensor = param.detach().clone()
            if device:
                ema_tensor = ema_tensor.to(device)
            self.ema_state[name] = ema_tensor

    def _swap_in_ema_weights(self, pl_module: "L.LightningModule") -> None:  # type: ignore[name-defined]
        """Temporarily load EMA weights for evaluation."""
        if not self.ema_state or self._backup:
            return
        self._backup = {}
        for name, param in pl_module.named_parameters():
            ema_param = self.ema_state.get(name)
            if ema_param is None:
                continue
            self._backup[name] = param.data.clone()
            param.data.copy_(ema_param.to(param.device))

    def _restore_original_weights(self, pl_module: "L.LightningModule") -> None:  # type: ignore[name-defined]
        """Restore the original (non-EMA) weights after evaluation."""
        if not self._backup:
            return
        for name, param in pl_module.named_parameters():
            original = self._backup.get(name)
            if original is not None:
                param.data.copy_(original.to(param.device))
        self._backup = {}

    def on_save_checkpoint(
        self,
        trainer: "L.Trainer",  # type: ignore[name-defined]
        pl_module: "L.LightningModule",  # type: ignore[name-defined]
        checkpoint: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Persist EMA tensors so runs can be resumed with the same shadow weights."""
        del trainer, pl_module  # unused
        print('on_save_checkpoint')
     
        checkpoint['callbacks']['ema'] = {
            "decay": torch.tensor(self.decay),
            "ema_state": {k: v.detach().cpu() for k, v in self.ema_state.items()},
            "ema_device": self.ema_device,
        }

    def on_load_checkpoint(
        self,
        trainer: "L.Trainer",  # type: ignore[name-defined]
        pl_module: "L.LightningModule",  # type: ignore[name-defined]
        checkpoint: Dict[str, torch.Tensor],
    ) -> None:
        """Restore EMA tensors and place them on the requested device."""
        del trainer, pl_module  # unused
        self.decay = float(checkpoint.get("decay", torch.tensor(self.decay)))
        saved_state = checkpoint.get("ema_state", {})
        self.ema_device = checkpoint.get("ema_device", self.ema_device)
        device = torch.device(self.ema_device) if self.ema_device else None
        self.ema_state = {
            name: tensor.to(device) if device else tensor
            for name, tensor in saved_state.items()
        }