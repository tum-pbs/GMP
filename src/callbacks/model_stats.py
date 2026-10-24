from __future__ import annotations

from typing import Dict, Optional, Sequence, Any

import lightning as L
import torch
from lightning.pytorch import Callback
from lightning.pytorch.utilities.rank_zero import rank_zero_info, rank_zero_warn
from torch.profiler import ProfilerActivity, profile
from time import perf_counter


class ModelStatsCallback(Callback):
    """Profiles parameter count, GFLOPs and peak VRAM for a dummy input."""

    def __init__(
        self,
        input_size: Sequence[int],
        device: Optional[str] = None,
        warmup_steps: int = 1,
        batch_size: int = 1,
        throughput_steps: int = 10,
        throughput_rollout: int = 2, # 1 autoregressive steps
    ) -> None:
        super().__init__()
        if not input_size:
            raise ValueError("input_size must be a non-empty sequence of ints.")
        self.input_size = tuple(int(dim) for dim in input_size)
        self.device = device
        self.warmup_steps = max(0, warmup_steps)
        self.batch_size = batch_size
        # self.throughput_batch_size = throughput_batch_size
        self.throughput_steps = max(1, throughput_steps)
        self.throughput_rollout = throughput_rollout
        self._has_profiled = False
        self.stats: Dict[str, Optional[float]] = {}


    def on_fit_start(self, trainer: "L.Trainer", pl_module: "L.LightningModule") -> None:  # type: ignore[name-defined]
        if self._has_profiled:
            return
        self._has_profiled = True

        if (dataloader := trainer.val_dataloaders) is None:
            trainer.validate_loop.setup_data()
            dataloader = trainer.val_dataloaders

        try:
            self.stats = self._profile_model(pl_module, dataloader)
            self._log_stats(trainer, self.stats)
        except Exception as exc:  # noqa: BLE001
            rank_zero_warn(f"ModelStatsCallback failed to profile model: {exc}")
            raise exc

    def _profile_model(self, pl_module: "L.LightningModule", dataloader: Optional = None) -> Dict[str, Optional[float]]:  # type: ignore[name-defined]
        model_device = self._resolve_device(pl_module)
        dtype = self._resolve_dtype(pl_module)

        if dataloader is None:
            dummy = torch.randn(*self.input_size, device=model_device, dtype=dtype)
        else:
            dummy = next(iter(dataloader))

        was_training = pl_module.training
        pl_module.eval()

        with torch.no_grad():
            for _ in range(self.warmup_steps):
                _ = pl_module.predict(dummy, rollout_num=2)

            flops = self._measure_flops(pl_module, dummy)
            flops_preprocessing = self._measure_preprocessing_flops(pl_module, dummy)
            vram_mb = self._measure_vram(pl_module, dummy)
            throughput = self._measure_throughput(pl_module, dummy)
            throughput_preprocessing = self._measure_preprocessing_throughput(pl_module, dummy)

        if was_training:
            pl_module.train()

        num_params = float(sum(p.numel() for p in pl_module.parameters()))

        active_params = None
        active_pct = None
        inner_model = getattr(pl_module, "model", pl_module)
        if hasattr(inner_model, "count_active_parameters"):
            breakdown = inner_model.count_active_parameters()
            active_params = float(breakdown["active"])
            active_pct = float(breakdown["pct"])

        return {
            "num_params": num_params,
            "active_params": active_params,
            "active_params_pct": active_pct,
            "flops": flops,
            "preprocessing_flops": flops_preprocessing,
            "vram_mb": vram_mb,
            "throughput_sps": throughput,
            "preprocessing_throughput_sps": throughput_preprocessing,
        }

    def _log_stats(self, trainer: "L.Trainer", stats: Dict[str, Optional[float]]) -> None:  # type: ignore[name-defined]
        clean_stats = {k: v for k, v in stats.items() if v is not None}
        if not clean_stats:
            return

        display_stats = {}
        for k, v in clean_stats.items():
            if k == "num_params":
                display_stats["num_params"] = f"{v / 1e6:.3f} M"
            elif k == "active_params":
                display_stats["active_params"] = f"{v / 1e6:.3f} M"
            elif k == "active_params_pct":
                display_stats["active_params_pct"] = f"{v:.1f}%"
            elif k == "flops":
                display_stats["flops"] = f"{v / 1e9:.3f} G"
            elif k == "preprocessing_flops":
                display_stats["preprocessing_flops"] = f"{v / 1e9:.3f} G"
            elif isinstance(v, float):
                display_stats[k] = f"{v:.3f}"
            else:
                display_stats[k] = str(v)

        display = ", ".join(f"{k}={v}" for k, v in display_stats.items())
        rank_zero_info(f"[ModelStats] {display}")

        metrics = {f"model/{k}": v for k, v in clean_stats.items()}
        if trainer.loggers:
            for logger in trainer.loggers:
                if hasattr(logger, "log_metrics"):
                    logger.log_metrics(metrics, step=trainer.global_step)

    def _measure_flops(self, pl_module: "L.LightningModule", dummy: torch.Tensor) -> Optional[float]:  # type: ignore[name-defined]
        activities = [ProfilerActivity.CPU]
        if pl_module.device.type == "cuda":
            activities.append(ProfilerActivity.CUDA)

        try:
            with profile(activities=activities, record_shapes=False, with_flops=True) as prof:
                _ = pl_module.predict(dummy, rollout_num=2)

            flops = sum(evt.flops for evt in prof.key_averages() if evt.flops is not None)

            rank_zero_info(f'FLOPS: {flops} ({flops / 1e9:.3f} G)')

            return flops
        except Exception:
            return None

    def _measure_preprocessing_flops(self, pl_module: "L.LightningModule", dummy: torch.Tensor) -> Optional[float]:  # type: ignore[name-defined]
        activities = [ProfilerActivity.CPU]
        if pl_module.device.type == "cuda":
            activities.append(ProfilerActivity.CUDA)

        try:
            with profile(activities=activities, record_shapes=False, with_flops=True) as prof:
                _ = pl_module.preprocess_from_batch(dummy)

            flops = sum(evt.flops for evt in prof.key_averages() if evt.flops is not None)

            rank_zero_info(f'preprocessing FLOPS: {flops} ({flops / 1e9:.3f} G)')

            return flops
        except Exception:
            return None

    def _measure_vram(self, pl_module: "L.LightningModule", dummy: torch.Tensor) -> Optional[float]:  # type: ignore[name-defined]
        if pl_module.device.type != "cuda" or not torch.cuda.is_available():
            return None

        device_index = pl_module.device.index or torch.cuda.current_device()
        torch.cuda.reset_peak_memory_stats(device_index)
        torch.cuda.empty_cache()
        _ = pl_module.predict(dummy, rollout_num=2)
        torch.cuda.synchronize(device_index)
        peak_bytes = torch.cuda.max_memory_allocated(device_index)
        return peak_bytes / (1024**2)

    def _measure_throughput(self, pl_module: "L.LightningModule", dummy: Any) -> Optional[float]:  # type: ignore[name-defined]
        # Adjust batch size for throughput without changing spatial dims.
        # if self.throughput_batch_size is not None and self.throughput_batch_size > 0:
        #
        #     shape = (self.throughput_batch_size, *dummy.shape[1:])
        #
        #     dummy = torch.randn(*shape, device=dummy.device, dtype=dummy.dtype)

        try:
            if pl_module.device.type == "cuda" and torch.cuda.is_available():
                torch.cuda.synchronize(pl_module.device)
            start = perf_counter()
            with torch.no_grad():
                for _ in range(self.throughput_steps):
                    _ = pl_module.predict(dummy, rollout_num=self.throughput_rollout)
            if pl_module.device.type == "cuda" and torch.cuda.is_available():
                torch.cuda.synchronize(pl_module.device)
            duration = perf_counter() - start
            total_samples = self.throughput_steps * self.batch_size * (self.throughput_rollout-1)
            return total_samples / duration if duration > 0 else None
        except Exception:
            return None

    def _measure_preprocessing_throughput(self, pl_module: "L.LightningModule", dummy: Any) -> Optional[float]:

        try:
            if pl_module.device.type == "cuda" and torch.cuda.is_available():
                torch.cuda.synchronize(pl_module.device)
            start = perf_counter()
            with torch.no_grad():
                for _ in range(self.throughput_steps):
                    _ = pl_module.preprocess_from_batch(dummy)
            if pl_module.device.type == "cuda" and torch.cuda.is_available():
                torch.cuda.synchronize(pl_module.device)
            duration = perf_counter() - start
            total_samples = self.throughput_steps * self.batch_size
            return total_samples / duration if duration > 0 else None
        except Exception:
            return None

    def _resolve_device(self, pl_module: "L.LightningModule") -> torch.device:  # type: ignore[name-defined]
        if self.device:
            return torch.device(self.device)
        try:
            return next(pl_module.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def _resolve_dtype(self, pl_module: "L.LightningModule") -> torch.dtype:  # type: ignore[name-defined]
        try:
            return next(pl_module.parameters()).dtype
        except StopIteration:
            return torch.float32

