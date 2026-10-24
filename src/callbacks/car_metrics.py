import logging
from dataclasses import replace
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from lightning import Callback, LightningModule, Trainer
from lightning.fabric.utilities import rank_zero_only
from torchmetrics.utilities import rank_zero_warn

log = logging.getLogger(__name__)


class ComputeCarMetrics(Callback):
    """
    Compute per-channel MSE, relative L2 (%), median relative L1 (%),
    and maximum absolute error over the test set.

    - Runs on_test_start; iterates trainer.test_dataloaders once.
    - For each test sample and each channel c computes:
        * MSE_c = mean((p - t)^2)
        * relL2_c = sqrt(sum((p - t)^2) / sum(t^2)) * 100
        * relL1_c = median( |p - t| / (|t| + eps) ) * 100
        * MAE_c = max(|p - t|)
    - Aggregates per-channel values across all test samples and reports mean/std.
    - Under DDP, aggregates statistics across all ranks so metrics are global.
    - Writes final scalars to the logger run summary (e.g. wandb.summary) to avoid time series.
    """

    def __init__(self, batch_size_inference: int = 1, eps: float = 1e-8, chunk_size: int = 65536, **model_kwargs) -> None:
        super().__init__()
        self.batch_size_inference = int(batch_size_inference)
        self.eps = float(eps)
        if isinstance(chunk_size, str) and chunk_size.lower() == "none":
            self.chunk_size = None
        else:
            self.chunk_size = chunk_size
        self.model_kwargs = model_kwargs

    def setup_test_dataloader(self, trainer: Trainer) -> None:
        td = getattr(trainer, "test_dataloaders", None)
        if td is None:
            try:
                trainer.test_loop.setup_data()
            except Exception:
                # best-effort; caller will handle missing loader
                pass

    def _compute_metrics_for_pair(
        self, pred: Any, tgt: Any, normalize: Optional[Dict[str, torch.Tensor]] = None, batch: Optional[Any] = None
    ) -> Dict[str, List[float]]:
        """
        Compute metrics for a pair of prediction and target.
        If `normalize` is provided, normalize the values before computing metrics.
        If `batch` metadata specifies separate surface and volume points, slice each set cleanly.
        """
        # extract arrays
        p = pred.values if hasattr(pred, "values") else pred
        t = tgt.values if hasattr(tgt, "values") else tgt

        p = np.asarray(torch.as_tensor(p).detach().cpu())  # numpy array
        t = np.asarray(torch.as_tensor(t).detach().cpu())

        # Normalize if mean and std are provided
        if normalize:
            mean = torch.as_tensor(normalize["mean"]).detach().cpu().numpy().ravel()
            std = torch.as_tensor(normalize["std"]).detach().cpu().numpy().ravel()
            # Reshape mean and std to broadcast correctly with p and t
            if p.ndim == 4:
                mean = mean[None, None, :, None]
                std = std[None, None, :, None]
            elif p.ndim == 3:
                mean = mean[None, :, None]
                std = std[None, :, None]
            elif p.ndim == 2:
                mean = mean[:, None]
                std = std[:, None]
            p = (p - mean) / (std + self.eps)
            t = (t - mean) / (std + self.eps)

        # normalize shapes to (T, C, N)
        def to_tcn(x: np.ndarray) -> np.ndarray:
            if x.ndim == 4 and x.shape[0] == 1:
                return x[0]  # (T, C, N)
            if x.ndim == 3:
                # could be (T, C, N) or (B, C, N) with B==1
                if x.shape[0] == 1:
                    return x[0][None, ...]  # (T=1, C, N)
                # treat as (T=1, C, N) if ambiguous
                return x[None, ...]
            if x.ndim == 2:
                # (C, N) -> (T=1, C, N)
                return x[None, ...]
            raise ValueError(f"Unsupported tensor shape {x.shape} for metrics computation")

        p_arr = to_tcn(p)
        t_arr = to_tcn(t)

        # align time dim
        T = min(p_arr.shape[0], t_arr.shape[0])
        p_arr = p_arr[:T]
        t_arr = t_arr[:T]

        # now shapes: (T, C, N)
        _, C_total, N_total = p_arr.shape

        # Extract metadata if available to slice volume vs surface points cleanly
        N_surf, C_surf = 0, 0
        if batch is not None and hasattr(batch, "loading_metadata"):
            meta = batch.loading_metadata
            if hasattr(meta, "num_surface_points") and hasattr(meta, "num_surface_value_channels"):
                try:
                    N_surf = int(torch.as_tensor(meta.num_surface_points).flatten()[0].item())
                    C_surf = int(torch.as_tensor(meta.num_surface_value_channels).flatten()[0].item())
                except Exception:
                    pass

        N_vol = N_total - N_surf if (N_surf > 0 and N_surf < N_total) else 0
        C_vol = C_total - C_surf if (C_surf > 0 and C_surf < C_total) else 0

        mse_list: List[float] = []
        rel_l2_list: List[float] = []
        rel_l1_list: List[float] = []
        mae_list: List[float] = []

        for c in range(C_total):
            if N_surf > 0 and N_vol > 0 and C_surf > 0 and C_vol > 0:
                if c < C_vol:
                    # Volume channel: valid on first N_vol points only
                    p_c = p_arr[:, c, :N_vol].reshape(-1)
                    t_c = t_arr[:, c, :N_vol].reshape(-1)
                else:
                    # Surface channel: valid on last N_surf points only
                    p_c = p_arr[:, c, N_vol:].reshape(-1)
                    t_c = t_arr[:, c, N_vol:].reshape(-1)
            else:
                p_c = p_arr[:, c, :].reshape(-1)
                t_c = t_arr[:, c, :].reshape(-1)

            # Filter out non-finite values (NaN/Inf)
            filter_mask = np.isfinite(p_c) & np.isfinite(t_c)

            p_valid = p_c[filter_mask]
            t_valid = t_c[filter_mask]

            # MSE
            mse_c = float(np.mean((p_valid - t_valid) ** 2)) if p_valid.size > 0 else float("nan")

            # rel L2
            num = float(np.sum((p_valid - t_valid) ** 2))
            den = float(np.sum(t_valid**2))
            rel_l2_c = float(np.sqrt(num / den) * 100.0) if den > 0.0 else float("nan")

            # rel L1 (median of pointwise relative abs error in percent)
            non_pad_mask = ~((p_valid == 0.0) & (t_valid == 0.0))
            if np.any(non_pad_mask):
                denom = np.abs(t_valid[non_pad_mask]) + self.eps
                rel_l1_pts = np.abs(p_valid[non_pad_mask] - t_valid[non_pad_mask]) / denom
                rel_l1_c = float(np.median(rel_l1_pts) * 100.0)
            else:
                rel_l1_c = float("nan")

            # Maximum absolute error
            mae_c = float(np.max(np.abs(p_valid - t_valid))) if p_valid.size > 0 else float("nan")

            mse_list.append(mse_c)
            rel_l2_list.append(rel_l2_c)
            rel_l1_list.append(rel_l1_c)
            mae_list.append(mae_c)

        return {"mse": mse_list, "rel_l2": rel_l2_list, "rel_l1": rel_l1_list, "mae": mae_list}


    @staticmethod
    def _is_distributed(trainer: Trainer) -> bool:
        return int(getattr(trainer, "world_size", 1)) > 1

    @staticmethod
    def _new_running_stats(channel_count: int) -> Dict[str, Dict[str, np.ndarray]]:
        def _empty() -> Dict[str, np.ndarray]:
            return {
                "count": np.zeros(channel_count, dtype=np.float64),
                "sum": np.zeros(channel_count, dtype=np.float64),
                "sumsq": np.zeros(channel_count, dtype=np.float64),
            }

        base_metrics = ("mse", "rel_l2", "rel_l1", "mae")
        metric_keys = (*base_metrics, *(f"{k}_norm" for k in base_metrics))
        return {k: _empty() for k in metric_keys}

    @staticmethod
    def _update_running_stats(stats: Dict[str, Dict[str, np.ndarray]], key: str, values: List[float]) -> None:
        arr = np.asarray(values, dtype=np.float64)
        finite = np.isfinite(arr)
        if not np.any(finite):
            return
        stats[key]["count"][finite] += 1.0
        stats[key]["sum"][finite] += arr[finite]
        stats[key]["sumsq"][finite] += arr[finite] ** 2

    @staticmethod
    def _reduce_running_stats(trainer: Trainer, stats: Dict[str, Dict[str, np.ndarray]], device: torch.device) -> None:
        if int(getattr(trainer, "world_size", 1)) <= 1:
            return
        for metric_parts in stats.values():
            for part_name, arr in metric_parts.items():
                t = torch.as_tensor(arr, dtype=torch.float64, device=device)
                t = trainer.strategy.reduce(t, reduce_op="sum")
                metric_parts[part_name] = t.detach().cpu().numpy()

    @staticmethod
    def _finalize_mean_std(count: np.ndarray, sum_: np.ndarray, sumsq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mean = np.full_like(sum_, np.nan, dtype=np.float64)
        std = np.full_like(sum_, np.nan, dtype=np.float64)
        valid = count > 0.0
        if np.any(valid):
            mean[valid] = sum_[valid] / count[valid]
            var = (sumsq[valid] / count[valid]) - (mean[valid] ** 2)
            var = np.maximum(var, 0.0)
            std[valid] = np.sqrt(var)
        return mean, std

    @rank_zero_only
    def _print_and_log(self, trainer: Trainer, metrics: Dict[str, float]) -> None:
        # console print
        print("=== ComputeCarMetrics (test) ===")
        for k, v in sorted(metrics.items()):
            print(f"  {k}: {v}")

        # write final scalars to logger run summary (preferred for single final values)
        exp = getattr(trainer.logger, "experiment", None) if hasattr(trainer, "logger") else None
        if exp is not None and hasattr(exp, "summary"):
            try:
                # update summary keys (overwrites previous)
                for k, v in metrics.items():
                    exp.summary[k] = v
                # flush if available (wandb)
                if hasattr(exp, "flush"):
                    exp.flush()
            except Exception as e:
                log.warning("ComputeCarMetrics: failed to write to experiment.summary: %s", e)
        else:
            # fallback: try logger.log_metrics once, but avoid repeated calls elsewhere
            try:
                if hasattr(trainer, "logger") and trainer.logger is not None:
                    trainer.logger.log_metrics(metrics)
            except Exception:
                pass

    def on_test_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        # ensure dataloader ready
        try:
            self.setup_test_dataloader(trainer)
        except Exception as e:
            rank_zero_warn(f"ComputeCarMetrics: failed to setup test dataloader: {e}")
            return

        dataloader = getattr(trainer, "test_dataloaders", None)
        if dataloader is None:
            rank_zero_warn("ComputeCarMetrics: no test dataloader found.")
            return

        generator = torch.Generator(device=pl_module.device).manual_seed(trainer.global_rank)
        loaders = dataloader if isinstance(dataloader, (list, tuple)) else [dataloader]

        original_states = {}
        original_num_workers = {}

        if self.chunk_size is not None:
            def _disable_subsampling_and_save(dataset):
                if dataset is None:
                    return
                if hasattr(dataset, "datasets"):
                    for sub_ds in dataset.datasets:
                        _disable_subsampling_and_save(sub_ds)
                if hasattr(dataset, "dataset"):
                    _disable_subsampling_and_save(dataset.dataset)
                if id(dataset) not in original_states:
                    original_states[id(dataset)] = (dataset, getattr(dataset, "disable_subsampling", None))
                setattr(dataset, "disable_subsampling", True)

            for dl in loaders:
                _disable_subsampling_and_save(getattr(dl, "dataset", None))

        # Always force the test dataloaders to run with batch_size 1 and num_workers 0
        from src.data.loader.dataloader import UPDEDataloader
        new_loaders = []
        for dl in loaders:
            original_num_workers[id(dl)] = (dl, dl.num_workers)
            kwargs = {
                "batch_size": 1,
                "num_workers": 0,
                "pin_memory": dl.pin_memory,
                "drop_last": dl.drop_last,
                "timeout": dl.timeout,
                "worker_init_fn": dl.worker_init_fn,
                "collate_fn": getattr(dl, "collate_fn", None),
            }
            if isinstance(dl, UPDEDataloader):
                new_dl = UPDEDataloader(dl.dataset, **kwargs)
            else:
                new_dl = torch.utils.data.DataLoader(dl.dataset, **kwargs)
            new_loaders.append(new_dl)

        if hasattr(trainer, "test_loop") and trainer.test_loop is not None:
            try:
                trainer.test_loop.dataloaders = new_loaders
            except Exception:
                try:
                    trainer.test_loop._dataloaders = new_loaders
                except Exception:
                    pass
        loaders = new_loaders

        try:
            normalize = None
            try:
                batch = next(iter(loaders[0]))
                normalize = {
                    "mean": batch.loading_metadata.val_subtract.to(pl_module.device),
                    "std": batch.loading_metadata.val_divide.to(pl_module.device),
                }
            except Exception as e:
                rank_zero_warn(f"ComputeCarMetrics: failed to extract normalization parameters: {e}")

            channel_count: Optional[int] = None
            stats: Optional[Dict[str, Dict[str, np.ndarray]]] = None

            try:
                for dl in loaders:
                    total_samples = len(dl)
                    for idx, batch in enumerate(dl):
                        print(f"Processing sample {idx + 1}/{total_samples}...")
                        try:
                            if self.chunk_size is not None:
                                prediction, target = pl_module.predict_chunks(batch, generator=generator, chunk_size=self.chunk_size, **self.model_kwargs)
                            else:
                                prediction, target = pl_module.predict(batch, generator=generator, **self.model_kwargs)
                        except Exception as e:
                            rank_zero_warn(f"ComputeCarMetrics: pl_module.predict failed on a batch: {e}")
                            continue

                        # get values and batch size (support batched outputs)
                        p_vals = prediction.values if hasattr(prediction, "values") else prediction
                        B = int(p_vals.shape[0]) if isinstance(p_vals, torch.Tensor) and p_vals.ndim >= 3 else 1

                        for i in range(B):
                            # prepare single-sample objects (keep original structure where possible)
                            if hasattr(prediction, "values"):
                                p_single = _replace_unstructured_values(prediction, prediction.values[i : i + 1])
                            else:
                                p_single = prediction[i : i + 1]

                            if hasattr(target, "values"):
                                t_single = _replace_unstructured_values(target, target.values[i : i + 1])
                            else:
                                t_single = target[i : i + 1]

                            metrics_pair = self._compute_metrics_for_pair(p_single, t_single, batch=batch)

                            normalize_i = None
                            if normalize is not None:
                                mean = normalize["mean"]
                                std = normalize["std"]
                                if mean.ndim >= 1 and mean.shape[0] > i:
                                    mean = mean[i : i + 1]
                                    std = std[i : i + 1]
                                normalize_i = {"mean": mean, "std": std}

                            metrics_pair_norm = self._compute_metrics_for_pair(p_single, t_single, normalize=normalize_i, batch=batch)


                            C = len(metrics_pair["mse"])
                            if channel_count is None:
                                channel_count = C
                                stats = self._new_running_stats(C)
                            elif channel_count != C:
                                rank_zero_warn("ComputeCarMetrics: inconsistent channel count; skipping sample.")
                                continue

                            assert stats is not None
                            for key in ("mse", "rel_l2", "rel_l1", "mae"):
                                self._update_running_stats(stats, key, metrics_pair[key])
                                self._update_running_stats(stats, f"{key}_norm", metrics_pair_norm[key])
            except Exception as e:
                rank_zero_warn(f"ComputeCarMetrics: error during iteration: {e}")

            # Make channel count globally consistent across ranks.
            local_c = int(channel_count) if channel_count is not None else 0
            if self._is_distributed(trainer):
                c_tensor = torch.tensor([local_c], dtype=torch.int64, device=pl_module.device)
                gathered_c = trainer.strategy.all_gather(c_tensor)
                global_c = int(gathered_c.max().item())
            else:
                global_c = local_c

            if global_c == 0:
                rank_zero_warn("ComputeCarMetrics: no samples processed; nothing to report.")
                return

            if stats is None:
                stats = self._new_running_stats(global_c)
            elif channel_count != global_c:
                rank_zero_warn("ComputeCarMetrics: channel count differs across ranks; local stats will be padded/trimmed to match global.")
                resized = self._new_running_stats(global_c)
                keep = min(channel_count or 0, global_c)
                for k in stats.keys():
                    for part in ["count", "sum", "sumsq"]:
                        resized[k][part][:keep] = stats[k][part][:keep]
                stats = resized

            # Reduce count/sum/sumsq over all ranks to compute true global mean/std.
            self._reduce_running_stats(trainer, stats, device=pl_module.device)

            def _finalized(key: str) -> tuple[np.ndarray, np.ndarray]:
                return self._finalize_mean_std(
                    stats[key]["count"],
                    stats[key]["sum"],
                    stats[key]["sumsq"],
                )

            metrics_out: Dict[str, float] = {}
            finalized = {
                key: _finalized(key)
                for key in ("mse", "rel_l2", "rel_l1", "mae", "mse_norm", "rel_l2_norm", "rel_l1_norm", "mae_norm")
            }

            def _add_output(c: int, key: str, mean_name: str, std_name: str) -> None:
                mean_arr, std_arr = finalized[key]
                metrics_out[mean_name.format(c=c)] = float(mean_arr[c])
                metrics_out[std_name.format(c=c)] = float(std_arr[c])

            for c in range(global_c):
                _add_output(c, "mse", "test/mse_channel_{c}_mean", "test/mse_channel_{c}_std")
                _add_output(c, "rel_l2", "test/rel_l2_channel_{c}_mean_percent", "test/rel_l2_channel_{c}_std_percent")
                _add_output(
                    c,
                    "rel_l1",
                    "test/rel_l1_channel_{c}_median_mean_percent",
                    "test/rel_l1_channel_{c}_median_std_percent",
                )
                _add_output(c, "mae", "test/mae_channel_{c}_mean", "test/mae_channel_{c}_std")
                metrics_out[f"test/num_samples_channel_{c}"] = int(stats["mse"]["count"][c])

                _add_output(c, "mse_norm", "test/mse_channel_{c}_mean_normalized", "test/mse_channel_{c}_std_normalized")
                _add_output(
                    c,
                    "rel_l2_norm",
                    "test/rel_l2_channel_{c}_mean_percent_normalized",
                    "test/rel_l2_channel_{c}_std_percent_normalized",
                )
                _add_output(
                    c,
                    "rel_l1_norm",
                    "test/rel_l1_channel_{c}_median_mean_percent_normalized",
                    "test/rel_l1_channel_{c}_median_std_percent_normalized",
                )
                _add_output(
                    c,
                    "mae_norm",
                    "test/mae_channel_{c}_mean_normalized",
                    "test/mae_channel_{c}_std_normalized",
                )

            # Compute channel averages across all channels
            for key, base_name in [
                ("mse", "test/mse_mean_all_channels"),
                ("rel_l2", "test/rel_l2_mean_percent_all_channels"),
                ("rel_l1", "test/rel_l1_median_mean_percent_all_channels"),
                ("mae", "test/mae_mean_all_channels"),
                ("mse_norm", "test/mse_mean_normalized_all_channels"),
                ("rel_l2_norm", "test/rel_l2_mean_percent_normalized_all_channels"),
                ("rel_l1_norm", "test/rel_l1_median_mean_percent_normalized_all_channels"),
                ("mae_norm", "test/mae_mean_normalized_all_channels"),
            ]:
                mean_arr, _ = finalized[key]
                valid_vals = mean_arr[~np.isnan(mean_arr)]
                if len(valid_vals) > 0:
                    metrics_out[base_name] = float(np.mean(valid_vals))
            if "test/rel_l2_mean_percent_all_channels" in metrics_out:
                metrics_out["test/avg_rel_l2_percent"] = metrics_out["test/rel_l2_mean_percent_all_channels"]
            if "test/rel_l2_mean_percent_normalized_all_channels" in metrics_out:
                metrics_out["test/avg_rel_l2_percent_normalized"] = metrics_out["test/rel_l2_mean_percent_normalized_all_channels"]
            if "test/mse_mean_all_channels" in metrics_out:
                metrics_out["test/avg_mse"] = metrics_out["test/mse_mean_all_channels"]

            self._print_and_log(trainer, metrics_out)

        finally:
            # Restore original states
            for dataset, state in original_states.values():
                if state is None:
                    if hasattr(dataset, "disable_subsampling"):
                        delattr(dataset, "disable_subsampling")
                else:
                    setattr(dataset, "disable_subsampling", state)
            for dl, num_workers in original_num_workers.values():
                dl.num_workers = num_workers


def _replace_unstructured_values(orig: Any, new_values: Any) -> Any:
    """
    Return a shallow copy of orig where .values is replaced by new_values.
    Works with dataclass-like UnstructuredData; falls back to a simple object.
    """
    try:
        return replace(orig, values=new_values)
    except Exception:

        class _Tmp:
            pass

        tmp = _Tmp()
        for k in ["values", "pos", "attributes", "channels_last"]:
            if hasattr(orig, k):
                setattr(tmp, k, getattr(orig, k))
        setattr(tmp, "values", new_values)
        return tmp



class ComputeCrashMetrics(Callback):
    """
    Compute per-channel MSE, relative L2 (%), median relative L1 (%),
    and maximum absolute error over the test set for crash_steady data.

    - Runs on_test_start; iterates trainer.test_dataloaders once.
    - Designed specifically for crash_steady data (surface only, 4 channels: disp_x, disp_y, disp_z, stress_vm).
    - Does NOT apply masking of zero values since zero displacements/stresses are physical.
    - Aggregates per-channel values across all test samples and reports mean/std.
    - Under DDP, aggregates statistics across all ranks.
    - Writes final scalars to the logger run summary (e.g. wandb.summary).
    """

    def __init__(self, batch_size_inference: int = 1, eps: float = 1e-8, chunk_size: int = 65536, **model_kwargs) -> None:
        super().__init__()
        self.batch_size_inference = int(batch_size_inference)
        self.eps = float(eps)
        if isinstance(chunk_size, str) and chunk_size.lower() == "none":
            self.chunk_size = None
        else:
            self.chunk_size = chunk_size
        self.model_kwargs = model_kwargs

    def setup_test_dataloader(self, trainer: Trainer) -> None:
        td = getattr(trainer, "test_dataloaders", None)
        if td is None:
            try:
                trainer.test_loop.setup_data()
            except Exception:
                pass

    def _compute_metrics_for_pair(self, pred: Any, tgt: Any, normalize: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, List[float]]:
        # extract arrays
        p = pred.values if hasattr(pred, "values") else pred
        t = tgt.values if hasattr(tgt, "values") else tgt

        p = np.asarray(torch.as_tensor(p).detach().cpu())  # numpy array
        t = np.asarray(torch.as_tensor(t).detach().cpu())

        # Normalize if mean and std are provided
        if normalize:
            mean = normalize["mean"].unsqueeze(1).unsqueeze(-1)
            std = normalize["std"].unsqueeze(1).unsqueeze(-1)
            mean = mean.cpu().numpy()
            std = std.cpu().numpy()
            p = (p - mean) / std
            t = (t - mean) / std

        # normalize shapes to (T, C, N)
        def to_tcn(x: np.ndarray) -> np.ndarray:
            if x.ndim == 4 and x.shape[0] == 1:
                return x[0]  # (T, C, N)
            if x.ndim == 3:
                if x.shape[0] == 1:
                    return x[0][None, ...]  # (T=1, C, N)
                return x[None, ...]
            if x.ndim == 2:
                return x[None, ...]
            raise ValueError(f"Unsupported tensor shape {x.shape} for metrics computation")

        p_arr = to_tcn(p)
        t_arr = to_tcn(t)

        # align time dim
        T = min(p_arr.shape[0], t_arr.shape[0])
        p_arr = p_arr[:T]
        t_arr = t_arr[:T]

        # now shapes: (T, C, N)
        _, C, _ = p_arr.shape

        mse_list: List[float] = []
        rel_l2_list: List[float] = []
        rel_l1_list: List[float] = []
        mae_list: List[float] = []

        for c in range(C):
            p_c = p_arr[:, c, :].reshape(-1)
            t_c = t_arr[:, c, :].reshape(-1)

            # Compute metrics for all points (no filter mask)
            mse_c = float(np.mean((p_c - t_c) ** 2))

            # rel L2
            num = float(np.sum((p_c - t_c) ** 2))
            den = float(np.sum(t_c**2))
            rel_l2_c = float(np.sqrt(num / den) * 100.0) if den > 0.0 else float("nan")

            # rel L1 (median of pointwise relative abs error in percent)
            denom = np.abs(t_c) + self.eps
            rel_l1_pts = np.abs(p_c - t_c) / denom
            rel_l1_c = float(np.median(rel_l1_pts) * 100.0)

            # Maximum absolute error
            mae_c = float(np.max(np.abs(p_c - t_c))) if p_c.size > 0 else float("nan")

            mse_list.append(mse_c)
            rel_l2_list.append(rel_l2_c)
            rel_l1_list.append(rel_l1_c)
            mae_list.append(mae_c)

        return {"mse": mse_list, "rel_l2": rel_l2_list, "rel_l1": rel_l1_list, "mae": mae_list}

    @staticmethod
    def _is_distributed(trainer: Trainer) -> bool:
        return int(getattr(trainer, "world_size", 1)) > 1

    @staticmethod
    def _new_running_stats(channel_count: int) -> Dict[str, Dict[str, np.ndarray]]:
        def _empty() -> Dict[str, np.ndarray]:
            return {
                "count": np.zeros(channel_count, dtype=np.float64),
                "sum": np.zeros(channel_count, dtype=np.float64),
                "sumsq": np.zeros(channel_count, dtype=np.float64),
            }

        base_metrics = ("mse", "rel_l2", "rel_l1", "mae")
        metric_keys = (*base_metrics, *(f"{k}_norm" for k in base_metrics))
        return {k: _empty() for k in metric_keys}

    @staticmethod
    def _update_running_stats(stats: Dict[str, Dict[str, np.ndarray]], key: str, values: List[float]) -> None:
        arr = np.asarray(values, dtype=np.float64)
        finite = np.isfinite(arr)
        if not np.any(finite):
            return
        stats[key]["count"][finite] += 1.0
        stats[key]["sum"][finite] += arr[finite]
        stats[key]["sumsq"][finite] += arr[finite] ** 2

    @staticmethod
    def _reduce_running_stats(trainer: Trainer, stats: Dict[str, Dict[str, np.ndarray]], device: torch.device) -> None:
        if int(getattr(trainer, "world_size", 1)) <= 1:
            return
        for metric_parts in stats.values():
            for part_name, arr in metric_parts.items():
                t = torch.as_tensor(arr, dtype=torch.float64, device=device)
                t = trainer.strategy.reduce(t, reduce_op="sum")
                metric_parts[part_name] = t.detach().cpu().numpy()

    @staticmethod
    def _finalize_mean_std(count: np.ndarray, sum_: np.ndarray, sumsq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mean = np.full_like(sum_, np.nan, dtype=np.float64)
        std = np.full_like(sum_, np.nan, dtype=np.float64)
        valid = count > 0.0
        if np.any(valid):
            mean[valid] = sum_[valid] / count[valid]
            var = (sumsq[valid] / count[valid]) - (mean[valid] ** 2)
            var = np.maximum(var, 0.0)
            std[valid] = np.sqrt(var)
        return mean, std

    @rank_zero_only
    def _print_and_log(self, trainer: Trainer, metrics: Dict[str, float]) -> None:
        # console print
        print("=== ComputeCrashMetrics (test) ===")
        for k, v in sorted(metrics.items()):
            print(f"  {k}: {v}")

        # write final scalars to logger run summary
        exp = getattr(trainer.logger, "experiment", None) if hasattr(trainer, "logger") else None
        if exp is not None and hasattr(exp, "summary"):
            try:
                for k, v in metrics.items():
                    exp.summary[k] = v
                if hasattr(exp, "flush"):
                    exp.flush()
            except Exception as e:
                log.warning("ComputeCrashMetrics: failed to write to experiment.summary: %s", e)
        else:
            try:
                if hasattr(trainer, "logger") and trainer.logger is not None:
                    trainer.logger.log_metrics(metrics)
            except Exception:
                pass

    def on_test_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        try:
            self.setup_test_dataloader(trainer)
        except Exception as e:
            rank_zero_warn(f"ComputeCrashMetrics: failed to setup test dataloader: {e}")
            return

        dataloader = getattr(trainer, "test_dataloaders", None)
        if dataloader is None:
            rank_zero_warn("ComputeCrashMetrics: no test dataloader found.")
            return

        generator = torch.Generator(device=pl_module.device).manual_seed(trainer.global_rank)
        loaders = dataloader if isinstance(dataloader, (list, tuple)) else [dataloader]

        original_states = {}
        original_num_workers = {}

        if self.chunk_size is not None:
            def _disable_subsampling_and_save(dataset):
                if dataset is None:
                    return
                if hasattr(dataset, "datasets"):
                    for sub_ds in dataset.datasets:
                        _disable_subsampling_and_save(sub_ds)
                if hasattr(dataset, "dataset"):
                    _disable_subsampling_and_save(dataset.dataset)
                if id(dataset) not in original_states:
                    original_states[id(dataset)] = (dataset, getattr(dataset, "disable_subsampling", None))
                setattr(dataset, "disable_subsampling", True)

            for dl in loaders:
                _disable_subsampling_and_save(getattr(dl, "dataset", None))

        # Always force the test dataloaders to run with batch_size 1 and num_workers 0
        from src.data.loader.dataloader import UPDEDataloader
        new_loaders = []
        for dl in loaders:
            original_num_workers[id(dl)] = (dl, dl.num_workers)
            kwargs = {
                "batch_size": 1,
                "num_workers": 0,
                "pin_memory": dl.pin_memory,
                "drop_last": dl.drop_last,
                "timeout": dl.timeout,
                "worker_init_fn": dl.worker_init_fn,
                "collate_fn": getattr(dl, "collate_fn", None),
            }
            if isinstance(dl, UPDEDataloader):
                new_dl = UPDEDataloader(dl.dataset, **kwargs)
            else:
                new_dl = torch.utils.data.DataLoader(dl.dataset, **kwargs)
            new_loaders.append(new_dl)

        if hasattr(trainer, "test_loop") and trainer.test_loop is not None:
            try:
                trainer.test_loop.dataloaders = new_loaders
            except Exception:
                try:
                    trainer.test_loop._dataloaders = new_loaders
                except Exception:
                    pass
        loaders = new_loaders

        try:
            normalize = None
            try:
                batch = next(iter(loaders[0]))
                normalize = {
                    "mean": batch.loading_metadata.val_subtract.to(pl_module.device),
                    "std": batch.loading_metadata.val_divide.to(pl_module.device),
                }
            except Exception as e:
                rank_zero_warn(f"ComputeCrashMetrics: failed to extract normalization parameters: {e}")

            channel_count: Optional[int] = None
            stats: Optional[Dict[str, Dict[str, np.ndarray]]] = None

            try:
                for dl in loaders:
                    total_samples = len(dl)
                    for idx, batch in enumerate(dl):
                        print(f"Processing sample {idx + 1}/{total_samples}...")
                        try:
                            if self.chunk_size is not None:
                                prediction, target = pl_module.predict_chunks(batch, generator=generator, chunk_size=self.chunk_size, **self.model_kwargs)
                            else:
                                prediction, target = pl_module.predict(batch, generator=generator, **self.model_kwargs)
                        except Exception as e:
                            rank_zero_warn(f"ComputeCrashMetrics: pl_module.predict failed on a batch: {e}")
                            continue

                        p_vals = prediction.values if hasattr(prediction, "values") else prediction
                        B = int(p_vals.shape[0]) if isinstance(p_vals, torch.Tensor) and p_vals.ndim >= 3 else 1

                        for i in range(B):
                            if hasattr(prediction, "values"):
                                p_single = _replace_unstructured_values(prediction, prediction.values[i : i + 1])
                            else:
                                p_single = prediction[i : i + 1]

                            if hasattr(target, "values"):
                                t_single = _replace_unstructured_values(target, target.values[i : i + 1])
                            else:
                                t_single = target[i : i + 1]

                            metrics_pair = self._compute_metrics_for_pair(p_single, t_single)

                            normalize_i = None
                            if normalize is not None:
                                mean = normalize["mean"]
                                std = normalize["std"]
                                if mean.ndim >= 1 and mean.shape[0] > i:
                                    mean = mean[i : i + 1]
                                    std = std[i : i + 1]
                                normalize_i = {"mean": mean, "std": std}

                            metrics_pair_norm = self._compute_metrics_for_pair(p_single, t_single, normalize=normalize_i)

                            C = len(metrics_pair["mse"])
                            if channel_count is None:
                                channel_count = C
                                stats = self._new_running_stats(C)
                            elif channel_count != C:
                                rank_zero_warn("ComputeCrashMetrics: inconsistent channel count; skipping sample.")
                                continue

                            assert stats is not None
                            for key in ("mse", "rel_l2", "rel_l1", "mae"):
                                self._update_running_stats(stats, key, metrics_pair[key])
                                self._update_running_stats(stats, f"{key}_norm", metrics_pair_norm[key])
            except Exception as e:
                rank_zero_warn(f"ComputeCrashMetrics: error during iteration: {e}")

            local_c = int(channel_count) if channel_count is not None else 0
            if self._is_distributed(trainer):
                c_tensor = torch.tensor([local_c], dtype=torch.int64, device=pl_module.device)
                gathered_c = trainer.strategy.all_gather(c_tensor)
                global_c = int(gathered_c.max().item())
            else:
                global_c = local_c

            if global_c == 0:
                rank_zero_warn("ComputeCrashMetrics: no samples processed; nothing to report.")
                return

            if stats is None:
                stats = self._new_running_stats(global_c)
            elif channel_count != global_c:
                rank_zero_warn("ComputeCrashMetrics: channel count differs across ranks; local stats will be padded/trimmed to match global.")
                resized = self._new_running_stats(global_c)
                keep = min(channel_count or 0, global_c)
                for k in stats.keys():
                    for part in ["count", "sum", "sumsq"]:
                        resized[k][part][:keep] = stats[k][part][:keep]
                stats = resized

            self._reduce_running_stats(trainer, stats, device=pl_module.device)

            def _finalized(key: str) -> tuple[np.ndarray, np.ndarray]:
                return self._finalize_mean_std(
                    stats[key]["count"],
                    stats[key]["sum"],
                    stats[key]["sumsq"],
                )

            metrics_out: Dict[str, float] = {}
            finalized = {
                key: _finalized(key)
                for key in ("mse", "rel_l2", "rel_l1", "mae", "mse_norm", "rel_l2_norm", "rel_l1_norm", "mae_norm")
            }

            def _add_output(c_name: str, c: int, key: str, mean_name: str, std_name: str) -> None:
                mean_arr, std_arr = finalized[key]
                metrics_out[mean_name.format(c=c_name)] = float(mean_arr[c])
                metrics_out[std_name.format(c=c_name)] = float(std_arr[c])

            channel_names = ["disp_x", "disp_y", "disp_z", "stress_vm"]
            for c in range(global_c):
                c_name = channel_names[c] if c < len(channel_names) else f"channel_{c}"
                _add_output(c_name, c, "mse", "test/mse_{c}_mean", "test/mse_{c}_std")
                _add_output(c_name, c, "rel_l2", "test/rel_l2_{c}_mean_percent", "test/rel_l2_{c}_std_percent")
                _add_output(
                    c_name,
                    c,
                    "rel_l1",
                    "test/rel_l1_{c}_median_mean_percent",
                    "test/rel_l1_{c}_median_std_percent",
                )
                _add_output(c_name, c, "mae", "test/mae_{c}_mean", "test/mae_{c}_std")
                metrics_out[f"test/num_samples_{c_name}"] = int(stats["mse"]["count"][c])

                _add_output(c_name, c, "mse_norm", "test/mse_{c}_mean_normalized", "test/mse_{c}_std_normalized")
                _add_output(
                    c_name,
                    c,
                    "rel_l2_norm",
                    "test/rel_l2_{c}_mean_percent_normalized",
                    "test/rel_l2_{c}_std_percent_normalized",
                )
                _add_output(
                    c_name,
                    c,
                    "rel_l1_norm",
                    "test/rel_l1_{c}_median_mean_percent_normalized",
                    "test/rel_l1_{c}_median_std_percent_normalized",
                )
                _add_output(
                    c_name,
                    c,
                    "mae_norm",
                    "test/mae_{c}_mean_normalized",
                    "test/mae_{c}_std_normalized",
                )

            # Compute channel averages across all channels
            for key, base_name in [
                ("mse", "test/mse_mean_all_channels"),
                ("rel_l2", "test/rel_l2_mean_percent_all_channels"),
                ("rel_l1", "test/rel_l1_median_mean_percent_all_channels"),
                ("mae", "test/mae_mean_all_channels"),
                ("mse_norm", "test/mse_mean_normalized_all_channels"),
                ("rel_l2_norm", "test/rel_l2_mean_percent_normalized_all_channels"),
                ("rel_l1_norm", "test/rel_l1_median_mean_percent_normalized_all_channels"),
                ("mae_norm", "test/mae_mean_normalized_all_channels"),
            ]:
                mean_arr, _ = finalized[key]
                valid_vals = mean_arr[~np.isnan(mean_arr)]
                if len(valid_vals) > 0:
                    metrics_out[base_name] = float(np.mean(valid_vals))
            if "test/rel_l2_mean_percent_all_channels" in metrics_out:
                metrics_out["test/avg_rel_l2_percent"] = metrics_out["test/rel_l2_mean_percent_all_channels"]
            if "test/rel_l2_mean_percent_normalized_all_channels" in metrics_out:
                metrics_out["test/avg_rel_l2_percent_normalized"] = metrics_out["test/rel_l2_mean_percent_normalized_all_channels"]
            if "test/mse_mean_all_channels" in metrics_out:
                metrics_out["test/avg_mse"] = metrics_out["test/mse_mean_all_channels"]

            self._print_and_log(trainer, metrics_out)

        finally:
            for dataset, state in original_states.values():
                if state is None:
                    if hasattr(dataset, "disable_subsampling"):
                        delattr(dataset, "disable_subsampling")
                else:
                    setattr(dataset, "disable_subsampling", state)
            for dl, num_workers in original_num_workers.values():
                dl.num_workers = num_workers

