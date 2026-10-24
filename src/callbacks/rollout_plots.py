import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import wandb
from lightning import Callback, LightningModule, Trainer
from lightning.fabric.utilities import rank_zero_only
from lightning.pytorch.loggers.wandb import WandbLogger
from matplotlib import pyplot as plt
from torchmetrics.utilities import rank_zero_info, rank_zero_warn

from src.modules.unstructured_window import UnstructuredData

log = logging.getLogger(__name__)


class UnstructuredRollout2DVisualization(Callback):

    def __init__(self, frequency: int, max_visualizations: int = 4, rollout_num: int = 20, steps_plot: int = 5, test_only: bool = False, disabled: bool = False, **kwargs) -> None:

        super().__init__()

        self.frequency = frequency
        self.max_visualizations = max_visualizations
        self.test_only = test_only
        self.rollout_num = rollout_num
        self.num_frames = rollout_num
        self.steps_plot = steps_plot
        self.time_steps = np.arange(0, self.rollout_num, self.steps_plot)
        self.disabled = disabled
        self.model_kwargs = kwargs

    def setup_test_dataloader(self, trainer):
        test_dataloader = trainer.test_dataloaders
        if test_dataloader is None:
            trainer.test_loop.setup_data()
            _ = trainer.test_dataloaders

    def visualization_2d_unstructured(self, prediction: UnstructuredData, target: UnstructuredData, time_steps: np.ndarray, stage: str, sample_idx: int, savepath: Path) -> Dict[str, Any]:

        dict_ = {}

        plt.rcParams.update({"font.size": 16, "axes.titlesize": 16})

        fig, ax = plt.subplots(nrows=3, ncols=time_steps.shape[0], figsize=(3 * time_steps.shape[0], 9))
        plt.subplots_adjust(wspace=0.2, hspace=0.3)

        for t_idx, t in enumerate(time_steps):

            t_int = int(t)

            pred_vals = prediction.values[0, t_int].cpu().numpy()
            target_vals = target.values[0, t_int].cpu().numpy()

            pred_pos = prediction.pos[0, t_int].cpu().numpy()

            vmin = target_vals[0].min()
            vmax = target_vals[0].max()

            sc0 = ax[0, t_idx].scatter(pred_pos[0], pred_pos[1], c=pred_vals[0], s=2, cmap="twilight", vmin=vmin, vmax=vmax)
            ax[0, t_idx].set_aspect("equal")
            ax[0, t_idx].axis("off")

            sc1 = ax[1, t_idx].scatter(pred_pos[0], pred_pos[1], c=target_vals[0], s=2, cmap="twilight", vmin=vmin, vmax=vmax)
            ax[1, t_idx].set_aspect("equal")
            ax[1, t_idx].axis("off")

            # vmax = np.max(np.abs((pred_vals[2] - target_vals[2])))
            sc2 = ax[2, t_idx].scatter(pred_pos[0], pred_pos[1], c=pred_vals[0] - target_vals[0], s=2, cmap="RdBu_r", vmin=vmin, vmax=-vmin)
            ax[2, t_idx].set_aspect("equal")
            ax[2, t_idx].axis("off")

            ax[1, t_idx].set_title(f"RMSE: {np.sqrt(np.mean((pred_vals[0] - target_vals[0])**2)):.3f}", fontsize=24)
            ax[0, t_idx].set_title(f"t={t_int}", fontsize=24)

        cbars = [sc0, sc1, sc2]
        for i_cb, cbar in enumerate(cbars):
            pos = ax[i_cb, -1].get_position()
            pad = (pos.x1 - pos.x0) * 0.05
            cbar_width = (pos.x1 - pos.x0) * 0.1
            cax = fig.add_axes([pos.x1 + pad, pos.y0, cbar_width, pos.y1 - pos.y0])
            plt.colorbar(cbar, cax=cax)

            # make the ticks bigger
            cax.tick_params(labelsize=24)

        ax[0, 0].text(-0.1, 0.5, "Predicted", ha="center", va="center", transform=ax[0, 0].transAxes, rotation=90, fontsize=24)
        ax[1, 0].text(-0.1, 0.5, "Reference", ha="center", va="center", transform=ax[1, 0].transAxes, rotation=90, fontsize=24)
        ax[2, 0].text(-0.1, 0.5, "Error", ha="center", va="center", transform=ax[2, 0].transAxes, rotation=90, fontsize=24)

        # save figure
        fig_path = os.path.join(savepath, f"{stage}/sample{sample_idx}/rollout.png")
        os.makedirs(os.path.dirname(fig_path), exist_ok=True)
        fig.savefig(fig_path, bbox_inches="tight")
        plt.close(fig)

        dict_[f"{stage}/sample{sample_idx}/rollout"] = fig

        # save prediction.values, target.values, and prediction.pos as npy arrays for the first sample and all time steps
        print(stage)
        np.save(os.path.join(savepath, f"{stage}/sample{sample_idx}/prediction_values.npy"), prediction.values[0, : self.rollout_num].cpu().numpy())
        np.save(os.path.join(savepath, f"{stage}/sample{sample_idx}/target_values.npy"), target.values[0, : self.rollout_num].cpu().numpy())
        np.save(os.path.join(savepath, f"{stage}/sample{sample_idx}/prediction_pos.npy"), prediction.pos[0, : self.rollout_num].cpu().numpy())

        return dict_

    def compute_2d_unstructured(self, prediction: UnstructuredData, target: UnstructuredData, mean: torch.Tensor, std: torch.Tensor) -> Dict[str, float]:
        """Calculates RMSE for specific timesteps for a single sample."""
        metrics = {}
        mean = mean.cpu().numpy()
        std = std.cpu().numpy()
        mean = mean.reshape(-1, 1)  # (C, 1)
        std = std.reshape(-1, 1)  # (C, 1)

        # Ensure we don't go out of bounds if rollout_num is small
        for t in [1, 5, 10, 20]:
            if t < prediction.values.shape[1]:
                # Assuming shape [batch, time, nodes, features]
                # Adjust indexing if your UnstructuredData format differs
                p = prediction.values[0, t].cpu().numpy()
                gt = target.values[0, t].cpu().numpy()

                # Apply normalization per channel correctly
                p = (p - mean) / std
                gt = (gt - mean) / std

                rmse = np.sqrt(np.mean((p - gt) ** 2))
                metrics[f"rmse_{t}"] = rmse
        return metrics

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

    @rank_zero_only
    def log_video(self, trainer, pl_module, dataloader, stage=""):
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

        normalize = None
        try:
            batch = next(iter(dataloader))
            normalize = {
                "mean": batch.loading_metadata.val_subtract.to(pl_module.device),
                "std": batch.loading_metadata.val_divide.to(pl_module.device),
            }
        except Exception as e:
            rank_zero_warn(f"ComputeCarMetrics: failed to extract normalization parameters: {e}")

        mean = normalize["mean"][0]
        std = normalize["std"][0]

        try:

            savepath = Path(os.path.join(trainer.default_root_dir, "visualizations"))

            generator = torch.Generator(device=pl_module.device).manual_seed(trainer.global_rank)

            count = 0
            vis_dict = {}
            all_sample_metrics = []

            for batch in dataloader:

                if count > self.max_visualizations:
                    break

                prediction, target = pl_module.predict(batch, rollout_num=self.rollout_num, generator=generator, **self.model_kwargs)

                vis_dict.update(self.visualization_2d_unstructured(prediction=prediction, target=target, time_steps=self.time_steps, stage=stage, sample_idx=count, savepath=savepath))

                all_sample_metrics.append(self.compute_2d_unstructured(prediction, target, mean, std))

                count += 1

            if all_sample_metrics:
                avg_metrics = {}
                for key in set().union(*[set(m.keys()) for m in all_sample_metrics]):
                    vals = np.array([m.get(key, np.nan) for m in all_sample_metrics], dtype=np.float64)
                    mean_val = float(np.nanmean(vals)) if vals.size > 0 else float("nan")
                    std_val = float(np.nanstd(vals)) if vals.size > 0 else float("nan")
                    avg_metrics[f"{stage}/avg_{key}"] = mean_val
                    avg_metrics[f"{stage}/std_{key}"] = std_val

                # Log numerical metrics to console and wandb summary
                self._print_and_log(trainer, avg_metrics)
                # Add to vis_dict so they get logged in the standard log_visualizations pass
                vis_dict.update(avg_metrics)

            self.log_visualizations(trainer, vis_dict)

        except Exception as e:

            rank_zero_warn(f"Error while creating visualization: {e}")
            raise e

    @rank_zero_only
    def log_visualizations(self, trainer: Trainer, vis_dict: Dict[str, Any]):

        # iterate through all loggers
        for logger in trainer.loggers:
            # check if logger is wandb

            print("Logging visualizations to logger: ", logger)

            if isinstance(logger, WandbLogger):
                data_dict = {}
                img_dict = {}
                for k, v in vis_dict.items():

                    if isinstance(v, plt.Figure):
                        # rank_zero_info('Logging figure to wandb: ', k)
                        img_dict[k] = wandb.Image(v, caption=k)
                    else:
                        # rank_zero_info('Logging data to wandb: ', k, v)
                        data_dict[k] = v

                data_dict.update(img_dict)
                logger.experiment.log(data_dict)

            else:

                logger.log_metrics(vis_dict)

    def on_test_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.setup_test_dataloader(trainer)
        if not self.disabled:
            module_stage = getattr(pl_module, "stage", "last_ema")
            self.log_video(trainer, pl_module, trainer.test_dataloaders, stage=f"{module_stage}/test")


class UnstructuredRollout3DVisualization(Callback):

    def __init__(self, frequency: int, max_visualizations: int = 4, rollout_num: int = 20, steps_plot: int = 5, test_only: bool = False, disabled: bool = False, **kwargs) -> None:

        super().__init__()

        self.frequency = frequency
        self.max_visualizations = max_visualizations
        self.test_only = test_only
        self.rollout_num = rollout_num
        self.num_frames = rollout_num
        self.steps_plot = steps_plot
        self.time_steps = np.arange(0, self.rollout_num, self.steps_plot)
        self.disabled = disabled
        self.model_kwargs = kwargs

    def setup_test_dataloader(self, trainer):
        test_dataloader = trainer.test_dataloaders
        if test_dataloader is None:
            trainer.test_loop.setup_data()
            _ = trainer.test_dataloaders

    def visualization_3d_unstructured(self, prediction: UnstructuredData, target: UnstructuredData, time_steps: np.ndarray, stage: str, sample_idx: int, savepath: Path) -> Dict[str, Any]:

        dict_ = {}

        plt.rcParams.update({"font.size": 16, "axes.titlesize": 16})

        fig = plt.figure(figsize=(3 * time_steps.shape[0], 9))
        ax = np.empty((3, time_steps.shape[0]), dtype=object)  # 3 rows, len(time_steps) columns
        for i in range(3):
            for tt in range(time_steps.shape[0]):
                ax[i, tt] = fig.add_subplot(3, time_steps.shape[0], 1 + i * time_steps.shape[0] + tt, projection="3d")
        plt.subplots_adjust(wspace=0.2, hspace=0.3)

        # err_max = torch.linalg.norm(prediction.values[0, : self.rollout_num, :3], dim=0) - torch.linalg.norm(target.values[0, : self.rollout_num, :3], dim=0)
        # err_max = torch.max(torch.abs(err_max)).item() / 5
        err_max = 100

        vmin = torch.linalg.norm(target.values[0, : self.rollout_num, :3], dim=1).cpu().numpy().min()
        vmax = torch.linalg.norm(target.values[0, : self.rollout_num, :3], dim=1).cpu().numpy().max()

        for t_idx, t in enumerate(time_steps):

            t_int = int(t)

            pred_vals = torch.linalg.norm(prediction.values[0, t_int, :3], dim=0).cpu().numpy()
            target_vals = torch.linalg.norm(target.values[0, t_int, :3], dim=0).cpu().numpy()
            diff_vals = torch.linalg.norm(prediction.values[0, t_int, :3] - target.values[0, t_int, :3], dim=0).cpu().numpy()

            pred_pos = prediction.pos[0, t_int].cpu().numpy()

            sc0 = ax[0, t_idx].scatter(pred_pos[0], pred_pos[1], pred_pos[2], c=pred_vals, s=2, cmap="twilight", vmin=vmin, vmax=vmax)
            ax[0, t_idx].set_aspect("equal")
            ax[0, t_idx].axis("off")

            sc1 = ax[1, t_idx].scatter(pred_pos[0], pred_pos[1], pred_pos[2], c=target_vals, s=2, cmap="twilight", vmin=vmin, vmax=vmax)
            ax[1, t_idx].set_aspect("equal")
            ax[1, t_idx].axis("off")

            sc2 = ax[2, t_idx].scatter(pred_pos[0], pred_pos[1], pred_pos[2], c=diff_vals, s=2, cmap="Spectral_r", vmin=0, vmax=err_max)
            ax[2, t_idx].set_aspect("equal")
            ax[2, t_idx].axis("off")

            ax[2, t_idx].set_title(f"RMSE: {np.sqrt(np.mean(diff_vals**2)):.3f}", fontsize=24)
            ax[0, t_idx].set_title(f"t={t_int}", fontsize=24)

        cbars = [sc0, sc1, sc2]
        for i_cb, cbar in enumerate(cbars):
            pos = ax[i_cb, -1].get_position()
            pad = (pos.x1 - pos.x0) * 0.05
            cbar_width = (pos.x1 - pos.x0) * 0.1
            cax = fig.add_axes([pos.x1 + pad, pos.y0, cbar_width, pos.y1 - pos.y0])
            plt.colorbar(cbar, cax=cax)

            # make the ticks bigger
            cax.tick_params(labelsize=24)

        fig.text(0.02, 0.6, "Predicted", ha="center", va="center", transform=ax[0, 0].transAxes, rotation=90, fontsize=24)
        fig.text(0.02, 0.5, "Reference", ha="center", va="center", transform=ax[1, 0].transAxes, rotation=90, fontsize=24)
        fig.text(0.02, 0.4, "Error", ha="center", va="center", transform=ax[2, 0].transAxes, rotation=90, fontsize=24)

        # save figure
        fig_path = os.path.join(savepath, f"{stage}/sample{sample_idx}/rollout.png")
        os.makedirs(os.path.dirname(fig_path), exist_ok=True)
        fig.savefig(fig_path, bbox_inches="tight")
        plt.close(fig)

        dict_[f"{stage}/sample{sample_idx}/rollout"] = fig

        # save prediction.values, target.values, and prediction.pos as npy arrays for the first sample and all time steps
        print(stage)
        np.save(os.path.join(savepath, f"{stage}/sample{sample_idx}/prediction_values.npy"), prediction.values[0, : self.rollout_num].cpu().numpy())
        np.save(os.path.join(savepath, f"{stage}/sample{sample_idx}/target_values.npy"), target.values[0, : self.rollout_num].cpu().numpy())
        np.save(os.path.join(savepath, f"{stage}/sample{sample_idx}/prediction_pos.npy"), prediction.pos[0, : self.rollout_num].cpu().numpy())

        return dict_

    def compute_3d_unstructured(self, prediction: UnstructuredData, target: UnstructuredData, mean: torch.Tensor, std: torch.Tensor) -> Dict[str, float]:
        """Calculates RMSE for specific timesteps for a single sample."""
        metrics = {}
        mean = mean.cpu().numpy()
        std = std.cpu().numpy()
        mean = mean.reshape(-1, 1)  # (C, 1)
        std = std.reshape(-1, 1)  # (C, 1)

        # Ensure we don't go out of bounds if rollout_num is small
        for t in [1, 5, 10, 20]:
            if t < prediction.values.shape[1]:
                # Assuming shape [batch, time, nodes, features]
                # Adjust indexing if your UnstructuredData format differs
                p = prediction.values[0, t].cpu().numpy()
                gt = target.values[0, t].cpu().numpy()

                # Apply normalization per channel correctly
                p = (p - mean) / std
                gt = (gt - mean) / std

                rmse = np.sqrt(np.mean((p - gt) ** 2))
                metrics[f"rmse_{t}"] = rmse
        return metrics

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

    @rank_zero_only
    def log_video(self, trainer, pl_module, dataloader, stage=""):
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

        normalize = None
        try:
            batch = next(iter(dataloader))
            normalize = {
                "mean": batch.loading_metadata.val_subtract.to(pl_module.device),
                "std": batch.loading_metadata.val_divide.to(pl_module.device),
            }
        except Exception as e:
            rank_zero_warn(f"ComputeCarMetrics: failed to extract normalization parameters: {e}")

        mean = normalize["mean"][0]
        std = normalize["std"][0]

        try:

            savepath = Path(os.path.join(trainer.default_root_dir, "visualizations"))

            generator = torch.Generator(device=pl_module.device).manual_seed(trainer.global_rank)

            count = 0
            vis_dict = {}
            all_sample_metrics = []

            for batch in dataloader:

                if count > self.max_visualizations:
                    break

                prediction, target = pl_module.predict(batch, rollout_num=self.rollout_num, generator=generator, **self.model_kwargs)

                vis_dict.update(self.visualization_3d_unstructured(prediction=prediction, target=target, time_steps=self.time_steps, stage=stage, sample_idx=count, savepath=savepath))

                all_sample_metrics.append(self.compute_3d_unstructured(prediction, target, mean, std))

                count += 1

            if all_sample_metrics:
                avg_metrics = {}
                for key in set().union(*[set(m.keys()) for m in all_sample_metrics]):
                    vals = np.array([m.get(key, np.nan) for m in all_sample_metrics], dtype=np.float64)
                    mean_val = float(np.nanmean(vals)) if vals.size > 0 else float("nan")
                    std_val = float(np.nanstd(vals)) if vals.size > 0 else float("nan")
                    avg_metrics[f"{stage}/avg_{key}"] = mean_val
                    avg_metrics[f"{stage}/std_{key}"] = std_val

                # Log numerical metrics to console and wandb summary
                self._print_and_log(trainer, avg_metrics)
                # Add to vis_dict so they get logged in the standard log_visualizations pass
                vis_dict.update(avg_metrics)

            self.log_visualizations(trainer, vis_dict)

        except Exception as e:

            rank_zero_warn(f"Error while creating visualization: {e}")
            raise e

    @rank_zero_only
    def log_visualizations(self, trainer: Trainer, vis_dict: Dict[str, Any]):

        # iterate through all loggers
        for logger in trainer.loggers:
            # check if logger is wandb

            print("Logging visualizations to logger: ", logger)

            if isinstance(logger, WandbLogger):
                data_dict = {}
                img_dict = {}
                for k, v in vis_dict.items():

                    if isinstance(v, plt.Figure):
                        # rank_zero_info('Logging figure to wandb: ', k)
                        img_dict[k] = wandb.Image(v, caption=k)
                    else:
                        # rank_zero_info('Logging data to wandb: ', k, v)
                        data_dict[k] = v

                data_dict.update(img_dict)
                logger.experiment.log(data_dict)

            else:

                logger.log_metrics(vis_dict)

    def on_test_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.setup_test_dataloader(trainer)
        if not self.disabled:
            module_stage = getattr(pl_module, "stage", "last_ema")
            self.log_video(trainer, pl_module, trainer.test_dataloaders, stage=f"{module_stage}/test")