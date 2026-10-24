import os
import warnings
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import wandb
from lightning import Callback, LightningModule, Trainer
from lightning.fabric.utilities import rank_zero_only
from lightning.pytorch.loggers.wandb import WandbLogger
from matplotlib import pyplot as plt
from torchmetrics.utilities import rank_zero_info, rank_zero_warn

from src.modules.unstructured_window import UnstructuredData


class Scatter3DVisualization(Callback):

    def __init__(self, frequency: int, max_visualizations: int = 4, test_only: bool = False, plot_surface: bool = True, plot_volume: bool = True, **kwargs) -> None:

        super().__init__()

        self.frequency = frequency
        self.max_visualizations = max_visualizations
        self.test_only = test_only
        self.plot_surface = plot_surface
        self.plot_volume = plot_volume
        self.model_kwargs = kwargs

    def setup_test_dataloader(self, trainer):
        test_dataloader = trainer.test_dataloaders
        if test_dataloader is None:
            trainer.test_loop.setup_data()
            _ = trainer.test_dataloaders

    def visualization_2d_surface_scatter(self, prediction: UnstructuredData, target: UnstructuredData, stage: str, sample_idx: int, savepath: Path) -> Dict[str, Any]:
        dict_ = {}
        if prediction.values.numel() == 0 or target.values.numel() == 0:
            return dict_
        num_channels = prediction.values.shape[2]

        # Compute metrics for pressure
        mse_pressure = np.mean((prediction.values[0, 0, 0].cpu().numpy() - target.values[0, 0, 0].cpu().numpy()) ** 2)
        norm_pressure = np.mean(target.values[0, 0, 0].cpu().numpy() ** 2)
        rel_l2_pressure = np.sqrt(mse_pressure / norm_pressure) * 100
        pointwise_rel_l1_p = np.abs(prediction.values[0, 0, 0].cpu().numpy() - target.values[0, 0, 0].cpu().numpy()) / (np.abs(target.values[0, 0, 0].cpu().numpy()) + 1e-8)
        median_rel_l1_p = np.median(pointwise_rel_l1_p) * 100

        # Compute WSS magnitude
        if num_channels > 1:
            pred_wss = prediction.values[0, 0, 1:].cpu().numpy()  # WSS components (3 channels)
            target_wss = target.values[0, 0, 1:].cpu().numpy()  # WSS components (3 channels)
            pred_wss_magnitude = np.linalg.norm(pred_wss, axis=0)  # Magnitude of predicted WSS
            target_wss_magnitude = np.linalg.norm(target_wss, axis=0)  # Magnitude of target WSS

            mse_wss = np.mean((pred_wss_magnitude - target_wss_magnitude) ** 2)
            norm_wss = np.mean(target_wss_magnitude**2)
            rel_l2_wss = np.sqrt(mse_wss / norm_wss) * 100

            pointwise_rel_l1_wss = np.abs(pred_wss_magnitude - target_wss_magnitude) / (np.abs(target_wss_magnitude) + 1e-8)
            median_rel_l1_wss = np.median(pointwise_rel_l1_wss) * 100

        # --- Separate vmin and vmax for Pressure ---
        target_pressure = target.values[0, 0, 0].cpu().numpy().ravel()  # Pressure channel
        if target_pressure.size == 0:
            vmin_pressure, vmax_pressure = 0.0, 1.0
        else:
            vmin_pressure, vmax_pressure = float(target_pressure.min()), float(target_pressure.max())

        # --- Separate vmin and vmax for WSS Magnitude ---
        if num_channels > 1:
            if target_wss_magnitude.size == 0:
                vmin_wss, vmax_wss = 0.0, 1.0
            else:
                vmin_wss, vmax_wss = float(target_wss_magnitude.min()), float(target_wss_magnitude.max())

        plt.rcParams.update({"font.size": 16, "axes.titlesize": 16})

        # --- Pressure Plot ---
        fig_pressure = plt.figure(figsize=(20, 10))
        fig_pressure.suptitle(f"Pressure - MSE: {mse_pressure:.4e}, Rel L2: {rel_l2_pressure:.2f}%, Median Rel L1: {median_rel_l1_p:.2f}%", fontsize=20)
        ax = fig_pressure.add_subplot(121, projection="3d")
        ax2 = fig_pressure.add_subplot(122, projection="3d")
        plt.subplots_adjust(wspace=0.1, hspace=0.1)

        # Predicted Pressure
        pos_np = prediction.pos[0, 0].cpu().numpy()
        x, y, z = pos_np[0].ravel(), pos_np[1].ravel(), pos_np[2].ravel()
        c = prediction.values[0, 0, 0].cpu().numpy().ravel()  # Pressure channel
        cb = ax.scatter(x, y, z, c=c, s=2, cmap="Spectral_r", vmin=vmin_pressure, vmax=vmax_pressure)
        ax.set_aspect("equal")
        ax.set_title("Predicted Pressure")

        # Reference Pressure
        pos_np_t = target.pos[0, 0].cpu().numpy()
        x2, y2, z2 = pos_np_t[0].ravel(), pos_np_t[1].ravel(), pos_np_t[2].ravel()
        c2 = target.values[0, 0, 0].cpu().numpy().ravel()  # Pressure channel
        cb2 = ax2.scatter(x2, y2, z2, c=c2, s=2, cmap="Spectral_r", vmin=vmin_pressure, vmax=vmax_pressure)
        ax2.set_aspect("equal")
        ax2.set_title("Reference Pressure")

        ax.view_init(elev=25, azim=225)
        ax2.view_init(elev=25, azim=225)

        fig_pressure.colorbar(cb, ax=ax, fraction=0.046, pad=0.04, orientation="horizontal")
        fig_pressure.colorbar(cb2, ax=ax2, fraction=0.046, pad=0.04, orientation="horizontal")

        # Save Pressure Figure
        fig_path_pressure = os.path.join(savepath, f"{stage}/sample{sample_idx}/scatter_surface_pressure.png")
        os.makedirs(os.path.dirname(fig_path_pressure), exist_ok=True)
        fig_pressure.savefig(fig_path_pressure, bbox_inches="tight")
        plt.close(fig_pressure)
        np.savez(os.path.join(savepath, f"{stage}/sample{sample_idx}/surface_pressure_data.npz"), pred_pressure=prediction.values[0, 0, 0].cpu().numpy(), target_pressure=target.values[0, 0, 0].cpu().numpy(), positions=target.pos[0, 0].cpu().numpy())

        dict_[f"{stage}/sample{sample_idx}/scatter_surface_pressure"] = fig_pressure

        # --- WSS Magnitude Plot ---
        if num_channels > 1:
            fig_wss = plt.figure(figsize=(20, 10))
            fig_wss.suptitle(f"WSS Magnitude - MSE: {mse_wss:.4e}, Rel L2: {rel_l2_wss:.2f}%, Median Rel L1: {median_rel_l1_wss:.2f}%", fontsize=20)
            ax = fig_wss.add_subplot(121, projection="3d")
            ax2 = fig_wss.add_subplot(122, projection="3d")
            plt.subplots_adjust(wspace=0.1, hspace=0.1)

            # Predicted WSS Magnitude
            c = pred_wss_magnitude.ravel()
            cb = ax.scatter(x, y, z, c=c, s=2, cmap="Spectral_r", vmin=vmin_wss, vmax=vmax_wss)
            ax.set_aspect("equal")
            ax.set_title("Predicted WSS Magnitude")

            # Reference WSS Magnitude
            c2 = target_wss_magnitude.ravel()
            cb2 = ax2.scatter(x2, y2, z2, c=c2, s=2, cmap="Spectral_r", vmin=vmin_wss, vmax=vmax_wss)
            ax2.set_aspect("equal")
            ax2.set_title("Reference WSS Magnitude")

            ax.view_init(elev=25, azim=225)
            ax2.view_init(elev=25, azim=225)

            fig_wss.colorbar(cb, ax=ax, fraction=0.046, pad=0.04, orientation="horizontal")
            fig_wss.colorbar(cb2, ax=ax2, fraction=0.046, pad=0.04, orientation="horizontal")

            # Save WSS Figure
            fig_path_wss = os.path.join(savepath, f"{stage}/sample{sample_idx}/scatter_surface_wss.png")
            os.makedirs(os.path.dirname(fig_path_wss), exist_ok=True)
            fig_wss.savefig(fig_path_wss, bbox_inches="tight")
            plt.close(fig_wss)
            np.savez(os.path.join(savepath, f"{stage}/sample{sample_idx}/surface_wss_data.npz"), pred_wss_magnitude=pred_wss_magnitude, target_wss_magnitude=target_wss_magnitude, positions=target.pos[0, 0].cpu().numpy())

            dict_[f"{stage}/sample{sample_idx}/scatter_surface_wss"] = fig_wss

        return dict_

    def visualization_2d_volume_scatter(self, prediction: UnstructuredData, target: UnstructuredData, stage: str, sample_idx: int, savepath: Path) -> Dict[str, Any]:
        dict_ = {}
        if prediction.values.numel() == 0 or target.values.numel() == 0:
            return dict_
        num_channels = prediction.values.shape[2]

        # Compute metrics for pressure
        mse_pressure = np.mean((prediction.values[0, 0, 0].cpu().numpy() - target.values[0, 0, 0].cpu().numpy()) ** 2)
        norm_pressure = np.mean(target.values[0, 0, 0].cpu().numpy() ** 2)
        rel_l2_pressure = np.sqrt(mse_pressure / norm_pressure) * 100
        pointwise_rel_l1_p = np.abs(prediction.values[0, 0, 0].cpu().numpy() - target.values[0, 0, 0].cpu().numpy()) / (np.abs(target.values[0, 0, 0].cpu().numpy()) + 1e-8)
        median_rel_l1_p = np.median(pointwise_rel_l1_p) * 100

        # Compute velocity magnitude
        if num_channels > 1:
            pred_velocity = prediction.values[0, 0, 1:].cpu().numpy()  # Velocity components (3 channels)
            target_velocity = target.values[0, 0, 1:].cpu().numpy()  # Velocity components (3 channels)
            pred_velocity_magnitude = np.linalg.norm(pred_velocity, axis=0)  # Magnitude of predicted velocity
            target_velocity_magnitude = np.linalg.norm(target_velocity, axis=0)  # Magnitude of target velocity

            mse_velocity = np.mean((pred_velocity_magnitude - target_velocity_magnitude) ** 2)
            norm_velocity = np.mean(target_velocity_magnitude**2)
            rel_l2_velocity = np.sqrt(mse_velocity / norm_velocity) * 100

            pointwise_rel_l1_velocity = np.abs(pred_velocity_magnitude - target_velocity_magnitude) / (np.abs(target_velocity_magnitude) + 1e-8)
            median_rel_l1_velocity = np.median(pointwise_rel_l1_velocity) * 100

        # --- Separate vmin and vmax for Pressure ---
        target_pressure = target.values[0, 0, 0].cpu().numpy().ravel()  # Pressure channel
        if target_pressure.size == 0:
            vmin_pressure, vmax_pressure = 0.0, 1.0
        else:
            vmin_pressure, vmax_pressure = float(target_pressure.min()), float(target_pressure.max())

        # --- Separate vmin and vmax for Velocity Magnitude ---
        if num_channels > 1:
            if target_velocity_magnitude.size == 0:
                vmin_velocity, vmax_velocity = 0.0, 1.0
            else:
                vmin_velocity, vmax_velocity = float(target_velocity_magnitude.min()), float(target_velocity_magnitude.max())

        plt.rcParams.update({"font.size": 16, "axes.titlesize": 16})

        # --- Pressure Plot ---
        fig_pressure = plt.figure(figsize=(20, 10))
        fig_pressure.suptitle(f"Pressure - MSE: {mse_pressure:.4e}, Rel L2: {rel_l2_pressure:.2f}%, Median Rel L1: {median_rel_l1_p:.2f}%", fontsize=20)
        ax = fig_pressure.add_subplot(121, projection="3d")
        ax2 = fig_pressure.add_subplot(122, projection="3d")
        plt.subplots_adjust(wspace=0.1, hspace=0.1)

        # Predicted Pressure
        pos_np = prediction.pos[0, 0].cpu().numpy()
        x, y, z = pos_np[0].ravel(), pos_np[1].ravel(), pos_np[2].ravel()
        c = prediction.values[0, 0, 0].cpu().numpy().ravel()  # Pressure channel
        cb = ax.scatter(x, y, z, c=c, s=2, cmap="Spectral_r", vmin=vmin_pressure, vmax=vmax_pressure)
        ax.set_aspect("equal")
        ax.set_title("Predicted Pressure")

        # Reference Pressure
        pos_np_t = target.pos[0, 0].cpu().numpy()
        x2, y2, z2 = pos_np_t[0].ravel(), pos_np_t[1].ravel(), pos_np_t[2].ravel()
        c2 = target.values[0, 0, 0].cpu().numpy().ravel()  # Pressure channel
        cb2 = ax2.scatter(x2, y2, z2, c=c2, s=2, cmap="Spectral_r", vmin=vmin_pressure, vmax=vmax_pressure)
        ax2.set_aspect("equal")
        ax2.set_title("Reference Pressure")

        ax.view_init(elev=25, azim=225)
        ax2.view_init(elev=25, azim=225)

        fig_pressure.colorbar(cb, ax=ax, fraction=0.046, pad=0.04, orientation="horizontal")
        fig_pressure.colorbar(cb2, ax=ax2, fraction=0.046, pad=0.04, orientation="horizontal")

        # Save Pressure Figure
        fig_path_pressure = os.path.join(savepath, f"{stage}/sample{sample_idx}/scatter_volume_pressure.png")
        os.makedirs(os.path.dirname(fig_path_pressure), exist_ok=True)
        fig_pressure.savefig(fig_path_pressure, bbox_inches="tight")
        plt.close(fig_pressure)
        np.savez(os.path.join(savepath, f"{stage}/sample{sample_idx}/volume_pressure_data.npz"), pred_pressure=prediction.values[0, 0, 0].cpu().numpy(), target_pressure=target.values[0, 0, 0].cpu().numpy(), positions=target.pos[0, 0].cpu().numpy())

        dict_[f"{stage}/sample{sample_idx}/scatter_volume_pressure"] = fig_pressure

        # --- Velocity Magnitude Plot ---
        if num_channels > 1:
            fig_velocity = plt.figure(figsize=(20, 10))
            fig_velocity.suptitle(f"Velocity Magnitude - MSE: {mse_velocity:.4e}, Rel L2: {rel_l2_velocity:.2f}%, Median Rel L1: {median_rel_l1_velocity:.2f}%", fontsize=20)
            ax = fig_velocity.add_subplot(121, projection="3d")
            ax2 = fig_velocity.add_subplot(122, projection="3d")
            plt.subplots_adjust(wspace=0.1, hspace=0.1)

            # Predicted Velocity Magnitude
            c = pred_velocity_magnitude.ravel()
            cb = ax.scatter(x, y, z, c=c, s=2, cmap="Spectral_r", vmin=vmin_velocity, vmax=vmax_velocity)
            ax.set_aspect("equal")
            ax.set_title("Predicted Velocity Magnitude")

            # Reference Velocity Magnitude
            c2 = target_velocity_magnitude.ravel()
            cb2 = ax2.scatter(x2, y2, z2, c=c2, s=2, cmap="Spectral_r", vmin=vmin_velocity, vmax=vmax_velocity)
            ax2.set_aspect("equal")
            ax2.set_title("Reference Velocity Magnitude")

            ax.view_init(elev=25, azim=225)
            ax2.view_init(elev=25, azim=225)

            fig_velocity.colorbar(cb, ax=ax, fraction=0.046, pad=0.04, orientation="horizontal")
            fig_velocity.colorbar(cb2, ax=ax2, fraction=0.046, pad=0.04, orientation="horizontal")

            # Save Velocity Figure
            fig_path_velocity = os.path.join(savepath, f"{stage}/sample{sample_idx}/scatter_volume_velocity.png")
            os.makedirs(os.path.dirname(fig_path_velocity), exist_ok=True)
            fig_velocity.savefig(fig_path_velocity, bbox_inches="tight")
            plt.close(fig_velocity)
            np.savez(os.path.join(savepath, f"{stage}/sample{sample_idx}/volume_velocity_data.npz"), pred_velocity_magnitude=pred_velocity_magnitude, target_velocity_magnitude=target_velocity_magnitude, positions=target.pos[0, 0].cpu().numpy())

            dict_[f"{stage}/sample{sample_idx}/scatter_volume_velocity"] = fig_velocity

        return dict_

    @rank_zero_only
    def log_video(self, trainer, pl_module, dataloader, stage=""):
        original_states = {}
        def _set_disable_subsampling_and_save(dataset, value):
            if dataset is None:
                return
            if hasattr(dataset, "datasets"):
                for sub_ds in dataset.datasets:
                    _set_disable_subsampling_and_save(sub_ds, value)
            if hasattr(dataset, "dataset"):
                _set_disable_subsampling_and_save(dataset.dataset, value)
            if id(dataset) not in original_states:
                original_states[id(dataset)] = (dataset, getattr(dataset, "disable_subsampling", None))
            setattr(dataset, "disable_subsampling", value)

        try:

            savepath = Path(os.path.join(trainer.default_root_dir, "visualizations"))

            generator = torch.Generator(device=pl_module.device).manual_seed(trainer.global_rank)

            # Temporarily restore subsampling for visualization to use configured max_num_..._points and avoid OOM
            loaders = dataloader if isinstance(dataloader, (list, tuple)) else [dataloader]
            for dl in loaders:
                _set_disable_subsampling_and_save(getattr(dl, "dataset", None), False)

            count = 0
            vis_dict = {}

            for batch in dataloader:

                if count > self.max_visualizations:
                    break

                if self.plot_surface:
                    prediction, target = pl_module.predict(batch, generator=generator, mode="surface", **self.model_kwargs)
                    vis_dict.update(self.visualization_2d_surface_scatter(prediction=prediction, target=target, stage=stage, sample_idx=count, savepath=savepath))
                if self.plot_volume:
                    prediction, target = pl_module.predict(batch, generator=generator, mode="volume", **self.model_kwargs)
                    vis_dict.update(self.visualization_2d_volume_scatter(prediction=prediction, target=target, stage=stage, sample_idx=count, savepath=savepath))

                count += 1

            self.log_visualizations(trainer, vis_dict)

        except Exception as e:

            rank_zero_warn(f"Error while creating visualization: {e}")
            raise e
        finally:
            # Restore original states
            for dataset, state in original_states.values():
                if state is None:
                    if hasattr(dataset, "disable_subsampling"):
                        delattr(dataset, "disable_subsampling")
                else:
                    setattr(dataset, "disable_subsampling", state)

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
        module_stage = getattr(pl_module, "stage", "last_ema")
        self.log_video(trainer, pl_module, trainer.test_dataloaders, stage=f"{module_stage}/test")


class Scatter3DCrash(Callback):
    def __init__(self, frequency: int, max_visualizations: int = 4, test_only: bool = False, plot_surface: bool = True, plot_volume: bool = False, **kwargs) -> None:

        super().__init__()

        self.frequency = frequency
        self.max_visualizations = max_visualizations
        self.test_only = test_only
        self.plot_surface = plot_surface
        self.plot_volume = plot_volume
        self.model_kwargs = kwargs

    def setup_test_dataloader(self, trainer):
        test_dataloader = trainer.test_dataloaders
        if test_dataloader is None:
            trainer.test_loop.setup_data()
            _ = trainer.test_dataloaders

    def _compute_magnitude_metrics(self, pred_mag: np.ndarray, target_mag: np.ndarray) -> Dict[str, float]:
        mse = np.mean((pred_mag - target_mag) ** 2)
        norm = np.mean(target_mag**2)
        rel_l2 = np.sqrt(mse / norm) * 100 if norm > 0 else 0.0
        pointwise_rel_l1 = np.abs(pred_mag - target_mag) / (np.abs(target_mag) + 1e-8)
        median_rel_l1 = np.median(pointwise_rel_l1) * 100
        return {
            "mse": mse,
            "rel_l2": rel_l2,
            "median_rel_l1": median_rel_l1,
        }

    def _compute_channel_metrics(self, pred_vals: np.ndarray, target_vals: np.ndarray) -> Dict[str, float]:
        mse = np.mean((pred_vals - target_vals) ** 2)
        norm = np.mean(target_vals**2)
        rel_l2 = np.sqrt(mse / norm) * 100 if norm > 0 else 0.0
        pointwise_rel_l1 = np.abs(pred_vals - target_vals) / (np.abs(target_vals) + 1e-8)
        median_rel_l1 = np.median(pointwise_rel_l1) * 100
        return {
            "mse": mse,
            "rel_l2": rel_l2,
            "median_rel_l1": median_rel_l1,
        }

    def _extract_values_and_pos(self, data: UnstructuredData) -> Dict[str, np.ndarray]:
        if data.channels_last:
            values = data.values[0, :, 0, :].cpu().numpy().T
            pos = data.pos[0, :, 0, :].cpu().numpy().T
        else:
            values = data.values[0, 0].cpu().numpy()
            pos = data.pos[0, 0].cpu().numpy()
        return {"values": values, "pos": pos}

    def visualization_2d_surface_scatter(self, prediction: UnstructuredData, target: UnstructuredData, stage: str, sample_idx: int, savepath: Path) -> Dict[str, Any]:
        dict_ = {}
        if prediction.values.numel() == 0 or target.values.numel() == 0:
            return dict_
        num_channels = prediction.values.shape[2]

        if num_channels < 3:
            rank_zero_warn("Scatter3DCrash expects at least 3 channels for surface data.")
            return dict_

        pred_data = self._extract_values_and_pos(prediction)
        target_data = self._extract_values_and_pos(target)

        pred_values = pred_data["values"]
        target_values = target_data["values"]

        pred_first = pred_values[0:3]
        target_first = target_values[0:3]
        pred_last = pred_values[3:]
        target_last = target_values[3:]
        num_tail_channels = num_channels - 3

        pred_first_mag = np.linalg.norm(pred_first, axis=0)
        target_first_mag = np.linalg.norm(target_first, axis=0)
        first_metrics = self._compute_magnitude_metrics(pred_first_mag, target_first_mag)
        channel_metrics = [
            self._compute_channel_metrics(pred_last[idx], target_last[idx])
            for idx in range(num_tail_channels)
        ]

        if target_first_mag.size == 0:
            vmin_first, vmax_first = 0.0, 1.0
        else:
            vmin_first, vmax_first = float(target_first_mag.min()), float(target_first_mag.max())

        channel_ranges = []
        for idx in range(num_tail_channels):
            target_channel = target_last[idx]
            if target_channel.size == 0:
                channel_ranges.append((0.0, 1.0))
            else:
                channel_ranges.append((float(target_channel.min()), float(target_channel.max())))

        plt.rcParams.update({"font.size": 16, "axes.titlesize": 16})

        pos_np = pred_data["pos"]
        x, y, z = pos_np[0].ravel(), pos_np[1].ravel(), pos_np[2].ravel()
        pos_np_t = target_data["pos"]
        x2, y2, z2 = pos_np_t[0].ravel(), pos_np_t[1].ravel(), pos_np_t[2].ravel()

        fig_first = plt.figure(figsize=(20, 10))
        fig_first.suptitle(
            f"Displacement Magnitude - MSE: {first_metrics['mse']:.4e}, Rel L2: {first_metrics['rel_l2']:.2f}%, Median Rel L1: {first_metrics['median_rel_l1']:.2f}%",
            fontsize=20,
        )
        ax = fig_first.add_subplot(121, projection="3d")
        ax2 = fig_first.add_subplot(122, projection="3d")
        plt.subplots_adjust(wspace=0.1, hspace=0.1)

        cb = ax.scatter(x, y, z, c=pred_first_mag.ravel(), s=2, cmap="Spectral_r", vmin=vmin_first, vmax=vmax_first)
        ax.set_aspect("equal")
        ax.set_title("Predicted Displacement Magnitude")

        cb2 = ax2.scatter(x2, y2, z2, c=target_first_mag.ravel(), s=2, cmap="Spectral_r", vmin=vmin_first, vmax=vmax_first)
        ax2.set_aspect("equal")
        ax2.set_title("Reference Displacement Magnitude")

        ax.view_init(elev=25, azim=225)
        ax2.view_init(elev=25, azim=225)

        fig_first.colorbar(cb, ax=ax, fraction=0.046, pad=0.04, orientation="horizontal")
        fig_first.colorbar(cb2, ax=ax2, fraction=0.046, pad=0.04, orientation="horizontal")

        fig_path_first = os.path.join(savepath, f"{stage}/sample{sample_idx}/scatter_surface_displacement_magnitude.png")
        os.makedirs(os.path.dirname(fig_path_first), exist_ok=True)
        fig_first.savefig(fig_path_first, bbox_inches="tight")
        plt.close(fig_first)
        np.savez(
            os.path.join(savepath, f"{stage}/sample{sample_idx}/surface_displacement_magnitude_data.npz"),
            pred_displacement_magnitude=pred_first_mag,
            target_displacement_magnitude=target_first_mag,
            positions=pos_np_t,
        )

        dict_[f"{stage}/sample{sample_idx}/scatter_surface_displacement_magnitude"] = fig_first

        for idx in range(num_tail_channels):
            channel_idx = idx + 3
            c_name = "Stress" if channel_idx == 3 else f"Channel {channel_idx}"
            file_name = "stress" if channel_idx == 3 else f"channel{channel_idx}"
            vmin_last, vmax_last = channel_ranges[idx]
            metrics = channel_metrics[idx]
            pred_channel = pred_last[idx]
            target_channel = target_last[idx]

            fig_last = plt.figure(figsize=(20, 10))
            fig_last.suptitle(
                f"{c_name} - MSE: {metrics['mse']:.4e}, Rel L2: {metrics['rel_l2']:.2f}%, Median Rel L1: {metrics['median_rel_l1']:.2f}%",
                fontsize=20,
            )
            ax = fig_last.add_subplot(121, projection="3d")
            ax2 = fig_last.add_subplot(122, projection="3d")
            plt.subplots_adjust(wspace=0.1, hspace=0.1)

            cb = ax.scatter(x, y, z, c=pred_channel.ravel(), s=2, cmap="Spectral_r", vmin=vmin_last, vmax=vmax_last)
            ax.set_aspect("equal")
            ax.set_title(f"Predicted {c_name}")

            cb2 = ax2.scatter(x2, y2, z2, c=target_channel.ravel(), s=2, cmap="Spectral_r", vmin=vmin_last, vmax=vmax_last)
            ax2.set_aspect("equal")
            ax2.set_title(f"Reference {c_name}")

            ax.view_init(elev=25, azim=225)
            ax2.view_init(elev=25, azim=225)

            fig_last.colorbar(cb, ax=ax, fraction=0.046, pad=0.04, orientation="horizontal")
            fig_last.colorbar(cb2, ax=ax2, fraction=0.046, pad=0.04, orientation="horizontal")

            fig_path_last = os.path.join(savepath, f"{stage}/sample{sample_idx}/scatter_surface_{file_name}.png")
            os.makedirs(os.path.dirname(fig_path_last), exist_ok=True)
            fig_last.savefig(fig_path_last, bbox_inches="tight")
            plt.close(fig_last)
            np.savez(
                os.path.join(savepath, f"{stage}/sample{sample_idx}/surface_{file_name}_data.npz"),
                pred_channel=pred_channel,
                target_channel=target_channel,
                positions=pos_np_t,
            )

            dict_[f"{stage}/sample{sample_idx}/scatter_surface_{file_name}"] = fig_last

        return dict_

    @rank_zero_only
    def log_video(self, trainer, pl_module, dataloader, stage=""):
        original_states = {}
        def _set_disable_subsampling_and_save(dataset, value):
            if dataset is None:
                return
            if hasattr(dataset, "datasets"):
                for sub_ds in dataset.datasets:
                    _set_disable_subsampling_and_save(sub_ds, value)
            if hasattr(dataset, "dataset"):
                _set_disable_subsampling_and_save(dataset.dataset, value)
            if id(dataset) not in original_states:
                original_states[id(dataset)] = (dataset, getattr(dataset, "disable_subsampling", None))
            setattr(dataset, "disable_subsampling", value)

        try:

            savepath = Path(os.path.join(trainer.default_root_dir, "visualizations"))

            generator = torch.Generator(device=pl_module.device).manual_seed(trainer.global_rank)

            # Temporarily restore subsampling for visualization to use configured max_num_..._points and avoid OOM
            loaders = dataloader if isinstance(dataloader, (list, tuple)) else [dataloader]
            for dl in loaders:
                _set_disable_subsampling_and_save(getattr(dl, "dataset", None), False)

            count = 0
            vis_dict = {}

            for batch in dataloader:

                if count > self.max_visualizations:
                    break

                if self.plot_surface:
                    prediction, target = pl_module.predict(batch, generator=generator, mode="surface", **self.model_kwargs)
                    vis_dict.update(self.visualization_2d_surface_scatter(prediction=prediction, target=target, stage=stage, sample_idx=count, savepath=savepath))

                count += 1

            self.log_visualizations(trainer, vis_dict)

        except Exception as e:

            rank_zero_warn(f"Error while creating visualization: {e}")
            raise e
        finally:
            # Restore original states
            for dataset, state in original_states.values():
                if state is None:
                    if hasattr(dataset, "disable_subsampling"):
                        delattr(dataset, "disable_subsampling")
                else:
                    setattr(dataset, "disable_subsampling", state)

    @rank_zero_only
    def log_visualizations(self, trainer: Trainer, vis_dict: Dict[str, Any]):

        for logger in trainer.loggers:

            print("Logging visualizations to logger: ", logger)

            if isinstance(logger, WandbLogger):
                data_dict = {}
                img_dict = {}
                for k, v in vis_dict.items():

                    if isinstance(v, plt.Figure):
                        img_dict[k] = wandb.Image(v, caption=k)
                    else:
                        data_dict[k] = v

                data_dict.update(img_dict)
                logger.experiment.log(data_dict)

            else:

                logger.log_metrics(vis_dict)

    def on_test_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.setup_test_dataloader(trainer)
        module_stage = getattr(pl_module, "stage", "last_ema")
        self.log_video(trainer, pl_module, trainer.test_dataloaders, stage=f"{module_stage}/test")
    