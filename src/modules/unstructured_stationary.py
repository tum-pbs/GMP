import dataclasses
import math
import random
from typing import List, Optional, Tuple

import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn

from src.data.loader import DataSample
from src.models.adapters import UnstructuredTransformerAdapter
from src.models.dataclasses import UnstructuredData


def relative_l2_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Relative L2 loss as defined in Equation (10) of SMART (Li et al., 2026):
    L(Y_hat, Y) = (1 / N_c) * \\sum_{c=1}^{N_c} ||Y_hat_c - Y_c||_2 / (||Y_c||_2 + eps)
    where ||.|| denotes the L2 norm across spatial locations.
    """
    diff_norm = torch.linalg.vector_norm(pred - target, ord=2, dim=-1)
    target_norm = torch.linalg.vector_norm(target, ord=2, dim=-1)
    rel_l2_per_channel = diff_norm / (target_norm + eps)
    return torch.mean(rel_l2_per_channel)


def compute_block_loss(
    data_prediction: UnstructuredData,
    data_output: UnstructuredData,
    batch: DataSample,
    loss_fn=relative_l2_loss,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Computes loss and MSE considering only valid/observed blocks of the output tensor.

    Handles:
    1. Joint 2x2 block matrix predictions (e.g. SMART, GT, ABUPT on volume + surface).
    2. Single-domain models (Volume-only or Surface-only, e.g. Transolver with use_surface=False or use_volume=False).
    3. Separate or fallback prediction structures.
    """
    pred_vals = data_prediction.values
    target_vals = data_output.values

    # Retrieve surface metadata if present
    num_surf_pts = 0
    num_surf_ch = 0
    if hasattr(batch, "loading_metadata"):
        lm = batch.loading_metadata
        if hasattr(lm, "num_surface_points") and lm.num_surface_points is not None and lm.num_surface_points.numel() > 0:
            num_surf_pts = int(lm.num_surface_points.flatten()[0].item())
        if hasattr(lm, "num_surface_value_channels") and lm.num_surface_value_channels is not None and lm.num_surface_value_channels.numel() > 0:
            num_surf_ch = int(lm.num_surface_value_channels.flatten()[0].item())

    total_pts = target_vals.shape[-1]
    total_ch = target_vals.shape[2]

    num_vol_pts = max(0, total_pts - num_surf_pts)
    num_vol_ch = max(0, total_ch - num_surf_ch)

    has_vol = (num_vol_pts > 0) and (num_vol_ch > 0)
    has_surf = (num_surf_pts > 0) and (num_surf_ch > 0)

    # Detect if 2x2 block matrix format is present (both volume and surface present and shapes match 2x2 layout)
    is_2x2_block = (
        has_vol
        and has_surf
        and (pred_vals.shape[2] == total_ch)
        and (pred_vals.shape[-1] == total_pts)
    )

    if is_2x2_block:
        # Diagonal blocks slicing: Top-Left (Volume) and Bottom-Right (Surface)
        pred_vol = pred_vals[:, :, :num_vol_ch, :num_vol_pts]
        target_vol = target_vals[:, :, :num_vol_ch, :num_vol_pts]

        pred_surf = pred_vals[:, :, num_vol_ch:, num_vol_pts:]
        target_surf = target_vals[:, :, num_vol_ch:, num_vol_pts:]

        loss_vol = loss_fn(pred_vol, target_vol)
        loss_surf = loss_fn(pred_surf, target_surf)
        loss = (loss_vol + loss_surf) / 2.0

        mse_vol = torch.nn.functional.mse_loss(pred_vol, target_vol)
        mse_surf = torch.nn.functional.mse_loss(pred_surf, target_surf)
        mse = (mse_vol + mse_surf) / 2.0

    elif has_vol and not has_surf:
        # Volume only
        pred_vol = pred_vals[:, :, :num_vol_ch, :num_vol_pts] if (pred_vals.shape[2] >= num_vol_ch and pred_vals.shape[-1] >= num_vol_pts) else pred_vals
        target_vol = target_vals[:, :, :num_vol_ch, :num_vol_pts] if (target_vals.shape[2] >= num_vol_ch and target_vals.shape[-1] >= num_vol_pts) else target_vals

        loss_vol = loss_fn(pred_vol, target_vol)
        loss_surf = None
        loss = loss_vol

        mse_vol = torch.nn.functional.mse_loss(pred_vol, target_vol)
        mse_surf = None
        mse = mse_vol

    elif has_surf and not has_vol:
        # Surface only
        pred_surf = pred_vals[:, :, -num_surf_ch:, -num_surf_pts:] if (pred_vals.shape[2] >= num_surf_ch and pred_vals.shape[-1] >= num_surf_pts) else pred_vals
        target_surf = target_vals[:, :, -num_surf_ch:, -num_surf_pts:] if (target_vals.shape[2] >= num_surf_ch and target_vals.shape[-1] >= num_surf_pts) else target_vals

        loss_surf = loss_fn(pred_surf, target_surf)
        loss_vol = None
        loss = loss_surf

        mse_surf = torch.nn.functional.mse_loss(pred_surf, target_surf)
        mse_vol = None
        mse = mse_surf

    else:
        # Fallback (general loss over matching shapes)
        loss = loss_fn(pred_vals, target_vals)
        mse = torch.nn.functional.mse_loss(pred_vals, target_vals)
        loss_vol = None
        loss_surf = None

    return loss, mse, loss_vol, loss_surf


class UnstructuredTransformerStationary(UnstructuredTransformerAdapter):


    def __init__(self, model: nn.Module, optimizer: DictConfig) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["model", "optimizer"])
        self.model = model

        if isinstance(optimizer, DictConfig):
            self.optimizer_dict = OmegaConf.to_container(optimizer, resolve=True)
        else:
            self.optimizer_dict = optimizer

    def build_unstructured(self, batch: DataSample, mode: str | None = None) -> UnstructuredData:
        pos = batch.positions

        surface_points = batch.loading_metadata.num_surface_points.to(self.device)
        surface_value_channels = batch.loading_metadata.num_surface_value_channels.to(self.device)
        surface_attr_channels = batch.loading_metadata.num_surface_attr_channels.to(self.device)

        if mode == "input":
            values = batch.attributes
        elif mode == "output":
            values = batch.values
        else:
            raise ValueError(f"Invalid mode '{mode}'. Expected 'input' or 'output'.")
        num_points = batch.num_points
        global_params = batch.global_params
        pos = pos.to(self.device)
        values = values.to(self.device)
        num_points = num_points.to(self.device)
        global_params = global_params.to(self.device)

        if batch.loading_metadata.channel_last[0]:
            values = torch.permute(values, (0, 2, 3, 1))  # (B, N, T, C) -> (B, T, C, N)
            pos = torch.permute(pos, (0, 2, 3, 1))  # (B, N, T, D) -> (B, T, D, N)

        return UnstructuredData(values=values, pos=pos, num_points=num_points, global_params=global_params, channels_last=False, surface_points=surface_points, surface_value_channels=surface_value_channels, surface_attr_channels=surface_attr_channels)

    def _scale_data_attr(self, data: UnstructuredData, batch: DataSample) -> UnstructuredData:
        values = data.values  # (B, T, A, N)

        attr_subtract = batch.loading_metadata.attr_subtract.to(self.device)  # (B, A)
        attr_divide = batch.loading_metadata.attr_divide.to(self.device)  # (B, A)

        values = values - attr_subtract.unsqueeze(1).unsqueeze(-1)  # (B, T, A, N)
        values = values / attr_divide.unsqueeze(1).unsqueeze(-1)  # (B, T, A, N)

        pos = data.pos  # (B, T, D, N)
        pos_min = batch.physical_metadata.min_domain.to(self.device)  # (B, D)
        pos_max = batch.physical_metadata.max_domain.to(self.device)  # (B, D)
        pos = (pos - pos_min.unsqueeze(1).unsqueeze(-1)) / (pos_max - pos_min).unsqueeze(1).unsqueeze(-1)  # (B, T, D, N)

        data = dataclasses.replace(data, values=values, pos=pos, global_params=data.global_params)

        return data

    def _scale_data_val(self, data: UnstructuredData, batch: DataSample) -> UnstructuredData:
        values = data.values  # (B, T, A, N)

        val_subtract = batch.loading_metadata.val_subtract.to(self.device)  # (B, A)
        val_divide = batch.loading_metadata.val_divide.to(self.device)  # (B, A)

        values = values - val_subtract.unsqueeze(1).unsqueeze(-1)  # (B, T, A, N)
        values = values / val_divide.unsqueeze(1).unsqueeze(-1)  # (B, T, A, N)

        pos = data.pos  # (B, T, D, N)
        pos_min = batch.physical_metadata.min_domain.to(self.device)  # (B, D)
        pos_max = batch.physical_metadata.max_domain.to(self.device)  # (B, D)
        pos = (pos - pos_min.unsqueeze(1).unsqueeze(-1)) / (pos_max - pos_min).unsqueeze(1).unsqueeze(-1)  # (B, T, D, N)

        data = dataclasses.replace(data, values=values, pos=pos, global_params=data.global_params)

        return data

    def _unscale_data_attr(self, data: UnstructuredData, batch: DataSample) -> UnstructuredData:
        values = data.values  # (B, T, C, N)

        attr_subtract = batch.loading_metadata.attr_subtract.to(self.device)  # (B, C)
        attr_divide = batch.loading_metadata.attr_divide.to(self.device)  # (B, C)

        values = values * attr_divide.unsqueeze(1).unsqueeze(-1)  # (B, T, C, N)
        values = values + attr_subtract.unsqueeze(1).unsqueeze(-1)  # (B, T, C, N)

        pos = data.pos  # (B, T, D, N)
        pos_min = batch.physical_metadata.min_domain.to(self.device)  # (B, D)
        pos_max = batch.physical_metadata.max_domain.to(self.device)  # (B, D)
        pos = pos * (pos_max - pos_min).unsqueeze(1).unsqueeze(-1) + pos_min.unsqueeze(1).unsqueeze(-1)  # (B, T, D, N)

        data = dataclasses.replace(data, values=values, pos=pos, global_params=data.global_params)

        return data

    def _unscale_data_val(self, data: UnstructuredData, batch: DataSample) -> UnstructuredData:
        values = data.values  # (B, T, C, N)

        val_subtract = batch.loading_metadata.val_subtract.to(self.device)  # (B, C)
        val_divide = batch.loading_metadata.val_divide.to(self.device)  # (B, C)

        values = values * val_divide.unsqueeze(1).unsqueeze(-1)  # (B, T, C, N)
        values = values + val_subtract.unsqueeze(1).unsqueeze(-1)  # (B, T, C, N)

        pos = data.pos  # (B, T, D, N)
        pos_min = batch.physical_metadata.min_domain.to(self.device)  # (B, D)
        pos_max = batch.physical_metadata.max_domain.to(self.device)  # (B, D)
        pos = pos * (pos_max - pos_min).unsqueeze(1).unsqueeze(-1) + pos_min.unsqueeze(1).unsqueeze(-1)  # (B, T, D, N)

        data = dataclasses.replace(data, values=values, pos=pos, global_params=data.global_params)

        return data

    def _shared_step(self, batch: DataSample, stage: str) -> torch.Tensor:

        data_input = self.build_unstructured(batch, mode="input")
        data_output = self.build_unstructured(batch, mode="output")

        data_input = self._scale_data_attr(data_input, batch)
        data_output = self._scale_data_val(data_output, batch)

        data_prediction = self.forward(data_input)

        loss, mse, loss_vol, loss_surf = compute_block_loss(data_prediction, data_output, batch, loss_fn=relative_l2_loss)

        batch_size = batch.values.shape[0]

        self.log(f"{self.stage}/{stage}/loss", loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=batch_size)
        self.log(f"{self.stage}/{stage}/mse", mse, prog_bar=False, on_step=True, on_epoch=True, batch_size=batch_size)
        if loss_vol is not None:
            self.log(f"{self.stage}/{stage}/loss_vol", loss_vol, prog_bar=False, on_step=False, on_epoch=True, batch_size=batch_size)
        if loss_surf is not None:
            self.log(f"{self.stage}/{stage}/loss_surf", loss_surf, prog_bar=False, on_step=False, on_epoch=True, batch_size=batch_size)

        return loss
    
    def _shared_step_chunks(self, batch: DataSample, stage: str) -> torch.Tensor:
        # Build unstructured data for input and output
        data_input = self.build_unstructured(batch, mode="input")
        data_output = self.build_unstructured(batch, mode="output")

        # Scale the input and output data
        data_input = self._scale_data_attr(data_input, batch)
        data_output = self._scale_data_val(data_output, batch)

        # Chunking parameters
        chunk_size = 65536
        num_points = data_input.values.shape[-1]  # Total number of points
        batch_size = data_input.values.shape[0]  # Batch size

        # Generate random permutation of indices
        random_indices = torch.randperm(num_points, device=self.device)

        # Initialize tensors to store predictions
        predictions = []

        # Process the point cloud in random chunks
        for start_idx in range(0, num_points, chunk_size):
            end_idx = min(start_idx + chunk_size, num_points)

            # Select random indices for the chunk
            chunk_indices = random_indices[start_idx:end_idx]

            # Extract the chunk
            chunk_input = dataclasses.replace(
                data_input,
                values=data_input.values[..., chunk_indices],
                pos=data_input.pos[..., chunk_indices],
            )

            # Forward pass for the chunk
            chunk_prediction = self.forward(chunk_input)
            predictions.append(chunk_prediction.values)

        # Combine predictions from all chunks
        combined_predictions = torch.cat(predictions, dim=-1)

        # Replace the values in the prediction with the combined predictions
        data_prediction = dataclasses.replace(data_input, values=combined_predictions)
        
        # Also apply the permutation to the target data_output to ensure it is in the same order as the predictions
        data_output = dataclasses.replace(
            data_output,
            values=data_output.values[..., random_indices],
            pos=data_output.pos[..., random_indices],
        )

        # Compute the loss
        loss, mse, loss_vol, loss_surf = compute_block_loss(data_prediction, data_output, batch, loss_fn=relative_l2_loss)

        # Log the loss
        self.log(f"{self.stage}/{stage}/loss", loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=batch_size)
        self.log(f"{self.stage}/{stage}/mse", mse, prog_bar=False, on_step=True, on_epoch=True, batch_size=batch_size)
        if loss_vol is not None:
            self.log(f"{self.stage}/{stage}/loss_vol", loss_vol, prog_bar=False, on_step=False, on_epoch=True, batch_size=batch_size)
        if loss_surf is not None:
            self.log(f"{self.stage}/{stage}/loss_surf", loss_surf, prog_bar=False, on_step=False, on_epoch=True, batch_size=batch_size)

        return loss

    def _forward(self, data: UnstructuredData) -> UnstructuredData:

        return self.model._forward(data)

    def predict(self, batch: DataSample, mode='all', generator: Optional[torch.Generator] = None, **kwargs) -> Tuple[UnstructuredData, UnstructuredData]:

        data_input = self.build_unstructured(batch, mode="input")
        data_output = self.build_unstructured(batch, mode="output")

        data_input = self._scale_data_attr(data_input, batch)

        data_prediction = self.forward(data_input)

        data_prediction = self._unscale_data_val(data_prediction, batch)

        num_surface_points_tensor = batch.loading_metadata.num_surface_points
        num_surface_value_channels_tensor = batch.loading_metadata.num_surface_value_channels

        if num_surface_points_tensor.numel() > 1:
            if not torch.all(num_surface_points_tensor == num_surface_points_tensor.flatten()[0]):
                raise ValueError("num_surface_points must be identical across the batch.")
        if num_surface_value_channels_tensor.numel() > 1:
            if not torch.all(num_surface_value_channels_tensor == num_surface_value_channels_tensor.flatten()[0]):
                raise ValueError("num_surface_value_channels must be identical across the batch.")

        num_surface_points = int(num_surface_points_tensor.flatten()[0].item())
        num_surface_value_channels = int(num_surface_value_channels_tensor.flatten()[0].item())
        num_volume_points = data_prediction.values.shape[-1] - num_surface_points
        num_volume_value_channels = data_prediction.values.shape[2] - num_surface_value_channels
        if mode == 'surface':
            prediction_values = data_prediction.values[..., -num_surface_value_channels:, -num_surface_points:]
            prediction_pos = data_prediction.pos[..., -num_surface_points:]
            output_values = data_output.values[..., -num_surface_value_channels:, -num_surface_points:]
            output_pos = data_output.pos[..., -num_surface_points:]
        elif mode == 'volume':
            prediction_values = data_prediction.values[..., :num_volume_value_channels, :num_volume_points]
            prediction_pos = data_prediction.pos[..., :num_volume_points]
            output_values = data_output.values[..., :num_volume_value_channels, :num_volume_points]
            output_pos = data_output.pos[..., :num_volume_points]
        elif mode == 'all':
            return data_prediction, data_output
        else:
            raise ValueError(f"Invalid mode '{mode}'. Expected 'surface' or 'volume'.")

        data_prediction = dataclasses.replace(data_prediction, values=prediction_values, pos=prediction_pos)
        data_output = dataclasses.replace(data_output, values=output_values, pos=output_pos)

        return data_prediction, data_output
    
    def predict_chunks(
        self, batch: DataSample, generator: Optional[torch.Generator] = None, chunk_size: int = 65536, **kwargs
    ) -> Tuple[UnstructuredData, UnstructuredData]:
        # Build unstructured data for input and output
        data_input = self.build_unstructured(batch, mode="input")
        data_output = self.build_unstructured(batch, mode="output")

        # Scale the input data
        data_input = self._scale_data_attr(data_input, batch)

        # Extraction of metadata
        num_surface_points = int(data_input.surface_points.flatten()[0].item())
        num_surface_value_channels = int(data_input.surface_value_channels.flatten()[0].item())
        num_surface_attr_channels = int(data_input.surface_attr_channels.flatten()[0].item())

        total_points = data_input.values.shape[-1]
        num_volume_points = total_points - num_surface_points

        batch_size = data_input.values.shape[0]
        T = data_input.values.shape[1]

        # Calculate volume value channels in output
        total_out_channels = data_output.values.shape[2]
        volume_value_channels = total_out_channels - num_surface_value_channels

        # Slice volume points and surface points from data_input
        vol_pos = data_input.pos[:, :, :, :num_volume_points]
        vol_values = data_input.values[:, :, :, :num_volume_points]

        surf_pos = data_input.pos[:, :, :, num_volume_points:]
        surf_values = data_input.values[:, :, :, num_volume_points:]

        device = data_input.values.device
        dtype = data_input.values.dtype

        # Generate random permutations of indices to ensure uniform spatial distribution across chunks
        g = generator
        if num_surface_points > 0:
            surf_perm = torch.randperm(num_surface_points, generator=g, device=device)
        else:
            surf_perm = torch.zeros((0,), dtype=torch.long, device=device)

        if num_volume_points > 0:
            vol_perm = torch.randperm(num_volume_points, generator=g, device=device)
        else:
            vol_perm = torch.zeros((0,), dtype=torch.long, device=device)

        num_vol_chunks = (num_volume_points + chunk_size - 1) // chunk_size if num_volume_points > 0 else 0
        num_surf_chunks = (num_surface_points + chunk_size - 1) // chunk_size if num_surface_points > 0 else 0
        num_chunks = max(num_vol_chunks, num_surf_chunks)

        if num_chunks == 0:
            return self._unscale_data_val(data_input, batch), data_output

        pred_volume_vals = torch.zeros((batch_size, T, volume_value_channels, num_volume_points), device=device, dtype=dtype)
        pred_surface_vals = torch.zeros((batch_size, T, num_surface_value_channels, num_surface_points), device=device, dtype=dtype)

        for i in range(num_chunks):
            # Volume indices for this chunk
            v_start = i * chunk_size
            v_end = (i + 1) * chunk_size

            is_vol_valid = False
            valid_vol_len = 0
            if num_volume_points > 0:
                if v_start < num_volume_points:
                    is_vol_valid = True
                    chunk_vol_idx_unique = vol_perm[v_start:min(v_end, num_volume_points)]
                    valid_vol_len = len(chunk_vol_idx_unique)
                    if valid_vol_len < chunk_size:
                        # Pad with random points from the volume set
                        num_pad = chunk_size - valid_vol_len
                        pad_idx = torch.randint(0, num_volume_points, (num_pad,), generator=g, device=device)
                        chunk_vol_idx = torch.cat([chunk_vol_idx_unique, vol_perm[pad_idx]], dim=0)
                    else:
                        chunk_vol_idx = chunk_vol_idx_unique
                else:
                    # Volume is exhausted, sample randomly to fill the chunk size
                    pad_idx = torch.randint(0, num_volume_points, (chunk_size,), generator=g, device=device)
                    chunk_vol_idx = vol_perm[pad_idx]
            else:
                chunk_vol_idx = torch.zeros((0,), dtype=torch.long, device=device)

            # Surface indices for this chunk
            s_start = i * chunk_size
            s_end = (i + 1) * chunk_size

            is_surf_valid = False
            valid_surf_len = 0
            if num_surface_points > 0:
                if s_start < num_surface_points:
                    is_surf_valid = True
                    chunk_surf_idx_unique = surf_perm[s_start:min(s_end, num_surface_points)]
                    valid_surf_len = len(chunk_surf_idx_unique)
                    if valid_surf_len < chunk_size:
                        # Pad with random points from the surface set
                        num_pad = chunk_size - valid_surf_len
                        pad_idx = torch.randint(0, num_surface_points, (num_pad,), generator=g, device=device)
                        chunk_surf_idx = torch.cat([chunk_surf_idx_unique, surf_perm[pad_idx]], dim=0)
                    else:
                        chunk_surf_idx = chunk_surf_idx_unique
                else:
                    # Surface is exhausted, sample randomly to fill the chunk size
                    pad_idx = torch.randint(0, num_surface_points, (chunk_size,), generator=g, device=device)
                    chunk_surf_idx = surf_perm[pad_idx]
            else:
                chunk_surf_idx = torch.zeros((0,), dtype=torch.long, device=device)

            if (i % 20 == 0) or (i == num_chunks - 1):
                print(f"  [predict_chunks] chunk {i+1}/{num_chunks}: vol={v_start}/{num_volume_points}, surf={s_start}/{num_surface_points}")

            # Assemble chunk input
            chunk_vol_pos = vol_pos[..., chunk_vol_idx]
            chunk_vol_values = vol_values[..., chunk_vol_idx]

            chunk_surf_pos = surf_pos[..., chunk_surf_idx]
            chunk_surf_values = surf_values[..., chunk_surf_idx]

            chunk_pos = torch.cat([chunk_vol_pos, chunk_surf_pos], dim=-1)
            chunk_values = torch.cat([chunk_vol_values, chunk_surf_values], dim=-1)

            chunk_num_points = chunk_pos.shape[-1]
            chunk_surf_points = chunk_surf_pos.shape[-1]

            chunk_input = dataclasses.replace(
                data_input,
                values=chunk_values,
                pos=chunk_pos,
                num_points=torch.full_like(data_input.num_points, chunk_num_points),
                surface_points=torch.full_like(data_input.surface_points, chunk_surf_points),
            )

            chunk_prediction = self.forward(chunk_input)

            num_vol_in_chunk = chunk_vol_pos.shape[-1]
            if is_vol_valid:
                # Volume predictions are the first chunk_size points (and we only store the unique ones)
                pred_volume_vals[..., chunk_vol_idx_unique] = chunk_prediction.values[:, :, :volume_value_channels, :valid_vol_len]

            if is_surf_valid:
                # Surface predictions are the points after the volume points in the chunk (and we only store the unique ones)
                pred_surface_vals[..., chunk_surf_idx_unique] = chunk_prediction.values[:, :, volume_value_channels:, num_vol_in_chunk : num_vol_in_chunk + valid_surf_len]

        # Assemble volume and surface predictions in canonical order (volume points first, then surface points)
        # Pad volume predictions with zeros for the surface points
        values_volume = torch.cat([
            pred_volume_vals,
            torch.zeros((batch_size, T, volume_value_channels, num_surface_points), device=device, dtype=dtype)
        ], dim=-1)

        # Pad surface predictions with zeros for the volume points
        values_surface = torch.cat([
            torch.zeros((batch_size, T, num_surface_value_channels, num_volume_points), device=device, dtype=dtype),
            pred_surface_vals
        ], dim=-1)

        # Concatenate along channel dimension
        combined_predictions = torch.cat([values_volume, values_surface], dim=2)

        # Replace values in data_input
        data_prediction = dataclasses.replace(data_input, values=combined_predictions)

        # Unscale the predictions
        data_prediction = self._unscale_data_val(data_prediction, batch)

        return data_prediction, data_output

    def on_train_epoch_start(self):
        if hasattr(self.trainer.datamodule.train_dataset, "sim_indexes"):
            random.shuffle(self.trainer.datamodule.train_dataset.sim_indexes)
        super().on_train_epoch_start()

        self.trainer.datamodule.train_dataset.set_epoch(self.current_epoch)

    def configure_optimizers(self):

        # initialize optimizers with self.parameters()
        optimizer_init = self.optimizer_dict["optimizer"]
        if isinstance(optimizer_init, list):
            instances = [init_f(params=self.parameters()) for init_f in optimizer_init]
        else:
            instances = optimizer_init(params=self.parameters())

        optimizers = {"optimizer": instances}

        # initialize lr_schedulers with optimizers
        lr_scheduler_init = self.optimizer_dict["lr_scheduler"]

        # Compute steps_per_epoch for step-level schedulers (e.g. CosineWarmupLR)
        extra_kwargs = {}
        try:
            total_steps = getattr(self.trainer, "estimated_stepping_batches", None)
            max_epochs = getattr(self.trainer, "max_epochs", -1)
            if max_epochs and max_epochs > 0 and total_steps is not None and not (math.isinf(total_steps) or math.isnan(total_steps)):
                extra_kwargs["steps_per_epoch"] = max(1, int(total_steps // max_epochs))
        except Exception:
            pass

        def _create_scheduler(factory, optimizer):
            try:
                return factory(optimizer=optimizer, **extra_kwargs)
            except TypeError:
                return factory(optimizer=optimizer)

        if isinstance(lr_scheduler_init, list):
            scheduler_instances = [_create_scheduler(init_f["scheduler"], opt) for init_f, opt in zip(lr_scheduler_init, instances)]
        else:
            scheduler_instances = _create_scheduler(lr_scheduler_init["scheduler"], instances)

        lr_scheduler_configs = self.optimizer_dict["lr_scheduler"]

        if isinstance(lr_scheduler_configs, list):
            for lr_scheduler_config, scheduler in zip(lr_scheduler_configs, scheduler_instances):
                lr_scheduler_config["scheduler"] = scheduler
        else:
            lr_scheduler_configs["scheduler"] = scheduler_instances

        optimizers["lr_scheduler"] = lr_scheduler_configs

        self.optimizer_dict.update(optimizers)

        if isinstance(self.optimizer_dict["optimizer"], list):
            return self.optimizer_dict["optimizer"], self.optimizer_dict["lr_scheduler"]
        else:
            return self.optimizer_dict

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "val")

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        with torch.no_grad():
            return self._shared_step(batch, "test")
