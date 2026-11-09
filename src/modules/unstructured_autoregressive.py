import dataclasses
from typing import List, Optional, Tuple

import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn

from src.data.loader import DataSample
from src.models.adapters import UnstructuredTransformerAdapter
from src.models.dataclasses import UnstructuredData


def concatenate_unstructured(data_list: List[UnstructuredData], dim: int = 1) -> UnstructuredData:
    values = torch.cat([data.values for data in data_list], dim=dim)
    pos = torch.cat([data.pos for data in data_list], dim=dim)
    num_points = data_list[0].num_points
    return UnstructuredData(values=values, pos=pos, num_points=num_points, global_params=torch.zeros(0), surface_points=data_list[0].surface_points, surface_attr_channels=data_list[0].surface_attr_channels, surface_value_channels=data_list[0].surface_value_channels)


class UnstructuredTransformerAutoregressive(UnstructuredTransformerAdapter):

    def __init__(self, model: nn.Module, optimizer: DictConfig, spatial_dim: Optional[int] = None) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["model", "optimizer"])
        self.model = model
        self.spatial_dim = spatial_dim or getattr(model, "spatial_dim", getattr(model, "dim", 3))

        if isinstance(optimizer, DictConfig):
            self.optimizer_dict = OmegaConf.to_container(optimizer, resolve=True)
        else:
            self.optimizer_dict = optimizer

    def build_unstructured(self, batch: DataSample, start: int, end: int) -> UnstructuredData:
        pos = batch.positions
        values = batch.values
        num_points = batch.num_points
        global_params = batch.global_params

        surface_points = batch.loading_metadata.num_surface_points.to(self.device)
        surface_value_channels = batch.loading_metadata.num_surface_value_channels.to(self.device)
        surface_attr_channels = batch.loading_metadata.num_surface_attr_channels.to(self.device)

        pos = pos.to(self.device)
        values = values.to(self.device)
        num_points = num_points.to(self.device)
        global_params = global_params.to(self.device)

        channels_last = batch.loading_metadata.channel_last[0]
        if channels_last:
            values = torch.permute(values, (0, 2, 3, 1))  # (B, N, T, C) -> (B, T, C, N)
            pos = torch.permute(pos, (0, 2, 3, 1))  # (B, N, T, D) -> (B, T, D, N)

        # Automatically pad 2D spatial coordinates to 3D (with dummy z=0) if model expects 3D
        if pos.shape[2] < self.spatial_dim:
            diff = self.spatial_dim - pos.shape[2]
            zeros = torch.zeros(pos.shape[0], pos.shape[1], diff, pos.shape[3], device=pos.device, dtype=pos.dtype)
            pos = torch.cat([pos, zeros], dim=2)
        elif pos.shape[2] > self.spatial_dim:
            pos = pos[:, :, :self.spatial_dim, :]

        return UnstructuredData(values=values[:, start:end, ...], pos=pos[:, start:end, ...], global_params=global_params, num_points=num_points, channels_last=channels_last,
                                surface_points=surface_points, surface_attr_channels=surface_attr_channels, surface_value_channels=surface_value_channels)

    def _scale_data(self, data: UnstructuredData, batch: DataSample) -> UnstructuredData:
        values = data.values  # (B, T, A, N)

        val_subtract = batch.loading_metadata.val_subtract.to(self.device)  # (B, 1, A)
        val_divide = batch.loading_metadata.val_divide.to(self.device)  # (B, 1, A)
        if len(val_subtract.shape) == 3:
            val_subtract = val_subtract.squeeze(1)
        if len(val_divide.shape) == 3:
            val_divide = val_divide.squeeze(1)
        values = values - val_subtract.unsqueeze(1).unsqueeze(-1)  # (B, T, A, N)
        values = values / val_divide.unsqueeze(1).unsqueeze(-1)  # (B, T, A, N)

        pos = data.pos  # (B, T, D, N)
        pos_min = batch.physical_metadata.min_domain.to(self.device)  # (B, D)
        pos_max = batch.physical_metadata.max_domain.to(self.device)  # (B, D)
        if pos_min.shape[-1] < pos.shape[2]:
            diff = pos.shape[2] - pos_min.shape[-1]
            pos_min = torch.cat([pos_min, torch.zeros(*pos_min.shape[:-1], diff, device=self.device, dtype=pos_min.dtype)], dim=-1)
            pos_max = torch.cat([pos_max, torch.ones(*pos_max.shape[:-1], diff, device=self.device, dtype=pos_max.dtype)], dim=-1)
        elif pos_min.shape[-1] > pos.shape[2]:
            pos_min = pos_min[..., :pos.shape[2]]
            pos_max = pos_max[..., :pos.shape[2]]

        pos = (pos - pos_min.unsqueeze(1).unsqueeze(-1)) / (pos_max - pos_min).unsqueeze(1).unsqueeze(-1)  # (B, T, D, N)

        data = dataclasses.replace(data, values=values, pos=pos)

        return data

    def _unscale_data(self, data: UnstructuredData, batch: DataSample) -> UnstructuredData:
        values = data.values  # (B, T, C, N)

        val_subtract = batch.loading_metadata.val_subtract.to(self.device)  # (B, 1, C)
        val_divide = batch.loading_metadata.val_divide.to(self.device)  # (B, 1, C)
        if len(val_subtract.shape) == 3:
            val_subtract = val_subtract.squeeze(1)
        if len(val_divide.shape) == 3:
            val_divide = val_divide.squeeze(1)
        values = values * val_divide.unsqueeze(1).unsqueeze(-1)  # (B, T, C, N)
        values = values + val_subtract.unsqueeze(1).unsqueeze(-1)  # (B, T, C, N)

        pos = data.pos  # (B, T, D, N)
        pos_min = batch.physical_metadata.min_domain.to(self.device)  # (B, D)
        pos_max = batch.physical_metadata.max_domain.to(self.device)  # (B, D)
        if pos_min.shape[-1] < pos.shape[2]:
            diff = pos.shape[2] - pos_min.shape[-1]
            pos_min = torch.cat([pos_min, torch.zeros(*pos_min.shape[:-1], diff, device=self.device, dtype=pos_min.dtype)], dim=-1)
            pos_max = torch.cat([pos_max, torch.ones(*pos_max.shape[:-1], diff, device=self.device, dtype=pos_max.dtype)], dim=-1)
        elif pos_min.shape[-1] > pos.shape[2]:
            pos_min = pos_min[..., :pos.shape[2]]
            pos_max = pos_max[..., :pos.shape[2]]

        pos = pos * (pos_max - pos_min).unsqueeze(1).unsqueeze(-1) + pos_min.unsqueeze(1).unsqueeze(-1)  # (B, T, D, N)

        data = dataclasses.replace(data, values=values, pos=pos)

        return data

    def _shared_step(self, batch: DataSample, stage: str) -> torch.Tensor:

        data_input = self.build_unstructured(batch, start=0, end=1)
        data_output = self.build_unstructured(batch, start=1, end=2)

        data_input = self._scale_data(data_input, batch)
        data_output = self._scale_data(data_output, batch)

        data_prediction = self.forward(data_input)

        loss = torch.nn.functional.mse_loss(data_prediction.values, data_output.values)

        batch_size = batch.values.shape[0]

        self.log(f"{self.stage}/{stage}/loss", loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=batch_size)

        return loss

    def _forward(self, data: UnstructuredData) -> UnstructuredData:

        return self.model._forward(data)

    def preprocess(self, data: UnstructuredData, generator: Optional[torch.Generator] = None) -> None:
        pass

    def preprocess_from_batch(self, batch, generator: Optional[torch.Generator] = None) -> None:

        data_input = self.build_unstructured(batch, start=0, end=1)

        self.preprocess(data_input, generator=generator)

    def predict(self, batch, rollout_num: int, generator: Optional[torch.Generator] = None, **kwargs) -> Tuple[UnstructuredData, UnstructuredData]:

        data_input = self.build_unstructured(batch, start=0, end=1)
        data_output = self.build_unstructured(batch, start=0, end=rollout_num)

        data_input = self._scale_data(data_input, batch)

        scaled_data_prediction = self._unscale_data(data_input, batch)
        predict_list: List[UnstructuredData] = [scaled_data_prediction]

        if rollout_num >= 1:
            with torch.no_grad():
                for t in range(rollout_num - 1):
                    data_prediction = self.forward(data_input)
                    scaled_data_prediction = self._unscale_data(data_prediction, batch)
                    predict_list.append(scaled_data_prediction)
                    data_input = data_prediction

        data_prediction = concatenate_unstructured(predict_list)

        return data_prediction, data_output

    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        train_ds = getattr(self.trainer.datamodule, "train_dataset", None)
        if hasattr(train_ds, "set_epoch"):
            train_ds.set_epoch(self.current_epoch)

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
            total_steps = self.trainer.estimated_stepping_batches
            max_epochs = self.trainer.max_epochs
            if max_epochs and max_epochs > 0:
                extra_kwargs["steps_per_epoch"] = total_steps // max_epochs
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
