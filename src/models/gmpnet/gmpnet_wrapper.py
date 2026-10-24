import dataclasses

import torch

from src.models.dataclasses import UnstructuredData

from .gmpnet import GMPNet


class GMPNetWrapper(GMPNet):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.print_parameter_breakdown()

    def _get_batch_constant_int(self, name: str, value: torch.Tensor | int) -> int:
        if isinstance(value, torch.Tensor):
            flat = value.reshape(-1)
            if flat.numel() == 0:
                raise ValueError(f"{name} is empty; expected a scalar or batch-constant tensor.")
            first = flat[0]
            if not torch.all(flat == first).item():
                raise ValueError(f"{name} must be constant across the batch for GMPNetWrapper.")
            return int(first.item())

        return int(value)

    def _forward(self, data: UnstructuredData) -> UnstructuredData:

        num_surface_points = self._get_batch_constant_int("surface_points", data.surface_points)
        num_points = self._get_batch_constant_int("num_points", data.num_points)
        num_volume_points = num_points - num_surface_points

        # if num_surface_points.sum() == 0 or num_volume_points.sum() == 0:
        #     raise NotImplementedError("SMARTWrapper currently only supports both surface and volume points together.")

        pos = data.pos  # (B, T, D, N)
        values = data.values
        context = data.global_params  # (B, G)

        surface_attr_channels = data.surface_attr_channels
        volume_attr_channels = values.shape[2] - surface_attr_channels

        B, T, C, N = values.shape
        D = pos.shape[2]
        if T > 1:
            raise NotImplementedError("SMARTWrapper currently only supports T=1.")

        pos = pos.squeeze(1).permute((0, 2, 1)).contiguous()
        values = values.squeeze(1).permute((0, 2, 1)).contiguous()

        geometry_point_cloud = pos[:, -num_surface_points:, :]  # (batch size, number surface points, spatial_dim)
        surface_query_coordinates = pos[:, -num_surface_points:, :]  # (batch size, number surface points, spatial_dim)
        volume_query_coordinates = pos[:, :num_volume_points, :]  # (batch size, number volume points, spatial_dim)

        values_surface, values_volume = self.forward(geometry_point_cloud, None, None, None, surface_query_coordinates, volume_query_coordinates, context)  # (batch size, number surface points, surface_channels), (batch size, number volume points, volume_channels)
        values_surface = values_surface.permute((0, 2, 1)).unsqueeze(1)  # (B, 1, surface_channels, number surface points)
        values_volume = values_volume.permute((0, 2, 1)).unsqueeze(1)  # (B, 1, volume_channels, number volume points)

        volume_value_channels = values_volume.shape[2]
        surface_value_channels = values_surface.shape[2]

        device = values_surface.device
        dtype = values_surface.dtype
        values_volume = torch.cat([values_volume, torch.zeros((B, 1, volume_value_channels, num_surface_points), device=device, dtype=dtype)], dim=-1)
        values_surface = torch.cat([torch.zeros((B, 1, surface_value_channels, num_volume_points), device=device, dtype=dtype), values_surface], dim=-1)
        values = torch.cat([values_volume, values_surface], dim=2)

        out_data = dataclasses.replace(data, values=values)
        return out_data

# Alias for backwards compatibility
SMARTWrapper = GMPNetWrapper