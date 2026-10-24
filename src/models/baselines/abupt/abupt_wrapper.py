import dataclasses
from typing import Optional, Any

import torch
import torch.nn as nn

from src.models.dataclasses import UnstructuredData
from src.models.baselines.abupt.src.model import AnchoredBranchedUPT


class ABUPTWrapper(AnchoredBranchedUPT):

    def _get_batch_constant_int(self, name: str, value: torch.Tensor | int) -> int:
        if isinstance(value, torch.Tensor):
            flat = value.reshape(-1)
            if flat.numel() == 0:
                raise ValueError(f"{name} is empty; expected a scalar or batch-constant tensor.")
            first = flat[0]
            if not torch.all(flat == first).item():
                raise ValueError(f"{name} must be constant across the batch for ABUPTWrapper.")
            return int(first.item())

        return int(value)

    def _forward(self, data: UnstructuredData) -> UnstructuredData:

        num_surface_points = self._get_batch_constant_int("surface_points", data.surface_points)
        num_points = self._get_batch_constant_int("num_points", data.num_points)
        num_volume_points = num_points - num_surface_points

        pos = data.pos  # (B, T, D, N)
        values = data.values  
        context = data.global_params  # (B, G)

        surface_attr_channels = data.surface_attr_channels
        volume_attr_channels = values.shape[2] - surface_attr_channels

        B, T, C, N = values.shape
        if B != 1:
            raise NotImplementedError("ABUPTWrapper currently only supports B=1.")
        
        pos = (pos.squeeze(1).permute((0, 2, 1))*1000).contiguous()
        device = pos.device

        geometry_point_cloud = pos[0, num_volume_points:, :]  # (batch size, number surface points, spatial_dim)
        surface_query_coordinates = pos[:, num_volume_points:, :]  # (batch size, number surface points, spatial_dim)
        volume_query_coordinates = pos[:, :num_volume_points, :]  # (batch size, number volume points, spatial_dim)

        # pick supernode indices (coarse subset)
        num_supernodes = max(1, int(num_surface_points * self.supernode_fraction))
        geometry_supernode_idx = torch.randperm(num_surface_points, device=device)[:num_supernodes]

        geometry_batch_idx = None  # AB-UPT asserts bs==1 support
        
        surface_anchor_position = surface_query_coordinates[:, :self.num_anchors]  # (num_anchors, D)
        volume_anchor_position = volume_query_coordinates[:, :self.num_anchors]  # (num_anchors, D)
        
        surface_query_position = surface_query_coordinates[:, self.num_anchors:]  # (num_surface_points - num_anchors, D)
        volume_query_position = volume_query_coordinates[:, self.num_anchors:]  # (num_volume_points - num_anchors, D)

        # call model
        outputs = self.forward(
            geometry_position=geometry_point_cloud,
            geometry_supernode_idx=geometry_supernode_idx,
            geometry_batch_idx=geometry_batch_idx,
            surface_anchor_position=surface_anchor_position,
            volume_anchor_position=volume_anchor_position,
            surface_query_position=surface_query_position,
            volume_query_position=volume_query_position,
        )

        ps_anchor = outputs.get("surface_anchor_pressure")  
        wss_anchor = outputs.get("surface_anchor_wallshearstress")
        ps_query = outputs.get("surface_query_pressure")
        wss_query = outputs.get("surface_query_wallshearstress")
        combined_query = torch.cat([ps_query, wss_query], dim=1)
        combined_anchor = torch.cat([ps_anchor, wss_anchor], dim=1)
        combined = torch.cat([combined_anchor, combined_query], dim=0)
        values_surface = combined.transpose(0, 1).unsqueeze(0).unsqueeze(0)

        pv_anchor = outputs.get("volume_anchor_pressure")
        velocity_anchor = outputs.get("volume_anchor_velocity")
        pv_query = outputs.get("volume_query_pressure")
        velocity_query = outputs.get("volume_query_velocity")
        combined_volume_query = torch.cat([pv_query, velocity_query], dim=1)
        combined_volume_anchor = torch.cat([pv_anchor, velocity_anchor], dim=1)
        combined_volume = torch.cat([combined_volume_anchor, combined_volume_query], dim=0)
        values_volume = combined_volume.transpose(0, 1).unsqueeze(0).unsqueeze(0)

        volume_value_channels = values_volume.shape[2]
        surface_value_channels = values_surface.shape[2]

        device = values_surface.device
        dtype = values_surface.dtype
        values_volume = torch.cat([values_volume, torch.zeros((B, 1, volume_value_channels, num_surface_points), device=device, dtype=dtype)], dim=-1)
        values_surface = torch.cat([torch.zeros((B, 1, surface_value_channels, num_volume_points), device=device, dtype=dtype), values_surface], dim=-1)
        values = torch.cat([values_volume, values_surface], dim=2)

        out_data = dataclasses.replace(data, values=values)
        return out_data
