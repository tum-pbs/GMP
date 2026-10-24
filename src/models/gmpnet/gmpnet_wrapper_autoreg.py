import dataclasses

import torch

from src.models.dataclasses import UnstructuredData

from .gmpnet import GMPNet


class GMPNetWrapper(GMPNet):

    def __init__(
        self,
        *args,
        concat_pos_to_physics: bool = True,
        surface_channels_input: int = 0,
        volume_channels_input: int = 0,
        spatial_dim: int = 3,
        **kwargs,
    ):
        self.concat_pos_to_physics = concat_pos_to_physics
        self.raw_surface_channels_input = surface_channels_input
        self.raw_volume_channels_input = volume_channels_input

        # When concat_pos_to_physics is True, the physics branch takes [pos, physics]
        eff_surface_channels_input = (
            surface_channels_input + spatial_dim
            if (concat_pos_to_physics and surface_channels_input > 0)
            else surface_channels_input
        )
        eff_volume_channels_input = (
            volume_channels_input + spatial_dim
            if (concat_pos_to_physics and volume_channels_input > 0)
            else volume_channels_input
        )

        super().__init__(
            *args,
            spatial_dim=spatial_dim,
            surface_channels_input=eff_surface_channels_input,
            volume_channels_input=eff_volume_channels_input,
            **kwargs,
        )
        self.print_parameter_breakdown()

    def _get_batch_constant_int(self, name: str, value: torch.Tensor | int) -> int:
        if isinstance(value, torch.Tensor):
            flat = value.reshape(-1)
            if flat.numel() == 0:
                raise ValueError(f"{name} is empty; expected a scalar or batch-constant tensor.")
            first = flat[0]
            if not torch.all(flat == first).item():
                raise ValueError(f"{name} must be constant across the batch.")
            return int(first.item())

        return int(value)

    def _prepare_physics(
        self,
        pos_pts: torch.Tensor,
        val_pts: torch.Tensor | None,
        expected_channels: int,
    ) -> torch.Tensor | None:
        if expected_channels <= 0 or val_pts is None:
            return None
        if val_pts.shape[-1] < expected_channels:
            diff = expected_channels - val_pts.shape[-1]
            val_pts = torch.cat([val_pts, torch.zeros(*val_pts.shape[:-1], diff, device=val_pts.device, dtype=val_pts.dtype)], dim=-1)
        elif val_pts.shape[-1] > expected_channels:
            val_pts = val_pts[..., :expected_channels]

        if self.concat_pos_to_physics:
            return torch.cat([pos_pts, val_pts], dim=-1)
        return val_pts

    def _forward(self, data: UnstructuredData) -> UnstructuredData:
        pos = data.pos
        values = data.values
        context = data.global_params

        # Normalize context conditioning tensor
        if context is not None:
            if context.numel() == 0 or (context.ndim > 1 and context.shape[-1] == 0):
                context = None
            elif context.ndim == 1:
                context = context.unsqueeze(-1)

        # Standardize (pos, values) shapes to (B, N, D) and (B, N, C) for T=1
        if values.ndim == 4:
            B = values.shape[0]
            if values.shape[1] == 1:
                # Shape: (B, 1, C, N) -> (B, N, C)
                pos = pos.squeeze(1).permute(0, 2, 1).contiguous()
                values = values.squeeze(1).permute(0, 2, 1).contiguous()
            elif values.shape[2] == 1:
                # Shape: (B, N, 1, C) -> (B, N, C)
                pos = pos.squeeze(2).contiguous()
                values = values.squeeze(2).contiguous()
            else:
                raise NotImplementedError(
                    f"GMPNetWrapper expected T=1 for autoregressive rollout steps, got values shape {values.shape}."
                )
        elif values.ndim == 3:
            B = values.shape[0]
        else:
            raise ValueError(f"Unexpected values ndim: {values.ndim}, shape: {values.shape}")

        # Ensure spatial dimension compatibility (pad with zeros if fewer, truncate if more)
        if pos.shape[-1] < self.spatial_dim:
            diff = self.spatial_dim - pos.shape[-1]
            pos = torch.cat([pos, torch.zeros(*pos.shape[:-1], diff, device=pos.device, dtype=pos.dtype)], dim=-1)
        elif pos.shape[-1] > self.spatial_dim:
            pos = pos[:, :, :self.spatial_dim]

        total_points = pos.shape[1]
        total_channels = values.shape[-1]

        # Extract surface metadata if available
        num_surface_points = None
        if hasattr(data, "surface_points") and data.surface_points is not None:
            try:
                num_surface_points = self._get_batch_constant_int("surface_points", data.surface_points)
            except Exception:
                num_surface_points = None

        num_surface_val_channels = None
        if hasattr(data, "surface_value_channels") and data.surface_value_channels is not None:
            try:
                num_surface_val_channels = self._get_batch_constant_int("surface_value_channels", data.surface_value_channels)
            except Exception:
                num_surface_val_channels = None

        model_has_surface = (self.surface_channels > 0 or self.raw_surface_channels_input > 0)
        model_has_volume = (self.volume_channels > 0 or self.raw_volume_channels_input > 0)

        # Identify active modality
        if model_has_surface and not model_has_volume:
            use_surface = True
            use_volume = False
        elif model_has_volume and not model_has_surface:
            use_surface = False
            use_volume = True
        else:
            # Model supports both (or neither explicitly constrained): inspect data annotations
            # Note: in datasets like KS or Ellipse, data.surface_points is set to total_points,
            # but num_surface_value_channels is 0 (volume PDE data).
            if num_surface_val_channels is not None and num_surface_val_channels == 0:
                use_surface = False
                use_volume = True
            elif num_surface_points is not None:
                if num_surface_points == 0:
                    use_surface = False
                    use_volume = True
                elif num_surface_points == total_points:
                    use_surface = True
                    use_volume = False
                elif 0 < num_surface_points < total_points:
                    use_surface = True
                    use_volume = True
                else:
                    use_surface = False
                    use_volume = True
            elif total_channels == self.raw_volume_channels_input and total_channels != self.raw_surface_channels_input:
                use_surface = False
                use_volume = True
            elif total_channels == self.raw_surface_channels_input and total_channels != self.raw_volume_channels_input:
                use_surface = True
                use_volume = False
            else:
                use_surface = model_has_surface
                use_volume = model_has_volume

        # Execute forward pass based on identified modality
        if use_surface and use_volume:
            # Both surface and volume data
            num_vol_pts = total_points - (num_surface_points if num_surface_points is not None else 0)
            num_surf_pts = total_points - num_vol_pts

            pos_vol = pos[:, :num_vol_pts, :]
            pos_surf = pos[:, num_vol_pts:, :]

            raw_val_vol = values[:, :num_vol_pts, :self.raw_volume_channels_input] if self.raw_volume_channels_input > 0 else None
            if self.raw_surface_channels_input > 0:
                if values.shape[-1] >= self.raw_volume_channels_input + self.raw_surface_channels_input:
                    raw_val_surf = values[:, num_vol_pts:, self.raw_volume_channels_input:self.raw_volume_channels_input + self.raw_surface_channels_input]
                else:
                    raw_val_surf = values[:, num_vol_pts:, -self.raw_surface_channels_input:]
            else:
                raw_val_surf = None

            val_vol = self._prepare_physics(pos_vol, raw_val_vol, self.raw_volume_channels_input)
            val_surf = self._prepare_physics(pos_surf, raw_val_surf, self.raw_surface_channels_input)

            values_surface, values_volume = self.forward(
                geo_surface=pos_surf,
                geo_volume=pos_vol,
                physics_surface=val_surf,
                physics_volume=val_vol,
                surf_query_pos=pos_surf,
                vol_query_pos=pos_vol,
                params=context,
            )

            values_surface = values_surface.permute(0, 2, 1).unsqueeze(1)  # (B, 1, C_surf, N_surf)
            values_volume = values_volume.permute(0, 2, 1).unsqueeze(1)    # (B, 1, C_vol, N_vol)

            device = values_surface.device
            dtype = values_surface.dtype
            c_vol = values_volume.shape[2]
            c_surf = values_surface.shape[2]

            val_vol_padded = torch.cat([values_volume, torch.zeros(B, 1, c_vol, num_surf_pts, device=device, dtype=dtype)], dim=-1)
            val_surf_padded = torch.cat([torch.zeros(B, 1, c_surf, num_vol_pts, device=device, dtype=dtype), values_surface], dim=-1)
            out_values = torch.cat([val_vol_padded, val_surf_padded], dim=2)

        elif use_surface:
            # Surface data only
            pos_surf = pos
            raw_val_surf = values[:, :, :self.raw_surface_channels_input] if self.raw_surface_channels_input > 0 else None
            val_surf = self._prepare_physics(pos_surf, raw_val_surf, self.raw_surface_channels_input)

            values_surface, _ = self.forward(
                geo_surface=pos_surf,
                geo_volume=None,
                physics_surface=val_surf,
                physics_volume=None,
                surf_query_pos=pos_surf,
                vol_query_pos=None,
                params=context,
            )
            out_values = values_surface.permute(0, 2, 1).unsqueeze(1)  # (B, 1, C_surf, N)

        else:
            # Volume data only
            pos_vol = pos
            raw_val_vol = values[:, :, :self.raw_volume_channels_input] if self.raw_volume_channels_input > 0 else None
            val_vol = self._prepare_physics(pos_vol, raw_val_vol, self.raw_volume_channels_input)

            _, values_volume = self.forward(
                geo_surface=None,
                geo_volume=pos_vol,
                physics_surface=None,
                physics_volume=val_vol,
                surf_query_pos=None,
                vol_query_pos=pos_vol,
                params=context,
            )
            out_values = values_volume.permute(0, 2, 1).unsqueeze(1)  # (B, 1, C_vol, N)

        out_data = dataclasses.replace(data, values=out_values)
        return out_data


# Alias for backwards compatibility
SMARTWrapper = GMPNetWrapper