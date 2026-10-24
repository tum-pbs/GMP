from dataclasses import dataclass, fields, is_dataclass, replace

import torch


def asdict_shallow(obj) -> dict:
    assert is_dataclass(obj), "This method should be called on dataclass instances only."

    def _asdict_shallow_rec(obj):
        if is_dataclass(obj):
            return {f.name: _asdict_shallow_rec(getattr(obj, f.name)) for f in fields(obj)}
        elif isinstance(obj, list):
            return [_asdict_shallow_rec(v) for v in obj]
        elif isinstance(obj, dict):
            return {k: _asdict_shallow_rec(v) for k, v in obj.items()}
        else:
            # Return the object reference as-is (no deepcopy)
            return obj

    return _asdict_shallow_rec(obj)  # type: ignore


def from_dict(cls, data: dict):
    """Utility function to create a dataclass instance from a dictionary."""
    if not is_dataclass(cls):
        raise ValueError("from_dict can only be used with dataclass types.")

    def _from_dict_rec(cls, data: dict):
        if is_dataclass(cls):
            fieldtypes = {f.name: f.type for f in fields(cls)}
            return cls(**{f: _from_dict_rec(fieldtypes[f], data[f]) if is_dataclass(fieldtypes[f]) else data[f] for f in data})  # type: ignore

    return _from_dict_rec(cls, data)


def pretty_info_str(obj) -> str:
    if not is_dataclass(obj):
        raise ValueError("pretty_info_str can only be used with dataclass instances.")

    info_lines = [f"{obj.__class__.__name__}:"]  # type: ignore
    for field in fields(obj):
        value = getattr(obj, field.name)
        if is_dataclass(value):
            nested_info = pretty_info_str(value)
            nested_info_indented = "\n".join("    " + line for line in nested_info.split("\n"))
            info_lines.append(nested_info_indented)
        elif isinstance(value, torch.Tensor):
            info_lines.append(f"    {field.name}: Tensor(shape={value.shape}, dtype={value.dtype})")
        else:
            info_lines.append(f"    {field.name}: {value}")

    return "\n".join(info_lines)


@dataclass
class PhysicalMetadata:
    min_domain: torch.Tensor  # (D)
    max_domain: torch.Tensor  # (D)
    hparams: torch.Tensor  # (P)
    dt: torch.Tensor  # ()
    num_hparams: torch.Tensor  # () = P
    num_attrs: torch.Tensor  # () = A


@dataclass
class LoadingMetadata:
    name: str  # ()
    values_name: list[str]  # (C)
    attrs_name: list[str]  # (A)
    hparams_name: list[str]  # (P)
    grid_dims: tuple[int, ...]  # ()
    channel_last: bool  # ()
    val_subtract: torch.Tensor  # (T, C)
    val_divide: torch.Tensor  # (T, C)
    attr_subtract: torch.Tensor  # (T, A)
    attr_divide: torch.Tensor  # (T, A)
    num_surface_points: torch.Tensor 
    num_surface_value_channels: torch.Tensor
    num_surface_attr_channels: torch.Tensor


@dataclass
class DataSample:
    positions: torch.Tensor  # (T, N, D) or (N, T, D)
    values: torch.Tensor  # (T, N, C) or (N, T, C)
    attributes: torch.Tensor  # (T, N, A) or (N, T, A)
    num_points: torch.Tensor  # ()
    physical_metadata: PhysicalMetadata
    loading_metadata: LoadingMetadata
    global_params: torch.Tensor  # (G) - New field for global parameters

    def to_dict(self) -> dict:
        """Get a shallow dictionary representation of the DataSample.

        Returns:
            dict: A shallow dictionary representation of the DataSample.
        """
        return asdict_shallow(self)

    @classmethod
    def from_dict(cls, data: dict) -> "DataSample":
        """Create a DataSample instance from a dictionary.

        Args:
            data (dict): A dictionary representation of a DataSample.

        Returns:
            DataSample: A DataSample instance created from the dictionary.
        """
        return from_dict(cls, data)  # type: ignore

    def info_str(self) -> str:
        """Get a pretty string representation of the DataSample.

        Returns:
            str: A pretty string representation of the DataSample.
        """
        return pretty_info_str(self)

    def get_slice(self, start: int, end: int | None = None) -> "DataSample":
        """Get a temporal slice of the DataSample from start to end (exclusive).

        The slicing is done along the time dimension, also if this DataSample has a batch dimension (i.e., 4 dimensional positions/values/attributes).

        Args:
            start (int): The starting time index (inclusive).
            end (int | None): The ending time index (exclusive). If None, slice until the end.

        Returns:
            DataSample: A new DataSample instance containing the sliced data.
        """
        if self.positions.dim() == 3:
            if self.loading_metadata.channel_last:
                sliced_positions = self.positions[:, start:end]  # (N, T, D)
                sliced_values = self.values[:, start:end]  # (N, T, C)
                sliced_attributes = self.attributes[:, start:end]  # (N, T, A)
            else:
                sliced_positions = self.positions[start:end]  # (T, D, N)
                sliced_values = self.values[start:end]  # (T, C, N)
                sliced_attributes = self.attributes[start:end]  # (T, A, N)
        else:
            # All entries of the `channel_last` list must be the same across the batch
            assert all(self.loading_metadata.channel_last) or not any(self.loading_metadata.channel_last), "Inconsistent `channel_last` settings across batch entries are not supported for slicing."

            if self.loading_metadata.channel_last[0]:
                sliced_positions = self.positions[:, :, start:end]  # (B, N, T, D)
                sliced_values = self.values[:, :, start:end]  # (B, N, T, C)
                sliced_attributes = self.attributes[:, :, start:end]  # (B, N, T, A)
            else:
                sliced_positions = self.positions[:, start:end]  # (B, T, D, N)
                sliced_values = self.values[:, start:end]  # (B, T, C, N)
                sliced_attributes = self.attributes[start:end]  # (B, T, A, N)

        return DataSample(
            positions=sliced_positions,
            values=sliced_values,
            attributes=sliced_attributes,
            num_points=self.num_points,
            physical_metadata=self.physical_metadata,
            loading_metadata=self.loading_metadata,
            global_params=self.global_params,  # Include global_params unchanged
        )

    def to(self, device: torch.device) -> "DataSample":
        """Move all tensor fields of the DataSample to the specified device.

        Args:
            device (torch.device): The target device.

        Returns:
            DataSample: A new DataSample instance with all tensor fields moved to the specified device.
        """
        # Move tensors in PhysicalMetadata to the specified device
        physical_metadata = PhysicalMetadata(
            min_domain=self.physical_metadata.min_domain.to(device),
            max_domain=self.physical_metadata.max_domain.to(device),
            hparams=self.physical_metadata.hparams.to(device),
            dt=self.physical_metadata.dt.to(device),
            num_hparams=self.physical_metadata.num_hparams.to(device),
            num_attrs=self.physical_metadata.num_attrs.to(device),
        )

        # Move tensors in LoadingMetadata to the specified device
        loading_metadata = LoadingMetadata(
            name=self.loading_metadata.name,
            values_name=self.loading_metadata.values_name,
            attrs_name=self.loading_metadata.attrs_name,
            hparams_name=self.loading_metadata.hparams_name,
            grid_dims=self.loading_metadata.grid_dims,
            channel_last=self.loading_metadata.channel_last,
            val_subtract=self.loading_metadata.val_subtract.to(device),
            val_divide=self.loading_metadata.val_divide.to(device),
            attr_subtract=self.loading_metadata.attr_subtract.to(device),
            attr_divide=self.loading_metadata.attr_divide.to(device),
            num_surface_points=self.loading_metadata.num_surface_points.to(device),
            num_surface_value_channels=self.loading_metadata.num_surface_value_channels.to(device),
            num_surface_attr_channels=self.loading_metadata.num_surface_attr_channels.to(device),
        )

        # Move tensors in DataSample to the specified device
        return DataSample(
            positions=self.positions.to(device),
            values=self.values.to(device),
            attributes=self.attributes.to(device),
            num_points=self.num_points.to(device),
            physical_metadata=physical_metadata,
            loading_metadata=loading_metadata,
            global_params=self.global_params.to(device),  # Move global_params to the device
        )
