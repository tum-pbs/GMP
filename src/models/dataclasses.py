from torch import Tensor
from dataclasses import dataclass

@dataclass
class UnstructuredData:
    values: Tensor # (B, T, C, N) if channels_last=False else (B, N, T, C)
    pos: Tensor # (B, T, D, N) if channels_last=False else (B, N, T, D)
    num_points: Tensor # (B, T)
    global_params: Tensor # (B, G)
    surface_points: Tensor
    surface_value_channels: Tensor
    surface_attr_channels: Tensor
    channels_last: bool = False