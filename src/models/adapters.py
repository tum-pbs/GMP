from abc import ABC
from typing import Optional

import lightning as L
import torch

from src.models.dataclasses import UnstructuredData


class TransientEmulatorAdapter(L.LightningModule, ABC):

    def __init__(self) -> None:
        super().__init__()
        self.stage = "train"

    def forward(self, data: UnstructuredData) -> UnstructuredData:
        raise NotImplementedError("This is a base adapter class. " "Please implement the forward method " "in a subclass.")

    def preprocess(self, data: UnstructuredData, generator: Optional[torch.Generator] = None) -> None:
        pass

    def preprocess_from_batch(self, batch, generator: Optional[torch.Generator] = None) -> None:
        pass

    def set_stage(self, stage: str) -> None:
        self.stage = stage


class WindowTransformerAdapter(TransientEmulatorAdapter, ABC):

    def __init__(self) -> None:
        super().__init__()

        self.cached = False
        self.cached_windows = []
        self.cached_shifted_windows = []

    def forward(self, data: UnstructuredData, use_cache=False) -> UnstructuredData:

        windows = None
        shifted_windows = None

        if use_cache:
            windows = self.cached_windows
            shifted_windows = self.cached_shifted_windows

        return self._forward(data, windows, shifted_windows)

    def _forward(self, data: UnstructuredData, windows, shifted_windows):

        raise NotImplementedError("This is a base adapter class. ")

    def compute_windows(self, pos: torch.Tensor):

        raise NotImplementedError("This is a base adapter class. ")

    def compute_windows_from_batch(self, pos: torch.Tensor):

        raise NotImplementedError("This is a base adapter class. ")

    def _cache_windows(self, windows, shifted_windows):

        self.cached_windows = windows
        self.cached_shifted_windows = shifted_windows
        self.cached = True

    def clear_cache(self):

        self.cached_windows = []
        self.cached_shifted_windows = []
        self.cached = False


class UnstructuredTransformerAdapter(TransientEmulatorAdapter, ABC):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, data: UnstructuredData) -> UnstructuredData:

        return self._forward(data)

    def _forward(self, data: UnstructuredData):

        raise NotImplementedError("This is a base adapter class. ")


class SteadyStateEmulatorAdapter(L.LightningModule):

    def __init__(self) -> None:
        super().__init__()

    def forward(self):
        raise NotImplementedError("This is a base adapter class. " "Please implement the forward method " "in a subclass.")
