import os
from typing import Optional, List, Dict

from torch import Tensor
from torchmetrics import Metric
import torch
import pandas as pd
from einops import reduce

from src.metrics.simulation_metric import SimulationMetric


class MSE(SimulationMetric):

    def __init__(self, *args, **kwargs) -> None:

        super().__init__(*args, **kwargs)
        self.name = "mse"

    def _metric_impl(self, prediction: Tensor, target: Tensor, eps: float = 1e-7) -> Tensor:

        target = target[:, :prediction.shape[1]]

        assert prediction.shape == target.shape, \
            f"Prediction shape {prediction.shape} and target shape {target.shape} do not match."

        mse = (prediction - target) ** 2
        mse = reduce(mse, 'b t c ... -> b t', 'mean')

        return mse