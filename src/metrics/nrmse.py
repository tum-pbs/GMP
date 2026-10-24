import os
from typing import Optional, List, Dict

from math import prod
from torch import Tensor
from torchmetrics import Metric
import numpy as np
import torch
import pandas as pd
from einops import einsum, reduce

from src.metrics.simulation_metric import SimulationMetric


class NRMSE(SimulationMetric):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.name = "nrmse"

    def _metric_impl(self, prediction: Tensor, target: Tensor, eps: float = 1e-7) -> Tensor:
        """
        Computes the NRMSE between prediction and target
        :param prediction: Tensor of shape (batch_size, time, channels, ...) where spatial dimensions are at the end
        :param target: Tensor of shape (batch_size, time, channels, ...) where spatial dimensions are at the end
        :param eps: Normalization epsilon to avoid division by zero
        :return: NRMSE tensor of shape (batch_size, time)
        """
        target = target[:, :prediction.shape[1]]

        assert prediction.shape == target.shape, \
            f"Prediction shape {prediction.shape} and target shape {target.shape} do not match."

        mse = (prediction - target) ** 2
        mse = reduce(mse, "i j ...-> i j", "mean")

        target_squared = target ** 2
        target_squared = reduce(target_squared, "i j ...-> i j", "mean")

        temp = mse / (target_squared + eps)

        nrmse = torch.sqrt(temp)

        return nrmse