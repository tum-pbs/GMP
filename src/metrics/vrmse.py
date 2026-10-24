import os
from typing import Optional, List, Dict

from torch import Tensor
from torchmetrics import Metric
import torch
import pandas as pd
from einops import reduce

from src.metrics.simulation_metric import SimulationMetric


class VRMSE(SimulationMetric):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.name = "vrmse"

    def _metric_impl(self, prediction: Tensor, target: Tensor, eps: float = 1e-7) -> Tensor:
        """
        Computes the VRMSE between prediction and target
        :param prediction: Tensor of shape (batch_size, time, channels, ...) where spatial dimensions are at the end
        :param target: Tensor of shape (batch_size, time, channels, ...) where spatial dimensions are at the end
        :param eps: Normalization epsilon to avoid division by zero
        :return: VRMSE tensor of shape (batch_size, time)
        """
        target = target[:, :prediction.shape[1]]

        assert prediction.shape == target.shape, \
            f"Prediction shape {prediction.shape} and target shape {target.shape} do not match."

        mse = (prediction - target) ** 2
        mse = reduce(mse, 'b t c ... -> b t', 'mean')

        target_avg = reduce(target, 'b t c ... -> b t', 'mean')
        temp = target - target_avg.view(target_avg.shape + (1,) * (len(target.shape) - 2))
        target_variance = reduce(temp ** 2, 'b t ... -> b t', 'mean')

        temp = mse * (1 / (target_variance + eps))
        vrmse = torch.sqrt(temp)

        return vrmse