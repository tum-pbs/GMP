import os
from abc import ABC, abstractmethod
from typing import Optional, List, Dict

from torch import Tensor
from torchmetrics import Metric
import torch
import pandas as pd
from einops import reduce

class SimulationMetric(Metric, ABC):

    def __init__(self, num_tasks: int, max_length: int = 100,
                 dataset_names: Optional[List[str]] = None, field_name: str = 'field_0', **kwargs):

        super().__init__(**kwargs)

        self.name = "default"
        self.times = list(range(max_length))
        self.num_tasks = num_tasks
        self.max_length = max_length

        self.num_fields = 1
        self.field_name = field_name

        if dataset_names is None:
            dataset_names = [f"dataset_{i}" for i in range(num_tasks)]

        self.dataset_names = dataset_names

        self.add_state("metric_sum", default=torch.zeros(size=(num_tasks, len(self.times))),
                       dist_reduce_fx="sum")

        self.add_state("total", default=torch.zeros(size=(num_tasks, len(self.times))),
                       dist_reduce_fx="sum")

    def _update(self, metric: Tensor, class_labels: Tensor) -> None:
        """
        Update the internal state with the given metric and class labels
        :param metric: tensor of shape (batch_size, num_time_dim) containing the metric values
        :param class_labels: tensor of shape (batch_size,) containing the class labels
        """

        num_time_dim = len(self.times)
        filtered_times = [time for time in self.times if time < metric.shape[1]]

        index_tensor = torch.tensor(filtered_times).to(metric.device).unsqueeze(0).repeat(metric.shape[0], 1)
        metric = metric.gather(dim=1, index=index_tensor)

        metric = torch.nn.functional.pad(metric, (0, num_time_dim - metric.shape[1]))

        mse_init = torch.zeros_like(self.metric_sum)

        # update mse_sum at index from class_labels
        mse = mse_init.scatter_add(dim=0, index=class_labels.unsqueeze(1).expand(-1, metric.shape[1]),
                                   src=metric)

        self.metric_sum += mse

        update_count = torch.zeros_like(self.total)
        index_tensor = class_labels.unsqueeze(1).expand(-1, update_count.shape[1])
        src = torch.ones_like(index_tensor).float()

        update_count = update_count.scatter_add(dim=0,
                                                index=index_tensor,
                                                src=src)

        self.total += update_count

    @abstractmethod
    def _metric_impl(self, prediction: Tensor, target: Tensor):
        raise NotImplementedError

    def update(self, prediction: Tensor, target: Tensor, dataset_idx: Tensor, eps: float = 1e-7) -> None:
        """
        Update the metric with predictions and target values.
        :param prediction: tensor of shape (batch_size, time, 1, spatial_dims...) containing the predictions
        :param target: tensor of shape (batch_size, time, 1, spatial_dims...) containing the target values
        :param dataset_idx: vector of shape (batch_size,) containing the class labels
        :param eps: small value to avoid division by zero
        :return:
        """

        value = self._metric_impl(prediction, target)

        self._update(value, dataset_idx)

    def compute(self) -> Tensor:

        result = self.metric_sum.float() / self.total
        result = torch.where(torch.isnan(result), torch.zeros_like(result), result)
        return result

    def get_dict(self, prefix: str = '') -> Dict[str, Tensor]:

        result: Dict[str, Tensor] = {}
        metric = self.compute().T
        for i, dataset_name in enumerate(self.dataset_names):
            for j, time in enumerate(self.times):
                result[f'{prefix}/{dataset_name}/rollout_{time}/{self.name}'] = metric[j, i]
        return result

    def save(self, savedir, prefix: str = '') -> None:

        savedir = os.path.join(savedir, prefix)

        # create path if not exists
        if not os.path.exists(savedir):
            os.makedirs(savedir)

        metric = self.compute().T.cpu().numpy()

        # save metric as csv
        metric_df = pd.DataFrame(metric[:, :len(self.dataset_names)], columns=self.dataset_names)
        metric_df["time"] = self.times
        metric_df = metric_df.set_index("time")
        metric_df.to_csv(os.path.join(savedir, f"{self.name}.csv"))