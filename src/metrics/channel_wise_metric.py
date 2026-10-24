from torchmetrics import Metric
from typing import Optional, List, Dict, Callable, Any

from torch import Tensor
import os

from src.modules.unstructured_window import UnstructuredData


class ChannelWiseMetric(Metric):
    def __init__(self, num_fields: int,
                 metrics: List[Metric],
                 field_names: Optional[List[str]] = None,
                 **kwargs):

        super().__init__(**kwargs)

        self.num_fields = num_fields

        if field_names is None:
            field_names = [f"field_{i}" for i in range(num_fields)]
        self.field_names = field_names

        # clone metrics for each field
        self.field_metrics = {
            field_name: [metric.clone() for metric in metrics]
            for field_name in field_names
        }


    def reset(self) -> None:
        """
        Reset the internal state of the metric.
        """

        for field_name in self.field_names:
            for field_metric in self.field_metrics[field_name]:
                field_metric.reset()

        super().reset()

    def update(self, prediction: UnstructuredData, target: UnstructuredData, dataset_idx: Tensor) -> None:

        self.field_metrics = {
            field_name: [
                metric.to(prediction.values.device) for metric in self.field_metrics[field_name]
            ] for field_name in self.field_names
        }

        for i, field_name in enumerate(self.field_names):
            for metric in self.field_metrics[field_name]:
                field_preds = prediction.values[:, :, i:i+1]
                field_target = target.values[:, :, i:i+1]
                metric.update(field_preds, field_target, dataset_idx)

    def compute(self) -> Dict[str, Tensor]:
        result: Dict[str, Tensor] = {}
        for field_name in self.field_names:
            for field_metric in self.field_metrics[field_name]:
                result[f'{field_name}_{field_metric.name}'] = field_metric.compute()
        return result # noqa

    def get_dict(self, prefix: str = '') -> Dict[str, Tensor]:

        result: Dict[str, Tensor] = {}
        for field_name in self.field_names:
            for field_metric in self.field_metrics[field_name]:
                result.update(field_metric.get_dict(prefix=f'{prefix}/{field_name}'))
        return result

    def save(self, savedir, prefix: str = '') -> None:

        if not os.path.exists(savedir):
            os.makedirs(savedir)

        for field_name in self.field_names:
            for metric in self.field_metrics[field_name]:
                metric.save(savedir, prefix=f'{prefix}/{field_name}')

    def persistent(self, mode: bool = False) -> None:
        """Change if metric state is persistent (save as part of state_dict) or not.

        Args:
            mode: bool indicating if all states should be persistent or not

        """
        for field_metric in self.field_metrics:
            field_metric.persistent(mode=mode)

    def __repr__(self) -> str:
        """Return a representation of the compositional metric, including the two inputs it was formed from."""
        _op_metrics = f"(\n  {',\n  '.join([repr(metric) for metric in self.field_metrics])}\n)"
        return f"{self.__class__.__name__}(\n  num_fields={self.num_fields},\n  field_names={self.field_names},\n  field_metrics={_op_metrics}\n)"

    def _wrap_compute(self, compute: Callable) -> Callable:
        """No wrapping necessary for compositional metrics."""
        return compute

import torch.jit

@torch.jit.unused
def forward(self, *args: Any, **kwargs: Any) -> Any:
    """Calculate metric on current batch and accumulate to global state."""
    val_list = []
    for field_metric in self.field_metrics:
        val = field_metric(*args, **kwargs)
        val_list.append(val)

    self._forward_cache = val_list
    return self._forward_cache