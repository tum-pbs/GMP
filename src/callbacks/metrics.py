
from typing import Any, Dict, List

from lightning import Callback, Trainer, LightningModule
from lightning.fabric.utilities import rank_zero_only
import torch
from torchmetrics import Metric
import os

import logging
log = logging.getLogger(__name__)

class MetricsUnstructured(Callback):
    def __init__(self, frequency: int,
                 batch_size_inference: int,
                 rollout_num: int = 20,
                 log_first_step: bool = False,
                 test_only: bool = True,
                 disabled: bool = False,
                 chunk_size: int = 65536,
                 metrics: List[Metric] = None,
                 **kwargs):
        """

        :param frequency:
        :param batch_size_inference:
        :param rollout_num:
        :param log_first_step:
        :param test_only:
        :param chunk_size:
        :param disabled:
        :param reference_boundary:
        :param metrics:
        """

        super().__init__()

        self.rollout_num = rollout_num
        self.frequency = frequency
        self.batch_size_inference = batch_size_inference
        self.disabled = disabled
        self.test_only = test_only
        self.chunk_size = chunk_size
        self.log_first_step = log_first_step
        self.metrics = metrics
        self.model_kwargs = kwargs
        self.temp_directory = "/tmp"

    def update_metrics(self, trainer, pl_module, dataloader, stage='test'):

        self.metrics = [metric.to(pl_module.device) for metric in self.metrics]

        generator = torch.Generator(device=pl_module.device).manual_seed(trainer.global_rank)

        for batch in dataloader:

            # TODO: Fix dataset idx for unstructured data
            # dataset_idx = batch["loading_metadata"]["dataset_idx"].flatten().to(pl_module.device)
            dataset_idx = torch.zeros(self.batch_size_inference, dtype=torch.long, device=pl_module.device)

            prediction, target = pl_module.predict(batch,
                                                 rollout_num=self.rollout_num,
                                                 generator=generator,
                                                 **self.model_kwargs)

            for metric in self.metrics:

                metric.update(prediction, target, dataset_idx=dataset_idx)

        metric_dict = {"epoch": trainer.current_epoch}

        for metric in self.metrics:

            metric.compute()

            if trainer.global_rank == 0:

                metric_dict.update(metric.get_dict(prefix=f'{stage}'))

                savedir = os.path.join(trainer.default_root_dir, 'metrics')
                metric.save(savedir, prefix=stage)

        self.log_metrics(trainer, metric_dict)

    @rank_zero_only
    def log_metrics(self, trainer: Trainer, metric_dict: Dict[str, Any]):
        trainer.logger.experiment.log(metric_dict)

    def check_frequency(self, check_idx):
        # check_idx + 1 since current_idx starts with 0
        return ((check_idx + 1) % self.frequency) == 0 or (check_idx == 0 and self.log_first_step)

    def on_test_start(self, trainer: Trainer, pl_module: LightningModule) -> None:

        if not self.disabled:

            with torch.no_grad():

                dataloader = trainer.test_dataloaders
                module_stage = getattr(pl_module, 'stage', "last_ema")
                self.update_metrics(trainer, pl_module, dataloader, stage=f'{module_stage}/test')

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule):

        if ((not self.disabled) and (not self.test_only) and
                self.check_frequency(pl_module.current_epoch)):

            if (dataloader := trainer.test_dataloaders) is None:
                trainer.test_loop.setup_data()
                dataloader = trainer.test_dataloaders

            with torch.no_grad():
                self.update_metrics(trainer, pl_module, dataloader, stage=f'fit/epoch_{trainer.current_epoch}')