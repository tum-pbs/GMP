from ..dataclasses import DataSample, LoadingMetadata, PhysicalMetadata, asdict_shallow

import h5py
import multiprocessing as mp
import os
import torch
import numpy as np


class CrashSteady(torch.utils.data.Dataset):
    def __init__(
        self,
        file_path: str,
        sim_indexes: list[int] | None = None,
        fields: list[str] | None = None,
        min: list[int] | None = None,
        max: list[int] | None = None,
        time_steps: int = 1,
        target_step: int = -1,
        max_num_points: int = 32768,
        min_domain: list[float] | None = None,
        max_domain: list[float] | None = None,
        globals: list[str] | None = None,
        global_parameter_min: list[float] | None = None,
        global_parameter_max: list[float] | None = None,
        **kwargs,
    ):
        """Dataset class for unstructured (pointwise) simulation data stored in various file formats.

        Args:
            file_path (str): Path to the data file
            sim_indexes (list[int] | None, optional): List of simulation indexes to load. Defaults to None.
            min (list[int] | None, optional): Minimum value for normalization. Defaults to None.
            max (list[int] | None, optional): Maximum value for normalization. Defaults to None.

        Raises:
            ValueError:
        """

        self.value_subtract = torch.tensor([0.5 * (min[i] + max[i]) for i in range(len(min))], dtype=torch.float32)  # (C)
        self.value_divide = torch.tensor([0.5 * (max[i] - min[i]) for i in range(len(min))], dtype=torch.float32)  # (C)

        self.min_domain = torch.tensor(min_domain, dtype=torch.float32)
        self.max_domain = torch.tensor(max_domain, dtype=torch.float32)

        self.globals = globals
        self.global_parameter_min = global_parameter_min
        self.global_parameter_max = global_parameter_max

        self.sim_indexes = sim_indexes
        self.fields = fields

        if time_steps != 1:
            raise ValueError(f"CrashSteady predicts one steady target step; time_steps must be 1, but got {time_steps}.")
        self.time_steps = time_steps
        self.target_step = target_step
        self.max_num_points = max_num_points
        self.epoch = mp.Value("i", 0, lock=False)

        self.data_path = []
        for idx in self.sim_indexes:
            if not os.path.exists(file_path.format(i=idx)):
                raise ValueError(f"File {file_path.format(i=idx)} does not exist.")
            self.data_path.append(file_path.format(i=idx))

    def __len__(self) -> int:
        return len(self.data_path)

    def __getitem__(self, idx) -> dict:
        with h5py.File(self.data_path[idx], "r") as f:
            N = f["coordinates"].shape[0]
            num_points = torch.tensor(N, dtype=torch.long)

            coords = np.asarray(f["coordinates"][:], dtype=np.float32)  # (N, 3)
            coords = torch.from_numpy(coords).unsqueeze(1)  # (N, 1, 3)

            values = torch.zeros(N, 1, self.value_subtract.shape[0], dtype=torch.float32)
            last_idx = 0
            for field in self.fields:
                field_data = f[field]
                target = field_data[self.target_step]
                if field_data.ndim == 3 and field_data.shape[-1] == 3:
                    target = np.asarray(target, dtype=np.float32)  # (N, 3)
                    values[:, 0, last_idx : last_idx + 3] = torch.from_numpy(target)
                    last_idx += 3
                else:
                    target = np.asarray(target, dtype=np.float32)  # (N)
                    values[:, 0, last_idx] = torch.from_numpy(target)
                    last_idx += 1

            global_params = torch.zeros(len(self.globals), dtype=torch.float32)
            for i, global_param in enumerate(self.globals):
                global_params[i] = f.attrs[global_param][()] - self.global_parameter_min[i]
                global_params[i] /= self.global_parameter_max[i] - self.global_parameter_min[i]

        if not getattr(self, "disable_subsampling", False) and self.max_num_points is not None and num_points > self.max_num_points:
            epoch = self.epoch.value if hasattr(self.epoch, "value") else self.epoch
            rng = np.random.default_rng(epoch * 10000 + idx)
            total_points = coords.shape[0]
            perm = rng.choice(total_points, size=self.max_num_points, replace=False)
            perm.sort()
            coords = torch.from_numpy(coords.numpy()[perm])
            values = torch.from_numpy(values.numpy()[perm])
            num_points = torch.tensor(self.max_num_points)

        physical_metadata = PhysicalMetadata(
            min_domain=self.min_domain,
            max_domain=self.max_domain,
            hparams=torch.zeros(0),
            dt=torch.tensor(1.0),
            num_hparams=torch.tensor(0, dtype=torch.long),
            num_attrs=torch.tensor(0, dtype=torch.long),
        )

        loading_metadata = LoadingMetadata(
            name=f"sim{idx}",
            values_name=self.fields,
            attrs_name=[],
            hparams_name=[],
            grid_dims=tuple(),
            channel_last=True,
            val_subtract=self.value_subtract,
            val_divide=self.value_divide,
            attr_subtract=torch.zeros(0, dtype=torch.float32),  # (A)
            attr_divide=torch.ones(0, dtype=torch.float32),  # (A)
            num_surface_points=num_points.detach().clone(),
            num_surface_value_channels=torch.tensor([values.shape[-1]], dtype=torch.long),  # (T, C)
            num_surface_attr_channels=torch.zeros(1, dtype=torch.long),  # (T, A)
        )

        return asdict_shallow(
            DataSample(
                positions=coords,
                values=values,
                attributes=torch.zeros(values.shape[0], 1, 0, dtype=torch.float32),
                num_points=num_points,
                global_params=global_params,
                physical_metadata=physical_metadata,
                loading_metadata=loading_metadata,
            )
        )

    def set_epoch(self, epoch):
        if hasattr(self.epoch, "value"):
            self.epoch.value = int(epoch)
        else:
            self.epoch = int(epoch)


