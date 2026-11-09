from ..dataclasses import DataSample, LoadingMetadata, PhysicalMetadata, asdict_shallow

import h5py
import multiprocessing as mp
import os
import torch
import numpy as np

class KSDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        file_path: str,
        sim_indexes: list[int] | None = None,
        mean: float = 0.0,
        std: float = 1.0,
        time_steps: int = 1,
        start: int = 0,
        max_num_points: int = 1024,
        min_domain: list[float] | None = None,
        max_domain: list[float] | None = None,
        global_parameter_min: list[float] | None = None,
        global_parameter_max: list[float] | None = None,
        **kwargs,
    ):
        """Dataset class for unstructured (pointwise) simulation data stored in various file formats.

        Args:
            file_path (str): Path to the data file
            sim_indexes (list[int] | None, optional): List of simulation indexes to load. Defaults to None.
            mean (float, optional): Mean value for normalization. Defaults to 0.0.
            std (float, optional): Standard deviation for normalization. Defaults to 1.0.

        Raises:
            ValueError:
        """

        self.value_subtract = [mean]
        self.value_divide = [std]

        self.min_domain = torch.tensor(min_domain)
        self.max_domain = torch.tensor(max_domain)

        self.global_parameter_min = global_parameter_min
        self.global_parameter_max = global_parameter_max
        
        self.sim_indexes = sim_indexes

        self.time_steps = time_steps
        self.start = start
        self.max_num_points = max_num_points
        self.epoch = mp.Value("i", 0, lock=False)

        f = h5py.File(file_path, "r")
        f = f["sims"]

        if self.sim_indexes is None:
            self.sim_indexes = list(range(len(f)))
        sim_keys = [f"sim{idx + self.start}" for idx in self.sim_indexes]

        self.data = []

        for sim_key in sim_keys:
            N = f[sim_key]["N"][()]
            num_points = torch.tensor(N, dtype=torch.long)
            L = f[sim_key]["Lx"][()]
            positions = np.array(f[sim_key]["coordinates"][()].transpose(2, 0, 1), dtype=np.float32)
            positions = torch.from_numpy(positions)/L  # (N, T, 2)
            values = np.array(f[sim_key]["values"][()].squeeze(), dtype=np.float32)
            values = torch.from_numpy(values)
            if self.time_steps >= values.shape[0]:
                raise ValueError(f"time_steps ({self.time_steps}) must be less than the number of time steps in the data ({values.shape[0]}).")
            for i in range(0,values.shape[0]-self.time_steps, self.time_steps):
                values_t = values[i:i+self.time_steps+1].permute(1,0).unsqueeze(-1)  # (N, T, C)
                positions_t = positions[:, i:i+self.time_steps+1]  # (N, T, D)
                self.data.append({"positions": positions_t, "values": values_t, "num_points": num_points, "L": L, "key": sim_key})
                
        print(f"Total number of samples: {len(self.data)}")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx) -> dict:
        
        coords = self.data[idx]["positions"]
        values = self.data[idx]["values"]
        num_points = self.data[idx]["num_points"]
        L = torch.tensor(self.data[idx]["L"], dtype=torch.float32)

        if self.global_parameter_max is not None:
            L = (L - torch.tensor(self.global_parameter_min)) / (torch.tensor(self.global_parameter_max) - torch.tensor(self.global_parameter_min))

        physical_metadata = PhysicalMetadata(
            min_domain=self.min_domain,
            max_domain=self.max_domain,
            hparams=torch.zeros(0),
            dt=torch.tensor(1.0),
            num_hparams=torch.tensor(0, dtype=torch.long),
            num_attrs=torch.tensor(1, dtype=torch.long),
        )

        loading_metadata = LoadingMetadata(
            name=f"sim{idx}",
            values_name=["values"],
            attrs_name=[],
            hparams_name=[],
            grid_dims=tuple(),
            channel_last=True,
            val_subtract=torch.tensor(self.value_subtract),  # (T, C)
            val_divide=torch.tensor(self.value_divide),  # (T, C)
            attr_subtract=torch.zeros(1),  # (T, A)
            attr_divide=torch.ones(1),  # (T, A)
            num_surface_points=torch.tensor(num_points, dtype=torch.long),
            num_surface_value_channels=torch.zeros(1, dtype=torch.long),  # (T, C)
            num_surface_attr_channels=torch.zeros(1, dtype=torch.long),  # (T, A)
        )

        return asdict_shallow(
            DataSample(
                positions=coords,
                values=values,
                attributes=torch.zeros(0),
                num_points=num_points,
                global_params=L,
                physical_metadata=physical_metadata,
                loading_metadata=loading_metadata,
            )
        )
    
    def set_epoch(self, epoch):
        if hasattr(self.epoch, "value"):
            self.epoch.value = int(epoch)
        else:
            self.epoch = int(epoch)