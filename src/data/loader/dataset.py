from typing import Type
from dataclasses import dataclass
from abc import ABC, abstractmethod

import h5py
import torch
import os

import src.data.loader.logutils as log

from .dataclasses import DataSample, LoadingMetadata, PhysicalMetadata, asdict_shallow


@dataclass
class FileLoaderConfig:
    """Configuration for file loaders used in UPDEDataset and MultiUPDEDataset.

    Args:
        sim_selection (dict | None): Dictionary specifying simulation indices to use ({"range": [start, end]} or {"list": [i0, i1, ...]}).
        channel_last (bool): Whether the channel dimension should be last in the returned samples.
        calc_norm_data (bool): Whether to automatically calculate and store normalization data.
        time_steps (int): Number of time steps to include in a sample.
        trim_start (int): Number of initial frames to trim from each simulation.
        trim_end (int): Number of final frames to trim from each simulation.
        time_stride (int): Stride in time steps between different samples.
        intermediate_frames (bool): Whether to include intermediate frames between start and end frame of a sample.
    """

    sim_selection: dict | None = None
    channel_last: bool = False
    calc_norm_data: bool = False
    time_steps: int = 1
    trim_start: int = 0
    trim_end: int = 0
    time_stride: int = 1
    intermediate_frames: bool = False

class UPDEDataset(torch.utils.data.Dataset, ABC):
    def __init__(self, file_path: str, file_loader_config: FileLoaderConfig):
        """Dataset class for unstructured (pointwise) simulation data stored in various file formats.

        Args:
            file_path (str): Path to the data file
            file_loader_config (FileLoaderConfig): Configuration for the file loader

        Raises:
            ValueError:
        """
        if not os.path.exists(file_path):
            raise ValueError(f"File not found: {file_path}")

        self.file_path = file_path
        self.file_loader_config = file_loader_config

        if file_loader_config.sim_selection is None:
            self.sim_selection = None
        else:
            if "range" in file_loader_config.sim_selection.keys():
                start, end = file_loader_config.sim_selection["range"]
                self.sim_selection = (start, end)
            elif "list" in file_loader_config.sim_selection.keys():
                sim_list = file_loader_config.sim_selection["list"]
                self.sim_selection = list(sim_list)
            else:
                raise ValueError("sim_selection dict must contain either 'range' or 'list' key.")

        self.num_sims, self.samples_per_sim = self.initialize(self.file_path, self.file_loader_config)
        
        if self.sim_selection is not None and any(idx >= self.num_sims or idx < 0 for idx in self.sim_selection):
            raise ValueError("sim_selection contains invalid simulation indices.")
        
        if not self.check_norm_data(recalculate=self.file_loader_config.calc_norm_data):
            log.warn(f"Missing or invalid normalization data or encountered an error during its recalculation.")
            log.warn(f"Consider setting `calc_norm_data=True` to calculate and store normalization data if supported by the used file loader.")

        log.info(f"Loaded UPDEDataset from file: {self.file_path} | Sim selection: {self.sim_selection if self.sim_selection is not None else 'All simulations'} | Length: {len(self)}")

    def __len__(self) -> int:
        if self.sim_selection is None:
            self.size = self.num_sims * self.samples_per_sim
        elif isinstance(self.sim_selection, tuple):
            self.size = (self.sim_selection[1] - self.sim_selection[0]) * self.samples_per_sim
        elif isinstance(self.sim_selection, list):
            self.size = len(self.sim_selection) * self.samples_per_sim
        else:
            raise ValueError(f"Invalid sim_selection type. Actual type: {type(self.sim_selection)}")

        return self.size

    def __getitem__(self, idx) -> DataSample:
        sim_idx = idx // self.samples_per_sim
        sample_idx = idx % self.samples_per_sim

        if isinstance(self.sim_selection, tuple):
            sim_idx += self.sim_selection[0]
        elif isinstance(self.sim_selection, list):
            sim_idx = self.sim_selection[sim_idx]

        if sim_idx < 0 or sim_idx >= self.num_sims:
            raise IndexError("Simulation index out of range.")

        return self.load_sample(sim_idx, sample_idx)

    def info_str(self) -> str:
        info_lines = ["UPDEDataset Info:"]
        info_lines.append(f"    File: {self.file_path}")
        info_lines.append(f"    Number of simulations: {self.num_sims}")
        info_lines.append(f"    Samples per simulation: {self.samples_per_sim}")
        info_lines.append(f"    Simulation selection: {self.sim_selection if self.sim_selection is not None else 'All simulations'}")
        info_lines.append(f"    Total number of samples: {len(self)}")
        return "\n".join(info_lines)
        
    @abstractmethod
    def initialize(self, file_path, config: FileLoaderConfig) -> tuple[int, int]:
        """Method called for initialization of the file loading

        Args:
            file_path (unknown): path to the file to be loaded
            config (FileLoaderConfig): Configuration for loading the file

        Returns:
            tuple[int, int]: number of simulations and number of samples per simulation provided by the file

        Throws:
            ArgumentError: If entries of `config` are invalid for the given file
        """
        pass
    
    @abstractmethod
    def check_norm_data(self, recalculate: bool) -> bool:
        """Method to check for and, if enabled, calculate and store normalization data on the dataset

        Args:
            recalculate (bool): Whether to recalculate normalization data and store it in the file (overwriting existing data)

        Returns:
            bool: True if normalization data is available after the call, False otherwise
        """
        pass

    @abstractmethod
    def load_sample(self, sim_idx, sample_idx) -> DataSample:
        """Method to load a single smaple from the file

        Args:
            sim_idx (int): Index of the simulation to load from
            sample_idx (int): Index of the sample to load from the simulation

        Returns:
            DataSample: Loaded data sample
        """
        pass
    
    def close(self) -> None:
        """Method to close the dataset and release resources."""
        pass


class MultiUPDEDataset(torch.utils.data.Dataset):
    def __init__(self, datasets: UPDEDataset | list[UPDEDataset], index_start: int = -1, index_end: int =-1):
        
        self.datasets = datasets if isinstance(datasets, list) else list(datasets)
            
        if index_start >= 0 and index_end > index_start:
            file_paths = [fpath.format(i=i) for i in range(index_start, index_end + 1) for fpath in file_paths]
            file_loader_configs = file_loader_configs * (index_end - index_start + 1)
            
        expected_channel_last_setting = datasets[0].file_loader_config.channel_last
        for dataset in datasets:
            if dataset.file_loader_config.channel_last != expected_channel_last_setting:
                log.warn("Detected inconsistent `channel_last` settings across file loader configurations. "
                         "This may lead to unexpected behavior (like large paddings) when loading data in batches "
                         "and prohibits slicing of DataSamples.")

        cumulative_sizes = [0]
        for dataset in self.datasets:
            cumulative_sizes.append(cumulative_sizes[-1] + len(dataset))
        self.cumulative_sizes = cumulative_sizes[1:]

    def __len__(self) -> int:
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx) -> DataSample:
        dataset_idx = 0
        while idx >= self.cumulative_sizes[dataset_idx]:
            dataset_idx += 1

        if dataset_idx > 0:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        else:
            sample_idx = idx

        return self.datasets[dataset_idx][sample_idx]

    def info_str(self) -> str:
        info_lines = ["MultiUPDEDataset Info:"]
        for i, dataset in enumerate(self.datasets):
            info_lines.append(f"  Dataset {i}:")
            dataset_info = dataset.info_str().split("\n")
            for line in dataset_info:
                info_lines.append(f"    {line}")
        return "\n".join(info_lines)

    def close(self) -> None:
        """Method to close all datasets and release resources."""
        for dataset in self.datasets:
            dataset.close()

    def set_epoch(self, epoch: int):
        """Method to set epoch on all sub-datasets."""
        for dataset in self.datasets:
            if hasattr(dataset, "set_epoch"):
                dataset.set_epoch(epoch)

