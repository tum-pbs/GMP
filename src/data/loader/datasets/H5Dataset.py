import math
import torch
import h5py
import numpy as np

import src.data.loader.logutils as log

from ..dataset import UPDEDataset, FileLoaderConfig
from ..dataclasses import DataSample, PhysicalMetadata, LoadingMetadata


################################################################################
### H5FileLoader ###############################################################
################################################################################

# Constants for expected HDF5 file structure


# Dataset Data Keys
KEY_SIM_GROUP = 'sims'

# Dataset Metadata Keys
KEY_DSET_NAME = 'dataset_name'
KEY_VALUES_NAME = 'values_name'
KEY_ATTRS_NAME = 'attrs_name'
KEY_HPARAMS_NAME = 'hparams_name'

# Simulation Data Keys
KEY_POSITIONS_DATA = 'positions'
KEY_VALUES_DATA = 'values'
KEY_ATTRIBUTES_DATA = 'attributes'
KEY_NUM_POINTS = 'N'

# Simulation Metadata Keys
KEY_DOMAIN_SIZE_X = 'Lx'
KEY_DOMAIN_SIZE_Y = 'Ly'
KEY_HPARAMS = 'hparams'
KEY_DT = 'dt'

# Normalization Keys
KEY_NORM_MEAN_POS = 'norm_mean_pos'
KEY_NORM_STD_POS = 'norm_std_pos'
KEY_NORM_MIN_POS = 'norm_min_pos'
KEY_NORM_MAX_POS = 'norm_max_pos'
KEY_NORM_MEAN_VAL = 'norm_mean_val'
KEY_NORM_STD_VAL = 'norm_std_val'
KEY_NORM_MIN_VAL = 'norm_min_val'
KEY_NORM_MAX_VAL = 'norm_max_val'


class H5Dataset(UPDEDataset):
    """UPDEDataset subclass for loading unstructured simulation data from H5/HDF5 files.
    """

    def initialize(self, file_path, config: FileLoaderConfig) -> tuple[int, int]:
        self.file_path = file_path
        self.file = h5py.File(file_path, 'r')
        self.config = config

        # Validate file structure
        if KEY_SIM_GROUP not in self.file or not isinstance(self.file[KEY_SIM_GROUP], h5py.Group):
            raise ValueError(f"HDF5 file must contain a '{KEY_SIM_GROUP}' group.")

        sim_group = self.file[KEY_SIM_GROUP]

        self.positions_shape = None
        self.values_shape = None
        self.attributes_shape = None
        self.hparams_shape = None

        for sim_name in sim_group:
            assert sim_name.startswith(
                'sim'), f"Simulation groups must be named 'sim{{index}}'. Found invalid group name: {sim_name}"
            assert sim_name[3:].isdigit(
            ), f"Simulation groups must be named 'sim{{index}}'. Found invalid group name: {sim_name}"

            # Mandatory Data checks (positions, values, num_points)
            sim_data = sim_group[sim_name]
            if sim_data is None or not isinstance(sim_data, h5py.Group):
                raise ValueError(f"Simulation {sim_name} must be its own group within the '{KEY_SIM_GROUP}' group.")
            if KEY_POSITIONS_DATA not in sim_data.keys():
                raise ValueError(f"Simulation {sim_name} must contain data with key '{KEY_POSITIONS_DATA}.")
            if KEY_VALUES_DATA not in sim_data.keys():
                raise ValueError(f"Simulation {sim_name} must contain data with key '{KEY_VALUES_DATA}.")
            if KEY_NUM_POINTS not in sim_data.keys():
                raise ValueError(f"Simulation {sim_name} must contain data with key '{KEY_NUM_POINTS}.")

            if not self.positions_shape:
                self.positions_shape = sim_data[KEY_POSITIONS_DATA].shape
            elif self.positions_shape != sim_data[KEY_POSITIONS_DATA].shape:
                raise ValueError(
                    f"All simulations must have the same shape for '{KEY_POSITIONS_DATA}'. Mismatch found in {sim_name}.")
            if not self.values_shape:
                self.values_shape = sim_data[KEY_VALUES_DATA].shape
            elif self.values_shape != sim_data[KEY_VALUES_DATA].shape:
                raise ValueError(
                    f"All simulations must have the same shape for '{KEY_VALUES_DATA}'. Mismatch found in {sim_name}.")
            if sim_data[KEY_NUM_POINTS].shape != ():
                raise ValueError(f"'{KEY_NUM_POINTS}' in simulation {sim_name} must be a scalar value.")

            # Check for optional Attributes Data (else tensor with 0 channels)
            if KEY_ATTRIBUTES_DATA not in sim_data.keys():
                log.debug(
                    f"Simulation {sim_name} does not contain data with key '{KEY_ATTRIBUTES_DATA}'. Defaulting to shape of `position` with 0 channels.")
                expected_attributes_shape = (self.positions_shape[0], 0) + self.positions_shape[2:]
            else:
                expected_attributes_shape = sim_data[KEY_ATTRIBUTES_DATA].shape

            if not self.attributes_shape:
                self.attributes_shape = expected_attributes_shape
            elif self.attributes_shape != expected_attributes_shape:
                raise ValueError(
                    f"All simulations must have the same shape for '{KEY_ATTRIBUTES_DATA}'. Mismatch found in {sim_name}.")

            # Expect data in file to be stored as (T, D, N1, ..., Nk) for positions, (T, C, N1, ..., Nk) for values, and (T, A, N1, ..., Nk) for attributes
            if self.positions_shape[0] != self.values_shape[0] or self.positions_shape[0] != self.attributes_shape[0]:
                raise ValueError(
                    f"Time dimension mismatch between '{KEY_POSITIONS_DATA}' and '{KEY_VALUES_DATA}' in simulation {sim_name}.")
            if self.positions_shape[2:] != self.values_shape[2:] or self.positions_shape[2:] != self.attributes_shape[2:]:
                raise ValueError(
                    f"Point dimension mismatch between '{KEY_POSITIONS_DATA}' and '{KEY_VALUES_DATA}' in simulation {sim_name}.")

            # Check Simulation-Level Metadata
            if KEY_DOMAIN_SIZE_X not in sim_data.attrs or KEY_DOMAIN_SIZE_Y not in sim_data.attrs:
                log.debug(
                    f"Simulation {sim_name} does not contain complete domain size attributes '{KEY_DOMAIN_SIZE_X}' and '{KEY_DOMAIN_SIZE_Y}'. Defaulting to unit domain size.")
            else:
                domain_size_x = sim_data.attrs[KEY_DOMAIN_SIZE_X]
                domain_size_y = sim_data.attrs[KEY_DOMAIN_SIZE_Y]
                if not (isinstance(domain_size_x, (int, float)) and isinstance(domain_size_y, (int, float))):
                    raise ValueError(
                        f"Domain size attributes '{KEY_DOMAIN_SIZE_X}' and '{KEY_DOMAIN_SIZE_Y}' in simulation {sim_name} must be numeric values.")

            if KEY_HPARAMS not in sim_data.attrs:
                log.debug(
                    f"Simulation {sim_name} does not contain hyperparameters attribute '{KEY_HPARAMS}'. Defaulting to zero hyperparameters.")
                curr_hparams_shape = (0,)
            else:
                curr_hparams_shape = sim_data.attrs[KEY_HPARAMS].shape

            if self.hparams_shape is None:
                self.hparams_shape = curr_hparams_shape
            elif self.hparams_shape != curr_hparams_shape:
                raise ValueError(
                    f"All simulations must have the same shape for hyperparameters attribute '{KEY_HPARAMS}'. Mismatch found in {sim_name}.")

            if KEY_DT not in sim_data.attrs:
                log.debug(
                    f"Simulation {sim_name} does not contain time step attribute '{KEY_DT}'. Defaulting to dt=1.0.")
            else:
                dt_value = sim_data.attrs[KEY_DT]
                if not isinstance(dt_value, (int, float)):
                    raise ValueError(
                        f"Time step attribute '{KEY_DT}' in simulation {sim_name} must be a numeric value.")

        # Extract dataset-level metadata
        if not KEY_DSET_NAME in self.file.attrs:
            log.debug(f"HDF5 file does not contain dataset-level attribute '{KEY_DSET_NAME}'. Defaulting to filename.")
            self.dset_name = self.file_path.split('/')[-1]
        else:
            self.dset_name = self.file.attrs[KEY_DSET_NAME]

        if not KEY_VALUES_NAME in self.file.attrs:
            log.debug(
                f"HDF5 file does not contain dataset-level attribute '{KEY_VALUES_NAME}'. Defaulting to generic value names.")
            self.values_name = [f"value_{i}" for i in range(self.values_shape[1])]
        else:
            self.values_name = list(self.file.attrs[KEY_VALUES_NAME])

        if not KEY_ATTRS_NAME in self.file.attrs:
            log.debug(
                f"HDF5 file does not contain dataset-level attribute '{KEY_ATTRS_NAME}'. Defaulting to generic attribute names.")
            self.attrs_name = [f"attr_{i}" for i in range(self.attributes_shape[1])]
        else:
            self.attrs_name = list(self.file.attrs[KEY_ATTRS_NAME])

        if not KEY_HPARAMS_NAME in self.file.attrs:
            log.debug(
                f"HDF5 file does not contain dataset-level attribute '{KEY_HPARAMS_NAME}'. Defaulting to generic hyperparameter names.")
            self.hparams_name = [f"hparam_{i}" for i in range(self.hparams_shape[0])]
        else:
            self.hparams_name = list(self.file.attrs[KEY_HPARAMS_NAME])

        self.num_sims = len(sim_group)
        self.samples_per_sim = max(math.ceil(
            (self.positions_shape[0] - config.trim_start - config.trim_end - config.time_steps) / config.time_stride
        ), 0)

        return self.num_sims, self.samples_per_sim

    def check_norm_data(self, recalculate: bool) -> bool:
        if not recalculate:
            if KEY_NORM_MEAN_POS in self.file.keys() and KEY_NORM_STD_POS in self.file.keys() and \
                    KEY_NORM_MIN_POS in self.file.keys() and KEY_NORM_MAX_POS in self.file.keys() and \
                    KEY_NORM_MEAN_VAL in self.file.keys() and KEY_NORM_STD_VAL in self.file.keys() and \
                    KEY_NORM_MIN_VAL in self.file.keys() and KEY_NORM_MAX_VAL in self.file.keys():
                log.info("Normalization data found in HDF5 file.")
                self.norm_mean_pos = torch.tensor(self.file[KEY_NORM_MEAN_POS][()])
                self.norm_std_pos = torch.tensor(self.file[KEY_NORM_STD_POS][()])
                self.norm_min_pos = torch.tensor(self.file[KEY_NORM_MIN_POS][()])
                self.norm_max_pos = torch.tensor(self.file[KEY_NORM_MAX_POS][()])
                self.norm_mean_val = torch.tensor(self.file[KEY_NORM_MEAN_VAL][()])
                self.norm_std_val = torch.tensor(self.file[KEY_NORM_STD_VAL][()])
                self.norm_min_val = torch.tensor(self.file[KEY_NORM_MIN_VAL][()])
                self.norm_max_val = torch.tensor(self.file[KEY_NORM_MAX_VAL][()])
                return True
            else:
                log.warn("Normalization data not found in HDF5 file.")
                self.norm_mean_pos = None
                self.norm_std_pos = None
                self.norm_min_pos = None
                self.norm_max_pos = None
                self.norm_mean_val = None
                self.norm_std_val = None
                self.norm_min_val = None
                self.norm_max_val = None
                return False

        try:
            log.info("Calculating normalization data from dataset...")

            # Reopen file in read-write mode to store normalization data
            self.file.close()
            self.file = h5py.File(self.file_path, 'r+')

            # The shapes of the reduced tensors (reduce over point dimensions)
            pos_norm_shape = [self.positions_shape[0], self.positions_shape[1]] + [1] * (len(self.positions_shape) - 2)
            pos_norm_shape = tuple(pos_norm_shape)
            val_norm_shape = [self.values_shape[0], self.values_shape[1]] + [1] * (len(self.values_shape) - 2)
            val_norm_shape = tuple(val_norm_shape)

            # The dimensions to normalize over (point dimensions)
            norm_dims = [i for i in range(len(self.positions_shape)) if i > 1]
            norm_dims = tuple(norm_dims)

            self.norm_mean_pos = torch.zeros(pos_norm_shape)
            self.norm_std_pos = torch.zeros(pos_norm_shape)
            self.norm_min_pos = torch.zeros(pos_norm_shape)
            self.norm_max_pos = torch.zeros(pos_norm_shape)
            self.norm_mean_val = torch.zeros(val_norm_shape)
            self.norm_std_val = torch.zeros(val_norm_shape)
            self.norm_min_val = torch.zeros(val_norm_shape)
            self.norm_max_val = torch.zeros(val_norm_shape)

            # Iterate over simulations to compute normalization data
            # NOTE: NaN values are ignored in the calculations
            # NOTE: This assumes that all simulations fit into memory when processed one by one
            for sim_name in self.file[KEY_SIM_GROUP]:
                sim_data: h5py.Group = self.file[KEY_SIM_GROUP][sim_name]
                pos_data: np.ndarray = sim_data[KEY_POSITIONS_DATA][()]
                val_data: np.ndarray = sim_data[KEY_VALUES_DATA][()]

                self.norm_mean_pos += np.nanmean(pos_data, axis=norm_dims, keepdims=True)
                self.norm_std_pos += np.nanstd(pos_data, axis=norm_dims, keepdims=True)
                self.norm_min_pos = np.minimum(self.norm_min_pos, np.nanmin(pos_data, axis=norm_dims, keepdims=True))
                self.norm_max_pos = np.maximum(self.norm_max_pos, np.nanmax(pos_data, axis=norm_dims, keepdims=True))
                self.norm_mean_val += np.nanmean(val_data, axis=norm_dims, keepdims=True)
                self.norm_std_val += np.nanstd(val_data, axis=norm_dims, keepdims=True)
                self.norm_min_val = np.minimum(self.norm_min_val, np.nanmin(val_data, axis=norm_dims, keepdims=True))
                self.norm_max_val = np.maximum(self.norm_max_val, np.nanmax(val_data, axis=norm_dims, keepdims=True))
            num_sims = len(self.file[KEY_SIM_GROUP])
            self.norm_mean_pos /= num_sims
            self.norm_std_pos /= num_sims
            self.norm_min_pos = self.norm_min_pos
            self.norm_max_pos = self.norm_max_pos
            self.norm_mean_val /= num_sims
            self.norm_std_val /= num_sims
            self.norm_min_val = self.norm_min_val
            self.norm_max_val = self.norm_max_val

            # Squeeze out remaining point dimensions (keep time/channel dimensions of size 1)
            squeeze_dims = tuple(i for i in range(2, len(pos_norm_shape)))
            self.norm_mean_pos = self.norm_mean_pos.squeeze(dim=squeeze_dims)
            self.norm_std_pos = self.norm_std_pos.squeeze(dim=squeeze_dims)
            self.norm_min_pos = self.norm_min_pos.squeeze(dim=squeeze_dims)
            self.norm_max_pos = self.norm_max_pos.squeeze(dim=squeeze_dims)
            self.norm_mean_val = self.norm_mean_val.squeeze(dim=squeeze_dims)
            self.norm_std_val = self.norm_std_val.squeeze(dim=squeeze_dims)
            self.norm_min_val = self.norm_min_val.squeeze(dim=squeeze_dims)
            self.norm_max_val = self.norm_max_val.squeeze(dim=squeeze_dims)

            # Store normalization data in HDF5 file
            log.debug("Storing normalization data into HDF5 file...")
            for key in [KEY_NORM_MEAN_POS, KEY_NORM_STD_POS, KEY_NORM_MIN_POS, KEY_NORM_MAX_POS,
                        KEY_NORM_MEAN_VAL, KEY_NORM_STD_VAL, KEY_NORM_MIN_VAL, KEY_NORM_MAX_VAL]:
                try:
                    del self.file[key]
                except KeyError:
                    pass

            self.file[KEY_NORM_MEAN_POS] = self.norm_mean_pos
            self.file[KEY_NORM_STD_POS] = self.norm_std_pos
            self.file[KEY_NORM_MIN_POS] = self.norm_min_pos
            self.file[KEY_NORM_MAX_POS] = self.norm_max_pos
            self.file[KEY_NORM_MEAN_VAL] = self.norm_mean_val
            self.file[KEY_NORM_STD_VAL] = self.norm_std_val
            self.file[KEY_NORM_MIN_VAL] = self.norm_min_val
            self.file[KEY_NORM_MAX_VAL] = self.norm_max_val

            # Reopen file in read-only mode
            self.file.close()
            self.file = h5py.File(self.file_path, 'r')
            return True
        except Exception as e:
            log.error(f"Error calculating normalization data: {e}")
            return False

    def load_sample(self, sim_idx, sample_idx) -> DataSample:
        time_start = self.config.trim_start + sample_idx * self.config.time_stride
        time_end = time_start + self.config.time_steps

        if self.config.intermediate_frames:
            slice_indices = torch.arange(time_start, time_end+1, dtype=torch.long)
        else:
            slice_indices = torch.tensor([time_start, time_end], dtype=torch.long)

        sims_group = self.file[KEY_SIM_GROUP]
        sim_name = f"sim{sim_idx}"
        sim_data = sims_group[sim_name]

        positions = torch.tensor(sim_data[KEY_POSITIONS_DATA][slice_indices.tolist()]).flatten(start_dim=2)
        values = torch.tensor(sim_data[KEY_VALUES_DATA][slice_indices.tolist()]).flatten(start_dim=2)

        if KEY_ATTRIBUTES_DATA not in sim_data.keys():
            attributes_sample_shape = (positions.shape[0], 0) + positions.shape[2:]
            attributes = torch.zeros(attributes_sample_shape).flatten(start_dim=2)
        else:
            attributes = torch.tensor(sim_data[KEY_ATTRIBUTES_DATA][slice_indices.tolist()]).flatten(start_dim=2)

        if self.config.channel_last:
            positions = positions.permute(2, 0, 1)      # (N, T, D)
            values = values.permute(2, 0, 1)            # (N, T, C)
            attributes = attributes.permute(2, 0, 1)    # (N, T, A)

        num_points = torch.tensor(sim_data[KEY_NUM_POINTS][()])

        min_domain = torch.zeros((self.positions_shape[1]))
        if not KEY_DOMAIN_SIZE_X in sim_data.attrs or not KEY_DOMAIN_SIZE_Y in sim_data.attrs:
            max_domain = torch.ones((self.positions_shape[1]))
        else:
            max_domain = torch.tensor([sim_data.attrs[KEY_DOMAIN_SIZE_X],
                                      sim_data.attrs[KEY_DOMAIN_SIZE_Y]], dtype=torch.float32)

        if not KEY_HPARAMS in sim_data.attrs:
            hparams = torch.zeros(self.hparams_shape)
        else:
            hparams = torch.tensor(sim_data.attrs[KEY_HPARAMS])

        if not KEY_DT in sim_data.attrs:
            dt = torch.tensor(1.0)
        else:
            dt = torch.tensor(sim_data.attrs[KEY_DT])

        physical_metadata = PhysicalMetadata(
            min_domain=min_domain,
            max_domain=max_domain,
            hparams=hparams,
            dt=dt,
            num_hparams=torch.tensor(self.hparams_shape[0], dtype=torch.long),
            num_attrs=torch.tensor(self.attributes_shape[1], dtype=torch.long),
        )

        loading_metadata = LoadingMetadata(
            name=self.dset_name + "_sim_" + str(sim_idx),
            values_name=self.values_name,
            attrs_name=self.attrs_name,
            hparams_name=self.hparams_name,
            grid_dims=tuple(self.positions_shape[2:]),
            channel_last=self.config.channel_last,
            norm_mean_pos=self.norm_mean_pos,
            norm_std_pos=self.norm_std_pos,
            norm_min_pos=self.norm_min_pos,
            norm_max_pos=self.norm_max_pos,
            norm_mean_val=self.norm_mean_val,
            norm_std_val=self.norm_std_val,
            norm_min_val=self.norm_min_val,
            norm_max_val=self.norm_max_val,
        )

        return DataSample(
            positions=positions,
            values=values,
            attributes=attributes,
            num_points=num_points,
            physical_metadata=physical_metadata,
            loading_metadata=loading_metadata,
        )

    def finalize(self) -> None:
        self.file.close()
