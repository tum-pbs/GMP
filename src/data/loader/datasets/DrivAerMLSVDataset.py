import multiprocessing as mp
import os
from typing import List, Optional

import h5py
import numpy as np
import pandas as pd
import torch

from ..dataclasses import DataSample, LoadingMetadata, PhysicalMetadata, asdict_shallow


class DrivAerMLSVDataset(torch.utils.data.Dataset):
    """
    Merge surface + volume files per simulation index into a single point cloud sample.

    Behavior:
    - Samples up to `max_num_points` from the concatenation of volume+surface points.
      Sampling is done on the combined index space so returned points preserve the
      selection proportion between volume and surface. Volume points are placed first
      in the returned arrays, surface points follow.
    - Returned `values` and `attributes` use a canonical channel ordering:
        values_name = volume_value_keys + surface_value_keys
      For each point we produce a fixed-length channel vector where fields absent for
      the given point-source are filled with zeros. This keeps channel semantics stable.
    - If no value/attr keys are provided for both sources, returned tensors have
      channel dimension == 0 (shape (N, 1, 0)).
    """

    def __init__(
        self,
        use_volume: bool,
        use_surface: bool,
        volume_file_path: str,
        surface_file_path: str,
        global_parameter_path: str,
        global_parameter_names: List[str],
        global_parameter_min: List[float],
        global_parameter_max: List[float],
        sim_indexes: List[int],
        min_domain: List[float],
        max_domain: List[float],
        attrs_surface: Optional[List[dict]] = None,
        values_surface: Optional[List[dict]] = None,
        attrs_volume: Optional[List[dict]] = None,
        values_volume: Optional[List[dict]] = None,
        max_num_surface_points: int = 16384,
        max_num_volume_points: int = 16384,
        use_cache: bool = False,
        **kwargs,
    ):
        if not sim_indexes:
            raise ValueError("sim_indexes must be provided")
        self.sim_indexes = list(sim_indexes)

        # parse keys/channels preserving order
        self.s_attr_keys = [a["name"] for a in (attrs_surface or [])] if use_surface else []
        self.s_attr_channels = [a["channels"] for a in (attrs_surface or [])] if use_surface else []
        self.s_value_keys = [v["name"] for v in (values_surface or [])] if use_surface else []
        self.s_value_channels = [v["channels"] for v in (values_surface or [])] if use_surface else []

        self.v_attr_keys = [a["name"] for a in (attrs_volume or [])] if use_volume else []
        self.v_attr_channels = [a["channels"] for a in (attrs_volume or [])] if use_volume else []
        self.v_value_keys = [v["name"] for v in (values_volume or [])] if use_volume else []
        self.v_value_channels = [v["channels"] for v in (values_volume or [])] if use_volume else []

        # build per-channel normalization vectors in canonical order:
        # first volume value channels, then surface value channels
        self.val_subtract = []
        self.val_divide = []
        if use_volume:
            # process volume value specs first
            for v in values_volume or []:
                norm_mode = v.get("norm_mode", "meanstd")
                params = v.get("norm_params", [])
                for c in range(v["channels"]):
                    if norm_mode == "minmax":
                        self.val_subtract.append(params[2*c])
                        self.val_divide.append(params[2*c + 1] - params[2*c])
                    elif norm_mode == "meanstd":
                        self.val_subtract.append(params[2*c])
                        self.val_divide.append(params[2*c + 1])
                    else:
                        raise ValueError(f"Unsupported normalization mode: {norm_mode}")
        if use_surface:
            # then surface value specs
            for v in values_surface or []:
                norm_mode = v.get("norm_mode", "meanstd")
                params = v.get("norm_params", [])
                for c in range(v["channels"]):
                    if norm_mode == "minmax":
                        self.val_subtract.append(params[2*c])
                        self.val_divide.append(params[2*c + 1] - params[2*c])
                    elif norm_mode == "meanstd":
                        self.val_subtract.append(params[2*c])
                        self.val_divide.append(params[2*c + 1])
                    else:
                        raise ValueError(f"Unsupported normalization mode: {norm_mode}")

        # attributes: same canonical ordering (volume attrs first, then surface attrs)
        self.attr_subtract = []
        self.attr_divide = []
        if use_volume:
            for a in attrs_volume or []:
                norm_mode = a.get("norm_mode", "meanstd")
                params = a.get("norm_params", [])
                for c in range(a["channels"]):
                    if norm_mode == "minmax":
                        self.attr_subtract.append(params[2*c])
                        self.attr_divide.append(params[2*c + 1] - params[2*c])
                    elif norm_mode == "meanstd":
                        self.attr_subtract.append(params[2*c])
                        self.attr_divide.append(params[2*c + 1])
                    else:
                        raise ValueError(f"Unsupported normalization mode: {norm_mode}")
        if use_surface:
            for a in attrs_surface or []:
                norm_mode = a.get("norm_mode", "meanstd")
                params = a.get("norm_params", [])
                for c in range(a["channels"]):
                    if norm_mode == "minmax":
                        self.attr_subtract.append(params[2*c])
                        self.attr_divide.append(params[2*c + 1] - params[2*c])
                    elif norm_mode == "meanstd":
                        self.attr_subtract.append(params[2*c])
                        self.attr_divide.append(params[2*c + 1])
                    else:
                        raise ValueError(f"Unsupported normalization mode: {norm_mode}")

        self.min_domain = torch.tensor(min_domain)
        self.max_domain = torch.tensor(max_domain)
        self.epoch = mp.Value("i", 0, lock=False)

        # collect file paths (parallel lists)
        potential_vol = [volume_file_path.format(i=idx) for idx in self.sim_indexes]
        potential_surf = [surface_file_path.format(i=idx) for idx in self.sim_indexes]

        self.file_paths_vol = []
        self.file_paths_surf = []
        valid_sim = []
        for pv, ps, simidx in zip(potential_vol, potential_surf, self.sim_indexes):
            if os.path.exists(pv) and os.path.exists(ps):
                self.file_paths_vol.append(pv)
                self.file_paths_surf.append(ps)
                valid_sim.append(simidx)
            else:
                print(f"Warning: missing vol/surf for sim {simidx} -> skipping")

        self.sim_indexes = valid_sim
        if len(self.file_paths_vol) == 0:
            raise RuntimeError("No valid (volume+surface) files found")

        self.use_volume = use_volume
        self.use_surface = use_surface
        self.max_num_surface_points = max_num_surface_points if use_surface else 0
        self.max_num_volume_points = max_num_volume_points if use_volume else 0
        self.use_cache = use_cache
        self.data = {}  # simple per-sim cache (optional)

        # load global parameters and normalize
        df = pd.read_csv(global_parameter_path)
        df.columns = df.columns.str.strip()
        # select rows by run number (aligned with self.sim_indexes), not by CSV position
        df = df.set_index("Run")
        missing = [s for s in self.sim_indexes if s not in df.index]
        if missing:
            raise ValueError(f"No global parameters in {global_parameter_path} for runs {missing}")
        gp = torch.tensor(df.loc[self.sim_indexes, global_parameter_names].values, dtype=torch.float32)
        self.global_parameters = (gp - torch.tensor(global_parameter_min)) / (torch.tensor(global_parameter_max) - torch.tensor(global_parameter_min))
        self.global_parameters = self.global_parameters.to(torch.float32)

        # prepare canonical names and channel totals
        self.values_name = self.v_value_keys + self.s_value_keys
        self.attrs_name = self.v_attr_keys + self.s_attr_keys
        self.total_val_ch_v = int(np.sum(self.v_value_channels)) if len(self.v_value_channels) else 0
        self.total_val_ch_s = int(np.sum(self.s_value_channels)) if len(self.s_value_channels) else 0
        self.total_attr_ch_v = int(np.sum(self.v_attr_channels)) if len(self.v_attr_channels) else 0
        self.total_attr_ch_s = int(np.sum(self.s_attr_channels)) if len(self.s_attr_channels) else 0

        self.total_val_ch = self.total_val_ch_v + self.total_val_ch_s
        self.total_attr_ch = self.total_attr_ch_v + self.total_attr_ch_s

    def __len__(self):
        return len(self.file_paths_vol)

    def __getitem__(self, idx):
        vol_path = self.file_paths_vol[idx]
        surf_path = self.file_paths_surf[idx]
        sim_index = self.sim_indexes[idx]

        # global_parameters rows are aligned with self.sim_indexes / file lists
        global_params = self.global_parameters[idx]

        # optional simple cache
        if self.use_cache and sim_index in self.data:
            entry = self.data[sim_index]
            coords_s = entry["coords_s"]
            coords_v = entry["coords_v"]
            attributes_s = entry["attributes_s"]
            attributes_v = entry["attributes_v"]
            values_s = entry["values_s"]
            values_v = entry["values_v"]
            Ns = entry["Ns"]
            Nv = entry["Nv"]
        else:
            with h5py.File(vol_path, "r") as hf_v, h5py.File(surf_path, "r") as hf_s:
                n_v = int(hf_v["coords"].shape[0])
                n_s = int(hf_s["coords"].shape[0])
                total = n_v + n_s

                # read coords
                coords_v = np.asarray(hf_v["coords"][:]) if self.use_volume else np.empty((0, 3), dtype=np.float32)
                coords_s = np.asarray(hf_s["coords"][:]) if self.use_surface else np.empty((0, 3), dtype=np.float32)
                coords_v = torch.as_tensor(coords_v).unsqueeze(1)  # (Nv,1,D)
                coords_s = torch.as_tensor(coords_s).unsqueeze(1)  # (Ns,1,D)

                Nv = coords_v.shape[0]
                Ns = coords_s.shape[0]

                # helper to read keys from h5 for selected indices -> returns (N, total_ch_for_source)
                def _read_keys(hf, keys, channels):
                    if len(keys) == 0:
                        return torch.empty((0, 1, 0))
                    out_parts = []
                    for key, ch in zip(keys, channels):
                        arr = np.asarray(hf[key][:])
                        t = torch.as_tensor(arr)
                        # normalize to (N, ch)
                        if t.ndim == 1:
                            t = t.unsqueeze(-1)
                        if t.ndim == 2:
                            # ensure shape (N, ch)
                            pass
                        # finally store as (N, 1, ch)
                        out_parts.append(t.unsqueeze(1) if t.ndim == 2 else t.view(t.shape[0], 1, -1))
                    if len(out_parts) == 0:
                        return torch.empty((0, 1, 0))
                    return torch.cat(out_parts, dim=-1)  # (N,1,sum_ch)

                attributes_v = _read_keys(hf_v, self.v_attr_keys, self.v_attr_channels)
                values_v = _read_keys(hf_v, self.v_value_keys, self.v_value_channels)
                attributes_s = _read_keys(hf_s, self.s_attr_keys, self.s_attr_channels)
                values_s = _read_keys(hf_s, self.s_value_keys, self.s_value_channels)

                # normalize empty-source tensors to shape (N_source, 1, 0) when needed
                def _ensure_n_first_dim(t: torch.Tensor, N: int):
                    # if t is empty (0,1,0) produce zeros with N rows and 0 channels
                    if t.numel() == 0:
                        return t.new_zeros((N, 1, 0))
                    # if t has matching first dim keep; else error (unexpected)
                    if t.shape[0] != N:
                        raise RuntimeError(f"Source tensor first dim {t.shape[0]} != expected {N}")
                    return t

                attributes_v = _ensure_n_first_dim(attributes_v, Nv)
                attributes_s = _ensure_n_first_dim(attributes_s, Ns)
                values_v = _ensure_n_first_dim(values_v, Nv)
                values_s = _ensure_n_first_dim(values_s, Ns)

            # optionally cache the constructed sample
            if self.use_cache:
                self.data[sim_index] = {
                    "coords_s": coords_s,
                    "coords_v": coords_v,
                    "attributes_s": attributes_s,
                    "attributes_v": attributes_v,
                    "values_s": values_s,
                    "values_v": values_v,
                    "Ns": Ns,
                    "Nv": Nv,
                }

        # attributes_v = torch.cat([attributes_v, torch.zeros((Nv, 1, self.total_attr_ch_s))], dim=-1)
        # attributes_s = torch.cat([torch.zeros((Ns, 1, self.total_attr_ch_v)), attributes_s], dim=-1)
        # values_v = torch.cat([values_v, torch.zeros((Nv, 1, self.total_val_ch_s))], dim=-1)
        # values_s = torch.cat([torch.zeros((Ns, 1, self.total_val_ch_v)), values_s], dim=-1)

        if len(self.attr_subtract) > 0:
           attr_sub_t = torch.tensor(self.attr_subtract, dtype=torch.float32)
           attr_pad_v = attr_sub_t[self.total_attr_ch_v:].view(1, 1, -1).expand(Nv, 1, -1)
           attr_pad_s = attr_sub_t[:self.total_attr_ch_v].view(1, 1, -1).expand(Ns, 1, -1)
        else:
           attr_pad_v = torch.zeros((Nv, 1, self.total_attr_ch_s), dtype=torch.float32)
           attr_pad_s = torch.zeros((Ns, 1, self.total_attr_ch_v), dtype=torch.float32)

        if len(self.val_subtract) > 0:
           val_sub_t = torch.tensor(self.val_subtract, dtype=torch.float32)
           val_pad_v = val_sub_t[self.total_val_ch_v:].view(1, 1, -1).expand(Nv, 1, -1)
           val_pad_s = val_sub_t[:self.total_val_ch_v].view(1, 1, -1).expand(Ns, 1, -1)
        else:
           val_pad_v = torch.zeros((Nv, 1, self.total_val_ch_s), dtype=torch.float32)
           val_pad_s = torch.zeros((Ns, 1, self.total_val_ch_v), dtype=torch.float32)

        attributes_v = torch.cat([attributes_v, attr_pad_v], dim=-1)
        attributes_s = torch.cat([attr_pad_s, attributes_s], dim=-1)
        values_v = torch.cat([values_v, val_pad_v], dim=-1)
        values_s = torch.cat([val_pad_s, values_s], dim=-1)



        # combine volume and surface data (volume first, then surface)
        coords = torch.cat([coords_v, coords_s], dim=0)
        attributes = torch.cat([attributes_v, attributes_s], dim=0)
        values = torch.cat([values_v, values_s], dim=0)
        num_points = torch.tensor(coords.shape[0])

        # from Nv and Ns get max_num_points random indices
        if getattr(self, "disable_subsampling", False):
            perm_v = np.arange(Nv)
            perm_s = np.arange(Ns)
        else:
            epoch = self.epoch.value if hasattr(self.epoch, "value") else self.epoch
            rng = np.random.default_rng(epoch * 10000 + idx)
            sz_v = min(self.max_num_volume_points, Nv)
            perm_v = rng.choice(Nv, size=sz_v, replace=False)
            perm_v = np.sort(perm_v)
            sz_s = min(self.max_num_surface_points, Ns)
            perm_s = rng.choice(Ns, size=sz_s, replace=False)
            perm_s = np.sort(perm_s)

        selected_indices = torch.cat([torch.from_numpy(perm_v), torch.from_numpy(perm_s + Nv)], dim=0)

        selected_indices_np = selected_indices.numpy()
        coords = torch.from_numpy(coords.numpy()[selected_indices_np])
        attributes = torch.from_numpy(attributes.numpy()[selected_indices_np])
        values = torch.from_numpy(values.numpy()[selected_indices_np])
        num_points = torch.tensor(coords.shape[0])

        physical_metadata = PhysicalMetadata(
            min_domain=self.min_domain,
            max_domain=self.max_domain,
            hparams=torch.zeros(0),
            dt=torch.tensor(1.0),
            num_hparams=torch.tensor(0, dtype=torch.long),
            num_attrs=torch.tensor(len(self.attrs_name), dtype=torch.long),
        )

        if len(self.val_subtract) == 0:
            val_sub_t = torch.tensor([], dtype=torch.float32)
            val_div_t = torch.tensor([], dtype=torch.float32)
        else:
            val_sub_t = torch.tensor(self.val_subtract, dtype=torch.float32)
            val_div_t = torch.tensor(self.val_divide, dtype=torch.float32)

        if len(self.attr_subtract) == 0:
            attr_sub_t = torch.tensor([], dtype=torch.float32)
            attr_div_t = torch.tensor([], dtype=torch.float32)
        else:
            attr_sub_t = torch.tensor(self.attr_subtract, dtype=torch.float32)
            attr_div_t = torch.tensor(self.attr_divide, dtype=torch.float32)

        loading_metadata = LoadingMetadata(
            name=f"sim_{sim_index}",
            values_name=self.values_name,
            attrs_name=self.attrs_name,
            hparams_name=[],
            grid_dims=tuple(),
            channel_last=True,
            val_subtract=val_sub_t,
            val_divide=val_div_t,
            attr_subtract=attr_sub_t,
            attr_divide=attr_div_t,
            num_surface_points=torch.tensor(len(perm_s), dtype=torch.long),
            num_surface_value_channels=torch.tensor(self.total_val_ch_s, dtype=torch.long),
            num_surface_attr_channels=torch.tensor(self.total_attr_ch_s, dtype=torch.long),
        )

        return asdict_shallow(
            DataSample(
                positions=coords,
                values=values,
                attributes=attributes,
                num_points=num_points,
                global_params=global_params,
                physical_metadata=physical_metadata,
                loading_metadata=loading_metadata,
            )
        )

    def set_epoch(self, epoch: int):
        if hasattr(self.epoch, "value"):
            self.epoch.value = int(epoch)
        else:
            self.epoch = int(epoch)
