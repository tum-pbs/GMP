import os
import torch
from lightning.pytorch.utilities.rank_zero import rank_zero_info
from src.models.baselines.transolver_pp.transolver_pp_wrapper import TransolverPPWrapper


class TransolverFFTWrapper(TransolverPPWrapper):
    def __init__(
        self,
        path_pretrained_encoder_weights="",
        path_pretrained_decoder_weights="",
        path_pretrained_weights="",
        exclude_prefixes=[],
        module_encoder_prefixes=[],
        module_decoder_prefixes=[],
        **kwargs,
    ) -> None:
        # 1. Initialize the base TransolverPPWrapper model
        super().__init__(**kwargs)

        # 2. Resolve pretrained paths
        geo_path = path_pretrained_encoder_weights or path_pretrained_weights
        phy_path = path_pretrained_decoder_weights or (path_pretrained_weights if not path_pretrained_encoder_weights else "")

        state_dict_geo = {}
        state_dict_phy = {}

        if geo_path:
            if not os.path.exists(geo_path):
                raise ValueError(f"Pretrained encoder weights not found at {geo_path}.")
            
            if geo_path.endswith(".ckpt"):
                checkpoint_encoder = torch.load(geo_path, map_location="cpu", weights_only=False)
                state_dict_geo = checkpoint_encoder.get("state_dict", checkpoint_encoder)
                state_dict_geo = {k.replace("model.", "", 1) if k.startswith("model.") else k: v for k, v in state_dict_geo.items()}
            elif geo_path.endswith(".pth") or geo_path.endswith(".pt"):
                ckpt = torch.load(geo_path, map_location="cpu")
                state_dict_geo = ckpt.get("model_state_dict", ckpt.get("ema_state_dict", ckpt))

            # Strip common wrapper prefixes
            cleaned_geo = {}
            for k, v in state_dict_geo.items():
                new_k = k
                for prefix in ["transolver.", "model.", "module."]:
                    if new_k.startswith(prefix):
                        new_k = new_k[len(prefix):]
                cleaned_geo[new_k] = v
            state_dict_geo = cleaned_geo

        if phy_path:
            if not os.path.exists(phy_path):
                raise ValueError(f"Pretrained decoder weights not found at {phy_path}.")

            if phy_path.endswith(".ckpt"):
                checkpoint_decoder = torch.load(phy_path, map_location="cpu", weights_only=False)
                state_dict_phy = checkpoint_decoder.get("state_dict", checkpoint_decoder)
                state_dict_phy = {k.replace("model.", "", 1) if k.startswith("model.") else k: v for k, v in state_dict_phy.items()}
            elif phy_path.endswith(".pth") or phy_path.endswith(".pt"):
                ckpt = torch.load(phy_path, map_location="cpu")
                state_dict_phy = ckpt.get("model_state_dict", ckpt.get("ema_state_dict", ckpt))

            cleaned_phy = {}
            for k, v in state_dict_phy.items():
                new_k = k
                for prefix in ["transolver.", "model.", "module."]:
                    if new_k.startswith(prefix):
                        new_k = new_k[len(prefix):]
                cleaned_phy[new_k] = v
            state_dict_phy = cleaned_phy

        modules_from_geo = tuple(module_encoder_prefixes)
        modules_from_phy = tuple(module_decoder_prefixes)
        exclude_prefixes_tuple = tuple(exclude_prefixes)

        # Filter out excluded keys and mismatched keys
        filtered_state_dict = {}
        for k, v in self.state_dict().items():
            if exclude_prefixes_tuple and any(k.startswith(p) for p in exclude_prefixes_tuple):
                rank_zero_info(f"Excluding key: {k}")
                continue
            
            loaded_from_checkpoint = False

            if modules_from_geo and any(k.startswith(p) for p in modules_from_geo):
                if k in state_dict_geo:
                    if self.state_dict()[k].shape == state_dict_geo[k].shape:
                        filtered_state_dict[k] = state_dict_geo[k]
                        rank_zero_info(f"Including key from geo checkpoint: {k}")
                        loaded_from_checkpoint = True
                    else:
                        rank_zero_info(f"Skipping key {k} from geo checkpoint due to shape mismatch (checkpoint: {state_dict_geo[k].shape}, model: {self.state_dict()[k].shape})")

            if not loaded_from_checkpoint and modules_from_phy and any(k.startswith(p) for p in modules_from_phy):
                if k in state_dict_phy:
                    if self.state_dict()[k].shape == state_dict_phy[k].shape:
                        filtered_state_dict[k] = state_dict_phy[k]
                        rank_zero_info(f"Including key from phy checkpoint: {k}")
                        loaded_from_checkpoint = True
                    else:
                        rank_zero_info(f"Skipping key {k} from phy checkpoint due to shape mismatch (checkpoint: {state_dict_phy[k].shape}, model: {self.state_dict()[k].shape})")

            # Fallback: if no specific encoder/decoder prefixes were specified, match directly from geo or phy state dict
            if not loaded_from_checkpoint and not modules_from_geo and not modules_from_phy:
                source_dict = state_dict_geo or state_dict_phy
                if k in source_dict:
                    if self.state_dict()[k].shape == source_dict[k].shape:
                        filtered_state_dict[k] = source_dict[k]
                        rank_zero_info(f"Including key from checkpoint: {k}")
                        loaded_from_checkpoint = True
                    else:
                        rank_zero_info(f"Skipping key {k} due to shape mismatch (checkpoint: {source_dict[k].shape}, model: {self.state_dict()[k].shape})")

            if not loaded_from_checkpoint:
                rank_zero_info(f"Key {k} not found in checkpoint (or skipped), initializing randomly.")

        missing, unexpected = self.load_state_dict(filtered_state_dict, strict=False)
        self.missing_keys = missing
        rank_zero_info(f"Loaded {len(filtered_state_dict)} tensors for TransolverFFTWrapper.")
        if missing:
            rank_zero_info(f"The following layers will be trained from SCRATCH: {missing}")
