import os
import torch
from lightning.pytorch.utilities.rank_zero import rank_zero_info
from src.models.baselines.abupt.abupt_wrapper import ABUPTWrapper

class AbuptFFTWrapper(ABUPTWrapper):
    def __init__(
        self,
        path_pretrained_encoder_weights="",
        path_pretrained_decoder_weights="",
        exclude_prefixes=[],
        module_encoder_prefixes=[],
        module_decoder_prefixes=[],
        **kwargs,
    ) -> None:
        # 1. Initialize the base ABUPTWrapper model
        super().__init__(**kwargs)

        # 2. Check for weight files
        state_dict_geo = {}
        state_dict_phy = {}

        if path_pretrained_encoder_weights:
            if not os.path.exists(path_pretrained_encoder_weights):
                raise ValueError(f"Pretrained encoder weights not found at {path_pretrained_encoder_weights}.")
            
            if path_pretrained_encoder_weights.endswith(".ckpt"):
                checkpoint_encoder = torch.load(path_pretrained_encoder_weights, map_location="cpu", weights_only=False)
                state_dict_geo = checkpoint_encoder.get("state_dict", checkpoint_encoder)
                state_dict_geo = {k.replace("model.", "", 1) if k.startswith("model.") else k: v for k, v in state_dict_geo.items()}
            elif path_pretrained_encoder_weights.endswith(".pth") or path_pretrained_encoder_weights.endswith(".pt"):
                ckpt = torch.load(path_pretrained_encoder_weights, map_location="cpu")
                state_dict_geo = ckpt.get("model_state_dict", ckpt.get("ema_state_dict", ckpt))

            # Strip abupt. and module. prefixes if present
            cleaned_geo = {}
            for k, v in state_dict_geo.items():
                new_k = k
                if new_k.startswith("abupt."):
                    new_k = new_k[len("abupt."):]
                if new_k.startswith("module."):
                    new_k = new_k[len("module."):]
                if new_k.startswith("abupt."):
                    new_k = new_k[len("abupt."):]
                cleaned_geo[new_k] = v
            state_dict_geo = cleaned_geo

        if path_pretrained_decoder_weights:
            if not os.path.exists(path_pretrained_decoder_weights):
                raise ValueError(f"Pretrained decoder weights not found at {path_pretrained_decoder_weights}.")

            if path_pretrained_decoder_weights.endswith(".ckpt"):
                checkpoint_decoder = torch.load(path_pretrained_decoder_weights, map_location="cpu", weights_only=False)
                state_dict_phy = checkpoint_decoder.get("state_dict", checkpoint_decoder)
                state_dict_phy = {k.replace("model.", "", 1) if k.startswith("model.") else k: v for k, v in state_dict_phy.items()}
            elif path_pretrained_decoder_weights.endswith(".pth") or path_pretrained_decoder_weights.endswith(".pt"):
                ckpt = torch.load(path_pretrained_decoder_weights, map_location="cpu")
                state_dict_phy = ckpt.get("model_state_dict", ckpt.get("ema_state_dict", ckpt))

            # Strip abupt. and module. prefixes if present
            cleaned_phy = {}
            for k, v in state_dict_phy.items():
                new_k = k
                if new_k.startswith("abupt."):
                    new_k = new_k[len("abupt."):]
                if new_k.startswith("module."):
                    new_k = new_k[len("module."):]
                if new_k.startswith("abupt."):
                    new_k = new_k[len("abupt."):]
                cleaned_phy[new_k] = v
            state_dict_phy = cleaned_phy

        modules_from_geo = tuple(module_encoder_prefixes)
        modules_from_phy = tuple(module_decoder_prefixes)
        exclude_prefixes = tuple(exclude_prefixes)

        # Filter out excluded keys and mismatched keys
        filtered_state_dict = {}
        for k, v in self.state_dict().items():
            if any(k.startswith(p) for p in exclude_prefixes):
                rank_zero_info(f"Excluding key: {k}")
                continue
            elif any(k.startswith(p) for p in modules_from_geo):
                if k in state_dict_geo:
                    if self.state_dict()[k].shape == state_dict_geo[k].shape:
                        filtered_state_dict[k] = state_dict_geo[k]
                        rank_zero_info(f"Including key from geo checkpoint: {k}")
                    else:
                        rank_zero_info(f"Skipping key {k} from geo checkpoint due to shape mismatch (checkpoint: {state_dict_geo[k].shape}, model: {self.state_dict()[k].shape})")
            elif any(k.startswith(p) for p in modules_from_phy):
                if k in state_dict_phy:
                    if self.state_dict()[k].shape == state_dict_phy[k].shape:
                        filtered_state_dict[k] = state_dict_phy[k]
                        rank_zero_info(f"Including key from phy checkpoint: {k}")
                    else:
                        rank_zero_info(f"Skipping key {k} from phy checkpoint due to shape mismatch (checkpoint: {state_dict_phy[k].shape}, model: {self.state_dict()[k].shape})")
            else:
                rank_zero_info(f"Key {k} not found in either checkpoint, initializing randomly.")

        missing, unexpected = self.load_state_dict(filtered_state_dict, strict=False)
        self.missing_keys = missing

        rank_zero_info(f"Loaded {len(filtered_state_dict)} tensors. "
                    f"The following layers will be trained from SCRATCH: {missing}")
