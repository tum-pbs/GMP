import os
import torch
from torch import nn
from lightning.pytorch.utilities.rank_zero import rank_zero_info

from ..gmpnet.gmpnet_wrapper import GMPNetWrapper
from ..gmpnet.gmpnet_wrapper_autoreg import GMPNetWrapper as GMPNetWrapperAutoreg


def load_filtered_pretrained_weights(
    model: nn.Module,
    path_pretrained_encoder_weights: str = "",
    path_pretrained_decoder_weights: str = "",
    exclude_prefixes: list | tuple = (),
    module_encoder_prefixes: list | tuple = (),
    module_decoder_prefixes: list | tuple = (),
) -> None:
    """Loads matching weights from checkpoint files into the model while excluding specified heads."""
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
            try:
                checkpoint_encoder = torch.load(path_pretrained_encoder_weights, map_location="cpu", weights_only=False)
            except TypeError:
                checkpoint_encoder = torch.load(path_pretrained_encoder_weights, map_location="cpu")
            state_dict_geo = checkpoint_encoder.get("model_state_dict", checkpoint_encoder.get("ema_state_dict", checkpoint_encoder))
            state_dict_geo = {k.replace("model.", "", 1) if k.startswith("model.") else k: v for k, v in state_dict_geo.items()}

    if path_pretrained_decoder_weights:
        if not os.path.exists(path_pretrained_decoder_weights):
            raise ValueError(f"Pretrained decoder weights not found at {path_pretrained_decoder_weights}.")
        if path_pretrained_decoder_weights.endswith(".ckpt"):
            checkpoint_decoder = torch.load(path_pretrained_decoder_weights, map_location="cpu", weights_only=False)
            state_dict_phy = checkpoint_decoder.get("state_dict", checkpoint_decoder)
            state_dict_phy = {k.replace("model.", "", 1) if k.startswith("model.") else k: v for k, v in state_dict_phy.items()}
        elif path_pretrained_decoder_weights.endswith(".pth") or path_pretrained_decoder_weights.endswith(".pt"):
            try:
                checkpoint_decoder = torch.load(path_pretrained_decoder_weights, map_location="cpu", weights_only=False)
            except TypeError:
                checkpoint_decoder = torch.load(path_pretrained_decoder_weights, map_location="cpu")
            state_dict_phy = checkpoint_decoder.get("model_state_dict", checkpoint_decoder.get("ema_state_dict", checkpoint_decoder))
            state_dict_phy = {k.replace("model.", "", 1) if k.startswith("model.") else k: v for k, v in state_dict_phy.items()}

    modules_from_geo = tuple(module_encoder_prefixes)
    modules_from_phy = tuple(module_decoder_prefixes)
    exclude = tuple(exclude_prefixes)

    filtered_state_dict = {}
    current_state = model.state_dict()
    for k, v in current_state.items():
        if any(k.startswith(p) for p in exclude):
            rank_zero_info(f"Excluding key: {k}")
            continue
        elif any(k.startswith(p) for p in modules_from_geo):
            if k in state_dict_geo:
                if v.shape == state_dict_geo[k].shape:
                    filtered_state_dict[k] = state_dict_geo[k]
                    rank_zero_info(f"Including key from geo checkpoint: {k}")
                else:
                    rank_zero_info(f"Skipping key {k} from geo checkpoint due to shape mismatch ({state_dict_geo[k].shape} vs {v.shape})")
        elif any(k.startswith(p) for p in modules_from_phy):
            if k in state_dict_phy:
                if v.shape == state_dict_phy[k].shape:
                    filtered_state_dict[k] = state_dict_phy[k]
                    rank_zero_info(f"Including key from phy checkpoint: {k}")
                else:
                    rank_zero_info(f"Skipping key {k} from phy checkpoint due to shape mismatch ({state_dict_phy[k].shape} vs {v.shape})")
        else:
            rank_zero_info(f"Key {k} not found in either checkpoint, initializing randomly.")

    if filtered_state_dict:
        missing, unexpected = model.load_state_dict(filtered_state_dict, strict=False)
        model.missing_keys = missing
        rank_zero_info(f"Loaded {len(filtered_state_dict)} tensors. The following layers will be trained from SCRATCH: {missing}")


class GMPNetFFTWrapper(GMPNetWrapper):
    """Full fine-tuning (FFT) wrapper for stationary GMPNet models."""

    def __init__(
        self,
        path_pretrained_encoder_weights: str = "",
        path_pretrained_decoder_weights: str = "",
        exclude_prefixes: list | tuple = (),
        module_encoder_prefixes: list | tuple = (),
        module_decoder_prefixes: list | tuple = (),
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        load_filtered_pretrained_weights(
            self,
            path_pretrained_encoder_weights=path_pretrained_encoder_weights,
            path_pretrained_decoder_weights=path_pretrained_decoder_weights,
            exclude_prefixes=exclude_prefixes,
            module_encoder_prefixes=module_encoder_prefixes,
            module_decoder_prefixes=module_decoder_prefixes,
        )


class GMPNetAutoregFFTWrapper(GMPNetWrapperAutoreg):
    """Full fine-tuning (FFT) wrapper for autoregressive GMPNet models."""

    def __init__(
        self,
        path_pretrained_encoder_weights: str = "",
        path_pretrained_decoder_weights: str = "",
        exclude_prefixes: list | tuple = (),
        module_encoder_prefixes: list | tuple = (),
        module_decoder_prefixes: list | tuple = (),
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        load_filtered_pretrained_weights(
            self,
            path_pretrained_encoder_weights=path_pretrained_encoder_weights,
            path_pretrained_decoder_weights=path_pretrained_decoder_weights,
            exclude_prefixes=exclude_prefixes,
            module_encoder_prefixes=module_encoder_prefixes,
            module_decoder_prefixes=module_decoder_prefixes,
        )


# Aliases for backwards compatibility
Smart2FFTWrapper = GMPNetFFTWrapper