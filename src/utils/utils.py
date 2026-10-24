from pathlib import Path
import os

from lightning.pytorch.utilities.rank_zero import rank_zero_only, rank_zero_info
from omegaconf import OmegaConf

import secrets
import string


DEFAULT_CFG_NAME = "config.yaml"


def resolve_dir(trainer: "L.Trainer", suffix='checkpoint') -> Path:
    """Determines model checkpoint save directory at runtime. Reference attributes from the trainer's logger to
    determine where to save checkpoints. The path for saving weights is set in this priority:

    1.  The ``Logger``'s ``log_dir`` if the trainer has loggers
    2.  The ``Trainer``'s ``default_root_dir`` if the trainer has no loggers

    The path gets extended with subdirectory "checkpoints".

    """

    # if len(trainer.loggers) > 0:
    #     if trainer.loggers[0].save_dir is not None:
    #         save_dir = trainer.loggers[0].save_dir
    #     else:
    #         save_dir = trainer.default_root_dir
    #     name = trainer.loggers[0].name
    #     version = trainer.loggers[0].version
    #     version = version if isinstance(version, str) else f"version_{version}"
    #     dir_path = os.path.join(save_dir, str(name), version, suffix)
    # else:
    # if no loggers, use default_root_dir

    dir_path = os.path.join(trainer.default_root_dir, suffix)

    dir_path = Path(dir_path)

    return dir_path

@rank_zero_only
def save_config(config, config_dir):
    config_dir.mkdir(exist_ok=True, parents=True)
    config_file = config_dir.joinpath(DEFAULT_CFG_NAME)
    rank_zero_info(f'Saving config.')
    OmegaConf.save(config, config_file)

def load_run_id(project_path):

    file = project_path.joinpath(DEFAULT_CFG_NAME)

    if file.exists():
        config = OmegaConf.load(file)
        return config.run_id
    else:
        raise FileNotFoundError(f"Config file {file} not found.")

# see wandb.sdk.lib.runid
def generate_id(length: int = 8) -> str:
    """Generate a random base-36 string of `length` digits."""
    # There are ~2.8T base-36 8-digit strings. If we generate 210k ids,
    # we'll have a ~1% chance of collision.
    alphabet = string.ascii_lowercase + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(length))


def setup_training_mode(cfg):
    """
    Configures trainer settings for epoch-based training.
    """
    rank_zero_info(
        f"[Training Mode: EPOCH] max_epochs: {getattr(cfg, 'max_epochs', 200)} | "
        f"warmup_epochs: {getattr(cfg, 'warmup_epochs', 10)}"
    )