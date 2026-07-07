from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml


@dataclass
class ModelConfig:
    channels: int = 4
    dilations: List[int] = field(default_factory=lambda: [1, 1, 1, 2, 2, 4, 4])
    bottleneck_dilation: int = 8
    base_channels: int = 64    # first-level width; every level doubles from here


@dataclass
class TrainConfig:
    optimizer: str = "AdamW"   # SGD, Adam, AdamW
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    batch_size: int = 8
    epochs: int = 40
    crop_len: int = 7          # seconds
    clip: float = 5.0          # max gradient norm
    loss: str = "kldiv"        # 'kldiv' (distribution), 'snr' (SI-SNR, sign-invariant), or 'corr' (sign-sensitive)


@dataclass
class DataConfig:
    train_dir: str = "lib/tune-ssnet/training/training_clips_mono/fetal-train"
    test_dir: str = "lib/tune-ssnet/training/training_clips_mono/fetal-test"
    num_workers: int = 4
    n_fft: int = 1024
    hop_length: int = 256


@dataclass
class Config:
    model: ModelConfig
    train: TrainConfig
    data: DataConfig
    model_dir: str
    resume: Optional[str] = None   # path to a model checkpoint (.pt) to resume training from


def _resolve_path(base_dir: Path, value: Optional[str]) -> Optional[str]:
    """Resolve ``value`` relative to ``base_dir``, or leave it absolute.

    ``base_dir / value`` yields ``value`` unchanged when it is already absolute;
    ``resolve()`` normalises ``..`` segments and makes it absolute. ``None`` passes
    through so optional keys (e.g. resume) stay unset.
    """
    if value is None:
        return None
    return str((base_dir / value).resolve())


def load_config(path: str) -> Config:
    # Path fields in the yaml are resolved relative to the config file's own
    # directory (not the process CWD), so the same yaml works no matter where
    # python is invoked from. Absolute paths are left as-is.
    config_path = Path(path).resolve()
    base_dir = config_path.parent

    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)

    config = Config(
        model=ModelConfig(**raw.get("model", {})),
        train=TrainConfig(**raw.get("train", {})),
        data=DataConfig(**raw.get("data", {})),
        model_dir=raw["model_dir"],
        resume=raw.get("resume"),
    )

    config.data.train_dir = _resolve_path(base_dir, config.data.train_dir)
    config.data.test_dir = _resolve_path(base_dir, config.data.test_dir)
    config.model_dir = _resolve_path(base_dir, config.model_dir)
    config.resume = _resolve_path(base_dir, config.resume)

    return config
