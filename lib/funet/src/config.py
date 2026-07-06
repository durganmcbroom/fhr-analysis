from dataclasses import dataclass, field
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
    loss: str = "kldiv"        # 'kldiv' (distribution) or 'snr' (SI-SNR signal loss)


@dataclass
class DataConfig:
    train_dir: str = "lib/tune-ssnet/training/training_clips_mono/fetal"
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


def load_config(path: str) -> Config:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    return Config(
        model=ModelConfig(**raw.get("model", {})),
        train=TrainConfig(**raw.get("train", {})),
        data=DataConfig(**raw.get("data", {})),
        model_dir=raw["model_dir"],
        resume=raw.get("resume"),
    )
