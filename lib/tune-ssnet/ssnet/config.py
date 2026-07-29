from dataclasses import dataclass, field

from common.config import Config, DataConfig, TrainConfig
from common.config import load_config as _load_config


@dataclass(kw_only=True)
class SSNetModelConfig:
    """Mirrors neossnet MaskNet's constructor exactly, so ``MaskNet(**asdict(model))`` works.

    Everything here is architecture: change any of it and an existing checkpoint no longer
    loads, which is the test for belonging in the model section.
    """

    num_sources: int = 2
    stochastic: bool = False

    # encoder/decoder parameters
    enc_kernel_size: int = 512
    enc_num_feats: int = 512
    enc_type: str = "convolution"    # convolution, spectrogram, spectrogram2
    dec_type: str = "convolution"    # convolution, group-convolution, spectrogram

    # mask generator parameters
    msk_num_feats: int = 256
    msk_num_heads: int = 4
    msk_ffn_expand: int = 4
    msk_num_layers: int = 4
    msk_use_conv: bool = True
    msk_kernel_size: int = 3
    msk_conv_layers: int = 6
    msk_dropout: float = 0.3
    msk_type: str = "transformer"    # conformer, transformer, transformer_relative
    msk_individual_mask: bool = True

    # wavelet analysis
    use_wavelet: bool = False
    wavelet_scale: int = 8
    mother_wavelet: str = "db10"


@dataclass(kw_only=True)
class SSNetTrainConfig(TrainConfig):
    """Base knobs plus the per-source loss weighting neossnet's losses take."""

    loss: str = "SNR"        # 'SDR' | 'LogMSE' | 'SNR' | 'SASDR' | 'MSE'; see task.LOSSES
    weights: str = "1.0, 1.0"   # per-source loss weights, one per num_sources


# No SSNetDataConfig: common.DataConfig already covers train_dir/val_fraction/num_workers.
# ssnet has no val_dir -- it carves validation out of train_dir by index (val_fraction).


@dataclass(kw_only=True)
class SSNetConfig(Config):
    model: SSNetModelConfig = field(default_factory=SSNetModelConfig)
    train: SSNetTrainConfig = field(default_factory=SSNetTrainConfig)
    data: DataConfig = field(default_factory=DataConfig)


def load_config(path: str) -> SSNetConfig:
    from ssnet.task import SSNetTask   # local import: task.py imports this module
    return _load_config(path, SSNetTask())
