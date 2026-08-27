from dataclasses import dataclass, field
from typing import List, Optional

from common.config import Config, DataConfig, TrainConfig
from common.config import load_config as _load_config


@dataclass(kw_only=True)
class PALNetModelConfig:
    """Architecture plus the input contract.

    ``n_fft``/``hop_length``/``freq_crop_hz`` live here rather than in ``train`` by the usual
    test: a checkpoint cannot be run without them. They are also, deliberately, FUNet's --
    PALNet exists to test the backbone, and that is only answerable if both models are fed the
    identical tensor.

    The AudioSet checkpoint itself is not a field. There is one checkpoint (see
    ``palnet.model.CHECKPOINT``, pinned by revision); a different one is a different study.
    """

    channels: int = 3            # abdomen fibers, stacked as separate spectrograms

    # The control for this model's whole premise. False keeps the architecture but throws the
    # AudioSet features away, so a run measures what the head can do over a random projection
    # of the same shape. If the pretrained arm does not beat it, 59.5M parameters are
    # contributing nothing.
    #
    # Not searched, deliberately: it is an experiment arm, not a hyperparameter.
    pretrained: bool = True
    backbone_seed: int = 0       # makes the control reproducible; ignored when pretrained

    # Where framewise embeddings are read off the trunk. 'after1' is PANNs' own 2048-d
    # embedding and the most semantic point in the network; 'layer4' (512-d) and 'layer3'
    # (256-d) are shallower and often the more transferable features. Time resolution is
    # identical at all three -- the trunk's pools are frequency-only.
    feature_layer: str = "after1"

    # --- the front-end, matching funet-v36 ---
    # At n_fft 1024 the bin width is 3.906 Hz, so [80, 350] keeps 71 rows, which the trunk's
    # five frequency halvings floor to exactly 64 spanning 78.1-328.1 Hz.
    n_fft: int = 1024
    hop_length: int = 256        # 64 ms frames, and the target's bin width
    freq_crop_hz: Optional[List[float]] = field(default_factory=lambda: [80.0, 350.0])

    head_hidden: int = 256       # bottleneck width of the trainable MLP
    # Linear layers in the head. 3 is in -> hidden -> hidden -> out; 1 is a plain linear probe
    # (head_hidden then does nothing), the standard baseline for a frozen backbone.
    head_layers: int = 1
    dropout: float = 0.0         # between the MLP's layers; 0 = off

    # How frame-rate activity is filled back onto the sample grid at readout (inference, the HR
    # metric and the diagnostic all follow this). 'linear' = straight lines between frames;
    # 'pchip' = shape-preserving cubic, smooth and still non-negative.
    interpolation: str = "linear"


@dataclass(kw_only=True)
class PALNetTrainConfig(TrainConfig):
    """Base knobs plus the loss options and masks shared with FUNet's beat objectives.

    ``crop_len`` stays the base class's plain integer seconds: the trunk is fully convolutional
    in time, so any length works and nothing has to be aligned.
    """

    loss: str = "mse"                # 'snr' | 'corr' | 'corr_amp' | 'mse'; see task.LOSSES
    amp_weight: float = 0.1          # corr_amp only: weight on the d' peak-contrast term
    amp_beat_threshold: float = 0.1  # corr_amp only: frac of per-item target peak counting as a beat

    # SpecAugment-style masks, max width of one zeroed band per sample; 0 = off.
    freq_mask: int = 0   # freq rows (of the 64 kept); try ~8
    time_mask: int = 0   # time frames; keep under one beat interval (~6 at hop 256), try 4-6


# No PALNetDataConfig: common.DataConfig already covers train_dir/val_dir/val_fraction/
# num_workers/preprocess, and nothing about where the snippets live is PALNet-specific.


@dataclass(kw_only=True)
class PALNetConfig(Config):
    model: PALNetModelConfig = field(default_factory=PALNetModelConfig)
    train: PALNetTrainConfig = field(default_factory=PALNetTrainConfig)
    data: DataConfig = field(default_factory=DataConfig)


def load_config(path: str) -> PALNetConfig:
    """Load a PALNet config. Same signature as funet.config.load_config and
    tslnet.config.load_config so external callers (the analyze pipeline, rtmon) don't need to
    know about tasks."""
    from palnet.task import PALNetTask   # local import: task.py imports this module
    return _load_config(path, PALNetTask())
