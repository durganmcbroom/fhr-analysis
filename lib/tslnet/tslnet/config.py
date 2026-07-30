from dataclasses import dataclass, field
from typing import List, Optional

from common.config import Config, DataConfig, TrainConfig
from common.config import load_config as _load_config

from tslnet.model import DEFAULT_CHECKPOINT


@dataclass(kw_only=True)
class TSLNetModelConfig:
    """Architecture plus the input contract.

    The envelope front-end (n_fft/hop_length/band/log_envelope) lives here rather than in
    ``data`` for the same reason FUNet's spectrogram geometry does: it fixes the series the
    backbone sees, and a checkpoint cannot be run without it. ``crop_len`` stays in ``train``
    -- it only sets how many patches a batch covers, and any multiple of ``patch_length`` up
    to the context works.
    """

    channels: int = 3            # abdomen fibers stacked as separate univariate series
    checkpoint: str = DEFAULT_CHECKPOINT
    head_hidden: int = 256       # bottleneck width of the trainable MLP
    # Linear layers in the head. 3 is in -> hidden -> hidden -> out; 1 is a plain linear probe
    # (head_hidden then does nothing), which is the standard baseline for a frozen backbone.
    head_layers: int = 3
    dropout: float = 0.0         # between the MLP's layers; 0 = off

    # Declared rather than read off the backbone so check_feasible stays offline -- otherwise
    # rejecting a bad crop_len would first cost a 2 GB download. build_model verifies both
    # against the checkpoint it actually loaded and raises if they disagree.
    context_length: int = 2048
    patch_length: int = 32

    # Envelope front-end. hop_length sets the frame rate (SAMPLE_RATE / hop = 250 Hz at 16),
    # which is what makes a 32-frame patch 0.13 s; n_fft sets the analysis window (32 ms at
    # 128), long enough to resolve the 100-300 Hz band the beats live in.
    n_fft: int = 128
    hop_length: int = 16
    # Passband in Hz for the envelope, or null for the full spectrum. Defaults to the same
    # 100-300 Hz the analyze pipeline bandpasses fibers to before FUNet sees them.
    band: Optional[List[float]] = field(default_factory=lambda: [100.0, 300.0])
    # log1p-compress the envelope. On, it matches FUNet's front-end and tames the dynamic
    # range of raw fiber audio; off, beats stay peakier and the backbone's own z-scoring
    # handles the scale. Genuinely unclear which transfers better, so it is searched.
    log_envelope: bool = True


@dataclass(kw_only=True)
class TSLNetTrainConfig(TrainConfig):
    """Base knobs plus the loss options shared with FUNet's beat-activity objectives."""

    loss: str = "corr_amp"   # 'kldiv' | 'snr' | 'corr' | 'corr_amp' | 'mse'; see task.LOSSES
    amp_weight: float = 0.1          # corr_amp only: weight on the d' peak-contrast term
    amp_beat_threshold: float = 0.1  # corr_amp only: frac of per-item target peak counting as a beat


# No TSLNetDataConfig: common.DataConfig already covers train_dir/val_dir/val_fraction/
# num_workers, and nothing about where the snippets live is TSLNet-specific.


@dataclass(kw_only=True)
class TSLNetConfig(Config):
    model: TSLNetModelConfig = field(default_factory=TSLNetModelConfig)
    train: TSLNetTrainConfig = field(default_factory=TSLNetTrainConfig)
    data: DataConfig = field(default_factory=DataConfig)


def load_config(path: str) -> TSLNetConfig:
    """Load a TSLNet config. Same signature as funet.config.load_config so external callers
    (the analyze pipeline) don't need to know about tasks."""
    from tslnet.task import TSLNetTask   # local import: task.py imports this module
    return _load_config(path, TSLNetTask())
