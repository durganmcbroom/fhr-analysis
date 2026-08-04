from dataclasses import dataclass, field

from common.config import Config, DataConfig, TrainConfig
from common.config import load_config as _load_config

from tslnet.model import DEFAULT_CHECKPOINT


@dataclass(kw_only=True)
class TSLNetModelConfig:
    """Architecture plus the input contract.

    ``model_hz`` lives here rather than in ``data`` for the same reason FUNet's spectrogram
    geometry does: it fixes the series the backbone sees, and a checkpoint cannot be run
    without it. ``crop_len`` stays in ``train`` -- it only sets how many patches a batch
    covers, and any multiple of ``patch_length`` up to the context works.
    """

    channels: int = 3            # abdomen fibers stacked as separate univariate series
    checkpoint: str = DEFAULT_CHECKPOINT

    # The control for this model's whole premise. False keeps the architecture named by
    # `checkpoint` but throws the pretrained weights away and initialises randomly, so a run
    # measures what the head can do over a random projection of the same shape. If the
    # pretrained arm does not beat it, TimesFM's pretraining is contributing nothing here.
    #
    # Not searched, deliberately: it is an experiment arm, not a hyperparameter.
    pretrained: bool = True
    # Makes the control reproducible, so head-only checkpoints stay valid for that arm too.
    # Ignored when pretrained is True. One seed is one draw -- repeat at 2-3 before reading
    # much into a small gap.
    backbone_seed: int = 0
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

    # Rate the waveform is decimated to before the backbone sees it. Bounded below by Nyquist
    # (the fetal band runs to 300 Hz, and an anti-alias filter needs headroom above that) and
    # above by the context, since crop_len * model_hz steps have to fit in context_length.
    # Must divide SAMPLE_RATE evenly so decimation and the target pooling are exact.
    model_hz: int = 800


@dataclass(kw_only=True)
class TSLNetTrainConfig(TrainConfig):
    """Base knobs plus the loss options shared with FUNet's beat-activity objectives."""

    # Widened from the base class's int. TSLNet wants a crop that fills the backbone's context
    # exactly -- 2048 steps at 800 Hz is 2.56 s -- and rounding to whole seconds would either
    # overflow the context or leave a fifth of it unused. Only tslnet.data reads this, so
    # funet and ssnet keep their integer seconds.
    crop_len: float = 2.56

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
