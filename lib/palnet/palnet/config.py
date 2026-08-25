from dataclasses import dataclass, field

from common.config import Config, DataConfig, TrainConfig
from common.config import load_config as _load_config

from palnet.model import DEFAULT_CHECKPOINT, DEFAULT_REVISION


@dataclass(kw_only=True)
class PALNetModelConfig:
    """Architecture plus the input contract.

    ``model_hz`` and ``hop`` live here rather than in ``train`` by the usual test: a checkpoint
    cannot be run without them. They are also the two knobs that decide what the frozen
    front-end sees at all -- see ``palnet.model``'s docstring for why the feeding rate is a
    design decision and not a formality. ``crop_len`` stays in ``train``: the backbone is fully
    convolutional in time, so any aligned length works.
    """

    channels: int = 3            # abdomen fibers
    # Fetched to the Hugging Face cache on first run (~259 MB), then reused. Pinned by
    # revision: the repo is a personal re-upload, and a silent re-push would otherwise change
    # what an existing checkpoint was trained against.
    checkpoint: str = DEFAULT_CHECKPOINT
    revision: str = DEFAULT_REVISION

    # The control for this model's whole premise. False keeps the architecture and the
    # front-end (a DFT basis and a mel filterbank -- randomising those would break the feature
    # extractor rather than control for it) but throws the AudioSet features away, so a run
    # measures what the head can do over a random projection of the same shape.
    #
    # Not searched, deliberately: it is an experiment arm, not a hyperparameter.
    pretrained: bool = True
    backbone_seed: int = 0       # makes the control reproducible; ignored when pretrained

    # 'per_fiber' puts the fibers on the batch axis and concatenates their per-frame embeddings
    # in the head -- the pretrained 1-channel stem is used exactly as published, at C times the
    # compute. 'stack' folds the stem across C input channels instead: 1x the compute, but the
    # stem is no longer the pretrained one.
    channel_mode: str = "per_fiber"

    # Where framewise embeddings are read off. 'after1' is PANNs' own 2048-d embedding at a /32
    # time stride; 'layer4' (512-d, /16) and 'layer3' (256-d, /8) are shallower, cheaper for a
    # given output frame rate, and often the more transferable features.
    feature_layer: str = "after1"

    # 'all' | 'after:1'..'after:4' | 'none'. 'after:N' trains resnet.layerN onwards.
    freeze: str = "all"
    # One label-free pass over the training data re-estimating every BatchNorm running
    # statistic. AudioSet's are the wrong ones here and knowably so; see check_feasible's
    # report on dead mel bins. Forces a full-backbone checkpoint (the statistics have to travel
    # with it), which is why it is off by default rather than always on.
    bn_recalibrate: bool = False
    bn_recalibrate_batches: int = 32
    bn0_trainable: bool = False  # unfreeze the input normaliser alone (128 params)

    # --- the front-end's two settings ---
    # Rate the 4 kHz snippets are resampled to before the backbone sees them. This is NOT a
    # convenience: the mel filterbank maps FFT bin *index*, so the rate decides where the fetal
    # band lands on the mel scale (8000 puts 100-300 Hz on 16 of the 64 bins; a nominally
    # correct 32000 leaves it 5) and simultaneously how long the fixed 1024-tap window is
    # (128 ms at 8000). check_feasible reports both.
    model_hz: int = 8000
    # The STFT convolution's stride, and the only front-end value that is not a stored weight.
    # Output frame stride is `hop x the feature layer's time downsample` -- 8 x 32 = 256
    # samples = 32 ms at model_hz 8000.
    hop: int = 8

    # Declared rather than read off the checkpoint so check_feasible stays offline -- otherwise
    # rejecting a bad crop would first cost a 259 MB download. build_model verifies both
    # against the backbone it loaded and raises if they disagree. Neither is adjustable: n_fft
    # is the length of the STFT conv kernel, which *is* the windowed DFT basis.
    n_fft: int = 1024
    mel_bins: int = 64

    head_hidden: int = 256       # bottleneck width of the trainable MLP
    # Linear layers in the head. 3 is in -> hidden -> hidden -> out; 1 is a plain linear probe
    # (head_hidden then does nothing), the standard baseline for a frozen backbone.
    head_layers: int = 3
    dropout: float = 0.0         # between the MLP's layers; 0 = off

    # How frame-rate activity is filled back onto the sample grid at readout (inference, the HR
    # metric and the diagnostic all follow this). 'linear' = straight lines between frames;
    # 'pchip' = shape-preserving cubic, smooth and still non-negative.
    interpolation: str = "linear"


@dataclass(kw_only=True)
class PALNetTrainConfig(TrainConfig):
    """Base knobs plus the loss options shared with FUNet's and TSLNet's beat objectives."""

    # Widened from the base class's int, as TSLNet's is. A crop has to be a whole number of
    # output frames or the pooling stages floor and drop the tail, and at 32 ms per frame the
    # aligned lengths are not integers -- 4.096 s is 128 frames. palnet.data floors whatever is
    # given here to the nearest aligned length and check_feasible reports it.
    crop_len: float = 4.096

    loss: str = "mse"                # 'snr' | 'corr' | 'corr_amp' | 'mse'; see task.LOSSES
    amp_weight: float = 0.1          # corr_amp only: weight on the d' peak-contrast term
    amp_beat_threshold: float = 0.1  # corr_amp only: frac of per-item target peak counting as a beat


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
