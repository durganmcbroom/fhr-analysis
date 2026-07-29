"""SSNet's Task: the glue between neossnet's MaskNet and common's phases.

neossnet is used for the model and the separation losses only. Its train.py -- the loop,
optimiser/loss factories and checkpointing -- is deliberately NOT used: common owns those, so
ssnet gets atomic checkpoints, archived configs, all three LR schedules and the Optuna search
for free, and the old monkeypatch that wrapped neossnet's module-level train/test just to
record losses for a plot is gone.
"""

import copy
import functools
from dataclasses import asdict

import torch
from torch import nn

from common.audio import SAMPLE_RATE
from common.errors import InfeasibleConfig
from common.optim import OPTIMIZERS
from common.task import Task

from ssnet.config import SSNetConfig
from ssnet.data import make_dataloader

# Top-level names belonging to the lib/neossnet submodule, which is vendored
# unmodified and uses bare imports internally; pyproject maps them in as-is.
from models import MaskNet
from models.transformer import TransformerEncoder
from loss_fn import LogMSE_Loss, SASDR_Loss, SDR_Loss, SNR_Loss


def _parse_weights(weights: str) -> torch.Tensor:
    return torch.tensor([float(x) for x in weights.split(",")])


@functools.cache
def _transformer_max_frames() -> int:
    """How many frames neossnet's 'transformer' mask generator can attend over.

    It builds its sinusoidal positional encoding at a fixed max_len (a literal inside
    TransformerEncoder, not a constructor argument), so a longer sequence fails on the
    ``x + pos_enc(seq_len)`` broadcast rather than anything informative. Read the table's real
    size instead of copying the number here, so this check can't drift from neossnet.
    ``num_layers=0`` makes the probe an empty ModuleList around the encoding.
    """
    probe = TransformerEncoder(encoder_dim=2, num_layers=0, num_attention_heads=1)
    return probe.pos_enc.pe.shape[1]


def _encoder_frames(config) -> int:
    """Frames the encoder emits for one training crop -- the mask generator's sequence length.

    Mirrors MaskNet: the waveform is zero-padded up to a whole number of strides
    (``_align_num_frames_with_strides``) and then run through a Conv1d of ``enc_kernel_size``
    at stride ``enc_kernel_size // 2`` with ``stride`` padding. The spectrogram encoders use
    the same window and hop, so the count holds for them too.
    """
    kernel = config.model.enc_kernel_size
    stride = kernel // 2
    samples = config.train.crop_len * SAMPLE_RATE

    remainder = (samples - kernel % 2) % stride
    padded = samples + (stride - remainder if remainder else 0)
    return (padded + 2 * stride - kernel) // stride + 1


# loss name -> config -> loss module. All are separation losses over (B, S, T) and all are
# lower-is-better, which is what lets the search minimise validation loss directly.
LOSSES = {
    "MSE":    lambda cfg: nn.MSELoss(),
    "SDR":    lambda cfg: SDR_Loss(_parse_weights(cfg.train.weights)),
    "LogMSE": lambda cfg: LogMSE_Loss(_parse_weights(cfg.train.weights)),
    "SNR":    lambda cfg: SNR_Loss(_parse_weights(cfg.train.weights)),
    "SASDR":  lambda cfg: SASDR_Loss(_parse_weights(cfg.train.weights)),
}


class SSNetTask(Task):
    name = "ssnet"
    ConfigType = SSNetConfig
    device_env_vars = ("TUNE_DEVICE",)

    # ------------------------------------------------------------------ required
    def build_model(self, config) -> MaskNet:
        return MaskNet(**asdict(config.model))

    def build_loss(self, config):
        try:
            factory = LOSSES[config.train.loss]
        except KeyError:
            raise ValueError(
                f"Unknown loss: {config.train.loss!r} (expected one of {list(LOSSES)})") from None
        print(f"Loss: {config.train.loss}")
        return factory(config)

    def make_train_loader(self, config):
        return make_dataloader(config, train=True)

    def make_val_loader(self, config):
        return make_dataloader(config, train=False)

    # ------------------------------------------------------------------ optional
    def adapt_state_dict(self, state_dict: dict, config) -> dict:
        """Warm-start a multi-channel encoder from a 1-channel checkpoint by tiling the single
        input-channel weight across the new channels (divided to keep scale)."""
        target_channels = 1
        weight = state_dict.get("encoder.weight")
        if weight is None or weight.shape[1] == target_channels:
            return state_dict
        adapted = dict(state_dict)
        adapted["encoder.weight"] = weight.repeat(1, target_channels, 1) / target_channels
        return adapted

    def check_feasible(self, config) -> None:
        """Reject configs the transformer/conformer mask generator cannot build."""
        m = config.model
        if m.msk_num_feats % m.msk_num_heads:
            raise InfeasibleConfig(
                f"msk_num_feats ({m.msk_num_feats}) must be divisible by msk_num_heads "
                f"({m.msk_num_heads}) for multi-head attention to split evenly")

        n_weights = len(config.train.weights.split(","))
        if config.train.loss in ("SDR", "LogMSE", "SNR") and n_weights != m.num_sources:
            raise InfeasibleConfig(
                f"train.weights has {n_weights} entr(ies) but num_sources is "
                f"{m.num_sources}; the {config.train.loss} loss weights each source")

        # A smaller encoder kernel means a smaller stride, which means more frames per crop --
        # and the plain transformer's positional encoding is only so long. (The relative and
        # conformer mask generators encode position per-attention, so they have no such cap.)
        if m.msk_type == "transformer":
            frames, limit = _encoder_frames(config), _transformer_max_frames()
            if frames > limit:
                raise InfeasibleConfig(
                    f"a {config.train.crop_len}s crop at enc_kernel_size {m.enc_kernel_size} "
                    f"(stride {m.enc_kernel_size // 2}) is {frames} frames, over the "
                    f"{limit}-frame limit of the 'transformer' mask generator's positional "
                    f"encoding; raise enc_kernel_size, shorten crop_len, or use msk_type "
                    f"'transformer_relative'/'conformer'")

    # ------------------------------------------------------- optimize phase only
    def suggest(self, trial, base):
        """Return a copy of ``base`` with this trial's hyperparameters applied.

        Keep in sync with ``searched_fields``. The encoder/decoder types and the wavelet
        options are left alone deliberately: they change the model family rather than its
        size, and are better compared as separate studies than mixed into one search.
        """
        config = copy.deepcopy(base)
        model, train = config.model, config.train

        # -- Mask generator --
        # num_feats is sampled as a multiple of num_heads so attention always splits evenly
        # (check_feasible enforces the same invariant for hand-written configs).
        model.msk_num_heads = trial.suggest_categorical("msk_num_heads", [2, 4, 8])
        head_dim = trial.suggest_categorical("msk_head_dim", [16, 32, 64, 96])
        model.msk_num_feats = model.msk_num_heads * head_dim
        model.msk_num_layers = trial.suggest_int("msk_num_layers", 2, 8)
        model.msk_ffn_expand = trial.suggest_categorical("msk_ffn_expand", [2, 4])
        model.msk_dropout = trial.suggest_float("msk_dropout", 0.0, 0.5)
        model.msk_use_conv = trial.suggest_categorical("msk_use_conv", [True, False])
        model.msk_kernel_size = trial.suggest_categorical("msk_kernel_size", [3, 5, 7])
        model.msk_conv_layers = trial.suggest_int("msk_conv_layers", 2, 8)

        # -- Encoder --
        # Kernels below 128 are left out: at the crop lengths these configs use they always
        # blow the transformer's positional-encoding limit, so offering them would just burn a
        # quarter of the trials on configs check_feasible prunes on sight. That check is still
        # the backstop -- long enough crops put even 128 over the line.
        model.enc_num_feats = trial.suggest_categorical("enc_num_feats", [128, 256, 512])
        model.enc_kernel_size = trial.suggest_categorical("enc_kernel_size", [128, 256, 512])

        # -- Optimisation --
        train.optimizer = trial.suggest_categorical("optimizer", list(OPTIMIZERS))
        train.learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
        train.weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-1, log=True)
        # Fraction of the peak LR, so the cosine floor can never land above the peak.
        train.min_lr = train.learning_rate * trial.suggest_float("min_lr_frac", 1e-3, 1e-1, log=True)

        return config

    def searched_fields(self, config) -> dict:
        return {
            "model": {
                "msk_num_heads": config.model.msk_num_heads,
                "msk_num_feats": config.model.msk_num_feats,
                "msk_num_layers": config.model.msk_num_layers,
                "msk_ffn_expand": config.model.msk_ffn_expand,
                "msk_dropout": config.model.msk_dropout,
                "msk_use_conv": config.model.msk_use_conv,
                "msk_kernel_size": config.model.msk_kernel_size,
                "msk_conv_layers": config.model.msk_conv_layers,
                "enc_num_feats": config.model.enc_num_feats,
                "enc_kernel_size": config.model.enc_kernel_size,
            },
            "train": {
                "optimizer": config.train.optimizer,
                "learning_rate": config.train.learning_rate,
                "weight_decay": config.train.weight_decay,
                "min_lr": config.train.min_lr,
            },
        }
