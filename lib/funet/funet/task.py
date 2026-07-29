"""FUNet's Task: the only glue between the model and common's phases."""

import copy

from torch import nn

from common.errors import InfeasibleConfig
from common.optim import OPTIMIZERS
from common.task import Task

from funet.config import FUNetConfig
from funet.data import make_dataloader, stft_output_shape
from funet.loss import CorrAmpLoss, CorrelationLoss, MSELoss, SNRLoss
from funet.model import FUNet

# loss name -> (config -> loss module, matching model output head). Factories take the config
# so amplitude-aware losses can read their hyperparameters from it.
#
# Single source of truth for the loss<->head coupling: the train entry point and inference
# used to keep two hand-maintained copies of this mapping, which could drift.
LOSSES = {
    "kldiv":    (lambda cfg: nn.KLDivLoss(reduction="batchmean"), "logprob"),
    "snr":      (lambda cfg: SNRLoss(), "signal"),
    "corr":     (lambda cfg: CorrelationLoss(), "signal"),   # sign-sensitive; fixes the SI-SNR sign-flip
    "corr_amp": (lambda cfg: CorrAmpLoss(amp_weight=cfg.train.amp_weight,   # corr + d' peak-contrast
                                         beat_threshold=cfg.train.amp_beat_threshold), "signal"),
    "mse":      (lambda cfg: MSELoss(), "signal"),   # per-frame regression to a unit-peak comb
}


class FUNetTask(Task):
    name = "funet"
    ConfigType = FUNetConfig
    device_env_vars = ("FUNET_DEVICE",)

    def head_for(self, config) -> str:
        """Which output head this config's loss trains."""
        try:
            return LOSSES[config.train.loss][1]
        except KeyError:
            raise ValueError(
                f"Unknown loss: {config.train.loss!r} (expected one of {list(LOSSES)})") from None

    # ------------------------------------------------------------------ required
    def build_model(self, config) -> FUNet:
        m = config.model
        return FUNet(
            channels=m.channels,
            dilations=m.dilations,
            bottleneck_dilation=m.bottleneck_dilation,
            bottleneck_convs=m.bottleneck_convs,
            codec_convolutions=m.codec_convolutions,
            base_channels=m.base_channels,
            head=self.head_for(config),
            # Inactive under eval(), but dropout>0 shifts Sequential state_dict keys, so the
            # architecture must match the checkpoint's training config to load it.
            dropout=m.dropout,
        )

    def build_loss(self, config):
        head = self.head_for(config)   # validates the name
        print(f"Loss: {config.train.loss} (model head: {head})")
        return LOSSES[config.train.loss][0](config)

    def make_train_loader(self, config):
        return make_dataloader(config, config.data.train_dir, train=True)

    def make_val_loader(self, config):
        return make_dataloader(config, config.data.val_dir, train=False)

    # ------------------------------------------------------------------ optional
    def check_feasible(self, config) -> None:
        """Reject a network deeper than the spectrogram can support.

        FUNet halves freq and time once per level, so it needs 2**depth <= both the freq-bin
        count and the frame count (see FUNet.forward / data.__getitem__). A deep net combined
        with a small n_fft or a large hop violates that.
        """
        freq_bins, time_frames = stft_output_shape(config)
        divisor = 2 ** len(config.model.dilations)
        if divisor > freq_bins or divisor > time_frames:
            raise InfeasibleConfig(
                f"depth {len(config.model.dilations)} needs freq and time >= {divisor}, but "
                f"this config yields freq={freq_bins}, time={time_frames}")

    # ------------------------------------------------------- optimize phase only
    def suggest(self, trial, base):
        """Return a copy of ``base`` with the searched hyperparameters replaced for this trial.

        Only the fields set here are touched; every other field on ``base`` is inherited as-is.
        Keep this in sync with ``searched_fields`` (which serialises the same set back to YAML).
        """
        config = copy.deepcopy(base)
        model, train = config.model, config.train

        # -- Architecture --
        # `dilations` encodes both the network depth (its length) and the per-level dilation,
        # so we sample a depth and then one dilation per level. Depth drives the 2**depth
        # downsampling that must divide the spectrogram (enforced by check_feasible); it is
        # capped at 6 because deeper nets stop fitting the smaller n_fft / larger hops below.
        depth = trial.suggest_int("depth", 3, 6)
        model.dilations = [
            trial.suggest_categorical(f"dilation_l{i}", [1, 2, 4, 8]) for i in range(depth)
        ]
        model.bottleneck_dilation = trial.suggest_categorical("bottleneck_dilation", [1, 2, 4, 8])
        model.bottleneck_convs = trial.suggest_int("bottleneck_convs", 1, 4)
        model.codec_convolutions = trial.suggest_int("codec_convolutions", 1, 4)
        model.base_channels = trial.suggest_categorical("base_channels", [8, 12, 16, 24, 32])
        model.dropout = trial.suggest_float("dropout", 0.0, 0.5)

        # -- Spectrogram (part of the input contract, hence model config) --
        model.n_fft = trial.suggest_categorical("n_fft", [512, 1024, 2048])
        model.hop_length = trial.suggest_categorical("hop_length", [64, 128, 256, 512])

        # -- Optimisation --
        train.optimizer = trial.suggest_categorical("optimizer", list(OPTIMIZERS))
        train.learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
        train.weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-1, log=True)
        # min_lr is the cosine floor, so it is only meaningful below the peak LR. Sampling it
        # as a fraction of learning_rate guarantees min_lr < learning_rate for every trial
        # (sampling the two independently could put the floor above the peak). Only used when
        # the base config's lr_schedule is 'cosine'; harmless otherwise.
        train.min_lr = train.learning_rate * trial.suggest_float("min_lr_frac", 1e-3, 1e-1, log=True)

        # -- SpecAugment (regularisation, hence train config) --
        train.freq_mask = trial.suggest_int("freq_mask", 0, 64)   # max freq bins zeroed; 0 = off
        train.time_mask = trial.suggest_int("time_mask", 0, 8)    # max time frames zeroed; 0 = off

        return config

    def searched_fields(self, config) -> dict:
        """The searched fields of ``config``, shaped like the config YAML. Mirrors the set
        ``suggest`` assigns -- keep the two together."""
        return {
            "model": {
                "dilations": config.model.dilations,
                "bottleneck_dilation": config.model.bottleneck_dilation,
                "bottleneck_convs": config.model.bottleneck_convs,
                "codec_convolutions": config.model.codec_convolutions,
                "base_channels": config.model.base_channels,
                "dropout": config.model.dropout,
                "n_fft": config.model.n_fft,
                "hop_length": config.model.hop_length,
            },
            "train": {
                "optimizer": config.train.optimizer,
                "learning_rate": config.train.learning_rate,
                "weight_decay": config.train.weight_decay,
                "min_lr": config.train.min_lr,
                "freq_mask": config.train.freq_mask,
                "time_mask": config.train.time_mask,
            },
        }
