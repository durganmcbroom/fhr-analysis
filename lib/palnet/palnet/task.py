"""PALNet's Task: the only glue between the model and common's phases.

Same seam FUNet and TSLNet implement, so PALNet inherits the training loop, atomic
checkpointing, config archiving, all three LR schedules and the Optuna search unchanged.
"""

import copy

import numpy as np
from torch import nn

from common.audio import SAMPLE_RATE
from common.errors import InfeasibleConfig
from common.losses import CorrAmpLoss, CorrelationLoss, MSELoss, SNRLoss
from common.metrics import FETAL_BPM_RANGE, HRMetrics
from common.optim import OPTIMIZERS
from common.phases.inference import activity_postprocess
from common.preprocess import BANDPASS_HZ
from common.task import Task

from palnet.config import PALNetConfig
from palnet.data import (
    freq_crop_bins, make_dataloader, model_rows, spectrogram_shape,
)
from palnet.model import PALNet
from palnet.panns import FREQ_DOWNSAMPLE, TAPS

# loss name -> a config -> loss module. All of common's affine-invariant and regression
# objectives apply unchanged: PALNet emits (batch, frames) beat activity, exactly as FUNet does.
#
# 'kldiv' is deliberately absent. It needs a log_softmax head, which the other two carry as a
# 'logprob' mode; PALNet emits one raw value per frame and nothing else, because the KL head was
# already on its way out of TSLNet and a second dead code path is not worth carrying.
LOSSES = {
    "snr":      lambda cfg: SNRLoss(),
    "corr":     lambda cfg: CorrelationLoss(),
    "corr_amp": lambda cfg: CorrAmpLoss(amp_weight=cfg.train.amp_weight,
                                        beat_threshold=cfg.train.amp_beat_threshold),
    "mse":      lambda cfg: MSELoss(),
}

# A frame is the finest thing the model can place a beat within, so it has to be well under one
# beat interval. The fastest plausible fetal rate is ~200 bpm = 0.3 s. Same constants and same
# reasoning as tslnet.task.
MAX_PATCH_BEAT_FRACTION = 0.5
FASTEST_FETAL_INTERVAL = 60.0 / 200.0   # seconds
TYPICAL_FETAL_BPM = 140.0   # only for reporting how many beats a crop covers


class PALNetTask(Task):
    name = "palnet"
    ConfigType = PALNetConfig
    device_env_vars = ("PALNET_DEVICE",)

    # The passband is a measured choice about which rows are signal, not a knob to sweep: it
    # follows from data.preprocess's bandpass, and letting a search wander it would trade a
    # known compute saving for trials that differ in what they can see. interpolation joins it
    # because it is a readout decision -- a search allowed to vary it could win by picking the
    # readout that localises beats best rather than the model that does. Both are inherited
    # verbatim per trial and common.phases.optimize enforces that (see Task.frozen_fields).
    frozen_fields = ("model.freq_crop_hz", "model.interpolation")

    # hop_length changes what the loss *means*, not just the model: data builds the target comb
    # on the spectrogram's own frame grid, so halving the hop doubles the frame count and
    # spreads the same beats over more (mostly-empty) frames -- mechanically lowering the MSE.
    # FUNet's v33 search took exactly that loophole and returned a config that beat the
    # hand-tuned one on paper while being worse in practice.
    #
    # Note this is narrower than FUNet's, which also lists n_fft. The frame count is
    # `1 + crop_len * SAMPLE_RATE // hop_length` and n_fft does not enter it -- n_fft moves the
    # *row* count, which the loss never averages over. Listing only what actually moves the
    # scale keeps the guard honest.
    loss_scale_fields = ("model.hop_length",)

    # No prepool_attr: PALNet's frequency collapse is a mean deep inside the trunk, and the
    # diagnostic's "input" column already shows the spectrogram the model is fed.

    # ------------------------------------------------------------------ required
    def build_model(self, config) -> PALNet:
        m = config.model
        rows = model_rows(config)
        model = PALNet(
            channels=m.channels,
            freq_rows=rows,
            pretrained=m.pretrained,
            backbone_seed=m.backbone_seed,
            feature_layer=m.feature_layer,
            head_hidden=m.head_hidden,
            head_layers=m.head_layers,
            dropout=m.dropout,
        )

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        arm = "pretrained" if m.pretrained else f"RANDOM CONTROL seed {m.backbone_seed}"
        print(f"PALNet: {trainable:,} trainable params over {frozen:,} frozen "
              f"({arm}, tap={m.feature_layer}, {rows} rows)")
        return model

    def build_loss(self, config):
        try:
            factory = LOSSES[config.train.loss]
        except KeyError:
            raise ValueError(
                f"Unknown loss: {config.train.loss!r} (expected one of {list(LOSSES)}); "
                "PALNet has no log-probability head, so 'kldiv' is not available") from None
        print(f"Loss: {config.train.loss}")
        return factory(config)

    def make_train_loader(self, config):
        return make_dataloader(config, config.data.train_dir, train=True)

    def make_val_loader(self, config):
        # With no val_dir the split comes out of train_dir by index (val_fraction);
        # check_feasible has already rejected the case where neither is set.
        return make_dataloader(config, config.data.val_dir or config.data.train_dir, train=False)

    # ------------------------------------------------------------------ optional
    def check_feasible(self, config) -> None:
        """Reject a config whose spectrogram cannot be fed to the trunk.

        Everything here is computed from declared values only -- no checkpoint download -- so
        the optimize phase can prune a bad trial before spending anything on it.
        """
        m, data = config.model, config.data

        if m.head_layers < 1:
            raise InfeasibleConfig(
                f"model.head_layers must be at least 1, got {m.head_layers}; 1 is a linear "
                "probe straight from the frozen embeddings to the frame grid")
        if m.feature_layer not in TAPS:
            raise InfeasibleConfig(
                f"model.feature_layer must be one of {list(TAPS)}, got {m.feature_layer!r}")
        if not data.val_dir and not 0 < data.val_fraction < 1:
            raise InfeasibleConfig(
                "no validation split: set data.val_dir (a held-out patient, preferred) or "
                "data.val_fraction in (0, 1) to carve one out of train_dir")

        rows, frames = spectrogram_shape(config)   # raises on a malformed freq_crop_hz
        kept = rows - rows % FREQ_DOWNSAMPLE
        bin_hz = SAMPLE_RATE / m.n_fft

        if kept < FREQ_DOWNSAMPLE:
            crop = freq_crop_bins(config)
            band = (f" (from freq_crop_hz {list(m.freq_crop_hz)}, which keeps rows "
                    f"{crop[0]}-{crop[1] - 1})" if crop else "")
            raise InfeasibleConfig(
                f"the trunk halves frequency {FREQ_DOWNSAMPLE.bit_length() - 1} times, so it "
                f"needs at least {FREQ_DOWNSAMPLE} spectrogram rows, but this config yields "
                f"{rows}{band}. Widen freq_crop_hz, or raise n_fft "
                f"(at n_fft {m.n_fft} a row is {bin_hz:.2f} Hz, so {FREQ_DOWNSAMPLE} rows span "
                f"{FREQ_DOWNSAMPLE * bin_hz:.0f} Hz)")
        if frames < 1:
            raise InfeasibleConfig(
                f"a {config.train.crop_len}s crop at hop {m.hop_length} yields {frames} "
                "frames; lengthen crop_len or lower hop_length")

        frame_seconds = m.hop_length / SAMPLE_RATE
        if frame_seconds > MAX_PATCH_BEAT_FRACTION * FASTEST_FETAL_INTERVAL:
            raise InfeasibleConfig(
                f"hop {m.hop_length} makes one frame {frame_seconds:.3f}s, over "
                f"{MAX_PATCH_BEAT_FRACTION:g} of the {FASTEST_FETAL_INTERVAL:.2f}s fastest "
                "fetal beat interval; adjacent beats would share a frame. Lower hop_length")

        if "bandpass" not in data.preprocess:
            print("WARNING: data.preprocess has no 'bandpass'. freq_crop_hz drops the rows "
                  "outside the band, but nothing has attenuated maternal sound or motion "
                  "*inside* it, and the crop was measured on a bandpassed mix.")

        crop = freq_crop_bins(config)
        low = (crop[0] if crop else 0) * bin_hz
        beats = config.train.crop_len * TYPICAL_FETAL_BPM / 60.0
        print(f"Spectrogram: n_fft {m.n_fft} ({bin_hz:.2f} Hz/row), hop {m.hop_length} "
              f"({frame_seconds * 1000:.0f} ms/frame)")
        print(f"Rows: {rows} kept -> {kept} after the trunk's frequency floor, "
              f"spanning {low:.1f}-{low + kept * bin_hz:.1f} Hz")
        print(f"Frames: {config.train.crop_len}s crop -> {frames} frames, ~{beats:.1f} beats")

    def make_input(self, config, x, src_hz):
        """The exact tensor PALNet is fed for ONE window: ``(channels, rows, frames)``.

        Lazy import: palnet.inference imports this module.
        """
        from palnet.inference import spectrogram_input
        return spectrogram_input(config, x, src_hz)

    def run_on_waveform(self, config, model, x, src_hz, device=None):
        """Beat activity over a whole recording, via the real inference path."""
        from palnet.inference import run_palnet
        return run_palnet(x, src_hz, model, config, device=device)

    def make_val_scorer(self, config):
        """Score validation by HR-trace agreement, along the real inference path.

        Identical in construction to FUNet's -- the same detector the deployed pipeline runs,
        the same postprocess and upsample -- because PALNet emits the same thing on the same
        grid.

        Imported lazily and locally: ``analyze`` is the analysis application and pulls in
        matplotlib and the neossnet utils, which a plain ``--objective loss`` run has no
        business loading.
        """
        postprocess = activity_postprocess(config.train.loss)

        def detect(activity, hz):
            from analyze.data import Audio
            from analyze.hr.detect_v2 import v2_beat_detector
            time = np.arange(activity.size) / hz
            # out=None suppresses the detector's diagnostic PNG, making it pure and in-memory.
            return v2_beat_detector(Audio(time, hz, activity), FETAL_BPM_RANGE, None)["times"]

        # Shared across epochs so each target's beats are detected once, not every epoch.
        reference_beats: dict = {}
        return lambda: HRMetrics(
            detect=detect,
            hop_length=config.model.hop_length,
            sample_rate=SAMPLE_RATE,
            postprocess=postprocess,
            reference_beats=reference_beats,
            interpolation=config.model.interpolation,
        )

    def period_probe(self, config):
        """Expose the detector's period choice and how close the runner-up was.

        Lazy and local for the same reason ``make_val_scorer``'s detector is.
        """
        from analyze.hr.period import period_probe
        return period_probe

    # ------------------------------------------------------- optimize phase only
    def suggest(self, trial, base):
        """Return a copy of ``base`` with the searched hyperparameters replaced for this trial.

        Not searched, deliberately: ``pretrained`` and ``backbone_seed`` (an experiment arm,
        not a hyperparameter) and the fields in ``frozen_fields``.

        Keep in sync with ``searched_fields`` and ``baseline_params``.
        """
        config = copy.deepcopy(base)
        model, train = config.model, config.train

        # -- Head --
        # Depth is worth searching precisely because 1 (a linear probe) is a real hypothesis
        # here: if the frozen features are already linearly separable, the extra layers are
        # only capacity to overfit two patients with.
        model.head_layers = trial.suggest_int("head_layers", 1, 3)
        model.head_hidden = trial.suggest_categorical("head_hidden", [64, 128, 256, 512])
        model.dropout = trial.suggest_float("dropout", 0.0, 0.5)
        model.feature_layer = trial.suggest_categorical("feature_layer", list(TAPS))

        # -- Front-end --
        # freq_crop_hz stays frozen, so n_fft moves the row count: 512 -> 32 rows, 1024 -> 64,
        # 2048 -> 128 over the shipped band. All clear the trunk's 32-row floor.
        model.n_fft = trial.suggest_categorical("n_fft", [512, 1024, 2048])
        model.hop_length = trial.suggest_categorical("hop_length", [64, 128, 256])

        # -- Optimisation --
        # A head over frozen features tolerates (and wants) a higher LR than a net trained from
        # scratch, so this floor sits above FUNet's 1e-5.
        train.optimizer = trial.suggest_categorical("optimizer", list(OPTIMIZERS))
        train.learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True)
        train.weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-1, log=True)
        # Fraction of the peak LR, so the cosine floor can never land above the peak. Only used
        # when the base config's lr_schedule is 'cosine'; harmless otherwise.
        train.min_lr = train.learning_rate * trial.suggest_float("min_lr_frac", 1e-3, 1e-1, log=True)

        return config

    def searched_fields(self, config) -> dict:
        """The searched fields of ``config``, shaped like the config YAML. Mirrors the set
        ``suggest`` assigns -- keep the two together."""
        return {
            "model": {
                "head_layers": config.model.head_layers,
                "head_hidden": config.model.head_hidden,
                "dropout": config.model.dropout,
                "feature_layer": config.model.feature_layer,
                "n_fft": config.model.n_fft,
                "hop_length": config.model.hop_length,
            },
            "train": {
                "optimizer": config.train.optimizer,
                "learning_rate": config.train.learning_rate,
                "weight_decay": config.train.weight_decay,
                "min_lr": config.train.min_lr,
            },
        }

    def baseline_params(self, base) -> dict:
        """The ``suggest`` parameters that reproduce ``base``.

        Enqueued as the study's first trial, so the search answers the only question that
        matters -- "can it beat the config I already have?" -- instead of reporting a winner
        nobody has compared against. Must name every parameter ``suggest`` requests; the
        optimize phase rebuilds the config from these and checks.
        """
        m, t = base.model, base.train
        return {
            "head_layers": m.head_layers,
            "head_hidden": m.head_hidden,
            "dropout": m.dropout,
            "feature_layer": m.feature_layer,
            "n_fft": m.n_fft,
            "hop_length": m.hop_length,
            "optimizer": t.optimizer,
            "learning_rate": t.learning_rate,
            "weight_decay": t.weight_decay,
            "min_lr_frac": t.min_lr / t.learning_rate,
        }
