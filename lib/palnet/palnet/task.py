"""PALNet's Task: the only glue between the model and common's phases.

Same seam FUNet and TSLNet implement, so PALNet inherits the training loop, atomic
checkpointing, config archiving, all three LR schedules and the Optuna search unchanged.
"""

import copy

import numpy as np
import torch

from common.audio import SAMPLE_RATE
from common.device import pick_device
from common.errors import InfeasibleConfig
from common.losses import CorrAmpLoss, CorrelationLoss, MSELoss, SNRLoss
from common.metrics import FETAL_BPM_RANGE, HRMetrics
from common.optim import OPTIMIZERS
from common.phases.inference import activity_postprocess
from common.preprocess import BANDPASS_HZ
from common.task import Task

from palnet import panns
from palnet.config import PALNetConfig
from palnet.data import (
    crop_samples, frame_stride, make_dataloader, model_frames, native_stride, time_downsample,
)
from palnet.model import FREEZE_MODES, PALNet

# loss name -> a config -> loss module. All of common's affine-invariant and regression
# objectives apply unchanged: PALNet emits (batch, frames) beat activity, exactly as FUNet and
# TSLNet do.
#
# 'kldiv' is deliberately absent. It needs a log_softmax head, which the other two carry as a
# 'logprob' mode; PALNet emits one raw value per frame and nothing else, because the KL head
# was already on its way out of TSLNet and a second dead code path is not worth carrying.
LOSSES = {
    "snr":      lambda cfg: SNRLoss(),
    "corr":     lambda cfg: CorrelationLoss(),
    "corr_amp": lambda cfg: CorrAmpLoss(amp_weight=cfg.train.amp_weight,
                                        beat_threshold=cfg.train.amp_beat_threshold),
    "mse":      lambda cfg: MSELoss(),
}

# An analysis unit -- a frame, or the STFT window one is built from -- is the finest thing the
# model can place a beat within, so it has to be well under one beat interval. The fastest
# plausible fetal rate is ~200 bpm = 0.3 s. Shared with tslnet.task's reasoning, and the same
# numbers.
MAX_PATCH_BEAT_FRACTION = 0.5
FASTEST_FETAL_INTERVAL = 60.0 / 200.0   # seconds
TYPICAL_FETAL_BPM = 140.0   # only for reporting how many beats a crop covers

# Rates the 4 kHz snippets may be resampled to before the backbone sees them. Each has to
# divide or be divided by SAMPLE_RATE evenly, so the crop and target grids land on whole
# samples, and each places the fetal band differently on the frozen mel scale:
#
#    4000 Hz -> 256 ms window, 22 of 64 mel bins on 100-300 Hz   (no resampling; window is long)
#    8000 Hz -> 128 ms window, 16 mel bins                       (default)
#   16000 Hz ->  64 ms window, 10 mel bins
#
# 32000 -- the checkpoint's nominal rate, where Hz would mean Hz -- is not offered: it leaves
# the band on 5 bins. That is the counter-intuitive centre of this model's design and is
# explained in palnet.data.
MODEL_RATES = [4000, 8000, 16000]

# Refuse a rate that resolves the fetal band with fewer mel bins than this. Well below the
# default's 16; the point is to reject a configuration that cannot see the signal at all.
MIN_MEL_BINS_ON_BAND = 8


def stft_window_seconds(config) -> float:
    """Duration of the backbone's fixed 1024-tap analysis window at this feeding rate."""
    return config.model.n_fft / config.model.model_hz


def mel_report(config) -> tuple[int, int]:
    """(mel bins covering the fetal band, mel bins with no signal reaching them).

    Both computed in closed form from the mel scale's own definition -- no checkpoint needed,
    which is what lets a bad rate be rejected before anything downloads 259 MB. The second
    number counts bins sitting entirely above the 4 kHz snippets' Nyquist, which after
    ``power_to_db`` are exactly -100 dB: a constant plane handed to a BatchNorm carrying
    AudioSet's statistics. See the note on ``bandpass`` in ``check_feasible``.
    """
    m = config.model
    on_band = len(panns.mel_bins_covering(*BANDPASS_HZ, m.model_hz))
    scale = panns.MEL_SR / float(m.model_hz)
    support = panns.mel_support_hz(m.mel_bins)
    dead = int(np.count_nonzero(support[:, 0] > (SAMPLE_RATE / 2) * scale))
    return on_band, dead


class PALNetTask(Task):
    name = "palnet"
    ConfigType = PALNetConfig
    device_env_vars = ("PALNET_DEVICE",)

    # Not knobs: n_fft is the length of the STFT conv kernel and mel_bins the width of the mel
    # matrix, both stored tensors. They are declared in the config only so check_feasible can
    # do its arithmetic offline, and a search that moved them would produce configs no
    # checkpoint can load. interpolation joins them because it is a readout decision -- letting
    # a search vary it lets a trial win by picking the readout that localises beats best rather
    # than the model that does.
    frozen_fields = ("model.n_fft", "model.mel_bins", "model.interpolation")

    # All three change how many frames the loss averages over, and the target is built on that
    # same grid, so a trial that merely picks a finer grid spreads the same beats over more
    # mostly-empty frames and mechanically lowers the MSE. FUNet's v33 search lost a week to
    # exactly this with hop_length. Declaring them here makes the optimize phase refuse to rank
    # on loss when the search moves them.
    loss_scale_fields = ("model.model_hz", "model.hop", "model.feature_layer")

    # The log-mel, arranged as (batch, channels, mel_bins, frames). Not a pre-pool map in
    # FUNet's sense -- PALNet's frequency collapse is a mean deep inside the backbone -- but it
    # is the map worth drawing: these are the 64 bins the pretrained convolutions see, and
    # whether a beat is visible in them is what decides whether this model can work.
    prepool_attr = "mel_view"

    # ------------------------------------------------------------------ required
    def build_model(self, config) -> PALNet:
        m = config.model
        model = PALNet(
            channels=m.channels,
            hop=m.hop,
            checkpoint=m.checkpoint,
            revision=m.revision,
            pretrained=m.pretrained,
            backbone_seed=m.backbone_seed,
            channel_mode=m.channel_mode,
            feature_layer=m.feature_layer,
            freeze=m.freeze,
            bn0_trainable=m.bn0_trainable,
            head_hidden=m.head_hidden,
            head_layers=m.head_layers,
            dropout=m.dropout,
        )

        # The config declares these so check_feasible can run without the download; now that
        # the real backbone is here, hold it to them. A mismatch means every frame count and
        # mel-coverage figure computed so far was against the wrong geometry.
        for name, declared in (("n_fft", m.n_fft), ("mel_bins", m.mel_bins)):
            actual = getattr(model.backbone, name)
            if actual != declared:
                raise ValueError(
                    f"config.model.{name} is {declared} but checkpoint {m.checkpoint!r} has "
                    f"{actual}; fix the config to match the checkpoint")

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        arm = "pretrained" if m.pretrained else f"RANDOM CONTROL seed {m.backbone_seed}"
        print(f"PALNet: {trainable:,} trainable params over {frozen:,} frozen "
              f"({m.checkpoint}, {arm}, freeze={m.freeze}, {m.channel_mode}, "
              f"tap={m.feature_layer})")
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
    def prepare_model(self, config, model, train_loader) -> None:
        """Re-estimate the backbone's BatchNorm statistics on this dataset, when asked to.

        Not the training loader that was handed over: that one is augmented and randomly
        cropped, and these are meant to be the statistics of the real input distribution rather
        than of a gained-and-noised one. A deterministic pass over the same directory is built
        here instead, shuffled so a 32-batch sample is not just the first 32 snippets.
        """
        if not config.model.bn_recalibrate:
            return
        loader = make_dataloader(config, config.data.train_dir, train=False, shuffle=True)
        model.recalibrate_bn(loader, device=pick_device(*self.device_env_vars),
                             batches=config.model.bn_recalibrate_batches)

    def check_feasible(self, config) -> None:
        """Reject a config whose front-end geometry cannot work.

        Everything here is computed from declared values only -- no checkpoint download -- so
        the optimize phase can prune a bad trial before spending anything on it.
        """
        m, data = config.model, config.data

        if m.head_layers < 1:
            raise InfeasibleConfig(
                f"model.head_layers must be at least 1, got {m.head_layers}; 1 is a linear "
                "probe straight from the frozen embeddings to the frame grid")
        if m.freeze not in FREEZE_MODES:
            raise InfeasibleConfig(
                f"model.freeze must be one of {list(FREEZE_MODES)}, got {m.freeze!r}")
        if m.channel_mode not in ("per_fiber", "stack"):
            raise InfeasibleConfig(
                f"model.channel_mode must be 'per_fiber' or 'stack', got {m.channel_mode!r}")
        if m.feature_layer not in panns.TAPS:
            raise InfeasibleConfig(
                f"model.feature_layer must be one of {list(panns.TAPS)}, "
                f"got {m.feature_layer!r}")

        if not data.val_dir and not 0 < data.val_fraction < 1:
            raise InfeasibleConfig(
                "no validation split: set data.val_dir (a held-out patient, preferred) or "
                "data.val_fraction in (0, 1) to carve one out of train_dir")

        # These are stored tensors, not settings. Caught here rather than at build_model so a
        # typo costs nothing instead of a 259 MB download.
        if m.n_fft != panns.N_FFT or m.mel_bins != panns.MEL_BINS:
            raise InfeasibleConfig(
                f"this checkpoint's front-end is fixed at n_fft {panns.N_FFT} / "
                f"{panns.MEL_BINS} mel bins (the STFT kernel *is* the windowed DFT basis); "
                f"config declares {m.n_fft} / {m.mel_bins}")

        if m.model_hz <= 0 or (SAMPLE_RATE % m.model_hz and m.model_hz % SAMPLE_RATE):
            raise InfeasibleConfig(
                f"model_hz {m.model_hz} must divide or be divided by the {SAMPLE_RATE} Hz "
                f"snippet rate evenly, so resampling and the target pooling land on whole "
                f"samples; the usable rates are {MODEL_RATES}")
        if m.hop < 1:
            raise InfeasibleConfig(f"model.hop must be at least 1, got {m.hop}")
        if (frame_stride(config) * SAMPLE_RATE) % m.model_hz:
            raise InfeasibleConfig(
                f"an output frame is {frame_stride(config)} samples at model_hz {m.model_hz}, "
                f"which is not a whole number of {SAMPLE_RATE} Hz samples; the target cannot "
                "be pooled onto that grid. Adjust hop or model_hz")

        # --- can this rate see the fetal band at all? ---
        on_band, dead = mel_report(config)
        if on_band < MIN_MEL_BINS_ON_BAND:
            raise InfeasibleConfig(
                f"at model_hz {m.model_hz} the {BANDPASS_HZ[0]:.0f}-{BANDPASS_HZ[1]:.0f} Hz "
                f"fetal band lands on only {on_band} of the {m.mel_bins} pretrained mel bins "
                f"(the filterbank maps FFT bin *index*, so the feeding rate decides where the "
                f"band sits on it). Lower model_hz to spread the band over more bins -- "
                f"{MODEL_RATES} are the usable rates")

        # --- is one analysis unit short enough to hold a single beat? ---
        window = stft_window_seconds(config)
        if window > FASTEST_FETAL_INTERVAL:
            raise InfeasibleConfig(
                f"model_hz {m.model_hz} makes the fixed {m.n_fft}-tap window {window:.3f}s, "
                f"longer than the {FASTEST_FETAL_INTERVAL:.2f}s fastest fetal beat interval; "
                "consecutive beats would fall inside one analysis window. Raise model_hz")
        if window > MAX_PATCH_BEAT_FRACTION * FASTEST_FETAL_INTERVAL:
            print(f"WARNING: the {m.n_fft}-tap window is {window * 1000:.0f} ms at model_hz "
                  f"{m.model_hz}, over {MAX_PATCH_BEAT_FRACTION:g} of the fastest beat "
                  f"interval. Unlike a patch this window overlaps and is centred, so it smears "
                  f"the envelope rather than merging beats -- but beat timing pays for it.")

        frame_seconds = frame_stride(config) / m.model_hz
        if frame_seconds > MAX_PATCH_BEAT_FRACTION * FASTEST_FETAL_INTERVAL:
            raise InfeasibleConfig(
                f"hop {m.hop} at tap {m.feature_layer} (time downsample "
                f"{time_downsample(config)}) makes one output frame {frame_seconds:.3f}s, over "
                f"{MAX_PATCH_BEAT_FRACTION:g} of the {FASTEST_FETAL_INTERVAL:.2f}s fastest "
                "fetal beat interval; adjacent beats would share a frame. Lower hop, or tap a "
                "shallower layer")

        frames = model_frames(config)
        if frames < 1:
            raise InfeasibleConfig(
                f"a {config.train.crop_len}s crop is under one output frame "
                f"({native_stride(config)} samples at {SAMPLE_RATE} Hz); lengthen crop_len, or "
                "lower hop")

        # --- the bandpass, both ways ---
        if "bandpass" in data.preprocess:
            print(f"NOTE: data.preprocess includes 'bandpass', so {m.mel_bins - on_band} of "
                  f"{m.mel_bins} mel bins receive no signal and sit at the -100 dB floor. bn0 "
                  f"carries AudioSet's running statistics and will map that constant well off "
                  f"distribution. Consider dropping 'bandpass' (the backbone was trained on "
                  f"full-band audio) or setting model.bn_recalibrate.")
        elif dead:
            print(f"NOTE: {dead} of {m.mel_bins} mel bins sit entirely above the "
                  f"{SAMPLE_RATE // 2} Hz snippet Nyquist and are at the -100 dB floor; the "
                  f"other {m.mel_bins - dead} carry real content.")

        aligned = crop_samples(config) / SAMPLE_RATE
        beats = aligned * TYPICAL_FETAL_BPM / 60.0
        print(f"Front-end: {m.model_hz} Hz, {m.n_fft}-tap window {window * 1000:.0f} ms "
              f"({m.model_hz / m.n_fft:.2f} Hz/bin), {on_band}/{m.mel_bins} mel bins on the "
              f"{BANDPASS_HZ[0]:.0f}-{BANDPASS_HZ[1]:.0f} Hz band")
        print(f"Frames: crop {config.train.crop_len:g}s -> {aligned:.3f}s aligned, "
              f"{crop_samples(config) * m.model_hz // SAMPLE_RATE // m.hop + 1} STFT frames "
              f"-> {frames} output frames of {frame_seconds * 1000:.0f} ms; ~{beats:.1f} beats")

    def make_input(self, config, x, src_hz):
        """The exact tensor PALNet is fed for ONE window: ``(channels, samples)`` at model_hz.

        Lazy import: palnet.inference imports this module.
        """
        from palnet.inference import waveform_input

        series = waveform_input(config, x, src_hz)
        stride = frame_stride(config)
        return series[..., :series.shape[-1] - series.shape[-1] % stride]

    def run_on_waveform(self, config, model, x, src_hz, device=None):
        """Beat activity over a whole recording, via the real inference path."""
        from palnet.inference import run_palnet
        return run_palnet(x, src_hz, model, config, device=device)

    def make_val_scorer(self, config):
        """Score validation by HR-trace agreement, along the real inference path.

        Identical in construction to FUNet's -- the same detector the deployed pipeline runs,
        the same postprocess and upsample -- because PALNet emits the same thing on the same
        kind of grid. The only difference is the grid's pitch: a frame is
        ``native_stride`` samples of the 4 kHz signal rather than ``hop_length``.

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
            hop_length=native_stride(config),
            sample_rate=SAMPLE_RATE,
            postprocess=postprocess,
            reference_beats=reference_beats,
            interpolation=config.model.interpolation,
        )

    # ------------------------------------------------------- optimize phase only
    def suggest(self, trial, base):
        """Return a copy of ``base`` with the searched hyperparameters replaced for this trial.

        What is not searched, deliberately: ``pretrained`` and ``checkpoint`` (an experiment
        arm and a model family -- compared as separate studies), and the front-end's frozen
        tensors (see ``frozen_fields``). What is left is the head's capacity, how much of the
        backbone is allowed to move, and the two knobs that decide what it sees.

        Keep in sync with ``searched_fields`` and ``baseline_params``.
        """
        config = copy.deepcopy(base)
        model, train = config.model, config.train

        # -- Head --
        # Depth is worth searching precisely because 1 (a linear probe) is a real hypothesis
        # here: if the frozen features are already linearly separable, the extra layers are
        # only capacity to overfit two patients with.
        model.head_layers = trial.suggest_int("head_layers", 1, 4)
        model.head_hidden = trial.suggest_categorical("head_hidden", [64, 128, 256, 512])
        model.dropout = trial.suggest_float("dropout", 0.0, 0.5)

        # -- What the backbone sees, and how much of it moves --
        model.model_hz = trial.suggest_categorical("model_hz", MODEL_RATES)
        model.hop = trial.suggest_categorical("hop", [4, 8, 16])
        model.feature_layer = trial.suggest_categorical(
            "feature_layer", list(panns.TAPS))
        model.freeze = trial.suggest_categorical("freeze", ["all", "after:3", "after:4"])
        model.bn_recalibrate = trial.suggest_categorical("bn_recalibrate", [False, True])

        # -- Optimisation --
        train.optimizer = trial.suggest_categorical("optimizer", list(OPTIMIZERS))
        # A head over frozen features tolerates (and wants) a higher LR than a net trained from
        # scratch, so this floor sits above FUNet's 1e-5. When `freeze` lets part of the
        # backbone move the same LR is applied to it, which is the usual reason a fine-tuning
        # trial goes worse than a probe -- and is the search's problem to discover.
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
                "model_hz": config.model.model_hz,
                "hop": config.model.hop,
                "feature_layer": config.model.feature_layer,
                "freeze": config.model.freeze,
                "bn_recalibrate": config.model.bn_recalibrate,
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
            "model_hz": m.model_hz,
            "hop": m.hop,
            "feature_layer": m.feature_layer,
            "freeze": m.freeze,
            "bn_recalibrate": m.bn_recalibrate,
            "optimizer": t.optimizer,
            "learning_rate": t.learning_rate,
            "weight_decay": t.weight_decay,
            "min_lr_frac": t.min_lr / t.learning_rate,
        }
