"""TSLNet's Task: the only glue between the model and common's phases.

Same seam FUNet implements, so TSLNet inherits the training loop, atomic checkpointing,
config archiving, all three LR schedules and the Optuna search unchanged.
"""

import copy

from torch import nn

from common.audio import SAMPLE_RATE
from common.errors import InfeasibleConfig
from common.losses import CorrAmpLoss, CorrelationLoss, MSELoss, SNRLoss
from common.optim import OPTIMIZERS
from common.preprocess import BANDPASS_HZ
from common.task import Task

from tslnet.config import TSLNetConfig
from tslnet.data import make_dataloader, model_steps
from tslnet.model import TSLNet

# loss name -> (config -> loss module, matching model output head), identical to FUNet's table:
# both models emit (batch, frames) beat activity, so the same objectives and the same
# loss<->head coupling apply. Single source of truth for that coupling.
LOSSES = {
    "kldiv":    (lambda cfg: nn.KLDivLoss(reduction="batchmean"), "logprob"),
    "snr":      (lambda cfg: SNRLoss(), "signal"),
    "corr":     (lambda cfg: CorrelationLoss(), "signal"),
    "corr_amp": (lambda cfg: CorrAmpLoss(amp_weight=cfg.train.amp_weight,
                                         beat_threshold=cfg.train.amp_beat_threshold), "signal"),
    "mse":      (lambda cfg: MSELoss(), "signal"),
}

# A patch is the finest thing the model can place a beat within, so it has to be well under
# one beat interval. The fastest plausible fetal rate is ~200 bpm = 0.3 s; a patch longer than
# this fraction of that starts merging adjacent beats into one embedding.
MAX_PATCH_BEAT_FRACTION = 0.5
FASTEST_FETAL_INTERVAL = 60.0 / 200.0   # seconds
TYPICAL_FETAL_BPM = 140.0   # only for reporting how many beats a crop covers

# Warn when Nyquist is under this multiple of the passband top. An anti-alias filter needs a
# transition band, so sitting right on the edge quietly costs signal: decimating to 600 Hz
# leaves a 295 Hz component at 0.59 amplitude even though 295 < 300 = Nyquist.
NYQUIST_HEADROOM = 1.25


# Model rates that divide 4 kHz evenly (so decimation and target pooling are exact) AND keep
# Nyquist clear of the fetal band with headroom. The list is short because 4000 = 2^5 * 5^3
# leaves only 800, 1000 and 2000 as divisors at or above 800, and each rate implies the crop
# that exactly fills the context:
#
#     800 Hz -> 2.56 s, 6.0 beats, 40 ms patches      (default: most beats with a clean band)
#    1000 Hz -> 2.05 s, 4.8 beats, 32 ms patches
#
# 2000 Hz divides evenly and has ample Nyquist, but its context covers 1.02 s -- 2.4 beats --
# which is too little rhythm to be worth the finer patches. Below 800 nothing is legal: 500 Hz
# divides evenly but its 250 Hz Nyquist cuts into the band, and 600 Hz neither divides 4000
# nor keeps 295 Hz intact (it survives at 0.59 amplitude).
MODEL_RATES = [800, 1000]


def context_filling_crop(config, model_hz: int) -> float:
    """The crop length, in seconds, whose decimated length is exactly ``context_length`` steps.

    The backbone costs the same whether the context is full or half empty, so there is no
    reason to run it short -- and a crop rounded to whole seconds would leave up to a fifth of
    the context unused at 800 Hz. This is why TSLNetTrainConfig widens crop_len to a float.
    """
    return config.model.context_length / model_hz


class TSLNetTask(Task):
    name = "tslnet"
    ConfigType = TSLNetConfig
    device_env_vars = ("TSLNET_DEVICE",)

    def head_for(self, config) -> str:
        """Which output head this config's loss trains."""
        try:
            return LOSSES[config.train.loss][1]
        except KeyError:
            raise ValueError(
                f"Unknown loss: {config.train.loss!r} (expected one of {list(LOSSES)})") from None

    # ------------------------------------------------------------------ required
    def build_model(self, config) -> TSLNet:
        m = config.model
        model = TSLNet(
            channels=m.channels,
            checkpoint=m.checkpoint,
            pretrained=m.pretrained,
            backbone_seed=m.backbone_seed,
            head_hidden=m.head_hidden,
            head_layers=m.head_layers,
            dropout=m.dropout,
            head=self.head_for(config),
        )

        # The config declares these so check_feasible can run without the download; now that
        # the real backbone is here, hold it to them. A mismatch means every frame/patch sum
        # computed so far was against the wrong geometry.
        for name, declared in (("context_length", m.context_length),
                               ("patch_length", m.patch_length)):
            actual = getattr(model, name)
            if actual != declared:
                raise ValueError(
                    f"config.model.{name} is {declared} but checkpoint {m.checkpoint!r} has "
                    f"{actual}; fix the config to match the checkpoint")

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        arm = "pretrained" if m.pretrained else f"RANDOM CONTROL seed {m.backbone_seed}"
        print(f"TSLNet: {trainable:,} trainable head params over {frozen:,} frozen "
              f"backbone params ({m.checkpoint}, {arm})")
        return model

    def build_loss(self, config):
        head = self.head_for(config)   # validates the name
        print(f"Loss: {config.train.loss} (model head: {head})")
        return LOSSES[config.train.loss][0](config)

    def make_train_loader(self, config):
        return make_dataloader(config, config.data.train_dir, train=True)

    def make_val_loader(self, config):
        # With no val_dir the split comes out of train_dir by index (val_fraction);
        # check_feasible has already rejected the case where neither is set.
        return make_dataloader(config, config.data.val_dir or config.data.train_dir, train=False)

    # ------------------------------------------------------------------ optional
    def check_feasible(self, config) -> None:
        """Reject a config whose decimated waveform cannot be fed to the backbone.

        Everything here is computed from declared values only -- no checkpoint download -- so
        the optimize phase can prune a bad trial before spending anything on it.
        """
        m, data = config.model, config.data

        if m.head_layers < 1:
            raise InfeasibleConfig(
                f"model.head_layers must be at least 1, got {m.head_layers}; 1 is a linear "
                "probe straight from the patch embeddings to the frame grid")

        if not data.val_dir and not 0 < data.val_fraction < 1:
            raise InfeasibleConfig(
                "no validation split: set data.val_dir (a held-out patient, preferred) or "
                "data.val_fraction in (0, 1) to carve one out of train_dir")

        if m.model_hz <= 0 or SAMPLE_RATE % m.model_hz:
            raise InfeasibleConfig(
                f"model_hz {m.model_hz} must divide the {SAMPLE_RATE} Hz snippet rate evenly, "
                f"so decimation and the target pooling land on whole samples; valid rates "
                f"near the usable range are {MODEL_RATES}")

        # The whole reason the front-end is a decimated waveform rather than a spectrogram is
        # that Nyquist permits it. Enforce that it actually does.
        nyquist = m.model_hz / 2
        if "bandpass" in config.data.preprocess:
            top = BANDPASS_HZ[1]
            if nyquist <= top:
                raise InfeasibleConfig(
                    f"model_hz {m.model_hz} has Nyquist {nyquist:.0f} Hz, at or below the "
                    f"{top:.0f} Hz top of the bandpass; decimation would alias the fetal band "
                    f"away. Raise model_hz (try {MODEL_RATES[0]})")
            if nyquist < top * NYQUIST_HEADROOM:
                print(f"WARNING: Nyquist {nyquist:.0f} Hz is close to the {top:.0f} Hz band "
                      f"top; the anti-alias filter attenuates content near the edge.")
        else:
            print("WARNING: data.preprocess has no 'bandpass'. The waveform front-end does no "
                  "band-limiting of its own -- decimation only removes content above "
                  f"{nyquist:.0f} Hz, so maternal sounds and motion below the fetal band "
                  "reach the model unchanged.")

        steps = model_steps(config)
        patches = steps // m.patch_length
        if patches < 1:
            raise InfeasibleConfig(
                f"a {config.train.crop_len}s crop at model_hz {m.model_hz} is {steps} steps, "
                f"under the backbone's patch length ({m.patch_length}); lengthen crop_len or "
                "raise model_hz")
        if patches * m.patch_length > m.context_length:
            raise InfeasibleConfig(
                f"a {config.train.crop_len}s crop at model_hz {m.model_hz} is "
                f"{patches * m.patch_length} steps, over the backbone's context length "
                f"({m.context_length}); shorten crop_len (to "
                f"{context_filling_crop(config, m.model_hz):.2f}s it fits exactly) or lower "
                "model_hz")

        patch_seconds = m.patch_length / m.model_hz
        if patch_seconds > MAX_PATCH_BEAT_FRACTION * FASTEST_FETAL_INTERVAL:
            raise InfeasibleConfig(
                f"model_hz {m.model_hz} makes one {m.patch_length}-step patch "
                f"{patch_seconds:.3f}s, over {MAX_PATCH_BEAT_FRACTION:g} of the "
                f"{FASTEST_FETAL_INTERVAL:.2f}s fastest fetal beat interval; adjacent beats "
                "would share an embedding. Raise model_hz")

        beats = config.train.crop_len * TYPICAL_FETAL_BPM / 60.0
        print(f"Waveform: {m.model_hz} Hz (Nyquist {nyquist:.0f}), {steps} steps/crop "
              f"-> {patches} patches of {patch_seconds * 1000:.0f} ms; "
              f"{config.train.crop_len:.2f}s ~ {beats:.1f} beats, "
              f"{patches * m.patch_length / m.context_length:.0%} of the context")

    # ------------------------------------------------------- optimize phase only
    def suggest(self, trial, base):
        """Return a copy of ``base`` with the searched hyperparameters replaced for this trial.

        The backbone is frozen and never searched -- there is one checkpoint and no
        architecture to vary. What is left is the head's capacity, the model rate that
        decides what the backbone even sees, and the usual optimisation knobs.
        Keep in sync with ``searched_fields``.
        """
        config = copy.deepcopy(base)
        model, train = config.model, config.train

        # -- Head --
        # Depth is worth searching precisely because 1 (a linear probe) is a real hypothesis
        # here, not a degenerate corner: if the frozen features are already linearly separable,
        # the extra layers are just capacity to overfit two patients with.
        model.head_layers = trial.suggest_int("head_layers", 1, 4)
        model.head_hidden = trial.suggest_categorical("head_hidden", [64, 128, 256, 512])
        model.dropout = trial.suggest_float("dropout", 0.0, 0.5)

        # -- Waveform front-end (part of the input contract, hence model config) --
        # model_hz trades context against patch resolution, and crop_len follows from it: only
        # the rates in MODEL_RATES are legal at all (they divide 4 kHz and clear Nyquist), and
        # for each one there is exactly one crop that fills the context. Sampling the pair
        # together means no trial is spent on a config check_feasible would reject on sight.
        model.model_hz = trial.suggest_categorical("model_hz", MODEL_RATES)
        train.crop_len = context_filling_crop(config, model.model_hz)

        # -- Optimisation --
        train.optimizer = trial.suggest_categorical("optimizer", list(OPTIMIZERS))
        # A head over frozen features tolerates (and wants) a higher LR than a net trained from
        # scratch, so this floor sits above FUNet's 1e-5.
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
            },
            "train": {
                # Derived from model_hz rather than sampled, but emitted so the written config
                # is runnable as-is instead of silently keeping the base file's crop_len.
                "crop_len": config.train.crop_len,
                "optimizer": config.train.optimizer,
                "learning_rate": config.train.learning_rate,
                "weight_decay": config.train.weight_decay,
                "min_lr": config.train.min_lr,
            },
        }
