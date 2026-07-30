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
from common.task import Task

from tslnet.config import TSLNetConfig
from tslnet.data import band_bins, envelope_frames, make_dataloader
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


def feasible_hop_range(config) -> tuple[int, int]:
    """Inclusive ``(lo, hi)`` hop_length bounds for this config's crop_len, from the two
    constraints ``check_feasible`` enforces.

    They pull in opposite directions and leave far less room than it looks:

    * a smaller hop means more frames per crop, and the crop has to fit the context ->
      ``hop >= crop_len * SAMPLE_RATE / context_length``
    * a larger hop means a longer patch, and a patch has to stay well inside one beat ->
      ``hop <= MAX_PATCH_BEAT_FRACTION * FASTEST_FETAL_INTERVAL * SAMPLE_RATE / patch_length``

    At crop_len 7 with the 2.0-500m checkpoint that is [14, 18] -- which is why the search
    derives the range instead of offering a categorical. Sampling from [8, 16, 32] pruned two
    trials in three before either had built a model.

    ``lo > hi`` means no hop works at this crop_len (above ~9.6 s nothing does, since filling
    the context then forces a patch past half a beat); the caller is expected to leave the
    config alone and let ``check_feasible`` produce the real error.
    """
    lo = -(-config.train.crop_len * SAMPLE_RATE // config.model.context_length)   # ceil
    hi = int(MAX_PATCH_BEAT_FRACTION * FASTEST_FETAL_INTERVAL * SAMPLE_RATE
             / config.model.patch_length)
    return lo, hi


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
        print(f"TSLNet: {trainable:,} trainable head params over {frozen:,} frozen "
              f"backbone params ({m.checkpoint})")
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
        """Reject a config whose envelope cannot be fed to the backbone.

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

        try:
            lo, hi = band_bins(m.n_fft, m.band)
        except ValueError as e:
            raise InfeasibleConfig(str(e)) from None

        frames = envelope_frames(config)
        patches = frames // m.patch_length
        if patches < 1:
            raise InfeasibleConfig(
                f"a {config.train.crop_len}s crop at hop_length {m.hop_length} is {frames} "
                f"envelope frames, under the backbone's patch length ({m.patch_length}); "
                "lengthen crop_len or lower hop_length")
        if patches * m.patch_length > m.context_length:
            raise InfeasibleConfig(
                f"a {config.train.crop_len}s crop at hop_length {m.hop_length} is "
                f"{patches * m.patch_length} envelope frames, over the backbone's context "
                f"length ({m.context_length}); shorten crop_len or raise hop_length")

        patch_seconds = m.patch_length * m.hop_length / SAMPLE_RATE
        if patch_seconds > MAX_PATCH_BEAT_FRACTION * FASTEST_FETAL_INTERVAL:
            raise InfeasibleConfig(
                f"hop_length {m.hop_length} makes one {m.patch_length}-frame patch "
                f"{patch_seconds:.3f}s, over {MAX_PATCH_BEAT_FRACTION:g} of the "
                f"{FASTEST_FETAL_INTERVAL:.2f}s fastest fetal beat interval; adjacent beats "
                "would share an embedding. Lower hop_length")

        print(f"Envelope: {SAMPLE_RATE / m.hop_length:.0f} Hz, {frames} frames/crop "
              f"-> {patches} patches of {patch_seconds * 1000:.0f} ms, "
              f"STFT bins [{lo}, {hi}) of {m.n_fft // 2 + 1}")

    # ------------------------------------------------------- optimize phase only
    def suggest(self, trial, base):
        """Return a copy of ``base`` with the searched hyperparameters replaced for this trial.

        The backbone is frozen and never searched -- there is one checkpoint and no
        architecture to vary. What is left is the head's capacity, the envelope front-end that
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

        # -- Envelope front-end (part of the input contract, hence model config) --
        # hop_length sets both the frame rate and the patch duration, so it is squeezed
        # between the context length and the beat interval; see feasible_hop_range. Sampling
        # inside the derived range means no trial is spent on a config check_feasible would
        # reject on sight. When the range is empty the base value is kept and check_feasible
        # raises the real error -- suggest() runs outside objective()'s InfeasibleConfig
        # handler, so it must not raise itself.
        lo, hi = feasible_hop_range(config)
        if lo <= hi:
            model.hop_length = trial.suggest_int("hop_length", lo, hi)
        model.n_fft = trial.suggest_categorical("n_fft", [64, 128, 256])
        model.log_envelope = trial.suggest_categorical("log_envelope", [True, False])
        # Sampled as (low, span) rather than two independent edges so high > low always holds.
        band_low = trial.suggest_categorical("band_low", [0, 50, 100, 150])
        band_span = trial.suggest_categorical("band_span", [150, 200, 400, 800])
        model.band = [float(band_low), float(band_low + band_span)]

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
                "hop_length": config.model.hop_length,
                "n_fft": config.model.n_fft,
                "log_envelope": config.model.log_envelope,
                "band": config.model.band,
            },
            "train": {
                "optimizer": config.train.optimizer,
                "learning_rate": config.train.learning_rate,
                "weight_decay": config.train.weight_decay,
                "min_lr": config.train.min_lr,
            },
        }
