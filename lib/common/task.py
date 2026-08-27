"""The seam every model implements.

A Task supplies the handful of things that are genuinely model-specific -- the architecture,
the loss, the datasets -- and inherits the training loop, checkpointing, config handling,
inference windowing and the Optuna search from ``common``.

Required (used by the train and inference phases):
    build_model, build_loss, make_train_loader, make_val_loader

Optional, with shared defaults:
    build_optimizer, build_scheduler, check_feasible

Optional, only the optimize phase calls these:
    suggest, searched_fields

A task that does not define ``suggest``/``searched_fields`` still trains and runs inference;
it just cannot be searched, and says so clearly instead of failing somewhere obscure.
"""

from abc import ABC, abstractmethod
from typing import Callable, Optional

from torch import nn
from torch.utils.data import DataLoader

from common import optim as _optim


class Task(ABC):
    #: Short identifier, used for study names and messages.
    name: str = "task"

    #: The task's Config dataclass (a subclass of common.config.Config). The loader builds
    #: the whole nested structure from this one declaration.
    ConfigType: type

    #: Dotted config keys holding filesystem paths, resolved relative to the YAML's directory.
    path_keys: tuple = ("data.train_dir", "data.val_dir", "model_dir", "resume")

    #: Dotted config keys the optimize phase must never change, e.g. ("model.freq_crop_hz",).
    #: Every trial inherits these from the base config verbatim. Declaring one here is
    #: enforced by common.phases.optimize on each trial rather than trusted to ``suggest``:
    #: a guard written inside ``suggest`` is removed by the very edit that would break it.
    frozen_fields: tuple = ()

    #: Dotted config keys whose value changes the *scale* of the training loss, so two trials
    #: that differ in them cannot be ranked against each other by loss -- the smaller-loss one
    #: may simply have picked the easier units. Declaring one here lets the optimize phase
    #: refuse to rank on loss when the search moves it, which is the difference between a
    #: search that finds a better model and one that finds a cheaper yardstick.
    loss_scale_fields: tuple = ()

    #: Name of the submodule whose *input* is the last 2-D feature map the model builds
    #: before collapsing it to a per-frame vector, e.g. FUNet's frequency-attention pooling.
    #: Set it and the diagnostic draws that map as its own column; leave it None and the column
    #: is omitted. A name rather than a hook so a task only has to point at the boundary.
    prepool_attr: Optional[str] = None

    #: Extra environment variables that override device selection, most specific first.
    device_env_vars: tuple = ()

    # ---------------------------------------------------------------- required
    @abstractmethod
    def build_model(self, config) -> nn.Module:
        """Construct the (untrained) model described by ``config.model``."""

    @abstractmethod
    def build_loss(self, config) -> Callable:
        """Construct the loss for ``config.train.loss``."""

    @abstractmethod
    def make_train_loader(self, config) -> DataLoader:
        """Training DataLoader: shuffled, augmented, randomly cropped."""

    @abstractmethod
    def make_val_loader(self, config) -> DataLoader:
        """Validation DataLoader: deterministic, un-augmented."""

    # ---------------------------------------------------------------- optional
    def build_optimizer(self, config, model: nn.Module):
        return _optim.build_optimizer(config, model)

    def build_scheduler(self, config, optimiser):
        return _optim.build_scheduler(config, optimiser)

    def check_feasible(self, config) -> None:
        """Reject a config that cannot produce a runnable model, by raising
        ``common.errors.InfeasibleConfig``.

        Called by *both* phases: the optimize phase turns it into a pruned trial, the train
        phase into a readable error before anything is built. Raise the shared exception, not
        ``optuna.TrialPruned`` -- task code must not import optuna.
        """
        return None

    def adapt_state_dict(self, state_dict: dict, config) -> dict:
        """Adapt a resume checkpoint before it is loaded into a fresh model.

        Hook for warm-starting from weights that don't quite match the current architecture
        (e.g. tiling a 1-channel encoder across more input channels). Default: unchanged.
        """
        return state_dict

    def config_overrides(self, config) -> dict:
        """Fields to overlay onto the raw YAML when archiving ``config``.

        The train phase archives the config unchanged (``{}``). The optimize phase overrides
        this with the trial's searched fields so the emitted config is directly runnable.
        """
        return {}

    # ------------------------------------------------------- optimize phase only
    def suggest(self, trial, base):
        """Return a copy of ``base`` with this trial's searched hyperparameters applied.

        ``trial`` is an ``optuna.Trial``, but it arrives as a plain object -- implementations
        call ``trial.suggest_*`` without importing optuna.
        """
        raise NotImplementedError(
            f"task {self.name!r} defines no search space; the optimize phase needs suggest()")

    def searched_fields(self, config) -> dict:
        """The searched fields of ``config``, shaped like the config YAML (section -> fields).

        Mirrors exactly what ``suggest`` assigns; used to write the winning trial back out as
        a runnable config. Keep the two in sync.
        """
        raise NotImplementedError(
            f"task {self.name!r} defines no search space; the optimize phase needs "
            "searched_fields()")

    def make_val_scorer(self, config) -> Optional[Callable[[], object]]:
        """A factory building a fresh ``common.metrics.HRMetrics`` per validation pass, or
        None for tasks with no beat notion (the default).

        Whatever this returns is measured on *every* validation pass, logged and plotted, so it
        must stay cheap -- the FUNet scorer costs ~40 ms per epoch, i.e. seconds across a whole
        run. It never changes how a model trains: the checkpoint, early stopping and the LR
        schedule all key off validation loss regardless. It only becomes decisive when a search
        is asked to rank trials by it (``optimize --objective hr_corr``).
        """
        return None

    def period_probe(self, config):
        """A callable ``(activity, hz, bpm_range, ref_beats) -> dict | None``, or None.

        The seam for showing *why* a beat detector chose the rate it did. Beat detection in
        this repo estimates one cardiac period per call and decodes the whole signal against
        it, so a wrong period is not a scatter of wrong beats -- it is a smooth, confident,
        uniformly wrong rate that no existing panel distinguishes from a right one. The probe
        returns the autocorrelation the detector picked from, which does distinguish them.

        Lives on the task, like ``make_val_scorer``, because the detector is in ``analyze`` and
        ``common`` must not import it. Default None, which simply drops the column.
        """
        return None

    def make_input(self, config, x, src_hz):
        """The exact tensor this model is fed for one window of waveform ``x``.

        The seam that lets a continuous recording be scored *as a sequence of training-sized
        windows* rather than in one pass: build the input, run the model, and every downstream
        step -- the frame-rate output, the upsample, the beat detection -- is identical to the
        snippet path, so the figure is identical too. Contrast ``run_on_waveform``, which runs a
        whole recording and hands back a native-rate signal with no frame grid to show.
        """
        raise NotImplementedError(
            f"task {self.name!r} cannot build a model input from a waveform; it has no "
            "make_input().")

    def run_on_waveform(self, config, model, x, src_hz, device=None):
        """Run ``model`` over a raw waveform, returning a signal of the same length at
        ``src_hz`` whose peaks are the heartbeats.

        The seam for scoring a continuous recording rather than snippets. What the signal *is*
        differs per task -- a beat activity for FUNet, the separated heart sound for a
        separation model -- but both feed the same beat detector, so a caller holding one does
        not need to know which. Tasks that have no recording-level inference leave this raising.
        """
        raise NotImplementedError(
            f"task {self.name!r} cannot run on a raw waveform; it has no run_on_waveform(). "
            "Only snippet directories can be scored for it.")

    def use_snippet_dir(self, config, snippet_dir) -> None:
        """Point this task's *validation* loader at ``snippet_dir``, in place.

        What "the validation set" means is the task's own business -- a held-out directory for
        some, a tail fraction of one directory for others -- so redirecting it is too. The
        default suits a task whose val loader reads ``data.val_dir``; anything else overrides.

        Used by the standalone diagnostic, which draws whatever the val loader yields, so
        redirecting that loader is all "run this on one patient" needs to mean.
        """
        config.data.val_dir = str(snippet_dir)

    def baseline_params(self, base) -> Optional[dict]:
        """The ``suggest`` parameters that reproduce ``base``, or None to skip anchoring.

        The optimize phase enqueues this as the study's first trial, which makes the search
        answer the only question that matters -- "can it beat the config I already have?" --
        instead of reporting a winner nobody has compared against. It also hands TPE one
        known-good point to model from rather than starting cold.

        Must name every parameter ``suggest`` requests, including derived ones (a depth plus
        its per-level dilations, a ratio rather than the absolute value it scales). The
        optimize phase checks that by rebuilding the config from these params, so a parameter
        added to ``suggest`` and forgotten here fails loudly instead of leaving the "baseline"
        trial quietly sampling that dimension at random.
        """
        return None

    @property
    def supports_optimize(self) -> bool:
        return type(self).suggest is not Task.suggest
