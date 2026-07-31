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
from typing import Callable

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

    @property
    def supports_optimize(self) -> bool:
        return type(self).suggest is not Task.suggest
