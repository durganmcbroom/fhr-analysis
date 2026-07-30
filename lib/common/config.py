"""Config dataclasses shared by every model, plus the YAML loader.

A model's config YAML has three sections plus a couple of top-level keys::

    model:      architecture + input contract (task-specific dataclass)
    train:      the optimisation loop (common.TrainConfig, extended per task)
    data:       where the snippets live and how they load (common.DataConfig, shared as-is)
    model_dir:  where checkpoints/curves/config are written
    resume:     optional checkpoint to warm-start from

The dividing line between ``model`` and ``train``: **if changing a field invalidates an
existing checkpoint, it belongs in ``model``**. FUNet's ``n_fft``/``hop_length`` are model
fields by that test (a checkpoint cannot be used without them), while ``crop_len`` is not --
a fully-convolutional net runs on any compatible length, so it is a training choice.

``DataConfig`` is deliberately shared unmodified: it describes where data lives and how it is
loaded, never anything model-specific.
"""

from dataclasses import MISSING, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import List, Optional, get_type_hints

import yaml


@dataclass(kw_only=True)
class TrainConfig:
    """The optimisation loop. Tasks subclass this to add their own knobs."""

    optimizer: str = "AdamW"        # SGD, Adam, AdamW
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    momentum: float = 0.9           # SGD only
    amsgrad: bool = False           # Adam/AdamW only
    batch_size: int = 8
    epochs: int = 40
    crop_len: int = 7               # seconds
    clip: Optional[float] = 5.0     # max gradient norm; None = no clipping
    loss: str = "mse"               # interpreted by the task's own loss registry

    # 'none' (constant LR) | 'cosine' (anneal learning_rate -> min_lr over epochs)
    # | 'plateau' (ReduceLROnPlateau on the validation loss)
    lr_schedule: str = "none"
    min_lr: float = 1e-5            # cosine floor; unused by the other schedules
    plateau_factor: float = 0.5     # plateau only: LR multiplier on each reduction
    plateau_patience: int = 3       # plateau only: epochs without improvement before reducing

    # Stop after this many epochs with no validation improvement; None = train all epochs.
    early_stop_patience: Optional[int] = None

    # Train-only input augmentation: subset of common.augment.AUGMENTATIONS. Regularisation,
    # so it lives with the other anti-overfitting knobs rather than in `data`.
    augment: List[str] = field(default_factory=list)


@dataclass(kw_only=True)
class DataConfig:
    """Where the snippets are and how they load. Shared by every model unmodified."""

    train_dir: str = ""
    # Held-out split used to select checkpoints and score search trials. Leave empty and set
    # val_fraction instead to carve the split out of train_dir by index.
    val_dir: str = ""
    val_fraction: float = 0.0   # >0: hold out this fraction of train_dir (tail) as validation
    num_workers: int = 4

    # Deterministic input transforms: subset of common.preprocess.PREPROCESSORS
    # ('bandpass' | 'normalize'). Distinct from train.augment in two ways that matter -- these
    # are not random, and they apply to EVERY split and to inference, not to training only.
    #
    # By the "does it invalidate a checkpoint" test this is really part of the input contract
    # and belongs alongside a model's n_fft. It sits here instead because it is identical for
    # every model, and a per-task copy in three ModelConfigs would be three things to keep in
    # step. The archived config next to each checkpoint records it either way, so a checkpoint
    # is never separated from the preprocessing that produced it.
    preprocess: List[str] = field(default_factory=list)


@dataclass(kw_only=True)
class Config:
    """Base config. Tasks subclass this and add a ``model`` field of their own type."""

    train: TrainConfig = field(default_factory=TrainConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model_dir: str = ""
    resume: Optional[str] = None    # checkpoint (.pt) to resume training from

    # Set by load_config to the YAML this came from. The archived config written next to each
    # checkpoint is produced by re-reading this file, so the archive keeps the original's
    # relative paths instead of this machine's absolute ones.
    source_path: Optional[str] = field(default=None, repr=False)


# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------

def _build(cls, raw, where: str = ""):
    """Recursively construct dataclass ``cls`` from a raw YAML mapping.

    Any field whose annotated type is itself a dataclass is built from the matching
    sub-mapping (or from defaults when the section is absent), which is what lets a task
    declare one ``ConfigType`` and get the whole nested structure for free. Unknown keys
    raise rather than being silently dropped -- a typo'd hyperparameter that quietly keeps
    the default is an expensive way to lose a run.
    """
    raw = raw or {}
    if not isinstance(raw, dict):
        raise TypeError(f"{where or cls.__name__}: expected a mapping, got {type(raw).__name__}")

    hints = get_type_hints(cls)
    known = {f.name for f in fields(cls)}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ValueError(
            f"unknown key(s) {unknown} in {where or cls.__name__}; "
            f"valid keys: {sorted(known)}"
        )

    kwargs = {}
    for f in fields(cls):
        ftype = hints[f.name]
        prefix = f"{where}.{f.name}" if where else f.name
        if is_dataclass(ftype):
            # Always build nested sections, so a config omitting e.g. `data:` still gets a
            # fully-populated DataConfig of defaults.
            kwargs[f.name] = _build(ftype, raw.get(f.name), prefix)
        elif f.name in raw:
            kwargs[f.name] = raw[f.name]
        elif f.default is MISSING and f.default_factory is MISSING:
            raise ValueError(f"missing required key '{prefix}'")
    return cls(**kwargs)


def _resolve_path(base_dir: Path, value: Optional[str]) -> Optional[str]:
    """Resolve ``value`` relative to ``base_dir``, or leave it absolute.

    ``base_dir / value`` yields ``value`` unchanged when it is already absolute;
    ``resolve()`` normalises ``..`` segments. ``None``/empty pass through so optional keys
    stay unset.
    """
    if not value:
        return value
    return str((base_dir / value).resolve())


def _resolve_paths(config, path_keys, base_dir: Path) -> None:
    """Resolve each dotted key (e.g. ``"data.train_dir"``) in place, relative to ``base_dir``."""
    for dotted in path_keys:
        obj = config
        *parents, attr = dotted.split(".")
        for name in parents:
            obj = getattr(obj, name)
        setattr(obj, attr, _resolve_path(base_dir, getattr(obj, attr)))


def load_config(path: str, task):
    """Load ``path`` into ``task.ConfigType``, resolving ``task.path_keys`` relative to the
    YAML's own directory (not the process CWD), so the same file works from anywhere."""
    config_path = Path(path).resolve()
    with open(config_path) as f:
        raw = yaml.safe_load(f) or {}

    config = _build(task.ConfigType, raw)
    _resolve_paths(config, task.path_keys, config_path.parent)
    config.source_path = str(config_path)
    return config
