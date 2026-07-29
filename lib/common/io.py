"""Atomic file writes, config emission, and the training-curve plot.

Everything written during a run goes through ``atomic_save`` so a crash or preemption can
never leave a half-written file: a reader sees either the previous complete file or the new
one, never a truncated mix.
"""

import os
import shutil
from typing import Callable, Dict, List, Optional

import yaml


def atomic_save(write_fn: Callable[[str], None], path: str) -> None:
    """Write ``path`` via a temp file + os.replace. ``write_fn`` receives the destination
    path and does the real write. Parent directories are created as needed."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp"
    write_fn(tmp)
    os.replace(tmp, path)


def atomic_copy(src: str, dst: str) -> None:
    """Copy ``src`` to ``dst`` atomically, so a reader of ``dst`` never sees a partial copy."""
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    tmp = f"{dst}.tmp"
    shutil.copyfile(src, tmp)
    os.replace(tmp, dst)


def write_config(base_config_path: str, out_path: str,
                 overrides: Optional[Dict[str, dict]] = None) -> None:
    """Write a runnable config = the raw base YAML with ``overrides`` merged in.

    Overlaying onto the *raw* YAML rather than serialising the loaded Config object is
    deliberate: loading resolves every path to an absolute one, so a dumped Config would bake
    this machine's paths into the archive and stop being portable. The raw YAML keeps its
    relative paths and every field the loader does not model.

    ``overrides`` maps section name -> fields to replace, e.g.
    ``{"model": {"base_channels": 32}, "train": {"learning_rate": 1e-3}}``. The optimize
    phase passes the searched fields; the train phase passes nothing and simply archives the
    config that produced the checkpoint sitting next to it.
    """
    with open(base_config_path) as f:
        raw = yaml.safe_load(f)
    for section, fields in (overrides or {}).items():
        if isinstance(raw.get(section), dict):
            raw[section].update(fields)
        else:
            raw[section] = fields

    atomic_save(lambda p: _dump_yaml(raw, p), out_path)


def _dump_yaml(raw: dict, path: str) -> None:
    with open(path, "w") as f:
        yaml.safe_dump(raw, f, sort_keys=False)


def plot_training_curves(
        train_losses: List[Optional[float]],
        val_losses: List[Optional[float]],
        out_path: str,
) -> None:
    """Save a train/validation loss-vs-epoch plot to ``out_path`` (PNG).

    Imports matplotlib lazily and forces the headless Agg backend, so it works when training
    runs without a display (remote box, CI, nohup, ...).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = list(range(1, len(train_losses) + 1))
    # A None loss (e.g. an epoch that produced no batch loss) becomes nan so the line simply
    # breaks there instead of erroring.
    train = [float("nan") if v is None else v for v in train_losses]
    val = [float("nan") if v is None else v for v in val_losses]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train, label="Train", color="#1f77b4")
    ax.plot(epochs, val, label="Validation", color="#d62728")

    # Mark the lowest validation loss (the checkpoint saved as model_best.pt).
    finite = [(e, v) for e, v in zip(epochs, val) if v == v]  # v == v drops nan
    if finite:
        best_e, best_v = min(finite, key=lambda ev: ev[1])
        ax.scatter([best_e], [best_v], color="#d62728", zorder=5)
        ax.annotate(f"best: {best_v:.4f} @ epoch {best_e}", (best_e, best_v),
                    textcoords="offset points", xytext=(0, 9), ha="center", fontsize=8)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training curves")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    # Force PNG rather than letting savefig infer from the extension: atomic_save hands this a
    # ``<name>.png.tmp`` path, whose ".tmp" suffix has no inferable image format.
    fig.savefig(out_path, dpi=120, format="png")
    plt.close(fig)
