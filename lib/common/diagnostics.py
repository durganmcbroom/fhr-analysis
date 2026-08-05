"""Optional end-of-training diagnostic figure: the best model against real snippets.

One row per validation snippet, two columns:

    left   the target ("SOT") activity and the model's activity on the same axes, both at the
           native sample rate the beat detector actually sees, with each detector's beat marks
           underneath. This is where over- and under-detection, smeared peaks and a raised
           noise floor are visible at a glance.
    right  the two BPM traces those beat trains produce, which is literally what the HR
           correlation scores. The per-snippet r is in the title.

Everything is taken from the same objects the metric uses -- ``Task.make_val_scorer``'s scorer
supplies the postprocess, the upsample and the detector -- so the picture cannot drift from
the number. Nothing here is imported by the training path.

REMOVING THIS FEATURE: delete this file and the block marked ``--- diagnostics ---`` in the
model's train entry point (``grep -rn diagnostics lib/*/*/train.py jobs/``). Nothing else
refers to it.
"""

from typing import List, Optional, Tuple

import numpy as np
import torch

from common.device import pick_device
from common.io import atomic_save
from common.metrics import bpm_trace, trace_correlation
from common.phases.inference import frames_to_native

#: Rows in the figure, i.e. how many validation snippets to draw.
DEFAULT_SNIPPETS = 6

TARGET_COLOUR = "#d62728"
MODEL_COLOUR = "#1f77b4"


def _collect(loader, n: int) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """The first ``n`` (input, target) pairs from a loader, item by item."""
    inputs, targets = [], []
    for batch_in, batch_target in loader:
        for i in range(batch_in.shape[0]):
            inputs.append(batch_in[i])
            targets.append(batch_target[i])
            if len(inputs) == n:
                return inputs, targets
    return inputs, targets


def _unit(x: np.ndarray) -> np.ndarray:
    """Scale to unit peak for display only -- the two activities live on unrelated scales and
    would otherwise not be comparable on one axis."""
    peak = float(np.max(np.abs(x)))
    return x / peak if peak > 0 else x


def plot_snippet_diagnostics(
        task,
        config,
        checkpoint_path: str,
        out_path: str,
        n_snippets: int = DEFAULT_SNIPPETS,
) -> Optional[str]:
    """Draw ``n_snippets`` validation snippets through ``checkpoint_path``. Returns the path
    written, or None if the task cannot supply a beat detector (no HR column to draw)."""
    make_scorer = task.make_val_scorer(config)
    if make_scorer is None:
        print(f"Diagnostics skipped: task {task.name!r} provides no validation scorer.")
        return None

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scorer = make_scorer()
    device = pick_device(*task.device_env_vars)

    model = task.build_model(config)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device).eval()

    inputs, targets = _collect(task.make_val_loader(config), n_snippets)
    if not inputs:
        print("Diagnostics skipped: the validation loader yielded no snippets.")
        return None

    with torch.no_grad():
        outputs = model(torch.stack(inputs).to(device))
    if scorer.postprocess is not None:
        outputs = scorer.postprocess(outputs)
    outputs = outputs.detach().float().cpu().numpy()

    rows = len(inputs)
    fig, axes = plt.subplots(rows, 2, figsize=(15, 2.6 * rows), squeeze=False)

    for row, (pred_frames, target_frames) in enumerate(zip(outputs, targets)):
        target_frames = target_frames.detach().float().cpu().numpy()

        # The signal the detector is handed, not the frame-rate output: peaks are picked after
        # frames_to_native, so drawing the frames would not show what was actually detected.
        native_kwargs = dict(hop_length=scorer.hop_length, model_hz=scorer.sample_rate,
                             src_hz=scorer.sample_rate)
        pred_native = frames_to_native(
            pred_frames, n_native=pred_frames.size * scorer.hop_length, **native_kwargs)
        target_native = frames_to_native(
            target_frames, n_native=target_frames.size * scorer.hop_length, **native_kwargs)
        seconds = np.arange(pred_native.size) / scorer.sample_rate

        pred_beats = scorer.beats(pred_frames)
        ref_beats = scorer.beats(target_frames)

        # ---- left: activities, with each detector's beats marked ----
        ax = axes[row][0]
        ax.plot(seconds, _unit(target_native), color=TARGET_COLOUR, lw=1.0, label="target (SOT)")
        ax.plot(seconds, _unit(pred_native), color=MODEL_COLOUR, lw=1.0, alpha=0.8, label="model")
        for t in ref_beats:
            ax.axvline(t, color=TARGET_COLOUR, alpha=0.35, lw=0.8)
        for t in pred_beats:
            ax.axvline(t, color=MODEL_COLOUR, alpha=0.35, lw=0.8, ls="--")
        ax.set_ylabel(f"snippet {row}")
        ax.set_title(f"activity -- beats: {len(ref_beats)} target / {len(pred_beats)} model",
                     fontsize=9)
        if row == 0:
            ax.legend(fontsize=8, loc="upper right")

        # ---- right: the BPM traces the score is computed from ----
        ax = axes[row][1]
        tr, br = bpm_trace(ref_beats, scorer.bpm_range)
        tp, bp = bpm_trace(pred_beats, scorer.bpm_range)
        ax.plot(tr, br, color=TARGET_COLOUR, marker="o", ms=3, lw=1.0, label="target BPM")
        ax.plot(tp, bp, color=MODEL_COLOUR, marker="o", ms=3, lw=1.0, alpha=0.8, label="model BPM")
        r = trace_correlation(pred_beats, ref_beats, scorer.bpm_range)
        ax.set_title(f"BPM -- r = {'degenerate' if r is None else f'{r:.3f}'}", fontsize=9)
        ax.set_ylim(*scorer.bpm_range)
        ax.set_ylabel("bpm")
        if row == 0:
            ax.legend(fontsize=8, loc="upper right")

    for ax in axes[-1]:
        ax.set_xlabel("seconds")
    for row in axes:
        for ax in row:
            ax.grid(True, alpha=0.3)

    fig.suptitle(f"Best model vs target -- {checkpoint_path}", fontsize=10)
    fig.tight_layout()
    # format="png" because atomic_save hands savefig a ".png.tmp" path, whose suffix has no
    # inferable image format.
    atomic_save(lambda p: fig.savefig(p, dpi=110, format="png"), out_path)
    plt.close(fig)
    print(f"Saved snippet diagnostics to {out_path}")
    return out_path
