"""Optional end-of-training diagnostic figure: the best model against real snippets.

One row per validation snippet, three columns following the inference path in order:

    left    what the model actually emits: one frame per ``hop_length`` samples, drawn as
            discrete points so the frame grid is visible. At hop 256 that is 64 ms per point.
    middle  the same signal after ``frames_to_native`` -- the upsampled waveform the beat
            detector is handed -- with the original frames overlaid as dots and each side's
            detected beats marked. This is the step where sub-frame beat timing comes from,
            so it is worth seeing: a beat mark that sits between two frame dots is timing the
            detector recovered that the frame grid alone could not represent.

            The target drawn here is the **original** heart snippet at the native rate, read
            back off disk -- not the loader's target upsampled again. Those are different
            signals: the loader's target is this one rectified and averaged into hop-sized
            bins, so round-tripping it through ``frames_to_native`` would draw a red trace
            already quantised onto the same grid as the model's, and the frame grid's cost
            would be invisible in the one panel meant to expose it. Against the true signal,
            the gap between a target beat's real position and the nearest frame dot is the
            quantisation error, drawn to scale.
    right   the two BPM traces those beat trains produce, which is literally what the HR
            correlation scores. The per-snippet r is in the title.

Over- and under-detection, smeared peaks and a raised noise floor are all legible in the first
two columns; whether that translates into an agreeing heart rate is the third.

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

from common.audio import SAMPLE_RATE, load_wav, snippet_indices
from common.device import pick_device
from common.io import atomic_save
from common.metrics import bpm_trace, trace_correlation
from common.phases.inference import frames_to_native

#: Rows per figure. The whole validation split is drawn, split across as many files as that
#: takes: matplotlib's Agg backend refuses any figure over 2**16 px on a side, which at this
#: row height is ~229 rows, and a single image that tall is unreadable anyway.
ROWS_PER_PAGE = 40

ROW_HEIGHT_IN = 2.6
TARGET_COLOUR = "#d62728"
MODEL_COLOUR = "#1f77b4"


def _predict(task, config, model, device, limit: Optional[int]):
    """Run ``model`` over the validation split, returning per-snippet (prediction, target)
    frame arrays.

    Only frame-rate arrays are kept (a few hundred floats each), never the upsampled signals:
    the whole split at native rate would be tens of MB, and each row needs its own only while
    it is being drawn. Batches come straight from the loader, so a large split is never
    materialised as one tensor.
    """
    scorer_postprocess = task.make_val_scorer(config)().postprocess
    preds, targets = [], []
    with torch.no_grad():
        for batch_in, batch_target in task.make_val_loader(config):
            out = model(batch_in.to(device))
            if scorer_postprocess is not None:
                out = scorer_postprocess(out)
            preds.extend(out.detach().float().cpu().numpy())
            targets.extend(batch_target.detach().float().cpu().numpy())
            if limit is not None and len(preds) >= limit:
                return preds[:limit], targets[:limit]
    return preds, targets


def _page_paths(out_path: str, pages: int) -> List[str]:
    """One path when everything fits on a page, else ``name-001.png``, ``name-002.png``, ..."""
    if pages == 1:
        return [out_path]
    stem, dot, ext = out_path.rpartition(".")
    stem = stem or out_path
    return [f"{stem}-{i + 1:03d}{dot}{ext}" for i in range(pages)]


def _unit(x: np.ndarray) -> np.ndarray:
    """Scale to unit peak for display only -- the two activities live on unrelated scales and
    would otherwise not be comparable on one axis."""
    peak = float(np.max(np.abs(x)))
    return x / peak if peak > 0 else x


def _sot_indices(config, scorer) -> Optional[List[int]]:
    """Validation snippet indices, in the order the loader yields them, or None.

    ``snippet_indices`` sorts, and the validation loader neither shuffles nor drops, so row i
    of the figure is snippet ``indices[i]``. None disables the original-SOT trace and the
    caller falls back to upsampling the loader's target: this is an end-of-training extra and
    must never take a completed run down with it.
    """
    if scorer.sample_rate != SAMPLE_RATE:
        # Snippets on disk are at SAMPLE_RATE; a model that scores at another rate would put
        # the two traces on different time axes. Better no trace than a misaligned one.
        return None
    try:
        return snippet_indices(config.data.val_dir)
    except (OSError, FileNotFoundError, AttributeError):
        return None


def _original_sot(snippet_dir: str, index: int, n_native: int) -> Optional[np.ndarray]:
    """The un-pooled heart snippet behind one validation target, at the native rate.

    ``clamp_min(0)`` and nothing else, because that is exactly what the loader's target is
    before the binning -- it rectifies and then averages each ``hop_length`` window (see
    funet.data.FetalPairs.__getitem__). Taking the first ``n_native`` samples matches the
    loader too: a validation crop is deterministic from sample 0, and ``n_native`` is the
    frame count times the hop, i.e. the same span the target covers.

    Read per row rather than up front: the whole split at native rate is tens of MB, and a
    row needs its own only while it is being drawn.
    """
    try:
        heart = load_wav(f"{snippet_dir}/{index}_heart.wav")[0].numpy()
    except (OSError, FileNotFoundError, ValueError):
        return None
    heart = np.clip(heart[:n_native], 0, None)
    if heart.size < n_native:      # short snippet: the loader zero-pads, so match it
        heart = np.pad(heart, (0, n_native - heart.size))
    return heart


def plot_snippet_diagnostics(
        task,
        config,
        checkpoint_path: str,
        out_path: str,
        n_snippets: Optional[int] = None,
        rows_per_page: int = ROWS_PER_PAGE,
) -> List[str]:
    """Draw the validation split through ``checkpoint_path``, every snippet by default.

    ``n_snippets`` caps the count (None = all). Output is split into pages of
    ``rows_per_page``; returns the paths written, empty if the task supplies no beat detector.
    """
    make_scorer = task.make_val_scorer(config)
    if make_scorer is None:
        print(f"Diagnostics skipped: task {task.name!r} provides no validation scorer.")
        return []

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scorer = make_scorer()
    device = pick_device(*task.device_env_vars)

    model = task.build_model(config)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device).eval()

    preds, targets = _predict(task, config, model, device, n_snippets)
    if not preds:
        print("Diagnostics skipped: the validation loader yielded no snippets.")
        return []

    pages = (len(preds) + rows_per_page - 1) // rows_per_page
    paths = _page_paths(out_path, pages)
    print(f"Diagnostics: drawing all {len(preds)} validation snippets "
          f"over {pages} page(s)...")

    # Row i is snippet indices[i]; None means the originals are unreadable and the target
    # trace falls back to the loader's target upsampled (see _sot_indices).
    indices = _sot_indices(config, scorer)
    if indices is None:
        print("  note: original snippets unavailable, drawing the upsampled target instead")

    written = []
    for page, path in enumerate(paths):
        first = page * rows_per_page
        page_preds = preds[first:first + rows_per_page]
        page_targets = targets[first:first + rows_per_page]
        page_indices = None if indices is None else indices[first:first + rows_per_page]
        _draw_page(plt, scorer, page_preds, page_targets, first, path,
                   checkpoint_path, page, pages,
                   snippet_dir=config.data.val_dir, indices=page_indices)
        written.append(path)
    return written


def _draw_page(plt, scorer, preds, targets, first_index: int, out_path: str,
               checkpoint_path: str, page: int, pages: int,
               snippet_dir: Optional[str] = None,
               indices: Optional[List[int]] = None) -> None:
    """Render one page of rows and save it."""
    rows = len(preds)
    fig, axes = plt.subplots(rows, 3, figsize=(21, ROW_HEIGHT_IN * rows), squeeze=False)

    for row, (pred_frames, target_frames) in enumerate(zip(preds, targets)):

        # The signal the detector is handed, not the frame-rate output: peaks are picked after
        # frames_to_native, so drawing only the frames would not show what was actually detected.
        native_kwargs = dict(hop_length=scorer.hop_length, model_hz=scorer.sample_rate,
                             src_hz=scorer.sample_rate, interpolation=scorer.interpolation)
        pred_native = frames_to_native(
            pred_frames, n_native=pred_frames.size * scorer.hop_length, **native_kwargs)
        # The target at native rate is the ORIGINAL snippet off disk, not the loader's
        # (already hop-pooled) target upsampled again -- see the module docstring. The
        # fallback keeps the panel populated when the snippets can't be read.
        n_native = target_frames.size * scorer.hop_length
        target_native = None
        if indices is not None:
            target_native = _original_sot(snippet_dir, indices[row], n_native)
        target_is_original = target_native is not None
        if target_native is None:
            target_native = frames_to_native(target_frames, n_native=n_native, **native_kwargs)
        seconds = np.arange(pred_native.size) / scorer.sample_rate
        # Where each model frame sits on that same axis, so the two left columns line up.
        frame_seconds = np.arange(pred_frames.size) * scorer.hop_length / scorer.sample_rate

        pred_beats = scorer.beats(pred_frames)
        ref_beats = scorer.beats(target_frames)

        pred_unit, target_unit = _unit(pred_native), _unit(target_native)
        pred_frames_unit, target_frames_unit = _unit(pred_frames), _unit(target_frames)

        # ---- left: the raw frame-rate output, before any upsampling ----
        ax = axes[row][0]
        ax.plot(frame_seconds, target_frames_unit, color=TARGET_COLOUR, lw=0.9,
                marker="o", ms=2.5, label="target (SOT)")
        ax.plot(frame_seconds, pred_frames_unit, color=MODEL_COLOUR, lw=0.9, alpha=0.8,
                marker="o", ms=2.5, label="model")
        ax.set_ylabel(f"snippet {first_index + row}")
        ax.set_title(f"model output -- {pred_frames.size} frames "
                     f"@ {1000 * scorer.hop_length / scorer.sample_rate:.0f} ms", fontsize=9)
        if row == 0:
            ax.legend(fontsize=8, loc="upper right")

        # ---- middle: after frames_to_native, i.e. what the detector sees ----
        ax = axes[row][1]
        # The original is a rectified waveform at the full rate, so it draws as a dense band
        # rather than a curve -- thin and semi-transparent so it reads as the envelope it is
        # and does not bury the model trace on top of it.
        ax.plot(seconds, target_unit, color=TARGET_COLOUR,
                lw=0.4 if target_is_original else 1.0,
                alpha=0.55 if target_is_original else 1.0)
        ax.plot(seconds, pred_unit, color=MODEL_COLOUR, lw=1.0, alpha=0.8)
        # The original frames on top of the interpolation, so the upsampling is visible.
        ax.plot(frame_seconds, target_frames_unit, color=TARGET_COLOUR, ls="none",
                marker="o", ms=2.5, alpha=0.5)
        ax.plot(frame_seconds, pred_frames_unit, color=MODEL_COLOUR, ls="none",
                marker="o", ms=2.5, alpha=0.5)
        # Beats as ticks above the traces rather than full-height rules. Against the original
        # SOT that is not cosmetic: the rectified waveform is itself a picket of narrow red
        # verticals, and red axvlines drawn through it are impossible to tell apart from the
        # signal they are marking.
        if len(ref_beats):
            ax.plot(ref_beats, np.full(len(ref_beats), 1.16), marker="v", ms=4, ls="none",
                    color=TARGET_COLOUR, clip_on=False)
        if len(pred_beats):
            ax.plot(pred_beats, np.full(len(pred_beats), 1.05), marker="v", ms=4, ls="none",
                    color=MODEL_COLOUR, clip_on=False)
        ax.set_ylim(-0.05, 1.25)
        source = "original SOT" if target_is_original else "upsampled target"
        ax.set_title(f"model upsampled to {scorer.sample_rate} Hz vs {source} + beats: "
                     f"{len(ref_beats)} target / {len(pred_beats)} model", fontsize=9)

        # ---- right: the BPM traces the score is computed from ----
        ax = axes[row][2]
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

    page_note = f"  (page {page + 1} of {pages})" if pages > 1 else ""
    fig.suptitle(f"Best model vs target -- {checkpoint_path}{page_note}", fontsize=10)
    fig.tight_layout()
    # format="png" because atomic_save hands savefig a ".png.tmp" path, whose suffix has no
    # inferable image format.
    atomic_save(lambda p: fig.savefig(p, dpi=110, format="png"), out_path)
    plt.close(fig)
    print(f"  saved {out_path} ({rows} snippets)")
