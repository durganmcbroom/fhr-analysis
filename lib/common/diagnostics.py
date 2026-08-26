"""Optional end-of-training diagnostic figure: the best model against real snippets.

One row per validation snippet, following the inference path left to right. Four columns for a
model that emits on a frame grid (FUNet); three for one that already emits per input sample
(SSNet and any other separation model), where the raw-output and upsampled panels would
otherwise be the same picture:

    input   the spectrogram actually fed to the network: preprocessing applied and the
            passband rows cropped, exactly as the loader hands it over, averaged across the
            input channels so one image stands for the whole stack. Both beat trains are
            marked on it, so a beat the model missed can be checked against whether anything
            was visible at that instant -- the difference between a model that cannot see a
            beat and one that sees it and mistimes it.
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
            metric scores. The title carries all three readings: the agreement fraction a
            search ranks by, the median |delta| in bpm, and Pearson r. They can disagree --
            over a snippet the rate is nearly flat, so r is decided by the worst beat or two
            while agreement reflects the whole trace (see common.metrics).

Over- and under-detection, smeared peaks and a raised noise floor are all legible in the
output columns; whether that translates into an agreeing heart rate is the last one.

Everything is taken from the same objects the metric uses -- ``Task.make_val_scorer``'s scorer
supplies the postprocess, the upsample and the detector -- so the picture cannot drift from
the number. Nothing here is imported by the training path.

Runnable on its own, so a trained model can be inspected on any snippet directory without
retraining:

    fhr-diagnose --task funet lib/funet/models/funet-v36/config.yaml \\
        --snippet-dir lib/funet/training/stereo_v13/fetal-test \\
        --out .out/diagnostics/v36-pt13.png

``--snippet-dir`` simply replaces ``data.val_dir`` for the run, which is what makes it usable
per patient: point it at one patient's snippets and the figure covers exactly that patient.

REMOVING THIS FEATURE: delete this file, the block marked ``--- diagnostics ---`` in the
model's train entry point, and the ``fhr-diagnose`` entry point in pyproject.toml
(``grep -rn diagnostics lib/*/*/train.py jobs/ pyproject.toml``). Nothing else refers to it.
"""

import argparse
import importlib
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch

from common.audio import SAMPLE_RATE, load_wav, snippet_indices
from common.device import pick_device
from common.io import atomic_save
from common.metrics import bpm_trace, snippet_hr
from common.phases.inference import frames_to_native

#: Rows per figure. The whole validation split is drawn, split across as many files as that
#: takes: matplotlib's Agg backend refuses any figure over 2**16 px on a side, which at this
#: row height is ~229 rows, and a single image that tall is unreadable anyway.
ROWS_PER_PAGE = 40

ROW_HEIGHT_IN = 2.6
#: Vertical space held back for the figure title, in inches (see _draw_page).
SUPTITLE_IN = 0.7
TARGET_COLOUR = "#d62728"
MODEL_COLOUR = "#1f77b4"


class _PrepoolCapture:
    """Records the last 2-D feature map a model builds before collapsing it to a vector.

    A forward *pre*-hook on the module named by ``Task.prepool_attr``: its input is the map,
    its output is already partway through the collapse. Channels are averaged on the way in,
    so what is kept is one ``(freq, time)`` image per item rather than the full stack.

    Costs nothing when the task names no module -- ``maps`` stays empty and the column is
    dropped -- and rides the forward pass the diagnostic was doing anyway.
    """

    def __init__(self, task, model):
        self.maps: List[np.ndarray] = []
        self.handle = None
        attr = getattr(task, "prepool_attr", None)
        module = getattr(model, attr, None) if attr else None
        if module is not None:
            self.handle = module.register_forward_pre_hook(self._capture)

    def _capture(self, module, args):
        x = args[0]
        if x.ndim == 4:                       # (batch, channels, freq, time)
            self.maps.extend(x.detach().float().cpu().numpy().mean(axis=1))

    def close(self) -> None:
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


def _predict(task, config, model, device, limit: Optional[int]):
    """Run ``model`` over the validation split, returning per-snippet
    (input spectrogram, prediction, target) arrays.

    The inputs are kept as the loader produced them -- preprocessed and passband-cropped --
    because that is precisely what the input column has to show; re-deriving them later would
    risk drawing something the model never saw. They are the bulky part (a cropped spectrogram
    is a few hundred KB per snippet, so tens of MB across a large split), but everything else
    stays frame-rate: the upsampled signals are rebuilt per row, only while it is drawn.
    Batches come straight from the loader, so the split is never one big tensor.
    """
    scorer = task.make_val_scorer(config)()
    # Both hooks, not just the output one: a task whose target carries a source axis narrows it
    # the same way the metric does, or every downstream step here gets a 2-D "frame" array.
    post_out, post_target = scorer.postprocess, scorer.target_postprocess
    inputs, preds, targets = [], [], []
    capture = _PrepoolCapture(task, model)
    with torch.no_grad():
        for batch_in, batch_target in task.make_val_loader(config):
            out = model(batch_in.to(device))
            if post_out is not None:
                out = post_out(out)
            if post_target is not None:
                batch_target = post_target(batch_target)
            inputs.extend(batch_in.detach().float().cpu().numpy())
            preds.extend(out.detach().float().cpu().numpy())
            targets.extend(batch_target.detach().float().cpu().numpy())
            if limit is not None and len(preds) >= limit:
                break
    capture.close()
    if limit is not None:
        inputs, preds, targets = inputs[:limit], preds[:limit], targets[:limit]
        capture.maps = capture.maps[:limit]
    return inputs, preds, targets, capture.maps


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



def _patient_rows(task, config, model, device, scorer, patient_dir: str,
                  window_s: Optional[float], fibers: Optional[List[str]] = None,
                  limit: Optional[int] = None):
    """Rows for a continuous recording, each window treated exactly as a training snippet.

    Returns the same ``(inputs, predictions, targets, reference beats)`` the snippet path
    returns, built the same way: ``Task.make_input`` produces the very tensor the model is fed,
    the model is called directly, and the frame-rate output goes on to the same upsample and
    the same detector. Everything downstream -- all four columns -- is therefore identical to
    the figure training draws, which is the point: the only difference between the two is where
    the audio and the reference came from.

    Two things are inherent to a recording rather than choices:

    * The reference is the **microphone SOT**, not a ``_heart.wav``: hand-marked
      ``mic_beats.npy`` when the patient has one, else the v7 detector on the band-limited mic.
      Those beat times are carried through rather than re-detected, because unlike a snippet
      target there is no activity signal they were derived from.
    * The drawn target is an impulse per SOT beat on the model's own frame grid, since the mic
      is a different sensor at a different rate and has no trace on that axis.

    ``analyze`` is imported here, not at module scope: it pulls in matplotlib, the neossnet
    utils and the whole analysis stack, and a training run must never load it.
    """
    from pathlib import Path as _Path

    from analyze.constants import ABDOMEN_FIBER_NAMES
    from analyze.data import load_data
    from analyze.hr import sot_beats
    from analyze.hr.detect_v7 import v7_beat_detector
    from analyze.sot import load_sot_no_ppg

    data = load_data(patient_dir)
    # Which fibres, in what order, is part of the model's input contract -- a 3-channel
    # checkpoint trained on ["1B","2A","2B"] cannot be handed all five, and a different three
    # means channel i is a different sensor than it was in training. Default to the first
    # `model.channels` in canonical order, the list analyze.funet_runner uses.
    if fibers:
        missing = [n for n in fibers if n not in data.abdomen]
        if missing:
            raise SystemExit(f"--fibers {missing} not in this recording "
                             f"(have {list(data.abdomen)})")
        names = list(fibers)
    else:
        names = [n for n in ABDOMEN_FIBER_NAMES if n in data.abdomen]
        wanted = getattr(config.model, "channels", None)
        if wanted:
            names = names[:wanted]

    stacked = np.stack([data.abdomen[n].data for n in names]).astype(np.float32)
    hz = data.abdomen[names[0]].hz
    time = np.asarray(data.abdomen[names[0]].time)
    print(f"  fibers: {names} @ {hz} Hz, {stacked.shape[-1] / hz:.1f} s")

    sot = sot_beats(v7_beat_detector, out=_Path(patient_dir) / ".diagnostics",
                    data_dir=patient_dir)(load_sot_no_ppg()(patient_dir))
    ref_all = np.asarray(sot.mic_beats, dtype=float)
    print(f"  SOT: {ref_all.size} fetal beats from the microphone")

    # The microphone is typically far shorter than the fibre recording -- 20-40% of it on the
    # Banner sessions -- and past its end there is no reference to score against, so those
    # windows would cost a forward pass each and draw a row with an empty target.
    total_s = stacked.shape[-1] / hz
    mic_end = float(np.asarray(sot.mic.time)[-1]) if sot.mic is not None else None
    if mic_end is not None:
        usable = max(0.0, mic_end - float(time[0]))
        if usable < total_s:
            print(f"  mic covers {usable:.0f}s of {total_s:.0f}s; scoring stops there "
                  f"(no reference beyond it)")
            total_s = usable
    if total_s <= 0:
        raise SystemExit("The microphone does not overlap the fibre recording; nothing to score.")

    # One row per training-sized crop by default, so a row is the same extent the model was
    # trained on and the figure is directly comparable to the one training draws.
    span = float(window_s) if window_s else float(config.train.crop_len)
    n_rows = max(1, int(total_s // span))
    if limit is not None and limit < n_rows:
        n_rows = limit
    print(f"  {n_rows} window(s) of {span:g}s")

    inputs, preds, targets, ref_beats = [], [], [], []
    capture = _PrepoolCapture(task, model)
    model.eval()
    with torch.no_grad():
        for row in range(n_rows):
            lo = row * span
            a, b = int(lo * hz), int((lo + span) * hz)
            chunk = stacked[:, a:b]
            if chunk.shape[-1] < int(span * hz):
                break                       # a short tail cannot make a full-length window

            x = task.make_input(config, chunk, hz)
            out = model(x[None, ...].to(device))
            if scorer.postprocess is not None:
                out = scorer.postprocess(out)
            frames = out[0].detach().float().cpu().numpy()

            # Only the span the model actually saw. make_input floors the time axis to the
            # network's downsampling factor, so a 7 s request becomes 6.656 s at hop 256;
            # counting reference beats past that would mark beats missed from audio the model
            # was never shown.
            covered = frames.size * scorer.hop_length / scorer.sample_rate
            offset = time[a] if a < time.size else lo
            in_window = ref_all[(ref_all >= offset) & (ref_all < offset + covered)] - offset

            # The reference on the model's own frame grid, so the target column and the
            # upsample behave exactly as they do for a snippet target.
            target = np.zeros_like(frames)
            idx = (in_window * scorer.sample_rate / scorer.hop_length).astype(int)
            idx = idx[(idx >= 0) & (idx < target.size)]
            target[idx] = 1.0

            inputs.append(x.detach().float().cpu().numpy())
            preds.append(frames)
            targets.append(target)
            ref_beats.append(in_window)

    capture.close()
    print(f"  scored {len(preds)} window(s), "
          f"{sum(len(r) for r in ref_beats)} SOT beats in total")
    return inputs, preds, targets, ref_beats, capture.maps


def plot_snippet_diagnostics(
        task,
        config,
        checkpoint_path: str,
        out_path: str,
        n_snippets: Optional[int] = None,
        rows_per_page: int = ROWS_PER_PAGE,
        patient_dir: Optional[str] = None,
        window_s: Optional[float] = None,
        fibers: Optional[List[str]] = None,
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

    # A patient directory is a continuous recording with a microphone reference; a snippet
    # directory is paired files with their own targets. Both reduce to the same per-row data.
    forced_refs = None
    if patient_dir:
        inputs, preds, targets, forced_refs, maps = _patient_rows(
            task, config, model, device, scorer, patient_dir, window_s, fibers, n_snippets)
    else:
        inputs, preds, targets, maps = _predict(task, config, model, device, n_snippets)
    if not preds:
        print("Diagnostics skipped: nothing to draw.")
        return []

    pages = (len(preds) + rows_per_page - 1) // rows_per_page
    paths = _page_paths(out_path, pages)
    print(f"Diagnostics: drawing all {len(preds)} validation snippets "
          f"over {pages} page(s)...")

    # Row i is snippet indices[i]; None means the originals are unreadable and the target
    # trace falls back to the loader's target upsampled (see _sot_indices).
    indices = None if patient_dir else _sot_indices(config, scorer)
    if indices is None and not patient_dir:
        print("  note: original snippets unavailable, drawing the upsampled target instead")

    written = []
    for page, path in enumerate(paths):
        first = page * rows_per_page
        page_inputs = inputs[first:first + rows_per_page]
        page_preds = preds[first:first + rows_per_page]
        page_targets = targets[first:first + rows_per_page]
        page_indices = None if indices is None else indices[first:first + rows_per_page]
        page_refs = None if forced_refs is None else forced_refs[first:first + rows_per_page]
        page_maps = maps[first:first + rows_per_page] if maps else None
        _draw_page(plt, scorer, page_inputs, page_preds, page_targets, first, path,
                   checkpoint_path, page, pages,
                   snippet_dir=config.data.val_dir, indices=page_indices,
                   forced_refs=page_refs, row_label="window" if patient_dir else "snippet",
                   maps=page_maps, probe=task.period_probe(config))
        written.append(path)
    return written


#: At or below this many rows, the "input" panel is a stack of channels rather than a
#: spectrogram, and is drawn as overlaid waveforms. A spectrogram with this few frequency bins
#: would carry nothing worth looking at, so there is no ambiguous case in practice.
MAX_WAVEFORM_ROWS = 8


def _draw_page(plt, scorer, inputs, preds, targets, first_index: int, out_path: str,
               checkpoint_path: str, page: int, pages: int,
               snippet_dir: Optional[str] = None,
               indices: Optional[List[int]] = None,
               forced_refs: Optional[List[np.ndarray]] = None,
               row_label: str = "snippet",
               maps: Optional[List[np.ndarray]] = None,
               probe=None) -> None:
    """Render one page of rows and save it."""
    rows = len(preds)
    # A model that already emits one value per input sample (hop_length 1, e.g. a separation
    # model) has no frame grid distinct from the sample grid, so the "raw frames" panel would
    # be a pixel-for-pixel copy of the one beside it. Drop it rather than print it twice.
    has_frame_grid = scorer.hop_length > 1
    has_maps = bool(maps)
    # Columns follow the data flow, and each is present only when it says something: the
    # pre-pool map when the task exposes one, the raw-frames panel only when the frame grid
    # differs from the sample grid.
    order = (["input"] + (["prepool"] if has_maps else [])
             + (["frames"] if has_frame_grid else []) + ["native", "bpm"]
             + (["period"] if probe is not None else []))
    at = {name: i for i, name in enumerate(order)}
    ncols = len(order)
    col_frames = at.get("frames")
    col_native, col_bpm = at["native"], at["bpm"]
    col_period = at.get("period")
    fig, axes = plt.subplots(rows, ncols, figsize=(7 * ncols, ROW_HEIGHT_IN * rows),
                             squeeze=False)

    for row, (spec, pred_frames, target_frames) in enumerate(zip(inputs, preds, targets)):

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
        # A recording's reference beats come from the microphone SOT and were never detected
        # from `target_frames`, so re-detecting them would measure the impulse train we drew
        # rather than the truth it stands for.
        ref_beats = (scorer.beats(target_frames) if forced_refs is None
                     else np.asarray(forced_refs[row], dtype=float))

        pred_unit, target_unit = _unit(pred_native), _unit(target_native)
        pred_frames_unit, target_frames_unit = _unit(pred_frames), _unit(target_frames)

        # ---- input: the spectrogram the network is actually given ----
        # Averaged over the channel axis: FUNet takes one row per fibre and they are hard to
        # read stacked, while the mean still shows where energy sits in time and frequency.
        # extent puts it on the same seconds axis as every other column, so a beat mark lines
        # up across the row. Rows are labelled as bins rather than Hz because the passband crop
        # has already removed the low bins, and this module cannot know the offset that applied.
        ax = axes[row][0]
        image = spec.mean(axis=0) if spec.ndim == 3 else spec
        channels = spec.shape[0] if spec.ndim == 3 else 1
        duration = pred_frames.size * scorer.hop_length / scorer.sample_rate
        # Not every model is fed a spectrogram: a separation model takes the waveform straight,
        # which arrives here as a single row. Drawn as an image that is a meaningless smear of
        # colour, so draw whatever it actually is.
        if image.shape[0] > MAX_WAVEFORM_ROWS:
            ax.imshow(image, origin="lower", aspect="auto", cmap="magma",
                      extent=(0.0, duration, 0.0, float(image.shape[0])))
            ax.set_title(f"model input -- {image.shape[0]} freq bins x "
                         f"{image.shape[1]} frames, mean of {channels} ch", fontsize=9)
        else:
            # A handful of rows is a channel stack, not a spectrogram -- PALNet is fed the
            # fibres as raw waveforms, and drawing three of them as an image is a meaningless
            # smear that the "freq bins" title above would then misdescribe. Overlaid, so a
            # fibre that has gone quiet or saturated is visible against the others.
            in_seconds = np.linspace(0.0, duration, image.shape[1])
            for ch in range(image.shape[0]):
                ax.plot(in_seconds, image[ch], lw=0.4, alpha=0.75)
            what = "waveform" if image.shape[0] == 1 else f"{image.shape[0]} channels"
            ax.set_title(f"model input -- {what}, {image.shape[1]} samples", fontsize=9)
        for t in ref_beats:
            ax.axvline(t, color=TARGET_COLOUR, alpha=0.55, lw=0.8)
        for t in pred_beats:
            ax.axvline(t, color=MODEL_COLOUR, alpha=0.55, lw=0.8, ls="--")
        ax.set_ylabel(f"{row_label} {first_index + row}")

        # ---- pre-pool: the last map that still has a frequency axis ----
        if has_maps:
            ax = axes[row][at["prepool"]]
            fmap = maps[row]
            ax.imshow(fmap, origin="lower", aspect="auto", cmap="viridis",
                      extent=(0.0, duration, 0.0, float(fmap.shape[0])))
            for t in ref_beats:
                ax.axvline(t, color=TARGET_COLOUR, alpha=0.45, lw=0.8)
            for t in pred_beats:
                ax.axvline(t, color=MODEL_COLOUR, alpha=0.45, lw=0.8, ls="--")
            ax.set_title(f"pre-pool feature map -- {fmap.shape[0]} x {fmap.shape[1]}, "
                         f"collapsed over freq into the output", fontsize=9)

        # ---- output: the raw frame-rate output, before any upsampling ----
        if col_frames is not None:
            ax = axes[row][col_frames]
            ax.plot(frame_seconds, target_frames_unit, color=TARGET_COLOUR, lw=0.9,
                    marker="o", ms=2.5, label="target (SOT)")
            ax.plot(frame_seconds, pred_frames_unit, color=MODEL_COLOUR, lw=0.9, alpha=0.8,
                    marker="o", ms=2.5, label="model")
            ax.set_title(f"model output -- {pred_frames.size} frames "
                         f"@ {1000 * scorer.hop_length / scorer.sample_rate:.0f} ms", fontsize=9)
            if row == 0:
                ax.legend(fontsize=8, loc="upper right")

        # ---- middle: after frames_to_native, i.e. what the detector sees ----
        ax = axes[row][col_native]
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
        # "upsampled" is only true when there was a frame grid to lift off; at hop 1 the model
        # already emitted this, and claiming otherwise would misdescribe the panel.
        what = (f"model upsampled to {scorer.sample_rate} Hz" if has_frame_grid
                else f"model output @ {scorer.sample_rate} Hz")
        ax.set_title(f"{what} vs {source} + beats: "
                     f"{len(ref_beats)} target / {len(pred_beats)} model", fontsize=9)

        # ---- right: the BPM traces the score is computed from ----
        ax = axes[row][col_bpm]
        tr, br = bpm_trace(ref_beats, scorer.bpm_range)
        tp, bp = bpm_trace(pred_beats, scorer.bpm_range)
        ax.plot(tr, br, color=TARGET_COLOUR, marker="o", ms=3, lw=1.0, label="target BPM")
        ax.plot(tp, bp, color=MODEL_COLOUR, marker="o", ms=3, lw=1.0, alpha=0.8, label="model BPM")
        # All three, so the headline number can be checked against the picture: agreement is
        # what a search ranks by, median|d| is the reading in bpm, r is the legacy statistic
        # (and the one that disagrees with your eye when the rate is flat).
        hr = snippet_hr(pred_beats, ref_beats, scorer.bpm_range, scorer.tolerance_bpm)
        if hr is None:
            title = "BPM -- degenerate (no comparable trace)"
        else:
            r_text = "n/a" if hr.corr is None else f"{hr.corr:+.3f}"
            title = (f"BPM -- within{scorer.tolerance_bpm:.0f} {hr.within_tol:.2f}, "
                     f"median|d| {hr.median_delta:.1f} bpm, r {r_text}")
        ax.set_title(title, fontsize=9)
        ax.set_ylim(*scorer.bpm_range)
        ax.set_ylabel("bpm")
        if row == 0:
            ax.legend(fontsize=8, loc="upper right")

        # ---- period: which rate the detector locked onto, and by how little it won ----
        # Beat detection here estimates ONE cardiac period per call and decodes the whole
        # window against it, so a wrong period is a uniformly wrong rate rather than a scatter
        # of wrong beats -- indistinguishable from a right one in every panel to the left. This
        # is the panel where they differ: a tall runner-up at the reference rate means the
        # detector had the right answer and lost it on a hair.
        if col_period is not None:
            ax = axes[row][col_period]
            info = probe(pred_native, float(scorer.sample_rate), scorer.bpm_range, ref_beats)
            if info is None:
                ax.set_title("period -- not enough signal to autocorrelate", fontsize=9)
            else:
                ax.plot(info["lags_bpm"], info["ac"], color=MODEL_COLOUR, lw=1.0)
                ax.axvline(info["chosen_bpm"], color=MODEL_COLOUR, lw=1.2, ls="--",
                           label=f"detector {info['chosen_bpm']:.0f}")
                bits = [f"chose {info['chosen_bpm']:.0f}"]
                if info["ref_bpm"] is not None:
                    ax.axvline(info["ref_bpm"], color=TARGET_COLOUR, lw=1.2,
                               label=f"target {info['ref_bpm']:.0f}")
                    bits.append(f"target {info['ref_bpm']:.0f}")
                    ratio = info["chosen_bpm"] / info["ref_bpm"]
                    if not 0.85 <= ratio <= 1.15:
                        bits.append(f"** {ratio:.2f}x OFF **")
                    if not info["ref_in_band"]:
                        bits.append("target OUTSIDE the searched band")
                if info["margin"] is not None:
                    # Negative is not a weak runner-up, it is a different statement: the
                    # activity is ANTI-correlated at the reference period, i.e. whatever the
                    # model is emitting peaks where the beats are not.
                    bits.append(f"runner-up {info['margin']:.0%}" if info["margin"] >= 0
                                else f"ANTI-correlated at target ({info['margin']:.0%})")
                ax.set_title("period -- " + ", ".join(bits), fontsize=9)
                ax.set_xlabel("bpm")
                ax.set_ylabel("autocorr (rel. peak)")
                ax.set_ylim(min(0.0, float(np.min(info["ac"]))), 1.05)
                if row == 0:
                    ax.legend(fontsize=8, loc="upper right")

    for ax in axes[-1]:
        if ax is not None and getattr(ax, "get_xlabel", lambda: "")() != "bpm":
            ax.set_xlabel("seconds")
    image_cols = 1 + (1 if has_maps else 0)     # input, and the pre-pool map when present
    for row in axes:
        for ax in row[image_cols:]:   # not the images: a grid over one is just noise
            ax.grid(True, alpha=0.3)

    page_note = f"  (page {page + 1} of {pages})" if pages > 1 else ""
    # Both the reserved strip and the title's own position are computed from the figure height
    # rather than left at matplotlib's defaults, which are fractions: on a 40-row page the
    # figure is ~100 in tall, so the default y=0.98 puts the title two inches down, on top of
    # the first row's column headings. Working in inches keeps it the same at any row count.
    headroom = SUPTITLE_IN / (ROW_HEIGHT_IN * rows)
    fig.suptitle(f"Best model vs target -- {checkpoint_path}{page_note}",
                 fontsize=10, y=1.0 - 0.4 * headroom, va="center")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 1.0 - headroom))
    # format="png" because atomic_save hands savefig a ".png.tmp" path, whose suffix has no
    # inferable image format.
    atomic_save(lambda p: fig.savefig(p, dpi=110, format="png"), out_path)
    plt.close(fig)
    print(f"  saved {out_path} ({rows} snippets)")


# --------------------------------------------------------------------------------------
# Standalone CLI
# --------------------------------------------------------------------------------------

#: task name -> (module, class). Imported lazily inside main() only: this module is pulled in
#: by the training path, which must not drag every model package (and, through the FUNet task,
#: the analysis stack) into a run that never asks for a diagnostic.
TASKS = {
    "funet": ("funet.task", "FUNetTask"),
    "ssnet": ("ssnet.task", "SSNetTask"),
    "tslnet": ("tslnet.task", "TSLNetTask"),
    "palnet": ("palnet.task", "PALNetTask"),
}

DEFAULT_PLOT = "snippet_diagnostics.png"


def _build_task(name: str):
    module_name, class_name = TASKS[name]
    try:
        module = importlib.import_module(module_name)
    except ImportError as e:
        raise SystemExit(f"Cannot import {module_name} for --task {name}: {e}")
    return getattr(module, class_name)()


def parse_args(argv):
    p = argparse.ArgumentParser(
        prog="fhr-diagnose", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config", type=Path, help="config describing the model to inspect")
    p.add_argument("--task", choices=sorted(TASKS), default="funet",
                   help="which model the config belongs to (default: funet)")
    p.add_argument("--snippet-dir", type=Path, default=None,
                   help="snippets to draw, replacing the config's validation set")
    p.add_argument("--patient-dir", type=Path, default=None,
                   help="a continuous recording to draw instead of snippets, e.g. "
                        "Banner_data/Banner_test_20251220/PT13_1. The model runs over the "
                        "fibres via its real recording-level inference and is scored against "
                        "the microphone SOT (hand-marked mic_beats.npy if present, else v7).")
    p.add_argument("--fibers", default=None, metavar="A,B,C",
                   help="abdomen fibres to stack, in the order the model was trained on "
                        "(default: the first model.channels of "
                        + ",".join(["1B", "2A", "2B", "2C", "2D"]) + ")")
    p.add_argument("--window", type=float, default=None, metavar="SECONDS",
                   help="split a --patient-dir recording into rows this long "
                        "(default: the whole recording as one row)")
    p.add_argument("--model-dir", type=Path, default=None,
                   help="directory holding the checkpoint, overriding the config's model_dir "
                        "(relative to the CWD, not to the config file)")
    p.add_argument("--checkpoint", type=Path, default=None,
                   help="exact weights file, overriding --model-dir")
    p.add_argument("--out", type=Path, default=None,
                   help=f"output png (default: <model-dir>/{DEFAULT_PLOT}). Paginated into "
                        f"-001, -002 ... when the split needs more than one page.")
    p.add_argument("--snippets", type=int, default=None,
                   help="cap the number of snippets drawn (default: all of them)")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(sys.argv[1:] if argv is None else argv)

    # Imported here rather than at module scope so the training path never pays for them.
    from common.config import load_config
    from common.phases.train import BEST_MODEL

    task = _build_task(args.task)
    config = load_config(str(args.config), task)

    # Same resolution order as fhr-mine-failures: an exact file, else BEST_MODEL under an
    # explicit dir, else under the config's own model_dir.
    if args.checkpoint:
        checkpoint = Path(args.checkpoint)
    else:
        model_dir = Path(args.model_dir) if args.model_dir else Path(config.model_dir)
        checkpoint = model_dir / BEST_MODEL
    if not checkpoint.is_file():
        source = ("--checkpoint" if args.checkpoint else
                  "--model-dir" if args.model_dir else
                  f"the config model_dir {config.model_dir!r}")
        raise SystemExit(
            f"No checkpoint at '{checkpoint}'.\n  Resolved from: {source}\n"
            f"  Point --model-dir at the directory holding {BEST_MODEL}, or --checkpoint at "
            f"the file itself.")

    # Redirecting the task's own validation loader is all that "run this on one patient"
    # requires -- no second loader, no duplicate of the preprocessing chain. How to redirect it
    # differs per task, hence the hook (ssnet splits a fraction off train_dir rather than
    # keeping a val_dir at all).
    if args.snippet_dir and args.patient_dir:
        raise SystemExit("--snippet-dir and --patient-dir are alternatives; pass one.")
    for flag, value in (("--snippet-dir", args.snippet_dir), ("--patient-dir", args.patient_dir)):
        if value and not Path(value).is_dir():
            raise SystemExit(f"{flag} '{value}' is not a directory.")
    if args.snippet_dir:
        task.use_snippet_dir(config, args.snippet_dir)

    out = Path(args.out) if args.out else checkpoint.parent / DEFAULT_PLOT
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"Diagnosing {args.task} on {args.patient_dir or config.data.val_dir}")
    written = plot_snippet_diagnostics(
        task, config, checkpoint_path=str(checkpoint), out_path=str(out),
        n_snippets=args.snippets,
        patient_dir=str(args.patient_dir) if args.patient_dir else None,
        window_s=args.window,
        fibers=args.fibers.split(',') if args.fibers else None)
    if not written:
        raise SystemExit(
            f"Nothing drawn. Task {task.name!r} provides no validation scorer "
            f"(Task.make_val_scorer), which this figure needs for its beat detection.")


if __name__ == "__main__":
    main()
