#!/usr/bin/env python3
"""
plot_beat_buildup.py — build the fetal beat detectors up one step at a time.

Renders one figure per detector variant (v2, V9, v7). Each figure is a vertical
stack of stages: stage 1 is the raw waveform, and every stage after it is exactly
*one* transformation, ending in the detected beats. The point is to be able to
point at any stage and name the thing that was added.

All three variants open with the same stages ------------------------------------

    1. raw waveform (microphone or fiber)
    2. Chebyshev bandpass, 190-220 Hz (the fetal acoustic band)
    3. sliding-window RMS envelope

-- and then diverge --------------------------------------------------------------

    v2  light smoothing -> distance-only peak picking -> recentre each peak on its
        half-max lobe midpoint.
    V9  Gaussian smoothing -> peak picking gated on a *local* height floor (a percentile
        measured per 4 s block, so the gate steps with the loudness of each stretch rather
        than sitting at one global level) -> grow each peak to its half-maximum (FWHM)
        extent -> merge and duration-gate the regions -> region midpoints ->
        amplitude-aware IBI rejection.
    v7  resample to a normalised 100 Hz envelope -> autocorrelation for the cycle length
        and systolic interval -> Gaussian duration priors -> HSMM Viterbi over
        S1/systole/S2/diastole -> beats are the S1 segment midpoints.

THIS FILE IS DELIBERATELY STANDALONE ---------------------------------------------

It imports nothing from ``analyze`` and re-implements every step on numpy/scipy, so
the whole story reads top to bottom in one place and nothing elsewhere in the repo
can silently change these figures.

The trade-off is that it is *not* the shipped code -- the detectors in ``analyze.hr``
are the source of truth, this is the explanation of them. Three deliberate
divergences:

  * every variant uses an **RMS envelope**, where the real v2/v7 use Shannon energy
    and the real V9 uses the Hilbert analytic envelope;
  * machinery that does not earn a stage is dropped (v7's transient suppression,
    v2's disabled MAD floor, v2's Hilbert-of-the-already-smoothed-energy); and
  * parameters are stated in seconds rather than taps, so the figures look the same
    at 8 kHz and 44.1 kHz.

Beat counts here will therefore not match ``plot_peak_detectors.py``.

The input is either the microphone or one fiber channel (``--source``); everything
downstream of stage 1 is identical either way. The hand-marked overlay comes from
``mic_beats.npy``, which is marked on the *mic* timebase -- over a fiber trace it is
drawn uncorrected, so a constant shift against the detected beats is the offset
between the two acquisitions, not detector error.

Examples
--------
    # default patient and window, all three variants + the comparison figure
    python src/fhr_bin/peak_det/plot_beat_buildup.py

    # one variant, explicit window
    python src/fhr_bin/peak_det/plot_beat_buildup.py "Patient 6" --variant V9 \
        --start 200 --end 210

    # off a fiber instead of the mic (1B by default)
    python src/fhr_bin/peak_det/plot_beat_buildup.py --source fiber
    python src/fhr_bin/peak_det/plot_beat_buildup.py --source fiber --fiber 2C
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
from matplotlib import pyplot as plt
from scipy.io.wavfile import read as wav_read
from scipy.ndimage import gaussian_filter1d, uniform_filter1d
from scipy.signal import cheby1, find_peaks, sosfiltfilt

# --------------------------------------------------------------------------------
# Constants. Mirrored from analyze.constants rather than imported -- see the module
# docstring on why this file stands alone.
# --------------------------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parents[3]
DEFAULT_DATA_DIR = PROJECT_DIR / "Banner_data" / "Banner_test_20251220"
DEFAULT_PATIENT = "patient8-session1"
MIC_FILE = "microphone.wav"
MIC_BEATS_FILE = "mic_beats.npy"          # hand-marked beats from the beat-marking app
OUT_DIR = Path(__file__).resolve().parent / "out"

# Fiber bundles. Column 0 of each .npy is time; the rest are channels. This mirrors
# analyze.data.load_fibers: 1B lives in the ps4000 (chest device) bundle, 2A-2D in the
# ps3000a (abdomen device) bundle. Chest (ps4000 col 1) is maternal and not offered
# here -- the fetal band would be the wrong filter for it.
FIBER_BUNDLE_A = "ps4000.npy"
FIBER_BUNDLE_B = "ps3000a.npy"
FIBER_CHANNELS = {
    "1B": (FIBER_BUNDLE_A, 2),
    "2A": (FIBER_BUNDLE_B, 1),
    "2B": (FIBER_BUNDLE_B, 2),
    "2C": (FIBER_BUNDLE_B, 3),
    "2D": (FIBER_BUNDLE_B, 4),
}
DEFAULT_FIBER = "1B"

FETAL_BAND_HZ = (190.0, 220.0)            # fetal acoustic band
BP_ORDER = 3
BP_RIPPLE_DB = 1.0
FETAL_BPM_RANGE = (90.0, 280.0)

RMS_WIN_S = 0.100                         # shared envelope window (all three variants)

# v2
V2_SMOOTH_S = 0.0125                      # = the original's 100-tap moving average at 8 kHz
V2_ENERGY_RANGE = 0.5                     # lobe edge = this fraction of the peak

# V9
V9_SIGMA_S = 0.020                        # Gaussian smoothing sigma
V9_HEIGHT_PCT = 20.0                      # percentile of the block: the local baseline estimate
V9_FLOOR_SCALE = 1                     # multiplier on that percentile -- THIS is the knob that
                                          # goes low. A percentile is an order statistic, so it
                                          # bottoms out at the block minimum (q=0) no matter how
                                          # small q gets; a scale factor has no lower bound. Below
                                          # ~0.3 the floor drops under the block minimum, at which
                                          # point the height gate stops rejecting anything; 0.0
                                          # disables it outright.
V9_FLOOR_WINDOW_S = 4.0                   # window the gate is measured in (local, not global)
V9_HALF_MAX_LIMIT_S = 0.120               # cap on how far a peak may grow to half-max
V9_MERGE_GAP_S = 0.040                    # regions closer than this are merged
V9_MIN_DUR_S, V9_MAX_DUR_S = 0.040, 0.300  # physiological region duration gate

# v7
V7_FEAT_FS = 100.0                        # feature grid for the HSMM
V7_NORM_PCT = (10.0, 95.0)                # robust envelope scaling
V7_M_S1_S, V7_M_S2_S = 0.090, 0.070       # mean S1 / S2 sound durations
V7_SD_S1_S, V7_SD_S2_S = 0.020, 0.020
V7_SD_SYS_S, V7_SD_DIA_S = 0.030, 0.045
S1, SYS, S2, DIA = 0, 1, 2, 3
STATE_NAMES = ["S1", "systole", "S2", "diastole"]
NEG = -1e18

# Palette
C_RAW = "0.45"
C_FILT = "#c0392b"
C_ENV = "tab:purple"
C_SMOOTH = "tab:blue"
C_PEAK = "k"
C_BEAT = "tab:green"
C_REJECT = "tab:red"
C_GT = "black"
C_GHOST = "0.75"
_STATE_COLORS = {S1: "tab:green", SYS: "0.88", S2: "tab:orange", DIA: "0.96"}


# --------------------------------------------------------------------------------
# I/O
# --------------------------------------------------------------------------------

def resolve_patient_dir(patient: str) -> Path:
    """Accept a bare patient name (under DEFAULT_DATA_DIR) or a full path."""
    candidates = [Path(patient), DEFAULT_DATA_DIR / patient]
    for c in candidates:
        if (c / MIC_FILE).exists():
            return c
    tried = "\n  ".join(str(c / MIC_FILE) for c in candidates)
    raise FileNotFoundError(f"Could not find {MIC_FILE} for {patient!r}. Tried:\n  {tried}")


def load_mic(path: Path, start: float, end: float) -> Tuple[np.ndarray, np.ndarray, float]:
    """Return ``(t, x, fs)`` for ``microphone.wav`` cropped to ``[start, end]``."""
    fs, arr = wav_read(str(path / MIC_FILE))
    arr = np.asarray(arr, dtype=float)
    if arr.ndim > 1:
        arr = arr[:, 0]
    fs = float(fs)
    i0 = max(0, int(round(start * fs)))
    i1 = min(len(arr), int(round(end * fs)))
    if i1 - i0 < 16:
        raise ValueError(f"window {start}-{end}s is empty or too short for this file")
    x = arr[i0:i1]
    t = np.arange(i0, i1) / fs
    return t, x, fs


def load_fiber(path: Path, channel: str, start: float, end: float
               ) -> Tuple[np.ndarray, np.ndarray, float]:
    """Return ``(t, x, fs)`` for one fiber channel cropped to ``[start, end]``.

    The bundle is memory-mapped and sliced rather than read whole: these are 9M-row
    files and a build-up window is a few seconds of them.
    """
    if channel not in FIBER_CHANNELS:
        raise ValueError(f"unknown fiber {channel!r}; choose from "
                         f"{', '.join(FIBER_CHANNELS)}")
    fname, col = FIBER_CHANNELS[channel]
    fp = path / fname
    if not fp.exists():
        raise FileNotFoundError(f"fiber {channel} needs {fp}, which does not exist")

    arr = np.load(fp, mmap_mode="r")
    t0, t1 = float(arr[0, 0]), float(arr[1, 0])
    fs = 1.0 / (t1 - t0)
    i0 = max(0, int(round((start - t0) * fs)))
    i1 = min(arr.shape[0], int(round((end - t0) * fs)))
    if i1 - i0 < 16:
        raise ValueError(f"window {start}-{end}s is empty or too short for {fp.name}")
    return (np.asarray(arr[i0:i1, 0], dtype=float),
            np.asarray(arr[i0:i1, col], dtype=float),
            round(fs))


def load_beats(path: Path, start: float, end: float) -> Optional[np.ndarray]:
    """Hand-marked beat times (s) inside the window, or None if not marked."""
    p = path / MIC_BEATS_FILE
    if not p.exists():
        return None
    arr = np.asarray(np.load(p, allow_pickle=False), dtype=float)
    if arr.ndim == 2 and arr.shape[1] >= 1:
        arr = arr[:, 0]
    arr = np.sort(arr.ravel()[np.isfinite(arr.ravel())])
    return arr[(arr >= start) & (arr <= end)]


# --------------------------------------------------------------------------------
# Shared DSP -- the pieces every variant is built from
# --------------------------------------------------------------------------------

def bandpass(x: np.ndarray, fs: float, lo: float, hi: float) -> np.ndarray:
    """Zero-phase Chebyshev type-I bandpass (same shape as analyze.filters.bp_filter)."""
    sos = cheby1(BP_ORDER, rp=BP_RIPPLE_DB, Wn=[lo, hi], fs=fs, btype="bandpass", output="sos")
    return sosfiltfilt(sos, x)


def rms_envelope(x: np.ndarray, fs: float, win_s: float) -> np.ndarray:
    """Sliding-window RMS. Collapses each oscillation packet into a single lobe --
    this is the step that makes peak picking mean 'find a heart sound' rather than
    'find a carrier cycle'."""
    n = max(2, int(round(win_s * fs)))
    return np.sqrt(np.maximum(uniform_filter1d(x ** 2, size=n, mode="reflect"), 0.0))


def moving_average(x: np.ndarray, fs: float, win_s: float) -> np.ndarray:
    n = max(1, int(round(win_s * fs)))
    return uniform_filter1d(x, size=n, mode="reflect") if n > 1 else x.copy()


def min_gap_samples(fs: float, bpm_max: float) -> int:
    """Minimum inter-beat spacing in samples implied by the fastest allowed rate."""
    return max(1, int(round(60.0 / bpm_max * fs)))


def block_percentile(x: np.ndarray, fs: float, pct: float, win_s: float) -> np.ndarray:
    """Percentile of ``x`` within each non-overlapping ``win_s`` block, held flat across the
    block, as a per-sample array.

    A single percentile over the whole clip is a *global* floor: one loud stretch lifts it
    for the entire recording, so quiet beats elsewhere fall under the gate and are lost.
    Measuring it per block gates each beat against its own stretch of signal instead.

    The blocks tumble rather than slide, so the floor is a step function -- it changes only
    at block edges, and the figure shows exactly which block's threshold judged each beat.
    The cost is that the gate jumps mid-signal: two beats either side of an edge are judged
    against different levels.
    """
    n = len(x)
    win = max(1, int(round(win_s * fs)))
    edges = list(range(0, n, win))
    # A short final block would take its percentile from very few samples; fold it into the
    # previous one rather than ending the clip on a noisy step.
    if len(edges) > 1 and n - edges[-1] < win // 2:
        edges.pop()

    out = np.empty(n, dtype=float)
    for i, lo in enumerate(edges):
        hi = edges[i + 1] if i + 1 < len(edges) else n
        out[lo:hi] = np.percentile(x[lo:hi], pct)
    return out


def median_bpm(times: Sequence[float]) -> float:
    times = np.asarray(times, dtype=float)
    if times.size < 2:
        return float("nan")
    return 60.0 / float(np.median(np.diff(times)))


def _fmt_t(v: float) -> str:
    """Window bound for filenames and titles. ``:g`` rather than ``:.0f`` so a
    sub-second window keeps its precision -- otherwise 100-100.5s and 100-100.4s
    both render as '100-100' and quietly overwrite each other."""
    return f"{v:g}"


def _bpm_note(times: Sequence[float], label: str = "beats") -> str:
    times = np.asarray(times, dtype=float)
    if times.size < 2:
        return f"{times.size} {label}"
    return f"{times.size} {label}   median {median_bpm(times):.1f} BPM"


# --------------------------------------------------------------------------------
# Figure framework
# --------------------------------------------------------------------------------

@dataclass
class Stage:
    """One transformation. ``draw`` paints it; ``aux`` marks a stage whose x axis is
    not time (an autocorrelation lag, a duration prior) so it gets its own row slot
    instead of being forced onto the shared time axis."""
    title: str
    draw: Callable[[plt.Axes], None]
    aux: bool = False


def _group_rows(stages: List[Stage]) -> List[List[int]]:
    """Consecutive aux stages share a row (up to 2); time stages get a full row."""
    rows: List[List[int]] = []
    i = 0
    while i < len(stages):
        if stages[i].aux:
            grp = []
            while i < len(stages) and stages[i].aux and len(grp) < 2:
                grp.append(i)
                i += 1
            rows.append(grp)
        else:
            rows.append([i])
            i += 1
    return rows


def render(stages: List[Stage], suptitle: str, out_path: Path, dpi: int,
           xlim: Tuple[float, float]) -> None:
    """Stack ``stages`` vertically over a shared time axis and write a PNG."""
    rows = _group_rows(stages)
    ncols = 2 if any(len(r) > 1 for r in rows) else 1

    fig = plt.figure(figsize=(16, 1.95 * len(rows) + 0.9), constrained_layout=True)
    gs = fig.add_gridspec(len(rows), ncols)

    share: Optional[plt.Axes] = None
    time_axes: List[plt.Axes] = []
    for r, idxs in enumerate(rows):
        for c, pi in enumerate(idxs):
            p = stages[pi]
            if p.aux:
                span = gs[r, c] if len(idxs) > 1 else gs[r, :]
                ax = fig.add_subplot(span)
            else:
                ax = fig.add_subplot(gs[r, :], sharex=share)
                share = share or ax
                time_axes.append(ax)
            p.draw(ax)
            ax.set_title(f"Stage {pi + 1} — {p.title}", loc="center", fontsize=9.5)
            ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.35)
            if not p.aux:
                ax.margins(x=0)
                ax.set_xlim(*xlim)

    # Only the bottom time stage carries the shared axis' ticks and label.
    for ax in time_axes[:-1]:
        ax.tick_params(labelbottom=False)
    if time_axes:
        time_axes[-1].set_xlabel("Time (s)")
    fig.suptitle(suptitle, fontsize=13)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f"saved -> {out_path}")


def _note(ax: plt.Axes, msg: str, loc: str = "upper right") -> None:
    """Numeric readout for a stage. Measured values only -- counts, rates, widths,
    thresholds. No prose: titles name the operation and its parameters, this box
    reports numbers, and nothing on a figure interprets them."""
    xy = {"upper right": (0.995, 0.95, "right", "top"),
          "upper left": (0.005, 0.95, "left", "top"),
          "lower right": (0.995, 0.05, "right", "bottom")}[loc]
    ax.text(xy[0], xy[1], msg, transform=ax.transAxes, ha=xy[2], va=xy[3], fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", alpha=0.85))


def _beats_panel(t: np.ndarray, sig: np.ndarray, beats: np.ndarray,
                 gt: Optional[np.ndarray], label: str) -> Callable[[plt.Axes], None]:
    """Final stage: detected beats drawn on the band-limited waveform the detector
    actually saw, with the hand-marked beats overlaid when the recording has them."""

    def draw(ax: plt.Axes) -> None:
        ax.plot(t, sig, lw=0.5, color=C_FILT, alpha=0.85, rasterized=True)
        lo, hi = float(np.min(sig)), float(np.max(sig))
        if len(beats):
            ax.vlines(beats, lo, hi, color=C_BEAT, lw=1.1, label=f"{label} beats")
        if gt is not None and len(gt):
            ax.plot(gt, np.full(len(gt), hi * 0.92), "v", color=C_GT, ms=5,
                    label="hand-marked")
        ax.set_ylabel("Amplitude")
        ax.legend(loc="lower right", fontsize=7.5, ncol=2)
        msg = _bpm_note(beats)
        if gt is not None and len(gt) >= 2:
            msg += f"\nhand-marked: {len(gt)} beats   median {median_bpm(gt):.1f} BPM"
        _note(ax, msg)

    return draw


# --------------------------------------------------------------------------------
# The shared spine: stages 1-5, identical for every variant
# --------------------------------------------------------------------------------

@dataclass
class Spine:
    """Everything the common stages produced, for the variants to build on."""
    t: np.ndarray
    raw: np.ndarray
    filt: np.ndarray
    env: np.ndarray
    fs: float
    stages: List[Stage]
    naive_times: np.ndarray       # stage 3: pure peak detection
    spaced_times: np.ndarray      # stage 4: + minimum spacing


def build_spine(t: np.ndarray, raw: np.ndarray, fs: float, rms_win_s: float,
                source_label: str) -> Spine:
    lo, hi = FETAL_BAND_HZ
    bpm_max = FETAL_BPM_RANGE[1]

    filt = bandpass(raw, fs, lo, hi)
    env = rms_envelope(filt, fs, rms_win_s)

    # Peak detection straight off the band-limited waveform, with and without a
    # spacing constraint. Not a stage in the per-variant figures -- these only feed
    # the comparison figure's two baseline rows.
    gap = min_gap_samples(fs, bpm_max)
    naive_idx, _ = find_peaks(filt)
    spaced_idx, _ = find_peaks(filt, distance=gap)
    naive_times, spaced_times = t[naive_idx], t[spaced_idx]

    span = max(t[-1] - t[0], 1e-9)

    def s1(ax: plt.Axes) -> None:
        ax.plot(t, raw, lw=0.5, color=C_RAW, rasterized=True)
        ax.set_ylabel("Amplitude")
        _note(ax, f"{fs / 1000:.1f} kHz   {len(raw)} samples   {span:.2f} s")

    def s2(ax: plt.Axes) -> None:
        # Filtered trace only. The raw signal is stage 1 immediately above and shares
        # this x axis, so ghosting it here just crowds the band-limited waveform.
        ax.plot(t, filt, lw=0.5, color=C_FILT, rasterized=True)
        ax.set_ylabel("Amplitude")

    def s3(ax: plt.Axes) -> None:
        ax.plot(t, np.abs(filt), lw=0.4, color=C_GHOST, alpha=0.7, rasterized=True,
                label="|filtered|")
        ax.plot(t, env, lw=1.0, color=C_ENV, label="RMS")
        ax.set_ylabel("RMS (a.u.)")
        ax.legend(loc="upper right", fontsize=7.5, ncol=2)

    stages = [
        Stage(f"Raw {source_label} signal", s1),
        Stage(f"Chebyshev bandpass  [{lo:.0f}–{hi:.0f} Hz, order {BP_ORDER}, "
              f"{BP_RIPPLE_DB:.0f} dB ripple]", s2),
        Stage(f"RMS envelope  [{rms_win_s * 1000:.0f} ms sliding window]", s3),
    ]
    return Spine(t, raw, filt, env, fs, stages, naive_times, spaced_times)


# --------------------------------------------------------------------------------
# v2 -- envelope peaks, distance only, recentred on the lobe midpoint
# --------------------------------------------------------------------------------

def build_v2(sp: Spine) -> Tuple[List[Stage], np.ndarray]:
    t, fs, env = sp.t, sp.fs, sp.env
    bpm_max = FETAL_BPM_RANGE[1]

    smooth = moving_average(env, fs, V2_SMOOTH_S)
    gap = min_gap_samples(fs, bpm_max)
    pk, _ = find_peaks(smooth, distance=gap)

    # Recentre each peak on the midpoint of its half-max lobe. The search is confined
    # to the peak's Voronoi cell (the midpoints to its neighbours): without that, a
    # quiet beat's threshold sits so low that the envelope never falls below it before
    # a louder neighbour, and the midpoint gets dragged on top of that neighbour.
    lobes: List[Tuple[int, int]] = []
    centres: List[int] = []
    n = len(pk)
    for i, p in enumerate(pk):
        bound = V2_ENERGY_RANGE * smooth[p]
        left_lim = (pk[i - 1] + p) // 2 if i > 0 else 0
        right_lim = (p + pk[i + 1]) // 2 if i < n - 1 else len(smooth)
        left = np.where(smooth[left_lim:p] <= bound)[0]
        right = np.where(smooth[p + 1:right_lim] <= bound)[0]
        if len(left) and len(right):
            li, ri = left_lim + left[-1], p + 1 + right[0]
            centres.append((li + ri) // 2)
        else:
            li = ri = p                      # lobe never closes -> trust the peak
            centres.append(p)
        lobes.append((li, ri))
    centres_idx = np.unique(centres) if centres else np.array([], dtype=int)
    beats = t[centres_idx] if len(centres_idx) else np.array([], dtype=float)

    delta_pct = 100.0 * float(np.max(np.abs(smooth - env))) / (float(np.max(env)) + 1e-12)
    widths_ms = np.array([(t[ri] - t[li]) * 1000.0 for li, ri in lobes if ri > li])

    def s6(ax: plt.Axes) -> None:
        ax.plot(t, env, lw=0.6, color=C_GHOST, label="RMS")
        ax.plot(t, smooth, lw=1.1, color=C_SMOOTH, label="smoothed")
        ax.set_ylabel("RMS (a.u.)")
        ax.legend(loc="lower right", fontsize=7.5, ncol=2)
        _note(ax, f"max |Δ| = {delta_pct:.1f}% of peak")

    def s7(ax: plt.Axes) -> None:
        ax.plot(t, smooth, lw=0.9, color=C_SMOOTH)
        if len(pk):
            ax.plot(t[pk], smooth[pk], "x", color=C_PEAK, ms=6, mew=1.2)
        ax.set_ylabel("RMS (a.u.)")
        _note(ax, _bpm_note(t[pk], "peaks"))

    def s8(ax: plt.Axes) -> None:
        ax.plot(t, smooth, lw=0.9, color=C_SMOOTH)
        for li, ri in lobes:
            if ri > li:
                ax.axvspan(t[li], t[ri], color=C_BEAT, alpha=0.18)
        if len(pk):
            ax.plot(t[pk], smooth[pk], "x", color=C_PEAK, ms=5, mew=1.0,
                    label="envelope peak")
        if len(centres_idx):
            ax.plot(t[centres_idx], smooth[centres_idx], "o", color=C_BEAT, ms=4,
                    markeredgecolor="k", markeredgewidth=0.4, label="lobe midpoint")
        ax.set_ylabel("RMS (a.u.)")
        ax.legend(loc="lower right", fontsize=7.5, ncol=2)
        _note(ax, f"{len(lobes)} lobes   {len(centres_idx)} midpoints   "
                  f"median width "
                  f"{np.median(widths_ms) if widths_ms.size else float('nan'):.0f} ms")

    stages = [
        Stage(f"Moving-average smoothing  [{V2_SMOOTH_S * 1000:.1f} ms]", s6),
        Stage(f"Peak picking  [min separation {60.0 / bpm_max * 1000:.0f} ms, "
              f"no height gate]", s7),
        Stage(f"Half-max lobe midpoint  [{V2_ENERGY_RANGE:.0%} of peak, within each "
              f"peak's Voronoi cell]", s8),
    ]
    return stages, beats


# --------------------------------------------------------------------------------
# V9 -- FWHM regions, duration gate, amplitude-aware IBI rejection
# --------------------------------------------------------------------------------

def build_V9(sp: Spine) -> Tuple[List[Stage], np.ndarray]:
    t, fs, env = sp.t, sp.fs, sp.env
    bpm_max = FETAL_BPM_RANGE[1]

    smooth = gaussian_filter1d(env, sigma=V9_SIGMA_S * fs, truncate=4.0, mode="reflect")
    gap = min_gap_samples(fs, bpm_max)
    # Local floor: the gate is the percentile of the V9_FLOOR_WINDOW_S block each sample
    # falls in, so find_peaks compares every candidate against its own stretch of signal.
    # Passing an array here is element-wise, one threshold per sample.
    height_th = V9_FLOOR_SCALE * block_percentile(smooth, fs, V9_HEIGHT_PCT, V9_FLOOR_WINDOW_S)
    pk, _ = find_peaks(smooth, distance=gap, height=height_th)

    # Grow each peak out to where the envelope drops below half its height, capped so
    # a peak sitting on a plateau cannot swallow the whole window.
    half_cap = int(np.floor(V9_HALF_MAX_LIMIT_S * fs))
    n = len(smooth)
    raw_regions = np.zeros((len(pk), 2), dtype=float)
    for k, p in enumerate(pk):
        hv = 0.5 * smooth[p]
        li = p
        while li > 0 and smooth[li] > hv and (p - li) < half_cap:
            li -= 1
        ri = p
        while ri < n - 1 and smooth[ri] > hv and (ri - p) < half_cap:
            ri += 1
        raw_regions[k] = (t[li], t[ri])

    # Merge regions that nearly touch, then keep only physiologically-sized ones.
    merged: List[List[float]] = []
    for s, e in raw_regions[np.argsort(raw_regions[:, 0])] if len(raw_regions) else []:
        if merged and (s - merged[-1][1]) <= V9_MERGE_GAP_S:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    merged_arr = np.asarray(merged, dtype=float) if merged else np.zeros((0, 2), float)
    if len(merged_arr):
        dur = merged_arr[:, 1] - merged_arr[:, 0]
        keep_mask = (dur >= V9_MIN_DUR_S) & (dur <= V9_MAX_DUR_S)
    else:
        keep_mask = np.zeros(0, dtype=bool)
    kept = merged_arr[keep_mask] if len(merged_arr) else merged_arr
    dropped = merged_arr[~keep_mask] if len(merged_arr) else merged_arr

    centres = np.mean(kept, axis=1) if len(kept) else np.zeros(0, dtype=float)

    # Amplitude-aware IBI rejection: while any pair is closer than the physiological
    # minimum, drop the quieter of the closest such pair (ties drop the earlier).
    min_ibi = 60.0 / bpm_max
    keep = np.ones(len(centres), dtype=bool)

    def amp_at(c: float) -> float:
        return float(smooth[int(np.argmin(np.abs(t - c)))])

    while True:
        idx = np.flatnonzero(keep)
        if len(idx) < 2:
            break
        ibis = np.diff(centres[idx])
        bad = np.flatnonzero(ibis < min_ibi)
        if len(bad) == 0:
            break
        worst = bad[int(np.argmin(ibis[bad]))]
        i0, i1 = idx[worst], idx[worst + 1]
        keep[i0 if amp_at(centres[i0]) <= amp_at(centres[i1]) else i1] = False
    beats = centres[keep]
    rejected = centres[~keep]

    def s6(ax: plt.Axes) -> None:
        ax.plot(t, env, lw=0.6, color=C_GHOST, label="RMS")
        ax.plot(t, smooth, lw=1.1, color=C_SMOOTH, label="smoothed")
        ax.set_ylabel("RMS (a.u.)")
        ax.legend(loc="lower right", fontsize=7.5, ncol=2)

    def s7(ax: plt.Axes) -> None:
        ax.plot(t, smooth, lw=0.9, color=C_SMOOTH)
        ax.plot(t, height_th, color=C_REJECT, lw=0.9, ls="--",
                label=f"{V9_FLOOR_SCALE:g} x {V9_HEIGHT_PCT:.0f}th pct "
                      f"per {V9_FLOOR_WINDOW_S:.0f}s block")
        if len(pk):
            ax.plot(t[pk], smooth[pk], "x", color=C_PEAK, ms=6, mew=1.2)
        ax.set_ylabel("RMS (a.u.)")
        ax.legend(loc="lower right", fontsize=7.5)
        _note(ax, _bpm_note(t[pk], "peaks"))

    def s8(ax: plt.Axes) -> None:
        ax.plot(t, smooth, lw=0.9, color=C_SMOOTH)
        for s, e in raw_regions:
            ax.axvspan(s, e, color=C_BEAT, alpha=0.20)
        if len(pk):
            ax.plot(t[pk], smooth[pk], "x", color=C_PEAK, ms=5, mew=1.0)
        ax.set_ylabel("RMS (a.u.)")
        w = (raw_regions[:, 1] - raw_regions[:, 0]) * 1000.0 if len(raw_regions) else np.array([])
        _note(ax, f"{len(raw_regions)} regions   median width "
                  f"{np.median(w) if w.size else float('nan'):.0f} ms")

    def s9(ax: plt.Axes) -> None:
        ax.plot(t, smooth, lw=0.9, color=C_SMOOTH)
        for s, e in kept:
            ax.axvspan(s, e, color=C_BEAT, alpha=0.25)
        for s, e in dropped:
            ax.axvspan(s, e, color=C_REJECT, alpha=0.25, hatch="//")
        ax.set_ylabel("RMS (a.u.)")
        _note(ax, f"{len(merged_arr)} merged   {len(kept)} kept   "
                  f"{len(dropped)} outside gate")

    def s10(ax: plt.Axes) -> None:
        ax.plot(t, smooth, lw=0.9, color=C_SMOOTH)
        lo_y, hi_y = ax.get_ylim()
        if len(beats):
            ax.vlines(beats, lo_y, hi_y, color=C_BEAT, lw=1.0, label="kept")
        if len(rejected):
            ax.vlines(rejected, lo_y, hi_y, color=C_REJECT, lw=1.4, ls="--",
                      label="rejected")
        ax.set_ylabel("RMS (a.u.)")
        ax.legend(loc="lower right", fontsize=7.5, ncol=2)
        _note(ax, f"{len(centres)} midpoints   {len(beats)} kept   "
                  f"{len(rejected)} rejected")

    stages = [
        Stage(f"Gaussian smoothing  [σ = {V9_SIGMA_S * 1000:.0f} ms]", s6),
        Stage(f"Peak picking  [min separation {60.0 / bpm_max * 1000:.0f} ms, "
              f"height ≥ {V9_FLOOR_SCALE:g} x the {V9_HEIGHT_PCT:.0f}th percentile per "
              f"{V9_FLOOR_WINDOW_S:.0f} s block]", s7),
        Stage(f"Half-maximum extent  [FWHM, capped at "
              f"{V9_HALF_MAX_LIMIT_S * 1000:.0f} ms]", s8),
        Stage(f"Region merge and duration gate  [merge < "
              f"{V9_MERGE_GAP_S * 1000:.0f} ms, keep "
              f"{V9_MIN_DUR_S * 1000:.0f}–{V9_MAX_DUR_S * 1000:.0f} ms]", s9),
        Stage(f"Region midpoints, amplitude-aware IBI rejection  "
              f"[min {min_ibi * 1000:.0f} ms, drop lower-amplitude]", s10),
    ]
    return stages, beats


# --------------------------------------------------------------------------------
# v7 -- duration-dependent HMM over the cardiac cycle
# --------------------------------------------------------------------------------

def _autocorr_cycle_systole(a: np.ndarray, fs: float, min_cycle_s: float, max_cycle_s: float
                     ) -> Tuple[np.ndarray, float, float]:
    """``(autocorrelation, cycle, systole)`` in samples.

    ``cycle`` is the S1->S1 cardiac cycle length: the dominant autocorrelation lag
    inside the physiological band. (Not "RR" -- that is the ECG R-wave-to-R-wave
    interval, and there is no R wave in an acoustic signal. The rest of this repo
    calls the beat-to-beat spacing IBI.)

    The systolic interval is the dominant lag in ``[0.30 cycle, 0.45 cycle]``, clamped
    to that band. In practice the clamp is doing all the work: the envelope
    autocorrelation has no local maximum there, so the argmax lands on a band edge --
    measured over a spread of patients, windows and channels it hit an edge every
    time, almost always the left one. Treat the returned systole as ``0.30 * cycle``
    with a nominal search around it, not as an independent measurement.
    """
    a0 = a - float(np.mean(a))
    ac = np.correlate(a0, a0, mode="full")[len(a) - 1:]
    lo, hi = int(round(min_cycle_s * fs)), min(int(round(max_cycle_s * fs)), len(ac) - 1)
    cycle = float(lo + int(np.argmax(ac[lo:hi + 1]))) if hi > lo else 0.5 * (min_cycle_s + max_cycle_s) * fs
    slo, shi = int(round(0.30 * cycle)), min(int(round(0.45 * cycle)), len(ac) - 1)
    sys = float(slo + int(np.argmax(ac[slo:shi + 1]))) if shi > slo else 0.38 * cycle
    return ac, cycle, sys


def _dur_logpdf(mean: float, std: float, fs: float, hard_min_s: float = 0.02
                ) -> Tuple[int, int, np.ndarray]:
    """Gaussian duration log-pdf over an integer sample range ``[dmin, dmax]``."""
    mean = max(mean, hard_min_s * fs)
    std = max(std, 0.012 * fs)
    dmin = max(1, int(round(mean - 2.5 * std)))
    dmax = max(dmin + 1, int(round(mean + 2.5 * std)))
    d = np.arange(dmax + 1, dtype=float)
    lp = -0.5 * ((d - mean) / std) ** 2 - np.log(std)
    lp[:dmin] = NEG
    return dmin, dmax, lp


def _build_durations(cycle: float, sys: float, fs: float
                     ) -> Tuple[List[Tuple[int, int, np.ndarray]], List[float]]:
    """Duration priors for [S1, systole, S2, diastole], in samples, plus each one's
    Gaussian mean. Systole is the S1->S2 gap minus the S1 sound; diastole is the
    remaining (longer) gap -- which is what lets the decoder tell S1 from S2 without
    ever looking at amplitude.

    The four means sum to ``cycle`` only while the ``max()`` floors stay inactive.
    Since ``sys`` is effectively ``0.30 * cycle`` (see ``_autocorr_cycle_systole``),
    the systole floor engages once ``0.30 * cycle - 90 ms < 30 ms``, i.e. above
    ~150 BPM -- well inside the fetal range. Past that the means over-run the measured
    cycle: +6% at 180 BPM, +26% at 280. The Viterbi still trades duration against
    emission, so the effect is a bias toward slower decoded rates, not a hard failure.

    The means are returned rather than recovered from ``(dmin, dmax)`` downstream:
    ``_dur_logpdf`` clamps ``dmin`` to >= 1, so for a short-mean state the support is
    asymmetric about the mean and its midpoint is not the mean.
    """
    m_s1, m_s2 = V7_M_S1_S * fs, V7_M_S2_S * fs
    m_sys = max(0.03 * fs, sys - m_s1)
    m_dia = max(0.05 * fs, cycle - sys - m_s2)
    means = [m_s1, m_sys, m_s2, m_dia]
    stds = [V7_SD_S1_S * fs, V7_SD_SYS_S * fs, V7_SD_S2_S * fs, V7_SD_DIA_S * fs]
    return [_dur_logpdf(m, s, fs) for m, s in zip(means, stds)], means


def _log_emission(a: np.ndarray) -> np.ndarray:
    """Per-sample log-emission, shape ``(T, 4)``. S1/S2 want sound present, the two
    gaps want silence. Nothing here distinguishes S1 from S2 -- that is the duration
    model's job alone."""
    p = np.empty((len(a), 4), dtype=float)
    p[:, S1] = a
    p[:, SYS] = 1.0 - a
    p[:, S2] = a
    p[:, DIA] = 1.0 - a
    return np.log(p + 1e-3)


def _viterbi_hsmm(log_e: np.ndarray, durations: List[Tuple[int, int, np.ndarray]]
                  ) -> np.ndarray:
    """Most-likely S1->systole->S2->diastole tiling of the window.

    Segments tile ``[0, T)`` exactly and follow the fixed cyclic order; each pays its
    Gaussian duration log-pdf plus the summed log-emission over its span (O(1) via a
    cumulative sum). The first segment may start in any state, since the window opens
    mid-cycle. Returns a per-sample state label.
    """
    T, Sn = log_e.shape
    csum = np.vstack([np.zeros(Sn), np.cumsum(log_e, axis=0)])       # (T+1, Sn)

    delta = np.full((T + 1, Sn), NEG)
    back_start = np.full((T + 1, Sn), -1, dtype=int)
    back_prev = np.full((T + 1, Sn), -1, dtype=int)

    for j in range(Sn):                                             # first segment
        dmin, dmax, lp = durations[j]
        for e in range(dmin, min(dmax, T) + 1):
            sc = lp[e] + (csum[e, j] - csum[0, j])
            if sc > delta[e, j]:
                delta[e, j], back_start[e, j], back_prev[e, j] = sc, 0, -1

    for e in range(1, T + 1):
        for j in range(Sn):
            i = (j - 1) % Sn                                        # only i -> j
            dmin, dmax, lp = durations[j]
            s_lo, s_hi = max(1, e - dmax), e - dmin
            if s_hi < s_lo:
                continue
            s = np.arange(s_lo, s_hi + 1)
            prev = delta[s, i]
            sc = prev + lp[e - s] + (csum[e, j] - csum[s, j])
            sc[prev <= NEG / 2] = NEG
            k = int(np.argmax(sc))
            if sc[k] > delta[e, j]:
                delta[e, j], back_start[e, j], back_prev[e, j] = sc[k], int(s[k]), i

    labels = np.zeros(T, dtype=int)
    e, j = T, int(np.argmax(delta[T]))
    while e > 0 and j >= 0:
        s = back_start[e, j]
        if s < 0:
            break
        labels[s:e] = j
        e, j = s, back_prev[e, j]
    return labels


def build_v7(sp: Spine) -> Tuple[List[Stage], np.ndarray]:
    t, fs, env = sp.t, sp.fs, sp.env
    min_cycle_s = 60.0 / FETAL_BPM_RANGE[1]
    max_cycle_s = 60.0 / FETAL_BPM_RANGE[0]
    ffs = V7_FEAT_FS

    # Resample to a coarse grid and scale robustly, so the emission model is
    # amplitude-invariant and the Viterbi is cheap.
    grid = np.arange(t[0], t[-1], 1.0 / ffs)
    eg = np.interp(grid, t, env)
    lo_p, hi_p = np.percentile(eg, V7_NORM_PCT)
    a = np.clip((eg - lo_p) / (hi_p - lo_p + 1e-12), 0.0, 1.0)

    ac, cycle, sys = _autocorr_cycle_systole(a, ffs, min_cycle_s, max_cycle_s)
    durations, dur_means = _build_durations(cycle, sys, ffs)
    log_e = _log_emission(a)
    labels = _viterbi_hsmm(log_e, durations)

    # Beat = the centre of each decoded S1 segment, not its leading edge. Padding with
    # a zero on both sides makes the diff give inclusive starts (+1) and exclusive
    # ends (-1), so an S1 run touching either end of the window is still bounded.
    is_s1 = (labels == S1).astype(int)
    edges = np.diff(np.concatenate([[0], is_s1, [0]]))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    beats = ((grid[starts] + grid[ends - 1]) / 2.0 if len(starts)
             else np.array([], dtype=float))
    cycle_bpm = 60.0 * ffs / cycle if cycle else float("nan")

    def s6(ax: plt.Axes) -> None:
        ax.plot(grid, a, lw=1.0, color=C_ENV)
        ax.set_ylabel("soundness a")
        ax.set_ylim(-0.05, 1.05)
        _note(ax, f"{len(grid)} samples   {ffs:.0f} Hz   "
                  f"scale [{lo_p:.4g}, {hi_p:.4g}]")

    def s7(ax: plt.Axes) -> None:
        lag_ms = np.arange(len(ac)) / ffs * 1000.0
        keep = lag_ms <= max_cycle_s * 1000.0 * 1.2
        ax.plot(lag_ms[keep], ac[keep], lw=1.0, color=C_SMOOTH)
        ax.axvline(cycle / ffs * 1000.0, color=C_BEAT, lw=1.2,
                   label=f"cycle S1→S1 = {cycle / ffs * 1000:.0f} ms ({cycle_bpm:.0f} BPM)")
        ax.axvline(sys / ffs * 1000.0, color=C_REJECT, lw=1.2, ls="--",
                   label=f"systole = {sys / ffs * 1000:.0f} ms")
        ax.axvspan(0.30 * cycle / ffs * 1000.0, 0.45 * cycle / ffs * 1000.0,
                   color=C_REJECT, alpha=0.10)
        ax.set_xlabel("Lag (ms)")
        ax.set_ylabel("autocorr.")
        ax.legend(loc="upper right", fontsize=7.5)

    def s8(ax: plt.Axes) -> None:
        for st, (dmin, dmax, lp) in enumerate(durations):
            d = np.arange(dmin, dmax + 1)
            ax.plot(d / ffs * 1000.0, np.exp(lp[dmin:dmax + 1] - lp[dmin:dmax + 1].max()),
                    lw=1.2, label=f"{STATE_NAMES[st]}  "
                                  f"({dur_means[st] / ffs * 1000:.0f} ms)")
        ax.set_xlabel("Duration (ms)")
        ax.set_ylabel("prior (norm.)")
        ax.legend(loc="upper right", fontsize=7.5)

    def s9(ax: plt.Axes) -> None:
        ax.plot(grid, a, lw=1.0, color=C_ENV, label="sound present  (S1, S2)")
        ax.plot(grid, 1.0 - a, lw=1.0, color="0.6", label="silence  (systole, diastole)")
        ax.set_ylabel("likelihood")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(loc="lower right", fontsize=7.5, ncol=2)

    def s10(ax: plt.Axes) -> None:
        ax.plot(grid, a, lw=0.9, color=C_ENV, zorder=4)
        for st in (S1, SYS, S2, DIA):
            ax.fill_between(grid, 0, 1, where=(labels == st), color=_STATE_COLORS[st],
                            alpha=0.55, step="mid", label=STATE_NAMES[st], lw=0)
        ax.set_ylabel("state")
        ax.set_ylim(0, 1)
        ax.legend(loc="lower right", fontsize=7.5, ncol=4)
        counts = "   ".join(f"{STATE_NAMES[s]} {int(np.sum(labels == s))}" for s in range(4))
        _note(ax, f"samples per state:   {counts}")

    stages = [
        Stage(f"Envelope resampled and normalised  [{ffs:.0f} Hz grid, "
              f"{V7_NORM_PCT[0]:.0f}th–{V7_NORM_PCT[1]:.0f}th percentile scaling]", s6),
        Stage("Autocorrelation", s7, aux=True),
        Stage("Duration priors  [Gaussian, per state]", s8, aux=True),
        Stage("Emission model  [sound present vs. silence]", s9),
        Stage("HSMM Viterbi decode  [S1 → systole → S2 → diastole]", s10),
    ]
    return stages, beats


# --------------------------------------------------------------------------------
# Capstone: every stage of the build-up on one waveform
# --------------------------------------------------------------------------------

def render_comparison(sp: Spine, results: List[Tuple[str, np.ndarray]],
                      gt: Optional[np.ndarray], suptitle: str, out_path: Path,
                      dpi: int, xlim: Tuple[float, float]) -> None:
    rows = [("naive (no constraints)", sp.naive_times, C_REJECT),
            ("naive + min spacing", sp.spaced_times, "tab:orange")]
    rows += [(lbl, b, C_BEAT) for lbl, b in results]
    if gt is not None and len(gt):
        rows.append(("hand-marked", gt, C_GT))

    fig, axes = plt.subplots(len(rows), 1, figsize=(16, 1.7 * len(rows) + 0.8),
                             sharex=True, constrained_layout=True)
    lo_y, hi_y = float(np.min(sp.filt)), float(np.max(sp.filt))
    for ax, (label, beats, color) in zip(np.atleast_1d(axes), rows):
        ax.plot(sp.t, sp.filt, lw=0.45, color=C_GHOST, rasterized=True)
        inw = np.asarray(beats)[(np.asarray(beats) >= xlim[0]) & (np.asarray(beats) <= xlim[1])]
        if len(inw):
            ax.vlines(inw, lo_y, hi_y, color=color, lw=1.0, alpha=0.8)
        ax.set_ylabel(label, rotation=0, ha="right", va="center", fontsize=9,
                      fontweight="bold")
        ax.margins(x=0)
        ax.set_xlim(*xlim)
        ax.grid(True, axis="x", linestyle=":", linewidth=0.5, alpha=0.35)
        ax.set_title(_bpm_note(inw), loc="left", fontsize=8.5)
    np.atleast_1d(axes)[-1].set_xlabel("Time (s)")
    fig.suptitle(suptitle, fontsize=13)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f"saved -> {out_path}")


# --------------------------------------------------------------------------------

VARIANTS = {"v2": build_v2, "V9": build_V9, "v7": build_v7}
VARIANT_BLURB = {
    "v2": "envelope peaks, distance only",
    "V9": "FWHM regions + duration gate + IBI rejection",
    "v7": "duration-dependent HMM over the cardiac cycle",
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("patient", nargs="?", default=DEFAULT_PATIENT,
                    help=f"Patient dir under {DEFAULT_DATA_DIR} or a full path "
                         f"(default: {DEFAULT_PATIENT})")
    ap.add_argument("--source", choices=["mic", "fiber"], default="mic",
                    help="Signal to build up from (default: mic)")
    ap.add_argument("--fiber", default=DEFAULT_FIBER, choices=list(FIBER_CHANNELS),
                    help=f"Fiber channel when --source fiber (default: {DEFAULT_FIBER})")
    ap.add_argument("--start", type=float, default=100.0, help="Window start (s), default 100")
    ap.add_argument("--end", type=float, default=108.0, help="Window end (s), default 108")
    ap.add_argument("--variant", choices=[*VARIANTS, "all"], default="all",
                    help="Which build-up to render (default: all)")
    ap.add_argument("--zoom", type=float, default=1.5, metavar="SECONDS",
                    help="Also write a zoomed figure over the first N seconds of the "
                         "window; 0 disables (default: 1.5)")
    ap.add_argument("--rms-win", type=float, default=RMS_WIN_S, metavar="SECONDS",
                    help=f"RMS envelope window (default: {RMS_WIN_S})")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR, help=f"default: {OUT_DIR}")
    ap.add_argument("--dpi", type=int, default=140, help="default: 140")
    args = ap.parse_args()

    data_dir = resolve_patient_dir(args.patient)
    name = data_dir.name

    if args.source == "fiber":
        t, raw, fs = load_fiber(data_dir, args.fiber, args.start, args.end)
        src_label, src_tag = f"fiber {args.fiber}", f"fiber{args.fiber}"
    else:
        t, raw, fs = load_mic(data_dir, args.start, args.end)
        src_label, src_tag = "microphone", "mic"

    # mic_beats.npy is marked on the microphone timebase. The mic and fiber are
    # separate acquisitions, so over a fiber trace these sit wherever the two devices'
    # clock offset puts them -- they are drawn uncorrected, and any constant shift
    # against the detected beats is that offset rather than detector error.
    gt = load_beats(data_dir, args.start, args.end)
    gt_msg = str(len(gt)) if gt is not None else "none (no mic_beats.npy)"

    print(f"patient: {name}   source: {src_label}   {fs / 1000:.1f} kHz   "
          f"window {args.start:g}-{args.end:g}s   ({len(raw)} samples)")
    print(f"hand-marked beats: {gt_msg}")

    sp = build_spine(t, raw, fs, args.rms_win, src_label)
    window = (float(t[0]), float(t[-1]))
    # Skip the zoom when it would just reproduce the main figure.
    zoom = None
    if args.zoom > 0 and args.zoom < window[1] - window[0]:
        zoom = (window[0], window[0] + args.zoom)

    win_tag = f"{_fmt_t(args.start)}-{_fmt_t(args.end)}"
    win_title = f"{_fmt_t(args.start)}–{_fmt_t(args.end)}s"

    chosen = list(VARIANTS) if args.variant == "all" else [args.variant]
    results: List[Tuple[str, np.ndarray]] = []

    for v in chosen:
        stages, beats = VARIANTS[v](sp)
        results.append((v, beats))
        print(f"  {v}: {_bpm_note(beats)}")

        full = sp.stages + stages
        full.append(Stage("Detected beats on the band-limited waveform",
                          _beats_panel(t, sp.filt, beats, gt, v)))
        stem = f"{name}_buildup_{src_tag}_{v}_{win_tag}"
        title = (f"{v} beat detector — {VARIANT_BLURB[v]}   |   {name}, {src_label}, "
                 f"{win_title}   |   RMS envelope "
                 f"({args.rms_win * 1000:.0f} ms)")
        render(full, title, args.out_dir / f"{stem}.png", args.dpi, window)
        if zoom:
            render(full, title + f"   |   zoom {zoom[0]:.1f}–{zoom[1]:.1f}s",
                   args.out_dir / f"{stem}_zoom.png", args.dpi, zoom)

    if len(results) > 1:
        stem = f"{name}_buildup_{src_tag}_compare_{win_tag}"
        title = (f"Beat detection build-up — naive → v2 → V9 → v7   |   {name}, "
                 f"{src_label}, {win_title}")
        render_comparison(sp, results, gt, title, args.out_dir / f"{stem}.png",
                          args.dpi, window)
        if zoom:
            render_comparison(sp, results, gt, title + f"   |   zoom",
                              args.out_dir / f"{stem}_zoom.png", args.dpi, zoom)


if __name__ == "__main__":
    main()
