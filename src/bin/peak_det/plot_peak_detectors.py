#!/usr/bin/env python3
"""Compare every fetal peak detector on one patient's mic (SOT) recording.

Loads a patient's ``microphone.wav`` as the SOT source, band-limits it to the
fetal acoustic band, then runs *every* beat detector in ``analyze.hr``
(``v1_beat_detector`` .. ``v8_beat_detector``) on the same windowed segment.
Each detector's beats are drawn on its own subplot, stacked vertically over a
shared time axis, so the differences between detectors are visible at a glance.

Two figures are written to ``bin/peak_det/out/``:
  * ``<patient>_peaks_<start>-<end>.png`` — each detector's beats on the waveform.
  * ``<patient>_hr_<start>-<end>.png`` — instantaneous HR (60 / IBI): one subplot
    per detector, each showing **two** traces: the hand-marked SOT HR (from
    ``mic_beats.npy``, the file the beat-marking app exports next to
    ``microphone.wav``) and that detector's HR, so each detector can be scored
    against ground truth. Pass ``--smooth N`` to moving-average the HR traces.

Examples
--------
    # default patient/window
    python bin/peak_det/plot_peak_detectors.py 5ch_belly_machine_1

    # explicit window and a full path
    python bin/peak_det/plot_peak_detectors.py PT12_1 --start 60 --end 100

    # smooth the HR traces over a 5-beat window
    python bin/peak_det/plot_peak_detectors.py PT12_1 --smooth 5
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

# bin/peak_det/plot_peak_detectors.py -> parents[2] == src/
SRC_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SRC_DIR))

from constants import (  # noqa: E402
    DEFAULT_DATA_DIR,
    FETAL_ACOUSTIC_BAND_HZ,
    FETAL_BPM_RANGE,
    MIC_FILE,
    MIC_BEATS_FILE,
)
from analyze.filters import bp_filter  # noqa: E402
from analyze.sot import load_sot_no_ppg  # noqa: E402
from analyze.hr.detect import v1_beat_detector  # noqa: E402
from analyze.hr.detect_v2 import v2_beat_detector  # noqa: E402
from analyze.hr.detect_v3 import v3_beat_detector  # noqa: E402
from analyze.hr.detect_v4 import v4_beat_detector  # noqa: E402
from analyze.hr.detect_v5 import v5_beat_detector  # noqa: E402
from analyze.hr.detect_v6 import v6_beat_detector  # noqa: E402
from analyze.hr.detect_v7 import v7_beat_detector  # noqa: E402
from analyze.hr.detect_v8 import v8_beat_detector  # noqa: E402

# Every detector shares the (X: Audio, bpm_range, out, tag) -> {"peaks", "times"}
# contract. ``out=None`` is passed so each detector skips its own debug figure;
# the combined comparison figure below is the only output.
DETECTORS = [
    ("v1", v1_beat_detector),
    ("v2", v2_beat_detector),
    ("v3", v3_beat_detector),
    ("v4", v4_beat_detector),
    ("v5", v5_beat_detector),
    ("v6", v6_beat_detector),
    ("v7", v7_beat_detector),
    ("v8", v8_beat_detector),
]

OUT_DIR = Path(__file__).resolve().parent / "out"


def resolve_patient_dir(patient: str) -> Path:
    """Accept either a bare patient name (under DEFAULT_DATA_DIR) or a full path."""
    p = Path(patient)
    candidates = [p, Path(DEFAULT_DATA_DIR) / patient]
    for c in candidates:
        if (c / MIC_FILE).exists():
            return c
    tried = "\n  ".join(str(c / MIC_FILE) for c in candidates)
    raise FileNotFoundError(f"Could not find {MIC_FILE} for '{patient}'. Tried:\n  {tried}")


def median_bpm(times: np.ndarray) -> float:
    times = np.asarray(times, dtype=float)
    if times.size < 2:
        return float("nan")
    return 60.0 / float(np.median(np.diff(times)))


def load_beat_npy(path) -> np.ndarray:
    """Sorted beat timestamps (s) from a .npy written by the beat-marking app.
    Accepts a 1-D array of times, or a 2-D array whose first column is time."""
    arr = np.asarray(np.load(path, allow_pickle=False), dtype=float)
    if arr.ndim == 2 and arr.shape[1] >= 1:
        arr = arr[:, 0]
    arr = arr.ravel()
    return np.sort(arr[np.isfinite(arr)])


def compute_hr(times: np.ndarray):
    """Instantaneous HR (60 / IBI). Returns (t_mid, bpm): one point per consecutive
    beat pair, placed at the pair's midpoint."""
    times = np.sort(np.asarray(times, dtype=float))
    if times.size < 2:
        return np.array([], dtype=float), np.array([], dtype=float)
    ibi = np.diff(times)
    bpm = 60.0 / np.clip(ibi, 1e-9, None)
    t_mid = (times[:-1] + times[1:]) / 2.0
    return t_mid, bpm


def smooth_series(y: np.ndarray, window: int) -> np.ndarray:
    """Centred moving average over the HR points; ``window`` is a count of points
    (<= 1 disables it). Edges shrink the window to whatever fits."""
    y = np.asarray(y, dtype=float)
    if window is None or window <= 1 or y.size == 0:
        return y
    half = window // 2
    out = np.empty_like(y)
    for i in range(y.size):
        a, b = max(0, i - half), min(y.size, i + half + 1)
        out[i] = float(np.mean(y[a:b]))
    return out


def run_detectors(mic_bp, out_dir: Path):
    """Run every detector on the band-limited mic. Returns list of (label, times, error)."""
    results = []
    for label, detector in DETECTORS:
        try:
            res = detector(mic_bp, FETAL_BPM_RANGE, None, tag=label)
            times = np.asarray(res["times"], dtype=float)
            print(f"  {label}: {len(times):4d} beats   median {median_bpm(times):6.1f} BPM")
            results.append((label, times, None))
        except Exception as exc:  # keep going so one bad detector doesn't sink the figure
            print(f"  {label}: FAILED ({type(exc).__name__}: {exc})")
            results.append((label, np.array([], dtype=float), f"{type(exc).__name__}: {exc}"))
    return results


def plot(mic_bp, results, patient: str, start: float, end: float, out_path: Path, dpi: int):
    t = np.asarray(mic_bp.time, dtype=float)
    x = np.asarray(mic_bp.data, dtype=float)
    lo, hi = float(t[0]), float(t[-1])

    n = len(results)
    fig, axes = plt.subplots(n, 1, figsize=(16, 2.2 * n), sharex=True,
                             constrained_layout=True)
    if n == 1:
        axes = [axes]

    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    for ax, (label, times, err), color in zip(axes, results, colors):
        ax.plot(t, x, color="0.55", lw=0.5, alpha=0.9, rasterized=True)
        in_win = times[(times >= lo) & (times <= hi)]
        for b in in_win:
            ax.axvline(b, color=color, lw=1.0, alpha=0.55)
        if in_win.size:
            y = np.interp(in_win, t, x)
            ax.plot(in_win, y, "o", color=color, ms=4,
                    markeredgecolor="black", markeredgewidth=0.4, zorder=6)
        ax.set_ylabel(label, rotation=0, ha="right", va="center", fontsize=11,
                      fontweight="bold")
        ax.margins(x=0)
        ax.grid(True, axis="x", linestyle=":", linewidth=0.5, alpha=0.4)

        if err is None:
            title = f"{label}_beat_detector  |  {in_win.size} beats  |  median {median_bpm(times):.1f} BPM"
            tcolor = "black"
        else:
            title = f"{label}_beat_detector  |  ERROR: {err}"
            tcolor = "tab:red"
        ax.set_title(title, loc="left", fontsize=9, color=tcolor)

    axes[-1].set_xlabel("Time (s)")
    axes[-1].set_xlim(lo, hi)
    fig.suptitle(
        f"Fetal peak detectors on {patient}  —  mic (SOT), "
        f"{FETAL_ACOUSTIC_BAND_HZ[0]:.0f}-{FETAL_ACOUSTIC_BAND_HZ[1]:.0f} Hz  "
        f"|  window {start:.0f}-{end:.0f}s",
        fontsize=13,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f"\nsaved -> {out_path}")


def plot_hr(results, sot_times, patient: str, start: float, end: float,
            out_path: Path, dpi: int, smooth_window: int):
    """Second figure: instantaneous HR (60/IBI), one subplot per detector, each
    overlaying the SOT HR (from mic_beats.npy) against that detector's HR."""
    n = len(results)
    fig, axes = plt.subplots(n, 1, figsize=(16, 2.0 * n), sharex=True, sharey=True,
                             constrained_layout=True)
    if n == 1:
        axes = [axes]
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    # SOT ground-truth HR (may be absent if mic_beats.npy wasn't found).
    if sot_times is not None and np.asarray(sot_times).size:
        sot_t, sot_hr = compute_hr(sot_times)
        sot_hr_s = smooth_series(sot_hr, smooth_window)
    else:
        sot_t, sot_hr, sot_hr_s = np.array([]), np.array([]), np.array([])

    # Shared y-scale from robust percentiles of every HR trace, so one runaway
    # detector doesn't flatten the rest; glitchy points just clip.
    pooled = [sot_hr]
    per_det_hr = []
    for label, times, err in results:
        _, hr = compute_hr(times[(times >= start) & (times <= end)])
        per_det_hr.append(hr)
        pooled.append(hr)
    pooled = np.concatenate([h for h in pooled if h.size]) if any(h.size for h in pooled) else np.array([120.0])
    lo_y = max(30.0, float(np.percentile(pooled, 2)) - 15.0)
    hi_y = min(320.0, float(np.percentile(pooled, 98)) + 15.0)
    if hi_y - lo_y < 20.0:
        hi_y = lo_y + 20.0

    for ax, (label, times, err), color, det_hr in zip(axes, results, colors, per_det_hr):
        in_win = times[(times >= start) & (times <= end)]
        d_t, d_hr = compute_hr(in_win)
        if d_hr.size:
            ax.plot(d_t, smooth_series(d_hr, smooth_window), color=color, lw=1.2,
                    marker=".", ms=3, label=label)
        if sot_t.size:
            # Exact same line+marker style as the detector (just black, drawn on top)
            # so the SOT's variability reads identically to a detector trace.
            ax.plot(sot_t, sot_hr_s, color="black", lw=1.2, marker=".", ms=3,
                    alpha=0.85, zorder=5, label="SOT (mic_beats)")
        ax.set_ylabel(label, rotation=0, ha="right", va="center", fontsize=11, fontweight="bold")
        ax.set_ylim(lo_y, hi_y)
        ax.margins(x=0)
        ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.4)

        det_med = median_bpm(in_win)
        if err is None:
            sot_med = median_bpm(sot_times) if sot_times is not None else float("nan")
            title = (f"{label}_beat_detector  |  median {det_med:.1f} BPM"
                     + (f"  (SOT {sot_med:.1f})" if sot_t.size else ""))
            tcolor = "black"
        else:
            title = f"{label}_beat_detector  |  ERROR: {err}"
            tcolor = "tab:red"
        ax.set_title(title, loc="left", fontsize=9, color=tcolor)

    axes[0].legend(loc="upper right", fontsize=8, ncol=2)
    axes[-1].set_xlabel("Time (s)")
    axes[-1].set_xlim(start, end)
    smooth_note = f"  |  smoothing: {smooth_window}-beat MA" if smooth_window and smooth_window > 1 else ""
    sot_note = "SOT = mic_beats.npy" if sot_t.size else "SOT missing (no mic_beats.npy)"
    fig.suptitle(
        f"Instantaneous HR (60/IBI) on {patient}  —  detector vs {sot_note}  "
        f"|  window {start:.0f}-{end:.0f}s{smooth_note}",
        fontsize=13,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f"saved -> {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("patient", nargs="?", default="5ch_belly_machine_1",
                        help="Patient dir name (under DEFAULT_DATA_DIR) or a full path "
                             "(default: 5ch_belly_machine_1)")
    parser.add_argument("--start", type=float, default=30.0, help="Window start (s), default 30")
    parser.add_argument("--end", type=float, default=60.0, help="Window end (s), default 60")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR,
                        help=f"Output directory (default: {OUT_DIR})")
    parser.add_argument("--dpi", type=int, default=140, help="Output resolution (default: 140)")
    parser.add_argument("--sot-beats", type=Path, default=None,
                        help=f"Hand-marked SOT beats .npy for the HR comparison "
                             f"(default: <patient>/{MIC_BEATS_FILE})")
    parser.add_argument("--smooth", type=int, default=0, metavar="N",
                        help="Moving-average window (in beats) for the HR traces; 0 = off (default)")
    args = parser.parse_args()

    data_dir = resolve_patient_dir(args.patient)
    name = Path(str(data_dir).rstrip("/")).name
    print(f"patient: {name}   dir: {data_dir}   window: {args.start:.0f}-{args.end:.0f}s")

    # Load full mic, window, then band-limit — same order as the SOT pipeline
    # (analyze.hr.sot_beats / bin/s1s2_diagnostic).
    sot = load_sot_no_ppg()(str(data_dir)).window(args.start, args.end)
    mic_bp = bp_filter(sot.mic, FETAL_ACOUSTIC_BAND_HZ[0], FETAL_ACOUSTIC_BAND_HZ[1])

    # Hand-marked SOT beats (mic_beats.npy) for the HR comparison, windowed to match.
    sot_beats_path = args.sot_beats or (Path(data_dir) / MIC_BEATS_FILE)
    sot_times = None
    if sot_beats_path.exists():
        sot_all = load_beat_npy(sot_beats_path)
        sot_times = sot_all[(sot_all >= args.start) & (sot_all <= args.end)]
        _, _sot_hr = compute_hr(sot_times)
        spread = (f"HR {np.min(_sot_hr):.0f}-{np.max(_sot_hr):.0f}, std {np.std(_sot_hr):.1f}"
                  if _sot_hr.size else "no HR")
        print(f"SOT beats: {len(sot_times)} from {sot_beats_path.name}   "
              f"median {median_bpm(sot_times):6.1f} BPM   ({spread})")
    else:
        print(f"SOT beats: {sot_beats_path} not found — HR plot will show detectors only")

    print("running detectors:")
    results = run_detectors(mic_bp, args.out_dir)

    out_path = args.out_dir / f"{name}_peaks_{args.start:.0f}-{args.end:.0f}.png"
    plot(mic_bp, results, name, args.start, args.end, out_path, args.dpi)

    hr_path = args.out_dir / f"{name}_hr_{args.start:.0f}-{args.end:.0f}.png"
    plot_hr(results, sot_times, name, args.start, args.end, hr_path, args.dpi, args.smooth)


if __name__ == "__main__":
    main()
