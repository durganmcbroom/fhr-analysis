from pathlib import Path
from typing import Tuple

import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import uniform_filter1d

from analyze.hr import fHROutput, fHRMultiOutput
from analyze.sot import SOTResult
from analyze.util import moving_average, moving_average_v2
from constants import FETAL_BPM_RANGE

# ---------------------------------------------------------------------------
# Every individual subplot in this module is 4:3 (width:height). Subplots are
# stacked vertically and share a width, so the total figure height is just
# n_rows subplot-heights.
# ---------------------------------------------------------------------------
FIG_ASPECT = 4.0 / 3.0  # per-subplot width:height


def _figsize(n_rows: int, width: float = 8.0) -> Tuple[float, float]:
    """Figsize so each of ``n_rows`` stacked subplots is 4:3 (width:height)."""
    return width, n_rows * width / FIG_ASPECT


# ---------------------------------------------------------------------------
# Instantaneous heart rate: fiber-pipeline detections vs SOT reference
# ---------------------------------------------------------------------------
# Instantaneous HR is the beat-to-beat rate 60 / IBI, where IBI = diff(beats),
# so each value is plotted at the *second* beat of its pair (beats[1:]) — the
# same convention the SOT and detector code already use (60.0 / diff(times)).


def _inst_hr_v2(
        beats: np.ndarray,
        band: Tuple[float, float],
) -> Tuple[np.ndarray, np.ndarray]:
    """Instantaneous HR (60/IBI) as (time, bpm), clipped to ``band`` and smoothed.

    Beats are sorted first so a stray out-of-order detection can't fold the line.
    """
    beats = np.sort(np.asarray(beats, dtype=float))
    print(f"{beats} L:{len(beats)}")
    if beats.size < 2:
        return np.array([]), np.array([])

    bpm = 60.0 / np.clip(np.diff(beats), 1e-6, None)
    t = beats[1:]

    keep = (bpm >= band[0]) & (bpm <= band[1])
    bpm, t = bpm[keep], t[keep]

    # bpm = np.clip(bpm, band[0], band[1])
    # Centered average with edge replication (not zero-pad) so ends don't sag.
    bpm = moving_average_v2(bpm, 5)
    # bpm = uniform_filter1d(bpm, size=min(5, bpm.size), mode='nearest')
    return t, bpm


def _inst_hr(
        beats: np.ndarray,
        band: Tuple[float, float],
) -> Tuple[np.ndarray, np.ndarray]:
    """Instantaneous HR (60/IBI) as (time, bpm), clipped to ``band`` and smoothed.

    Beats are sorted first so a stray out-of-order detection can't fold the line.
    """
    beats = np.sort(np.asarray(beats, dtype=float))
    print(f"{beats} L:{len(beats)}")
    if beats.size < 2:
        return np.array([]), np.array([])

    bpm = 60.0 / np.clip(np.diff(beats), 1e-6, None)
    bpm = np.clip(bpm, band[0], band[1])
    # Centered average with edge replication (not zero-pad) so ends don't sag.
    # bpm = moving_average_v2(bpm, 5)
    bpm = uniform_filter1d(bpm, size=min(5, bpm.size), mode='nearest')
    return beats[1:], bpm

    window_len = 5
    step = 1
    ret = []
    max = round(beats.max())
    for start_s in range(0, max, step):
        end_s = min(start_s + window_len, max)

        window = (beats > start_s) & (beats < end_s)
        num = np.count_nonzero(window)

        bpm_in_range = (num / window_len) * 60

        tight_window = (beats > start_s) & (beats < min(start_s + step, max))
        tight_num = np.count_nonzero(tight_window)
        ret.extend([bpm_in_range] * tight_num)

    return beats, np.array(ret)


def _hr_ylim(traces, band: Tuple[float, float], pad: float = 0.1):
    """Robust y-limits from the values that fall inside a plausible HR band,
    so a few spurious (missed/extra-beat) spikes don't flatten the axis."""
    vals = np.concatenate([y for (_, y) in traces if y.size]) if traces else np.array([])
    vals = vals[(vals >= band[0]) & (vals <= band[1])]
    if vals.size == 0:
        return band
    lo, hi = float(np.min(vals)), float(np.max(vals))
    margin = pad * max(hi - lo, 1.0)
    return lo - margin, hi + margin


def _plot_hr_axis(
        ax,
        sot_beats: np.ndarray,
        pred_beats: np.ndarray,
        sot_color: str,
        pred_color: str,
        sot_label: str,
        pred_label: str,
        title: str,
        band: Tuple[float, float],
) -> None:
    if sot_beats is not None:
        sot_t, sot_y = _inst_hr_v2(sot_beats, band)

    pred_t, pred_y = _inst_hr_v2(pred_beats, band)

    if sot_beats is not None and sot_t.size:
        med = float(np.median(sot_y))
        ax.plot(sot_t, sot_y, color=sot_color, lw=1.4, marker='o', ms=3,
                alpha=0.9, label=f'{sot_label} (median {med:.1f})')
    if pred_t.size:
        med = float(np.median(pred_y))
        ax.plot(pred_t, pred_y, color=pred_color, lw=1.1, marker='s', ms=3,
                alpha=0.8, label=f'{pred_label} (median {med:.1f})')

    ylim_traces = [(pred_t, pred_y)]
    if sot_beats is not None:
        ylim_traces.append((sot_t, sot_y))

    ax.set_ylim(*_hr_ylim(ylim_traces, band))
    ax.set_ylabel("Instantaneous HR (BPM)", fontsize=8)
    ax.grid(True, alpha=0.25)
    ax.legend(loc='upper right', fontsize=8)
    ax.set_title(title, fontsize=9)


def _plot_hr_axis_multi(
        ax,
        sot_beats: np.ndarray | None,
        pred_beats: dict,
        sot_color: str,
        sot_label: str,
        title: str,
        band: Tuple[float, float],
) -> None:
    """Like ``_plot_hr_axis``, but overlays one trace per entry of ``pred_beats``
    (name -> beat times) instead of a single prediction."""
    cmap = plt.get_cmap("tab10")
    traces = [(name, *_inst_hr_v2(beats, band)) for name, beats in pred_beats.items()]

    if sot_beats is not None:
        sot_t, sot_y = _inst_hr_v2(sot_beats, band)
        if sot_t.size:
            med = float(np.median(sot_y))
            ax.plot(sot_t, sot_y, color=sot_color, lw=1.4, marker='o', ms=3,
                    alpha=0.9, label=f'{sot_label} (median {med:.1f})')

    for i, (name, t, y) in enumerate(traces):
        if t.size:
            med = float(np.median(y))
            ax.plot(t, y, color=cmap(i % 10), lw=1.1, marker='s', ms=3,
                    alpha=0.8, label=f'{name} (median {med:.1f})')

    ylim_traces = [(t, y) for (_, t, y) in traces]
    if sot_beats is not None:
        ylim_traces.append((sot_t, sot_y))

    ax.set_ylim(*_hr_ylim(ylim_traces, band))
    ax.set_ylabel("Instantaneous HR (BPM)", fontsize=8)
    ax.grid(True, alpha=0.25)
    ax.legend(loc='upper right', fontsize=8)
    ax.set_title(title, fontsize=9)


def plot_hr_multi_comparison(
        multi: fHRMultiOutput,
        sot: SOTResult | None,
        out: Path,
        filename: str = "hr_comparison_multi.png",
) -> None:
    """One fetal-HR panel with every abdomen fiber's trace overlaid (plus an
    optional maternal panel), for spotting overall trends across fibers."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    if sot is not None:
        t_start = float(min(
            sot.ppg.time[0] if len(sot.ppg.time) else 0.0,
            sot.mic.time[0] if len(sot.mic.time) else 0.0,
        ))
        t_end = float(max(
            sot.ppg.time[-1] if len(sot.ppg.time) else 1.0,
            sot.mic.time[-1] if len(sot.mic.time) else 1.0,
        ))
    else:
        any_source = next(iter(multi.fetal_sources.values()))
        t_start = float(any_source.time[0])
        t_end = float(any_source.time[-1])

    if multi.maternal_beats is not None:
        fig, (ax_m, ax_f) = plt.subplots(2, 1, figsize=_figsize(2), sharex=True,
                                         constrained_layout=True)
        _plot_hr_axis(
            ax_m,
            sot_beats=sot.ppg_beats if sot is not None else None,
            pred_beats=multi.maternal_beats,
            sot_color='tab:blue', pred_color='tab:orange',
            sot_label='PPG (SOT)', pred_label='Fiber chest',
            title="Maternal instantaneous HR — fiber vs SOT",
            band=(30.0, 160.0),
        )
    else:
        fig, ax_f = plt.subplots(1, 1, figsize=_figsize(1), constrained_layout=True)

    _plot_hr_axis_multi(
        ax_f,
        sot_beats=sot.mic_beats if sot is not None else None,
        pred_beats=multi.fetal_beats,
        sot_color='tab:red',
        sot_label='Mic (SOT)',
        title="Fetal instantaneous HR — all abdomen fibers",
        band=FETAL_BPM_RANGE,
    )

    ax_f.set_xlabel("Time (s)", fontsize=8)
    ax_f.set_xlim(t_start, t_end)
    fig.suptitle("Instantaneous heart rate: all abdomen fibers vs SOT", fontsize=11)
    plt.savefig(out / filename, dpi=150)
    plt.close()


def plot_hr_comparison(
        fetal_result: fHROutput,
        sot: SOTResult | None,
        out: Path,
        filename: str = "hr_comparison.png",
) -> None:
    """Two stacked panels — maternal (top) and fetal (bottom) — each comparing
    the fiber pipeline's instantaneous HR against the SOT reference."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    if sot is not None:
        if sot.ppg is not None:
            t_start = float(min(
                sot.ppg.time[0] if len(sot.ppg.time) else 0.0,
                sot.mic.time[0] if len(sot.mic.time) else 0.0,
            ))
            t_end = float(max(
                sot.ppg.time[-1] if len(sot.ppg.time) else 1.0,
                sot.mic.time[-1] if len(sot.mic.time) else 1.0,
            ))
        else:
            t_start = sot.mic.time[0]
            t_end = sot.mic.time[-1]
    else:
        t = fetal_result.fetal_source.time
        t_start = t[0]
        t_end = t[-1]

    fig, (ax_m, ax_f) = plt.subplots(2, 1, figsize=_figsize(2), sharex=True,
                                     constrained_layout=True)

    # Top: maternal (fiber chest vs PPG SOT). Colors match evaluate.py.
    if fetal_result.maternal_beats is not None:
        _plot_hr_axis(
            ax_m,
            sot_beats=sot.ppg_beats if sot is not None else None,
            pred_beats=fetal_result.maternal_beats,
            sot_color='tab:blue', pred_color='tab:orange',
            sot_label='PPG (SOT)', pred_label='Fiber chest',
            title="Maternal instantaneous HR — fiber vs SOT",
            band=(30.0, 160.0),
        )

    # Bottom: fetal (fiber fetal vs mic SOT).
    _plot_hr_axis(
        ax_f,
        sot_beats=sot.mic_beats if sot is not None else None,
        pred_beats=fetal_result.fetal_beats,
        sot_color='tab:red', pred_color='tab:green',
        sot_label='Mic (SOT)', pred_label='Fiber fetal',
        title="Fetal instantaneous HR — fiber vs SOT",
        band=FETAL_BPM_RANGE,
    )

    ax_f.set_xlabel("Time (s)", fontsize=8)
    ax_f.set_xlim(t_start, t_end)
    fig.suptitle("Instantaneous heart rate: fiber pipeline vs SOT", fontsize=11)
    plt.savefig(out / filename, dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Pipeline stage factory
# ---------------------------------------------------------------------------

def plot_hr(sot: SOTResult, out: Path):
    """Pipeline stage: write the maternal/fetal instantaneous-HR comparison plot.

    Pass-through — returns its input unchanged so it can be dropped into a
    Pipeline anywhere after beat detection. Accepts a FetalHRResult, or any
    result wrapping one (e.g. EvaluationResult / EvaluationResultV2 via
    ``.fetal_result``).
    """

    def run_plot_hr(result):
        fetal_result = getattr(result, "fetal_result", result)
        plot_hr_comparison(fetal_result, sot, out)
        return result

    run_plot_hr.__name__ = "plot_hr"
    return run_plot_hr


def plot_multi_hr(sot: SOTResult | None, out: Path):
    """Pipeline stage: write the all-abdomen-fibers instantaneous-HR overlay
    plot. Pass-through, like ``plot_hr``. ``sot`` is optional."""

    def run_plot_multi_hr(result: fHRMultiOutput):
        plot_hr_multi_comparison(result, sot, out)
        return result

    run_plot_multi_hr.__name__ = "plot_multi_hr"
    return run_plot_multi_hr


def _peak_rows(result):
    """Normalize a fHRMultiOutput or fHROutput into (label, source, beats) rows."""
    rows = []
    if isinstance(result, fHRMultiOutput):
        if result.maternal_source is not None:
            rows.append(("maternal", result.maternal_source, result.maternal_beats))
        rows.extend((name, source, result.fetal_beats[name]) for name, source in result.fetal_sources.items())
    else:
        if result.maternal_source is not None:
            rows.append(("maternal", result.maternal_source, result.maternal_beats))
        rows.append(("fetal", result.fetal_source, result.fetal_beats))
    return rows


def plot_peaks(out: Path, filename: str = "peaks.png"):
    """Pipeline stage: plot each channel's waveform (stacked vertically, one
    row per channel) with its detected beats marked on the trace. Accepts
    either a fHRMultiOutput (one row per abdomen fiber, plus maternal if
    present) or a single fHROutput (maternal + fetal rows)."""

    def run_plot_peaks(result):
        rows = _peak_rows(result)

        out.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(len(rows), 1, figsize=_figsize(len(rows)), squeeze=False, sharex=True)

        for row, (label, source, beats) in enumerate(rows):
            ax = axes[row][0]
            data = np.asarray(source.data, dtype=float)

            mn = data.min()
            mx = data.max()

            ax.plot(source.time, data, lw=0.5, color="steelblue")
            beats = np.asarray(beats, dtype=float) if beats is not None else np.array([])
            if beats.size:
                ax.vlines(beats, mn, mx, color="lightcoral", linestyles="--", label=f"peaks (n={beats.size})", lw=0.8)
                ax.legend(loc='upper right', fontsize=7)
            ax.set_title(label, fontsize=9)
            ax.set_ylabel("Amplitude", fontsize=8)
            ax.tick_params(labelsize=7)

        axes[-1][0].set_xlabel("Time (s)", fontsize=8)
        fig.suptitle("Detected peaks on waveform", fontsize=11)
        fig.tight_layout()
        out_file = out / filename
        plt.savefig(out_file, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[plot_peaks] saved visualization → {out_file}")

        return result

    run_plot_peaks.__name__ = "plot_peaks"
    return run_plot_peaks
