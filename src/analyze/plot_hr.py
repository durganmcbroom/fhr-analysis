from pathlib import Path
from typing import Tuple

import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import uniform_filter1d

from analyze.hr import fHROutput
from analyze.sot import SOTResult


# ---------------------------------------------------------------------------
# Instantaneous heart rate: fiber-pipeline detections vs SOT reference
# ---------------------------------------------------------------------------
# Instantaneous HR is the beat-to-beat rate 60 / IBI, where IBI = diff(beats),
# so each value is plotted at the *second* beat of its pair (beats[1:]) — the
# same convention the SOT and detector code already use (60.0 / diff(times)).


def _inst_hr(
        beats: np.ndarray,
        band: Tuple[float, float],
) -> Tuple[np.ndarray, np.ndarray]:
    """Instantaneous HR (60/IBI) as (time, bpm), clipped to ``band`` and smoothed.

    Beats are sorted first so a stray out-of-order detection can't fold the line.
    """
    beats = np.sort(np.asarray(beats, dtype=float))
    if beats.size < 2:
        return np.array([]), np.array([])

    bpm = 60.0 / np.clip(np.diff(beats), 1e-6, None)
    bpm = np.clip(bpm, band[0], band[1])
    # Centered average with edge replication (not zero-pad) so ends don't sag.
    bpm = uniform_filter1d(bpm, size=min(5, bpm.size), mode='nearest')
    return beats[1:], bpm


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
        sot_t, sot_y = _inst_hr(sot_beats, band)

    pred_t, pred_y = _inst_hr(pred_beats, band)

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
        t_start = float(min(
            sot.ppg.time[0] if len(sot.ppg.time) else 0.0,
            sot.mic.time[0] if len(sot.mic.time) else 0.0,
        ))
        t_end = float(max(
            sot.ppg.time[-1] if len(sot.ppg.time) else 1.0,
            sot.mic.time[-1] if len(sot.mic.time) else 1.0,
        ))
    else:
        t = fetal_result.fetal_source.time
        t_start = t[0]
        t_end = t[-1]

    fig, (ax_m, ax_f) = plt.subplots(2, 1, figsize=(14, 7), sharex=True,
                                     constrained_layout=True)

    # Top: maternal (fiber chest vs PPG SOT). Colours match evaluate.py.
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
        band=(60.0, 240.0),
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
