from dataclasses import replace
from itertools import combinations
from pathlib import Path
from typing import Tuple

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import to_rgba
from scipy.ndimage import uniform_filter1d

from analyze.drift import correct_drift
from analyze.hr import fHROutput, fHRMultiOutput
from analyze.sot import SOTResult
from analyze.util import moving_average, moving_average_v2
from analyze.constants import FETAL_BPM_RANGE

# ---------------------------------------------------------------------------
# Every individual subplot in this module is 4:3 (width:height). Subplots are
# stacked vertically and share a width, so the total figure height is just
# n_rows subplot-heights.
# ---------------------------------------------------------------------------
FIG_ASPECT = 16.0 / 6.0  # per-subplot width:height
DEFAULT_WIDTH = 8.0
ROW_HEIGHT = DEFAULT_WIDTH / FIG_ASPECT  # height of one subplot at the default width


def _figsize(n_rows: int, width: float = DEFAULT_WIDTH,
             row_height: float | None = None) -> Tuple[float, float]:
    """Figsize so each of ``n_rows`` stacked subplots is 4:3 (width:height).

    ``row_height`` pins each row's height instead of deriving it from the width, so a long
    recording can be stretched along the time axis without the figure growing as tall as
    it is wide.
    """
    return width, n_rows * (row_height if row_height is not None else width / FIG_ASPECT)


# ---------------------------------------------------------------------------
# Instantaneous heart rate: fiber-pipeline detections vs SOT reference
# ---------------------------------------------------------------------------
# Instantaneous HR is the beat-to-beat rate 60 / IBI, where IBI = diff(beats),
# so each value is plotted at the *second* beat of its pair (beats[1:]) — the
# same convention the SOT and detector code already use (60.0 / diff(times)).


def _current_hr_func():
    return _inst_hr_v2

def _inst_hr_v2(
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
    t = beats[1:]

    keep = (bpm >= band[0]) & (bpm <= band[1])
    bpm, t = bpm[keep], t[keep]

    # bpm = np.clip(bpm, band[0], band[1])
    # Centered average with edge replication (not zero-pad) so ends don't sag.
    bpm = moving_average_v2(bpm, 20)
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
    if beats.size < 2:
        return np.array([]), np.array([])

    bpm = 60.0 / np.clip(np.diff(beats), 1e-6, None)
    bpm = np.clip(bpm, band[0], band[1])
    # Centered average with edge replication (not zero-pad) so ends don't sag.
    # bpm = moving_average_v2(bpm, 5)
    bpm = uniform_filter1d(bpm, size=min(5, bpm.size), mode='nearest')
    return beats[1:], bpm

# def _inst_hr_v3(
#         beats: np.ndarray,
#         band: Tuple[float, float],
# ) -> Tuple[np.ndarray, np.ndarray]:
#     beats = np.sort(beats)
#     window_len = 5.0
#     half = window_len / 2
#
#     left = np.searchsorted(beats, beats - half, side="left")
#     right = np.searchsorted(beats, beats + half, side="right")
#     counts = right - left
#
#     bpm = counts / window_len * 60.0          # one value per beat
#
#     keep = (bpm >= band[0]) & (bpm <= band[1])
#     bpm, beats = bpm[keep], beats[keep]
#
#     bpm = moving_average_v2(bpm, 10)          # 10-beat window, same length as beats
#
#     return beats, bpm

def _inst_hr_v3(
        beats: np.ndarray,
        band: Tuple[float, float],
) -> Tuple[np.ndarray, np.ndarray]:
    window_len = 5.0
    edges = np.arange(0, beats.max() + window_len, window_len)
    counts, _ = np.histogram(beats, bins=edges)

    bpm = counts / window_len * 60.0
    t = edges[:-1] + window_len / 2

    return t, bpm

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


def _pairwise_r(named, grid_hz: float = 4.0):
    """Pearson R and mean absolute error (BPM) between every pair of HR traces,
    over the span all of them share.

    Each trace is sampled at its own beat times, so they have to be resampled onto one
    common grid before they can be compared. Returns ``{(name_a, name_b): (r, mae)}``,
    omitting pairs with no shared span or a constant trace (where R is undefined). R
    captures whether the traces move together (pattern); MAE captures how far apart the
    numbers actually are (magnitude) -- two traces offset by a constant amount can still
    score R close to 1, so the pair is reported together rather than R alone.
    """
    usable = [(n, t, y) for (n, t, y) in named if t is not None and t.size >= 2]
    if len(usable) < 2:
        return {}

    lo = max(float(t[0]) for (_, t, _) in usable)
    hi = min(float(t[-1]) for (_, t, _) in usable)
    if hi <= lo:
        return {}

    grid = np.arange(lo, hi, 1.0 / grid_hz)
    resampled = {n: np.interp(grid, t, y) for (n, t, y) in usable}

    out = {}
    for a, b in combinations(resampled, 2):
        ya, yb = resampled[a], resampled[b]
        if ya.std() > 0 and yb.std() > 0:
            r = float(np.corrcoef(ya, yb)[0, 1])
            mae = float(np.mean(np.abs(ya - yb)))
            out[(a, b)] = (r, mae)
    return out


def _r_box(ax, lines) -> None:
    """Draw the R readout in the upper-left, opposite the legend."""
    ax.text(0.01, 0.98, "\n".join(lines), transform=ax.transAxes, va='top', ha='left',
            fontsize=7, family='monospace',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.7', alpha=0.8))


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
        marker_size: float = 3.0,
) -> None:
    if sot_beats is not None:
        sot_t, sot_y = _current_hr_func()(sot_beats, band)

    pred_t, pred_y = _current_hr_func()(pred_beats, band)

    # marker_size <= 0 draws the curves only. Over a long recording the per-beat markers
    # touch and read as a thick band, which hides the shape of the trace.
    ref_marker = 'o' if marker_size > 0 else None
    pred_marker = 's' if marker_size > 0 else None

    if sot_beats is not None and sot_t.size:
        med = float(np.median(sot_y))
        ax.plot(sot_t, sot_y, color=sot_color, lw=1.4, marker=ref_marker, ms=marker_size,
                alpha=0.9, label=f'{sot_label} (median {med:.1f})')
    if pred_t.size:
        med = float(np.median(pred_y))
        ax.plot(pred_t, pred_y, color=pred_color, lw=1.1, marker=pred_marker, ms=marker_size,
                alpha=0.8, label=f'{pred_label} (median {med:.1f})')

    # Agreement with the reference, on the same footing as the multi-trace panel.
    if sot_beats is not None and sot_t.size and pred_t.size:
        pair = _pairwise_r([(pred_label, pred_t, pred_y), (sot_label, sot_t, sot_y)])
        stat = next(iter(pair.values()), None)
        if stat is not None:
            r, mae = stat
            _r_box(ax, [f"R vs {sot_label}  {r:+.3f}", f"MAE          {mae:5.1f} bpm"])
            print(f"[plot_hr] {pred_label} vs {sot_label}: R={r:.3f}  MAE={mae:.1f} bpm")

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
        marker_size: float = 3.0,
) -> None:
    """Like ``_plot_hr_axis``, but overlays one trace per entry of ``pred_beats``
    (name -> beat times) instead of a single prediction."""
    # Drop the SOT's own colour from the palette: past four overlaid traces tab10 reaches
    # tab:red, and a prediction sharing the reference's colour is unreadable.
    cmap = plt.get_cmap("tab10")
    sot_rgba = to_rgba(sot_color)
    palette = [c for c in (cmap(i) for i in range(10)) if c != sot_rgba] or [cmap(i) for i in range(10)]
    traces = [(name, *_current_hr_func()(beats, band)) for name, beats in pred_beats.items()]

    # marker_size <= 0 draws lines only. Once enough traces overlap, per-beat markers stop
    # separating the series and just thicken them into a band.
    ref_marker = 'o' if marker_size > 0 else None
    trace_marker = 's' if marker_size > 0 else None

    if sot_beats is not None:
        sot_t, sot_y = _current_hr_func()(sot_beats, band)
        if sot_t.size:
            med = float(np.median(sot_y))
            ax.plot(sot_t, sot_y, color=sot_color, lw=1.4, marker=ref_marker, ms=marker_size,
                    alpha=0.9, label=f'{sot_label} (median {med:.1f})')

    for i, (name, t, y) in enumerate(traces):
        if t.size:
            med = float(np.median(y))
            ax.plot(t, y, color=palette[i % len(palette)], lw=1.1, marker=trace_marker,
                    ms=marker_size, alpha=0.8, label=f'{name} (median {med:.1f})')

    # Pairwise Pearson R between the overlaid HR traces -- how much the sources agree
    # beat-to-beat. The reference joins as an ordinary series, so each source's R against
    # it falls out of the same pass -- that is the number that ranks the sources.
    usable = list(traces)
    if sot_beats is not None:
        usable.append((sot_label, sot_t, sot_y))
    pair_r = _pairwise_r(usable)

    if pair_r:
        mean_r = float(np.mean([r for (r, _) in pair_r.values()]))
        # R against the reference is listed in full however many sources there are: it is
        # the comparison being made. The source-vs-source pairs stay on the stdout line,
        # where they don't crowd the axes.
        ref_r = {(a if b == sot_label else b): stat
                 for (a, b), stat in pair_r.items() if sot_label in (a, b)}
        if ref_r:
            width = max(len(n) for n in ref_r)
            lines = [f"{'':<{width}} {'R':>7} {'MAE':>7}"]
            lines += [f"{n:<{width}} {r:+.3f} {mae:6.1f}" for n, (r, mae) in ref_r.items()]
            mean_ref_r = float(np.mean([r for (r, _) in ref_r.values()]))
            mean_ref_mae = float(np.mean([mae for (_, mae) in ref_r.values()]))
            lines.append(f"{'mean':<{width}} {mean_ref_r:+.3f} {mean_ref_mae:6.1f}")
        else:
            lines = [f"R (mean) = {mean_r:.2f}"]
            if len(pair_r) <= 3:  # list each pair when there are few; else just the mean
                lines += [f"{a}-{b}: R={r:.2f} MAE={mae:.1f}" for (a, b), (r, mae) in pair_r.items()]

        _r_box(ax, lines)
        print("[plot_hr_multi] pairwise HR corr coef: "
              + ", ".join(f"{a}-{b}={r:.3f} (MAE {mae:.1f})" for (a, b), (r, mae) in pair_r.items())
              + f" (mean R {mean_r:.3f})")
        if ref_r:
            print(f"[plot_hr_multi] HR corr coef vs {sot_label}: "
                  + ", ".join(f"{n}={r:.3f} (MAE {mae:.1f})" for n, (r, mae) in ref_r.items())
                  + f" (mean R {float(np.mean([r for (r, _) in ref_r.values()])):.3f})")

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
        title: str = "Instantaneous heart rate: all abdomen fibers vs SOT",
        panel_title: str = "Fetal instantaneous HR — all abdomen fibers",
        width: float = 8.0,
        marker_size: float = 3.0,
        row_height: float = ROW_HEIGHT,
) -> None:
    """One fetal-HR panel with every trace in ``multi.fetal_beats`` overlaid (plus an
    optional maternal panel), for spotting overall trends across them.

    The titles default to the abdomen-fiber wording this was written for; callers
    overlaying something else (model versions, detectors) should pass their own."""
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
        fig, (ax_m, ax_f) = plt.subplots(2, 1, figsize=_figsize(2, width, row_height),
                                         sharex=True, constrained_layout=True)
        _plot_hr_axis(
            ax_m,
            sot_beats=sot.ppg_beats if sot is not None else None,
            pred_beats=multi.maternal_beats,
            sot_color='tab:blue', pred_color='tab:orange',
            sot_label='PPG (SOT)', pred_label='Fiber chest',
            title="Maternal instantaneous HR — fiber vs SOT",
            band=(30.0, 160.0),
            marker_size=marker_size,
        )
    else:
        fig, ax_f = plt.subplots(1, 1, figsize=_figsize(1, width, row_height),
                                 constrained_layout=True)

    _plot_hr_axis_multi(
        ax_f,
        sot_beats=sot.mic_beats if sot is not None else None,
        pred_beats=multi.fetal_beats,
        sot_color='tab:red',
        sot_label='Mic (SOT)',
        title=panel_title,
        band=FETAL_BPM_RANGE,
        marker_size=marker_size,
    )

    ax_f.set_xlabel("Time (s)", fontsize=8)
    ax_f.set_xlim(t_start, t_end)
    fig.suptitle(title, fontsize=11)
    plt.savefig(out / filename, dpi=500)
    plt.close()


def plot_hr_comparison(
        fetal_result: fHROutput,
        sot: SOTResult | None,
        out: Path,
        filename: str = "hr_comparison.png",
        width: float = 3 * DEFAULT_WIDTH,
        marker_size: float = 0.0,
        row_height: float = ROW_HEIGHT,
) -> None:
    """Two stacked panels — maternal (top) and fetal (bottom) — each comparing
    the fiber pipeline's instantaneous HR against the SOT reference.

    Drawn three times the default width, at unchanged panel height, so a full recording
    spreads along the time axis instead of collapsing into a band; ``marker_size=0`` leaves
    the curves without per-beat markers for the same reason.
    """
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

    fig, (ax_m, ax_f) = plt.subplots(2, 1, figsize=_figsize(2, width, row_height),
                                     sharex=True, constrained_layout=True)

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
            marker_size=marker_size,
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
        marker_size=marker_size,
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


def plot_hr_corrected(sot: SOTResult, drift_log_path, out: Path, filename: str = "hr_comparison_corrected.png"):
    """Pipeline stage: like ``plot_hr``, but first corrects ``sot.mic_beats``
    for NST clock drift (dropped-sample gaps, see ``analyze.drift``) and
    writes to a separate file so the corrected plot can be compared against
    the uncorrected ``hr_comparison.png`` rather than replacing it. A no-op
    correction (original beats, unchanged) if ``drift_log_path`` doesn't exist.
    """

    def run_plot_hr_corrected(result):
        fetal_result = getattr(result, "fetal_result", result)
        corrected_sot = replace(sot, mic_beats=correct_drift(sot.mic_beats, drift_log_path))
        plot_hr_comparison(fetal_result, corrected_sot, out, filename=filename)
        return result

    run_plot_hr_corrected.__name__ = "plot_hr_corrected"
    return run_plot_hr_corrected


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
