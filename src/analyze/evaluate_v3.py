from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import numpy.typing as npt
from matplotlib import pyplot as plt
from scipy.signal import correlate, correlation_lags

from analyze.hr import fHROutput
from analyze.sot import SOTResult
from analyze.util import moving_average_v2
from constants import FETAL_BPM_RANGE

# Reuse the v1 beat-matcher verbatim so the timing-error semantics (MAD outlier
# rejection, ±250 ms acceptance, ±50 ms "correct") stay identical to v1/v2 and
# the three evaluators remain directly comparable. Only the *lag estimation* and
# the *headline score* differ in v3.
from analyze.evaluate import _robust_match


# ---------------------------------------------------------------------------
# v3 design notes
# ---------------------------------------------------------------------------
# v1 cross-correlates Gaussian impulse trains of the beat times; v2 cross-
# correlates sigmoid soft-steps per window. v3 instead works one level up, on
# the *instantaneous heart rate*:
#
#   1. Build the 60/IBI HR trace (the same "60 over inter-beat-interval" method
#      the plot_hr panels use) for both the SOT reference beats and the fiber
#      prediction, per channel.
#
#   2. Resample both HR traces onto a common uniform grid and cross-correlate
#      them. The lag of the largest correlation is the SOT-vs-fiber delay, and
#      the correlation is *normalised* (divided by the geometric mean of the two
#      energies) so the peak is a genuine correlation coefficient in [-1, 1].
#
#   3. Score the timing errors exactly like v1/v2 (correct the prediction by the
#      lag, robust-match against the SOT, count ±50 ms hits), and report the peak
#      correlation coefficient as the overall score.
#
# Lag sign convention matches v1: with ``correlate(ref, pred)`` +
# ``correlation_lags(len(ref), len(pred))`` the prediction is aligned to the
# reference by ``pred_beats + lag`` (see evaluate.py `_eval_channel`).

# HR varies slowly, so the correlation grid need not be dense. 50 Hz resolves the
# lag to 20 ms before the sub-sample parabolic refinement below — fine to feed
# into a ±50 ms match — while staying far cheaper than the 200 Hz impulse grid.
HR_XCORR_FS = 50.0

# Beats averaged (moving_average_v2) into each 60/IBI HR trace. Two smoothings,
# deliberately decoupled:
#
#   * DISPLAY (the HR panel) scales with the analysis-window length so a long
#     recording reads as a trend, not a dense cloud of jumpy points:
#     `HR_SMOOTH_PER_60S` beats per 60 s (10 at 60 s, 20 at 120 s, ...), floored
#     at `HR_SMOOTH_MIN`. Override with an explicit ``hr_smooth`` beat count.
#   * CORRELATION (the lag + the score) uses a fixed, light `HR_XCORR_SMOOTH`, so
#     the lag stays as well-localised as possible and the correlation coefficient
#     is comparable across window lengths — independent of the display smoothing.
#
# HR is smooth, so the coefficient is a trend-agreement measure either way and the
# lag is only trend-accurate regardless (coarse at sub-beat scale).
HR_SMOOTH_PER_60S = 10
HR_SMOOTH_MIN = 3
HR_XCORR_SMOOTH = 5


def _smooth_for_window(span_s: float) -> int:
    """Beats to average into the HR trace for an analysis window of ``span_s``.

    Anchored so a 60 s window smooths over ``HR_SMOOTH_PER_60S`` beats and scales
    linearly with duration; never below ``HR_SMOOTH_MIN``.
    """
    return max(HR_SMOOTH_MIN, int(round(HR_SMOOTH_PER_60S * float(span_s) / 60.0)))

# Plausible HR bands used only to clip the 60/IBI trace (drop spurious
# missed/extra-beat spikes). Fetal reuses the project range; maternal matches the
# band the plot_hr comparison panels already use.
MATERNAL_HR_BAND = (30.0, 160.0)


# ---------------------------------------------------------------------------
# Instantaneous heart rate (60 / IBI)
# ---------------------------------------------------------------------------

def _inst_hr(
        beats: np.ndarray,
        band: Tuple[float, float],
        smooth: int = HR_SMOOTH_PER_60S,
) -> Tuple[np.ndarray, np.ndarray]:
    """Instantaneous HR as ``(time, bpm)`` via the 60/IBI method.

    ``bpm = 60 / diff(beats)`` placed at the second beat of each pair
    (``beats[1:]``) — the same convention as plot_hr and the detectors. Beats are
    sorted first so an out-of-order detection can't fold the line, values outside
    ``band`` are dropped, and the trace is lightly averaged (``moving_average_v2``
    over ``smooth`` beats) to tame beat-to-beat noise before correlation.
    """
    beats = np.sort(np.asarray(beats, dtype=float))
    beats = beats[np.isfinite(beats)]
    if beats.size < 2:
        return np.array([]), np.array([])

    bpm = 60.0 / np.clip(np.diff(beats), 1e-6, None)
    t = beats[1:]

    keep = (bpm >= band[0]) & (bpm <= band[1])
    bpm, t = bpm[keep], t[keep]
    if bpm.size == 0:
        return np.array([]), np.array([])

    bpm = moving_average_v2(bpm, smooth)
    return t, bpm


def _resample_to_grid(t: np.ndarray, hr: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Linear-interpolate an HR trace onto ``grid``; NaN outside its support.

    Leaving out-of-support samples as NaN (rather than extrapolating) lets the
    cross-correlation treat them as "no data" — after demeaning they become zero
    and contribute nothing to either the correlation or the energy.
    """
    out = np.full(grid.shape, np.nan, dtype=float)
    if t.size < 2:
        return out
    inside = (grid >= t[0]) & (grid <= t[-1])
    out[inside] = np.interp(grid[inside], t, hr)
    return out


def _demean_zerofill(x: np.ndarray) -> np.ndarray:
    """Subtract the mean over the valid (finite) samples, then zero the rest."""
    m = np.isfinite(x)
    if not np.any(m):
        return np.zeros_like(x)
    out = x.copy()
    out[m] = out[m] - float(np.mean(out[m]))
    out[~m] = 0.0
    return out


def _parabolic_peak(lags: np.ndarray, corr: np.ndarray, i: int) -> Tuple[float, float]:
    """Sub-sample peak of ``corr`` near index ``i`` by a 3-point parabola fit.

    Returns the refined ``(lag, correlation)``. Falls back to the sampled point
    at the array edges or when the three points are collinear.
    """
    if i <= 0 or i >= len(corr) - 1:
        return float(lags[i]), float(corr[i])
    y0, y1, y2 = float(corr[i - 1]), float(corr[i]), float(corr[i + 1])
    denom = y0 - 2.0 * y1 + y2
    if abs(denom) < 1e-12:
        return float(lags[i]), float(corr[i])
    delta = 0.5 * (y0 - y2) / denom          # in samples, |delta| <= 1
    peak = y1 - 0.25 * (y0 - y2) * delta
    step = float(lags[1] - lags[0])
    return float(lags[i] + delta * step), float(peak)


# ---------------------------------------------------------------------------
# HR cross-correlation → best lag + best correlation coefficient
# ---------------------------------------------------------------------------

def _hr_xcorr(
        ref_t: np.ndarray,
        ref_hr: np.ndarray,
        pred_t: np.ndarray,
        pred_hr: np.ndarray,
        lag_bound_s: float,
        fs: float = HR_XCORR_FS,
) -> dict:
    """Cross-correlate two HR traces; return the lag of the largest correlation.

    Both traces are resampled onto one uniform ``fs`` grid spanning their union,
    demeaned (out-of-support samples fall to zero), and cross-correlated. The
    correlation is normalised by ``sqrt(E_ref · E_pred)`` so, by Cauchy–Schwarz,
    every value is a correlation coefficient in ``[-1, 1]`` and the in-band argmax
    is the *best correlation coefficient*. The search is restricted to
    ``|lag| <= lag_bound_s`` and the peak is parabolically refined to sub-grid
    resolution.

    Sign: ``lag_s`` follows v1, i.e. the prediction is aligned to the reference by
    ``pred_beats + lag_s``.
    """
    empty = {"lag_s": 0.0, "corr": 0.0,
             "lags": np.array([]), "corr_curve": np.array([])}
    if ref_t.size < 2 or pred_t.size < 2:
        return empty

    t0 = float(min(ref_t[0], pred_t[0]))
    t1 = float(max(ref_t[-1], pred_t[-1]))
    grid = np.arange(t0, t1, 1.0 / fs)
    if grid.size < 4:
        return empty

    ref_g = _demean_zerofill(_resample_to_grid(ref_t, ref_hr, grid))
    pred_g = _demean_zerofill(_resample_to_grid(pred_t, pred_hr, grid))

    denom = float(np.sqrt(np.sum(ref_g * ref_g) * np.sum(pred_g * pred_g)))
    if denom <= 1e-12:
        return empty

    corr = correlate(ref_g, pred_g, mode='full') / denom
    lags = correlation_lags(len(ref_g), len(pred_g), mode='full') / fs

    mask = np.abs(lags) <= lag_bound_s
    if not np.any(mask):
        return empty
    sub_lags = lags[mask]
    sub_corr = corr[mask]

    idx = int(np.argmax(sub_corr))
    lag_s, peak = _parabolic_peak(sub_lags, sub_corr, idx)
    return {"lag_s": lag_s, "corr": peak, "lags": sub_lags, "corr_curve": sub_corr}


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ChannelEvalV3:
    label: str                       # "maternal" or "fetal"
    lag_s: float                     # HR-xcorr lag of the largest correlation
    best_corr: float                 # peak correlation coefficient (the score)
    n_ref: int                       # SOT beat count (in analysis window)
    n_pred: int                      # fiber beat count
    n_correct: int                   # beats within ±50 ms after lag correction
    n_accepted: int                  # beats within acceptance window (±250 ms, inlier)
    precision: float                 # n_correct / n_pred
    recall: float                    # n_correct / n_ref
    f1: float
    dt_signed: npt.NDArray[np.float64]     # signed timing errors (matched pairs)
    accepted_mask: npt.NDArray[np.bool_]
    matched_ref_t: npt.NDArray[np.float64]
    # HR traces (60/IBI) kept for plotting
    ref_hr_t: npt.NDArray[np.float64]
    ref_hr: npt.NDArray[np.float64]
    pred_hr_t: npt.NDArray[np.float64]
    pred_hr: npt.NDArray[np.float64]
    # HR cross-correlation curve
    xcorr_lags: npt.NDArray[np.float64] | None = None
    xcorr_corr: npt.NDArray[np.float64] | None = None
    band: Tuple[float, float] = FETAL_BPM_RANGE
    hr_smooth_used: int = HR_SMOOTH_PER_60S   # beats averaged into the HR trace
    # stored to allow combine / plotting over the full span
    ref_times: npt.NDArray[np.float64] | None = None
    pred_times: npt.NDArray[np.float64] | None = None
    t_start: float | None = None
    t_end: float | None = None
    lag_bound_s: float = 5.0


@dataclass
class EvaluationResultV3:
    maternal: Optional[ChannelEvalV3]
    fetal: ChannelEvalV3
    fetal_result: fHROutput

    @property
    def overall_score(self) -> float:
        """Overall score = the best HR correlation coefficient.

        The fetal channel is this pipeline's primary target, so its peak
        correlation is the headline number (the maternal coefficient is still
        computed, printed and plotted when a chest channel is present).
        """
        return self.fetal.best_corr


# ---------------------------------------------------------------------------
# Per-channel evaluation
# ---------------------------------------------------------------------------

def _eval_channel_v3(
        label: str,
        ref_times: np.ndarray,
        pred_times: np.ndarray,
        t_start: float,
        t_end: float,
        band: Tuple[float, float],
        lag_bound_s: float = 5.0,
        hr_fs: float = HR_XCORR_FS,
        hr_smooth: Optional[int] = None,
        xcorr_smooth: int = HR_XCORR_SMOOTH,
        correct_thr: float = 0.05,
) -> ChannelEvalV3:
    ref_times = np.sort(np.asarray(ref_times, dtype=float))
    pred_times = np.sort(np.asarray(pred_times, dtype=float))

    # DISPLAY smoothing scales with the window length (60 s -> 10 beats) unless the
    # caller pins it, so long recordings read as a trend rather than a jumpy point
    # cloud. The CORRELATION below uses its own fixed light `xcorr_smooth` so the
    # lag/score stay well-localised and comparable regardless of the display.
    disp_smooth = hr_smooth if hr_smooth is not None else _smooth_for_window(t_end - t_start)

    # Reference beats for the HR trace may reach a lag_bound margin past the
    # analysis window so the shifted correlation keeps support at the edges; the
    # timing-error scoring below uses only the in-window reference beats.
    margin = lag_bound_s + 2.0
    ref_hr_beats = ref_times[(ref_times >= t_start - margin) & (ref_times <= t_end + margin)]
    pred_win = pred_times[(pred_times >= t_start) & (pred_times <= t_end)]

    # Correlation traces (fixed light smoothing) -> lag + score.
    ref_xt, ref_xhr = _inst_hr(ref_hr_beats, band, xcorr_smooth)
    pred_xt, pred_xhr = _inst_hr(pred_win, band, xcorr_smooth)
    xc = _hr_xcorr(ref_xt, ref_xhr, pred_xt, pred_xhr, lag_bound_s, hr_fs)

    # Display traces (window-scaled smoothing) -> the HR panel.
    ref_hr_t, ref_hr = _inst_hr(ref_hr_beats, band, disp_smooth)
    pred_hr_t, pred_hr = _inst_hr(pred_win, band, disp_smooth)
    lag_s = xc["lag_s"] if np.isfinite(xc["lag_s"]) else 0.0
    best_corr = float(xc["corr"])

    # Timing errors: align the prediction by +lag (v1 convention), then robust-
    # match against the in-window SOT and count ±50 ms hits.
    ref_score = ref_times[(ref_times >= t_start) & (ref_times <= t_end)]
    match = _robust_match(ref_score, pred_win + lag_s)
    dt = np.asarray(match["dt_signed"], dtype=float)
    accepted = np.asarray(match["accepted_mask"], dtype=bool)
    dt_acc = dt[accepted]

    n_ref = len(ref_score)
    n_pred = len(pred_win)
    n_correct = int(np.sum(np.abs(dt_acc) <= correct_thr))
    n_accepted = int(np.sum(accepted))
    precision = n_correct / max(n_pred, 1)
    recall = n_correct / max(n_ref, 1)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return ChannelEvalV3(
        label=label,
        lag_s=lag_s,
        best_corr=best_corr,
        n_ref=n_ref,
        n_pred=n_pred,
        n_correct=n_correct,
        n_accepted=n_accepted,
        precision=precision,
        recall=recall,
        f1=f1,
        dt_signed=dt,
        accepted_mask=accepted,
        matched_ref_t=np.asarray(match["matched_ref_t"], dtype=float),
        ref_hr_t=ref_hr_t,
        ref_hr=ref_hr,
        pred_hr_t=pred_hr_t,
        pred_hr=pred_hr,
        xcorr_lags=xc["lags"],
        xcorr_corr=xc["corr_curve"],
        band=band,
        hr_smooth_used=disp_smooth,
        ref_times=ref_score,
        pred_times=pred_win,
        t_start=t_start,
        t_end=t_end,
        lag_bound_s=lag_bound_s,
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _hr_ylim(traces, band: Tuple[float, float], pad: float = 0.1):
    """Robust y-limits from HR values inside ``band`` (so spurious spikes from a
    missed/extra beat don't flatten the axis). Mirrors plot_hr._hr_ylim."""
    vals = np.concatenate([y for (_, y) in traces if y.size]) if traces else np.array([])
    vals = vals[(vals >= band[0]) & (vals <= band[1])]
    if vals.size == 0:
        return band
    lo, hi = float(np.min(vals)), float(np.max(vals))
    margin = pad * max(hi - lo, 1.0)
    return lo - margin, hi + margin


def _plot_channel_v3(
        ax_hr,
        ax_xcorr,
        ax_dt,
        ev: ChannelEvalV3,
        ref_color: str,
        pred_color: str,
        ref_label: str,
        pred_label: str,
        t_start: float,
        t_end: float,
) -> None:
    # --- HR panel: SOT and fiber 60/IBI traces overlaid. The prominent fiber
    #     trace is the *lag-aligned* one (shifted by the xcorr lag to meet the
    #     SOT); the raw, un-shifted fiber HR is kept as a pale dashed background.
    if ev.ref_hr_t.size:
        med = float(np.median(ev.ref_hr))
        ax_hr.plot(ev.ref_hr_t, ev.ref_hr, color=ref_color, lw=1.4, marker='o', ms=3,
                   alpha=0.9, label=f'{ref_label} (median {med:.1f})')
    if ev.pred_hr_t.size:
        med = float(np.median(ev.pred_hr))
        # Pale dashed background: raw fiber HR before lag alignment.
        ax_hr.plot(ev.pred_hr_t, ev.pred_hr, color=pred_color, lw=1.0, ls='--',
                   alpha=0.35, label=f'{pred_label} (raw)', zorder=2)
        # Prominent: lag-aligned fiber HR (shifted by +lag onto the SOT).
        ax_hr.plot(ev.pred_hr_t + ev.lag_s, ev.pred_hr, color=pred_color, lw=1.1,
                   marker='s', ms=3, alpha=0.85, zorder=4,
                   label=f'{pred_label} +lag {ev.lag_s:+.3f}s (median {med:.1f})')
    ax_hr.set_ylim(*_hr_ylim([(ev.ref_hr_t, ev.ref_hr), (ev.pred_hr_t, ev.pred_hr)], ev.band))
    ax_hr.set_xlim(t_start, t_end)
    ax_hr.set_xlabel("Time (s)", fontsize=7)
    ax_hr.set_ylabel("Instantaneous HR (BPM)", fontsize=7)
    ax_hr.grid(True, alpha=0.25)
    ax_hr.legend(loc='upper right', fontsize=7)
    ax_hr.set_title(
        f"{ev.label} HR (60/IBI, smooth={ev.hr_smooth_used} beats) -- "
        f"best corr={ev.best_corr:.3f} @ lag={ev.lag_s:+.3f}s",
        fontsize=8,
    )

    # --- XCorr panel (like v1): the HR correlation-coefficient curve, with the
    #     best lag marked at the peak.
    if ev.xcorr_lags is not None and len(ev.xcorr_lags):
        ax_xcorr.plot(ev.xcorr_lags, ev.xcorr_corr, color='0.2', lw=1.2)
        ax_xcorr.axvline(ev.lag_s, color='crimson', lw=1.2, ls='--',
                         label=f'lag={ev.lag_s:+.3f}s')
    ax_xcorr.axhline(0, color='0.7', lw=0.6, ls=':')
    ax_xcorr.axvline(0, color='0.7', lw=0.6, ls=':')
    ax_xcorr.set_xlabel("Lag (s)", fontsize=7)
    ax_xcorr.set_ylabel("HR corr coef", fontsize=7)
    ax_xcorr.legend(loc='upper right', fontsize=7)
    ax_xcorr.set_title(f"{ev.label} HR xcorr  (peak r={ev.best_corr:.3f})", fontsize=8)

    # --- Timing-error panel (like v1/v2): signed dt scatter with the ±50/±250 ms
    #     bands, accepted vs rejected, and the accepted median.
    dt = ev.dt_signed
    accepted = ev.accepted_mask
    ref_t = ev.matched_ref_t
    if len(dt):
        ax_dt.scatter(ref_t[accepted], dt[accepted], s=22, color='tab:blue', zorder=5,
                      label=f'accepted n={accepted.sum()}')
        if np.any(~accepted):
            ax_dt.scatter(ref_t[~accepted], dt[~accepted], s=30, marker='x',
                          color='tab:red', zorder=5, label=f'rejected n={(~accepted).sum()}')
        ax_dt.axhline(0.0, color='k', lw=0.8)
        ax_dt.axhspan(-0.05, 0.05, color='tab:green', alpha=0.15, label='±50 ms')
        ax_dt.axhspan(-0.25, 0.25, color='tab:orange', alpha=0.08, label='±250 ms')
        med = float(np.median(dt[accepted])) if accepted.any() else 0.0
        ax_dt.axhline(med, color='0.4', ls='--', lw=0.9, label=f'median={med:.3f}s')
    ax_dt.set_xlim(t_start, t_end)
    ax_dt.set_xlabel("Ref time (s)", fontsize=7)
    ax_dt.set_ylabel("dt (s)", fontsize=7)
    ax_dt.legend(loc='upper right', fontsize=7)
    ax_dt.set_title(f"{ev.label} timing error  correct={ev.n_correct}/{ev.n_ref}", fontsize=8)


def plot_evaluation_v3(
        ev_result: EvaluationResultV3,
        out: Path,
) -> None:
    m = ev_result.maternal
    f = ev_result.fetal

    fig = plt.figure(figsize=(20, 8), constrained_layout=True)
    gs = fig.add_gridspec(2, 4)

    ax_m_hr = fig.add_subplot(gs[0, 0:2])
    ax_m_xcorr = fig.add_subplot(gs[0, 2])
    ax_m_dt = fig.add_subplot(gs[0, 3])

    ax_f_hr = fig.add_subplot(gs[1, 0:2])
    ax_f_xcorr = fig.add_subplot(gs[1, 2])
    ax_f_dt = fig.add_subplot(gs[1, 3])

    if m is not None:
        _plot_channel_v3(
            ax_m_hr, ax_m_xcorr, ax_m_dt, m,
            ref_color='tab:blue', pred_color='tab:orange',
            ref_label='PPG (SOT)', pred_label='Fiber chest',
            t_start=float(m.t_start), t_end=float(m.t_end),
        )
    else:
        for ax in (ax_m_hr, ax_m_xcorr, ax_m_dt):
            ax.set_axis_off()

    _plot_channel_v3(
        ax_f_hr, ax_f_xcorr, ax_f_dt, f,
        ref_color='tab:red', pred_color='tab:green',
        ref_label='Mic (SOT)', pred_label='Fiber fetal',
        t_start=float(f.t_start), t_end=float(f.t_end),
    )

    m_corr = m.best_corr if m is not None else float('nan')
    fig.suptitle(
        f"HR cross-correlation score (best corr coef)   "
        f"Overall (fetal)={ev_result.overall_score:.3f}   |   "
        f"Maternal={m_corr:.3f}  acc@50ms={ev_result.maternal.recall if m is not None else 0:.1%}   |   "
        f"Fetal={f.best_corr:.3f}  acc@50ms={f.recall:.1%}",
        fontsize=10,
    )
    plt.savefig(out / "evaluation_v3.png", dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Pipeline stage factory
# ---------------------------------------------------------------------------

def evaluate_v3(
        sot: SOTResult,
        out: Path,
        lag_bound_s: float = 5.0,
        hr_fs: float = HR_XCORR_FS,
        hr_smooth: Optional[int] = None,
        xcorr_smooth: int = HR_XCORR_SMOOTH,
        fetal_band: Tuple[float, float] = FETAL_BPM_RANGE,
        maternal_band: Tuple[float, float] = MATERNAL_HR_BAND,
):
    """Pipeline stage factory: HR-cross-correlation evaluation.

    For each channel it builds the 60/IBI heart-rate trace of the SOT reference
    and the fiber prediction, cross-correlates the two HR traces to find the lag
    of the largest correlation, then scores the timing errors like v1/v2 (correct
    the prediction by that lag, robust-match, count ±50 ms hits). The overall
    score is the best (peak) HR correlation coefficient — see
    ``EvaluationResultV3.overall_score``.

    ``hr_smooth`` is the beats averaged into the *displayed* HR trace. Left as
    ``None`` it scales with the analysis-window length (``HR_SMOOTH_PER_60S`` beats
    per 60 s, e.g. 10 at 60 s, 20 at 120 s) so long recordings read as a trend
    rather than a jumpy point cloud; pass an int to pin it. The *correlation* (lag
    + score) uses its own fixed light ``xcorr_smooth`` so it stays comparable
    across window lengths regardless of the display smoothing.

    Like ``evaluate_v2``, ``sot`` is the **full**, un-windowed SOT (the analysis
    window is taken from the fiber result); scoring is restricted to that window
    while the out-of-window SOT only supports the lag search near the edges.

    Accepts a ``fHROutput`` directly, or any result wrapping one via
    ``.fetal_result`` (e.g. an EvaluationResult / EvaluationResultV2), so it can
    also be dropped in after another evaluator. Returns ``EvaluationResultV3``.
    """
    out = Path(out)

    def _span(audio) -> Tuple[float, float]:
        return float(audio.time[0]), float(audio.time[-1])

    def run_evaluate_v3(result) -> EvaluationResultV3:
        fetal_result: fHROutput = getattr(result, "fetal_result", result)
        out.mkdir(parents=True, exist_ok=True)

        maternal_ev = None
        if fetal_result.maternal_source is not None and sot.ppg_beats is not None:
            m_t0, m_t1 = _span(fetal_result.maternal_source)
            maternal_ev = _eval_channel_v3(
                "maternal",
                ref_times=sot.ppg_beats,
                pred_times=fetal_result.maternal_beats,
                t_start=m_t0,
                t_end=m_t1,
                band=maternal_band,
                lag_bound_s=lag_bound_s,
                hr_fs=hr_fs,
                hr_smooth=hr_smooth,
                xcorr_smooth=xcorr_smooth,
            )

        f_t0, f_t1 = _span(fetal_result.fetal_source)
        fetal_ev = _eval_channel_v3(
            "fetal",
            ref_times=sot.mic_beats,
            pred_times=fetal_result.fetal_beats,
            t_start=f_t0,
            t_end=f_t1,
            band=fetal_band,
            lag_bound_s=lag_bound_s,
            hr_fs=hr_fs,
            hr_smooth=hr_smooth,
            xcorr_smooth=xcorr_smooth,
        )

        result_v3 = EvaluationResultV3(maternal=maternal_ev, fetal=fetal_ev, fetal_result=fetal_result)
        plot_evaluation_v3(result_v3, out)

        for ev in (maternal_ev, fetal_ev):
            if ev is not None:
                print(
                    f"  {ev.label.capitalize():8s}  best-corr={ev.best_corr:+.3f}"
                    f"  lag={ev.lag_s:+.3f}s"
                    f"  ref={ev.n_ref}  pred={ev.n_pred}"
                    f"  correct={ev.n_correct}/{ev.n_ref} ({ev.recall:.1%})"
                    f"  F1={ev.f1:.2f}"
                )
        print(f"  Overall score (best corr coef): {result_v3.overall_score:.3f}")

        return result_v3

    run_evaluate_v3.__name__ = "evaluate_v3"
    return run_evaluate_v3
