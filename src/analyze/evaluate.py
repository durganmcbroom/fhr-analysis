from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
from fontTools.qu2cu.qu2cu import List
from matplotlib import pyplot as plt

from matplotlib.transforms import blended_transform_factory
from scipy.signal import correlate, correlation_lags

from analyze.data import Audio
from analyze.hr import fHROutput
from analyze.sot import SOTResult
from constants import XCORR_TARGET_FS


# ---------------------------------------------------------------------------
# Signal utilities (standalone copies; avoid importing private symbols)
# ---------------------------------------------------------------------------

def _gaussian_smooth(x: np.ndarray, sigma_samples: float) -> np.ndarray:
    sigma_samples = max(1.0, float(sigma_samples))
    radius = int(3 * sigma_samples)
    grid = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (grid / sigma_samples) ** 2)
    kernel /= kernel.sum()
    return np.convolve(x, kernel, mode='same')


def _impulse_train(times: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
    impulse = np.zeros_like(t_grid, dtype=float)
    if len(times) == 0:
        return impulse
    dt = t_grid[1] - t_grid[0]
    idx = np.round((times - t_grid[0]) / dt).astype(int)
    idx = idx[(idx >= 0) & (idx < len(t_grid))]
    impulse[idx] = 1.0
    return impulse


# ---------------------------------------------------------------------------
# Cross-correlation lag estimation
# Port of impulse_xcorr_details (clean_data_template.py:1573)
# ---------------------------------------------------------------------------

def _xcorr_lag(
        ref_times: np.ndarray,
        pred_times: np.ndarray,
        t_start: float,
        t_end: float,
        lag_bound_s: float = 5.0,
        target_fs: float = XCORR_TARGET_FS,
        smooth_sigma_s: float = 0.08,
) -> dict:
    """Estimate lag of pred relative to ref via impulse-train cross-correlation.

    Positive lag_s means pred events occur lag_s seconds after ref events.
    Correct pred by subtracting lag_s to align with ref.
    """
    t_grid = np.arange(t_start, t_end, 1.0 / target_fs)
    ref_imp = _gaussian_smooth(_impulse_train(ref_times, t_grid), smooth_sigma_s * target_fs)
    pred_imp = _gaussian_smooth(_impulse_train(pred_times, t_grid), smooth_sigma_s * target_fs)

    corr = correlate(ref_imp, pred_imp, mode='full')
    lags = correlation_lags(len(ref_imp), len(pred_imp), mode='full') / target_fs
    mask = (lags >= -lag_bound_s) & (lags <= lag_bound_s) & (lags <= 0)
    if not np.any(mask):
        return {"lag_s": 0.0, "score": 0.0, "lags": np.array([]), "corr": np.array([])}

    sub_corr = corr[mask]
    sub_lags = lags[mask]
    idx = int(np.argmax(sub_corr))
    lag_s = float(sub_lags[idx])
    score = float(sub_corr[idx] / (np.std(sub_corr) + 1e-9))
    return {"lag_s": lag_s, "score": score, "lags": sub_lags, "corr": sub_corr}


# ---------------------------------------------------------------------------
# Beat matching with MAD-based outlier rejection
# Faithful port of robust_match_peaks (clean_data_template.py:1966)
# ---------------------------------------------------------------------------

def _robust_match(
        ref_times: np.ndarray,
        pred_times: np.ndarray,
        max_abs_dt: float = 0.25,
        outlier_k: float = 3.5,
) -> dict:
    ref_times = np.sort(np.asarray(ref_times, float))
    pred_times = np.sort(np.asarray(pred_times, float))
    ref_times = ref_times[np.isfinite(ref_times)]
    pred_times = pred_times[np.isfinite(pred_times)]

    used = np.zeros(len(pred_times), dtype=bool)
    matched_ref, matched_pred, dt = [], [], []

    for tr in ref_times:
        left = np.searchsorted(pred_times, tr - max_abs_dt, side='left')
        right = np.searchsorted(pred_times, tr + max_abs_dt, side='right')
        if right <= left:
            continue
        cand = np.arange(left, right)
        cand = cand[~used[cand]]
        if cand.size == 0:
            continue
        j = cand[np.argmin(np.abs(pred_times[cand] - tr))]
        used[j] = True
        matched_ref.append(float(tr))
        matched_pred.append(float(pred_times[j]))
        dt.append(float(pred_times[j] - tr))

    if not dt:
        return {
            "matched_ref_t": np.array([], dtype=float),
            "matched_pred_t": np.array([], dtype=float),
            "dt_signed": np.array([], dtype=float),
            "accepted_mask": np.array([], dtype=bool),
            "median_dt": np.nan,
            "sigma_mad": np.nan,
        }

    dt = np.asarray(dt, dtype=float)
    med = float(np.median(dt))
    mad = float(np.median(np.abs(dt - med)))
    sigma = 1.4826 * mad
    if sigma <= 1e-12:
        sigma = float(np.std(dt)) if np.std(dt) > 0 else 1e-6
    accepted_mask = np.abs(dt - med) <= outlier_k * sigma

    return {
        "matched_ref_t": np.asarray(matched_ref),
        "matched_pred_t": np.asarray(matched_pred),
        "dt_signed": dt,
        "accepted_mask": accepted_mask,
        "median_dt": med,
        "sigma_mad": sigma,
    }


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ChannelEval:
    label: str  # "maternal" or "fetal"
    lag_s: float  # estimated pred-relative-to-ref lag
    xcorr_score: float  # normalised xcorr peak height
    n_ref: int  # SOT beat count
    n_pred: int  # fiber beat count
    n_correct: int  # beats within ±50 ms after lag correction
    n_accepted: int  # beats within acceptance window (±250 ms, inlier)
    precision: float  # n_correct / n_pred
    recall: float  # n_correct / n_ref
    f1: float
    dt_signed: npt.NDArray[np.float64]  # signed time errors (accepted pairs)
    accepted_mask: npt.NDArray[np.bool_]
    matched_ref_t: npt.NDArray[np.float64]
    xcorr_lags: npt.NDArray[np.float64] | None = None
    xcorr_corr: npt.NDArray[np.float64] | None = None
    # stored to allow combine_evaluations to recompute xcorr over the full span
    ref_times: npt.NDArray[np.float64] | None = None
    pred_times: npt.NDArray[np.float64] | None = None
    t_start: float | None = None
    t_end: float | None = None
    lag_bound_s: float = 5.0


@dataclass
class EvaluationResult:
    maternal: ChannelEval
    fetal: ChannelEval
    fetal_result: fHROutput


# ---------------------------------------------------------------------------
# Per-channel evaluation
# ---------------------------------------------------------------------------

def _eval_channel(
        label: str,
        ref_times: np.ndarray,
        pred_times: np.ndarray,
        t_start: float,
        t_end: float,
        lag_bound_s: float = 5.0,
) -> ChannelEval:
    ref_times = np.asarray(ref_times, float)
    pred_times = np.asarray(pred_times, float)

    xcorr = _xcorr_lag(ref_times, pred_times, t_start, t_end, lag_bound_s)
    lag_s = xcorr["lag_s"] if np.isfinite(xcorr["lag_s"]) else 0.0

    # Shift pred into ref time frame before matching
    match = _robust_match(ref_times, pred_times + lag_s)
    dt = np.asarray(match["dt_signed"], dtype=float)
    accepted = np.asarray(match["accepted_mask"], dtype=bool)
    dt_acc = dt[accepted]

    n_ref = len(ref_times)
    n_pred = len(pred_times)
    n_correct = int(np.sum(np.abs(dt_acc) <= 0.05))
    n_accepted = int(np.sum(accepted))
    precision = n_correct / max(n_pred, 1)
    recall = n_correct / max(n_ref, 1)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return ChannelEval(
        label=label,
        lag_s=lag_s,
        xcorr_score=xcorr["score"],
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
        xcorr_lags=xcorr["lags"],
        xcorr_corr=xcorr["corr"],
        ref_times=ref_times,
        pred_times=pred_times,
        t_start=t_start,
        t_end=t_end,
        lag_bound_s=lag_bound_s,
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_channel(
        ax_signal,
        ax_xcorr,
        ax_dt,
        raw_signal,
        sot_signal,
        ref_times: np.ndarray,
        pred_times: np.ndarray,
        ev: ChannelEval,
        ref_color: str,
        pred_color: str,
        ref_label: str,
        pred_label: str,
        t_start: float,
        t_end: float,
) -> None:
    signal_frac = 0.40
    pad = 0.25  # fraction of amp added as breathing room at each outer edge

    fiber_data = raw_signal.data - float(np.mean(raw_signal.data))
    sot_data   = sot_signal.data - float(np.mean(sot_signal.data))
    fiber_amp  = max(float(np.percentile(np.abs(fiber_data), 99)), 1e-12)
    sot_amp    = max(float(np.percentile(np.abs(sot_data),   99)), 1e-12)

    ax_signal.plot(raw_signal.time, fiber_data, color=pred_color, lw=0.6, alpha=0.45)
    ax_signal.set_ylim(-fiber_amp * (1 + pad), -fiber_amp + 2 * fiber_amp / signal_frac)
    ax_signal.set_yticks([])

    ax2 = ax_signal.twinx()
    ax2.plot(sot_signal.time - ev.lag_s, sot_data, color=ref_color, lw=0.6, alpha=0.45)
    ax2.set_ylim(sot_amp - 2 * sot_amp / signal_frac, sot_amp * (1 + pad))
    ax2.set_yticks([])

    trans = blended_transform_factory(ax_signal.transData, ax_signal.transAxes)
    ax_signal.vlines(pred_times,           0, 1, transform=trans,
                     color=pred_color, lw=0.9, ls='--', alpha=0.65)
    ax_signal.vlines(ref_times - ev.lag_s, 0, 1, transform=trans,
                     color=ref_color,  lw=0.9, ls=':',  alpha=0.65)

    ax_signal.set_xlim(float(raw_signal.time[0]), float(raw_signal.time[-1]))

    fiber_ylim  = ax_signal.get_ylim()
    sot_ylim    = ax2.get_ylim()
    fiber_yfrac = (0.0 - fiber_ylim[0]) / (fiber_ylim[1] - fiber_ylim[0])
    sot_yfrac   = (0.0 - sot_ylim[0])   / (sot_ylim[1]   - sot_ylim[0])

    ax_signal.text(
        -0.02, fiber_yfrac, pred_label,
        transform=ax_signal.transAxes, ha='right', va='center',
        color=pred_color, fontsize=8, fontweight='bold', clip_on=False,
    )
    ax_signal.text(
        -0.02, sot_yfrac, ref_label,
        transform=ax_signal.transAxes, ha='right', va='center',
        color=ref_color, fontsize=8, fontweight='bold', clip_on=False,
    )

    ax_signal.set_title(
        f"{ev.label} -- lag={ev.lag_s:+.3f}s  acc@50ms={ev.recall:.1%}  F1={ev.f1:.2f}", fontsize=8
    )
    # XCorr curve — use the stored result from eval time so the crimson line
    # is guaranteed to sit exactly at the peak of the plotted curve.
    if ev.xcorr_lags is not None and len(ev.xcorr_lags):
        ax_xcorr.plot(ev.xcorr_lags, ev.xcorr_corr, color='0.2', lw=1.2)
        ax_xcorr.axvline(ev.lag_s, color='crimson', lw=1.2, ls='--',
                         label=f'lag={ev.lag_s:+.3f}s')
    ax_xcorr.axvline(0, color='0.7', lw=0.6, ls=':')
    ax_xcorr.set_xlabel("Lag (s)", fontsize=7)
    ax_xcorr.set_ylabel("XCorr", fontsize=7)
    ax_xcorr.legend(loc='upper right', fontsize=7)
    ax_xcorr.set_title(f"{ev.label} xcorr  (score={ev.xcorr_score:.1f})", fontsize=8)

    # dt scatter
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


def plot_evaluation(
        ev_result: EvaluationResult,
        sot: SOTResult,
        out: Path,
) -> None:
    t_start = float(sot.ppg.time[0])
    t_end = float(sot.ppg.time[-1])

    fig = plt.figure(figsize=(20, 8), constrained_layout=True)
    gs = fig.add_gridspec(2, 4)

    ax_m_sig = fig.add_subplot(gs[0, 0:2])
    ax_m_xcorr = fig.add_subplot(gs[0, 2])
    ax_m_dt = fig.add_subplot(gs[0, 3])

    ax_f_sig = fig.add_subplot(gs[1, 0:2])
    ax_f_xcorr = fig.add_subplot(gs[1, 2])
    ax_f_dt = fig.add_subplot(gs[1, 3])

    _plot_channel(
        ax_m_sig, ax_m_xcorr, ax_m_dt,
        raw_signal=ev_result.fetal_result.maternal_source,
        sot_signal=sot.ppg,
        ref_times=sot.ppg_beats,
        pred_times=ev_result.fetal_result.maternal_beats,
        ev=ev_result.maternal,
        ref_color='tab:blue', pred_color='tab:orange',
        ref_label='PPG (SOT)', pred_label='Fiber chest',
        t_start=t_start, t_end=t_end,
    )
    _plot_channel(
        ax_f_sig, ax_f_xcorr, ax_f_dt,
        raw_signal=ev_result.fetal_result.fetal_source,
        sot_signal=sot.mic,
        ref_times=sot.mic_beats,
        pred_times=ev_result.fetal_result.fetal_beats,
        ev=ev_result.fetal,
        ref_color='tab:red', pred_color='tab:green',
        ref_label='Mic (SOT)', pred_label='Fiber fetal',
        t_start=t_start, t_end=t_end,
    )

    fig.suptitle(
        f"Maternal — acc@50ms={ev_result.maternal.recall:.1%}  F1={ev_result.maternal.f1:.2f}   |   "
        f"Fetal — acc@50ms={ev_result.fetal.recall:.1%}  F1={ev_result.fetal.f1:.2f}",
        fontsize=10,
    )
    plt.savefig(out / "evaluation.png", dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Pipeline stage factory
# ---------------------------------------------------------------------------

def evaluate(
        sot: SOTResult,
        out_dir: str,
        lag_bound_s: float = 5.0,
):
    """Pipeline stage factory: cross-correlate fiber detections against SOT and score.

    Takes FetalHRResult (output of fetal_hr stage), returns EvaluationResult.
    SOT is captured from the sot_pipe run in model.py.
    """

    def run_evaluate(result: fHROutput) -> EvaluationResult:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)

        t_start = float(min(
            sot.ppg.time[0] if len(sot.ppg.time) else 0,
            sot.mic.time[0] if len(sot.mic.time) else 0,
        ))
        t_end = float(max(
            sot.ppg.time[-1] if len(sot.ppg.time) else 1,
            sot.mic.time[-1] if len(sot.mic.time) else 1,
        ))

        maternal_ev = _eval_channel(
            "maternal",
            ref_times=sot.ppg_beats,
            pred_times=result.maternal_beats,
            t_start=t_start,
            t_end=t_end,
            lag_bound_s=lag_bound_s,
        )
        fetal_ev = _eval_channel(
            "fetal",
            ref_times=sot.mic_beats,
            pred_times=result.fetal_beats,
            t_start=t_start,
            t_end=t_end,
            lag_bound_s=lag_bound_s,
        )

        result = EvaluationResult(maternal=maternal_ev, fetal=fetal_ev, fetal_result=result)
        plot_evaluation(result, sot, out)

        for ev in (maternal_ev, fetal_ev):
            print(
                f"  {ev.label.capitalize():8s}  lag={ev.lag_s:+.3f}s  xcorr={ev.xcorr_score:.1f}"
                f"  ref={ev.n_ref}  pred={ev.n_pred}"
                f"  correct={ev.n_correct}/{ev.n_ref} ({ev.recall:.1%})"
                f"  F1={ev.f1:.2f}"
            )

        return result

    run_evaluate.__name__ = "evaluate"
    return run_evaluate

def combine_evaluations(
        results: List[EvaluationResult],
) -> EvaluationResult:
    def combine_channels(label: str, ev: List[ChannelEval]):
        combined_ref = np.concatenate([e.ref_times for e in ev])
        combined_pred = np.concatenate([e.pred_times for e in ev])
        t_start = float(min(e.t_start for e in ev))  # type: ignore[arg-type]
        t_end = float(max(e.t_end for e in ev))      # type: ignore[arg-type]
        lag_bound_s = float(ev[0].lag_bound_s)
        xcorr = _xcorr_lag(combined_ref, combined_pred, t_start, t_end, lag_bound_s)
        lag_s = xcorr["lag_s"] if np.isfinite(xcorr["lag_s"]) else 0.0

        n_ref = sum(e.n_ref for e in ev)
        n_pred = sum(e.n_pred for e in ev)
        n_correct = sum(e.n_correct for e in ev)
        precision = n_correct / max(n_pred, 1)
        recall = n_correct / max(n_ref, 1)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return ChannelEval(
            label=label,
            lag_s=lag_s,
            xcorr_score=xcorr["score"],
            n_ref=n_ref,
            n_pred=n_pred,
            n_correct=n_correct,
            n_accepted=sum(e.n_accepted for e in ev),
            precision=precision,
            recall=recall,
            f1=f1,
            dt_signed=np.concatenate([e.dt_signed for e in ev]),
            accepted_mask=np.concatenate([e.accepted_mask for e in ev]),
            matched_ref_t=np.concatenate([e.matched_ref_t for e in ev]),
            xcorr_lags=xcorr["lags"],
            xcorr_corr=xcorr["corr"],
            ref_times=combined_ref,
            pred_times=combined_pred,
            t_start=t_start,
            t_end=t_end,
            lag_bound_s=lag_bound_s,
        )

    return EvaluationResult(
        maternal=combine_channels("maternal", [e.maternal for e in results]),
        fetal=combine_channels("fetal", [e.fetal for e in results]),
        fetal_result=fHROutput(
            Audio(
                np.concatenate([e.fetal_result.fetal_source.time for e in results]),
                results[0].fetal_result.fetal_source.hz,
                np.concatenate([e.fetal_result.fetal_source.data for e in results]),
            ),
            np.concatenate([e.fetal_result.fetal_beats for e in results]),
            Audio(
                np.concatenate([e.fetal_result.maternal_source.time for e in results]),
                results[0].fetal_result.maternal_source.hz,
                np.concatenate([e.fetal_result.maternal_source.data for e in results]),
            ),
            np.concatenate([e.fetal_result.maternal_beats for e in results]),
        ),
    )