from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import numpy.typing as npt
from matplotlib import pyplot as plt
from matplotlib.transforms import blended_transform_factory
from scipy.signal import correlate, correlation_lags
from scipy.special import expit

from analyze.data import Audio
from analyze.hr import fHROutput
from analyze.sot import SOTResult
from constants import XCORR_TARGET_FS

# Reuse the v1 beat-matcher verbatim so the matching semantics (MAD outlier
# rejection, ±250 ms acceptance) stay identical and the two evaluators remain
# directly comparable. Only the *delay estimation* and *scoring* differ in v2.
from analyze.evaluate import _robust_match


# ---------------------------------------------------------------------------
# v2 design notes
# ---------------------------------------------------------------------------
# Two changes relative to evaluate.py:
#
#   1. Sigmoid (soft-step) representation instead of impulse trains.
#      Each beat is rendered as a smoothed box-car — a rising sigmoid minus a
#      falling sigmoid — of half-width `half_width_s`. Cross-correlating these
#      soft plateaus (rather than narrow Gaussian-smoothed deltas) gives a
#      broader, more stable alignment peak, and the final score is a sigmoid of
#      the timing error rather than a hard ±50 ms indicator.
#
#   2. Per-window delay. The recording is split into `window_s` (5 s) windows,
#      each refining its own best-fit lag. This tracks the slowly-accumulating
#      SOT drift (caused by dropped samples) far better than a single global lag.
#
# Directionality + monotonicity. The SOT and fiber start synced (lag = 0), and
# the SOT only ever *drops* samples, so the prediction can only fall further
# behind: |lag| increases monotonically and never recovers. Concretely the lag
# is <= 0 (correct via `pred + lag`, as in v1) and monotonically non-increasing
# across windows. Starting from 0 and only allowing a sub-IBI downward step per
# window also dispatches the beat-aliasing problem for free — the estimate
# cannot hop to the true ± one-beat-interval correlation peak.

# ---------------------------------------------------------------------------
# Sigmoid (soft-step) beat representation
# ---------------------------------------------------------------------------

def _sigmoid_train(
        times: np.ndarray,
        t_grid: np.ndarray,
        half_width_s: float = 0.05,
        edge_s: float = 0.015,
) -> np.ndarray:
    """Render beats as a sum of soft box-cars on `t_grid`.

    Each beat contributes ``sigmoid((t - (tb - w))/s) - sigmoid((t - (tb + w))/s)``
    — a plateau of half-width ``w = half_width_s`` with sigmoid edges of scale
    ``s = edge_s``. This is the "step/sigmoid instead of impulse train"
    representation used for both alignment and scoring.
    """
    train = np.zeros_like(t_grid, dtype=float)
    times = np.asarray(times, dtype=float)
    times = times[np.isfinite(times)]
    if times.size == 0:
        return train
    # Beat counts per window are small (~10), so a simple loop is plenty fast
    # and keeps memory flat regardless of grid length.
    for tb in times:
        train += (expit((t_grid - (tb - half_width_s)) / edge_s)
                  - expit((t_grid - (tb + half_width_s)) / edge_s))
    return train


def _sigmoid_score(dt: np.ndarray, thr: float = 0.05, tau: float = 0.05) -> np.ndarray:
    """Soft, lenient version of the hard |dt| <= thr indicator.

    Flat **1.0 for every beat inside the ±`thr` window** (so the soft count never
    penalises a hard-correct beat ⇒ ``soft_recall >= recall`` always), then a
    gentle **sigmoid tail** for near-misses outside it. ``tau`` sets the tail
    width — larger is more lenient. With the default 50 ms the tail still gives
    meaningful credit across the ±250 ms acceptance window (≈0.54 at 100 ms,
    ≈0.24 at 150 ms, ≈0.04 at 250 ms) instead of collapsing just past 50 ms.

    Implemented as a sigmoid scaled by 2 and clamped at 1: inside the window the
    scaled sigmoid exceeds 1 and saturates (the flat top); outside it decays.
    """
    a = np.abs(np.asarray(dt, dtype=float))
    return np.minimum(1.0, 2.0 * expit((thr - a) / tau))


# ---------------------------------------------------------------------------
# Per-window directional lag estimation
# ---------------------------------------------------------------------------

def _window_lag(
        ref_w_times: np.ndarray,
        pred_ext_times: np.ndarray,
        grid_lo: float,
        grid_hi: float,
        lag_lo: float,
        lag_hi: float,
        target_fs: float = XCORR_TARGET_FS,
        half_width_s: float = 0.05,
        edge_s: float = 0.015,
) -> dict:
    """Best-fit lag aligning `pred` to `ref` over the lag band ``[lag_lo, lag_hi]``.

    `ref_w_times` are the reference beats *inside* the window; `pred_ext_times`
    are predictions spanning the window plus a tail to the right (the prediction
    is late, so its matching beats sit later in time). Both ``lag_lo`` and
    ``lag_hi`` are <= 0 — the same one direction as v1 (``pred + lag``).

    The band is kept narrower than one beat interval so the in-band argmax
    cannot hop to the quasi-periodic alias (a near-equal correlation peak at the
    true lag ± a beat interval). The caller slides the band downward window by
    window, which is what makes that safe.

    Returns ``lag_s`` (the in-band argmax) and a normalised ``score`` (peak
    height over the in-band spread) used to decide whether to trust it.
    """
    t_grid = np.arange(grid_lo, grid_hi, 1.0 / target_fs)
    if t_grid.size < 4:
        return {"lag_s": np.nan, "score": 0.0}

    ref_sig = _sigmoid_train(ref_w_times, t_grid, half_width_s, edge_s)
    pred_sig = _sigmoid_train(pred_ext_times, t_grid, half_width_s, edge_s)
    if not np.any(ref_sig) or not np.any(pred_sig):
        return {"lag_s": np.nan, "score": 0.0}

    corr = correlate(ref_sig, pred_sig, mode='full')
    lags = correlation_lags(len(ref_sig), len(pred_sig), mode='full') / target_fs
    mask = (lags >= lag_lo) & (lags <= lag_hi)
    if not np.any(mask):
        return {"lag_s": np.nan, "score": 0.0}

    sub_corr = corr[mask]
    sub_lags = lags[mask]
    idx = int(np.argmax(sub_corr))
    lag_s = float(sub_lags[idx])
    score = float(sub_corr[idx] / (np.std(sub_corr) + 1e-9))
    return {"lag_s": lag_s, "score": score}


def _median_ibi(times: np.ndarray, lo: float = 0.05, hi: float = 2.0) -> float:
    """Median inter-beat interval, ignoring implausible gaps."""
    ibi = np.diff(np.sort(np.asarray(times, dtype=float)))
    ibi = ibi[(ibi > lo) & (ibi < hi)]
    return float(np.median(ibi)) if ibi.size else 0.5


def _window_starts(t_start: float, t_end: float, window_s: float, hop_s: float) -> Tuple[list, float]:
    """Start times for fixed-width `window_s` windows stepping by `hop_s`.

    Every window is *exactly* ``window_s`` long — never shorter, never longer.
    Windows start at ``t_start`` and step by ``hop_s``. The default ``hop_s ==
    window_s`` tiles the recording into contiguous, non-overlapping windows, so
    that shifting each window by its own lag opens a clean gap between windows
    (see `_plot_channel_v2`). The final window is anchored to end exactly at
    ``t_end``, so the tail is always covered by a full-length window rather than a
    truncated remainder. Recordings shorter than one window collapse to a single
    best-effort window spanning the whole recording.

    Returns ``(starts, win_len)`` — ``win_len`` equals ``window_s`` except in the
    degenerate short-recording case.
    """
    span = t_end - t_start
    if span <= window_s:
        return [t_start], max(span, 0.0)

    last_start = t_end - window_s
    starts, k = [], 0
    while t_start + k * hop_s < last_start - 1e-9:
        starts.append(t_start + k * hop_s)
        k += 1
    starts.append(last_start)
    return starts, window_s


def _estimate_window_lags(
        ref_times: np.ndarray,
        pred_times: np.ndarray,
        t_start: float,
        t_end: float,
        window_s: float = 5.0,
        hop_s: Optional[float] = None,
        lag_bound_s: float = 5.0,
        min_window_score: float = 2.0,
        min_ref_in_window: int = 2,
        half_width_s: float = 0.05,
        edge_s: float = 0.015,
) -> dict:
    """Best-fit lag over `window_s` windows: baseline offset + monotone drift.

    ``ref_times`` is the **full SOT** (not windowed to ``[t_start, t_end]``), so
    every correlation has reference beats beyond the analysis-window edges and is
    never truncated there. The windows themselves still tile only the analysis
    window ``[t_start, t_end]``.

    The lag is modelled as ``lag(t) = baseline + drift(t)``:

      * **Baseline** — a *fixed* fiber-vs-SOT delay present from the start, of
        *either sign* (e.g. pulse transit time makes the chest fiber lead the
        PPG). The first confident window searches symmetrically within
        ``±band_w``. The full SOT matters here: the search reaches ``band_w``
        past the window's left edge with real reference beats instead of running
        off the end of a pre-windowed SOT. (The search stays within ``±band_w``,
        i.e. under half a beat interval, rather than going wide — a wide search
        would just lock onto a quasi-periodic alias.)

      * **Drift** — the SOT then only drops samples, so from the baseline the
        lag can only *decrease*. Once the baseline is locked every window
        searches *downward only*, within ``[carry - band_w, carry]``.

    ``band_w = 0.4·IBI`` is narrower than half a beat interval, so no single
    window step can hop to the quasi-periodic alias. Sparse or low-confidence
    windows hold the previous lag (``carry``).
    """
    ref_times = np.sort(np.asarray(ref_times, dtype=float))
    pred_times = np.sort(np.asarray(pred_times, dtype=float))
    if hop_s is None:
        hop_s = window_s  # contiguous, non-overlapping windows

    # Step cap per window: under half a beat interval so it cannot reach the
    # next alias, and it also bounds how fast the lag is allowed to move.
    band_w = float(np.clip(0.4 * _median_ibi(ref_times), 0.15, lag_bound_s))

    def beats_on(lo, hi):
        """Full-SOT ref and pred beats on the extended span [lo, hi]."""
        r = ref_times[(ref_times >= lo - 0.2) & (ref_times < hi + 0.2)]
        p = pred_times[(pred_times >= lo - 0.2) & (pred_times < hi + 0.2)]
        return r, p

    starts, win_len = _window_starts(t_start, t_end, window_s, hop_s)

    # --- Per-window: baseline (symmetric, first window) then monotone drift.
    #     Reference beats come from the full SOT so the correlation is never
    #     truncated at the analysis-window edges.
    centers, lags, scores, edges = [], [], [], []
    carry = 0.0            # running lag: baseline offset, then monotone drift
    baseline_locked = False

    for w0 in starts:
        w1 = w0 + win_len
        ref_in_win = ref_times[(ref_times >= w0) & (ref_times < w1)]  # confidence
        # Grid spans a band to the left (positive baseline, prediction leads) and
        # lag_bound to the right (accumulating drift, prediction lags).
        grid_lo, grid_hi = w0 - band_w, w1 + lag_bound_s
        ref_grid, pred_ext = beats_on(grid_lo, grid_hi)

        if baseline_locked:
            # Monotone drift: from the baseline the lag may only decrease.
            lag_lo = max(-lag_bound_s, carry - band_w)
            lag_hi = carry
        else:
            # Baseline: fixed start delay, searched symmetrically about 0.
            lag_lo = max(-lag_bound_s, -band_w)
            lag_hi = min(lag_bound_s, band_w)

        res = _window_lag(ref_grid, pred_ext, grid_lo, grid_hi, lag_lo, lag_hi,
                          half_width_s=half_width_s, edge_s=edge_s)
        lag = res["lag_s"]
        if (not np.isfinite(lag)
                or res["score"] < min_window_score
                or ref_in_win.size < min_ref_in_window):
            lag = carry
        else:
            if baseline_locked:
                lag = min(lag, carry)  # enforce monotone drift
            carry = lag
            baseline_locked = True

        edges.append(w0)
        centers.append(0.5 * (w0 + w1))
        lags.append(lag)
        scores.append(res["score"])

    return {
        "edges": np.asarray(edges, dtype=float),
        "centers": np.asarray(centers, dtype=float),
        "lags": np.asarray(lags, dtype=float),
        "scores": np.asarray(scores, dtype=float),
        "window_s": float(window_s),
        "win_len": float(win_len),
    }


def _lag_at(times: np.ndarray, edges: np.ndarray, lags: np.ndarray) -> np.ndarray:
    """Piecewise-constant (staircase) per-window lag at arbitrary `times`.

    Each window holds a single lag across its whole extent; the lag steps at the
    window boundaries (left ``edges``). This is deliberately a staircase, not an
    interpolation: shifting the SOT by a per-window-constant lag is exactly what
    opens a gap in the waveform at each step — window k+1 shifts right more than
    window k, so a slice of empty time appears between them. That gap is the
    intended visualization of the samples the SOT dropped over that window.
    """
    times = np.asarray(times, dtype=float)
    edges = np.asarray(edges, dtype=float)
    lags = np.asarray(lags, dtype=float)
    if edges.size == 0:
        return np.zeros_like(times)
    idx = np.searchsorted(edges, times, side='right') - 1
    idx = np.clip(idx, 0, len(lags) - 1)
    return lags[idx]


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ChannelEvalV2:
    label: str  # "maternal" or "fetal"
    lag_s: float  # representative (median) per-window lag, for display/compat
    n_ref: int  # SOT beat count
    n_pred: int  # fiber beat count
    n_correct: int  # beats within ±50 ms after per-window correction (hard)
    n_accepted: int  # beats within acceptance window (±250 ms, inlier)
    soft_correct: float  # sum of per-beat sigmoid scores (soft "correct" count)
    precision: float  # n_correct / n_pred (hard)
    recall: float  # n_correct / n_ref (hard)
    soft_recall: float  # soft_correct / n_ref  (the v2 primary metric)
    f1: float
    dt_signed: npt.NDArray[np.float64]  # signed time errors (matched pairs)
    accepted_mask: npt.NDArray[np.bool_]
    matched_ref_t: npt.NDArray[np.float64]
    sigmoid_scores: npt.NDArray[np.float64]  # per matched pair, sigmoid of |dt|
    # per-window delay track (the v2 lag estimate)
    window_edges: npt.NDArray[np.float64]
    window_centers: npt.NDArray[np.float64]
    window_lags: npt.NDArray[np.float64]
    window_scores: npt.NDArray[np.float64]
    # stored to allow combine_evaluations_v2 / plotting over the full span
    ref_times: npt.NDArray[np.float64] | None = None
    pred_times: npt.NDArray[np.float64] | None = None
    pred_times_corrected: npt.NDArray[np.float64] | None = None
    t_start: float | None = None
    t_end: float | None = None
    window_s: float = 5.0
    window_len: float = 5.0  # actual window length (= window_s except degenerate)
    lag_bound_s: float = 5.0


@dataclass
class EvaluationResultV2:
    maternal: ChannelEvalV2
    fetal: ChannelEvalV2
    fetal_result: fHROutput


# ---------------------------------------------------------------------------
# Per-channel evaluation
# ---------------------------------------------------------------------------

def _eval_channel_v2(
        label: str,
        ref_times: np.ndarray,
        pred_times: np.ndarray,
        t_start: float,
        t_end: float,
        window_s: float = 5.0,
        hop_s: Optional[float] = None,
        lag_bound_s: float = 5.0,
        correct_thr: float = 0.05,
        soft_tau: float = 0.05,
) -> ChannelEvalV2:
    ref_full = np.asarray(ref_times, dtype=float)   # full SOT — for the lag search
    pred_times = np.asarray(pred_times, dtype=float)

    win = _estimate_window_lags(
        ref_full, pred_times, t_start, t_end,
        window_s=window_s, hop_s=hop_s, lag_bound_s=lag_bound_s,
        half_width_s=correct_thr,
    )

    # Apply the per-window staircase lag to each prediction, then match. Scoring
    # is restricted to the analysis window [t_start, t_end]: the full SOT is only
    # for the lag search, so out-of-window SOT beats must not inflate n_ref.
    ref_score = ref_full[(ref_full >= t_start) & (ref_full <= t_end)]
    pred_corrected = pred_times + _lag_at(pred_times, win["edges"], win["lags"])
    match = _robust_match(ref_score, pred_corrected)
    dt = np.asarray(match["dt_signed"], dtype=float)
    accepted = np.asarray(match["accepted_mask"], dtype=bool)
    dt_acc = dt[accepted]

    # Sigmoid scoring: soft "correct" count over accepted pairs.
    sig = _sigmoid_score(dt, thr=correct_thr, tau=soft_tau)
    soft_correct = float(np.sum(sig[accepted])) if dt.size else 0.0

    n_ref = len(ref_score)
    n_pred = len(pred_times)
    n_correct = int(np.sum(np.abs(dt_acc) <= correct_thr))
    n_accepted = int(np.sum(accepted))
    precision = n_correct / max(n_pred, 1)
    recall = n_correct / max(n_ref, 1)
    soft_recall = soft_correct / max(n_ref, 1)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    lag_s = float(np.median(win["lags"])) if win["lags"].size else 0.0

    return ChannelEvalV2(
        label=label,
        lag_s=lag_s,
        n_ref=n_ref,
        n_pred=n_pred,
        n_correct=n_correct,
        n_accepted=n_accepted,
        soft_correct=soft_correct,
        precision=precision,
        recall=recall,
        soft_recall=soft_recall,
        f1=f1,
        dt_signed=dt,
        accepted_mask=accepted,
        matched_ref_t=np.asarray(match["matched_ref_t"], dtype=float),
        sigmoid_scores=sig,
        window_edges=win["edges"],
        window_centers=win["centers"],
        window_lags=win["lags"],
        window_scores=win["scores"],
        ref_times=ref_score,
        pred_times=pred_times,
        pred_times_corrected=pred_corrected,
        t_start=t_start,
        t_end=t_end,
        window_s=window_s,
        window_len=win["win_len"],
        lag_bound_s=lag_bound_s,
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_channel_v2(
        ax_signal,
        ax_lag,
        ax_dt,
        raw_signal,
        sot_signal,
        ref_times: np.ndarray,
        pred_times: np.ndarray,
        ev: ChannelEvalV2,
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
    sot_data = sot_signal.data - float(np.mean(sot_signal.data))
    fiber_amp = max(float(np.percentile(np.abs(fiber_data), 99)), 1e-12)
    sot_amp = max(float(np.percentile(np.abs(sot_data), 99)), 1e-12)

    # Per-window staircase correction shifts the SOT later (time - lag, lag <= 0)
    # to meet the late prediction. Each window is shifted right by a constant, and
    # because later windows shift right more, a gap opens between windows — the
    # samples the SOT dropped. The connecting line left intact across each gap
    # reads as a short horizontal segment (no NaN break, per request).
    ref_lag = _lag_at(ref_times, ev.window_edges, ev.window_lags)
    sot_lag = _lag_at(sot_signal.time, ev.window_edges, ev.window_lags)

    ax_signal.plot(raw_signal.time, fiber_data, color=pred_color, lw=0.6, alpha=0.45)
    ax_signal.set_ylim(-fiber_amp * (1 + pad), -fiber_amp + 2 * fiber_amp / signal_frac)
    ax_signal.set_yticks([])

    ax2 = ax_signal.twinx()
    ax2.plot(sot_signal.time - sot_lag, sot_data, color=ref_color, lw=0.6, alpha=0.45)
    ax2.set_ylim(sot_amp - 2 * sot_amp / signal_frac, sot_amp * (1 + pad))
    ax2.set_yticks([])

    trans = blended_transform_factory(ax_signal.transData, ax_signal.transAxes)
    ax_signal.vlines(pred_times, 0, 1, transform=trans,
                     color=pred_color, lw=0.9, ls='--', alpha=0.65)
    ax_signal.vlines(ref_times - ref_lag, 0, 1, transform=trans,
                     color=ref_color, lw=0.9, ls=':', alpha=0.65)

    sig_xlim = (float(raw_signal.time[0]), float(raw_signal.time[-1]))
    ax_signal.set_xlim(*sig_xlim)

    fiber_ylim = ax_signal.get_ylim()
    sot_ylim = ax2.get_ylim()
    fiber_yfrac = (0.0 - fiber_ylim[0]) / (fiber_ylim[1] - fiber_ylim[0])
    sot_yfrac = (0.0 - sot_ylim[0]) / (sot_ylim[1] - sot_ylim[0])

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
        f"{ev.label} -- median lag={ev.lag_s:+.3f}s  "
        f"soft@50ms={ev.soft_recall:.1%}  acc@50ms={ev.recall:.1%}  F1={ev.f1:.2f}",
        fontsize=8,
    )

    # Per-window delay staircase, drawn in the SAME shifted frame as the SOT
    # waveform above (and sharing its x-limits): each window is one horizontal
    # segment exactly `window_len` long, placed at `edge - lag` .. `edge + len -
    # lag`, with a real gap to the next segment whose width is the step in lag.
    # So every stair step sits directly above the matching gap in the SOT.
    if ev.window_edges.size:
        seg_x, seg_y = [], []
        for s, lag in zip(ev.window_edges, ev.window_lags):
            seg_x += [s - lag, s + ev.window_len - lag, np.nan]
            seg_y += [lag, lag, np.nan]
        ax_lag.plot(seg_x, seg_y, color='crimson', lw=1.6,
                    label=f'per-window lag ({ev.window_len:.0f} s segments)')
        ax_lag.axhline(ev.lag_s, color='0.4', ls='--', lw=0.9,
                       label=f'median={ev.lag_s:+.3f}s')
    ax_lag.axhline(0, color='0.7', lw=0.6, ls=':')
    ax_lag.set_xlim(*sig_xlim)
    ax_lag.set_xlabel("Time (s)", fontsize=7)
    ax_lag.set_ylabel("Lag (s)", fontsize=7)
    ax_lag.legend(loc='upper right', fontsize=7)
    ax_lag.set_title(f"{ev.label} per-window delay ({ev.window_len:.0f} s windows)", fontsize=8)

    # dt scatter, coloured by sigmoid score.
    dt = ev.dt_signed
    accepted = ev.accepted_mask
    ref_t = ev.matched_ref_t
    sig = ev.sigmoid_scores
    if len(dt):
        sc = ax_dt.scatter(ref_t[accepted], dt[accepted], s=24, c=sig[accepted],
                           cmap='viridis', vmin=0.0, vmax=1.0, zorder=5,
                           label=f'accepted n={accepted.sum()}')
        if np.any(~accepted):
            ax_dt.scatter(ref_t[~accepted], dt[~accepted], s=30, marker='x',
                          color='tab:red', zorder=5, label=f'rejected n={(~accepted).sum()}')
        plt.colorbar(sc, ax=ax_dt, fraction=0.046, pad=0.04, label='sigmoid')
        ax_dt.axhline(0.0, color='k', lw=0.8)
        ax_dt.axhspan(-0.05, 0.05, color='tab:green', alpha=0.15, label='±50 ms')
        ax_dt.axhspan(-0.25, 0.25, color='tab:orange', alpha=0.08, label='±250 ms')
        med = float(np.median(dt[accepted])) if accepted.any() else 0.0
        ax_dt.axhline(med, color='0.4', ls='--', lw=0.9, label=f'median={med:.3f}s')
    ax_dt.set_xlim(t_start, t_end)
    ax_dt.set_xlabel("Ref time (s)", fontsize=7)
    ax_dt.set_ylabel("dt (s)", fontsize=7)
    ax_dt.legend(loc='upper right', fontsize=7)
    ax_dt.set_title(
        f"{ev.label} timing error  soft={ev.soft_correct:.1f}/{ev.n_ref}  "
        f"correct={ev.n_correct}/{ev.n_ref}", fontsize=8,
    )


def plot_evaluation_v2(
        ev_result: EvaluationResultV2,
        sot: SOTResult,
        out: Path,
) -> None:
    # Analysis windows come from each channel's eval (the fiber span). The SOT is
    # full, so window its signals for display (plus a lag_bound margin so the
    # shifted SOT still fills the panel) instead of plotting the whole recording.
    m = ev_result.maternal
    f = ev_result.fetal
    m_t0, m_t1 = float(m.t_start), float(m.t_end)
    f_t0, f_t1 = float(f.t_start), float(f.t_end)
    pad = float(f.lag_bound_s) + 1.0
    ppg_disp = sot.ppg.window(m_t0 - pad, m_t1 + pad)
    mic_disp = sot.mic.window(f_t0 - pad, f_t1 + pad)

    fig = plt.figure(figsize=(20, 8), constrained_layout=True)
    gs = fig.add_gridspec(2, 4)

    ax_m_sig = fig.add_subplot(gs[0, 0:2])
    ax_m_lag = fig.add_subplot(gs[0, 2])
    ax_m_dt = fig.add_subplot(gs[0, 3])

    ax_f_sig = fig.add_subplot(gs[1, 0:2])
    ax_f_lag = fig.add_subplot(gs[1, 2])
    ax_f_dt = fig.add_subplot(gs[1, 3])

    _plot_channel_v2(
        ax_m_sig, ax_m_lag, ax_m_dt,
        raw_signal=ev_result.fetal_result.maternal_source,
        sot_signal=ppg_disp,
        ref_times=m.ref_times,
        pred_times=ev_result.fetal_result.maternal_beats,
        ev=m,
        ref_color='tab:blue', pred_color='tab:orange',
        ref_label='PPG (SOT)', pred_label='Fiber chest',
        t_start=m_t0, t_end=m_t1,
    )
    _plot_channel_v2(
        ax_f_sig, ax_f_lag, ax_f_dt,
        raw_signal=ev_result.fetal_result.fetal_source,
        sot_signal=mic_disp,
        ref_times=f.ref_times,
        pred_times=ev_result.fetal_result.fetal_beats,
        ev=f,
        ref_color='tab:red', pred_color='tab:green',
        ref_label='Mic (SOT)', pred_label='Fiber fetal',
        t_start=f_t0, t_end=f_t1,
    )

    fig.suptitle(
        f"Maternal — soft@50ms={ev_result.maternal.soft_recall:.1%}  "
        f"acc@50ms={ev_result.maternal.recall:.1%}  F1={ev_result.maternal.f1:.2f}   |   "
        f"Fetal — soft@50ms={ev_result.fetal.soft_recall:.1%}  "
        f"acc@50ms={ev_result.fetal.recall:.1%}  F1={ev_result.fetal.f1:.2f}",
        fontsize=10,
    )
    plt.savefig(out / "evaluation_v2.png", dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Pipeline stage factory
# ---------------------------------------------------------------------------

def evaluate_v2(
        sot: SOTResult,
        out: Path,
        window_s: float = 5.0,
        hop_s: Optional[float] = None,
        lag_bound_s: float = 5.0,
):
    """Pipeline stage factory: sigmoid-scored, per-window-delay evaluation.

    Drop-in replacement for ``evaluate`` (evaluate.py). Differs in two ways:
    beats are scored with a sigmoid of the timing error rather than a hard
    ±50 ms count, and the SOT-vs-fiber delay is estimated as a staircase over
    fixed `window_s` (5 s) windows stepping by `hop_s` (default `window_s` ⇒
    contiguous, non-overlapping), preserving v1's one-directional lag constraint.

    ``sot`` is the **full**, un-windowed SOT (the analysis window is taken from
    the fiber result instead). Scoring is restricted to the analysis window; the
    out-of-window SOT only widens the initial-lag search.

    Takes FetalHRResult (output of fetal_hr stage), returns EvaluationResultV2.
    """

    def _span(audio):
        return float(audio.time[0]), float(audio.time[-1])

    def run_evaluate_v2(result: fHROutput) -> EvaluationResultV2:
        out.mkdir(parents=True, exist_ok=True)

        # Analysis window comes from the (already-windowed) fiber sources, per
        # channel — not from the now-full SOT.
        m_t0, m_t1 = _span(result.maternal_source)
        f_t0, f_t1 = _span(result.fetal_source)

        maternal_ev = _eval_channel_v2(
            "maternal",
            ref_times=sot.ppg_beats,
            pred_times=result.maternal_beats,
            t_start=m_t0,
            t_end=m_t1,
            window_s=window_s,
            hop_s=hop_s,
            lag_bound_s=lag_bound_s,
        )
        fetal_ev = _eval_channel_v2(
            "fetal",
            ref_times=sot.mic_beats,
            pred_times=result.fetal_beats,
            t_start=f_t0,
            t_end=f_t1,
            window_s=window_s,
            hop_s=hop_s,
            lag_bound_s=lag_bound_s,
        )

        result_v2 = EvaluationResultV2(maternal=maternal_ev, fetal=fetal_ev, fetal_result=result)
        plot_evaluation_v2(result_v2, sot, out)

        for ev in (maternal_ev, fetal_ev):
            print(
                f"  {ev.label.capitalize():8s}  median-lag={ev.lag_s:+.3f}s"
                f"  ref={ev.n_ref}  pred={ev.n_pred}"
                f"  soft={ev.soft_correct:.1f}/{ev.n_ref} ({ev.soft_recall:.1%})"
                f"  correct={ev.n_correct}/{ev.n_ref} ({ev.recall:.1%})"
                f"  F1={ev.f1:.2f}"
            )

        return result_v2

    run_evaluate_v2.__name__ = "evaluate_v2"
    return run_evaluate_v2


def combine_evaluations_v2(
        results: List[EvaluationResultV2],
) -> EvaluationResultV2:
    """Concatenate windowed-pipeline v2 evaluations into a single result."""

    def combine_channels(label: str, ev: List[ChannelEvalV2]) -> ChannelEvalV2:
        combined_ref = np.concatenate([e.ref_times for e in ev])
        combined_pred = np.concatenate([e.pred_times for e in ev])
        combined_pred_corr = np.concatenate([e.pred_times_corrected for e in ev])
        t_start = float(min(e.t_start for e in ev))  # type: ignore[arg-type]
        t_end = float(max(e.t_end for e in ev))      # type: ignore[arg-type]

        window_lags = np.concatenate([e.window_lags for e in ev])
        n_ref = sum(e.n_ref for e in ev)
        n_pred = sum(e.n_pred for e in ev)
        n_correct = sum(e.n_correct for e in ev)
        soft_correct = float(sum(e.soft_correct for e in ev))
        precision = n_correct / max(n_pred, 1)
        recall = n_correct / max(n_ref, 1)
        soft_recall = soft_correct / max(n_ref, 1)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        lag_s = float(np.median(window_lags)) if window_lags.size else 0.0

        return ChannelEvalV2(
            label=label,
            lag_s=lag_s,
            n_ref=n_ref,
            n_pred=n_pred,
            n_correct=n_correct,
            n_accepted=sum(e.n_accepted for e in ev),
            soft_correct=soft_correct,
            precision=precision,
            recall=recall,
            soft_recall=soft_recall,
            f1=f1,
            dt_signed=np.concatenate([e.dt_signed for e in ev]),
            accepted_mask=np.concatenate([e.accepted_mask for e in ev]),
            matched_ref_t=np.concatenate([e.matched_ref_t for e in ev]),
            sigmoid_scores=np.concatenate([e.sigmoid_scores for e in ev]),
            window_edges=np.concatenate([e.window_edges for e in ev]),
            window_centers=np.concatenate([e.window_centers for e in ev]),
            window_lags=window_lags,
            window_scores=np.concatenate([e.window_scores for e in ev]),
            ref_times=combined_ref,
            pred_times=combined_pred,
            pred_times_corrected=combined_pred_corr,
            t_start=t_start,
            t_end=t_end,
            window_s=float(ev[0].window_s),
            window_len=float(ev[0].window_len),
            lag_bound_s=float(ev[0].lag_bound_s),
        )

    return EvaluationResultV2(
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
