"""
detect_v3.py — robust fetal/maternal beat detection for acoustic fiber/mic signals.

Why this exists
---------------
``detect_v2._detect_beats`` picks peaks on a raw Hilbert envelope using a
per-window *additive* floor (``median + k*MAD``). When a loud transient sits in a
window it inflates the MAD, which raises the floor, so genuine low-amplitude
beats in that same window fall below threshold and are dropped. The missed beats
then corrupt the impulse-train cross-correlation lag in ``evaluate._xcorr_lag``,
so the whole ``acc@50ms`` score is mis-reported. This is the "large transient
spikes mask the small beats" failure.

This module is a SELF-CONTAINED replacement detector. ``v3_beat_detector``
follows the ``(X: Audio, bpm_range, out, tag) -> {peaks, times}`` contract, like
``v2_beat_detector``. Use it by passing it to ``hr.fiber_beats`` /
``hr.sot_beats`` (e.g. ``fiber_beats(v3_beat_detector, out)``).

Three layers, each independently testable:

  L1  Amplitude-invariant detection function.
      Optional local transient suppression, analytic envelope, then AGC
      normalisation by a slow local RMS so a small beat and a large beat reach a
      *comparable normalised height*. A single dimensionless relative threshold
      then catches both. (Mirrors the proven adaptive-local-threshold scheme that
      ``sot.detect_mic_fetal_beats`` already uses for the reference detector.)

  L2  Rhythm-aware tracking.
      Estimate the local beat period by short-time autocorrelation of the
      detection function, then a Viterbi/DP beat tracker selects the
      period-consistent subsequence of candidate peaks: it recovers on-rhythm
      low-amplitude beats, rejects off-rhythm transients, and tolerates a single
      dropped beat (~2 periods). It NEVER fabricates beats out of nothing, so the
      result stays honest against the uniform-grid control.

  L3  Detection-independent alignment.
      ``envelope_xcorr_lag()`` is a robust drop-in alternative to
      ``evaluate._xcorr_lag``: it correlates the *continuous* normalised
      envelopes of two signals instead of detected-beat impulse trains, so the
      estimated lag no longer collapses when individual beats are missed. Same
      sign convention as ``evaluate._xcorr_lag`` (``correlate(ref, pred)``), so
      its ``lag_s`` is usable identically.
"""

from typing import Optional, Tuple

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import correlate, correlation_lags, find_peaks, hilbert

from analyze.data import Audio
from constants import XCORR_TARGET_FS


# ---------------------------------------------------------------------------
# Small signal utilities (standalone; no shared deps with other modules)
# ---------------------------------------------------------------------------

def _moving_avg(x: np.ndarray, n: int) -> np.ndarray:
    if n <= 1:
        return np.asarray(x, float).copy()
    return np.convolve(x, np.ones(n, dtype=float) / n, mode="same")


def _suppress_transients(x: np.ndarray, hz: float, window_s: float = 0.04,
                         k: float = 3.0) -> np.ndarray:
    """Local-RMS transient suppression (port of ``sot._suppress_transients``).

    Loud short events get gain < 1; everything below the robust scale is left
    untouched. Run BEFORE enveloping so spikes don't dominate the AGC level.
    """
    n = max(1, int(round(window_s * hz)))
    local_rms = np.sqrt(np.maximum(_moving_avg(np.square(x), n), 0.0))
    med = float(np.median(local_rms))
    mad = float(np.median(np.abs(local_rms - med))) + 1e-9
    scale = med + k * (1.4826 * mad)
    gain = 1.0 / np.maximum(1.0, local_rms / max(scale, 1e-6))
    return x * gain


def _analytic_envelope(x: np.ndarray) -> np.ndarray:
    return np.abs(hilbert(x))


# ---------------------------------------------------------------------------
# L1 — amplitude-invariant detection function
# ---------------------------------------------------------------------------

def _detection_function(
        X: Audio,
        suppress_transients: bool = True,
        transient_k: float = 3.0,
        env_smooth_s: float = 0.012,
        env_fs: float = 500.0,
        agc_win_s: float = 1.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build an amplitude-invariant detection function on a uniform time grid.

    Returns ``(grid_times, dfun)`` where ``dfun`` is the analytic envelope
    normalised by a slow local RMS. After normalisation a beat appears as a value
    rising above a baseline of ~1 regardless of its absolute amplitude, so a
    *dimensionless* relative threshold detects small and large beats alike.

    Working on a downsampled ``env_fs`` grid (default 500 Hz, 2 ms resolution)
    keeps autocorrelation and the DP cheap while staying far finer than the
    ±50 ms scoring tolerance.
    """
    x = np.asarray(X.data, float)
    hz = float(X.hz)

    if suppress_transients:
        x = _suppress_transients(x, hz, k=transient_k)

    env = _analytic_envelope(x)
    env = _moving_avg(env, max(1, int(round(env_smooth_s * hz))))

    # Resample the envelope onto a uniform grid.
    t0, t1 = float(X.time[0]), float(X.time[-1])
    grid = np.arange(t0, t1, 1.0 / env_fs)
    if len(grid) < 4:
        return grid, np.zeros_like(grid)
    env_g = np.interp(grid, X.time, env)

    # AGC: divide by a slow local RMS (loudness), floored by a robust global
    # noise estimate so silent gaps are not amplified into spurious peaks.
    agc_n = max(3, int(round(agc_win_s * env_fs)))
    local_rms = np.sqrt(np.maximum(uniform_filter1d(env_g ** 2, size=agc_n, mode="reflect"), 0.0))
    med = float(np.median(env_g))
    mad = float(np.median(np.abs(env_g - med))) + 1e-12
    floor = med + 1.0 * 1.4826 * mad
    denom = np.maximum(local_rms, floor)
    dfun = env_g / np.maximum(denom, 1e-12)
    return grid, dfun


# ---------------------------------------------------------------------------
# L2 — rhythm estimation + Viterbi/DP beat tracking
# ---------------------------------------------------------------------------

def _period_track(
        dfun: np.ndarray,
        fs: float,
        min_period_s: float,
        max_period_s: float,
        win_s: float = 5.0,
        step_s: float = 1.0,
) -> np.ndarray:
    """Per-sample expected beat period via short-time autocorrelation of ``dfun``.

    The autocorrelation peak within the physiological lag band gives the dominant
    inter-beat interval in each window; the values are interpolated to every
    sample and median-smoothed. Falls back to the band centre when the signal is
    too short or the autocorrelation is flat.
    """
    n = len(dfun)
    centre = 0.5 * (min_period_s + max_period_s)
    if n < 4:
        return np.full(n, centre)

    lo = max(1, int(round(min_period_s * fs)))
    hi = int(round(max_period_s * fs))
    win = min(n, int(round(win_s * fs)))
    step = max(1, int(round(step_s * fs)))

    centres, periods = [], []
    for start in range(0, max(1, n - win + 1), step):
        seg = dfun[start:start + win]
        seg = seg - float(np.mean(seg))
        if float(np.std(seg)) < 1e-9:
            continue
        ac = correlate(seg, seg, mode="full", method="fft")
        ac = ac[len(seg) - 1:]  # non-negative lags
        hi_seg = min(hi, len(ac) - 1)
        if hi_seg <= lo:
            continue
        band = ac[lo:hi_seg + 1]
        if not len(band) or float(np.max(band)) <= 0:
            continue
        p = (lo + int(np.argmax(band))) / fs
        centres.append((start + win / 2.0) / fs)
        periods.append(p)

    grid_t = np.arange(n) / fs
    if not periods:
        return np.full(n, centre)
    periods = np.asarray(periods, float)
    # light median smoothing of the (sparse) period samples
    if len(periods) >= 3:
        periods = np.array([
            float(np.median(periods[max(0, i - 1):i + 2])) for i in range(len(periods))
        ])
    track = np.interp(grid_t, np.asarray(centres, float), periods)
    return np.clip(track, min_period_s, max_period_s)


def _viterbi_beats(
        times: np.ndarray,
        scores: np.ndarray,
        period_at: np.ndarray,
        min_ibi_s: float,
        lam: float = 1.0,
        miss_penalty: float = 0.4,
        gap_factor: float = 2.3,
) -> np.ndarray:
    """Select the rhythm-consistent subsequence of candidate beats by DP.

    Maximises ``sum(score) - lam * median_score * sum(penalty)`` where the
    transition penalty grows with deviation of the inter-beat interval from the
    locally expected period. A ~2x interval is allowed (one dropped beat) at an
    extra ``miss_penalty`` cost. Returns indices into ``times`` (sorted).
    """
    m = len(times)
    if m == 0:
        return np.array([], dtype=int)
    if m == 1:
        return np.array([0], dtype=int)

    med_s = float(np.median(scores)) or 1.0
    dp = scores.astype(float).copy()
    prev = np.full(m, -1, dtype=int)

    for i in range(m):
        p = float(period_at[i])
        gap_max = gap_factor * p
        for j in range(i - 1, -1, -1):
            ibi = times[i] - times[j]
            if ibi < min_ibi_s:
                continue
            if ibi > gap_max:
                break
            if ibi <= 1.5 * p:
                term = ((ibi - p) / p) ** 2
                miss = 0.0
            else:  # treat as a single dropped beat (~2 periods)
                term = ((ibi - 2.0 * p) / (2.0 * p)) ** 2
                miss = miss_penalty
            pen = lam * med_s * (term + miss)
            cand = dp[j] + scores[i] - pen
            if cand > dp[i]:
                dp[i] = cand
                prev[i] = j

    # backtrack from the best-scoring chain end
    end = int(np.argmax(dp))
    chain = []
    while end != -1:
        chain.append(end)
        end = prev[end]
    return np.array(chain[::-1], dtype=int)


# ---------------------------------------------------------------------------
# Public: signal-level detector
# ---------------------------------------------------------------------------

def v3_beat_detector(
        X: Audio,
        bpm_range: Tuple[float, float] = (100.0, 180.0),
        out=None,
        tag: str = "",
        *,
        suppress_transients: bool = True,
        transient_k: float = 3.0,
        env_fs: float = 500.0,
        agc_win_s: float = 1.5,
        cand_height: Optional[float] = None,
        cand_frac: float = 0.25,
        cand_prominence: float = 0.2,
        lam: float = 1.0,
        miss_penalty: float = 0.4,
        return_debug: bool = False,
) -> dict:
    """Detect quasi-periodic acoustic beats on a band-limited signal.

    ``X`` is expected to be already band-limited to the cardiac acoustic band
    (e.g. the 190-210 Hz NeoSSNet/ANC output, or a maternal-band chest signal).
    Returns the same dict shape as ``detect_v2._detect_beats``:
    ``{"peaks", "times", "ibi", "bpm"}``. With ``return_debug=True`` it also
    returns ``grid``, ``dfun``, ``period`` and ``cand_times`` for inspection /
    Layer-0 diagnostics (e.g. overlaying ``dfun`` against the mic SOT beats to
    confirm missed beats actually carry energy).
    """
    min_period = 60.0 / bpm_range[1]
    max_period = 60.0 / bpm_range[0]
    min_ibi = 0.85 * min_period  # allow modest beat-to-beat variability

    grid, dfun = _detection_function(
        X,
        suppress_transients=suppress_transients,
        transient_k=transient_k,
        env_fs=env_fs,
        agc_win_s=agc_win_s,
    )

    empty = {
        "peaks": np.array([], dtype=int),
        "times": np.array([], dtype=float),
    }
    if len(grid) < 4 or float(np.max(dfun)) <= 0:
        if return_debug:
            empty.update({"grid": grid, "dfun": dfun,
                          "period": np.array([]), "cand_times": np.array([])})
        return empty

    # --- candidates: permissive peaks on the normalised detection function ---
    # The beat-vs-background contrast varies enormously by signal (a clean mic
    # reaches ~5x background at beats; a marginally separated fiber only ~2x), so
    # a *fixed* normalised threshold either rejects the real (weak) beats or
    # floods with noise. Default to a threshold ADAPTIVE to this signal's own
    # distribution: median + cand_frac*(p90 - median). It self-calibrates low
    # enough to admit weak on-rhythm beats while staying above background; the
    # rhythm-aware DP below does the real signal/noise discrimination.
    if cand_height is None:
        base = float(np.median(dfun))
        hi = float(np.percentile(dfun, 90))
        cand_thr = base + cand_frac * max(hi - base, 0.0)
    else:
        cand_thr = cand_height
    # Loose spacing (~0.15 s) so a small on-rhythm beat near a transient survives
    # as a candidate; the DP enforces physiological spacing afterwards.
    cand_dist = max(1, int(round(0.15 * env_fs)))
    cand_idx, props = find_peaks(
        dfun, distance=cand_dist, height=cand_thr, prominence=cand_prominence
    )
    period = _period_track(dfun, env_fs, min_period, max_period)

    if len(cand_idx) == 0:
        if return_debug:
            empty.update({"grid": grid, "dfun": dfun,
                          "period": period, "cand_times": np.array([])})
        return empty

    cand_times = grid[cand_idx]
    cand_scores = dfun[cand_idx]
    period_at = period[cand_idx]

    # --- L2: rhythm-aware selection ---
    keep = _viterbi_beats(
        cand_times, cand_scores, period_at,
        min_ibi_s=min_ibi, lam=lam, miss_penalty=miss_penalty,
    )
    beat_times = cand_times[keep]

    # map beat times back to nearest sample indices in the original signal
    peaks = np.searchsorted(X.time, beat_times)
    peaks = np.clip(peaks, 0, len(X.time) - 1)

    result = {"peaks": peaks, "times": beat_times}
    if return_debug:
        result.update({"grid": grid, "dfun": dfun, "period": period,
                       "cand_times": cand_times})
    return result


# ---------------------------------------------------------------------------
# L3 — detection-independent alignment (robust lag estimate)
# ---------------------------------------------------------------------------

def envelope_xcorr_lag(
        ref: Audio,
        pred: Audio,
        lag_bound_s: float = 5.0,
        target_fs: float = XCORR_TARGET_FS,
        env_smooth_s: float = 0.02,
) -> dict:
    """Robust alternative to ``evaluate._xcorr_lag`` using continuous envelopes.

    Instead of cross-correlating impulse trains of *detected* beats (which
    collapses when beats are missed), this correlates the continuous normalised
    acoustic envelopes of ``ref`` and ``pred`` over their overlapping span. The
    rhythm survives even when individual beats are undetectable, so the lag is
    far more stable.

    Same sign convention as ``evaluate._xcorr_lag``: positive ``lag_s`` means
    ``pred`` events occur ``lag_s`` after ``ref``; correct ``pred`` by adding
    ``lag_s`` (i.e. ``pred_times + lag_s``) to align with ``ref`` for matching.

    Returns ``{"lag_s", "score", "lags", "corr"}``. NOTE: this is a standalone
    utility — wiring it into ``evaluate.py`` is intentionally left to the caller
    so this module stays detection-only.
    """
    t0 = max(float(ref.time[0]), float(pred.time[0]))
    t1 = min(float(ref.time[-1]), float(pred.time[-1]))
    if t1 - t0 <= 1.0 / target_fs:
        return {"lag_s": 0.0, "score": 0.0, "lags": np.array([]), "corr": np.array([])}

    grid = np.arange(t0, t1, 1.0 / target_fs)

    def _env(sig: Audio) -> np.ndarray:
        e = _analytic_envelope(np.asarray(sig.data, float))
        e = _moving_avg(e, max(1, int(round(env_smooth_s * sig.hz))))
        g = np.interp(grid, sig.time, e)
        g = g - float(np.mean(g))
        s = float(np.std(g))
        return g / s if s > 1e-12 else g

    rg = _env(ref)
    pg = _env(pred)

    # correlate(ref, pred) matches evaluate._xcorr_lag ordering / sign.
    corr = correlate(rg, pg, mode="full")
    lags = correlation_lags(len(rg), len(pg), mode="full") / target_fs
    mask = (lags >= -lag_bound_s) & (lags <= lag_bound_s)
    if not np.any(mask):
        return {"lag_s": 0.0, "score": 0.0, "lags": np.array([]), "corr": np.array([])}

    sub_corr = corr[mask]
    sub_lags = lags[mask]
    idx = int(np.argmax(sub_corr))
    lag_s = float(sub_lags[idx])
    score = float(sub_corr[idx] / (np.std(sub_corr) + 1e-9))
    return {"lag_s": lag_s, "score": score, "lags": sub_lags, "corr": sub_corr}
