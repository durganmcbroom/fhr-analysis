"""
detect_v6.py — period-locked S1 beat detector for fetal acoustic (SOT/mic) signals.

Why this exists
---------------
``detect_v2`` (and, in practice, ``detect_v5``) pick the tallest envelope lobe in
each neighbourhood. But every cardiac cycle produces *two* acoustic events — S1
(dominant, AV-valve closure, start of systole) and S2 (semilunar-valve closure,
start of diastole). Whenever S2 outshines S1 the marker jumps from S1 to S2, so
the beat train **flips between S1 and S2**, alternating short (systolic) and long
(diastolic) inter-beat intervals, and a tall S2 can suppress the following real
S1 (a missed beat). A single global amplitude floor compounds this by dropping
genuine low-amplitude beats whenever a loud transient inflates the floor.

This detector borrows the two robust ideas clinical monitors use:

  1. **Period lock (CTG autocorrelation).** Estimate the cardiac period T(t) by
     short-time autocorrelation of the envelope, then a Viterbi/DP tracker selects
     the candidate subsequence spaced at T(t). Enforcing one detection per *full*
     cardiac period is what stops the per-cycle S1<->S2 flipping: an S1->S2 hop
     (~0.35 T) or S2->S1 hop (~0.65 T) is heavily penalised versus staying on one
     phase (~1.0 T).

  2. **S1/S2 disambiguation (fPCG systole<diastole rule).** Period lock yields a
     clean one-per-cycle train but on *some* consistent phase — it could be S2.
     A single global vote using the fact that systole (S1->S2) is shorter than
     diastole (S2->next S1) can decide whether the locked phase is S1 or S2 and
     shift the whole train onto S1. This is implemented (``phase_correct``) but
     OFF by default: in validation the marginal vote sometimes mis-shifted onto a
     noisy secondary lobe and added artifacts, and beat *consistency* (idea 1) —
     not S1-vs-S2 identity — is what removes the HR jitter.

Plus an amplitude-invariant detection function (Shannon energy + AGC) so weak
on-rhythm beats survive alongside loud ones.

Contract matches the other detectors: ``v6_beat_detector(X, bpm_range, out,
energy_range=0.5, tag="", ...)`` -> ``{"peaks", "times"}``, so it is a drop-in for
``v2_beat_detector`` in ``fetal_detector`` / ``sot_beats`` / ``fiber_beats``.
"""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks

from analyze.data import Audio
# Reuse the proven building blocks from the earlier detectors.
from analyze.hr.detect_v3 import (
    _moving_avg,
    _period_track,
    _suppress_transients,
)
from analyze.hr.detect_v5 import _shannon_energy_envelope


# ---------------------------------------------------------------------------
# Amplitude-invariant detection function (Shannon energy + AGC on a grid)
# ---------------------------------------------------------------------------

def _detection_function(
        X: Audio,
        *,
        suppress_transients: bool = True,
        transient_k: float = 4.0,
        env_smooth_s: float = 0.015,
        env_fs: float = 500.0,
        agc_win_s: float = 1.2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build an amplitude-invariant detection function on a uniform time grid.

    The second-order Shannon energy of the (peak-normalised) signal gives a sharp
    two-lobe-per-cycle envelope; dividing by a slow local RMS (AGC) makes a small
    beat and a loud beat reach comparable normalised height so a *single*
    dimensionless threshold admits both. Working on a coarse ``env_fs`` grid keeps
    the autocorrelation and DP cheap while staying far finer than any tolerance we
    care about.

    Returns ``(grid_times, dfun, env_grid)`` where ``env_grid`` is the smoothed
    (un-normalised) Shannon envelope resampled onto the same grid — used for the
    S1/S2 phase vote and for snapping.
    """
    x = np.asarray(X.data, float)
    hz = float(X.hz)

    if suppress_transients:
        x = _suppress_transients(x, hz, k=transient_k)

    env = _shannon_energy_envelope(x)
    env = _moving_avg(env, max(1, int(round(env_smooth_s * hz))))

    t0, t1 = float(X.time[0]), float(X.time[-1])
    grid = np.arange(t0, t1, 1.0 / env_fs)
    if len(grid) < 4:
        return grid, np.zeros_like(grid), np.zeros_like(grid)
    env_g = np.interp(grid, X.time, env)

    agc_n = max(3, int(round(agc_win_s * env_fs)))
    local_rms = np.sqrt(np.maximum(uniform_filter1d(env_g ** 2, size=agc_n, mode="reflect"), 0.0))
    med = float(np.median(env_g))
    mad = float(np.median(np.abs(env_g - med))) + 1e-12
    floor = med + 1.0 * 1.4826 * mad
    denom = np.maximum(local_rms, floor)
    dfun = env_g / np.maximum(denom, 1e-12)
    return grid, dfun, env_g


# ---------------------------------------------------------------------------
# Period-synchronous beat tracking (phase-locked-loop over candidates)
# ---------------------------------------------------------------------------

def _period_lock_track(
        cand_pos: np.ndarray,
        cand_score: np.ndarray,
        period_samp: np.ndarray,
        *,
        tol: float = 0.28,
        max_skip: int = 2,
) -> np.ndarray:
    """Lock candidates to the (autocorrelation) cardiac period, one beat per cycle.

    A soft rhythm prior (v3's Viterbi) is structurally biased toward *more* beats:
    its ``((ibi-p)/p)**2`` penalty is nearly flat within +/-30% of the period, so
    adding an S2 candidate gains its envelope height for almost no cost -> doubling.
    Instead, anchor at the strongest candidate and walk outward, predicting the
    next beat one period ahead and snapping to the best-scoring candidate inside a
    ``+/-tol`` * period window. Because that window starts at ``(1-tol)`` * period
    (~0.72 T), the second heart sound (~0.4 T away) can never be chosen as the
    next beat -> one beat per cardiac cycle by construction. Up to ``max_skip``
    periods are searched so a genuinely dropped beat is bridged rather than
    triggering a phase slip.

    ``cand_pos`` are (sorted) candidate positions in grid samples; ``period_samp``
    is the per-grid-sample expected period in samples. Returns sorted indices into
    ``cand_pos``.
    """
    n = len(cand_pos)
    if n == 0:
        return np.array([], dtype=int)
    cand_pos = np.asarray(cand_pos)
    cand_score = np.asarray(cand_score)
    anchor = int(np.argmax(cand_score))
    chosen = {anchor}

    def extend(direction: int) -> None:
        cur = anchor
        while True:
            p = float(period_samp[int(cand_pos[cur])])
            nxt = None
            for k in range(1, max_skip + 1):
                target = cand_pos[cur] + direction * k * p
                lo, hi = target - tol * p, target + tol * p
                idxs = np.flatnonzero((cand_pos >= lo) & (cand_pos <= hi))
                idxs = (idxs[cand_pos[idxs] > cand_pos[cur]] if direction > 0
                        else idxs[cand_pos[idxs] < cand_pos[cur]])
                if len(idxs):
                    nxt = int(idxs[int(np.argmax(cand_score[idxs]))])
                    break
            if nxt is None or nxt in chosen:
                break
            chosen.add(nxt)
            cur = nxt

    extend(+1)
    extend(-1)
    return np.array(sorted(chosen), dtype=int)


# ---------------------------------------------------------------------------
# Global S1/S2 phase correction (systole < diastole)
# ---------------------------------------------------------------------------

def _phase_correct_to_s1(
        beat_idx: np.ndarray,
        cand_idx: np.ndarray,
        dfun: np.ndarray,
        *,
        min_pairs: int = 3,
        margin: float = 1.10,
) -> Tuple[np.ndarray, str]:
    """Decide whether the period-locked train sits on S1 or S2 and, if S2, shift
    it onto S1.

    For each consecutive locked pair ``(b_i, b_{i+1})`` the loudest *other*
    candidate strictly between them is that cycle's second heart sound. If the
    locked beats are S1 then ``gap1 = secondary - b_i`` (systole) is the SHORTER
    interval and ``gap2 = b_{i+1} - secondary`` (diastole) the longer; if the beats
    are S2 the ordering is reversed. A single global median vote (robust to a few
    missed cycles) sets the phase. ``margin`` requires diastole to beat systole by
    a clear factor before we trust a flip, so near-symmetric rhythms stay on the
    louder (Viterbi-preferred, usually S1) lobe.

    Returns ``(s1_beat_idx, decision)`` where ``decision`` is ``"kept"`` (already
    S1 / undecided) or ``"shifted"`` (relabelled S2 -> S1).
    """
    beat_idx = np.asarray(sorted(beat_idx), dtype=int)
    if len(beat_idx) < min_pairs + 1:
        return beat_idx, "kept"
    cand_idx = np.asarray(sorted(cand_idx), dtype=int)

    gap1, gap2, secondaries = [], [], []
    for a, b in zip(beat_idx[:-1], beat_idx[1:]):
        between = cand_idx[(cand_idx > a) & (cand_idx < b)]
        if len(between) == 0:
            secondaries.append(-1)
            continue
        s = int(between[int(np.argmax(dfun[between]))])
        secondaries.append(s)
        gap1.append(s - a)
        gap2.append(b - s)

    if len(gap1) < min_pairs:
        return beat_idx, "kept"

    med_systole_first = float(np.median(gap1))   # gap to the secondary
    med_diastole_first = float(np.median(gap2))   # gap from the secondary to next beat
    # Locked beats are S1 when the first gap (to the secondary) is the short one.
    if med_systole_first <= med_diastole_first * margin:
        return beat_idx, "kept"

    # Locked beats are S2 -> the secondaries are the S1s of the following cycles.
    s1 = np.asarray([s for s in secondaries if s >= 0], dtype=int)
    if len(s1) < min_pairs:
        return beat_idx, "kept"
    return np.unique(s1), "shifted"


# ---------------------------------------------------------------------------
# Snapping
# ---------------------------------------------------------------------------

def _snap_to_env(times: np.ndarray, env_time: np.ndarray, env: np.ndarray,
                 tol_s: float = 0.020) -> np.ndarray:
    """Nudge each beat time to the nearest local maximum of the full-resolution
    envelope within +/- ``tol_s`` for precise onset timing."""
    if len(times) == 0:
        return times
    out = np.empty_like(times)
    for k, tc in enumerate(times):
        lo = np.searchsorted(env_time, tc - tol_s)
        hi = np.searchsorted(env_time, tc + tol_s)
        if hi <= lo:
            out[k] = tc
        else:
            out[k] = env_time[lo + int(np.argmax(env[lo:hi]))]
    return out


# ---------------------------------------------------------------------------
# Public: signal-level detector
# ---------------------------------------------------------------------------

def v6_beat_detector(
        X: Audio,
        bpm_range: Tuple[float, float] = (90.0, 210.0),
        out=None,
        energy_range: float = 0.5,   # kept for signature-compat with v2/v5 (unused)
        tag: str = "",
        *,
        use_floor: bool = True,      # kept for signature-compat with v2 (unused)
        env_fs: float = 500.0,
        suppress_transients: bool = True,
        transient_k: float = 4.0,
        cand_frac: float = 0.20,
        cand_prominence: float = 0.15,
        cand_dist_s: float = 0.08,
        track_tol: float = 0.22,
        max_skip: int = 2,
        phase_correct: bool = False,
        phase_margin: float = 1.10,
        snap_tol_s: float = 0.020,
        return_debug: bool = False,
) -> dict:
    """Detect one S1 beat per cardiac cycle on a band-limited acoustic signal.

    ``bpm_range`` bounds the physiological FHR (min/max). Candidates are picked
    permissively on the AGC-normalised Shannon detection function (keeping *both*
    S1 and S2), then a period-synchronous tracker locks the one-per-cycle
    subsequence to the autocorrelation cardiac period. That already yields a clean
    train on a *consistent* heart sound (no within-cycle S1<->S2 flipping), which
    is what matters for HR. ``phase_correct`` additionally tries to force that
    consistent sound to be S1 specifically via a global systole<diastole vote, but
    it is OFF by default: across Patient 6/7 the marginal global vote occasionally
    mis-shifts onto a noisy secondary lobe and *added* flip artifacts, so it costs
    more than the S1-identity guarantee is worth. Returns
    ``{"peaks": indices into X.time, "times": beat times}``.

    ``track_tol`` is the fractional half-width of the next-beat search window; it
    must stay below ~0.5 so the S2 (at ~0.4 T) is never captured, and tighter
    (~0.22) keeps the lock rigid enough to avoid grabbing a wrong nearby candidate
    on easy windows. ``cand_dist_s`` must stay below the systolic interval so S1
    and S2 survive as *separate* candidates for the (optional) phase vote.
    """
    min_period = 60.0 / float(bpm_range[1])
    max_period = 60.0 / float(bpm_range[0])

    grid, dfun, env_g = _detection_function(
        X, suppress_transients=suppress_transients, transient_k=transient_k,
        env_fs=env_fs,
    )

    # Full-resolution envelope for snapping / diagnostics.
    env_full = _moving_avg(_shannon_energy_envelope(np.asarray(X.data, float)),
                           max(1, int(round(0.015 * float(X.hz)))))

    empty = {"peaks": np.array([], dtype=int), "times": np.array([], dtype=float)}

    def _finish(beat_times, cand_times=None, kept_cand=None, decision="",
                period=None):
        if out is not None:
            _plot_stages(X, grid, dfun, env_g, cand_times, kept_cand, beat_times,
                         decision, out, tag)
        if len(beat_times) == 0:
            res = dict(empty)
        else:
            peaks = np.clip(np.searchsorted(X.time, beat_times), 0, len(X.time) - 1)
            res = {"peaks": peaks, "times": np.asarray(beat_times, float)}
        if return_debug:
            res.update({"grid": grid, "dfun": dfun, "env_g": env_g,
                        "cand_times": cand_times, "period": period,
                        "decision": decision})
        return res

    if len(grid) < 4 or float(np.max(dfun)) <= 0:
        return _finish(np.array([], dtype=float))

    # --- candidates: permissive peaks on the normalised detection function ---
    base = float(np.median(dfun))
    hi = float(np.percentile(dfun, 90))
    cand_thr = base + cand_frac * max(hi - base, 0.0)
    cand_dist = max(1, int(round(cand_dist_s * env_fs)))
    cand_idx, _ = find_peaks(dfun, distance=cand_dist, height=cand_thr,
                             prominence=cand_prominence)
    period = _period_track(dfun, env_fs, min_period, max_period)

    if len(cand_idx) == 0:
        return _finish(np.array([], dtype=float), period=period)

    cand_times = grid[cand_idx]
    cand_scores = dfun[cand_idx]

    # --- period lock: one beat per cardiac cycle, consistent phase ---
    period_samp = np.maximum(period * env_fs, 1.0)
    keep = _period_lock_track(cand_idx, cand_scores, period_samp,
                              tol=track_tol, max_skip=max_skip)
    locked_grid_idx = cand_idx[keep]

    # --- global S1/S2 phase correction ---
    if phase_correct:
        s1_grid_idx, decision = _phase_correct_to_s1(
            locked_grid_idx, cand_idx, dfun, margin=phase_margin)
    else:
        s1_grid_idx, decision = locked_grid_idx, "no-phase"

    beat_times = grid[s1_grid_idx]
    beat_times = _snap_to_env(beat_times, X.time, env_full, tol_s=snap_tol_s)

    kept_cand = grid[locked_grid_idx]
    return _finish(beat_times, cand_times=cand_times, kept_cand=kept_cand,
                   decision=decision, period=period)


# ---------------------------------------------------------------------------
# Diagnostics (mirrors detect_v2/detect_v5 layout)
# ---------------------------------------------------------------------------

def _plot_stages(X, grid, dfun, env_g, cand_times, kept_cand, beat_times,
                 decision, out, tag):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(4, 1, figsize=(14, 9), sharex=True, constrained_layout=True)

    axes[0].plot(grid, env_g, lw=0.7, color="tab:purple")
    axes[0].set_title(f"{tag} Shannon energy envelope (peak-normalised)")

    axes[1].plot(grid, dfun, lw=0.7, color="tab:purple")
    if cand_times is not None and len(cand_times):
        y = np.interp(cand_times, grid, dfun)
        axes[1].plot(cand_times, y, "x", color="0.4", ms=5, label="candidates (S1+S2)")
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].set_title("AGC detection function + candidates")

    axes[2].plot(grid, dfun, lw=0.7, color="tab:purple")
    if kept_cand is not None and len(kept_cand):
        axes[2].vlines(kept_cand, 0, float(np.max(dfun)), color="tab:orange", lw=0.8,
                       label="period-locked phase")
    if len(beat_times):
        axes[2].vlines(beat_times, 0, float(np.max(dfun)), color="tab:green", lw=1.0,
                       label="S1 beats")
    axes[2].legend(loc="upper right", fontsize=8)
    axes[2].set_title(f"period lock + S1/S2 phase vote (decision: {decision})")

    axes[3].plot(X.time, X.data, lw=0.5, color="tab:blue", rasterized=True)
    if len(beat_times):
        axes[3].vlines(beat_times, float(np.min(X.data)), float(np.max(X.data)),
                       linestyles="--", color="tab:red", lw=0.8)
    axes[3].set_title(f"S1 beats on waveform (n={len(beat_times)})")
    axes[3].set_xlabel("Time (s)")

    fig.savefig(out / f"detect_v6_{tag}.png", dpi=150)
    plt.close(fig)
