"""
detect_v8.py — positive-peak detection with a LOCAL adaptive threshold.

Why this exists
---------------
The original Shannon-energy-plus-*global*-threshold approach (a single
``scale * (max + min)`` over the whole chunk, as in ``analyze_patient11_1.py``)
silently drops peaks inside quiet sub-regions: one loud passage sets the global
threshold and every softer beat elsewhere falls under it.

``detect_adaptive_positive_peaks`` fixes this by thresholding each sample against
a *local* rolling-max envelope (several cardiac cycles wide) floored by a robust
MAD-based noise estimate. A peak is kept only when it clears its own sample's
threshold, so "missed peaks in quiet regions" and "false positives in noisy
regions" are handled symmetrically.

Detection runs on an **amplitude envelope** of the signal (RMS by default) rather
than the raw waveform, so each heart-sound burst collapses to a single lobe. That
matters because ``find_peaks(distance=…)`` deduplicates on ``min_dist`` alone; on
the raw oscillation packet that constraint is forced large enough to collapse a
burst, which then also swallows genuinely close neighbouring beats. The envelope
choice is pluggable — see ``ENVELOPES`` (``"rms"``, ``"hilbert"``, ``"none"``) or
pass any ``callable(x, fs) -> envelope`` to ``envelope=``.

Contract matches the other detectors: ``v8_beat_detector(X, bpm_range, out,
energy_range=0.5, tag="", ...)`` -> ``{"peaks": indices into X.time, "times":
beat times, ...}``, a drop-in for ``v2_beat_detector``. Use it by passing it to
``hr.fiber_beats`` / ``hr.sot_beats`` (e.g. ``fiber_beats(v8_beat_detector, out)``).
"""

from pathlib import Path
from typing import Callable, Dict, Tuple, Union

import numpy as np
from matplotlib import pyplot as plt
from scipy import signal
from scipy.ndimage import maximum_filter1d

from analyze.data import Audio


# ---------------------------------------------------------------------------
# Envelope stage — turns the raw oscillation packet of each beat into one lobe.
# Each function is ``(x, fs, window_s) -> envelope`` (same length as ``x``, >= 0).
# Add a new method by dropping another entry in ``ENVELOPES``; or pass any
# ``callable(x, fs) -> envelope`` straight to ``envelope=``.
# ---------------------------------------------------------------------------

def _rms_envelope(x: np.ndarray, fs: float, window_s: float) -> np.ndarray:
    """Sliding-window RMS: ``sqrt(mean(x**2))`` over ``window_s``. Cheap, robust,
    and phase-free; window spans a few oscillation cycles so a burst becomes a
    single smooth lobe."""
    n = max(1, int(round(fs * window_s)))
    kernel = np.ones(n, dtype=float) / n
    return np.sqrt(np.convolve(x * x, kernel, mode="same"))


def _hilbert_envelope(x: np.ndarray, fs: float, window_s: float) -> np.ndarray:
    """Analytic-signal magnitude ``|hilbert(x)|``, optionally box-smoothed over
    ``window_s`` to tame residual carrier ripple."""
    env = np.abs(signal.hilbert(x))
    n = max(1, int(round(fs * window_s)))
    if n > 1:
        env = np.convolve(env, np.ones(n, dtype=float) / n, mode="same")
    return env


def _raw_envelope(x: np.ndarray, fs: float, window_s: float) -> np.ndarray:
    """No envelope — the legacy positive-half of the raw signal. Kept so the
    pre-envelope behaviour is still reachable via ``envelope="none"``."""
    return np.maximum(x, 0.0)


ENVELOPES: Dict[str, Callable[[np.ndarray, float, float], np.ndarray]] = {
    "rms": _rms_envelope,
    "hilbert": _hilbert_envelope,
    "none": _raw_envelope,
    "raw": _raw_envelope,
}

EnvelopeSpec = Union[str, Callable[[np.ndarray, float], np.ndarray]]


def _apply_envelope(x: np.ndarray, fs: float, envelope: EnvelopeSpec,
                    window_s: float) -> np.ndarray:
    """Resolve ``envelope`` (a name in ``ENVELOPES`` or a ``callable(x, fs)``) and
    apply it, returning a non-negative detection signal the same length as ``x``."""
    if callable(envelope):
        env = np.asarray(envelope(x, fs), dtype=float)
    else:
        try:
            fn = ENVELOPES[envelope]
        except KeyError:
            raise ValueError(
                f"unknown envelope {envelope!r}; choose from {sorted(ENVELOPES)} "
                f"or pass a callable(x, fs) -> envelope"
            )
        env = np.asarray(fn(x, fs, window_s), dtype=float)
    return np.maximum(env, 0.0)


def _collapse_keep_first(peaks: np.ndarray, window: int) -> np.ndarray:
    """Collapse runs of peaks that fall within ``window`` samples into their
    FIRST (earliest) member; return the indices (into ``peaks``) that survive.

    ``peaks`` must be sorted ascending. Each survivor anchors a cluster: any later
    peak within ``window`` of the anchor is dropped, so an S1/S2 pair (two lobes
    of one heartbeat) collapses to S1 rather than to whichever lobe is louder."""
    if len(peaks) == 0 or window <= 0:
        return np.arange(len(peaks))
    keep = [0]
    for i in range(1, len(peaks)):
        if peaks[i] - peaks[keep[-1]] >= window:
            keep.append(i)
    return np.asarray(keep, dtype=int)


def _enforce_min_dist_tallest(peaks: np.ndarray, heights: np.ndarray,
                              min_dist: int) -> np.ndarray:
    """Greedy tallest-first thinning so survivors are >= ``min_dist`` apart (the
    physiological beat spacing); returns surviving indices (into ``peaks``) sorted
    by time. Same rule scipy's ``find_peaks(distance=...)`` uses, applied to an
    already-selected peak set."""
    if len(peaks) == 0 or min_dist <= 1:
        return np.arange(len(peaks))
    accepted: list[int] = []
    for i in np.argsort(heights)[::-1]:          # tallest first
        p = peaks[i]
        if all(abs(p - peaks[a]) >= min_dist for a in accepted):
            accepted.append(int(i))
    return np.asarray(sorted(accepted), dtype=int)


def detect_adaptive_positive_peaks(
    t: np.ndarray,
    x: np.ndarray,
    min_interval_s: float,
    local_window_s: float = 1.5,
    local_frac: float = 0.20,
    global_floor_k: float = 2.5,
    smooth_s: float = 0.003,
    prominence_floor_k: float = 0.3,
    envelope: EnvelopeSpec = "raw",
    envelope_window_s: float = 0.03,
    s1s2_window_s: float = 0.0,
) -> Dict[str, np.ndarray | float]:
    """envelope peak detection with a LOCAL adaptive threshold.

     this fixes in the original Shannon-energy-plus-global-threshold
    approach used by analyze_patient11_1.py:
      * Global threshold (`scale * (max+min)` over the whole chunk) silently
        drops peaks inside quiet sub-regions.

    Algorithm:
      0. Reduce `x` to a non-negative amplitude `envelope` (RMS by default over
         `envelope_window_s`) so each heart-sound burst is a single lobe instead
         of a packet of oscillations — otherwise `distance` in step 4 has to be
         large enough to collapse the packet and then also drops close beats.
      1. Smooth the envelope over `smooth_s` (sub-ms) to merge numerical jitter.
      2. Build a rolling-max envelope over `local_window_s` — the local
         reference amplitude. Several cardiac cycles wide so it captures one
         strong peak per ~cycle.
      3. Per-sample adaptive threshold =
             max( local_frac * rolling_max,
                  global_floor_k * 1.4826 * MAD(x) )
         where the MAD term prevents false positives in deeply quiet regions.
      4. `scipy.signal.find_peaks` with `distance=min_interval_s`
         and a small absolute prominence floor (`prominence_floor_k * global_floor`)
         that uses proper topographic prominence, not argmax.
      5. Keep only peaks that clear their own sample's adaptive threshold.
      6. If `s1s2_window_s > 0`, resolve S1/S2 ambiguity: any two lobes within
         `s1s2_window_s` are the two heart sounds of one beat, so keep the FIRST
         (S1) instead of the tallest — otherwise the beat time flips between S1
         and S2 depending on which sound is louder that cycle. Genuine beats are
         then re-thinned to `min_interval_s` (tallest-wins). Default 0.0 = off,
         preserving the plain tallest-per-min_interval behaviour.

    A peak at 1500 units in a region where rolling_max is 2000 passes
    (ratio 0.75); a 200-unit bump in a region where rolling_max is 300 fails
    (absolute value below the MAD floor). So "missed peaks in quiet regions"
    and "false positives in noisy regions" are handled symmetrically.
    """
    fs = 1.0 / np.median(np.diff(t))
    x_arr = np.asarray(x, float)
    xpos = _apply_envelope(x_arr, fs, envelope, envelope_window_s)

    smooth_n = max(1, int(round(fs * smooth_s)))
    if smooth_n > 1:
        kernel = np.ones(smooth_n, dtype=float) / smooth_n
        xs = np.convolve(xpos, kernel, mode="same")
    else:
        xs = xpos.copy()

    local_win = max(3, int(round(fs * local_window_s)))
    if local_win % 2 == 0:
        local_win += 1
    local_max = maximum_filter1d(xs, size=local_win, mode="reflect")
    local_max = np.maximum(local_max, 1e-12)

    # Absolute noise floor: a low multiple of the robust noise scale. Measured on
    # the RAW signal, not the envelope -- MAD(raw) ignores the (brief) beats and
    # gives ~= the noise sigma, which is also the quiet-region level of an
    # amplitude envelope (RMS/|hilbert| of noise ~= sigma). Computing it on the
    # envelope instead uses median+k*MAD, which on a dense signal (envelope
    # energy everywhere, e.g. the mic) lands *above* the peak envelope and kills
    # every detection.
    med = float(np.median(x_arr))
    mad = float(np.median(np.abs(x_arr - med))) + 1e-12
    noise_scale = 1.4826 * mad
    global_floor = global_floor_k * noise_scale

    threshold_arr = np.maximum(local_frac * local_max, global_floor)

    min_dist = max(1, int(round(fs * min_interval_s)))
    s1s2_dist = int(round(fs * s1s2_window_s)) if s1s2_window_s and s1s2_window_s > 0 else 0

    if s1s2_dist <= 0:
        # Plain behaviour: one tallest peak per min_dist window.
        peaks, props = signal.find_peaks(
            xs,
            distance=min_dist,
            prominence=(prominence_floor_k * global_floor, None),
        )
        if len(peaks) > 0:
            keep = xs[peaks] >= threshold_arr[peaks]
            peaks = peaks[keep]
            prominences = props["prominences"][keep]
        else:
            prominences = np.array([], dtype=float)
    else:
        # S1/S2 mode: surface *every* lobe (no distance collapse), threshold it,
        # then collapse each S1/S2 pair to its first (S1) lobe and finally thin
        # to the physiological beat spacing.
        peaks, props = signal.find_peaks(
            xs,
            prominence=(prominence_floor_k * global_floor, None),
        )
        prominences = props["prominences"]
        if len(peaks) > 0:
            keep = xs[peaks] >= threshold_arr[peaks]
            peaks, prominences = peaks[keep], prominences[keep]
        if len(peaks) > 0:
            first = _collapse_keep_first(peaks, s1s2_dist)          # S1/S2 -> earliest
            peaks, prominences = peaks[first], prominences[first]
            spaced = _enforce_min_dist_tallest(peaks, xs[peaks], min_dist)  # beats -> min_dist
            peaks, prominences = peaks[spaced], prominences[spaced]

    beat_times = t[peaks] if len(peaks) else np.array([], dtype=float)
    heights = xs[peaks] if len(peaks) else np.array([], dtype=float)
    ibi = np.diff(beat_times) if len(beat_times) > 1 else np.array([], dtype=float)
    bpm = 60.0 / np.clip(ibi, 1e-6, None) if len(ibi) else np.array([], dtype=float)
    return {
        "score": float(np.mean(prominences)) if len(prominences) else -1.0,
        "peaks": peaks,
        "times": beat_times,
        "ibi": ibi,
        "bpm": bpm,
        "prominences": np.asarray(prominences, dtype=float),
        "heights": np.asarray(heights, dtype=float),
        "threshold_arr": threshold_arr,
        "local_max": local_max,
        "global_floor": float(global_floor),
        # Intermediates kept so the diagnostics can reconstruct, for every
        # local maximum, *which* gate dropped it (see `_plot_debug`).
        "xs": xs,
        "min_dist": int(min_dist),
        "prominence_floor": float(prominence_floor_k * global_floor),
        "envelope": envelope if isinstance(envelope, str)
        else getattr(envelope, "__name__", "custom"),
        "envelope_window_s": float(envelope_window_s),
        "s1s2_window_s": float(s1s2_window_s),
    }


# ---------------------------------------------------------------------------
# Public: signal-level detector (drop-in for v2_beat_detector)
# ---------------------------------------------------------------------------

def v8_beat_detector(
        X: Audio,
        bpm_range: Tuple[float, float] = (90.0, 240.0),
        out=None,
        energy_range: float = 0.5,   # signature-compat (unused)
        tag: str = "",
        use_floor: bool = False,     # signature-compat (unused)
        local_window_s: float = 1.5,
        local_frac: float = 0.20,
        global_floor_k: float = 2,
        smooth_s: float = 0.003,
        prominence_floor_k: float = 0.2,
        envelope: EnvelopeSpec = "rms",
        envelope_window_s: float = 0.03,
        s1s2_window_s: float = 0.15,
) -> dict:
    """Detect beats on an amplitude envelope with a local adaptive threshold.

    ``bpm_range`` bounds the physiological rate; the fastest rate
    (``bpm_range[1]``) sets the minimum inter-beat spacing passed to
    ``detect_adaptive_positive_peaks`` as ``min_interval_s = 60 / bpm_range[1]``.
    ``envelope`` selects the amplitude-envelope stage (``"rms"`` default,
    ``"hilbert"``, ``"none"`` for the raw positive-half, or any
    ``callable(x, fs) -> envelope``); ``envelope_window_s`` is its smoothing span.
    ``s1s2_window_s`` (default 0 = off) locks each beat to its first heart sound:
    two lobes closer than this are treated as one beat's S1/S2 pair and collapsed
    to the earlier (S1), instead of keeping whichever is louder.
    Returns ``{"peaks": indices into X.time, "times": beat times, ...}`` — the
    same contract as ``v2_beat_detector``.
    """
    min_interval_s = 60.0 / float(bpm_range[1])

    result = detect_adaptive_positive_peaks(
        X.time,
        X.data,
        min_interval_s,
        local_window_s=local_window_s,
        local_frac=local_frac,
        global_floor_k=global_floor_k,
        smooth_s=smooth_s,
        prominence_floor_k=prominence_floor_k,
        envelope=envelope,
        envelope_window_s=envelope_window_s,
        s1s2_window_s=s1s2_window_s,
    )

    if out is not None:
        _plot_debug(X, result, out, tag)

    return result


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def _classify_peaks(result: dict) -> Dict[str, np.ndarray]:
    """Reconstruct, for every local maximum of the detection function, *which*
    gate kept or dropped it. Every local max lands in exactly one bucket:

      * ``kept``               — survived all gates (these are the beats). This is
                                 the detector's actual output, so the diagnostic
                                 stays correct whichever spacing rule ran
                                 (tallest-per-min_dist, or S1/S2 keep-first).
      * ``below_prominence``   — topographic prominence < the prominence floor,
                                 so ``find_peaks`` never even proposed it.
      * ``distance_suppressed``— cleared prominence and the adaptive threshold but
                                 was removed by the spacing selection (min_dist
                                 tallest-wins, or S1/S2 collapse to the earlier lobe).
      * ``below_threshold``    — a real candidate, but its height fell under its
                                 own sample's local adaptive threshold.

    Returned indices point into the detection signal ``xs`` (== indices into
    ``X.time``), so a caller can look up time/height/threshold directly.
    """
    xs = result["xs"]
    threshold_arr = result["threshold_arr"]
    prom_floor = result["prominence_floor"]
    kept = np.asarray(result["peaks"], dtype=int)

    all_pk, _ = signal.find_peaks(xs)
    prom_pk, _ = signal.find_peaks(xs, prominence=(prom_floor, None))

    # Split the prominence-passing peaks the detector did NOT keep into "below
    # the adaptive threshold" vs "removed by the spacing selection".
    non_kept = prom_pk[~np.isin(prom_pk, kept)]
    below_thr = non_kept[xs[non_kept] < threshold_arr[non_kept]] if len(non_kept) else non_kept
    suppressed = non_kept[xs[non_kept] >= threshold_arr[non_kept]] if len(non_kept) else non_kept

    return {
        "kept": kept,
        "below_threshold": below_thr,
        "distance_suppressed": suppressed,
        "below_prominence": all_pk[~np.isin(all_pk, prom_pk)],
    }


def _plot_debug(X: Audio, result: dict, out, tag: str):
    """Save a peak-detection debug chart to ``out / f"detect_v8_{tag}.png"``.

    Panel 1 overlays the detection function, the rolling-max envelope, the
    adaptive threshold and the global floor, and marks every local maximum by its
    outcome (kept / below-threshold / distance-suppressed / below-prominence).
    Panel 2 draws a red shortfall stem from each below-threshold peak up to the
    threshold it needed to reach — a direct read of *why* it was rejected. Panel 3
    puts the kept beats back on the raw waveform.
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    time = X.time
    data = X.data
    xs = result["xs"]
    threshold_arr = result["threshold_arr"]
    local_max = result["local_max"]
    global_floor = result["global_floor"]
    prom_floor = result["prominence_floor"]

    cls = _classify_peaks(result)
    kept = cls["kept"]
    below_thr = cls["below_threshold"]
    dist_supp = cls["distance_suppressed"]
    below_prom = cls["below_prominence"]

    # The prominence/distance buckets are dominated by sub-noise-floor ripples the
    # detector is *designed* to ignore; plotting all of them buries the signal. So
    # only render markers above the global (MAD) floor — a peak that clears the
    # noise floor but was still dropped is the interesting case — while the legend
    # keeps the true "shown of total" counts.
    def _visible(idx):
        return idx[xs[idx] > global_floor] if len(idx) else idx

    below_prom_v = _visible(below_prom)
    dist_supp_v = _visible(dist_supp)

    fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True, constrained_layout=True)

    env_name = result.get("envelope", "rms")
    env_win = result.get("envelope_window_s", 0.0)
    s1s2_win = result.get("s1s2_window_s", 0.0)
    # When S1/S2 collapse is on, the "suppressed" bucket is the dropped S2 partners
    # (earlier lobe kept), not distance losers -- label it so the chart reads right.
    supp_label = "S1/S2-collapsed" if s1s2_win and s1s2_win > 0 else "distance-suppressed"

    # --- Panel 1: detection function, thresholds, and per-peak outcome --------
    ax = axes[0]
    ax.plot(time, xs, lw=0.6, color="tab:purple", label=f"detection fn ({env_name} envelope)")
    ax.plot(time, local_max, lw=0.7, color="tab:orange", alpha=0.8, label="local rolling-max")
    ax.plot(time, threshold_arr, lw=0.9, color="tab:red", label="adaptive threshold")
    ax.axhline(global_floor, color="0.5", lw=0.7, ls="--", label="global (MAD) floor")

    if len(below_prom_v):
        ax.plot(time[below_prom_v], xs[below_prom_v], ".", color="0.6", ms=5,
                label=f"below prominence ({len(below_prom_v)} of {len(below_prom)} above floor)")
    if len(dist_supp_v):
        ax.plot(time[dist_supp_v], xs[dist_supp_v], "o", mfc="none", mec="tab:orange",
                ms=7, mew=1.2, label=f"{supp_label} ({len(dist_supp_v)} of {len(dist_supp)} above floor)")
    if len(below_thr):
        ax.plot(time[below_thr], xs[below_thr], "x", color="tab:red", ms=8, mew=1.6,
                label=f"below threshold ({len(below_thr)})")
    if len(kept):
        ax.plot(time[kept], xs[kept], "^", color="tab:green", ms=7,
                label=f"KEPT beats ({len(kept)})")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    s1s2_note = f", S1/S2 collapse {s1s2_win*1000:.0f} ms" if s1s2_win and s1s2_win > 0 else ""
    ax.set_title(f"{tag} — peak detection diagnostics  |  {env_name} envelope "
                 f"(win {env_win*1000:.0f} ms), adaptive threshold, prominence/distance gates{s1s2_note}")

    # --- Panel 2: why the rejected candidates fell short ----------------------
    ax = axes[1]
    ax.plot(time, xs, lw=0.5, color="tab:purple", alpha=0.5)
    ax.plot(time, threshold_arr, lw=0.9, color="tab:red", label="adaptive threshold")
    # Red shortfall stems: from the peak height up to the threshold it missed.
    for p in below_thr:
        ax.plot([time[p], time[p]], [xs[p], threshold_arr[p]], color="tab:red", lw=1.2)
    if len(below_thr):
        ax.plot(time[below_thr], xs[below_thr], "x", color="tab:red", ms=7, mew=1.5,
                label="below-threshold peak")
    # Distance-suppressed (above the noise floor): the prominence floor was
    # cleared but a taller peak within min_dist removed this one.
    if len(dist_supp_v):
        ax.plot(time[dist_supp_v], xs[dist_supp_v], "o", mfc="none", mec="tab:orange",
                ms=6, mew=1.2, label="distance-suppressed")
    ax.axhline(prom_floor, color="0.5", lw=0.7, ls=":", label="prominence floor")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("rejected candidates — red stem = shortfall to the threshold it needed")

    # --- Panel 3: kept beats on the raw waveform ------------------------------
    ax = axes[2]
    ax.plot(time, data, lw=0.5, color="tab:blue", rasterized=True)
    if len(kept) and len(data):
        ax.vlines(time[kept], float(np.min(data)), float(np.max(data)),
                  linestyles="--", color="tab:green", lw=0.9)
    if len(below_thr) and len(data):
        ax.vlines(time[below_thr], float(np.min(data)), float(np.max(data)),
                  linestyles=":", color="tab:red", lw=0.7, alpha=0.7)
    ax.set_title(f"kept beats on waveform (n={len(kept)}; "
                 f"dotted red = missed below-threshold peaks)")
    ax.set_xlabel("Time (s)")

    fig.savefig(out / f"detect_v8_{tag}.png", dpi=150)
    plt.close(fig)
