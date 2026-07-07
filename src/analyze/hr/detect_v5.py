"""Beat detector v5 — verbatim port of the alternative pipeline's detector.

This reproduces ``detect_beats`` from the ``fiber_nst_model_pipeline`` project
(its ``beat_detect.py`` + the tuning in its ``config.py``) so the microphone /
model HR curves match that pipeline's output exactly.

Unlike :func:`analyze.hr.detect_v2.v2_beat_detector` (Shannon energy → moving
average → analytic envelope → distance-only peak picking → Voronoi lobe
midpoints), this detector:

* builds the envelope from the **peak-normalised, natural-log** 2nd-order
  Shannon energy (or the analytic envelope, selectable via ``method``),
* smooths it with a Gaussian (``ENVELOPE_SIGMA_S``),
* picks peaks with **both** a minimum separation (from ``bpm_range[1]``) **and**
  a height gate (``PEAK_HEIGHT_PCT`` percentile of the envelope),
* grows each peak to a half-maximum region (capped at ``HALF_MAX_LIMIT_S``),
  merges regions closer than ``MERGE_GAP_S``, and **rejects** regions whose
  duration falls outside ``[MIN_BEAT_DUR_S, MAX_BEAT_DUR_S]``,
* returns each beat as the midpoint of its region.

Signature/return shape are drop-in compatible with ``v2_beat_detector``
(``(X, bpm_range, out, energy_range=0.5, tag="")`` → ``{"peaks", "times"}``), so
it can be handed straight to ``fiber_beats`` / ``sot_beats`` in place of v2.
"""
from pathlib import Path
from typing import Tuple

import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import hilbert, find_peaks
from scipy.ndimage import gaussian_filter1d

from analyze.data import Audio

# ── Tuning — ported verbatim from the alternative pipeline's config.py ──────────
ENVELOPE_METHOD = "shannon"   # step-1 envelope: "hilbert" or "shannon" (2nd-order energy)
ENVELOPE_SIGMA_S = 0.020
HALF_MAX_LIMIT_S = 0.120
MIN_BEAT_DUR_S = 0.040
MAX_BEAT_DUR_S = 0.300
MERGE_GAP_S = 0.040
PEAK_HEIGHT_PCT = 65.0


def _shannon_energy_envelope(sig: np.ndarray) -> np.ndarray:
    """Second-order Shannon ENERGY envelope: E = -x^2 * log(x^2).

    Computed on the peak-normalised signal (Liang et al. heart-sound method). The
    x^2 factor is the "second order" (energy) weighting; the natural log
    de-emphasises the largest spikes and suppresses the low-amplitude noise floor
    relative to a plain |x|^2 envelope, giving sharper, better-separated bumps.
    """
    x = np.asarray(sig, dtype=float)
    peak = float(np.max(np.abs(x))) + 1e-12
    xn2 = (x / peak) ** 2
    xn2 = np.clip(xn2, 1e-12, None)          # floor avoids log(0) -> nan
    return -xn2 * np.log(xn2)


def v5_beat_detector(
        X: Audio,
        bpm_range: Tuple[float, float],
        out,
        energy_range: float = 0.5,
        tag: str = "",
        *,
        method: str = ENVELOPE_METHOD,
        envelope_sigma_s: float = ENVELOPE_SIGMA_S,
        half_max_limit_s: float = HALF_MAX_LIMIT_S,
        min_beat_dur_s: float = MIN_BEAT_DUR_S,
        max_beat_dur_s: float = MAX_BEAT_DUR_S,
        merge_gap_s: float = MERGE_GAP_S,
        peak_height_pct: float = PEAK_HEIGHT_PCT,
) -> dict:
    """Envelope-based beat detector (regions, not peaks).

    Drop-in replacement for ``v2_beat_detector``. ``method`` selects the step-1
    envelope: ``"shannon"`` (2nd-order Shannon energy) or ``"hilbert"`` (analytic
    signal magnitude); defaults to the alternative pipeline's ``"shannon"``.
    ``energy_range`` is the half-maximum fraction used to grow each beat region
    (0.5 = FWHM, matching the port). ``bpm_range[1]`` sets the minimum peak
    separation. Returns ``{"peaks": indices, "times": beat-centre times}``.
    """
    t = np.asarray(X.time, dtype=float)
    sig = np.asarray(X.data, dtype=float)

    if t.size < 2:
        return {"peaks": np.array([], dtype=int), "times": np.array([], dtype=float)}

    fs_s = 1.0 / float(np.median(np.diff(t)))

    # ── step 1: envelope ────────────────────────────────────────────────────────
    if method == "shannon":
        env = _shannon_energy_envelope(sig)
    else:
        env = np.abs(hilbert(sig))
    env_smooth = gaussian_filter1d(env, sigma=envelope_sigma_s * fs_s)

    # ── step 2: peak picking (min separation + height gate) ──────────────────────
    min_sep = max(1, int(fs_s * 60.0 / bpm_range[1]))
    height_th = np.percentile(env_smooth, peak_height_pct)
    pk_idx, _ = find_peaks(env_smooth, distance=min_sep, height=height_th)

    if len(pk_idx) == 0:
        if out is not None:
            _plot_stages(t, sig, [], env, env_smooth, pk_idx, height_th,
                         np.array([], dtype=float), out, tag)
        return {"peaks": np.array([], dtype=int), "times": np.array([], dtype=float)}

    # ── step 3: grow each peak to a half-maximum region ─────────────────────────
    half_samp = int(half_max_limit_s * fs_s)
    raw = []
    for pi in pk_idx:
        hv = energy_range * env_smooth[pi]
        li = pi
        while li > 0 and env_smooth[li] > hv and (pi - li) < half_samp:
            li -= 1
        ri = pi
        while ri < len(env_smooth) - 1 and env_smooth[ri] > hv and (ri - pi) < half_samp:
            ri += 1
        raw.append((float(t[li]), float(t[ri])))
    raw.sort()

    # ── step 4: merge near-touching regions, then reject by duration ────────────
    merged = []
    for s, e in raw:
        if merged and (s - merged[-1][1]) <= merge_gap_s:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append([s, e])
    beats = [(s, e) for s, e in merged if min_beat_dur_s <= (e - s) <= max_beat_dur_s]

    centers = np.array([(s + e) / 2.0 for s, e in beats], dtype=float)

    if out is not None:
        _plot_stages(t, sig, beats, env, env_smooth, pk_idx, height_th, centers, out, tag)

    if len(centers):
        peaks = np.clip(np.searchsorted(t, centers), 0, len(t) - 1)
    else:
        peaks = np.array([], dtype=int)

    return {"peaks": peaks, "times": centers}


# --- per-stage diagnostics (mirrors detect_v2._plot_stages) -----------------

def _plot_stages(time, data, beats, energy, env_smooth, pk_idx, height_th,
                 centers, out, tag):
    out = Path(out)
    fig, axes = plt.subplots(4, 1, figsize=(14, 9), sharex=True, constrained_layout=True)

    axes[0].plot(time, energy, lw=0.7, color="tab:blue")
    axes[0].set_title(f"{tag} Shannon energy (peak-normalised, ln)")

    axes[1].plot(time, env_smooth, lw=0.7, color="tab:purple")
    axes[1].axhline(height_th, color="tab:red", lw=0.8, label="height threshold")
    if len(pk_idx):
        axes[1].plot(time[pk_idx], env_smooth[pk_idx], "x", color="k", ms=5)
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].set_title("smoothed envelope + peaks + height gate")

    axes[2].plot(time, env_smooth, lw=0.7, color="tab:purple")
    for s, e in beats:
        axes[2].axvspan(s, e, color="tab:green", alpha=0.2)
    if len(centers):
        axes[2].vlines(centers, 0, float(env_smooth.max()), color="tab:green", lw=0.8)
    axes[2].set_title(f"beat regions + centres (n={len(centers)})")

    axes[3].plot(time, data, lw=0.7, color="tab:blue")
    if len(centers):
        axes[3].vlines(centers, float(data.min()), float(data.max()),
                       linestyles="--", color="tab:red", lw=0.8)
    for s, e in beats:
        axes[3].axvspan(s, e, color="tab:green", alpha=0.2)
    axes[3].set_title("beats on waveform")
    axes[3].set_xlabel("Time (s)")

    fig.savefig(out / f"detect_v5_{tag}.png", dpi=150)
    plt.close(fig)
