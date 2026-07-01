from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import maximum_filter1d
from scipy.signal import hilbert, find_peaks, fftconvolve

from analyze.data import Audio
from analyze.filters import bp_filter
from analyze.util import moving_average
from constants import MATERNAL_ACOUSTIC_BAND_HZ, MATERNAL_BPM_RANGE, FETAL_BPM_RANGE

def _local_envelope_stats(
    env: np.ndarray,
    hz: float,
    local_win_s: float = 1.5,
    floor_k: float = 2.0,
) -> Tuple[np.ndarray, float]:
    """Local max (over ~``local_win_s``) and a robust MAD floor for adaptive thresholding.

    Returns ``(local_max, floor)``. A detection threshold of
    ``max(frac * local_max, floor)`` tracks the local envelope, so a single large
    transient can no longer raise the bar across the whole window (the failure mode
    of a global ``max``-anchored threshold), while the floor keeps quiet, beat-free
    regions from admitting noise. Mirrors the pattern in
    ``sot.py:_detect_mic_fetal_beats``.
    """
    win = max(3, int(round(local_win_s * hz)))
    if win % 2 == 0:
        win += 1
    local_max = maximum_filter1d(env, size=win, mode='reflect')
    med = float(np.median(env))
    mad = float(np.median(np.abs(env - med))) + 1e-12
    floor = med + floor_k * 1.4826 * mad
    return local_max, floor


def detect_maternal_beats(
    chest_audio: Audio,
    maternal_band: Tuple[float, float] = MATERNAL_ACOUSTIC_BAND_HZ,
    bpm_range: Tuple[float, float] = MATERNAL_BPM_RANGE,
) -> Dict:
    hz = float(chest_audio.hz)
    filtered = bp_filter(chest_audio, maternal_band[0], maternal_band[1]).data

    scale = np.max(np.abs(filtered)) + 1e-12
    xn = filtered / scale
    x4 = np.clip(np.abs(xn) ** 4, 1e-12, None)
    se = -x4 * np.log(x4)

    smooth_n = max(1, int(round(0.005 * hz)))
    env = moving_average(se, smooth_n)

    local_max, floor = _local_envelope_stats(env, hz)
    active = env > np.maximum(0.25 * local_max, floor)

    changes = np.diff(active.astype(int), prepend=0, append=0)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)

    min_dist = max(1, int(round(60.0 / bpm_range[1] * 1.02 * hz)))
    pad = max(1, int(round(0.006 * hz)))
    ref_pos = np.maximum(filtered, 0.0)

    candidate_idx: List[int] = []
    candidate_strength: List[float] = []
    for s, e in zip(starts, ends):
        lo = max(0, s - pad)
        hi = min(len(ref_pos), e + pad)
        if hi <= lo:
            continue
        local = ref_pos[lo:hi]
        if local.size == 0 or float(np.max(local)) <= 0:
            continue
        idx = lo + int(np.argmax(local))
        candidate_idx.append(idx)
        candidate_strength.append(float(ref_pos[idx]))

    keep_idx: List[int] = []
    keep_str: List[float] = []
    for idx, strength in zip(candidate_idx, candidate_strength):
        if keep_idx and idx - keep_idx[-1] < min_dist:
            if strength > keep_str[-1]:
                keep_idx[-1] = idx
                keep_str[-1] = strength
        else:
            keep_idx.append(idx)
            keep_str.append(strength)

    peaks = np.asarray(keep_idx, dtype=int)
    beat_times = chest_audio.time[peaks] if len(peaks) else np.array([], dtype=float)
    ibi = np.diff(beat_times) if len(beat_times) > 1 else np.array([], dtype=float)
    bpm = 60.0 / np.clip(ibi, 1e-6, None) if len(ibi) else np.array([], dtype=float)

    return {"peaks": peaks, "times": beat_times, "ibi": ibi, "bpm": bpm, "filtered": filtered}


def _fetal_quality_score(
    beat_result: Dict,
    fetal_bpm_range: Tuple[float, float] = FETAL_BPM_RANGE,
) -> float:
    bpm = beat_result["bpm"]
    if len(bpm) < 2:
        return -np.inf
    valid = (bpm >= fetal_bpm_range[0]) & (bpm <= fetal_bpm_range[1])
    valid_frac = float(valid.mean())
    irregularity = float(np.std(bpm[valid]) / 60.0) if np.any(valid) else 1.0
    return valid_frac - 0.3 * irregularity + 0.01 * len(bpm)


def _shannon_energy_envelope(normalized: np.ndarray) -> np.ndarray:
    x2 = np.clip(normalized ** 2, 1e-12, None)
    return np.abs(hilbert(-x2 * np.log(x2)))


def _peaks_at_threshold(
    factor: float,
    env: np.ndarray,
    local_max: np.ndarray,
    floor: float,
    positive: np.ndarray,
    min_dist: int,
    pad: int,
    refine_radius: int,
    time: np.ndarray,
    filtered: np.ndarray,
) -> Dict:
    threshold = np.maximum(factor * local_max, floor)
    active = env > threshold
    changes = np.diff(active.astype(int), prepend=0, append=0)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)

    candidate_idx: List[int] = []
    candidate_strength: List[float] = []
    for s, e in zip(starts, ends):
        lo = max(0, s - pad)
        hi = min(len(positive), e + pad)
        if hi <= lo:
            continue
        local = positive[lo:hi]
        if local.size == 0 or float(np.max(local)) <= 0:
            continue
        idx = lo + int(np.argmax(local))
        candidate_idx.append(idx)
        candidate_strength.append(float(positive[idx]))

    keep_idx: List[int] = []
    keep_str: List[float] = []
    for idx, strength in zip(candidate_idx, candidate_strength):
        if keep_idx and idx - keep_idx[-1] < min_dist:
            if strength > keep_str[-1]:
                keep_idx[-1] = idx
                keep_str[-1] = strength
        else:
            keep_idx.append(idx)
            keep_str.append(strength)

    peaks = np.asarray(keep_idx, dtype=int)
    refined: List[int] = []
    for pk in peaks:
        lo = max(0, pk - refine_radius)
        hi = min(len(positive), pk + refine_radius + 1)
        refined.append(lo + int(np.argmax(positive[lo:hi])))
    peaks = np.asarray(refined, dtype=int)

    beat_times = time[peaks] if len(peaks) else np.array([], dtype=float)
    ibi = np.diff(beat_times) if len(beat_times) > 1 else np.array([], dtype=float)
    bpm = 60.0 / np.clip(ibi, 1e-6, None) if len(ibi) else np.array([], dtype=float)
    return {"peaks": peaks, "times": beat_times, "ibi": ibi, "bpm": bpm, "filtered": filtered}


def _best_shannon_result(
    env: np.ndarray,
    local_max: np.ndarray,
    floor: float,
    positive: np.ndarray,
    min_dist: int,
    pad: int,
    refine_radius: int,
    time: np.ndarray,
    filtered: np.ndarray,
) -> Tuple:
    best = None
    best_score = -np.inf
    for factor in np.linspace(0.10, 0.40, 13):
        result = _peaks_at_threshold(factor, env, local_max, floor, positive, min_dist, pad, refine_radius, time, filtered)
        score = _fetal_quality_score(result)
        if score > best_score:
            best_score = score
            best = result
    return best, best_score


def _build_template(
    normalized: np.ndarray,
    beat_times: np.ndarray,
    t0: float,
    hz: float,
    pre_s: float = 0.030,
    post_s: float = 0.050,
    min_snippets: int = 5,
) -> np.ndarray | None:
    pre = int(round(pre_s * hz))
    post = int(round(post_s * hz))
    snippets: List[np.ndarray] = []
    for bt in beat_times:
        c = int(round((bt - t0) * hz))
        if c - pre < 0 or c + post >= len(normalized):
            continue
        seg = normalized[c - pre: c + post + 1]
        s = float(np.std(seg))
        if s < 1e-12:
            continue
        snippets.append((seg - np.mean(seg)) / s)
    if len(snippets) < min_snippets:
        return None
    template = np.mean(np.vstack(snippets), axis=0)
    t_std = float(np.std(template))
    return template / t_std if t_std > 1e-12 else template


def _matched_filter_peaks(
    normalized: np.ndarray,
    template: np.ndarray,
    min_dist: int,
    hz: float,
    time: np.ndarray,
    filtered: np.ndarray,
) -> Dict | None:
    mf = np.abs(fftconvolve(normalized, template[::-1], mode='same'))
    mf = moving_average(mf, max(1, int(round(0.010 * hz))))
    mf_max = float(np.max(mf))
    if mf_max < 1e-12:
        return None
    mf /= mf_max
    peaks, _ = find_peaks(mf, distance=min_dist, prominence=0.15)
    if len(peaks) < 2:
        return None
    beat_times = time[peaks]
    ibi = np.diff(beat_times)
    bpm = 60.0 / np.clip(ibi, 1e-6, None)
    return {"peaks": peaks, "times": beat_times, "ibi": ibi, "bpm": bpm, "filtered": filtered}


def detect_fetal_beats(
    source_audio: Audio,
    min_interval_s: float = 0.27,
    threshold_factor: float = 0.40,
) -> Dict:
    hz = float(source_audio.hz)
    filtered = np.real(source_audio.data)
    scale = np.max(np.abs(filtered)) + 1e-12
    normalized = filtered / scale

    env           = _shannon_energy_envelope(normalized)
    local_max, floor = _local_envelope_stats(env, hz)
    min_dist      = max(1, int(round(min_interval_s * hz)))
    pad           = max(1, int(round(0.004 * hz)))
    refine_radius = max(1, int(round(0.015 * hz)))
    positive      = np.maximum(normalized, 0.0)

    best, best_score = _best_shannon_result(
        env, local_max, floor, positive, min_dist, pad, refine_radius, source_audio.time, filtered
    )
    if best is None:
        return _peaks_at_threshold(
            threshold_factor, env, local_max, floor, positive, min_dist, pad, refine_radius,
            source_audio.time, filtered,
        )

    template = _build_template(normalized, best["times"], float(source_audio.time[0]), hz)
    if template is None:
        return best

    mf_result = _matched_filter_peaks(normalized, template, min_dist, hz, source_audio.time, filtered)
    if mf_result is not None and _fetal_quality_score(mf_result) >= best_score - 0.05:
        return mf_result

    return best


def _plot_impulses(
    all_source_beats: Dict[str, Dict],
    maternal_result: Dict,
    classification: Dict[str, str],
    out: Path,
) -> None:
    fetal_name = next((n for n, l in classification.items() if l == "fetal_hr"), None)
    maternal_beats = maternal_result["times"]

    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True, constrained_layout=True)

    if fetal_name:
        fetal_beats = all_source_beats[fetal_name]["times"]
        axes[0].vlines(fetal_beats, 0.0, 1.0, color="tab:green", lw=1.2, alpha=0.9, label=f"Fetal ({fetal_name})")
    axes[0].vlines(maternal_beats, 0.0, 0.7, color="tab:orange", lw=1.2, alpha=0.9, label="Maternal (chest)")
    axes[0].set_yticks([])
    axes[0].set_ylabel("Impulses")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].set_title("Fetal vs. maternal beat impulse trains")

    colors = ["tab:green", "tab:orange", "tab:blue", "0.5"]
    for i, (name, beats) in enumerate(all_source_beats.items()):
        label = classification.get(name, "noise")
        offset = i * 1.2
        axes[1].vlines(beats["times"], offset, offset + 1.0,
                       color=colors[i % len(colors)], lw=1.0, alpha=0.85,
                       label=f"{name} [{label}]")
    axes[1].set_yticks([])
    axes[1].set_ylabel("All sources")
    axes[1].set_xlabel("Time (s)")
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].set_title("All separated source beat trains")

    plt.savefig(out / "fetal_hr_impulses.png", dpi=150)
    plt.close()


def v1_beat_detector(X: Audio, bpm_range, out=None, tag=""):
    """Core v1 detector on a single ``Audio`` (for fiber_beats / sot_beats).

    Follows the ``(X, bpm_range, out, tag) -> {peaks, times}`` contract. The
    maternal/chest channel uses the Shannon-energy maternal detector; any other
    channel uses the adaptive-threshold fetal detector. Wraps the shared cores
    (``detect_maternal_beats`` / ``detect_fetal_beats``) without changing them,
    so classify / ica / imf_select keep working.
    """
    if "maternal" in tag:
        res = detect_maternal_beats(X, bpm_range=bpm_range)
    else:
        res = detect_fetal_beats(X, min_interval_s=60.0 / bpm_range[1])
    return  {"peaks": res["peaks"], "times": res["times"]}
