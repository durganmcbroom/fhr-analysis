#!/usr/bin/env python3
# patient 8, 12_2 verified
import argparse
import itertools
import json
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import signal
from scipy.io import wavfile
from scipy.ndimage import maximum_filter1d
from sklearn.decomposition import FastICA
from sklearn.exceptions import ConvergenceWarning

from constants import (
    MATERNAL_BPM_RANGE, FETAL_BPM_RANGE, MATERNAL_ACOUSTIC_BAND_HZ,
    SOURCE_PREP_BAND_HZ, FETAL_ACOUSTIC_BAND_HZ,
    FIBER_BUNDLE_A, FIBER_BUNDLE_B, PVS_FILE, MIC_FILE,
)


os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


WINDOWS = [(250, 270)] #[(200.0, 250.0), (280.0, 330.0), (415.0, 490.0), (600.0, 645.0)]
CHUNK_SECONDS = 20.0
# DATASET_DIR = Path("./../Banner_data/patient8-session1")
# OUTPUT_DIR = Path("../out/claude_output_patient8")
DATASET_DIR = Path("../../Banner_data/Banner_test_20251220/PT13_1")
OUTPUT_DIR = Path("../out/claude_output_patient13_1")


@dataclass
class Config:
    dataset_dir: Path = DATASET_DIR
    output_dir: Path = OUTPUT_DIR
    windows: Sequence[Tuple[float, float]] = ((200.0, 500.0),)
    display_window: Tuple[float, float] = (250.0, 270.0)
    waveform_plot_windows: Sequence[Tuple[float, float]] = ((200.0, 220.0), (220.0, 240.0), (240.0, 260.0))
    ppg_lag_window: Tuple[float, float] = (255.0, 260.0)
    chunk_seconds: float = CHUNK_SECONDS
    ppg_lag_bounds: Tuple[float, float] = (1.5, 3.5)
    mic_lag_bounds: Tuple[float, float] = (0.0, 10.0)
    maternal_hr_bpm: Tuple[float, float] = MATERNAL_BPM_RANGE
    fetal_hr_bpm: Tuple[float, float] = FETAL_BPM_RANGE
    maternal_acoustic_band_hz: Tuple[float, float] = MATERNAL_ACOUSTIC_BAND_HZ
    fetal_acoustic_band_hz: Tuple[float, float] = (25.0, 120.0)
    residual_fetal_band_hz: Tuple[float, float] = (35.0, 140.0)
    source_prep_band_hz: Tuple[float, float] = SOURCE_PREP_BAND_HZ
    fetal_detection_band_hz: Tuple[float, float] = FETAL_ACOUSTIC_BAND_HZ
    selected_pair_idx: Tuple[int, int] = (0, 1)
    ica_fun: str = "exp"
    ica_tol: float = 1e-3
    ica_max_iter: int = 3000
    ica_seconds: float | None = None  # if set, ICA runs on this window; stats use xcorr_window_s
    xcorr_window_s: float = 5.0       # xcorr chunk size; controls stats granularity and lag range (±window/2)


def resample_uniform(t: np.ndarray, x: np.ndarray, fs_target: float) -> Tuple[np.ndarray, np.ndarray]:
    t0, t1 = float(t[0]), float(t[-1])
    n = int(np.floor((t1 - t0) * fs_target)) + 1
    tu = t0 + np.arange(n) / fs_target
    xu = np.interp(tu, t, x)
    return tu, xu


def butter_filter(
    x: np.ndarray,
    fs: float,
    low: float | None = None,
    high: float | None = None,
    order: int = 4,
) -> np.ndarray:
    nyq = fs / 2.0
    if low is None and high is None:
        return x.copy()
    if low is not None and low <= 0:
        low = None
    if high is not None and high >= nyq:
        high = None
    if low is None and high is None:
        return x.copy()
    if low is None:
        sos = signal.butter(order, high / nyq, btype="lowpass", output="sos")
    elif high is None:
        sos = signal.butter(order, low / nyq, btype="highpass", output="sos")
    else:
        sos = signal.butter(order, [low / nyq, high / nyq], btype="bandpass", output="sos")
    return signal.sosfiltfilt(sos, x)


def cheby1_bandpass_filter(x: np.ndarray, fs: float, band: Tuple[float, float], order: int = 6, rp: float = 1.0) -> np.ndarray:
    nyq = fs / 2.0
    low = band[0] / nyq
    high = band[1] / nyq
    sos = signal.cheby1(order, rp, [low, high], btype="bandpass", output="sos")
    return signal.sosfiltfilt(sos, x)


def robust_clip(x: np.ndarray, zmax: float = 6.0) -> np.ndarray:
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + 1e-9
    sigma = 1.4826 * mad
    return np.clip(x, med - zmax * sigma, med + zmax * sigma)


def normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    s = np.std(x)
    if s < 1e-12:
        return np.zeros_like(x)
    return (x - np.mean(x)) / s


def moving_average(x: np.ndarray, fs: float, seconds: float) -> np.ndarray:
    n = max(1, int(round(seconds * fs)))
    if n <= 1:
        return x.copy()
    kernel = np.ones(n, dtype=float) / n
    return np.convolve(x, kernel, mode="same")

def running_rms(x: np.ndarray, fs: float, seconds: float) -> np.ndarray:
    return np.sqrt(np.maximum(moving_average(np.square(x), fs, seconds), 0.0))

def envelope_from_band(x: np.ndarray, fs: float, band: Tuple[float, float], smooth_hz: float) -> np.ndarray:
    xb = butter_filter(x, fs, low=band[0], high=band[1], order=3)
    env = np.abs(signal.hilbert(xb))
    env = butter_filter(env, fs, high=smooth_hz, order=3)
    return np.maximum(env, 0.0)


def suppress_transients(x: np.ndarray, fs: float, window_s: float = 0.12) -> np.ndarray:
    local = running_rms(x, fs, window_s)
    scale = np.median(local) + 3.0 * (1.4826 * np.median(np.abs(local - np.median(local))) + 1e-9)
    gain = 1.0 / np.maximum(1.0, local / max(scale, 1e-6))
    return x * gain


def signal_quality(x: np.ndarray, fs: float, bpm_range: Tuple[float, float]) -> Dict[str, float]:
    x = normalize(x)
    freqs, psd = signal.welch(x, fs=fs, nperseg=min(len(x), max(256, int(fs * 16))))
    lo, hi = np.array(bpm_range) / 60.0
    in_band = (freqs >= lo) & (freqs <= hi)
    out_band = (freqs >= 0.1) & (freqs <= min(8.0, fs / 2.0))
    band_power = float(np.trapezoid(psd[in_band], freqs[in_band])) if np.any(in_band) else 0.0
    total_power = float(np.trapezoid(psd[out_band], freqs[out_band])) if np.any(out_band) else 0.0
    peak_freq = float(freqs[in_band][np.argmax(psd[in_band])]) if np.any(in_band) else np.nan
    snr = band_power / (max(total_power - band_power, 1e-9))
    return {
        "band_power": band_power,
        "total_power": total_power,
        "snr": snr,
        "peak_bpm": peak_freq * 60.0,
    }


def detect_beats(
    t: np.ndarray,
    x: np.ndarray,
    fs: float,
    bpm_range: Tuple[float, float],
    prefer_positive: bool | None = None,
    prominence_scale: float = 0.35,
) -> Dict[str, np.ndarray | float]:
    min_dist = max(1, int(fs * 60.0 / bpm_range[1] * 0.75))
    max_dist = max(min_dist + 1, int(fs * 60.0 / bpm_range[0] * 1.25))
    work = normalize(x)
    prom = max(0.15, prominence_scale * np.std(work))
    candidates: List[Tuple[float, np.ndarray, np.ndarray]] = []
    polarities = [1] if prefer_positive is True else [-1] if prefer_positive is False else [1, -1]
    for sign in polarities:
        peaks, props = signal.find_peaks(sign * work, distance=min_dist, prominence=prom)
        if len(peaks) < 2:
            candidates.append((-np.inf, peaks, np.zeros(0)))
            continue
        ibi = np.diff(t[peaks])
        bpm = 60.0 / np.clip(ibi, 1e-6, None)
        valid = (bpm >= bpm_range[0] * 0.8) & (bpm <= bpm_range[1] * 1.2)
        score = valid.mean() - 0.5 * np.std(ibi)
        if len(peaks) > 2:
            dense_penalty = max(0.0, len(peaks) / (len(t) / fs / 0.25) - 1.0)
            score -= 0.2 * dense_penalty
        candidates.append((score, peaks, props.get("prominences", np.zeros(len(peaks)))))
    score, peaks, prominences = max(candidates, key=lambda item: item[0])
    beat_times = t[peaks]
    ibi = np.diff(beat_times)
    bpm = 60.0 / np.clip(ibi, 1e-6, None)
    return {
        "score": float(score if np.isfinite(score) else -1.0),
        "peaks": peaks,
        "times": beat_times,
        "ibi": ibi,
        "bpm": bpm,
        "prominences": prominences,
    }


def detect_maternal_fiber_beats(
    t: np.ndarray,
    x: np.ndarray,
    fs: float,
    bpm_range: Tuple[float, float],
    target_count: int | None = None,
) -> Dict[str, np.ndarray | float]:
    return shannon_energy_peak_detection(
        t,
        detection_signal=x,
        refinement_signal=x,
        min_distance_s=60.0 / bpm_range[1] * 1.02,
        threshold_scale=None,
        smooth_s=None,
        bpm_range=bpm_range,
        positive_only=True,
        threshold_grid=np.linspace(0.10, 0.40, 13),
        smooth_grid=[0.003, 0.005, 0.007],
        target_count=target_count,
        count_range=(max(1, (target_count or 0) - 1), (target_count or 99) + 1) if target_count is not None else None,
    )


def detect_fetal_beats(
    t: np.ndarray,
    feature: np.ndarray,
    waveform: np.ndarray,
    fs: float,
    bpm_range: Tuple[float, float],
) -> Dict[str, np.ndarray | float]:
    feature = normalize(feature)
    work = np.maximum(feature, 0.0)
    min_dist = max(1, int(fs * 0.32))
    peaks, props = signal.find_peaks(
        work,
        distance=min_dist,
        prominence=max(0.22, 0.60 * np.std(work)),
        height=np.percentile(work, 82),
    )
    if len(peaks) == 0:
        return {
            "score": -1.0,
            "peaks": np.array([], dtype=int),
            "times": np.array([], dtype=float),
            "ibi": np.array([], dtype=float),
            "bpm": np.array([], dtype=float),
            "prominences": np.array([], dtype=float),
        }

    refine_radius = max(1, int(0.025 * fs))
    wpos = np.maximum(normalize(waveform), 0.0)
    refined = []
    refined_prom = []
    for pk, prom in zip(peaks, props.get("prominences", np.zeros(len(peaks)))):
        lo = max(0, pk - refine_radius)
        hi = min(len(wpos), pk + refine_radius + 1)
        local = lo + int(np.argmax(wpos[lo:hi]))
        refined.append(local)
        refined_prom.append(prom)
    refined = np.asarray(refined, dtype=int)
    order = np.argsort(refined)
    refined = refined[order]
    refined_prom = np.asarray(refined_prom)[order]

    keep_idx: List[int] = []
    keep_prom: List[float] = []
    for idx, prom in zip(refined, refined_prom):
        if keep_idx and idx - keep_idx[-1] < min_dist:
            if prom > keep_prom[-1]:
                keep_idx[-1] = int(idx)
                keep_prom[-1] = float(prom)
        else:
            keep_idx.append(int(idx))
            keep_prom.append(float(prom))
    peaks = np.asarray(keep_idx, dtype=int)
    refined_prom = np.asarray(keep_prom, dtype=float)
    beat_times = t[peaks]
    ibi = np.diff(beat_times)
    bpm = 60.0 / np.clip(ibi, 1e-6, None)
    valid = (bpm >= bpm_range[0] * 0.9) & (bpm <= bpm_range[1] * 1.05)
    score = float(valid.mean() - 0.25 * np.std(ibi)) if len(ibi) else -1.0
    return {
        "score": score,
        "peaks": peaks,
        "times": beat_times,
        "ibi": ibi,
        "bpm": bpm,
        "prominences": refined_prom[: len(peaks)],
    }


def manual_shannon_peak_detection(
    t: np.ndarray,
    waveform: np.ndarray,
    band_hz: Tuple[float, float],
    min_interval_s: float,
    threshold_factor: float = 0.40,
    order: int = 3,
    rp: float = 1.0,
) -> Dict[str, np.ndarray | float]:
    fs = 1.0 / np.median(np.diff(t))
    filtered = cheby1_bandpass_filter(np.asarray(waveform, float), fs, band_hz, order=order, rp=rp)
    scale = np.max(np.abs(filtered)) + 1e-12
    normalized = filtered / scale
    shannon = -(normalized**2) * np.log(np.clip(normalized**2, 1e-12, None))
    envelope = np.abs(signal.hilbert(shannon))
    threshold = threshold_factor * (float(np.max(envelope)) + float(np.min(envelope)))
    active = envelope > threshold

    changes = np.diff(active.astype(int), prepend=0, append=0)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    min_dist = max(1, int(round(min_interval_s * fs)))
    pad = max(1, int(round(0.004 * fs)))
    candidate_idx: List[int] = []
    candidate_strength: List[float] = []
    positive = np.maximum(normalized, 0.0)

    for start, end in zip(starts, ends):
        lo = max(0, start - pad)
        hi = min(len(positive), end + pad)
        if hi <= lo:
            continue
        local = positive[lo:hi]
        if local.size == 0 or float(np.max(local)) <= 0:
            continue
        idx = lo + int(np.argmax(local))
        candidate_idx.append(idx)
        candidate_strength.append(float(positive[idx]))

    keep_idx: List[int] = []
    keep_strength: List[float] = []
    for idx, strength in zip(candidate_idx, candidate_strength):
        if keep_idx and idx - keep_idx[-1] < min_dist:
            if strength > keep_strength[-1]:
                keep_idx[-1] = idx
                keep_strength[-1] = strength
        else:
            keep_idx.append(idx)
            keep_strength.append(strength)

    peaks = np.asarray(keep_idx, dtype=int)
    beat_times = t[peaks] if len(peaks) else np.array([], dtype=float)
    ibi = np.diff(beat_times) if len(beat_times) > 1 else np.array([], dtype=float)
    bpm = 60.0 / np.clip(ibi, 1e-6, None) if len(ibi) else np.array([], dtype=float)
    score = float(np.mean(keep_strength)) if keep_strength else -1.0
    return {
        "score": score,
        "peaks": peaks,
        "times": beat_times,
        "ibi": ibi,
        "bpm": bpm,
        "prominences": np.asarray(keep_strength, dtype=float),
        "filtered": filtered,
        "normalized": normalized,
        "shannon_energy": shannon,
        "envelope": envelope,
        "threshold": threshold,
    }


def detect_fetal_waveform_peaks(
    t: np.ndarray,
    waveform: np.ndarray,
    bpm_range: Tuple[float, float],
    target_count: int | None = None,
) -> Dict[str, np.ndarray | float]:
    _ = bpm_range, target_count
    return manual_shannon_peak_detection(
        t,
        waveform,
        band_hz=(190.0, 220.0),
        min_interval_s=0.27,
        threshold_factor=0.40,
        order=3,
        rp=1.0,
    )


def detect_adaptive_positive_peaks(
    t: np.ndarray,
    x: np.ndarray,
    min_interval_s: float,
    local_window_s: float = 1.5,
    local_frac: float = 0.20,
    global_floor_k: float = 2.5,
    smooth_s: float = 0.003,
    prominence_floor_k: float = 0.3,
) -> Dict[str, np.ndarray | float]:
    fs = 1.0 / np.median(np.diff(t))
    x_arr = np.asarray(x, float)
    xpos = np.maximum(x_arr, 0.0)

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

    med = float(np.median(x_arr))
    mad = float(np.median(np.abs(x_arr - med))) + 1e-12
    noise_scale = 1.4826 * mad
    global_floor = global_floor_k * noise_scale
    threshold_arr = np.maximum(local_frac * local_max, global_floor)

    min_dist = max(1, int(round(fs * min_interval_s)))
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
    }


def detect_microphone_s1_peaks(
    t: np.ndarray,
    waveform: np.ndarray,
    target_count: int | None = None,
) -> Dict[str, np.ndarray | float]:
    _ = target_count
    return detect_adaptive_positive_peaks(
        t,
        waveform,
        min_interval_s=0.30,
        local_window_s=1.5,
        local_frac=0.20,
        global_floor_k=2.5,
        prominence_floor_k=0.3,
    )


def shannon_energy_trace(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    scale = np.max(np.abs(x)) + 1e-12
    xn = x / scale
    x4 = np.clip(xn**4, 1e-12, None)
    return -x4 * np.log(x4)


def shannon_energy_peak_detection(
    t: np.ndarray,
    detection_signal: np.ndarray,
    refinement_signal: np.ndarray,
    min_distance_s: float,
    threshold_scale: float | None,
    smooth_s: float | None,
    bpm_range: Tuple[float, float],
    positive_only: bool = True,
    threshold_grid: Sequence[float] | None = None,
    smooth_grid: Sequence[float] | None = None,
    target_count: int | None = None,
    count_range: Tuple[int, int] | None = None,
) -> Dict[str, np.ndarray | float]:
    fs = 1.0 / np.median(np.diff(t))
    det = np.asarray(detection_signal, float)
    ref = np.asarray(refinement_signal, float)
    se = shannon_energy_trace(det)
    min_dist = max(1, int(round(min_distance_s * fs)))
    search_ref = np.maximum(ref, 0.0) if positive_only else np.abs(ref)
    threshold_candidates = list(threshold_grid) if threshold_grid is not None else [float(threshold_scale)]
    smooth_candidates = list(smooth_grid) if smooth_grid is not None else [float(smooth_s)]

    def empty_result(env: np.ndarray, threshold: float) -> Dict[str, np.ndarray | float]:
        return {
            "score": -1.0,
            "peaks": np.array([], dtype=int),
            "times": np.array([], dtype=float),
            "ibi": np.array([], dtype=float),
            "bpm": np.array([], dtype=float),
            "prominences": np.array([], dtype=float),
            "shannon_energy": se,
            "envelope": env,
            "threshold": threshold,
        }

    def run_once(local_smooth_s: float, local_threshold_scale: float) -> Dict[str, np.ndarray | float]:
        env = moving_average(se, fs, local_smooth_s)
        threshold = local_threshold_scale * (float(np.max(env)) + float(np.min(env)))
        active = env > threshold
        if not np.any(active):
            return empty_result(env, threshold)

        changes = np.diff(active.astype(int), prepend=0, append=0)
        starts = np.flatnonzero(changes == 1)
        ends = np.flatnonzero(changes == -1)
        pad = max(1, int(round(0.006 * fs)))
        candidate_idx: List[int] = []
        candidate_strength: List[float] = []
        for start, end in zip(starts, ends):
            lo = max(0, start - pad)
            hi = min(len(ref), end + pad)
            if hi <= lo:
                continue
            local = search_ref[lo:hi]
            if local.size == 0 or np.max(local) <= 0:
                continue
            idx = lo + int(np.argmax(local))
            candidate_idx.append(idx)
            candidate_strength.append(float(search_ref[idx]))

        if not candidate_idx:
            return empty_result(env, threshold)

        order = np.argsort(candidate_idx)
        candidate_idx = [int(np.asarray(candidate_idx)[order][i]) for i in range(len(order))]
        candidate_strength = [float(np.asarray(candidate_strength)[order][i]) for i in range(len(order))]

        keep_idx: List[int] = []
        keep_strength: List[float] = []
        for idx, strength in zip(candidate_idx, candidate_strength):
            if keep_idx and idx - keep_idx[-1] < min_dist:
                if strength > keep_strength[-1]:
                    keep_idx[-1] = idx
                    keep_strength[-1] = strength
            else:
                keep_idx.append(idx)
                keep_strength.append(strength)

        peaks = np.asarray(keep_idx, dtype=int)
        beat_times = t[peaks]
        ibi = np.diff(beat_times)
        bpm = 60.0 / np.clip(ibi, 1e-6, None)
        valid = (bpm >= bpm_range[0] * 0.9) & (bpm <= bpm_range[1] * 1.1)
        score = float(valid.mean() - 0.2 * np.std(ibi)) if len(ibi) else -1.0
        return {
            "score": score,
            "peaks": peaks,
            "times": beat_times,
            "ibi": ibi,
            "bpm": bpm,
            "prominences": np.asarray(keep_strength, dtype=float),
            "shannon_energy": se,
            "envelope": env,
            "threshold": threshold,
            "threshold_scale": float(local_threshold_scale),
            "smooth_s": float(local_smooth_s),
        }

    best: Dict[str, np.ndarray | float] | None = None
    best_obj = -np.inf
    duration_s = float(t[-1] - t[0]) if len(t) > 1 else 0.0
    default_target = int(round(duration_s * np.mean(bpm_range) / 60.0)) if duration_s > 0 else 0
    expected_target = target_count if target_count is not None else default_target
    for local_smooth_s in smooth_candidates:
        for local_threshold_scale in threshold_candidates:
            out = run_once(local_smooth_s, local_threshold_scale)
            count = len(out["times"])
            obj = float(out["score"])
            if count_range is not None:
                lo, hi = count_range
                if lo <= count <= hi:
                    obj += 2.0
                else:
                    obj -= 0.8 * min(abs(count - lo), abs(count - hi))
            if expected_target > 0:
                obj -= 0.45 * abs(count - expected_target)
            obj += 0.35 * float(local_threshold_scale)
            obj -= 0.5 * float(local_smooth_s)
            if best is None or obj > best_obj:
                best = out
                best_obj = obj

    assert best is not None
    return best


def beat_train(times: np.ndarray, fs: float, t_start: float, t_end: float, sigma_s: float) -> Tuple[np.ndarray, np.ndarray]:
    n = int(np.ceil((t_end - t_start) * fs)) + 1
    t = t_start + np.arange(n) / fs
    x = np.zeros(n, dtype=float)
    idx = np.round((times - t_start) * fs).astype(int)
    idx = idx[(idx >= 0) & (idx < n)]
    x[idx] = 1.0
    sigma = max(1.0, sigma_s * fs)
    radius = int(6 * sigma)
    grid = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (grid / sigma) ** 2)
    kernel /= kernel.sum()
    x = np.convolve(x, kernel, mode="same")
    return t, x


def impulse_train_from_times(times: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
    impulse = np.zeros_like(t_grid, dtype=float)
    if len(times) == 0:
        return impulse
    idx = np.round((times - t_grid[0]) / (t_grid[1] - t_grid[0])).astype(int)
    idx = idx[(idx >= 0) & (idx < len(t_grid))]
    impulse[idx] = 1.0
    return impulse


def gaussian_smooth_signal(x: np.ndarray, sigma_samples: float) -> np.ndarray:
    sigma_samples = max(1.0, float(sigma_samples))
    radius = int(3 * sigma_samples)
    grid = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (grid / sigma_samples) ** 2)
    kernel /= kernel.sum()
    return np.convolve(x, kernel, mode="same")


def estimate_lag_from_beats(
    ref_times: np.ndarray,
    mov_times: np.ndarray,
    bounds: Tuple[float, float],
    fs: float = 20.0,
    sigma_s: float = 0.12,
) -> Dict[str, float]:
    if len(ref_times) < 3 or len(mov_times) < 3:
        return {"lag_s": np.nan, "score": 0.0}
    t_start = min(ref_times.min(), mov_times.min()) - bounds[1] - 1.0
    t_end = max(ref_times.max(), mov_times.max()) + bounds[1] + 1.0
    _, ref = beat_train(ref_times, fs, t_start, t_end, sigma_s)
    _, mov = beat_train(mov_times, fs, t_start, t_end, sigma_s)
    corr = signal.correlate(ref - ref.mean(), mov - mov.mean(), mode="full")
    lags = signal.correlation_lags(len(ref), len(mov), mode="full") / fs
    mask = (lags >= bounds[0]) & (lags <= bounds[1])
    if not np.any(mask):
        return {"lag_s": np.nan, "score": 0.0}
    sub_corr = corr[mask]
    sub_lags = lags[mask]
    idx = int(np.argmax(sub_corr))
    lag = float(sub_lags[idx])
    score = float(sub_corr[idx] / (np.std(sub_corr) + 1e-9))
    return {"lag_s": lag, "score": score}


def estimate_peak_lag_grid(
    ref_times: np.ndarray,
    mov_times: np.ndarray,
    bounds: Tuple[float, float],
    step_s: float = 0.01,
    tol_s: float = 0.18,
) -> Dict[str, float]:
    if len(ref_times) < 3 or len(mov_times) < 3:
        return {"lag_s": np.nan, "score": 0.0}
    best = {"lag_s": np.nan, "score": -np.inf}
    for lag in np.arange(bounds[0], bounds[1] + 0.5 * step_s, step_s):
        shifted = mov_times + lag
        matched = 0
        errors = []
        for tm in shifted:
            idx = np.argmin(np.abs(ref_times - tm))
            err = abs(ref_times[idx] - tm)
            if err <= tol_s:
                matched += 1
                errors.append(err)
        if matched == 0:
            score = -1.0
        else:
            score = matched / max(len(shifted), 1) - 0.5 * float(np.mean(errors))
        if score > best["score"]:
            best = {"lag_s": float(lag), "score": float(score)}
    return best


def select_times_in_windows(times: np.ndarray, windows: Sequence[Tuple[float, float]]) -> np.ndarray:
    out = []
    for start, end in windows:
        mask = (times >= start) & (times < end)
        if np.any(mask):
            out.append(times[mask])
    return np.concatenate(out) if out else np.array([], dtype=float)


def estimate_lag_from_impulse_trains(
    ref_times: np.ndarray,
    mov_times: np.ndarray,
    windows: Sequence[Tuple[float, float]],
    bounds: Tuple[float, float],
    target_fs: float = 200.0,
    smooth_sigma_s: float = 0.10,
) -> Dict[str, float]:
    ref_sel = select_times_in_windows(ref_times, windows)
    mov_sel = select_times_in_windows(mov_times, windows)
    if len(ref_sel) < 3 or len(mov_sel) < 3:
        return {"lag_s": np.nan, "score": 0.0}

    t_start = min(ref_sel.min(), mov_sel.min())
    t_end = max(ref_sel.max(), mov_sel.max()) + 1.0 / target_fs
    t_grid = np.arange(t_start, t_end, 1.0 / target_fs)
    ref_impulse = impulse_train_from_times(ref_sel, t_grid)
    mov_impulse = impulse_train_from_times(mov_sel, t_grid)

    ref_smooth = gaussian_smooth_signal(ref_impulse, smooth_sigma_s * target_fs)
    mov_smooth = gaussian_smooth_signal(mov_impulse, smooth_sigma_s * target_fs)

    cross_corr = signal.correlate(ref_smooth, mov_smooth, mode="full")
    lags = signal.correlation_lags(len(ref_smooth), len(mov_smooth), mode="full") / target_fs
    mask = (lags >= bounds[0]) & (lags <= bounds[1])
    if not np.any(mask):
        return {"lag_s": np.nan, "score": 0.0}

    sub_corr = cross_corr[mask]
    sub_lags = lags[mask]
    idx = int(np.argmax(sub_corr))
    lag_s = float(sub_lags[idx])
    score = float(sub_corr[idx] / (np.std(sub_corr) + 1e-9))
    return {"lag_s": lag_s, "score": score}


def match_beats(ref_times: np.ndarray, pred_times: np.ndarray, tol_s: float = 0.15) -> Dict[str, float]:
    if len(ref_times) == 0 or len(pred_times) == 0:
        return {"matched": 0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "mean_abs_error_s": np.nan}
    used = np.zeros(len(pred_times), dtype=bool)
    errors = []
    matched = 0
    for rt in ref_times:
        idx = np.argmin(np.abs(pred_times - rt))
        err = abs(pred_times[idx] - rt)
        if err <= tol_s and not used[idx]:
            used[idx] = True
            matched += 1
            errors.append(float(err))
    precision = matched / max(len(pred_times), 1)
    recall = matched / max(len(ref_times), 1)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {
        "matched": matched,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_abs_error_s": float(np.mean(errors)) if errors else np.nan,
    }


def artifact_score(x: np.ndarray, fs: float) -> float:
    x = normalize(x)
    fast = np.mean(np.abs(np.diff(x)))
    burst = np.percentile(np.abs(x), 99) / (np.percentile(np.abs(x), 90) + 1e-9)
    drift = np.std(moving_average(x, fs, 1.5))
    return float(0.5 * fast + 0.3 * burst + 0.2 * drift)


def load_data(cfg: Config) -> Dict[str, object]:
    ps4000 = np.load(cfg.dataset_dir / FIBER_BUNDLE_A, mmap_mode="r")
    ps3000a = np.load(cfg.dataset_dir / FIBER_BUNDLE_B, mmap_mode="r")
    pvs = np.load(cfg.dataset_dir / PVS_FILE, mmap_mode="r")
    mic_fs, mic = wavfile.read(cfg.dataset_dir / MIC_FILE)
    mic = mic.astype(float)
    if mic.ndim > 1:
        mic = mic[:, 0]
    mic_t = np.arange(len(mic)) / mic_fs
    return {"ps4000": ps4000, "ps3000a": ps3000a, "pvs": pvs, "mic_fs": float(mic_fs), "mic": mic, "mic_t": mic_t}


def slice_data_for_focus(data: Dict[str, object], cfg: Config) -> Dict[str, object]:
    all_windows = list(cfg.windows) + [cfg.display_window, cfg.ppg_lag_window] + list(cfg.waveform_plot_windows)
    start = min(window[0] for window in all_windows)
    end = max(window[1] for window in all_windows)
    margin = cfg.mic_lag_bounds[1] + cfg.ppg_lag_bounds[1] + 2.0

    ps4000 = np.asarray(data["ps4000"])
    ps3000a = np.asarray(data["ps3000a"])
    pvs = np.asarray(data["pvs"])
    mic_t = np.asarray(data["mic_t"])
    mic = np.asarray(data["mic"])

    fiber_mask = (ps4000[:, 0] >= start - cfg.mic_lag_bounds[1] - 2.0) & (ps4000[:, 0] < end + 2.0)
    abdomen_mask = (ps3000a[:, 0] >= start - cfg.mic_lag_bounds[1] - 2.0) & (ps3000a[:, 0] < end + 2.0)
    ppg_mask = (pvs[:, 0] >= start - margin) & (pvs[:, 0] < end + 2.0)
    mic_mask = (mic_t >= start) & (mic_t < end)

    return {
        "ps4000": ps4000[fiber_mask],
        "ps3000a": ps3000a[abdomen_mask],
        "pvs": pvs[ppg_mask],
        "mic_fs": data["mic_fs"],
        "mic": mic[mic_mask],
        "mic_t": mic_t[mic_mask],
    }


def fixed_ppg_channel() -> int:
    return 1


def prepare_ppg(pvs: np.ndarray, col: int) -> Dict[str, np.ndarray | float]:
    t = np.asarray(pvs[:, 0], float)
    x = np.asarray(pvs[:, col], float)
    fs = 1.0 / np.median(np.diff(t))
    x = signal.detrend(x)
    x = butter_filter(x, fs, low=0.7, high=4.0, order=3)
    beats_pos = detect_beats(t, x, fs, (45.0, 140.0), prefer_positive=True)
    beats_neg = detect_beats(t, x, fs, (45.0, 140.0), prefer_positive=False)
    beats = beats_pos if beats_pos["score"] >= beats_neg["score"] else beats_neg
    return {"t": t, "x": x, "fs": fs, "beats": beats}


def prepare_fibers(ps4000: np.ndarray, ps3000a: np.ndarray, cfg: Config) -> Dict[str, object]:
    fiber_t = np.asarray(ps4000[:, 0], float)
    fiber_fs_native = 1.0 / np.median(np.diff(fiber_t))
    chest = np.asarray(ps4000[:, 1], float)
    abdomen = [np.asarray(ps4000[:, 2], float)] + [np.asarray(ps3000a[:, i], float) for i in range(1, 5)]
    abdomen_labels = ["ps4000_belly"] + [f"ps3000a_{i}" for i in range(1, 5)]
    chest_maternal_band = cheby1_bandpass_filter(chest, fiber_fs_native, cfg.maternal_acoustic_band_hz, order=6, rp=1.0)
    chest_beats = detect_maternal_fiber_beats(fiber_t, chest_maternal_band, fiber_fs_native, cfg.maternal_hr_bpm)

    return {
        "t": fiber_t,
        "fs": fiber_fs_native,
        "fs_native": fiber_fs_native,
        "chest": chest,
        "chest_maternal_band": chest_maternal_band,
        "chest_env": np.abs(chest_maternal_band),
        "chest_beats": chest_beats,
        "abdomen": abdomen,
        "abdomen_labels": abdomen_labels,
    }


def prepare_microphone(mic_t: np.ndarray, mic: np.ndarray, cfg: Config) -> Dict[str, np.ndarray | float]:
    t = np.asarray(mic_t, float)
    fs = 1.0 / np.median(np.diff(t))
    x_raw = signal.detrend(np.asarray(mic, float))
    x_proc = robust_clip(x_raw)
    hs = cheby1_bandpass_filter(x_proc, fs, cfg.fetal_detection_band_hz, order=3, rp=1.0)
    hs = suppress_transients(hs, fs, window_s=0.04)
    beats = detect_microphone_s1_peaks(t, x_proc)
    template = build_average_template(hs, beats["times"], fs, pre_s=0.01, post_s=0.03)
    feature = np.abs(hs)
    env = np.abs(signal.hilbert(hs))
    return {"t": t, "x": x_raw, "x_proc": x_proc, "hs": hs, "env": env, "feature": feature, "template": template, "fs": fs, "beats": beats}


def build_average_template(x: np.ndarray, beat_times: np.ndarray, fs: float, pre_s: float = 0.03, post_s: float = 0.05) -> np.ndarray:
    pre = int(round(pre_s * fs))
    post = int(round(post_s * fs))
    if len(beat_times) == 0:
        return np.array([1.0], dtype=float)
    snippets = []
    for bt in beat_times:
        c = int(round(bt * fs))
        if c - pre < 0 or c + post >= len(x):
            continue
        seg = x[c - pre : c + post + 1].astype(float)
        seg = normalize(seg)
        snippets.append(seg)
    if not snippets:
        return np.array([1.0], dtype=float)
    template = np.mean(np.vstack(snippets), axis=0)
    template = normalize(template)
    return template


def matched_feature(x: np.ndarray, template: np.ndarray, fs: float) -> np.ndarray:
    x = normalize(x)
    template = normalize(template)
    mf = signal.fftconvolve(x, template[::-1], mode="same")
    mf = np.abs(mf)
    mf = butter_filter(mf, fs, high=18.0, order=3)
    mf = moving_average(mf, fs, 0.012)
    return mf


def display_downsample(t: np.ndarray, x: np.ndarray, max_points: int = 800) -> Tuple[np.ndarray, np.ndarray]:
    if len(t) <= max_points:
        return t, x
    step = max(1, len(t) // max_points)
    return t[::step], x[::step]


def refine_peak_times_to_waveform(
    t: np.ndarray,
    waveform: np.ndarray,
    peak_times: np.ndarray,
    radius_s: float = 0.025,
    positive_only: bool = False,
) -> np.ndarray:
    if len(peak_times) == 0:
        return np.array([], dtype=float)
    fs = 1.0 / np.median(np.diff(t))
    radius = max(1, int(round(radius_s * fs)))
    x = np.asarray(waveform, float)
    w = np.maximum(x, 0.0) if positive_only else np.abs(x)
    refined = []
    for pt in peak_times:
        c = int(np.argmin(np.abs(t - pt)))
        lo = max(0, c - radius)
        hi = min(len(w), c + radius + 1)
        idx = lo + int(np.argmax(w[lo:hi]))
        refined.append(float(t[idx]))
    return np.asarray(refined, dtype=float)


def local_max_y(
    t: np.ndarray,
    x: np.ndarray,
    beat_times: np.ndarray,
    window_s: float = 0.04,
    positive_only: bool = False,
) -> np.ndarray:
    """For each beat time, return a local peak height around that time.

    When positive_only=True, markers are forced onto the strongest positive lobe
    in the local window so the plotted peak cannot land on a negative excursion.
    """
    if len(beat_times) == 0 or len(t) < 2:
        return np.zeros_like(beat_times, dtype=float)
    result = np.zeros(len(beat_times), dtype=float)
    metric = np.maximum(x, 0.0) if positive_only else np.abs(x)
    for i, bt in enumerate(beat_times):
        mask = (t >= bt - window_s) & (t <= bt + window_s)
        if np.any(mask):
            idx = np.argmax(metric[mask])
            local_x = x[mask]
            if positive_only and metric[mask][idx] <= 0:
                result[i] = float(np.maximum(local_x, 0.0).max(initial=0.0))
            else:
                result[i] = float(local_x[idx])
        else:
            j = int(np.clip(np.argmin(np.abs(t - bt)), 0, len(x) - 1))
            result[i] = float(max(x[j], 0.0) if positive_only else x[j])
    return result


def regression_beta(x: np.ndarray, ref: np.ndarray) -> float:
    denom = float(np.dot(ref, ref)) + 1e-9
    return float(np.dot(x, ref) / denom)


def generate_fetal_candidates(
    abdomen_chunk: np.ndarray,
    chest_x: np.ndarray,
    fs: float,
    cfg: Config,
) -> List[Dict[str, np.ndarray | float | str]]:
    X = np.asarray(abdomen_chunk, float)
    if X.ndim == 1:
        X = X[:, None]
    X = np.column_stack(
        [
            cheby1_bandpass_filter(np.asarray(X[:, i], float), fs, cfg.source_prep_band_hz, order=2, rp=1.0)
            for i in range(X.shape[1])
        ]
    )

    chest_maternal = cheby1_bandpass_filter(chest_x, fs, cfg.maternal_acoustic_band_hz, order=4, rp=1.0)
    candidates: List[Dict[str, np.ndarray | float | str]] = []

    # Manual-style method: fixed pair selection from the belly fibers.
    if X.shape[1] >= 2:
        pair_idx = [
            int(np.clip(cfg.selected_pair_idx[0], 0, X.shape[1] - 1)),
            int(np.clip(cfg.selected_pair_idx[1], 0, X.shape[1] - 1)),
        ]
    else:
        pair_idx = [0, 0]

    pair = np.column_stack([X[:, pair_idx[0]], X[:, pair_idx[1]]])
    raw_pair_a = pair[:, 0]
    raw_pair_b = pair[:, 1]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        try:
            ica = FastICA(n_components=2, random_state=42)
            sources = ica.fit_transform(pair)
        except Exception:
            sources = pair.copy()

    for k in range(sources.shape[1]):
        comp = np.asarray(sources[:, k], float)
        comp_maternal = cheby1_bandpass_filter(comp, fs, cfg.maternal_acoustic_band_hz, order=3, rp=1.0)
        if np.std(comp_maternal) < 1e-9 or np.std(chest_maternal) < 1e-9:
            corr = 0.0
        else:
            corr = float(np.corrcoef(comp_maternal, chest_maternal)[0, 1])
            if not np.isfinite(corr):
                corr = 0.0
        candidates.append(
            {
                "name": f"fastica_{k}",
                "signal": comp,
                "maternal_est": np.zeros_like(comp),
                "beta": abs(corr),
                "maternal_corr": corr,
                "family": "fastica_pair",
                "raw_pair_a": raw_pair_a,
                "raw_pair_b": raw_pair_b,
            }
        )

    return candidates


def choose_best_fetal_candidate(
    ft: np.ndarray,
    abdomen_chunk: np.ndarray,
    chest_x: np.ndarray,
    mic_t: np.ndarray,
    mic_feature: np.ndarray,
    mic_template: np.ndarray,
    mic_beats: np.ndarray,
    fs: float,
    cfg: Config,
) -> Dict[str, object]:
    candidates = generate_fetal_candidates(abdomen_chunk, chest_x, fs, cfg)
    best = None
    for cand in candidates:
        signal_x = np.asarray(cand["signal"])
        source_filtered = cheby1_bandpass_filter(signal_x, fs, cfg.fetal_detection_band_hz, order=3, rp=1.0)
        beats = detect_fetal_waveform_peaks(ft, signal_x, cfg.fetal_hr_bpm, target_count=len(mic_beats))
        source_times = np.asarray(beats["times"])
        if len(source_times) < 2 or len(mic_beats) < 2: # changed here
            continue
        t0 = float(min(np.min(source_times), np.min(mic_beats)))
        t1 = float(max(np.max(source_times), np.max(mic_beats)))
        lag_detail = impulse_xcorr_details(mic_beats, source_times, t0, t1, cfg.mic_lag_bounds, target_fs=200.0, smooth_sigma_s=0.08)
        lag_s = float(lag_detail["lag_s"]) if np.isfinite(lag_detail["lag_s"]) else 0.0
        matched = robust_match_peaks(source_times, mic_beats - lag_s, max_abs_dt=0.25, outlier_k=3.5)
        accepted = np.asarray(matched["accepted_mask"], dtype=bool)
        dt = np.asarray(matched["dt_signed"], dtype=float)
        dt_acc = dt[accepted]
        abs_dt_acc = np.abs(dt_acc)
        n_correct = int(np.sum(abs_dt_acc <= 0.05))
        accuracy = float(n_correct / len(mic_beats)) if len(mic_beats) else 0.0
        accepted_ratio = float(np.sum(accepted) / len(mic_beats)) if len(mic_beats) else 0.0
        count_similarity = 1.0 - abs(len(source_times) - len(mic_beats)) / max(len(mic_beats), 1)
        qual = signal_quality(source_filtered, fs, cfg.fetal_hr_bpm)
        score = (
            6.0 * accuracy
            + 1.5 * accepted_ratio
            + 0.5 * count_similarity
            + 0.25 * float(lag_detail["score"])
            + 0.10 * beats["score"]
            + 0.10 * qual["snr"]
            - 0.10 * abs(lag_s)
        )
        out = {
            "channel": str(cand["name"]),
            "score": float(score),
            "fetal_beats": beats,
            "fetal_match": {"f1": accepted_ratio, "mean_abs_error_s": float(np.mean(abs_dt_acc)) if abs_dt_acc.size else np.nan},
            "quality": qual,
            "artifact": np.nan,
            "signals": {
                "residual": np.asarray(cand["signal"]),
                "env": source_filtered,
                "maternal_est": np.asarray(cand["maternal_est"]),
                "source": source_filtered,
                "raw_pair_a": np.asarray(cand.get("raw_pair_a", signal_x)),
                "raw_pair_b": np.asarray(cand.get("raw_pair_b", signal_x)),
            },
            "corr_raw_mic": float(lag_detail["score"]),
            "maternal_corr": float(cand.get("maternal_corr", 0.0)),
            "lag_s": lag_s,
            "accuracy_50ms": accuracy,
        }
        if best is None or out["score"] > best["score"]:
            best = out
    return best if best is not None else {
        "channel": "",
        "score": -np.inf,
        "fetal_beats": {"times": np.array([]), "bpm": np.array([])},
        "fetal_match": {"f1": 0.0, "mean_abs_error_s": np.nan},
        "quality": {"snr": 0.0},
        "artifact": np.nan,
        "signals": {"residual": np.array([]), "env": np.array([]), "maternal_est": np.array([]), "source": np.array([]), "raw_pair_a": np.array([]), "raw_pair_b": np.array([])},
        "corr_raw_mic": 0.0,
    }


def estimate_ppg_fiber_lag(ppg: Dict[str, object], fibers: Dict[str, object], cfg: Config) -> Dict[str, float]:
    return estimate_lag_from_impulse_trains(
        np.asarray(fibers["chest_beats"]["times"]),
        np.asarray(ppg["beats"]["times"]),
        [cfg.ppg_lag_window],
        cfg.ppg_lag_bounds,
        target_fs=200.0,
        smooth_sigma_s=0.10,
    )


def estimate_fiber_mic_lag(
    fibers: Dict[str, object],
    mic: Dict[str, object],
    cfg: Config,
) -> Dict[str, float]:
    abdomen_matrix = np.column_stack(fibers["abdomen"])
    mic_t = np.asarray(mic["t"])
    mic_feature = np.asarray(mic["feature"])
    candidate = choose_best_fetal_candidate(
        np.asarray(fibers["t"]),
        abdomen_matrix,
        fibers["chest"],
        mic_t,
        mic_feature,
        np.asarray(mic["template"]),
        np.asarray(mic["beats"]["times"]),
        fibers["fs"],
        cfg,
    )
    beats = candidate["fetal_beats"]
    lag = estimate_lag_from_beats(np.asarray(mic["beats"]["times"]), np.asarray(beats["times"]), cfg.mic_lag_bounds, fs=20.0, sigma_s=0.1)
    lag["beat_score"] = float(beats["score"])
    return lag if np.isfinite(lag["lag_s"]) else {"lag_s": 0.0, "score": 0.0}


def chunk_mask(t: np.ndarray, start: float, end: float) -> np.ndarray:
    return (t >= start) & (t < end)


def beat_rows(
    modality: str,
    channel: str,
    beat_times: np.ndarray,
    ref_start: float,
    window_index: int,
    chunk_id: int,
) -> List[Dict[str, float | str]]:
    rows = []
    if len(beat_times) == 0:
        return rows
    ibi = np.diff(beat_times)
    inst_bpm = 60.0 / np.clip(ibi, 1e-6, None)
    for i, bt in enumerate(beat_times):
        row = {
            "window_index": window_index,
            "chunk_id": chunk_id,
            "modality": modality,
            "channel": channel,
            "beat_time_s": float(bt),
            "time_since_window_start_s": float(bt - ref_start),
            "ibi_s": float(ibi[i - 1]) if i > 0 else np.nan,
            "inst_bpm": float(inst_bpm[i - 1]) if i > 0 else np.nan,
        }
        rows.append(row)
    return rows


def analyze_chunks(
    ppg: Dict[str, object],
    fibers: Dict[str, object],
    mic: Dict[str, object],
    ppg_col: int,
    ppg_lag_s: float,
    mic_lag_s: float,
    cfg: Config,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    beat_rows_all: List[Dict[str, float | str]] = []
    chunk_rows: List[Dict[str, float | str]] = []
    chunk_id = 0

    for window_idx, (start, end) in enumerate(cfg.windows):
        for chunk_start in np.arange(start, end, cfg.chunk_seconds):
            chunk_end = min(chunk_start + cfg.chunk_seconds, end)
            chunk_id += 1

            mic_mask = chunk_mask(np.asarray(mic["t"]), chunk_start, chunk_end)
            mic_t = np.asarray(mic["t"])[mic_mask]
            mic_feature = np.asarray(mic["feature"])[mic_mask]
            mic_beats = detect_fetal_beats(mic_t, mic_feature, np.asarray(mic["hs"])[mic_mask], mic["fs"], cfg.fetal_hr_bpm)

            ppg_start = chunk_start - ppg_lag_s
            ppg_end = chunk_end - ppg_lag_s
            ppg_mask = chunk_mask(np.asarray(ppg["t"]), ppg_start, ppg_end)
            ppg_t = np.asarray(ppg["t"])[ppg_mask] + ppg_lag_s
            ppg_x = np.asarray(ppg["x"])[ppg_mask]
            ppg_beats = detect_beats(ppg_t, ppg_x, ppg["fs"], cfg.maternal_hr_bpm)

            fiber_start = chunk_start - mic_lag_s
            fiber_end = chunk_end - mic_lag_s
            fiber_mask = chunk_mask(np.asarray(fibers["t"]), fiber_start, fiber_end)
            ft = np.asarray(fibers["t"])[fiber_mask] + mic_lag_s
            chest_x = np.asarray(fibers["chest"])[fiber_mask]
            chest_band = np.asarray(fibers["chest_maternal_band"])[fiber_mask]
            chest_beats = detect_maternal_fiber_beats(ft, chest_band, fibers["fs"], cfg.maternal_hr_bpm)

            maternal_match = match_beats(np.asarray(ppg_beats["times"]), np.asarray(chest_beats["times"]), tol_s=0.2)
            mic_artifact = artifact_score(mic_feature, mic["fs"]) if len(mic_feature) else np.nan
            chest_artifact = artifact_score(chest_band, fibers["fs"]) if len(chest_band) else np.nan

            abdomen_matrix = np.column_stack([np.asarray(abdomen)[fiber_mask] for abdomen in fibers["abdomen"]])
            best_candidate = choose_best_fetal_candidate(
                ft,
                abdomen_matrix,
                chest_x,
                mic_t,
                mic_feature,
                np.asarray(mic["template"]),
                np.asarray(mic_beats["times"]),
                fibers["fs"],
                cfg,
            )

            beat_rows_all.extend(
                beat_rows("maternal_ppg", f"pvs_col_{ppg_col}", np.asarray(ppg_beats["times"]), chunk_start, window_idx, chunk_id)
            )
            beat_rows_all.extend(
                beat_rows("maternal_fiber", "fiber_1A", np.asarray(chest_beats["times"]), chunk_start, window_idx, chunk_id)
            )
            beat_rows_all.extend(
                beat_rows("fetal_microphone", "microphone", np.asarray(mic_beats["times"]), chunk_start, window_idx, chunk_id)
            )
            if best_candidate is not None:
                beat_rows_all.extend(
                    beat_rows(
                        "fetal_fiber",
                        best_candidate["channel"],
                        np.asarray(best_candidate["fetal_beats"]["times"]),
                        chunk_start,
                        window_idx,
                        chunk_id,
                    )
                )

            chunk_rows.append(
                {
                    "window_index": window_idx,
                    "chunk_id": chunk_id,
                    "chunk_start_s": chunk_start,
                    "chunk_end_s": chunk_end,
                    "ppg_lag_s": ppg_lag_s,
                    "fiber_to_mic_lag_s": mic_lag_s,
                    "selected_fetal_channel": best_candidate["channel"] if best_candidate else "",
                    "maternal_ppg_beats": len(ppg_beats["times"]),
                    "maternal_fiber_beats": len(chest_beats["times"]),
                    "fetal_mic_beats": len(mic_beats["times"]),
                    "fetal_fiber_beats": len(best_candidate["fetal_beats"]["times"]) if best_candidate else 0,
                    "maternal_bpm_ppg": float(np.nanmedian(ppg_beats["bpm"])) if len(ppg_beats["bpm"]) else np.nan,
                    "maternal_bpm_fiber": float(np.nanmedian(chest_beats["bpm"])) if len(chest_beats["bpm"]) else np.nan,
                    "fetal_bpm_mic": float(np.nanmedian(mic_beats["bpm"])) if len(mic_beats["bpm"]) else np.nan,
                    "fetal_bpm_fiber": float(np.nanmedian(best_candidate["fetal_beats"]["bpm"])) if best_candidate and len(best_candidate["fetal_beats"]["bpm"]) else np.nan,
                    "maternal_match_f1": maternal_match["f1"],
                    "maternal_match_mae_s": maternal_match["mean_abs_error_s"],
                    "fetal_match_f1": best_candidate["fetal_match"]["f1"] if best_candidate else 0.0,
                    "fetal_match_mae_s": best_candidate["fetal_match"]["mean_abs_error_s"] if best_candidate else np.nan,
                    "mic_artifact_score": mic_artifact,
                    "chest_artifact_score": chest_artifact,
                    "selected_fetal_artifact_score": best_candidate["artifact"] if best_candidate else np.nan,
                    "selected_fetal_quality_snr": best_candidate["quality"]["snr"] if best_candidate else np.nan,
                    "chunk_confidence": (
                        0.45 * maternal_match["f1"]
                        + 0.55 * (best_candidate["fetal_match"]["f1"] if best_candidate else 0.0)
                        - 0.05 * mic_artifact
                    ),
                }
            )

    return pd.DataFrame(beat_rows_all), pd.DataFrame(chunk_rows)


def plot_chunk_overlay(
    start: float,
    end: float,
    ppg: Dict[str, object],
    fibers: Dict[str, object],
    mic: Dict[str, object],
    ppg_lag_s: float,
    mic_lag_s: float,
    cfg: Config,
    output_path: Path,
) -> None:
    mic_mask = chunk_mask(np.asarray(mic["t"]), start, end)
    fiber_mask = chunk_mask(np.asarray(fibers["t"]) + mic_lag_s, start, end)
    ppg_mask = chunk_mask(np.asarray(ppg["t"]) + ppg_lag_s, start, end)

    mic_t = np.asarray(mic["t"])[mic_mask]
    mic_raw = normalize(np.asarray(mic["x"])[mic_mask])
    fiber_t = np.asarray(fibers["t"])[fiber_mask] + mic_lag_s
    chest_x = np.asarray(fibers["chest"])[fiber_mask]
    chest_band = np.asarray(fibers["chest_maternal_band"])[fiber_mask]
    ppg_t = np.asarray(ppg["t"])[ppg_mask] + ppg_lag_s
    ppg_x = np.asarray(ppg["x"])[ppg_mask]

    abdomen_matrix = np.column_stack([np.asarray(abdomen)[fiber_mask] for abdomen in fibers["abdomen"]])
    raw_abd = abdomen_matrix[:, 0]
    cand = choose_best_fetal_candidate(
        fiber_t,
        abdomen_matrix,
        chest_x,
        mic_t,
        np.asarray(mic["feature"])[mic_mask],
        np.asarray(mic["template"]),
        np.asarray(mic["beats"]["times"][(np.asarray(mic["beats"]["times"]) >= start) & (np.asarray(mic["beats"]["times"]) < end)]),
        fibers["fs"],
        cfg,
    )["signals"]

    fig, axes = plt.subplots(6, 1, figsize=(14, 14), sharex=True, constrained_layout=True)
    axes[0].plot(fiber_t, chest_band, color="tab:orange", lw=1.0, label="Fiber 1A 40-80 Hz")
    axes[0].legend(loc="upper right")
    axes[0].set_ylabel("Chest")

    axes[1].plot(ppg_t, ppg_x, color="tab:blue", lw=1.0, label="PPG filtered")
    axes[1].legend(loc="upper right")
    axes[1].set_ylabel("PPG")

    chest_beats = detect_maternal_fiber_beats(fiber_t, chest_band, fibers["fs"], cfg.maternal_hr_bpm)["times"]
    ppg_beats = detect_beats(ppg_t, ppg_x, ppg["fs"], cfg.maternal_hr_bpm, prefer_positive=True)["times"]
    for ax in axes[:2]:
        ax.vlines(chest_beats, ymin=ax.get_ylim()[0], ymax=ax.get_ylim()[1], color="k", alpha=0.25, lw=0.8)
        ax.vlines(ppg_beats, ymin=ax.get_ylim()[0], ymax=ax.get_ylim()[1], color="tab:red", alpha=0.18, lw=0.8)

    raw_abd_label = fibers.get("abdomen_labels", ["fiber_2"])[0]
    axes[2].plot(fiber_t, normalize(raw_abd), color="0.4", lw=0.9, label=f"abdomen raw ({raw_abd_label})")
    axes[2].plot(fiber_t, normalize(cand["source"]), color="tab:green", lw=1.0, label="Separated source")
    axes[2].plot(fiber_t, normalize(cand["maternal_est"]), color="tab:orange", lw=0.8, alpha=0.7, label="Estimated maternal")
    axes[2].legend(loc="upper right")
    axes[2].set_ylabel("Source")

    mic_t_plot, mic_raw_plot = display_downsample(mic_t, mic_raw)
    axes[3].plot(mic_t_plot, mic_raw_plot, color="tab:red", lw=0.9, alpha=0.9, label="NST raw")
    axes[3].set_ylabel("NST")

    src_norm = normalize(cand["source"])
    fiber_t_plot, src_plot = display_downsample(fiber_t, src_norm)
    axes[4].plot(fiber_t_plot, src_plot, color="tab:green", lw=0.9, alpha=0.95, label="Extracted source waveform")
    mic_det = detect_fetal_beats(mic_t, np.asarray(mic["feature"])[mic_mask], np.asarray(mic["hs"])[mic_mask], mic["fs"], cfg.fetal_hr_bpm)
    fib_det = detect_fetal_beats(fiber_t, cand["env"], cand["source"], fibers["fs"], cfg.fetal_hr_bpm)
    mic_beats = refine_peak_times_to_waveform(mic_t, mic_raw, mic_det["times"], radius_s=0.02)
    fiber_beats = refine_peak_times_to_waveform(fiber_t, src_norm, fib_det["times"], radius_s=0.02)

    # ylim from full-range of the raw waveforms (signed) with moderate padding
    mic_lo, mic_hi = float(np.min(mic_raw_plot)), float(np.max(mic_raw_plot))
    mic_pad = 0.08 * max(mic_hi - mic_lo, 1e-6)
    src_lo, src_hi = float(np.min(src_plot)), float(np.max(src_plot))
    src_pad = 0.08 * max(src_hi - src_lo, 1e-6)
    y3_lo, y3_hi = mic_lo - mic_pad, mic_hi + mic_pad
    y4_lo, y4_hi = src_lo - src_pad, src_hi + src_pad

    axes[3].vlines(mic_beats, y3_lo, y3_hi, color="tab:red", alpha=0.12, lw=0.8)
    # Place markers on the ENVELOPE at each detected beat -> markers land on the bump peaks
    mic_peak_y = np.interp(mic_beats, mic_t, mic_raw)
    axes[3].plot(
        mic_beats,
        mic_peak_y,
        "o",
        color="yellow",
        markeredgecolor="black",
        markeredgewidth=0.7,
        ms=6,
        zorder=6,
        label="NST peak",
    )
    axes[4].vlines(fiber_beats, y4_lo, y4_hi, color="tab:green", alpha=0.12, lw=0.8)
    fiber_peak_y = np.interp(fiber_beats, fiber_t, src_norm)
    axes[4].plot(
        fiber_beats,
        fiber_peak_y,
        "o",
        color="cyan",
        markeredgecolor="black",
        markeredgewidth=0.7,
        ms=6,
        zorder=6,
        label="Extracted peak",
    )
    axes[3].legend(loc="upper right")
    axes[4].legend(loc="upper right")
    axes[4].set_ylabel("Source")

    axes[5].vlines(fiber_beats, 0.0, 1.0, color="tab:blue", linewidth=1.3, alpha=0.95, label="Extracted beat")
    axes[5].vlines(mic_beats, 0.0, 1.7, color="tab:orange", linewidth=1.1, alpha=0.95, label="NST beat")
    axes[5].set_ylim(0, 2.0)
    axes[5].set_yticks([])
    axes[5].set_ylabel("Impulses")
    axes[5].legend(loc="upper right")

    for bt in mic_beats:
        axes[3].annotate(
            f"{bt-start:.2f}",
            xy=(bt, y3_hi),
            xytext=(0, 2),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=6,
            color="tab:red",
            rotation=90,
        )
    for bt in fiber_beats:
        axes[4].annotate(
            f"{bt-start:.2f}",
            xy=(bt, y4_hi),
            xytext=(0, 2),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=6,
            color="tab:green",
            rotation=90,
        )

    # Set ylim LAST so subsequent plot/legend/annotate calls cannot override
    axes[3].set_autoscale_on(True)
    axes[4].set_autoscale_on(True)
    # axes[3].set_ylim(y3_lo, y3_hi)
    # axes[4].set_ylim(y4_lo, y4_hi)

    axes[5].set_xlabel("Microphone time (s)")
    axes[5].set_title(f"Focused overlay {start:.0f}-{end:.0f} s")
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_window(
    window: Tuple[float, float],
    beat_df: pd.DataFrame,
    chunk_df: pd.DataFrame,
    ppg: Dict[str, object],
    fibers: Dict[str, object],
    mic: Dict[str, object],
    ppg_lag_s: float,
    mic_lag_s: float,
    cfg: Config,
    output_path: Path,
) -> None:
    start, end = window
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True, constrained_layout=True)

    mic_mask = chunk_mask(np.asarray(mic["t"]), start, end)
    mic_t = np.asarray(mic["t"])[mic_mask]
    mic_raw = normalize(np.asarray(mic["x"])[mic_mask])

    ppg_mask = chunk_mask(np.asarray(ppg["t"]) + ppg_lag_s, start, end)
    ppg_t = np.asarray(ppg["t"])[ppg_mask] + ppg_lag_s
    ppg_x = np.asarray(ppg["x"])[ppg_mask]

    fiber_mask = chunk_mask(np.asarray(fibers["t"]) + mic_lag_s, start, end)
    fiber_t = np.asarray(fibers["t"])[fiber_mask] + mic_lag_s
    chest_band = np.asarray(fibers["chest_maternal_band"])[fiber_mask]

    df0 = beat_df[(beat_df["beat_time_s"] >= start) & (beat_df["beat_time_s"] < end)]
    chest_times = df0[df0["modality"] == "maternal_fiber"]["beat_time_s"].to_numpy()
    ppg_times = df0[df0["modality"] == "maternal_ppg"]["beat_time_s"].to_numpy()

    axes[0].plot(fiber_t, chest_band, color="tab:orange", lw=1.0, alpha=0.95, label="Fiber 1A 40-80 Hz")
    axes[0].set_ylabel("Chest")
    axes[0].legend(loc="upper right")
    y0 = axes[0].get_ylim()
    axes[0].vlines(chest_times, y0[0], y0[1], color="k", alpha=0.28, lw=0.8)
    axes[0].vlines(ppg_times, y0[0], y0[1], color="tab:red", alpha=0.18, lw=0.8)

    axes[1].plot(ppg_t, ppg_x, color="tab:blue", lw=1.0, label="PPG filtered")
    axes[1].set_ylabel("PPG")
    axes[1].legend(loc="upper right")
    y1 = axes[1].get_ylim()
    axes[1].vlines(chest_times, y1[0], y1[1], color="k", alpha=0.28, lw=0.8)
    axes[1].vlines(ppg_times, y1[0], y1[1], color="tab:red", alpha=0.18, lw=0.8)

    mic_t_plot, mic_raw_plot = display_downsample(mic_t, mic_raw)
    axes[2].plot(mic_t_plot, mic_raw_plot, color="tab:red", lw=0.9, alpha=0.9, label="NST raw")
    df1 = df0[df0["modality"] == "fetal_fiber"]
    selected_channels = df1["channel"].unique().tolist()
    src_norm = None
    if selected_channels:
        abdomen_matrix = np.column_stack([np.asarray(abdomen)[fiber_mask] for abdomen in fibers["abdomen"]])
        best = choose_best_fetal_candidate(
            fiber_t,
            abdomen_matrix,
            np.asarray(fibers["chest"])[fiber_mask],
            mic_t,
            np.asarray(mic["feature"])[mic_mask],
            np.asarray(mic["template"]),
            np.asarray(mic["beats"]["times"][(np.asarray(mic["beats"]["times"]) >= start) & (np.asarray(mic["beats"]["times"]) < end)]),
            fibers["fs"],
            cfg,
        )
        src_norm = normalize(best["signals"]["source"])
        fiber_t_plot, src_plot = display_downsample(fiber_t, src_norm)
        axes[2].plot(fiber_t_plot, src_plot, color="tab:green", lw=0.9, alpha=0.95, label=best["channel"])
    mic_peak_times = df0[df0["modality"] == "fetal_microphone"]["beat_time_s"].to_numpy()
    fiber_peak_times = df0[df0["modality"] == "fetal_fiber"]["beat_time_s"].to_numpy()
    mic_lo, mic_hi = float(np.min(mic_raw_plot)), float(np.max(mic_raw_plot))
    mic_pad = 0.08 * max(mic_hi - mic_lo, 1e-6)
    if selected_channels:
        src_lo, src_hi = float(np.min(src_plot)), float(np.max(src_plot))
        src_pad = 0.08 * max(src_hi - src_lo, 1e-6)
        y2_lo = min(mic_lo - mic_pad, src_lo - src_pad)
        y2_hi = max(mic_hi + mic_pad, src_hi + src_pad)
    else:
        y2_lo, y2_hi = mic_lo - mic_pad, mic_hi + mic_pad
    for modality, color in [("fetal_microphone", "tab:red"), ("fetal_fiber", "tab:green")]:
        times = df0[df0["modality"] == modality]["beat_time_s"].to_numpy()
        axes[2].vlines(times, y2_lo, y2_hi, color=color, alpha=0.16, lw=0.8)
    if len(mic_peak_times):
        axes[2].plot(
            mic_peak_times,
            np.interp(mic_peak_times, mic_t, mic_raw),
            "o",
            color="yellow",
            markeredgecolor="black",
            markeredgewidth=0.6,
            ms=5,
            zorder=6,
        )
    if len(fiber_peak_times) and selected_channels:
        axes[2].plot(
            fiber_peak_times,
            np.interp(fiber_peak_times, fiber_t, src_norm),
            "o",
            color="cyan",
            markeredgecolor="black",
            markeredgewidth=0.6,
            ms=5,
            zorder=6,
        )
    axes[2].set_ylabel("Fetal Waveform")
    axes[2].legend(loc="upper right")

    for modality, color, label in [
        ("maternal_ppg", "tab:blue", "PPG maternal"),
        ("maternal_fiber", "tab:orange", "Fiber 1A maternal"),
        ("fetal_microphone", "tab:red", "Microphone fetal"),
        ("fetal_fiber", "tab:green", "Fiber fetal"),
    ]:
        sub = df0[df0["modality"] == modality]
        axes[3].plot(sub["beat_time_s"], sub["inst_bpm"], marker="o", ms=3, lw=1.0, color=color, label=label)

    cdf = chunk_df[(chunk_df["chunk_start_s"] >= start) & (chunk_df["chunk_start_s"] < end)]
    ax4 = axes[3].twinx()
    ax4.step(cdf["chunk_start_s"], cdf["chunk_confidence"], where="post", color="0.4", lw=1.2, label="Chunk confidence")
    ax4.set_ylabel("Confidence")
    ax4.set_ylim(min(-0.2, cdf["chunk_confidence"].min() - 0.1), max(1.5, cdf["chunk_confidence"].max() + 0.1))

    axes[3].set_ylabel("BPM")
    axes[3].set_xlabel("Aligned time (s)")
    axes[3].legend(loc="upper left", ncol=2)
    axes[3].set_title(f"Window {start:.0f}-{end:.0f} s")

    # Enforce ylim last (after all plot() calls) to prevent matplotlib autoscale override
    axes[2].set_autoscale_on(False)
    axes[2].set_ylim(y2_lo, y2_hi)

    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def chunk_bounds(window: Tuple[float, float], chunk_seconds: float) -> List[Tuple[float, float]]:
    start, end = window
    out: List[Tuple[float, float]] = []
    for chunk_start in np.arange(start, end, chunk_seconds):
        out.append((float(chunk_start), float(min(chunk_start + chunk_seconds, end))))
    return out


def window_tag(window: Tuple[float, float]) -> str:
    return f"{int(window[0])}_{int(window[1])}"


def impulse_xcorr_details(
    ref_times: np.ndarray,
    mov_times: np.ndarray,
    start: float,
    end: float,
    bounds: Tuple[float, float],
    target_fs: float = 200.0,
    smooth_sigma_s: float = 0.08,
) -> Dict[str, np.ndarray | float]:
    t_grid = np.arange(start, end, 1.0 / target_fs)
    ref_impulse = impulse_train_from_times(ref_times, t_grid)
    mov_impulse = impulse_train_from_times(mov_times, t_grid)
    ref_smooth = gaussian_smooth_signal(ref_impulse, smooth_sigma_s * target_fs)
    mov_smooth = gaussian_smooth_signal(mov_impulse, smooth_sigma_s * target_fs)
    corr = signal.correlate(ref_smooth, mov_smooth, mode="full")
    lags = signal.correlation_lags(len(ref_smooth), len(mov_smooth), mode="full") / target_fs
    mask = (lags >= bounds[0]) & (lags <= bounds[1])
    if np.any(mask):
        sub_corr = corr[mask]
        sub_lags = lags[mask]
        idx = int(np.argmax(sub_corr))
        lag_s = float(sub_lags[idx])
        score = float(sub_corr[idx] / (np.std(sub_corr) + 1e-9))
    else:
        sub_lags = np.array([], dtype=float)
        sub_corr = np.array([], dtype=float)
        lag_s = np.nan
        score = 0.0
    return {
        "t_grid": t_grid,
        "ref_impulse": ref_impulse,
        "mov_impulse": mov_impulse,
        "lags": sub_lags,
        "corr": sub_corr,
        "lag_s": lag_s,
        "score": score,
    }


def build_display_results(
    ppg: Dict[str, object],
    fibers: Dict[str, object],
    mic: Dict[str, object],
    ppg_lag_s: float,
    mic_lag_s: float,
    cfg: Config,
) -> Dict[str, object]:
    display_start, display_end = cfg.display_window
    chunks = chunk_bounds(cfg.display_window, cfg.chunk_seconds)

    maternal = []
    fetal = []
    for start, end in chunks:
        chest_mask = chunk_mask(np.asarray(fibers["t"]) + mic_lag_s, start, end)
        ppg_mask = chunk_mask(np.asarray(ppg["t"]) + mic_lag_s, start, end)
        mic_mask = chunk_mask(np.asarray(mic["t"]), start, end)
        if not np.any(chest_mask) or not np.any(ppg_mask) or not np.any(mic_mask):
            continue

        chest_t = np.asarray(fibers["t"])[chest_mask] + mic_lag_s
        chest_raw = np.asarray(fibers["chest"])[chest_mask]
        chest_band = np.asarray(fibers["chest_maternal_band"])[chest_mask]

        ppg_t = np.asarray(ppg["t"])[ppg_mask] + mic_lag_s
        ppg_raw = np.asarray(ppg["x"])[ppg_mask]
        ppg_beats = detect_beats(ppg_t, ppg_raw, ppg["fs"], cfg.maternal_hr_bpm, prefer_positive=True)
        chest_beats = detect_maternal_fiber_beats(
            chest_t,
            chest_band,
            fibers["fs"],
            cfg.maternal_hr_bpm,
            target_count=len(ppg_beats["times"]),
        )

        fiber_mask = chest_mask
        abdomen_matrix = np.column_stack([np.asarray(abdomen)[fiber_mask] for abdomen in fibers["abdomen"]])
        mic_t = np.asarray(mic["t"])[mic_mask]
        mic_raw = np.asarray(mic["x"])[mic_mask]
        mic_hs = np.asarray(mic["hs"])[mic_mask]
        mic_feature = np.asarray(mic["feature"])[mic_mask]
        fetal_target_count = int(round((end - start) * 100.0 / 60.0))
        mic_beats = detect_microphone_s1_peaks(mic_t, mic_raw, target_count=fetal_target_count)
        best = choose_best_fetal_candidate(
            chest_t - mic_lag_s,
            abdomen_matrix,
            chest_raw,
            mic_t,
            mic_feature,
            np.asarray(mic["template"]),
            np.asarray(mic_beats["times"]),
            fibers["fs"],
            cfg,
        )
        source_t_native = np.asarray(fibers["t"])[fiber_mask]
        source_t_aligned = source_t_native + mic_lag_s
        _raw_source = np.asarray(best["signals"]["source"])
        # If no valid ICA source found, fall back to zeros of the correct length
        source = _raw_source if len(_raw_source) == len(source_t_native) else np.zeros(len(source_t_native))
        source_feature = np.asarray(best["signals"]["env"])
        if len(source_feature) != len(source_t_native):
            source_feature = np.zeros(len(source_t_native))
        if len(source) >= 30:  # enough samples to filter
            source_beats_native = detect_fetal_waveform_peaks(
                source_t_native,
                source,
                cfg.fetal_hr_bpm,
                target_count=len(mic_beats["times"]),
            )
        else:
            source_beats_native = {"times": np.array([]), "bpm": np.array([]), "score": 0.0}
        source_beats_aligned = np.asarray(source_beats_native["times"]) + mic_lag_s

        maternal.append(
            {
                "start": start,
                "end": end,
                "chest_t": chest_t,
                "chest_raw": chest_raw,
                "chest_band": chest_band,
                "chest_beats": np.asarray(chest_beats["times"]),
                "ppg_t": ppg_t,
                "ppg_raw": ppg_raw,
                "ppg_beats": np.asarray(ppg_beats["times"]),
            }
        )
        fetal.append(
            {
                "start": start,
                "end": end,
                "mic_t": mic_t,
                "mic_raw": mic_raw,
                "mic_hs": mic_hs,
                "mic_beats": refine_peak_times_to_waveform(mic_t, mic_raw, np.asarray(mic_beats["times"]), radius_s=0.015, positive_only=True),
                "source_t_native": source_t_native,
                "source_t_aligned": source_t_aligned,
                "source_waveform": source,
                "source_beats_native": refine_peak_times_to_waveform(source_t_native, source, np.asarray(source_beats_native["times"]), radius_s=0.015, positive_only=True),
                "source_beats_aligned": refine_peak_times_to_waveform(source_t_aligned, source, source_beats_aligned, radius_s=0.015, positive_only=True),
                "channel": best["channel"],
            }
        )

    maternal_ref = np.concatenate([item["chest_beats"] for item in maternal]) if maternal else np.array([], dtype=float)
    maternal_mov = np.concatenate([item["ppg_beats"] for item in maternal]) if maternal else np.array([], dtype=float)
    fetal_ref = np.concatenate([item["mic_beats"] for item in fetal]) if fetal else np.array([], dtype=float)
    fetal_mov_native = np.concatenate([item["source_beats_native"] for item in fetal]) if fetal else np.array([], dtype=float)

    display_fetal = None
    mic_mask = chunk_mask(np.asarray(mic["t"]), display_start, display_end)
    fiber_mask = chunk_mask(np.asarray(fibers["t"]) + mic_lag_s, display_start, display_end)
    if np.any(mic_mask) and np.any(fiber_mask):
        mic_t = np.asarray(mic["t"])[mic_mask]
        mic_raw = np.asarray(mic["x"])[mic_mask]
        mic_feature = np.asarray(mic["feature"])[mic_mask]
        mic_beats_display_chunks: List[np.ndarray] = []
        for start, end in chunks:
            local_mask = chunk_mask(np.asarray(mic["t"]), start, end)
            if not np.any(local_mask):
                continue
            local_target = int(round((end - start) * 100.0 / 60.0))
            local = detect_microphone_s1_peaks(
                np.asarray(mic["t"])[local_mask],
                np.asarray(mic["x"])[local_mask],
                target_count=local_target,
            )
            mic_beats_display_chunks.append(np.asarray(local["times"]))
        mic_beats_display = np.concatenate(mic_beats_display_chunks) if mic_beats_display_chunks else np.array([], dtype=float)

        fiber_t_native = np.asarray(fibers["t"])[fiber_mask]
        fiber_t_aligned = fiber_t_native + mic_lag_s
        chest_x = np.asarray(fibers["chest"])[fiber_mask]
        abdomen_matrix = np.column_stack([np.asarray(abdomen)[fiber_mask] for abdomen in fibers["abdomen"]])
        whole = choose_best_fetal_candidate(
            fiber_t_aligned,
            abdomen_matrix,
            chest_x,
            mic_t,
            mic_feature,
            np.asarray(mic["template"]),
            mic_beats_display,
            fibers["fs"],
            cfg,
        )
        source = np.asarray(whole["signals"]["source"])
        raw_abd = abdomen_matrix[:, 0]
        source_beats_native = detect_fetal_waveform_peaks(
            fiber_t_native,
            source,
            cfg.fetal_hr_bpm,
            target_count=len(mic_beats_display),
        )
        source_beats_aligned = source_beats_native["times"] + mic_lag_s
        display_fetal = {
            "channel": whole["channel"],
            "raw_abd": raw_abd,
            "raw_abd_label": fibers.get("abdomen_labels", ["fiber_2"])[0],
            "maternal_est": np.asarray(whole["signals"]["maternal_est"]),
            "source_waveform": source,
            "source_t_native": fiber_t_native,
            "source_t_aligned": fiber_t_aligned,
            "mic_t": mic_t,
            "mic_raw": mic_raw,
            "mic_beats": refine_peak_times_to_waveform(mic_t, mic_raw, mic_beats_display, radius_s=0.015, positive_only=True),
            "source_beats_native": refine_peak_times_to_waveform(fiber_t_native, source, np.asarray(source_beats_native["times"]), radius_s=0.015, positive_only=True),
            "source_beats_aligned": refine_peak_times_to_waveform(fiber_t_aligned, source, source_beats_aligned, radius_s=0.015, positive_only=True),
        }
        fetal_ref = np.asarray(display_fetal["mic_beats"])
        fetal_mov_native = np.asarray(display_fetal["source_beats_native"])

    maternal_xcorr = impulse_xcorr_details(maternal_ref, maternal_mov, display_start, display_end, cfg.ppg_lag_bounds)
    fetal_xcorr = impulse_xcorr_details(fetal_ref, fetal_mov_native, display_start - cfg.mic_lag_bounds[1], display_end, cfg.mic_lag_bounds)

    return {
        "maternal_chunks": maternal,
        "fetal_chunks": fetal,
        "maternal_xcorr": maternal_xcorr,
        "fetal_xcorr": fetal_xcorr,
        "display_fetal": display_fetal,
        "display_start": display_start,
        "display_end": display_end,
    }


def plot_raw_signals(
    display: Dict[str, object],
    ppg_lag_s: float,
    mic_lag_s: float,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(15, 11), sharex=True, constrained_layout=True)

    for item in display["fetal_chunks"]:
        axes[0].plot(item["mic_t"], item["mic_raw"], color="firebrick", lw=0.8, alpha=0.9, rasterized=True)
        axes[3].plot(item["source_t_aligned"], item["source_waveform"], color="0.35", lw=0.8, alpha=0.9, rasterized=True)
    for item in display["maternal_chunks"]:
        axes[1].plot(item["ppg_t"] + ppg_lag_s, item["ppg_raw"], color="tab:blue", lw=0.8, alpha=0.95)
        axes[2].plot(item["chest_t"], item["chest_raw"], color="tab:orange", lw=0.8, alpha=0.95, rasterized=True)

    axes[0].set_ylabel("Mic raw")
    axes[1].set_ylabel("PPG raw\n(shifted)")
    axes[2].set_ylabel("Chest raw")
    axes[3].set_ylabel("Belly/source")
    axes[3].set_xlabel("Microphone time (s)")
    axes[0].set_title(f"Raw Signals {display['display_start']:.0f}-{display['display_end']:.0f} s")
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_maternal_alignment(display: Dict[str, object], output_path: Path) -> None:
    xcorr = display["maternal_xcorr"]
    lag_s = float(xcorr["lag_s"])
    fig, axes = plt.subplots(6, 1, figsize=(15, 15), sharex=False, constrained_layout=True)

    chest_beats_all: List[np.ndarray] = []
    ppg_beats_all: List[np.ndarray] = []
    for item in display["maternal_chunks"]:
        axes[0].plot(item["chest_t"], item["chest_band"], color="tab:orange", lw=1.0, rasterized=True)
        chest_y = np.interp(item["chest_beats"], item["chest_t"], item["chest_band"])
        axes[0].plot(item["chest_beats"], chest_y, "o", ms=6, color="cyan", markeredgecolor="black", markeredgewidth=0.6)
        chest_beats_all.append(item["chest_beats"])

        axes[1].plot(item["ppg_t"], item["ppg_raw"], color="tab:blue", lw=1.0)
        ppg_y = np.interp(item["ppg_beats"], item["ppg_t"], item["ppg_raw"])
        axes[1].plot(item["ppg_beats"], ppg_y, "o", ms=6, color="magenta", markeredgecolor="black", markeredgewidth=0.6)
        ppg_beats_all.append(item["ppg_beats"])

        axes[4].plot(item["chest_t"], item["chest_band"], color="tab:orange", lw=1.0, rasterized=True)
        axes[5].plot(item["ppg_t"] + lag_s, item["ppg_raw"], color="tab:blue", lw=1.0)
        axes[4].plot(item["chest_beats"], chest_y, "o", ms=5, color="cyan", markeredgecolor="black", markeredgewidth=0.6)
        axes[5].plot(item["ppg_beats"] + lag_s, ppg_y, "o", ms=5, color="magenta", markeredgecolor="black", markeredgewidth=0.6)

    chest_beats = np.concatenate(chest_beats_all) if chest_beats_all else np.array([], dtype=float)
    ppg_beats = np.concatenate(ppg_beats_all) if ppg_beats_all else np.array([], dtype=float)
    axes[2].vlines(chest_beats, 0.0, 1.0, color="tab:orange", linewidth=1.2, alpha=0.95, label="Chest beat")
    axes[2].vlines(ppg_beats, 0.0, 0.7, color="tab:blue", linewidth=1.2, alpha=0.95, label="PPG beat")
    axes[2].legend(loc="upper right")
    axes[2].set_yticks([])
    axes[2].set_ylabel("Impulse")

    axes[3].plot(xcorr["lags"], xcorr["corr"], color="0.2", lw=1.2)
    axes[3].axvline(lag_s, color="crimson", lw=1.2, ls="--", label=f"lag={lag_s:.3f}s")
    axes[3].legend(loc="upper right")
    axes[3].set_ylabel("XCorr")
    axes[3].set_xlabel("Lag (s)")

    axes[0].set_ylabel("Chest\n40-80 Hz")
    axes[1].set_ylabel("PPG\nfiltered")
    axes[4].set_ylabel("Chest\naligned ref")
    axes[5].set_ylabel("PPG\nshifted")
    axes[5].set_xlabel("Microphone time (s)")
    axes[0].set_title(f"Maternal Alignment {display['display_start']:.0f}-{display['display_end']:.0f} s")
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_fetal_alignment(display: Dict[str, object], output_path: Path) -> None:
    xcorr = display["fetal_xcorr"]
    lag_s = float(xcorr["lag_s"])
    fig, axes = plt.subplots(6, 1, figsize=(15, 16), sharex=False, constrained_layout=True)
    item = display.get("display_fetal")
    if item is None:
        plt.close(fig)
        return

    axes[0].plot(item["mic_t"], item["mic_raw"], color="firebrick", lw=0.8, alpha=0.9, rasterized=True)
    mic_y = np.interp(item["mic_beats"], item["mic_t"], item["mic_raw"])
    axes[0].plot(item["mic_beats"], mic_y, "o", ms=7, color="yellow", markeredgecolor="black", markeredgewidth=0.7)

    axes[1].plot(item["source_t_native"], item["source_waveform"], color="tab:green", lw=0.8, alpha=0.95, rasterized=True)
    src_y_native = local_max_y(
        item["source_t_native"],
        item["source_waveform"],
        np.asarray(item["source_beats_native"]),
        window_s=0.01,
        positive_only=True,
    )
    axes[1].plot(item["source_beats_native"], src_y_native, "o", ms=7, color="cyan", markeredgecolor="black", markeredgewidth=0.7)

    axes[4].plot(item["mic_t"], item["mic_raw"], color="firebrick", lw=0.8, alpha=0.9, rasterized=True)
    axes[4].plot(item["mic_beats"], mic_y, "o", ms=6, color="yellow", markeredgecolor="black", markeredgewidth=0.7)

    axes[5].plot(item["source_t_native"], item["source_waveform"], color="tab:green", lw=0.8, alpha=0.95, rasterized=True)
    axes[5].plot(item["source_beats_native"], src_y_native, "o", ms=6, color="cyan", markeredgecolor="black", markeredgewidth=0.7)

    mic_beats = np.asarray(item["mic_beats"])
    source_beats_native = np.asarray(item["source_beats_native"])
    mic_beats_shifted = mic_beats - lag_s
    mic_y_shifted = mic_y

    axes[4].cla()
    axes[4].plot(item["mic_t"] - lag_s, item["mic_raw"], color="firebrick", lw=0.8, alpha=0.9, rasterized=True)
    axes[4].plot(mic_beats_shifted, mic_y_shifted, "o", ms=6, color="yellow", markeredgecolor="black", markeredgewidth=0.7)

    axes[2].vlines(source_beats_native, 0.0, 1.0, color="tab:green", linewidth=1.2, alpha=0.95, label="Extracted beat")
    axes[2].vlines(mic_beats, 0.0, 0.7, color="firebrick", linewidth=1.2, alpha=0.95, label="Mic beat")
    axes[2].legend(loc="upper right")
    axes[2].set_yticks([])
    axes[2].set_ylabel("Impulse")

    axes[3].plot(xcorr["lags"], xcorr["corr"], color="0.2", lw=1.2)
    axes[3].axvline(lag_s, color="crimson", lw=1.2, ls="--", label=f"lag={lag_s:.3f}s")
    axes[3].legend(loc="upper right")
    axes[3].set_ylabel("XCorr")
    axes[3].set_xlabel("Lag (s)")

    axes[0].set_ylabel("Mic raw")
    axes[1].set_ylabel("Source\n(native)")
    axes[4].set_ylabel("Mic raw\n(shifted)")
    axes[5].set_ylabel("Source")
    axes[5].set_xlabel("Time (s)")
    axes[0].set_title(f"Fetal Extraction {display['display_start']:.0f}-{display['display_end']:.0f} s ({item['channel']})")
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_fetal_source_overlay(display: Dict[str, object], output_path: Path) -> None:
    item = display.get("display_fetal")
    if item is None:
        return
    fig, ax = plt.subplots(1, 1, figsize=(15, 3.2), constrained_layout=True)
    ax.plot(item["source_t_native"], normalize(item["raw_abd"]), color="0.35", lw=1.1, label=f"abdomen raw ({item.get('raw_abd_label', 'fiber_2')})")
    ax.plot(item["source_t_native"], normalize(item["source_waveform"]), color="tab:green", lw=1.2, label="Separated source")
    ax.plot(item["source_t_native"], normalize(item["maternal_est"]), color="tab:orange", lw=1.0, alpha=0.8, label="Estimated maternal")
    ax.set_xlabel("Fiber time (s)")
    ax.set_ylabel("Source")
    ax.set_title(f"Whole-window fetal source ({item['channel']})")
    ax.legend(loc="upper right")
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_focused_summary(
    cfg: Config,
    ppg_lag: Dict[str, float],
    mic_lag: Dict[str, float],
    display: Dict[str, object],
) -> None:
    payload = {
        "focus_window_s": list(cfg.windows[0]),
        "display_window_s": list(cfg.display_window),
        "ppg_to_chest_lag_s": float(ppg_lag["lag_s"]),
        "ppg_to_chest_score": float(ppg_lag["score"]),
        "fiber_to_microphone_lag_s": float(mic_lag["lag_s"]),
        "fiber_to_microphone_score": float(mic_lag["score"]),
        "maternal_display_lag_s": float(display["maternal_xcorr"]["lag_s"]),
        "fetal_display_lag_s": float(display["fetal_xcorr"]["lag_s"]),
        "num_display_chunks": int(len(display["maternal_chunks"])),
        "selected_fetal_channels": [item["channel"] for item in display["fetal_chunks"]],
    }
    with open(cfg.output_dir / f"focused_summary_{window_tag(cfg.windows[0])}.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def robust_match_peaks(
    ref_times: np.ndarray,
    pred_times: np.ndarray,
    max_abs_dt: float = 0.25,
    outlier_k: float = 3.5,
) -> Dict[str, np.ndarray | float]:
    ref_times = np.asarray(ref_times, dtype=float)
    pred_times = np.asarray(pred_times, dtype=float)
    ref_times = np.sort(ref_times[np.isfinite(ref_times)])
    pred_times = np.sort(pred_times[np.isfinite(pred_times)])

    used = np.zeros(len(pred_times), dtype=bool)
    ref_idx = []
    pred_idx = []
    dt = []

    for i, tr in enumerate(ref_times):
        left = np.searchsorted(pred_times, tr - max_abs_dt, side="left")
        right = np.searchsorted(pred_times, tr + max_abs_dt, side="right")
        if right <= left:
            continue
        cand = np.arange(left, right)
        cand = cand[~used[cand]]
        if cand.size == 0:
            continue
        j = cand[np.argmin(np.abs(pred_times[cand] - tr))]
        used[j] = True
        ref_idx.append(i)
        pred_idx.append(j)
        dt.append(float(pred_times[j] - tr))

    if not dt:
        empty_i = np.array([], dtype=int)
        empty_f = np.array([], dtype=float)
        return {
            "ref_idx": empty_i,
            "pred_idx": empty_i,
            "matched_ref_t": empty_f,
            "matched_pred_t": empty_f,
            "dt_signed": empty_f,
            "accepted_mask": np.array([], dtype=bool),
            "median_dt": np.nan,
            "sigma_mad": np.nan,
            "inlier_halfwidth": np.nan,
        }

    ref_idx = np.asarray(ref_idx, dtype=int)
    pred_idx = np.asarray(pred_idx, dtype=int)
    dt = np.asarray(dt, dtype=float)
    med = float(np.median(dt))
    mad = float(np.median(np.abs(dt - med)))
    sigma = 1.4826 * mad
    if sigma <= 1e-12:
        sigma = float(np.std(dt)) if np.std(dt) > 0 else 1e-6
    halfwidth = float(outlier_k * sigma)
    accepted_mask = np.abs(dt - med) <= halfwidth
    return {
        "ref_idx": ref_idx,
        "pred_idx": pred_idx,
        "matched_ref_t": ref_times[ref_idx],
        "matched_pred_t": pred_times[pred_idx],
        "dt_signed": dt,
        "accepted_mask": accepted_mask,
        "median_dt": med,
        "sigma_mad": sigma,
        "inlier_halfwidth": halfwidth,
    }


def plot_chunk_statistics(
    start: float,
    end: float,
    mic_times: np.ndarray,
    src_times_native: np.ndarray,
    lag_detail: Dict[str, np.ndarray | float],
    matched: Dict[str, np.ndarray | float],
    output_path: Path,
) -> None:
    lag_s = float(lag_detail["lag_s"])
    mic_times_shifted = mic_times - lag_s
    dt = np.asarray(matched["dt_signed"], dtype=float)
    accepted = np.asarray(matched["accepted_mask"], dtype=bool)

    fig, axes = plt.subplots(4, 1, figsize=(14, 12), constrained_layout=True)

    axes[0].vlines(src_times_native, 0.0, 1.0, color="tab:green", linewidth=1.2, alpha=0.95, label="Extracted beat")
    axes[0].vlines(mic_times, 0.0, 0.7, color="firebrick", linewidth=1.2, alpha=0.95, label="Mic beat")
    axes[0].set_xlim(float(min(np.min(src_times_native) if len(src_times_native) else start, start - 0.1)), float(end))
    axes[0].set_yticks([])
    axes[0].set_ylabel("Impulse")
    axes[0].set_title(f"Chunk {start:.0f}-{end:.0f} s: raw impulse trains")
    axes[0].legend(loc="upper right")

    axes[1].plot(lag_detail["lags"], lag_detail["corr"], color="0.2", lw=1.2)
    axes[1].axvline(lag_s, color="crimson", lw=1.2, ls="--", label=f"lag={lag_s:.3f}s")
    axes[1].set_ylabel("XCorr")
    axes[1].set_xlabel("Lag (s)")
    axes[1].legend(loc="upper right")

    axes[2].vlines(src_times_native, 0.0, 1.0, color="tab:green", linewidth=1.2, alpha=0.95, label="Extracted beat")
    axes[2].vlines(mic_times_shifted, 0.0, 0.7, color="firebrick", linewidth=1.2, alpha=0.95, label="Mic beat shifted")
    axes[2].set_xlim(float(start), float(end))
    axes[2].set_yticks([])
    axes[2].set_ylabel("Aligned")
    axes[2].set_title("Impulse trains after compensating lag (mic shifted)")
    axes[2].legend(loc="upper right")

    if dt.size:
        t_ref = np.asarray(matched["matched_ref_t"], dtype=float)
        axes[3].scatter(t_ref[accepted], dt[accepted], s=28, color="tab:blue", label=f"Accepted (n={accepted.sum()})")
        if np.any(~accepted):
            axes[3].scatter(t_ref[~accepted], dt[~accepted], s=38, marker="x", color="tab:red", label=f"Rejected (n={(~accepted).sum()})")
        axes[3].axhline(0.0, color="k", lw=1.0)
        axes[3].axhspan(-0.05, 0.05, color="tab:green", alpha=0.12, label="±0.05 s")
        axes[3].axhline(float(matched["median_dt"]), color="0.3", ls="--", lw=1.0, label=f"median={float(matched['median_dt']):.3f}s")
    axes[3].set_xlim(float(start), float(end))
    axes[3].set_xlabel("Microphone time (s)")
    axes[3].set_ylabel("dt (s)")
    axes[3].set_title("Relative time difference after robust matching")
    axes[3].legend(loc="upper right")

    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _xcorr_chunk_data(
    fibers: Dict[str, object],
    mic: Dict[str, object],
    ica_seg_cache: Dict[Tuple[float, float], Tuple[np.ndarray, np.ndarray]],
    cfg: "Config",
    xcorr_window_s: float = 5.0,
) -> Tuple[
    List[Tuple[np.ndarray, np.ndarray, float, float, float]],
    List[Dict[str, object]],
    int,
]:
    """Per-chunk xcorr: both ICA sources vs mic, global source selection.

    Pipeline per 20 s ICA segment:
      1. Apply fetal_detection_band to each of the 2 ICA sources
      2. Detect peaks on each filtered source → fiber_peaks_0, fiber_peaks_1
    Per 5 s chunk within the segment:
      3. Detect mic peaks in that chunk
      4. xcorr source_0_chunk vs mic_chunk  → lag0, score0
         xcorr source_1_chunk vs mic_chunk  → lag1, score1
    Globally across all chunks:
      5. chosen_k = argmax(median xcorr score) — held fixed for all chunks

    Returns (curves, csv_rows, chosen_k).
    """
    half_w            = xcorr_window_s / 2.0
    lag_bounds        = (-half_w, half_w)
    fs_fiber          = float(fibers["fs"])
    target_fs         = 200.0
    smooth_sigma_samp = 0.08 * target_fs   # 16 samples at 200 Hz

    # --- Mic peak detection over full analysis window (done once) ---
    mic_t_all    = np.asarray(mic["t"])
    mic_x_all    = np.asarray(mic["x"])
    win_s, win_e = float(cfg.windows[0][0]), float(cfg.windows[0][1])
    mic_win_mask = chunk_mask(mic_t_all, win_s, win_e)
    mic_t_win    = mic_t_all[mic_win_mask]
    mic_x_win    = mic_x_all[mic_win_mask]
    mic_beats_all = detect_microphone_s1_peaks(mic_t_win, mic_x_win)
    mic_peaks_all = refine_peak_times_to_waveform(
        mic_t_win, mic_x_win, np.asarray(mic_beats_all["times"]),
        radius_s=0.015, positive_only=True,
    )

    # --- Peak detection on each full 20 s segment (not per 5 s chunk) ---
    # Using the full segment gives the threshold estimator ~40 beats of context
    # instead of ~8, making it far more robust against edge effects and outliers.
    seg_peaks: Dict[Tuple[float, float], List[np.ndarray]] = {}  # seg_key → [peaks_k0, peaks_k1]
    for seg_key in sorted(ica_seg_cache.keys()):
        ft_seg, sources_seg = ica_seg_cache[seg_key]
        seg_start, seg_end = seg_key
        seg_mask = (ft_seg >= seg_start - 0.1) & (ft_seg < seg_end + 0.1)
        ft_s = ft_seg[seg_mask]
        peaks_per_k = []
        for k in range(2):
            src_filt = cheby1_bandpass_filter(
                sources_seg[seg_mask, k], fs_fiber, cfg.fetal_detection_band_hz, order=3, rp=1.0
            )
            # Use a lower threshold (0.15 vs default 0.40) for the xcorr path:
            # the 20 s segment may contain large transient spikes whose amplitude
            # would push the 0.40 threshold too high, causing regular heartbeat
            # peaks to be missed.  0.15 gives enough sensitivity on the full segment
            # while still requiring peaks to be above the noise floor.
            min_ibi_s = 60.0 / float(cfg.fetal_hr_bpm[1])  # e.g. 60/220 ≈ 0.27 s
            beats = manual_shannon_peak_detection(
                ft_s, src_filt,
                band_hz=cfg.fetal_detection_band_hz,
                min_interval_s=min_ibi_s,
                threshold_factor=0.15,
                order=3, rp=1.0,
            )
            peaks = refine_peak_times_to_waveform(
                ft_s, src_filt, np.asarray(beats["times"]), radius_s=0.015, positive_only=True
            )
            peaks_per_k.append(peaks)
        seg_peaks[seg_key] = peaks_per_k

    chunks     = list(chunk_bounds(cfg.windows[0], xcorr_window_s))
    chunk_data: List[Dict] = []

    for chunk_start, chunk_end in chunks:
        # Find the 20 s ICA segment that covers this chunk
        seg_key = None
        for k in ica_seg_cache:
            if k[0] <= chunk_start + 0.01 and k[1] >= chunk_end - 0.01:
                seg_key = k
                break
        if seg_key is None:
            chunk_data.append(None)
            continue

        # Mic peaks in this chunk → local time
        dur     = float(chunk_end - chunk_start)
        t_local = np.arange(0.0, dur, 1.0 / target_fs)
        mic_c   = mic_peaks_all[(mic_peaks_all >= chunk_start) & (mic_peaks_all < chunk_end)]
        if len(mic_c) < 2:
            chunk_data.append(None)
            continue
        imp_mic_s = gaussian_smooth_signal(
            impulse_train_from_times(mic_c - chunk_start, t_local), smooth_sigma_samp
        )

        row = {"chunk_start": chunk_start, "chunk_end": chunk_end,
               "lags": None, "corr": [None, None],
               "peak_lag": [np.nan, np.nan], "score": [0.0, 0.0]}

        for k in range(2):
            # Slice pre-detected segment peaks to this chunk window
            fiber_c = seg_peaks[seg_key][k]
            fiber_c = fiber_c[(fiber_c >= chunk_start) & (fiber_c < chunk_end)]
            if len(fiber_c) < 2:
                continue

            imp_fib_s = gaussian_smooth_signal(
                impulse_train_from_times(fiber_c - chunk_start, t_local), smooth_sigma_samp
            )
            corr   = signal.correlate(imp_fib_s, imp_mic_s, mode="full")
            lags   = signal.correlation_lags(len(imp_fib_s), len(imp_mic_s), mode="full") / target_fs
            m      = (lags >= lag_bounds[0]) & (lags <= lag_bounds[1])
            lags_m = lags[m]; corr_m = corr[m].astype(float)
            if len(corr_m) == 0:
                continue
            peak_idx           = int(np.argmax(corr_m))
            row["lags"]        = lags_m
            row["corr"][k]     = corr_m
            row["peak_lag"][k] = float(lags_m[peak_idx])
            row["score"][k]    = float(corr_m[peak_idx] / (np.std(corr_m) + 1e-9))

        chunk_data.append(row)

    # --- Global source selection: pick k with higher median xcorr score ---
    scores_by_k = [[], []]
    for row in chunk_data:
        if row is None:
            continue
        for k in range(2):
            if np.isfinite(row["peak_lag"][k]):
                scores_by_k[k].append(row["score"][k])
    med_scores = [np.median(s) if s else 0.0 for s in scores_by_k]
    chosen_k   = int(np.argmax(med_scores))

    # --- Build output using chosen source ---
    curves:   List[Tuple[np.ndarray, np.ndarray, float, float, float]] = []
    csv_rows: List[Dict[str, object]] = []
    for row in chunk_data:
        if row is None:
            continue
        cs, ce   = row["chunk_start"], row["chunk_end"]
        lags_m   = row["lags"]
        corr_m   = row["corr"][chosen_k]
        peak_lag = row["peak_lag"][chosen_k]
        if corr_m is None or lags_m is None:
            csv_rows.append({"chunk_start_s": cs, "chunk_end_s": ce, "peak_lag_s": float("nan")})
            continue
        curves.append((lags_m, corr_m, peak_lag, float(cs), float(ce)))
        csv_rows.append({"chunk_start_s": cs, "chunk_end_s": ce, "peak_lag_s": peak_lag})

    return curves, csv_rows, chosen_k, seg_peaks


def estimate_lag_via_local_xcorr(
    fibers: Dict[str, object],
    mic: Dict[str, object],
    ica_seg_cache: Dict[Tuple[float, float], Tuple[np.ndarray, np.ndarray]],
    cfg: "Config",
    xcorr_window_s: float = 5.0,
) -> float:
    """Estimate the fiber→mic delay as the median per-chunk local xcorr peak lag."""
    _, csv_rows, _, _ = _xcorr_chunk_data(fibers, mic, ica_seg_cache, cfg, xcorr_window_s)
    lags = np.array([row["peak_lag_s"] for row in csv_rows], dtype=float)
    valid = lags[np.isfinite(lags)]
    return float(np.median(valid)) if len(valid) > 0 else 0.0


def plot_xcorr_curve(
    fibers: Dict[str, object],
    mic: Dict[str, object],
    ica_seg_cache: Dict[Tuple[float, float], Tuple[np.ndarray, np.ndarray]],
    cfg: "Config",
    output_path: Path,
    csv_path: Path,
    xcorr_window_s: float = 5.0,
) -> None:
    """Grid of 4:3 per-chunk xcorr subplots — one panel per xcorr_window_s chunk.

    ICA runs on 20 s segments; both sources are xcorr'd against mic per 5 s chunk.
    The source with the higher median xcorr score is chosen as the fetal source.
    Positive lag = mic is ahead of fiber.
    """
    half_w     = xcorr_window_s / 2.0
    lag_bounds = (-half_w, half_w)
    win_start  = float(cfg.windows[0][0])
    win_end    = float(cfg.windows[0][1])
    group_size = max(1, int(round(float(cfg.ica_seconds or xcorr_window_s * 4) / xcorr_window_s)))

    curves, csv_rows, chosen_k, seg_peaks = _xcorr_chunk_data(fibers, mic, ica_seg_cache, cfg, xcorr_window_s)

    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
    if not curves:
        return

    cmap  = plt.cm.plasma
    ncols = group_size
    nrows = max(1, int(np.ceil(len(curves) / ncols)))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3 + 0.5), squeeze=False)
    t_min = min(c[3] for c in curves); t_max = max(c[3] for c in curves)

    for i, (lags, corr, peak_lag, t_start, t_end) in enumerate(curves):
        row, col = divmod(i, ncols)
        ax = axes[row][col]
        color = cmap((t_start - t_min) / max(t_max - t_min, 1.0))
        ax.plot(lags, corr, color=color, lw=1.2)
        ax.axhline(0, color="0.7", lw=0.5, ls="--")
        ax.axvline(0, color="0.7", lw=0.5, ls=":")
        if np.isfinite(peak_lag):
            peak_y = float(corr[np.argmin(np.abs(lags - peak_lag))])
            ax.plot(peak_lag, peak_y, "o", color="crimson", ms=5, zorder=6)
            ax.axvline(peak_lag, color="crimson", lw=1.0, ls="--", alpha=0.7)
        ax.set_xlim(lag_bounds[0], lag_bounds[1])
        ax.set_title(
            f"{t_start:.0f}–{t_end:.0f} s  (peak {peak_lag:+.2f} s)" if np.isfinite(peak_lag)
            else f"{t_start:.0f}–{t_end:.0f} s  (no peak)",
            fontsize=8, pad=3,
        )
        ax.tick_params(labelsize=7)
        ax.set_xlabel("Lag (s)", fontsize=7)
        ax.yaxis.set_ticklabels([])
        ax.grid(axis="x", alpha=0.25)

    for j in range(len(curves), nrows * ncols):
        row, col = divmod(j, ncols)
        axes[row][col].set_visible(False)

    fig.suptitle(
        f"xcorr per {xcorr_window_s:.0f} s chunk — {win_start:.0f}–{win_end:.0f} s  "
        f"(lag ±{half_w:.1f} s)  |  chosen ICA source: {chosen_k}  |  mic slid over fiber",
        fontsize=10, y=1.0,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- Waveform overview: chosen source (fetal-band filtered) + shifted mic chunks ---
    _plot_xcorr_waveform(fibers, mic, ica_seg_cache, seg_peaks, csv_rows, chosen_k,
                         cfg, win_start, win_end,
                         output_path.with_name(output_path.stem + "_waveform.png"))


def _plot_xcorr_waveform(
    fibers: Dict[str, object],
    mic: Dict[str, object],
    ica_seg_cache: Dict[Tuple[float, float], Tuple[np.ndarray, np.ndarray]],
    seg_peaks: Dict[Tuple[float, float], List[np.ndarray]],
    csv_rows: List[Dict[str, object]],
    chosen_k: int,
    cfg: "Config",
    win_start: float,
    win_end: float,
    output_path: Path,
) -> None:
    """Filtered fiber source (fixed) with detected peaks, and mic chunks shifted by lag.

    Top panel: fetal-band filtered chosen ICA source (from 20 s segments, stitched).
    Peaks are the same ones computed on the full 20 s segment (not re-detected per chunk).
    Bottom panel: mic chunks each shifted by their per-chunk lag onto fiber time axis.
    """
    fs_fiber = float(fibers["fs"])
    fig, axes = plt.subplots(2, 1, figsize=(16, 6), sharex=True, constrained_layout=True)

    # --- Top panel: stitch together fetal-filtered chosen source, reuse seg_peaks ---
    all_src_filt = []
    for seg_key in sorted(ica_seg_cache.keys()):
        ft_seg, sources_seg = ica_seg_cache[seg_key]
        seg_mask = (ft_seg >= win_start - 0.1) & (ft_seg < win_end + 0.1)
        if not np.any(seg_mask):
            continue
        ft_sub  = ft_seg[seg_mask]
        src_sub = sources_seg[seg_mask, chosen_k]
        filt    = cheby1_bandpass_filter(src_sub, fs_fiber, cfg.fetal_detection_band_hz, order=3, rp=1.0)
        axes[0].plot(ft_sub, filt, color="tab:green", lw=0.7, alpha=0.85)
        # Overlay pre-detected peaks (from full 20 s segment detection)
        if seg_key in seg_peaks:
            peaks = seg_peaks[seg_key][chosen_k]
            peaks_in_win = peaks[(peaks >= win_start) & (peaks < win_end)]
            if len(peaks_in_win):
                peak_vals = np.interp(peaks_in_win, ft_sub, filt)
                axes[0].plot(peaks_in_win, peak_vals, "o", color="cyan", ms=4, zorder=5)
        all_src_filt.append(filt)

    axes[0].set_ylabel(f"Source {chosen_k}\n(fetal band)", fontsize=8)
    axes[0].set_title(
        f"ICA source {chosen_k} — fetal-band filtered, peaks detected  "
        f"({win_start:.0f}–{win_end:.0f} s)", fontsize=9,
    )

    # --- Bottom panel: mic chunks shifted by per-chunk lag ---
    mic_t_all = np.asarray(mic["t"])
    mic_x_all = np.asarray(mic["x"])
    src_filt_all = np.concatenate(all_src_filt) if all_src_filt else np.array([1.0])
    scale = (float(np.std(src_filt_all)) or 1.0) / (float(np.std(mic_x_all)) or 1.0)

    for row in csv_rows:
        lag = float(row["peak_lag_s"])
        if not np.isfinite(lag):
            continue
        chunk_start, chunk_end = float(row["chunk_start_s"]), float(row["chunk_end_s"])
        m = chunk_mask(mic_t_all, chunk_start, chunk_end)
        if not np.any(m):
            continue
        axes[1].plot(mic_t_all[m] + lag, mic_x_all[m] * scale,
                     color="firebrick", lw=0.6, alpha=0.8)

    axes[1].set_ylabel("Mic (shifted)", fontsize=8)
    axes[1].set_xlabel("Fiber time (s)", fontsize=8)
    axes[1].set_title(
        "Mic chunks shifted by per-chunk lag onto fiber time axis  "
        "(gaps = mic ahead of fiber / packet drops)", fontsize=9,
    )
    axes[0].set_xlim(win_start, win_end)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_combined_impulse_trains(
    chunk_records: List[Dict[str, object]],
    window: Tuple[float, float],
    output_path: Path,
) -> None:
    if not chunk_records:
        return

    start, end = window
    fig, axes = plt.subplots(2, 1, figsize=(15, 7), sharex=True, constrained_layout=True)

    for rec in chunk_records:
        mic_times = np.asarray(rec["mic_times"], dtype=float)
        src_native = np.asarray(rec["src_times_native"], dtype=float)
        mic_shifted = np.asarray(rec["mic_times_shifted"], dtype=float)

        axes[0].vlines(src_native, 0.0, 1.0, color="tab:green", linewidth=1.0, alpha=0.95)
        axes[0].vlines(mic_times, 0.0, 0.7, color="firebrick", linewidth=1.0, alpha=0.95)

        axes[1].vlines(src_native, 0.0, 1.0, color="tab:green", linewidth=1.0, alpha=0.95)
        axes[1].vlines(mic_shifted, 0.0, 0.7, color="firebrick", linewidth=1.0, alpha=0.95)

        axes[0].axvspan(float(rec["start"]), float(rec["end"]), color="0.85", alpha=0.08)
        axes[1].axvspan(float(rec["start"]), float(rec["end"]), color="0.85", alpha=0.08)

    axes[0].set_xlim(float(start), float(end))
    axes[0].set_yticks([])
    axes[0].set_ylabel("Raw")
    axes[0].set_title(f"Combined impulse trains {start:.0f}-{end:.0f} s")
    axes[0].legend(
        handles=[
            plt.Line2D([0], [0], color="tab:green", lw=1.5, label="Extracted beat"),
            plt.Line2D([0], [0], color="firebrick", lw=1.5, label="Mic beat"),
        ],
        loc="upper right",
    )

    axes[1].set_xlim(float(start), float(end))
    axes[1].set_yticks([])
    axes[1].set_ylabel("Aligned")
    axes[1].set_xlabel("Microphone time (s)")
    axes[1].set_title("Combined impulse trains after per-chunk compensation (mic shifted)")

    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_shifted_waveform_window(
    window: Tuple[float, float],
    fibers: Dict[str, object],
    mic: Dict[str, object],
    mic_lag_s: float,
    ica_seg_cache: Dict[Tuple[float, float], Tuple[np.ndarray, np.ndarray]],
    cfg: Config,
    output_path: Path,
) -> None:
    """Four-panel plot using the shared 20 s ICA cache (consistent with stats CSV).

    Panel 1 – Mic raw shifted per 5 s xcorr lag.
    Panel 2 – 20 s ICA source (best component) with detected peaks.
    Panel 3 – Impulse trains: fiber peaks from 20 s ICA (blue), shifted mic peaks (orange).
    Panel 4 – dt scatter: robust_match_peaks on 20 s ICA peaks vs all shifted mic peaks.
    """
    start, end = window
    fs = float(fibers["fs"])
    fig, axes = plt.subplots(4, 1, figsize=(15, 13), sharex=True, constrained_layout=True,
                             gridspec_kw={"height_ratios": [2, 2, 1, 1.5]})

    # Find the ICA segment that covers this window (one 20 s segment per plot window)
    seg_key: Tuple[float, float] | None = None
    for k in ica_seg_cache:
        if k[0] <= start + 0.01 and k[1] >= end - 0.01:
            seg_key = k
            break
    # Fallback: find any overlapping segment
    if seg_key is None:
        for k in ica_seg_cache:
            if k[0] < end and k[1] > start:
                seg_key = k
                break

    # ------------------------------------------------------------------ #
    # Panel 2: pick best ICA component over the full 20 s window          #
    # ------------------------------------------------------------------ #
    src_peak_times = np.array([], dtype=float)   # fiber peaks (native time)
    ft_seg_full = np.array([], dtype=float)
    source_full = np.array([], dtype=float)

    if seg_key is not None:
        ft_seg_native, sources_seg = ica_seg_cache[seg_key]
        # Clip to the display window in native time
        win_mask = (ft_seg_native >= start - mic_lag_s - 0.1) & (ft_seg_native < end - mic_lag_s + 0.1)
        ft_seg_disp = ft_seg_native[win_mask]
        sources_disp = sources_seg[win_mask]

        # Score each component against full-window mic peaks to pick best
        mic_mask_full = chunk_mask(np.asarray(mic["t"]), start, end)
        mic_t_full_raw = np.asarray(mic["t"])[mic_mask_full]
        mic_raw_full_raw = np.asarray(mic["x"])[mic_mask_full]
        mic_beats_full = detect_microphone_s1_peaks(mic_t_full_raw, mic_raw_full_raw)
        mic_peak_times_full = refine_peak_times_to_waveform(
            mic_t_full_raw, mic_raw_full_raw,
            np.asarray(mic_beats_full["times"]), radius_s=0.015, positive_only=True,
        )

        best_source_filt: np.ndarray | None = None
        best_n_correct = -1
        for k in range(sources_disp.shape[1]):
            src_filt_k = cheby1_bandpass_filter(sources_disp[:, k], fs, cfg.fetal_detection_band_hz, order=3, rp=1.0)
            beats_k = detect_fetal_waveform_peaks(ft_seg_disp, src_filt_k, cfg.fetal_hr_bpm)
            src_times_k = np.asarray(beats_k["times"])
            if len(src_times_k) == 0 or len(mic_peak_times_full) == 0:
                continue
            ld_k = impulse_xcorr_details(
                mic_peak_times_full, src_times_k,
                float(start), float(end), cfg.mic_lag_bounds,
                target_fs=200.0, smooth_sigma_s=0.08,
            )
            lag_k = float(ld_k["lag_s"]) if np.isfinite(ld_k["lag_s"]) else 0.0
            m_k = robust_match_peaks(src_times_k, mic_peak_times_full - lag_k, max_abs_dt=0.25, outlier_k=3.5)
            dt_k = np.asarray(m_k["dt_signed"], dtype=float)
            acc_k = np.asarray(m_k["accepted_mask"], dtype=bool)
            n_corr_k = int(np.sum(np.abs(dt_k[acc_k]) <= 0.05))
            if n_corr_k > best_n_correct:
                best_n_correct = n_corr_k
                best_source_filt = src_filt_k

        if best_source_filt is not None:
            source_full = best_source_filt
            ft_seg_full = ft_seg_disp
            # Detect peaks on the full 20 s source
            seg_beats = detect_fetal_waveform_peaks(ft_seg_full, source_full, cfg.fetal_hr_bpm)
            src_peak_times = refine_peak_times_to_waveform(
                ft_seg_full, source_full,
                np.asarray(seg_beats["times"]), radius_s=0.015, positive_only=True,
            )
            # Plot Panel 2 waveform
            axes[1].plot(ft_seg_full, source_full, color="tab:green", lw=0.9, alpha=0.95, rasterized=True)
            if len(src_peak_times):
                src_peak_y = local_max_y(ft_seg_full, source_full, src_peak_times, window_s=0.01, positive_only=True)
                axes[1].plot(src_peak_times, src_peak_y, "o", color="cyan",
                             markeredgecolor="black", markeredgewidth=0.7, ms=6, zorder=6, label="Extracted peak (20 s ICA)")

    axes[1].set_ylabel("Extracted\n(20 s ICA)")
    axes[1].legend(loc="upper right")

    # ------------------------------------------------------------------ #
    # Panel 1: mic shifted per 5 s xcorr lag                             #
    # ------------------------------------------------------------------ #
    mic_peak_times_shifted_all: List[np.ndarray] = []
    mic_plot_t_all: List[np.ndarray] = []
    mic_plot_x_all: List[np.ndarray] = []

    for chunk_start, chunk_end in chunk_bounds(window, cfg.chunk_seconds):
        mic_mask = chunk_mask(np.asarray(mic["t"]), chunk_start, chunk_end)
        if not np.any(mic_mask):
            continue
        mic_t = np.asarray(mic["t"])[mic_mask]
        mic_raw = np.asarray(mic["x"])[mic_mask]
        mic_beats = detect_microphone_s1_peaks(mic_t, mic_raw)
        mic_peak_times_chunk = refine_peak_times_to_waveform(
            mic_t, mic_raw, np.asarray(mic_beats["times"]), radius_s=0.015, positive_only=True,
        )
        # Shift mic into fiber native time using the global mic_lag_s.
        # Per-chunk xcorr would be wrong here because mic peaks (mic-time) and
        # fiber peaks (native-time) live in different frames on the same grid.
        shifted_t   = mic_t                  - mic_lag_s
        shifted_pks = mic_peak_times_chunk   - mic_lag_s

        axes[0].plot(shifted_t, mic_raw, color="firebrick", lw=0.9, alpha=0.9, rasterized=True)
        mic_peak_times_shifted_all.append(shifted_pks)
        mic_plot_t_all.append(shifted_t)
        mic_plot_x_all.append(mic_raw)

    all_mic_shifted = np.concatenate(mic_peak_times_shifted_all) if mic_peak_times_shifted_all else np.array([], dtype=float)

    if mic_plot_t_all and len(all_mic_shifted):
        mic_t_cat = np.concatenate(mic_plot_t_all)
        mic_x_cat = np.concatenate(mic_plot_x_all)
        order = np.argsort(mic_t_cat)
        mic_peak_y = np.interp(all_mic_shifted, mic_t_cat[order], mic_x_cat[order])
        axes[0].plot(all_mic_shifted, mic_peak_y, "o", color="yellow",
                     markeredgecolor="black", markeredgewidth=0.7, ms=6, zorder=6,
                     label="Shifted microphone peak")

    axes[0].set_ylabel("Mic raw\n(shifted)")
    axes[0].legend(loc="upper right")

    # ------------------------------------------------------------------ #
    # Panel 3: impulse trains                                             #
    # ------------------------------------------------------------------ #
    if len(src_peak_times):
        axes[2].vlines(src_peak_times, 0.0, 1.0, color="tab:blue", linewidth=1.2, alpha=0.95, label="Fiber beat (20 s ICA)")
    if len(all_mic_shifted):
        axes[2].vlines(all_mic_shifted, 0.0, 1.0, color="tab:orange", linewidth=1.2, alpha=0.75, label="NST beat (shifted)")
    axes[2].set_ylim(0, 1.3)
    axes[2].set_yticks([])
    axes[2].set_ylabel("Impulses")
    axes[2].legend(loc="upper right", fontsize=8)
    axes[2].set_title("Impulse trains after shift (20 s ICA peaks)")

    # ------------------------------------------------------------------ #
    # Panel 4: dt scatter — 20 s ICA peaks vs all shifted mic peaks      #
    # ------------------------------------------------------------------ #
    if len(src_peak_times) >= 2 and len(all_mic_shifted) >= 2:
        matched = robust_match_peaks(src_peak_times, all_mic_shifted, max_abs_dt=0.25, outlier_k=3.5)
        all_ref_t   = np.asarray(matched["matched_ref_t"], dtype=float)
        all_dt      = np.asarray(matched["dt_signed"], dtype=float)
        all_accepted = np.asarray(matched["accepted_mask"], dtype=bool)
        overall_median = float(np.median(all_dt[all_accepted])) if np.any(all_accepted) else 0.0
        axes[3].scatter(all_ref_t[all_accepted], all_dt[all_accepted],
                        s=22, color="tab:blue", zorder=5,
                        label=f"Accepted (n={int(all_accepted.sum())})")
        if np.any(~all_accepted):
            axes[3].scatter(all_ref_t[~all_accepted], all_dt[~all_accepted],
                            s=32, marker="x", color="tab:red", zorder=5,
                            label=f"Rejected (n={int((~all_accepted).sum())})")
        axes[3].axhline(0.0, color="k", lw=1.0)
        axes[3].axhspan(-0.25, 0.25, color="tab:orange", alpha=0.10, label="±0.25 s acceptance window")
        axes[3].axhspan(-0.05, 0.05, color="tab:green", alpha=0.20, label="±0.05 s correct")
        axes[3].axhline(overall_median, color="0.35", ls="--", lw=1.0,
                        label=f"median={overall_median:.3f} s")
    axes[3].set_ylabel("dt (s)")
    axes[3].set_xlabel("Fiber time (s)")
    axes[3].legend(loc="upper right", fontsize=8)
    axes[3].set_title(f"dt: 20 s ICA peaks vs mic shifted by local xcorr lag ({mic_lag_s:.2f} s)")

    # x-axis in fiber native time (fixed reference).
    # mic is shifted to align with fiber, so gaps in mic chunks are visible.
    fiber_start = float(start) - mic_lag_s
    fiber_end   = float(end)   - mic_lag_s
    for ax in axes:
        ax.set_xlim(fiber_start, fiber_end)
    if int(start) == 600 and int(end) == 620:
        axes[0].set_ylim(-5000, 5000)

    axes[0].set_title(
        f"Fiber {fiber_start:.1f}–{fiber_end:.1f} s  "
        f"(mic {start:.0f}–{end:.0f} s shifted by local xcorr lag {mic_lag_s:.2f} s)"
    )
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def analyze_fetal_statistics(
    ppg: Dict[str, object],
    fibers: Dict[str, object],
    mic: Dict[str, object],
    ppg_lag_s: float,
    mic_lag_s: float,
    cfg: Config,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: List[Dict[str, float | str]] = []
    match_rows: List[Dict[str, float | str]] = []
    chunk_records: List[Dict[str, object]] = []
    pooled_total_mic_peaks = 0
    pooled_total_extracted_peaks = 0
    pooled_correct_50ms = 0
    pooled_dt_acc: List[float] = []

    for chunk_id, (start, end) in enumerate(chunk_bounds(cfg.windows[0], cfg.chunk_seconds), start=1):
        mic_mask = chunk_mask(np.asarray(mic["t"]), start, end)
        fiber_mask = chunk_mask(np.asarray(fibers["t"]) + mic_lag_s, start, end)
        if not np.any(mic_mask) or not np.any(fiber_mask):
            continue

        mic_t = np.asarray(mic["t"])[mic_mask]
        mic_raw = np.asarray(mic["x"])[mic_mask]
        mic_feature = np.asarray(mic["feature"])[mic_mask]
        chunk_target = int(round((end - start) * 100.0 / 60.0))
        mic_beats = detect_microphone_s1_peaks(mic_t, mic_raw, target_count=chunk_target)
        mic_peak_times = refine_peak_times_to_waveform(mic_t, mic_raw, np.asarray(mic_beats["times"]), radius_s=0.015, positive_only=True)

        fiber_t_native = np.asarray(fibers["t"])[fiber_mask]
        chest_x = np.asarray(fibers["chest"])[fiber_mask]
        abdomen_matrix = np.column_stack([np.asarray(abdomen)[fiber_mask] for abdomen in fibers["abdomen"]])
        best = choose_best_fetal_candidate(
            fiber_t_native + mic_lag_s,
            abdomen_matrix,
            chest_x,
            mic_t,
            mic_feature,
            np.asarray(mic["template"]),
            mic_peak_times,
            fibers["fs"],
            cfg,
        )
        source = np.asarray(best["signals"]["source"])
        source_beats = detect_fetal_waveform_peaks(
            fiber_t_native,
            source,
            cfg.fetal_hr_bpm,
            target_count=len(mic_peak_times),
        )
        source_peak_times_native = refine_peak_times_to_waveform(
            fiber_t_native,
            source,
            np.asarray(source_beats["times"]),
            radius_s=0.015,
            positive_only=True,
        )

        t0 = float(min(np.min(source_peak_times_native) if len(source_peak_times_native) else start - cfg.mic_lag_bounds[1], np.min(mic_peak_times) if len(mic_peak_times) else start))
        t1 = float(max(np.max(mic_peak_times) if len(mic_peak_times) else end, np.max(source_peak_times_native) if len(source_peak_times_native) else end))
        lag_detail = impulse_xcorr_details(
            mic_peak_times,
            source_peak_times_native,
            t0,
            t1,
            cfg.mic_lag_bounds,
            target_fs=200.0,
            smooth_sigma_s=0.08,
        )
        lag_s = float(lag_detail["lag_s"]) if np.isfinite(lag_detail["lag_s"]) else 0.0
        matched = robust_match_peaks(source_peak_times_native, mic_peak_times - lag_s, max_abs_dt=0.25, outlier_k=3.5)

        dt = np.asarray(matched["dt_signed"], dtype=float)
        accepted = np.asarray(matched["accepted_mask"], dtype=bool)
        dt_acc = dt[accepted]
        abs_dt_acc = np.abs(dt_acc)
        n_correct = int(np.sum(abs_dt_acc <= 0.05))
        accuracy = float(n_correct / len(mic_peak_times)) if len(mic_peak_times) else np.nan
        pooled_total_mic_peaks += int(len(mic_peak_times))
        pooled_total_extracted_peaks += int(len(source_peak_times_native))
        pooled_correct_50ms += n_correct
        pooled_dt_acc.extend(dt_acc.tolist())

        summary_rows.append(
            {
                "chunk_id": chunk_id,
                "chunk_start_s": float(start),
                "chunk_end_s": float(end),
                "selected_channel": best["channel"],
                "mic_peak_count": int(len(mic_peak_times)),
                "extracted_peak_count": int(len(source_peak_times_native)),
                "xcorr_lag_s": lag_s,
                "xcorr_score": float(lag_detail["score"]),
                "accepted_pairs": int(np.sum(accepted)),
                "mean_dt_s": float(np.mean(dt_acc)) if dt_acc.size else np.nan,
                "std_dt_s": float(np.std(dt_acc)) if dt_acc.size else np.nan,
                "mean_abs_dt_s": float(np.mean(abs_dt_acc)) if abs_dt_acc.size else np.nan,
                "std_abs_dt_s": float(np.std(abs_dt_acc)) if abs_dt_acc.size else np.nan,
                "median_abs_dt_s": float(np.median(abs_dt_acc)) if abs_dt_acc.size else np.nan,
                "accuracy_50ms": accuracy,
            }
        )

        for i in range(len(dt)):
            match_rows.append(
                {
                    "chunk_id": chunk_id,
                    "chunk_start_s": float(start),
                    "chunk_end_s": float(end),
                    "selected_channel": best["channel"],
                    "extracted_time_s": float(np.asarray(matched["matched_ref_t"])[i]),
                    "mic_time_shifted_s": float(np.asarray(matched["matched_pred_t"])[i]),
                    "dt_s": float(dt[i]),
                    "abs_dt_s": float(abs(dt[i])),
                    "accepted": bool(accepted[i]),
                    "correct_50ms": bool(abs(dt[i]) <= 0.05 and accepted[i]),
                }
            )

        chunk_records.append(
            {
                "start": float(start),
                "end": float(end),
                "mic_times": mic_peak_times,
                "src_times_native": source_peak_times_native,
                "mic_times_shifted": mic_peak_times - lag_s,
            }
        )

    plot_combined_impulse_trains(
        chunk_records,
        cfg.windows[0],
        cfg.output_dir / f"stats_impulses_full_{window_tag(cfg.windows[0])}.png",
    )
    pooled_dt_acc_arr = np.asarray(pooled_dt_acc, dtype=float)
    summary_rows.append(
        {
            "chunk_id": "pooled",
            "chunk_start_s": float(cfg.windows[0][0]),
            "chunk_end_s": float(cfg.windows[0][1]),
            "selected_channel": "ALL",
            "mic_peak_count": int(pooled_total_mic_peaks),
            "extracted_peak_count": int(pooled_total_extracted_peaks),
            "xcorr_lag_s": np.nan,
            "xcorr_score": np.nan,
            "accepted_pairs": int(len(pooled_dt_acc_arr)),
            "mean_dt_s": float(np.mean(pooled_dt_acc_arr)) if pooled_dt_acc_arr.size else np.nan,
            "std_dt_s": float(np.std(pooled_dt_acc_arr)) if pooled_dt_acc_arr.size else np.nan,
            "mean_abs_dt_s": float(np.mean(np.abs(pooled_dt_acc_arr))) if pooled_dt_acc_arr.size else np.nan,
            "std_abs_dt_s": float(np.std(np.abs(pooled_dt_acc_arr))) if pooled_dt_acc_arr.size else np.nan,
            "median_abs_dt_s": float(np.median(np.abs(pooled_dt_acc_arr))) if pooled_dt_acc_arr.size else np.nan,
            "accuracy_50ms": float(pooled_correct_50ms / pooled_total_mic_peaks) if pooled_total_mic_peaks else np.nan,
        }
    )
    return pd.DataFrame(summary_rows), pd.DataFrame(match_rows)


def build_ica_segment_cache(
    fibers: Dict[str, object],
    mic_lag_s: float,
    cfg: Config,
    ica_seconds: float,
) -> Dict[Tuple[float, float], Tuple[np.ndarray, np.ndarray]]:
    """Pre-compute FastICA for every ``ica_seconds`` segment of the analysis window.

    Returns a dict keyed by (seg_start_mic, seg_end_mic) → (ft_native, sources).
    ``ft_native`` is the native fiber timestamp array; ``sources`` is (T, 2).
    """
    fs = float(fibers["fs"])
    pair_idx = cfg.selected_pair_idx
    ft_all = np.asarray(fibers["t"], float)
    abdomen = fibers["abdomen"]
    ica_seg_cache: Dict[Tuple[float, float], Tuple[np.ndarray, np.ndarray]] = {}

    for seg_start, seg_end in chunk_bounds(cfg.windows[0], ica_seconds):
        seg_start_native = seg_start - mic_lag_s
        seg_end_native = seg_end - mic_lag_s
        seg_mask = (ft_all >= seg_start_native - 0.5) & (ft_all < seg_end_native + 0.5)
        if not np.any(seg_mask):
            continue
        ft_seg = ft_all[seg_mask]
        a = np.asarray(abdomen[pair_idx[0]], float)[seg_mask]
        b = np.asarray(abdomen[pair_idx[1]], float)[seg_mask]
        a_filt = cheby1_bandpass_filter(a, fs, cfg.source_prep_band_hz, order=2, rp=1.0)
        b_filt = cheby1_bandpass_filter(b, fs, cfg.source_prep_band_hz, order=2, rp=1.0)
        X = np.column_stack([a_filt, b_filt])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            try:
                ica = FastICA(n_components=2, random_state=42, max_iter=cfg.ica_max_iter, tol=cfg.ica_tol)
                sources = ica.fit_transform(X)
            except Exception:
                sources = X.copy()
        ica_seg_cache[(float(seg_start), float(seg_end))] = (ft_seg, sources)

    return ica_seg_cache


def analyze_fetal_statistics_two_level(
    ppg: Dict[str, object],
    fibers: Dict[str, object],
    mic: Dict[str, object],
    ppg_lag_s: float,
    mic_lag_s: float,
    cfg: Config,
    ica_seconds: float,
    ica_seg_cache: Dict[Tuple[float, float], Tuple[np.ndarray, np.ndarray]] | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Two-level chunking: ICA on ``ica_seconds`` segments, stats on ``cfg.chunk_seconds`` sub-chunks.

    If ``ica_seg_cache`` is supplied (pre-built by ``build_ica_segment_cache``) it is used
    directly; otherwise the cache is built here.  Identical output columns to
    ``analyze_fetal_statistics``.
    """
    summary_rows: List[Dict[str, float | str]] = []
    match_rows: List[Dict[str, float | str]] = []
    chunk_records: List[Dict[str, object]] = []
    pooled_total_mic_peaks = 0
    pooled_total_extracted_peaks = 0
    pooled_correct_50ms = 0
    pooled_dt_acc: List[float] = []
    fs = float(fibers["fs"])

    # -----------------------------------------------------------------------
    # Phase 1: ICA segment cache (reuse if already built)
    # -----------------------------------------------------------------------
    if ica_seg_cache is None:
        ica_seg_cache = build_ica_segment_cache(fibers, mic_lag_s, cfg, ica_seconds)

    # -----------------------------------------------------------------------
    # Phase 2: stats per cfg.chunk_seconds sub-chunk using cached ICA sources.
    # Component selection is done ONCE per 20 s ICA segment (outer loop) so
    # that selected_channel is consistent across all 5 s chunks in a segment.
    # -----------------------------------------------------------------------
    chunk_id = 0  # global counter across segments
    for seg_key in sorted(ica_seg_cache.keys()):
        seg_start, seg_end = seg_key          # mic-time coordinates
        ft_seg_native, sources_seg = ica_seg_cache[seg_key]

        # --- Full 20 s mic peaks for component scoring ---
        mic_mask_full = chunk_mask(np.asarray(mic["t"]), seg_start, seg_end)
        if not np.any(mic_mask_full):
            continue
        mic_t_full   = np.asarray(mic["t"])[mic_mask_full]
        mic_raw_full = np.asarray(mic["x"])[mic_mask_full]
        seg_target   = int(round((seg_end - seg_start) * 100.0 / 60.0))
        mic_beats_full = detect_microphone_s1_peaks(mic_t_full, mic_raw_full, target_count=seg_target)
        mic_peak_times_full = refine_peak_times_to_waveform(
            mic_t_full, mic_raw_full, np.asarray(mic_beats_full["times"]),
            radius_s=0.015, positive_only=True,
        )

        # --- Full 20 s source slice (native fiber time) ---
        win_mask = (ft_seg_native >= seg_start - mic_lag_s - 0.1) & \
                   (ft_seg_native <  seg_end   - mic_lag_s + 0.1)
        if not np.any(win_mask):
            continue
        ft_seg_disp  = ft_seg_native[win_mask]
        sources_disp = sources_seg[win_mask]

        # --- Score each ICA component on the full 20 s window; pick best_k ONCE ---
        best_k = 0
        best_n_correct_seg = -1
        for k in range(sources_disp.shape[1]):
            src_filt_k = cheby1_bandpass_filter(
                sources_disp[:, k], fs, cfg.fetal_detection_band_hz, order=3, rp=1.0
            )
            beats_k     = detect_fetal_waveform_peaks(ft_seg_disp, src_filt_k, cfg.fetal_hr_bpm)
            src_times_k = np.asarray(beats_k["times"])
            if len(src_times_k) == 0 or len(mic_peak_times_full) == 0:
                continue
            ld_k = impulse_xcorr_details(
                mic_peak_times_full, src_times_k,
                float(seg_start), float(seg_end), cfg.mic_lag_bounds,
                target_fs=200.0, smooth_sigma_s=0.08,
            )
            lag_k = float(ld_k["lag_s"]) if np.isfinite(ld_k["lag_s"]) else 0.0
            m_k   = robust_match_peaks(src_times_k, mic_peak_times_full - lag_k,
                                       max_abs_dt=0.25, outlier_k=3.5)
            dt_k  = np.asarray(m_k["dt_signed"], dtype=float)
            acc_k = np.asarray(m_k["accepted_mask"], dtype=bool)
            n_corr_k = int(np.sum(np.abs(dt_k[acc_k]) <= 0.05))
            if n_corr_k > best_n_correct_seg:
                best_n_correct_seg = n_corr_k
                best_k = k
        best_channel = f"ica_{best_k}"

        # --- Inner loop: per 5 s sub-chunk using the fixed component best_k ---
        for start, end in chunk_bounds((seg_start, seg_end), cfg.chunk_seconds):
            chunk_id += 1

            # Mic peaks for this chunk
            mic_mask = chunk_mask(np.asarray(mic["t"]), start, end)
            if not np.any(mic_mask):
                continue
            mic_t = np.asarray(mic["t"])[mic_mask]
            mic_raw = np.asarray(mic["x"])[mic_mask]
            chunk_target = int(round((end - start) * 100.0 / 60.0))
            mic_beats_result = detect_microphone_s1_peaks(mic_t, mic_raw, target_count=chunk_target)
            mic_peak_times = refine_peak_times_to_waveform(
                mic_t, mic_raw, np.asarray(mic_beats_result["times"]), radius_s=0.015, positive_only=True
            )

            # Slice the pre-selected component to this chunk's native fiber time range
            chunk_start_native = start - mic_lag_s
            chunk_end_native   = end   - mic_lag_s
            src_mask = (ft_seg_native >= chunk_start_native) & (ft_seg_native < chunk_end_native)
            if not np.any(src_mask):
                continue
            ft_chunk_native = ft_seg_native[src_mask]
            comp_chunk = sources_seg[src_mask, best_k]  # fixed component
            source = cheby1_bandpass_filter(comp_chunk, fs, cfg.fetal_detection_band_hz, order=3, rp=1.0)

            source_beats = detect_fetal_waveform_peaks(
                ft_chunk_native, source, cfg.fetal_hr_bpm, target_count=len(mic_peak_times)
            )
            source_peak_times_native = refine_peak_times_to_waveform(
                ft_chunk_native, source, np.asarray(source_beats["times"]), radius_s=0.015, positive_only=True
            )

            t0 = float(
                min(np.min(source_peak_times_native) if len(source_peak_times_native) else start - cfg.mic_lag_bounds[1],
                    np.min(mic_peak_times) if len(mic_peak_times) else start)
            )
            t1 = float(
                max(np.max(mic_peak_times) if len(mic_peak_times) else end,
                    np.max(source_peak_times_native) if len(source_peak_times_native) else end)
            )
            lag_detail = impulse_xcorr_details(
                mic_peak_times, source_peak_times_native,
                t0, t1, cfg.mic_lag_bounds, target_fs=200.0, smooth_sigma_s=0.08,
            )
            lag_s = float(lag_detail["lag_s"]) if np.isfinite(lag_detail["lag_s"]) else 0.0
            matched = robust_match_peaks(
                source_peak_times_native, mic_peak_times - lag_s, max_abs_dt=0.25, outlier_k=3.5
            )

            dt = np.asarray(matched["dt_signed"], dtype=float)
            accepted = np.asarray(matched["accepted_mask"], dtype=bool)
            dt_acc = dt[accepted]
            abs_dt_acc = np.abs(dt_acc)
            n_correct = int(np.sum(abs_dt_acc <= 0.05))
            accuracy = float(n_correct / len(mic_peak_times)) if len(mic_peak_times) else np.nan
            pooled_total_mic_peaks += int(len(mic_peak_times))
            pooled_total_extracted_peaks += int(len(source_peak_times_native))
            pooled_correct_50ms += n_correct
            pooled_dt_acc.extend(dt_acc.tolist())

            summary_rows.append(
                {
                    "chunk_id": chunk_id,
                    "chunk_start_s": float(start),
                    "chunk_end_s": float(end),
                    "selected_channel": best_channel,
                    "mic_peak_count": int(len(mic_peak_times)),
                    "extracted_peak_count": int(len(source_peak_times_native)),
                    "xcorr_lag_s": lag_s,
                    "xcorr_score": float(lag_detail["score"]),
                    "accepted_pairs": int(np.sum(accepted)),
                    "mean_dt_s": float(np.mean(dt_acc)) if dt_acc.size else np.nan,
                    "std_dt_s": float(np.std(dt_acc)) if dt_acc.size else np.nan,
                    "mean_abs_dt_s": float(np.mean(abs_dt_acc)) if abs_dt_acc.size else np.nan,
                    "std_abs_dt_s": float(np.std(abs_dt_acc)) if abs_dt_acc.size else np.nan,
                    "median_abs_dt_s": float(np.median(abs_dt_acc)) if abs_dt_acc.size else np.nan,
                    "accuracy_50ms": accuracy,
                }
            )
            for i in range(len(dt)):
                match_rows.append(
                    {
                        "chunk_id": chunk_id,
                        "chunk_start_s": float(start),
                        "chunk_end_s": float(end),
                        "selected_channel": best_channel,
                        "extracted_time_s": float(np.asarray(matched["matched_ref_t"])[i]),
                        "mic_time_shifted_s": float(np.asarray(matched["matched_pred_t"])[i]),
                        "dt_s": float(dt[i]),
                        "abs_dt_s": float(abs(dt[i])),
                        "accepted": bool(accepted[i]),
                        "correct_50ms": bool(abs(dt[i]) <= 0.05 and accepted[i]),
                    }
                )
            chunk_records.append(
                {
                    "start": float(start),
                    "end": float(end),
                    "mic_times": mic_peak_times,
                    "src_times_native": source_peak_times_native,
                    "mic_times_shifted": mic_peak_times - lag_s,
                }
            )

    plot_combined_impulse_trains(
        chunk_records,
        cfg.windows[0],
        cfg.output_dir / f"stats_impulses_full_{window_tag(cfg.windows[0])}.png",
    )
    pooled_dt_acc_arr = np.asarray(pooled_dt_acc, dtype=float)
    summary_rows.append(
        {
            "chunk_id": "pooled",
            "chunk_start_s": float(cfg.windows[0][0]),
            "chunk_end_s": float(cfg.windows[0][1]),
            "selected_channel": "ALL",
            "mic_peak_count": int(pooled_total_mic_peaks),
            "extracted_peak_count": int(pooled_total_extracted_peaks),
            "xcorr_lag_s": np.nan,
            "xcorr_score": np.nan,
            "accepted_pairs": int(len(pooled_dt_acc_arr)),
            "mean_dt_s": float(np.mean(pooled_dt_acc_arr)) if pooled_dt_acc_arr.size else np.nan,
            "std_dt_s": float(np.std(pooled_dt_acc_arr)) if pooled_dt_acc_arr.size else np.nan,
            "mean_abs_dt_s": float(np.mean(np.abs(pooled_dt_acc_arr))) if pooled_dt_acc_arr.size else np.nan,
            "std_abs_dt_s": float(np.std(np.abs(pooled_dt_acc_arr))) if pooled_dt_acc_arr.size else np.nan,
            "median_abs_dt_s": float(np.median(np.abs(pooled_dt_acc_arr))) if pooled_dt_acc_arr.size else np.nan,
            "accuracy_50ms": float(pooled_correct_50ms / pooled_total_mic_peaks) if pooled_total_mic_peaks else np.nan,
        }
    )
    return pd.DataFrame(summary_rows), pd.DataFrame(match_rows)


def save_summary(
    beat_df: pd.DataFrame,
    chunk_df: pd.DataFrame,
    cfg: Config,
    ppg_col: int,
    ppg_lag: Dict[str, float],
    mic_lag: Dict[str, float],
) -> None:
    metrics = {
        "dataset_dir": str(cfg.dataset_dir),
        "ppg_column": int(ppg_col),
        "ppg_to_fiber_lag_s": float(ppg_lag["lag_s"]),
        "ppg_to_fiber_score": float(ppg_lag["score"]),
        "fiber_to_microphone_lag_s": float(mic_lag["lag_s"]),
        "fiber_to_microphone_score": float(mic_lag["score"]),
        "num_chunks": int(len(chunk_df)),
        "mean_chunk_confidence": float(chunk_df["chunk_confidence"].mean()),
        "median_maternal_match_f1": float(chunk_df["maternal_match_f1"].median()),
        "median_fetal_match_f1": float(chunk_df["fetal_match_f1"].median()),
    }
    with open(cfg.output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    best_channels = (
        chunk_df.groupby("selected_fetal_channel")
        .agg(chunks=("chunk_id", "count"), median_fetal_f1=("fetal_match_f1", "median"))
        .reset_index()
        .sort_values(["chunks", "median_fetal_f1"], ascending=[False, False])
    )
    best_channels.to_csv(cfg.output_dir / "best_channels.csv", index=False)


# ---------------------------------------------------------------------------
# Grid-search helpers
# ---------------------------------------------------------------------------

def fft_candidate_wide_bands(
    fiber_a: np.ndarray,
    fiber_b: np.ndarray,
    fs: float,
    fetal_lo: float = 80.0,
    fetal_hi: float = 250.0,
) -> List[Tuple[float, float]]:
    """Return 2-3 candidate wide bandpass ranges derived from the combined PSD of a fiber pair.

    Always includes (5.0, 220.0) as a reference band, then adds 1-2 FFT-driven candidates.
    """
    nperseg = min(len(fiber_a), max(256, int(fs * 4.0)))
    _, psd_a = signal.welch(fiber_a, fs, nperseg=nperseg)
    freqs, psd_b = signal.welch(fiber_b, fs, nperseg=nperseg)
    psd = (psd_a + psd_b) / 2.0
    mask = (freqs >= fetal_lo) & (freqs <= fetal_hi)
    bands: List[Tuple[float, float]] = [(5.0, 220.0)]  # always include reference band
    if np.any(mask):
        sub = psd[mask]
        sub_f = freqs[mask]
        # power-weighted centroid
        centroid = float(np.sum(sub_f * sub) / (np.sum(sub) + 1e-12))
        # 5th / 95th percentile frequencies by cumulative power
        cs = np.cumsum(sub) / (np.sum(sub) + 1e-12)
        f_lo = float(sub_f[np.searchsorted(cs, 0.05)])
        f_hi = float(sub_f[min(len(sub_f) - 1, np.searchsorted(cs, 0.95, side="right") - 1)])
        # Wide band around 5-95% spectral content
        b_lo = max(5.0, f_lo - 20.0)
        b_hi = min(280.0, f_hi + 20.0)
        candidate = (round(b_lo, 1), round(b_hi, 1))
        if candidate not in bands:
            bands.append(candidate)
        # Tight band ±50 Hz around power centroid
        c_lo = max(40.0, centroid - 50.0)
        c_hi = min(280.0, centroid + 50.0)
        candidate2 = (round(c_lo, 1), round(c_hi, 1))
        if candidate2 not in bands and abs(c_hi - c_lo) >= 30.0:
            bands.append(candidate2)
    return bands


def precompute_full_window_ica(
    fibers: Dict[str, object],
    mic_lag_s: float,
    pair_idx: Tuple[int, int],
    wide_band: Tuple[float, float],
    cfg: Config,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run ICA once on the full analysis window for a given (fiber pair, wide band).

    Returns
    -------
    ft_native : np.ndarray, shape (T,)
        Fiber timestamps (native, no mic_lag offset).
    ica_sources : np.ndarray, shape (T, 2)
        Two ICA components.
    """
    win_start, win_end = cfg.windows[0]
    fiber_start = win_start - mic_lag_s - 1.0
    fiber_end = win_end - mic_lag_s + 1.0
    ft_full = np.asarray(fibers["t"], float)
    mask = (ft_full >= fiber_start) & (ft_full < fiber_end)
    ft = ft_full[mask]
    fs = float(fibers["fs"])
    abdomen = fibers["abdomen"]
    a = np.asarray(abdomen[pair_idx[0]], float)[mask]
    b = np.asarray(abdomen[pair_idx[1]], float)[mask]

    a_filt = cheby1_bandpass_filter(a, fs, wide_band, order=2, rp=1.0)
    b_filt = cheby1_bandpass_filter(b, fs, wide_band, order=2, rp=1.0)
    X = np.column_stack([a_filt, b_filt])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        try:
            ica = FastICA(n_components=2, random_state=42, max_iter=cfg.ica_max_iter, tol=cfg.ica_tol)
            sources = ica.fit_transform(X)
        except Exception:
            sources = X.copy()
    return ft, sources


def evaluate_combo_pooled_accuracy(
    mic: Dict[str, object],
    mic_lag_s: float,
    ica_sources: np.ndarray,
    fiber_t_full: np.ndarray,
    detection_band: Tuple[float, float],
    window_size: float,
    cfg: Config,
) -> float:
    """Compute pooled accuracy@50ms for one (detection_band, window_size) combo.

    Uses pre-computed full-window ICA sources to avoid repeated ICA runs.
    Calls manual_shannon_peak_detection, impulse_xcorr_details, and robust_match_peaks
    directly — none of those functions are modified.
    """
    win_start, win_end = cfg.windows[0]
    pooled_correct = 0
    pooled_mic = 0

    for chunk_start, chunk_end in chunk_bounds((win_start, win_end), window_size):
        # Mic peaks for this chunk
        mic_t_arr = np.asarray(mic["t"])
        mic_x_arr = np.asarray(mic["x"])
        mic_mask = (mic_t_arr >= chunk_start) & (mic_t_arr < chunk_end)
        if not np.any(mic_mask):
            continue
        mic_t_chunk = mic_t_arr[mic_mask]
        mic_x_chunk = mic_x_arr[mic_mask]
        mic_peaks_result = detect_microphone_s1_peaks(mic_t_chunk, mic_x_chunk)
        mic_peak_times = np.asarray(mic_peaks_result["times"])
        if len(mic_peak_times) == 0:
            continue
        pooled_mic += len(mic_peak_times)

        # ICA source slice (native fiber timestamps, no mic_lag offset)
        fiber_start_native = chunk_start - mic_lag_s
        fiber_end_native = chunk_end - mic_lag_s
        src_mask = (fiber_t_full >= fiber_start_native) & (fiber_t_full < fiber_end_native)
        if not np.any(src_mask):
            continue
        ft_native = fiber_t_full[src_mask]

        best_correct = 0
        for k in range(ica_sources.shape[1]):
            src_k = ica_sources[src_mask, k]
            # Direct call with sweep band — manual_shannon_peak_detection is unchanged
            peaks_result = manual_shannon_peak_detection(
                ft_native,
                src_k,
                band_hz=detection_band,
                min_interval_s=0.27,
                threshold_factor=0.40,
                order=3,
                rp=1.0,
            )
            src_peak_times_native = np.asarray(peaks_result["times"])
            if len(src_peak_times_native) == 0:
                continue

            # Align to mic time by adding mic_lag_s
            src_peak_times_aligned = src_peak_times_native + mic_lag_s

            # xcorr lag estimate — impulse_xcorr_details is unchanged
            t0 = float(
                min(
                    src_peak_times_aligned.min() if len(src_peak_times_aligned) else chunk_start,
                    mic_peak_times.min() if len(mic_peak_times) else chunk_start,
                )
            )
            t1 = float(
                max(
                    src_peak_times_aligned.max() if len(src_peak_times_aligned) else chunk_end,
                    mic_peak_times.max() if len(mic_peak_times) else chunk_end,
                )
            )
            lag_detail = impulse_xcorr_details(
                mic_peak_times,
                src_peak_times_aligned,
                t0,
                t1,
                cfg.mic_lag_bounds,
                target_fs=200.0,
                smooth_sigma_s=0.08,
            )
            lag_s = float(lag_detail["lag_s"]) if np.isfinite(lag_detail["lag_s"]) else 0.0

            # robust match — robust_match_peaks is unchanged
            matched = robust_match_peaks(
                src_peak_times_aligned,
                mic_peak_times - lag_s,
                max_abs_dt=0.25,
                outlier_k=3.5,
            )
            dt = np.asarray(matched["dt_signed"], dtype=float)
            accepted = np.asarray(matched["accepted_mask"], dtype=bool)
            n_correct = int(np.sum(np.abs(dt[accepted]) <= 0.05))
            if n_correct > best_correct:
                best_correct = n_correct

        pooled_correct += best_correct

    if pooled_mic == 0:
        return 0.0
    return float(pooled_correct / pooled_mic)


def run_grid_search(
    ppg: Dict[str, object],
    fibers: Dict[str, object],
    mic: Dict[str, object],
    ppg_lag_s: float,
    mic_lag_s: float,
    cfg: Config,
) -> pd.DataFrame:
    """Sweep all (fiber pair × wide band × detection band × window size) combos.

    For each (fiber pair, wide band), ICA is run once on the full analysis window.
    Then (detection band, window size) combos reuse the cached ICA sources.

    Returns a DataFrame sorted by pooled_accuracy descending.
    """
    fiber_label = {0: "1B", 1: "2A", 2: "2B", 3: "2C", 4: "2D"}
    fiber_pairs = list(itertools.combinations(range(5), 2))   # 10 pairs
    detection_bands = [(lo, lo + 30) for lo in range(80, 221, 10)]   # 15 bands
    window_sizes = [5.0, 10.0, 20.0]

    # Pre-compute FFT-based wide-band candidates for each fiber pair
    win_start, win_end = cfg.windows[0]
    ft_all = np.asarray(fibers["t"], float)
    fs = float(fibers["fs"])
    fiber_mask_full = (ft_all >= win_start - mic_lag_s - 1.0) & (ft_all < win_end - mic_lag_s + 1.0)

    pair_wide_bands: Dict[Tuple[int, int], List[Tuple[float, float]]] = {}
    for pair in fiber_pairs:
        a = np.asarray(fibers["abdomen"][pair[0]], float)[fiber_mask_full]
        b = np.asarray(fibers["abdomen"][pair[1]], float)[fiber_mask_full]
        pair_wide_bands[pair] = fft_candidate_wide_bands(a, b, fs)

    total_ica_configs = sum(len(pair_wide_bands[p]) for p in fiber_pairs)
    print(f"  Grid search: {total_ica_configs} ICA configs × {len(detection_bands)} det-bands"
          f" × {len(window_sizes)} win-sizes = {total_ica_configs * len(detection_bands) * len(window_sizes)} combos")

    rows = []
    done = 0
    for pair in fiber_pairs:
        pair_label = f"({fiber_label[pair[0]]},{fiber_label[pair[1]]})"
        for wide_band in pair_wide_bands[pair]:
            done += 1
            print(
                f"  [ICA {done:2d}/{total_ica_configs}] pair={pair_label}"
                f" wide=({wide_band[0]:.0f},{wide_band[1]:.0f}) Hz ...",
                flush=True,
            )
            ft_native, ica_sources = precompute_full_window_ica(
                fibers, mic_lag_s, pair, wide_band, cfg
            )
            for det_band in detection_bands:
                for win_size in window_sizes:
                    acc = evaluate_combo_pooled_accuracy(
                        mic, mic_lag_s,
                        ica_sources, ft_native,
                        det_band, win_size, cfg,
                    )
                    rows.append(
                        {
                            "pair": pair_label,
                            "pair_a": pair[0],
                            "pair_b": pair[1],
                            "wide_band_lo": float(wide_band[0]),
                            "wide_band_hi": float(wide_band[1]),
                            "det_band_lo": float(det_band[0]),
                            "det_band_hi": float(det_band[1]),
                            "window_size_s": float(win_size),
                            "pooled_accuracy": float(acc),
                        }
                    )

    df = pd.DataFrame(rows).sort_values("pooled_accuracy", ascending=False).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Beat-to-beat maternal/fetal extraction for patient11_1-like datasets.")
    parser.add_argument("--dataset-dir", default=str(DATASET_DIR), help="Directory containing microphone.wav, ps4000.npy, ps3000a.npy, and pvs.npy")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="Directory where CSV/JSON/PNG outputs will be saved")
    parser.add_argument("--window-start", type=float, default=None, help="Override analysis window start time (s)")
    parser.add_argument("--window-end", type=float, default=None, help="Override analysis window end time (s)")
    parser.add_argument("--chunk-seconds", type=float, default=None, help="Override chunk/window size (s)")
    parser.add_argument("--display-start", type=float, default=None, help="Override display window start time (s)")
    parser.add_argument("--display-end", type=float, default=None, help="Override display window end time (s)")
    parser.add_argument("--pair-idx-a", type=int, default=None, help="First belly-fiber index within [1B, 2A, 2B, 2C, 2D]")
    parser.add_argument("--pair-idx-b", type=int, default=None, help="Second belly-fiber index within [1B, 2A, 2B, 2C, 2D]")
    parser.add_argument("--ica-seconds", type=float, default=None, help="ICA window size (s); enables two-level mode: ICA on this segment, xcorr/stats on --xcorr-window sub-chunks")
    parser.add_argument("--xcorr-window", type=float, default=None, help="xcorr evaluation window (s); controls chunk size, lag range (±window/2), and stats granularity (default: 5.0)")
    parser.add_argument("--source-prep-band-lo",  type=float, default=None, help="Wide-band low cutoff Hz applied to fibers before ICA (default: 40.0)")
    parser.add_argument("--source-prep-band-hi",  type=float, default=None, help="Wide-band high cutoff Hz applied to fibers before ICA (default: 200.0)")
    parser.add_argument("--fetal-detect-band-lo", type=float, default=None, help="Fetal detection band low cutoff Hz after ICA (default: 190.0)")
    parser.add_argument("--fetal-detect-band-hi", type=float, default=None, help="Fetal detection band high cutoff Hz after ICA (default: 220.0)")
    parser.add_argument(
        "--grid-search",
        action="store_true",
        help=(
            "Run parameter grid search over all fiber pairs, wide bands (FFT-derived), "
            "detection bands (80–250 Hz, 30 Hz wide, 10 Hz step), and window sizes (5/10/20 s). "
            "Reports best combo by pooled accuracy@50ms and saves grid_search_results.csv."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config(dataset_dir=Path(args.dataset_dir), output_dir=Path(args.output_dir))
    if args.window_start is not None and args.window_end is not None:
        cfg.windows = ((float(args.window_start), float(args.window_end)),)
    if args.chunk_seconds is not None:
        cfg.chunk_seconds = float(args.chunk_seconds)
    if args.pair_idx_a is not None and args.pair_idx_b is not None:
        cfg.selected_pair_idx = (int(args.pair_idx_a), int(args.pair_idx_b))
    if args.ica_seconds is not None:
        cfg.ica_seconds = float(args.ica_seconds)
    if args.xcorr_window is not None:
        cfg.xcorr_window_s = float(args.xcorr_window)
    # chunk_seconds drives stats granularity — always match xcorr_window_s for coherent reporting
    cfg.chunk_seconds = cfg.xcorr_window_s
    if args.source_prep_band_lo is not None or args.source_prep_band_hi is not None:
        lo = args.source_prep_band_lo if args.source_prep_band_lo is not None else cfg.source_prep_band_hz[0]
        hi = args.source_prep_band_hi if args.source_prep_band_hi is not None else cfg.source_prep_band_hz[1]
        cfg.source_prep_band_hz = (float(lo), float(hi))
    if args.fetal_detect_band_lo is not None or args.fetal_detect_band_hi is not None:
        lo = args.fetal_detect_band_lo if args.fetal_detect_band_lo is not None else cfg.fetal_detection_band_hz[0]
        hi = args.fetal_detect_band_hi if args.fetal_detect_band_hi is not None else cfg.fetal_detection_band_hz[1]
        cfg.fetal_detection_band_hz = (float(lo), float(hi))
    if args.display_start is not None and args.display_end is not None:
        cfg.display_window = (float(args.display_start), float(args.display_end))
    else:
        cfg.display_window = cfg.windows[0]
    start, end = cfg.windows[0]
    cfg.waveform_plot_windows = tuple(
        (float(ws), float(min(ws + 20.0, end)))
        for ws in np.arange(start, end, 20.0)
    )
    ppg_mid = start + min(15.0, max(0.0, (end - start) / 2.0))
    cfg.ppg_lag_window = (ppg_mid, min(ppg_mid + 5.0, end))
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    data = slice_data_for_focus(load_data(cfg), cfg)

    ppg_col = fixed_ppg_channel()
    ppg = prepare_ppg(data["pvs"], ppg_col)
    fibers = prepare_fibers(data["ps4000"], data["ps3000a"], cfg)
    mic = prepare_microphone(data["mic_t"], data["mic"], cfg)

    ppg_lag = estimate_ppg_fiber_lag(ppg, fibers, cfg)

    # -----------------------------------------------------------------------
    # Grid search mode — run optimization, report best combo, then exit
    # -----------------------------------------------------------------------
    if args.grid_search:
        print(f"\n=== Parameter grid search (200–500 s window) ===")
        print(f"Fiber->mic lag (pre-estimated): {mic_lag['lag_s']:.3f} s")
        grid_df = run_grid_search(ppg, fibers, mic, ppg_lag["lag_s"], mic_lag["lag_s"], cfg)
        grid_path = cfg.output_dir / "grid_search_results.csv"
        grid_df.to_csv(grid_path, index=False)
        print(f"\nGrid search complete. Full results → {grid_path}")
        print(f"\nTop 10 combos (0=1B, 1=2A, 2=2B, 3=2C, 4=2D):")
        pd.set_option("display.width", 120)
        pd.set_option("display.float_format", "{:.4f}".format)
        print(grid_df.head(10).to_string(index=False))
        best = grid_df.iloc[0]
        fiber_label = {0: "1B", 1: "2A", 2: "2B", 3: "2C", 4: "2D"}
        print(f"\nBest combo:")
        print(f"  Fiber pair:     ({fiber_label[int(best['pair_a'])]}, {fiber_label[int(best['pair_b'])]})")
        print(f"  Wide band:      ({best['wide_band_lo']:.1f}, {best['wide_band_hi']:.1f}) Hz")
        print(f"  Detection band: ({best['det_band_lo']:.0f}, {best['det_band_hi']:.0f}) Hz")
        print(f"  Window size:    {best['window_size_s']:.0f} s")
        print(f"  Pooled acc@50ms:{best['pooled_accuracy']:.4f}")
        return

    # -----------------------------------------------------------------------
    # Normal analysis pipeline
    # -----------------------------------------------------------------------
    main_tag    = window_tag(cfg.windows[0])
    display_tag = window_tag(cfg.display_window)
    use_two_level = cfg.ica_seconds is not None

    # Build ICA cache (20 s segments, lag=0) — shared by xcorr, stats, and waveform plots.
    ica_cache = build_ica_segment_cache(fibers, 0.0, cfg, cfg.ica_seconds) if use_two_level else None
    mic_lag_s = 0.0
    if use_two_level and ica_cache:
        mic_lag_s = estimate_lag_via_local_xcorr(fibers, mic, ica_cache, cfg, cfg.xcorr_window_s)
        print(f"Local xcorr median lag: {mic_lag_s:.3f} s")
    mic_lag = {"lag_s": mic_lag_s, "score": 0.0}

    display = build_display_results(ppg, fibers, mic, ppg_lag["lag_s"], mic_lag["lag_s"], cfg)

    plot_raw_signals(display, ppg_lag["lag_s"], mic_lag["lag_s"], cfg.output_dir / f"raw_signals_{display_tag}.png")
    plot_maternal_alignment(display, cfg.output_dir / f"maternal_alignment_{display_tag}.png")
    plot_fetal_alignment(display, cfg.output_dir / f"fetal_alignment_{display_tag}.png")
    plot_fetal_source_overlay(display, cfg.output_dir / f"fetal_source_overlay_{display_tag}.png")
    for window in cfg.waveform_plot_windows:
        plot_shifted_waveform_window(
            window,
            fibers,
            mic,
            mic_lag["lag_s"],
            ica_cache if ica_cache is not None else {},
            cfg,
            cfg.output_dir / f"shifted_waveforms_{int(window[0])}_{int(window[1])}.png",
        )
    if use_two_level:
        print(f"Two-level mode: ICA={cfg.ica_seconds:.0f}s segments, xcorr/stats={cfg.xcorr_window_s:.0f}s chunks")
        stats_df, match_df = analyze_fetal_statistics_two_level(
            ppg, fibers, mic, ppg_lag["lag_s"], mic_lag["lag_s"], cfg, cfg.ica_seconds,
            ica_seg_cache=ica_cache,
        )
    else:
        stats_df, match_df = analyze_fetal_statistics(ppg, fibers, mic, ppg_lag["lag_s"], mic_lag["lag_s"], cfg)
    stats_df.to_csv(cfg.output_dir / f"fetal_chunk_statistics_{main_tag}.csv", index=False)
    match_df.to_csv(cfg.output_dir / f"fetal_peak_matches_{main_tag}.csv", index=False)
    if use_two_level and ica_cache:
        plot_xcorr_curve(
            fibers, mic, ica_cache, cfg,
            cfg.output_dir / f"xcorr_curve_{main_tag}.png",
            cfg.output_dir / f"xcorr_lags_{main_tag}.csv",
            xcorr_window_s=cfg.xcorr_window_s,
        )
    save_focused_summary(cfg, ppg_lag, mic_lag, display)

    print(f"Saved outputs to {cfg.output_dir}")
    print(f"PPG column: {ppg_col}")
    print(f"PPG->fiber lag: {ppg_lag['lag_s']:.3f} s (score {ppg_lag['score']:.2f})")
    print(f"Fiber->mic lag (local xcorr median): {mic_lag['lag_s']:.3f} s")
    print(f"Source prep band:  {cfg.source_prep_band_hz} Hz")
    print(f"Fetal detect band: {cfg.fetal_detection_band_hz} Hz")
    print(f"Display maternal lag: {display['maternal_xcorr']['lag_s']:.3f} s")
    print(f"Display fetal lag: {display['fetal_xcorr']['lag_s']:.3f} s")
    if not stats_df.empty:
        chunk_only = stats_df[stats_df["chunk_id"] != "pooled"]
        print(f"Mean chunk accuracy@50ms: {chunk_only['accuracy_50ms'].mean():.3f}")
        pooled_row = stats_df[stats_df["chunk_id"] == "pooled"]
        if not pooled_row.empty:
            print(f"Pooled accuracy@50ms:     {pooled_row['accuracy_50ms'].iloc[0]:.3f}")


if __name__ == "__main__":
    main()
