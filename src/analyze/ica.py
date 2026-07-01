"""ICA-based fetal HR pipeline stages.

Ports the logic from clean_data_template.py into three cacheable stages:

  load_ica_data   → ICAInputData     (raw load + window)
  prepare_signals → PreparedSignals  (filter, detect beats, estimate PPG→fiber lag)
  run_ica         → FiberData        (FastICA + per-segment xcorr source selection)

FiberData output has a single 'fetal' abdomen channel (selected ICA component).
Feed into abdomen_bp → classify_sources → detect_beats → evaluate downstream.

Pair selection matches clean_data_template.py: pair_idx=(0,1) uses
(ps4000[:,2], ps3000a[:,1]) — the ps4000 belly fiber + ps3000a channel 2A.
"""

import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import numpy.typing as npt
from scipy.io.wavfile import read as wav_read
from scipy.signal import detrend, correlate, correlation_lags, hilbert
from sklearn.decomposition import FastICA
from sklearn.exceptions import ConvergenceWarning

from analyze.data import Audio, FiberData
from analyze.evaluate import _gaussian_smooth, _impulse_train
from analyze.hr.detect import detect_maternal_beats
from analyze.filters import bp_filter
from analyze.sot import detect_ppg_beats, detect_mic_fetal_beats, _robust_clip, _suppress_transients
from analyze.util import normalize_path
from constants import (
    FIBER_BUNDLE_A, FIBER_BUNDLE_B, MIC_FILE, PVS_FILE, XCORR_TARGET_FS,
    FETAL_ACOUSTIC_BAND_HZ, SOURCE_PREP_BAND_HZ, MATERNAL_ACOUSTIC_BAND_HZ, MATERNAL_BPM_RANGE,
)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class ICAInputData:
    """Raw windowed data for all four modalities."""
    chest: Audio
    abdomen: List[Audio]        # abdomen[0]=ps4000[:,2], abdomen[1:]=ps3000a 2A–2D
    abdomen_labels: List[str]
    ppg: Audio                  # raw PPG, detrended
    mic_raw: Audio              # mic, detrended + robust-clipped
    mic_hs: Audio               # mic bandpassed at fetal_detect_band + transient-suppressed


@dataclass
class PreparedSignals:
    """Filtered signals, detected beats, and PPG→fiber lag. Ready for ICA."""
    chest: Audio                     # raw chest fiber
    chest_maternal: Audio            # chest at maternal_band (40–80 Hz)
    chest_beats: npt.NDArray[np.float64]   # maternal beat times from chest
    abdomen_prepped: List[Audio]     # each abdomen channel at source_prep_band
    abdomen_labels: List[str]
    fiber_fs: float
    ppg: Audio                       # butter 0.7–4 Hz filtered PPG
    ppg_beats: npt.NDArray[np.float64]
    ppg_lag_s: float                 # PPG→fiber lag (seconds)
    mic_raw: Audio
    mic_hs: Audio
    mic_feature: Audio               # abs(mic_hs)
    mic_template: npt.NDArray[np.float64]
    mic_beats: npt.NDArray[np.float64]   # fetal beat times from mic (SOT)


# ---------------------------------------------------------------------------
# Local helpers — ports of clean_data_template.py
# ---------------------------------------------------------------------------

def _normalize(x: np.ndarray) -> np.ndarray:
    s = np.std(x)
    return np.zeros_like(x) if s < 1e-12 else (x - np.mean(x)) / s


def _manual_shannon_peaks(
    t: np.ndarray,
    waveform: np.ndarray,
    band_hz: Tuple[float, float],
    min_interval_s: float,
    threshold_factor: float = 0.40,
    order: int = 3,
) -> npt.NDArray[np.float64]:
    """Return beat times via Shannon-energy peak detection.

    Port of manual_shannon_peak_detection (clean_data_template.py:292).
    Applies an internal cheby1 bandpass at band_hz (matching the template,
    which double-filters by also pre-filtering before calling this).
    """
    hz = float(np.round(1.0 / np.median(np.diff(t))))
    audio = Audio(t, int(hz), np.asarray(waveform, float))
    filtered = bp_filter(audio, band_hz[0], band_hz[1], order=order, filter_type='cheby1').data

    scale = np.max(np.abs(filtered)) + 1e-12
    normalized = filtered / scale
    shannon = -(normalized ** 2) * np.log(np.clip(normalized ** 2, 1e-12, None))
    envelope = np.abs(hilbert(shannon))
    threshold = threshold_factor * (float(np.max(envelope)) + float(np.min(envelope)))
    active = envelope > threshold

    changes = np.diff(active.astype(int), prepend=0, append=0)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    min_dist = max(1, int(round(min_interval_s * hz)))
    pad = max(1, int(round(0.004 * hz)))
    positive = np.maximum(normalized, 0.0)

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
    return t[peaks] if len(peaks) else np.array([], dtype=float)


def _refine_peaks(
    t: np.ndarray,
    x: np.ndarray,
    peak_times: npt.NDArray[np.float64],
    radius_s: float = 0.015,
    positive_only: bool = True,
) -> npt.NDArray[np.float64]:
    """Snap each peak time to the nearest local waveform maximum.

    Port of refine_peak_times_to_waveform (clean_data_template.py:893).
    """
    if len(peak_times) == 0:
        return np.array([], dtype=float)
    hz = float(np.round(1.0 / np.median(np.diff(t))))
    radius = max(1, int(round(radius_s * hz)))
    w = np.maximum(x, 0.0) if positive_only else np.abs(x)
    refined = []
    for pt in peak_times:
        c = int(np.argmin(np.abs(t - pt)))
        lo = max(0, c - radius)
        hi = min(len(w), c + radius + 1)
        refined.append(float(t[lo + int(np.argmax(w[lo:hi]))]))
    return np.asarray(refined, dtype=float)


def _build_template(
    mic_hs: Audio,
    beat_times: np.ndarray,
    pre_s: float = 0.01,
    post_s: float = 0.03,
) -> np.ndarray:
    """Port of build_average_template (clean_data_template.py:856)."""
    hz = float(mic_hs.hz)
    pre = int(round(pre_s * hz))
    post = int(round(post_s * hz))
    if len(beat_times) == 0:
        return np.array([1.0], dtype=float)
    snippets = []
    for bt in beat_times:
        c = int(round((bt - mic_hs.time[0]) * hz))
        if c - pre < 0 or c + post >= len(mic_hs.data):
            continue
        seg = mic_hs.data[c - pre: c + post + 1].astype(float)
        s = np.std(seg)
        if s < 1e-12:
            continue
        snippets.append((seg - np.mean(seg)) / s)
    if not snippets:
        return np.array([1.0], dtype=float)
    t = np.mean(np.vstack(snippets), axis=0)
    s = np.std(t)
    return t / s if s > 1e-12 else t


def _xcorr_lag_pair(
    ref_times: np.ndarray,
    pred_times: np.ndarray,
    t_start: float,
    t_end: float,
    lag_bound_s: float = 3.5,
    target_fs: float = XCORR_TARGET_FS,
    sigma_s: float = 0.08,
) -> Tuple[float, float]:
    """Return (lag_s, score) from impulse-train xcorr."""
    if len(ref_times) < 2 or len(pred_times) < 2:
        return 0.0, 0.0
    t_grid = np.arange(t_start, t_end, 1.0 / target_fs)
    ref_s = _gaussian_smooth(_impulse_train(ref_times, t_grid), sigma_s * target_fs)
    pred_s = _gaussian_smooth(_impulse_train(pred_times, t_grid), sigma_s * target_fs)
    corr = correlate(ref_s, pred_s, mode='full')
    lags = correlation_lags(len(ref_s), len(pred_s), mode='full') / target_fs
    mask = (lags >= -lag_bound_s) & (lags <= lag_bound_s)
    if not np.any(mask):
        return 0.0, 0.0
    sub, sub_lags = corr[mask], lags[mask]
    idx = int(np.argmax(sub))
    return float(sub_lags[idx]), float(sub[idx] / (np.std(sub) + 1e-9))


# ---------------------------------------------------------------------------
# Stage 1: load_ica_data
# ---------------------------------------------------------------------------

def load_ica_data(
    window_start: float,
    window_end: float,
    ppg_col: int = 1,
    fetal_detect_band: Tuple[float, float] = FETAL_ACOUSTIC_BAND_HZ,
):
    """Stage factory: load ps4000, ps3000a, pvs, microphone.wav and slice to window.

    abdomen[0] = ps4000[:,2] (belly fiber on same device as chest, matches
    clean_data_template.py prepare_fibers pair_idx=(0,1) → ps4000[:,2] + ps3000a[:,1]).
    abdomen[1:] = ps3000a channels 2A–2D.
    """
    def run_load_ica_data(data_dir: str) -> ICAInputData:
        path = normalize_path(data_dir)

        # --- Fibers ---
        ps4 = np.load(path + FIBER_BUNDLE_A)
        ps3 = np.load(path + FIBER_BUNDLE_B)

        t_fiber = ps4[:, 0].astype(float)
        fiber_hz = round(1.0 / float(np.median(np.diff(t_fiber))))
        fiber_mask = (t_fiber >= window_start) & (t_fiber <= window_end)
        t_fib = t_fiber[fiber_mask]

        chest = Audio(t_fib, fiber_hz, ps4[fiber_mask, 1].astype(float))

        # ps4000[:,2] is the belly fiber on the same device as the chest.
        # This matches clean_data_template.py's abdomen[0].
        ps4_belly = Audio(t_fib, fiber_hz, ps4[fiber_mask, 2].astype(float))

        t_abd = ps3[:, 0].astype(float)
        abd_hz = round(1.0 / float(np.median(np.diff(t_abd))))
        abd_mask = (t_abd >= window_start) & (t_abd <= window_end)
        t_a = t_abd[abd_mask]

        abdomen_labels = ["ps4_belly", "2A", "2B", "2C", "2D"]
        abdomen = [ps4_belly] + [
            Audio(t_a, abd_hz, ps3[abd_mask, col].astype(float))
            for col in range(1, min(5, ps3.shape[1]))
        ]

        # --- PPG ---
        pvs = np.load(path + PVS_FILE)
        t_ppg = pvs[:, 0].astype(float)
        ppg_mask = (t_ppg >= window_start) & (t_ppg <= window_end)
        t_ppg = t_ppg[ppg_mask]
        hz_ppg = round(1.0 / float(np.median(np.diff(t_ppg))))
        ppg = Audio(t_ppg, hz_ppg, detrend(pvs[ppg_mask, ppg_col].astype(float)))

        # --- Microphone ---
        mic_fs, mic_arr = wav_read(path + MIC_FILE)
        mic_arr = mic_arr.astype(float)
        if mic_arr.ndim > 1:
            mic_arr = mic_arr[:, 0]
        t_mic = np.arange(len(mic_arr)) / float(mic_fs)
        mic_mask = (t_mic >= window_start) & (t_mic <= window_end)
        t_mic = t_mic[mic_mask]
        mic_arr = _robust_clip(detrend(mic_arr[mic_mask]))
        mic_raw = Audio(t_mic, mic_fs, mic_arr)

        mic_hs_filt = bp_filter(mic_raw, fetal_detect_band[0], fetal_detect_band[1], filter_type='cheby1')
        mic_hs = Audio(t_mic, mic_fs,
                       _suppress_transients(mic_hs_filt.data, float(mic_fs), window_s=0.04))

        print(f"  Loaded: fiber {len(t_fib)} pts @ {fiber_hz} Hz | "
              f"abd {len(t_a)} pts @ {abd_hz} Hz | "
              f"mic {len(t_mic)} pts @ {mic_fs} Hz | "
              f"ppg {len(t_ppg)} pts @ {hz_ppg} Hz")

        return ICAInputData(
            chest=chest,
            abdomen=abdomen,
            abdomen_labels=abdomen_labels,
            ppg=ppg,
            mic_raw=mic_raw,
            mic_hs=mic_hs,
        )

    run_load_ica_data.__name__ = "load_ica_data"
    return run_load_ica_data


# ---------------------------------------------------------------------------
# Stage 2: prepare_signals
# ---------------------------------------------------------------------------

def prepare_signals(
    source_prep_band: Tuple[float, float] = SOURCE_PREP_BAND_HZ,
    maternal_band: Tuple[float, float] = MATERNAL_ACOUSTIC_BAND_HZ,
    maternal_bpm_range: Tuple[float, float] = MATERNAL_BPM_RANGE,
):
    """Stage factory: filter signals, detect beats, estimate PPG→fiber lag.

    Port of prepare_ppg + prepare_fibers + prepare_microphone (beat detection part)
    + estimate_ppg_fiber_lag from clean_data_template.py.
    """
    def run_prepare_signals(data: ICAInputData) -> PreparedSignals:
        # Chest: maternal band + beat detection
        chest_maternal = bp_filter(data.chest, maternal_band[0], maternal_band[1], filter_type='cheby1')
        chest_beats = detect_maternal_beats(data.chest, maternal_band, maternal_bpm_range)["times"]

        # Abdomen: prep band for ICA (order=2 matches template's cheby1_bandpass_filter call)
        abdomen_prepped = [
            bp_filter(a, source_prep_band[0], source_prep_band[1],
                      order=2, filter_type='cheby1')
            for a in data.abdomen
        ]

        # PPG: butter 0.7–4 Hz + beat detection
        ppg_filt = bp_filter(data.ppg, 0.7, 4.0, filter_type='butter')
        ppg_beats = detect_ppg_beats(ppg_filt)["times"]

        # PPG→fiber lag: xcorr of chest maternal beats vs PPG beats
        t_start = float(data.chest.time[0])
        t_end = float(data.chest.time[-1])
        ppg_lag_s, ppg_score = _xcorr_lag_pair(
            chest_beats, ppg_beats, t_start, t_end, lag_bound_s=3.5,
        )

        # Mic: fetal beat detection + average beat template
        mic_beats = detect_mic_fetal_beats(data.mic_raw)["times"]
        mic_template = _build_template(data.mic_hs, mic_beats)
        mic_feature = Audio(data.mic_hs.time, data.mic_hs.hz, np.abs(data.mic_hs.data))

        print(f"  Chest maternal: {len(chest_beats)} beats | "
              f"PPG maternal: {len(ppg_beats)} beats | "
              f"Mic fetal: {len(mic_beats)} beats")
        print(f"  PPG→fiber lag: {ppg_lag_s:+.3f}s  (score={ppg_score:.1f})")

        return PreparedSignals(
            chest=data.chest,
            chest_maternal=chest_maternal,
            chest_beats=chest_beats,
            abdomen_prepped=abdomen_prepped,
            abdomen_labels=data.abdomen_labels,
            fiber_fs=float(data.chest.hz),
            ppg=ppg_filt,
            ppg_beats=ppg_beats,
            ppg_lag_s=ppg_lag_s,
            mic_raw=data.mic_raw,
            mic_hs=data.mic_hs,
            mic_feature=mic_feature,
            mic_template=mic_template,
            mic_beats=mic_beats,
        )

    run_prepare_signals.__name__ = "prepare_signals"
    return run_prepare_signals


# ---------------------------------------------------------------------------
# Stage 3: run_ica
# ---------------------------------------------------------------------------

def run_ica(
    pair_idx: Tuple[int, int] = (0, 1),
    ica_seconds: float = 20.0,
    xcorr_window_s: float = 5.0,
    fetal_detect_band: Tuple[float, float] = FETAL_ACOUSTIC_BAND_HZ,
    ica_max_iter: int = 3000,
    ica_tol: float = 1e-3,
):
    """Stage factory: segmented FastICA + per-segment xcorr source selection.

    Matches clean_data_template.py _xcorr_chunk_data / build_ica_segment_cache:
      - FastICA run per ica_seconds segment on the chosen channel pair.
      - Beat detection for xcorr scoring uses manual Shannon energy with
        threshold_factor=0.15 (lower than default, for sensitivity on long segments)
        followed by refine_peak_times_to_waveform, matching the template exactly.
      - Source selection is global: median xcorr score across all 5-s chunks
        determines chosen_k once per ICA segment.
      - Returns FiberData with only the selected fetal source in abdomen.
    """
    def run_run_ica(prep: PreparedSignals) -> FiberData:
        fs = prep.fiber_fs
        target_fs = XCORR_TARGET_FS
        sigma_samp = 0.08 * target_fs
        half_w = xcorr_window_s / 2.0
        lag_bounds = (-half_w, half_w)
        min_ibi_s = 60.0 / float(fetal_detect_band[1])  # ~0.27 s at 220 BPM

        a_all = prep.abdomen_prepped[pair_idx[0]]
        b_all = prep.abdomen_prepped[pair_idx[1]]
        t_start = float(a_all.time[0])
        t_end = float(a_all.time[-1])

        # --- Mic peaks refined once over the full window (matches template) ---
        mic_beats_all = prep.mic_beats
        mic_beats_all = _refine_peaks(
            prep.mic_raw.time, prep.mic_raw.data, mic_beats_all,
            radius_s=0.015, positive_only=True,
        )

        # --- Build ICA segment cache ---
        # Port of build_ica_segment_cache (clean_data_template.py:2815).
        # Segments are non-overlapping; ±0.5 s margin on load to avoid edge artefacts.
        ica_cache: Dict[Tuple[float, float], Tuple[np.ndarray, np.ndarray]] = {}
        seg_start = t_start
        while seg_start < t_end:
            seg_end = min(seg_start + ica_seconds, t_end)
            if seg_end - seg_start < 2.0:
                break
            mask_a = (a_all.time >= seg_start - 0.5) & (a_all.time < seg_end + 0.5)
            mask_b = (b_all.time >= seg_start - 0.5) & (b_all.time < seg_end + 0.5)
            n = min(mask_a.sum(), mask_b.sum())
            if n > 0:
                t_seg = a_all.time[mask_a][:n]
                X = np.column_stack([a_all.data[mask_a][:n], b_all.data[mask_b][:n]])
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", ConvergenceWarning)
                    try:
                        ica = FastICA(n_components=2, random_state=42,
                                      max_iter=ica_max_iter, tol=ica_tol)
                        sources = ica.fit_transform(X)
                    except Exception:
                        sources = X.copy()
                ica_cache[(float(seg_start), float(seg_end))] = (t_seg, sources)
            seg_start = seg_end

        print(f"  ICA: {len(ica_cache)} segments × {ica_seconds:.0f}s")

        # --- Per-segment: detect fetal peaks on the full segment for both sources ---
        # Threshold 0.15 (vs default 0.40) gives more sensitivity on long segments,
        # preventing large transient spikes from pushing the threshold too high.
        # Matching clean_data_template.py _xcorr_chunk_data:2153–2165.
        seg_peaks: Dict[Tuple[float, float], List[npt.NDArray]] = {}
        for (seg_s, seg_e), (t_seg, sources_seg) in ica_cache.items():
            core = (t_seg >= seg_s) & (t_seg < seg_e)
            ft_s = t_seg[core]
            peaks_per_k: List[npt.NDArray] = []
            for k in range(2):
                src_filt_k = bp_filter(
                    Audio(ft_s, int(fs), np.real(sources_seg[core, k])),
                    fetal_detect_band[0], fetal_detect_band[1], filter_type='cheby1', order=3,
                ).data
                raw_times = _manual_shannon_peaks(
                    ft_s, src_filt_k, fetal_detect_band, min_ibi_s,
                    threshold_factor=0.15, order=3,
                )
                refined = _refine_peaks(ft_s, src_filt_k, raw_times,
                                        radius_s=0.015, positive_only=True)
                peaks_per_k.append(refined)
            seg_peaks[(seg_s, seg_e)] = peaks_per_k

        # --- Per xcorr chunk: score each source against mic ---
        # Matching _xcorr_chunk_data (clean_data_template.py:2091).
        scores_by_k: List[List[float]] = [[], []]

        chunk_start = t_start
        while chunk_start < t_end:
            chunk_end = min(chunk_start + xcorr_window_s, t_end)

            seg_key = next(
                (k for k in ica_cache if k[0] <= chunk_start + 0.01 and k[1] >= chunk_end - 0.01),
                None,
            )
            if seg_key is None:
                chunk_start = chunk_end
                continue

            dur = chunk_end - chunk_start
            t_local = np.arange(0.0, dur, 1.0 / target_fs)

            mic_c = mic_beats_all[(mic_beats_all >= chunk_start) & (mic_beats_all < chunk_end)]
            if len(mic_c) < 2:
                chunk_start = chunk_end
                continue

            imp_mic = _gaussian_smooth(
                _impulse_train(mic_c - chunk_start, t_local), sigma_samp
            )

            for k in range(2):
                fiber_c = seg_peaks[seg_key][k]
                fiber_c = fiber_c[(fiber_c >= chunk_start) & (fiber_c < chunk_end)]
                if len(fiber_c) < 2:
                    continue
                imp_fib = _gaussian_smooth(
                    _impulse_train(fiber_c - chunk_start, t_local), sigma_samp
                )
                corr = correlate(imp_fib, imp_mic, mode='full')
                lags = correlation_lags(len(imp_fib), len(imp_mic), mode='full') / target_fs
                m = (lags >= lag_bounds[0]) & (lags <= lag_bounds[1])
                if not np.any(m):
                    continue
                sub = corr[m]
                peak_idx = int(np.argmax(sub))
                scores_by_k[k].append(float(sub[peak_idx] / (np.std(sub) + 1e-9)))

            chunk_start = chunk_end

        med_scores = [float(np.median(s)) if s else 0.0 for s in scores_by_k]
        chosen_k = int(np.argmax(med_scores))
        print(f"  Source scores: k=0 → {med_scores[0]:.2f}  k=1 → {med_scores[1]:.2f}"
              f"  → chosen k={chosen_k}")

        # --- Stitch chosen source across the full window ---
        t_out = a_all.time
        stitched = np.zeros(len(t_out))
        for (seg_s, seg_e), (t_seg, sources_seg) in ica_cache.items():
            core_out = (t_out >= seg_s) & (t_out < seg_e)
            core_seg = (t_seg >= seg_s) & (t_seg < seg_e)
            out_idx = np.where(core_out)[0]
            seg_idx = np.where(core_seg)[0]
            n = min(len(out_idx), len(seg_idx))
            stitched[out_idx[:n]] = np.real(sources_seg[seg_idx[:n], chosen_k])

        return FiberData(
            prep.chest,
            {"fetal": Audio(t_out, int(fs), stitched)},
        )

    run_run_ica.__name__ = "run_ica"
    return run_run_ica
