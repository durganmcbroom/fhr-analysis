import os
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import numpy.typing as npt
from matplotlib import pyplot as plt
from scipy.io.wavfile import read as wav_read
from scipy.signal import detrend, find_peaks

from analyze.data import Audio
from analyze.filters import bp_filter
from analyze.util import normalize_path
from constants import FETAL_ACOUSTIC_BAND_HZ, MATERNAL_BPM_RANGE, MIC_FILE, PVS_FILE


def _window_beats(beats, start, end):
    """Restrict beat times to [start, end] and recompute IBI/BPM from them."""
    beats = np.asarray(beats, dtype=float)
    bw = beats[(beats >= start) & (beats <= end)]
    return bw


@dataclass
class SOTResult:
    ppg: Audio  # butter 0.7–4 Hz filtered PPG
    ppg_beats: npt.NDArray[np.float64]  # maternal beat times (s)

    mic: Audio  # cheby1 190–220 Hz + transient-suppressed mic
    mic_beats: npt.NDArray[np.float64]  # fetal beat times (s)

    def window(self, start, end):
        """Restrict every signal and beat array to [start, end].

        Lets the generic `windowed` stage (data.py) be reused for the SOT
        pipeline. eval_v2 pipelines skip this and pass the full SOT instead, so
        the initial-lag search has data beyond the analysis-window edges."""
        ppg_b = _window_beats(self.ppg_beats, start, end)
        mic_b = _window_beats(self.mic_beats, start, end)

        return SOTResult(
            ppg=self.ppg.window(start, end),
            ppg_beats=ppg_b,
            mic=self.mic.window(start, end),
            mic_beats=mic_b,
        )


# ---------------------------------------------------------------------------
# Small signal utilities (standalone; no shared deps with fetal_hr.py)
# ---------------------------------------------------------------------------

def _normalize(x: np.ndarray) -> np.ndarray:
    s = np.std(x)
    return np.zeros_like(x) if s < 1e-12 else (x - np.mean(x)) / s


def _moving_avg(x: np.ndarray, n: int) -> np.ndarray:
    if n <= 1:
        return x.copy()
    return np.convolve(x, np.ones(n, dtype=float) / n, mode='same')


def _robust_clip(x: np.ndarray, zmax: float = 6.0) -> np.ndarray:
    """Port of robust_clip from clean_data_template.py:101."""
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + 1e-9
    sigma = 1.4826 * mad
    return np.clip(x, med - zmax * sigma, med + zmax * sigma)


def _suppress_transients(x: np.ndarray, hz: float, window_s: float = 0.04) -> np.ndarray:
    """Local RMS-based transient suppression. Port of suppress_transients from clean_data_template.py:135."""
    n = max(1, int(round(window_s * hz)))
    local_rms = np.sqrt(np.maximum(_moving_avg(np.square(x), n), 0.0))
    med = np.median(local_rms)
    mad = np.median(np.abs(local_rms - med)) + 1e-9
    scale = med + 3.0 * (1.4826 * mad)
    gain = 1.0 / np.maximum(1.0, local_rms / max(scale, 1e-6))
    return x * gain


# ---------------------------------------------------------------------------
# Beat detection
# ---------------------------------------------------------------------------

def detect_ppg_beats(
        ppg: Audio,
        bpm_range: Tuple[float, float],
) -> dict:
    """Detect maternal HR peaks from filtered PPG.

    Port of detect_beats (clean_data_template.py:160), both polarities tried.
    """
    hz = float(ppg.hz)
    work = _normalize(ppg.data)
    min_dist = max(1, int(hz * 60.0 / bpm_range[1] * 0.75))
    prom = max(0.15, 0.35 * np.std(work))

    best_score = -np.inf
    best_peaks = np.array([], dtype=int)
    for sign in (1, -1):
        peaks, props = find_peaks(sign * work, distance=min_dist, prominence=prom)
        if len(peaks) < 2:
            continue
        ibi = np.diff(ppg.time[peaks])
        bpm_vals = 60.0 / np.clip(ibi, 1e-6, None)
        valid = (bpm_vals >= bpm_range[0] * 0.8) & (bpm_vals <= bpm_range[1] * 1.2)
        score = float(valid.mean()) - 0.5 * float(np.std(ibi))
        if len(peaks) > 2:
            dense_penalty = max(0.0, len(peaks) / (len(work) / hz / 0.25) - 1.0)
            score -= 0.2 * dense_penalty
        if score > best_score:
            best_score = score
            best_peaks = peaks

    beat_times = ppg.time[best_peaks] if len(best_peaks) else np.array([], dtype=float)
    # ibi = np.diff(beat_times) if len(beat_times) > 1 else np.array([], dtype=float)
    # bpm = 60.0 / np.clip(ibi, 1e-6, None) if len(ibi) else np.array([], dtype=float)
    return {"peaks": best_peaks, "times": beat_times}


def detect_mic_fetal_beats(
        mic_raw: Audio,
        min_interval_s: float = 0.25,  # 240 bpm cap; 0.30 (200 bpm) suppressed real beats
) -> dict:
    """Detect fetal HR peaks from microphone using adaptive local threshold.

    Port of detect_microphone_s1_peaks → detect_adaptive_positive_peaks
    (clean_data_template.py:445). Operates on the detrended+clipped signal.
    """
    # return v2_beat_detector(
    #     mic_raw,
    #     (100, 240),
    #     None,
    #     tag="fetal"
    # )
    raise Exception("not implemented")
    # hz = float(mic_raw.hz)
    # x = np.asarray(mic_raw.data, float)
    # xpos = np.maximum(x, 0.0)
    #
    # smooth_n = max(1, int(round(hz * 0.003)))
    # xs = _moving_avg(xpos, smooth_n)
    #
    # local_win = max(3, int(round(hz * 1.5)))
    # if local_win % 2 == 0:
    #     local_win += 1
    # local_max = np.maximum(maximum_filter1d(xs, size=local_win, mode='reflect'), 1e-12)
    #
    # med = float(np.median(x))
    # mad = float(np.median(np.abs(x - med))) + 1e-12
    # noise_scale = 1.4826 * mad
    # global_floor = 2.5 * noise_scale
    # threshold_arr = np.maximum(0.20 * local_max, global_floor)
    #
    # min_dist = max(1, int(round(min_interval_s * hz)))
    # peaks, props = find_peaks(xs, distance=min_dist, prominence=(0.3 * global_floor, None))
    #
    # if len(peaks) > 0:
    #     keep = xs[peaks] >= threshold_arr[peaks]
    #     peaks = peaks[keep]
    #
    # beat_times = mic_raw.time[peaks] if len(peaks) else np.array([], dtype=float)
    # ibi = np.diff(beat_times) if len(beat_times) > 1 else np.array([], dtype=float)
    # bpm = 60.0 / np.clip(ibi, 1e-6, None) if len(ibi) else np.array([], dtype=float)
    # return {"peaks": peaks, "times": beat_times, "ibi": ibi, "bpm": bpm}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_sot(result: SOTResult, out_dir: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), constrained_layout=True)

    # Instantaneous BPM recomputed from beat times on the fly (no stored bpm).
    def _bpm(beats):
        beats = np.asarray(beats, dtype=float)
        if beats.size < 2:
            return beats[1:], np.array([], dtype=float)
        return beats[1:], 60.0 / np.clip(np.diff(beats), 1e-6, None)

    ppg_t, ppg_bpm = _bpm(result.ppg_beats)
    mic_t, mic_bpm = _bpm(result.mic_beats)

    # Panel 1: PPG + maternal beats
    axes[0].plot(result.ppg.time, result.ppg.data, color='tab:blue', lw=0.9, alpha=0.9)
    if len(result.ppg_beats):
        y = np.interp(result.ppg_beats, result.ppg.time, result.ppg.data)
        axes[0].plot(result.ppg_beats, y, 'o', color='red', ms=5, zorder=6)
    med_bpm = float(np.median(ppg_bpm)) if ppg_bpm.size else float('nan')
    axes[0].set_title(f"PPG (maternal SOT)  |  {len(result.ppg_beats)} beats  |  median {med_bpm:.1f} BPM", fontsize=9)
    axes[0].set_ylabel("Amplitude")

    # Panel 2: Mic bandpassed + fetal beats
    axes[1].plot(result.mic.time, result.mic.data, color='tab:red', lw=0.7, alpha=0.85, rasterized=True)
    if len(result.mic_beats):
        y = np.interp(result.mic_beats, result.mic.time, result.mic.data)
        axes[1].plot(result.mic_beats, y, 'o', color='yellow', markeredgecolor='black',
                     markeredgewidth=0.6, ms=5, zorder=6)
    med_bpm_f = float(np.median(mic_bpm)) if mic_bpm.size else float('nan')
    axes[1].set_title(f"Mic 190–220 Hz (fetal SOT)  |  {len(result.mic_beats)} beats  |  median {med_bpm_f:.1f} BPM",
                      fontsize=9)
    axes[1].set_ylabel("Amplitude")

    # Panel 3: BPM over time
    if ppg_bpm.size:
        axes[2].plot(ppg_t, ppg_bpm, color='tab:blue', lw=1.2,
                     marker='o', ms=3, label='Maternal (PPG)')
    if mic_bpm.size:
        axes[2].plot(mic_t, mic_bpm, color='tab:red', lw=1.0,
                     marker='o', ms=2, alpha=0.85, label='Fetal (mic)')
    axes[2].axhline(120, color='gray', ls='--', lw=0.8, alpha=0.5)
    axes[2].axhline(160, color='gray', ls=':', lw=0.8, alpha=0.5)
    axes[2].set_ylabel("BPM")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_title("Sources of truth — instantaneous BPM")
    axes[2].legend(loc='upper right', fontsize=8)

    plt.savefig(out / "sot.png", dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Pipeline stage factory
# ---------------------------------------------------------------------------

@dataclass
class SOTData:
    ppg: Audio
    mic: Audio

    def window(self, start, end):
        return SOTData(
            ppg=self.ppg.window(start, end) if self.ppg is not None else None,
            mic=self.mic.window(start, end),
        )


def load_sot_no_ppg(
        fetal_band: Tuple[float, float] = FETAL_ACOUSTIC_BAND_HZ,
):
    def run_load_sot(data_dir: str) -> SOTData:
        path = normalize_path(data_dir)

        # --- Microphone (microphone.wav) ---
        mic_fs, mic_arr = wav_read(path + MIC_FILE)
        mic_arr = mic_arr.astype(float)
        if mic_arr.ndim > 1:
            mic_arr = mic_arr[:, 0]
        t_mic = np.arange(len(mic_arr)) / float(mic_fs)

        mic = Audio(t_mic, mic_fs, mic_arr)

        # mic = bp_filter(mic, fetal_band[0], fetal_band[1], filter_type='cheby1')
        # mic = Audio(t_mic, mic_fs, mic.data)

        return SOTData(
            None,
            mic
        )

    run_load_sot.__name__ = "load_sot"
    return run_load_sot


def load_sot(
        # out_dir: Path,
        # ppg_col: int = 1,
        fetal_band: Tuple[float, float] = FETAL_ACOUSTIC_BAND_HZ,
        maternal_bpm_range: Tuple[float, float] = MATERNAL_BPM_RANGE,
):
    """Pipeline stage factory: load and process PPG + mic sources of truth.

    Loads the *whole* recording — windowing is no longer applied here. Apply a
    `windowed(start, end)` stage afterwards for pipelines that need a windowed
    SOT (e.g. v1 `evaluate`); eval_v2 pipelines pass the full SOT so the initial
    lag can search beyond the analysis-window edges. Returns SOTResult.
    """

    def run_load_sot(data_dir: str) -> SOTResult:
        path = normalize_path(data_dir)

        # --- PPG (pvs.npy) ---
        pvs = np.load(path + PVS_FILE)
        t_ppg = pvs[:, 0].astype(float)
        x_ppg = pvs[:, 1].astype(float)
        hz_ppg = round(1.0 / float(np.median(np.diff(t_ppg))))

        x_ppg = detrend(x_ppg)
        ppg_raw = Audio(t_ppg, hz_ppg, x_ppg)
        ppg_filt = bp_filter(ppg_raw, 0.7, 4.0, filter_type='butter')
        # ppg_beats = detect_ppg_beats(ppg_filt, maternal_bpm_range)

        # --- Microphone (microphone.wav) ---
        mic_fs, mic_arr = wav_read(path + MIC_FILE)
        mic_arr = mic_arr.astype(float)
        if mic_arr.ndim > 1:
            mic_arr = mic_arr[:, 0]
        t_mic = np.arange(len(mic_arr)) / float(mic_fs)

        # mic_arr = detrend(mic_arr)
        # mic_arr = _robust_clip(mic_arr)
        # mic_arr = moving_average(mic_arr, round(mic_fs * 0.1))
        # mic_arr = np.abs(hilbert(mic_arr))

        # mic_arr = run_neossnet(mic_arr, mic_fs, FETAL_MODEL_PATH, FETAL_MODEL_CFG)
        mic = Audio(t_mic, mic_fs, mic_arr)

        mic = bp_filter(mic, fetal_band[0], fetal_band[1], filter_type='cheby1')
        mic = Audio(t_mic, mic_fs, mic.data)  # _suppress_transients(mic.data, float(mic_fs)))

        # mic_beats = detect_mic_fetal_beats(mic_hs_suppressed)
        #
        # med_ppg = float(np.median(ppg_beats['bpm'])) if len(ppg_beats['bpm']) else float('nan')
        # med_mic = float(np.median(mic_beats['bpm'])) if len(mic_beats['bpm']) else float('nan')
        # print(f"  SOT — PPG maternal: {len(ppg_beats['times'])} beats, median {med_ppg:.1f} BPM")
        # print(f"  SOT — Mic fetal:    {len(mic_beats['times'])} beats, median {med_mic:.1f} BPM")

        return SOTData(
            ppg_filt,
            mic
        )
        # result = SOTResult(
        #     ppg=ppg_filt,
        #     ppg_beats=ppg_beats["times"],
        #     ppg_ibi=ppg_beats["ibi"],
        #     ppg_bpm=ppg_beats["bpm"],
        #     mic=mic_hs_suppressed,
        #     mic=mic_raw_audio,
        #     mic_beats=mic_beats["times"],
        #     mic_ibi=mic_beats["ibi"],
        #     mic_bpm=mic_beats["bpm"],
        # )
        #
        # plot_sot(result, out_dir)

    run_load_sot.__name__ = "load_sot"
    return run_load_sot


def combine_sot_results(results: list[SOTResult]) -> SOTResult:
    return SOTResult(
        ppg=Audio(
            np.concatenate([e.ppg.time for e in results]),
            results[0].ppg.hz,
            np.concatenate([e.ppg.data for e in results]),
        ),
        ppg_beats=np.concatenate([e.ppg_beats for e in results]),
        mic=Audio(
            np.concatenate([e.mic.time for e in results]),
            results[0].mic.hz,
            np.concatenate([e.mic.data for e in results]),
        ),
        mic_beats=np.concatenate([e.mic_beats for e in results]),
    )


def plot_mic(out):
    def plot_mic_runner(sot: SOTData):
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 1, figsize=(15, 3), squeeze=False)

        data = sot.mic

        ax = axes[0][0]
        ax.plot(data.time, data.data, lw=0.5, color="steelblue")
        ax.set_title(f"Mic - output", fontsize=9)
        ax.set_xlabel("Time (s)", fontsize=8)
        ax.set_ylabel("Amplitude", fontsize=8)
        ax.tick_params(labelsize=7)

        fig.suptitle("ML source separation: heart", fontsize=11, y=1.01)
        fig.tight_layout()
        out_file = os.path.join(out, "mic_output.png")
        plt.savefig(out_file, dpi=150, bbox_inches="tight")
        plt.close()

        return sot
    return plot_mic_runner
