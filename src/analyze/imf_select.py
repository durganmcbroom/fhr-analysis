"""CEEMDAN IMF scoring, selection, and classification pipeline stage.

Input to select_best_imfs: (imfs, time, hz) tuple from ceemdan_decompose.
Output: FiberData ready for fetal_hr + evaluate stages.

Because the signal was bandpassed to 20–120 Hz before ANC and CEEMDAN,
the fetal_hr and evaluate stages downstream must be called with
detection_band=(20, 120) rather than the default (150, 250).
"""
from pathlib import Path
from typing import Tuple

import numpy as np
import numpy.typing as npt
from matplotlib import pyplot as plt
from scipy.signal import welch

from analyze.data import Audio, FiberData
from analyze.hr.detect import detect_fetal_beats, _fetal_quality_score
from constants import FETAL_BPM_RANGE


# ---------------------------------------------------------------------------
# IMF scoring
# ---------------------------------------------------------------------------

def _dominant_frequency(imf: np.ndarray, hz: float) -> float:
    """Return the frequency (Hz) of the peak power in the IMF."""
    freqs, psd = welch(imf, fs=hz, nperseg=min(len(imf), max(256, int(hz * 2))))
    return float(freqs[np.argmax(psd)])


def _score_imf(
    imf: np.ndarray,
    time: np.ndarray,
    hz: int,
    detection_band: Tuple[float, float],
    fetal_bpm_range: Tuple[float, float],
) -> dict:
    """Score a single IMF for fetal heartbeat content."""
    audio = Audio(time, hz, np.real(imf))
    beats = detect_fetal_beats(audio, detection_band=detection_band)
    quality = _fetal_quality_score(beats, fetal_bpm_range)
    dom_freq = _dominant_frequency(np.real(imf), float(hz))
    return {
        "quality": quality,
        "n_beats": len(beats["times"]),
        "median_bpm": float(np.median(beats["bpm"])) if len(beats["bpm"]) else float("nan"),
        "dominant_hz": dom_freq,
        "beats": beats,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_imf_scores(
    imfs: npt.NDArray,
    time: np.ndarray,
    hz: int,
    score_info: list,
    selected_indices: list,
    out: Path,
) -> None:
    n = len(score_info)
    fig, axes = plt.subplots(n, 2, figsize=(14, 3 * n), constrained_layout=True)
    if n == 1:
        axes = axes[np.newaxis, :]

    for row, info in enumerate(score_info):
        idx = info["index"]
        selected = idx in selected_indices
        color = "tab:green" if selected else "0.55"
        star = "  ★" if selected else ""
        title = (
            f"IMF {idx + 1}  quality={info['score']['quality']:.3f}"
            f"  beats={info['score']['n_beats']}"
            f"  median={info['score']['median_bpm']:.1f} BPM"
            f"  dom={info['score']['dominant_hz']:.1f} Hz{star}"
        )

        axes[row, 0].plot(time, np.real(imfs[idx]), color=color, lw=0.7, alpha=0.9)
        axes[row, 0].set_title(title, fontsize=8)
        axes[row, 0].set_ylabel("Amplitude", fontsize=7)

        freqs, psd = welch(np.real(imfs[idx]), fs=float(hz),
                           nperseg=min(len(imfs[idx]), max(256, int(hz * 2))))
        axes[row, 1].plot(freqs, psd, color=color, lw=0.9)
        axes[row, 1].set_xlim(0, min(200.0, hz / 2.0))
        axes[row, 1].set_title(f"PSD — IMF {idx + 1}", fontsize=8)
        axes[row, 1].set_ylabel("Power", fontsize=7)

    axes[-1, 0].set_xlabel("Time (s)", fontsize=8)
    axes[-1, 1].set_xlabel("Frequency (Hz)", fontsize=8)
    plt.savefig(out / "imf_scores.png", dpi=120)
    plt.close()


def _plot_combined(
    combined: np.ndarray,
    time: np.ndarray,
    beats: dict,
    selected_indices: list,
    out: Path,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), constrained_layout=True)

    axes[0].plot(time, combined, color="tab:green", lw=0.8, alpha=0.9)
    if len(beats["times"]):
        y = np.interp(beats["times"], time, combined)
        axes[0].plot(beats["times"], y, "o", color="red", ms=4, zorder=6)
    med = float(np.median(beats["bpm"])) if len(beats["bpm"]) else float("nan")
    axes[0].set_title(
        f"Combined IMFs {[i+1 for i in selected_indices]}"
        f"  |  {len(beats['times'])} beats  |  median {med:.1f} BPM",
        fontsize=9,
    )
    axes[0].set_ylabel("Amplitude")

    if len(beats["bpm"]) > 1:
        axes[1].plot(beats["times"][1:], beats["bpm"], color="tab:green",
                     lw=1.0, marker="o", ms=3)
    axes[1].axhline(120, color="gray", ls="--", lw=0.8, alpha=0.5)
    axes[1].axhline(160, color="gray", ls=":", lw=0.8, alpha=0.5)
    axes[1].set_ylabel("BPM")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_title("Combined IMF — instantaneous BPM")

    plt.savefig(out / "imf_combined.png", dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Pipeline stage factory
# ---------------------------------------------------------------------------

def select_best_imfs(
    out_dir: str,
    top_k: int = 4,
    detection_band: Tuple[float, float] = (20.0, 120.0),
    fetal_bpm_range: Tuple[float, float] = FETAL_BPM_RANGE,
):
    """Pipeline stage factory: score CEEMDAN IMFs and return top-K in a FiberData.

    The chest field of the returned FiberData is zeros because the chest fiber
    was discarded after ANC. Downstream fetal_hr will classify purely by fetal
    quality score (maternal correlation will be 0 for all channels).

    Call downstream fetal_hr and evaluate with detection_band=(20, 120) to
    match the 20–120 Hz range of the ANC'd CEEMDAN signal.
    """
    def run_select_best_imfs(data) -> FiberData:
        imfs, time, hz = data
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)

        n_imfs = imfs.shape[0]
        score_info = []

        for i in range(n_imfs):
            s = _score_imf(np.real(imfs[i]), time, int(hz), detection_band, fetal_bpm_range)
            score_info.append({"index": i, "score": s})
            print(
                f"  IMF {i+1:2d}:  quality={s['quality']:+.3f}"
                f"  beats={s['n_beats']:3d}"
                f"  median={s['median_bpm']:6.1f} BPM"
                f"  dom={s['dominant_hz']:6.1f} Hz"
            )

        # Rank by quality score, select top_k
        ranked = sorted(score_info, key=lambda x: x["score"]["quality"], reverse=True)
        selected_indices = [r["index"] for r in ranked[:top_k]]
        print(f"  Selected: IMFs {[i+1 for i in selected_indices]}")

        _plot_imf_scores(imfs, time, int(hz), score_info, selected_indices, out)

        # Build abdomen: individual selected IMFs + their sum
        combined_data = np.sum([np.real(imfs[i]) for i in selected_indices], axis=0)
        combined_audio = Audio(time, int(hz), combined_data)

        # Detect beats on combined for the combined plot
        combined_beats = detect_fetal_beats(combined_audio, detection_band=detection_band)
        _plot_combined(combined_data, time, combined_beats, selected_indices, out)

        abdomen = {
            f"imf_{i + 1}": Audio(time, int(hz), np.real(imfs[i]))
            for i in selected_indices
        }
        abdomen["imf_combined"] = combined_audio

        # Chest is zeros — no chest fiber survives ANC → fetal_hr classifies by quality only
        chest_zeros = Audio(time, int(hz), np.zeros(len(time)))

        return FiberData(chest_zeros, abdomen)

    run_select_best_imfs.__name__ = "select_best_imfs"
    return run_select_best_imfs
