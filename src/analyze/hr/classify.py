from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import correlate, correlation_lags

from analyze.data import Audio, FiberData, FiberPair
from analyze.hr.detect import detect_maternal_beats, detect_fetal_beats, _fetal_quality_score
from analyze.hr.utils import _gaussian_smooth, _impulse_train
from constants import MATERNAL_ACOUSTIC_BAND_HZ, FETAL_BPM_RANGE, XCORR_TARGET_FS


def _maternal_correlation_score(
    source_beat_times: np.ndarray,
    maternal_beat_times: np.ndarray,
    t_start: float,
    t_end: float,
    lag_bound_s: float = 5.0,
    target_fs: float = XCORR_TARGET_FS,
    smooth_sigma_s: float = 0.10,
) -> float:
    """Peak normalised xcorr between a source's beat train and the maternal beat train."""
    if len(source_beat_times) < 2 or len(maternal_beat_times) < 2:
        return 0.0

    t_grid = np.arange(t_start, t_end, 1.0 / target_fs)
    src_imp = _gaussian_smooth(_impulse_train(source_beat_times, t_grid), smooth_sigma_s * target_fs)
    mat_imp = _gaussian_smooth(_impulse_train(maternal_beat_times, t_grid), smooth_sigma_s * target_fs)

    corr = correlate(src_imp, mat_imp, mode='full')
    lags = correlation_lags(len(src_imp), len(mat_imp), mode='full') / target_fs
    mask = (lags >= -lag_bound_s) & (lags <= lag_bound_s)
    if not np.any(mask):
        return 0.0

    sub = corr[mask]
    std = float(np.std(sub))
    if std < 1e-9:
        return 0.0
    return float(sub.max() / std)


def _classify_sources(
    all_source_beats: Dict[str, Dict],
    maternal_beat_times: np.ndarray,
    audio_ref: Audio,
    fetal_bpm_range: Tuple[float, float] = FETAL_BPM_RANGE,
    maternal_corr_threshold: float = 0.5,
    use_bpm_exclusion: bool = False,
    bpm_exclusion_margin: float = 25.0,
    use_source_dedup: bool = False,
    dedup_xcorr_threshold: float = 0.3,
) -> Dict[str, str]:
    """Classify separated sources into fetal_hr / maternal_hr / fetal_hr_alt / noise."""
    t_start = float(audio_ref.time[0])
    t_end = float(audio_ref.time[-1])

    classification: Dict[str, str] = {}

    # --- Step 1: identify maternal sources ---
    if use_bpm_exclusion and len(maternal_beat_times) >= 2:
        maternal_median_bpm = 60.0 / float(np.median(np.diff(maternal_beat_times)))
        for name, beats in all_source_beats.items():
            if len(beats["bpm"]) < 1:
                continue
            if abs(float(np.median(beats["bpm"])) - maternal_median_bpm) < bpm_exclusion_margin:
                classification[name] = "maternal_hr"
    else:
        maternal_scores = {
            name: _maternal_correlation_score(beats["times"], maternal_beat_times, t_start, t_end)
            for name, beats in all_source_beats.items()
        }
        best_maternal = max(maternal_scores, key=lambda k: maternal_scores[k])
        if maternal_scores[best_maternal] >= maternal_corr_threshold:
            classification[best_maternal] = "maternal_hr"
        else:
            classification[best_maternal] = "noise"

    # --- Step 2: classify fetal candidates ---
    fetal_candidates = {n: b for n, b in all_source_beats.items() if n not in classification}

    if not fetal_candidates:
        fetal_scores_all = {n: _fetal_quality_score(b, fetal_bpm_range) for n, b in all_source_beats.items()}
        classification[max(fetal_scores_all, key=lambda k: fetal_scores_all[k])] = "fetal_hr"
        return classification

    if use_source_dedup and len(fetal_candidates) >= 2:
        names = list(fetal_candidates.keys())
        pair_xcorr: Dict[Tuple[str, str], float] = {}
        for i, n1 in enumerate(names):
            for n2 in names[i + 1:]:
                pair_xcorr[(n1, n2)] = _maternal_correlation_score(
                    fetal_candidates[n1]["times"],
                    fetal_candidates[n2]["times"],
                    t_start, t_end,
                )

        total_agreement = {n: 0.0 for n in names}
        for (n1, n2), score in pair_xcorr.items():
            if score > dedup_xcorr_threshold:
                total_agreement[n1] += score
                total_agreement[n2] += score

        fetal_scores = {n: _fetal_quality_score(b, fetal_bpm_range) for n, b in fetal_candidates.items()}
        combined = {n: fetal_scores[n] + 0.5 * total_agreement[n] for n in names}
        best_fetal = max(combined, key=lambda k: combined[k])
        classification[best_fetal] = "fetal_hr" if fetal_scores[best_fetal] > -np.inf else "noise"

        for name in names:
            if name in classification:
                continue
            key = (min(name, best_fetal), max(name, best_fetal))
            if pair_xcorr.get(key, 0.0) > dedup_xcorr_threshold:
                classification[name] = "fetal_hr_alt"
            else:
                median_ibi = float(np.median(fetal_candidates[name]["ibi"])) if len(fetal_candidates[name]["ibi"]) else np.inf
                classification[name] = "maternal_breathing" if median_ibi > 3.0 else "noise"
    else:
        fetal_scores = {n: _fetal_quality_score(b, fetal_bpm_range) for n, b in fetal_candidates.items()}
        best_fetal = max(fetal_scores, key=lambda k: fetal_scores[k])
        classification[best_fetal] = "fetal_hr" if fetal_scores[best_fetal] > -np.inf else "noise"
        for name in fetal_candidates:
            if name in classification:
                continue
            median_ibi = float(np.median(fetal_candidates[name]["ibi"])) if len(fetal_candidates[name]["ibi"]) else np.inf
            classification[name] = "maternal_breathing" if median_ibi > 3.0 else "noise"

    if "fetal_hr" not in classification.values():
        candidates = {n: b for n, b in all_source_beats.items() if classification.get(n) != "maternal_hr"}
        if candidates:
            fetal_scores_fb = {n: _fetal_quality_score(b, fetal_bpm_range) for n, b in candidates.items()}
            classification[max(fetal_scores_fb, key=lambda k: fetal_scores_fb[k])] = "fetal_hr"

    return classification


def _plot_sources(
    data: FiberData,
    all_source_beats: Dict[str, Dict],
    maternal_result: Dict,
    classification: Dict[str, str],
    out: Path,
) -> None:
    label_color = {
        "fetal_hr": "tab:green",
        "fetal_hr_alt": "tab:cyan",
        "maternal_hr": "tab:orange",
        "maternal_breathing": "tab:blue",
        "noise": "0.5",
    }

    n_sources = len(all_source_beats)
    fig, axes = plt.subplots(n_sources + 1, 1, figsize=(14, 4 * (n_sources + 1)), sharex=False)

    ax0 = axes[0]
    ax0.plot(data.chest.time, maternal_result["filtered"], color="tab:orange", lw=0.8, alpha=0.9)
    if len(maternal_result["times"]):
        y_mat = np.interp(maternal_result["times"], data.chest.time, maternal_result["filtered"])
        ax0.plot(maternal_result["times"], y_mat, "o", color="red", ms=5, zorder=6)
    mat_bpm_med = float(np.median(maternal_result["bpm"])) if len(maternal_result["bpm"]) else float("nan")
    ax0.set_title(f"Chest fiber (maternal band)  |  median BPM: {mat_bpm_med:.1f}", fontsize=9)
    ax0.set_ylabel("Amplitude")

    for i, (name, beats) in enumerate(all_source_beats.items(), start=1):
        label = classification.get(name, "noise")
        color = label_color[label]
        ax = axes[i]
        ax.plot(data.abdomen[name].time, beats["filtered"], color=color, lw=0.8, alpha=0.9)
        if len(beats["times"]):
            y_src = np.interp(beats["times"], data.abdomen[name].time, beats["filtered"])
            ax.plot(beats["times"], y_src, "o", color="red", ms=4, zorder=6)
        med_bpm = float(np.median(beats["bpm"])) if len(beats["bpm"]) else float("nan")
        ax.set_title(f"Source {name}  [{label}]  |  beats: {len(beats['times'])}  |  median BPM: {med_bpm:.1f}", fontsize=9)
        ax.set_ylabel("Amplitude")

    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.savefig(out / "fetal_hr_sources.png", dpi=150)
    plt.close()


def _plot_bpm(
    all_source_beats: Dict[str, Dict],
    maternal_result: Dict,
    classification: Dict[str, str],
    out: Path,
) -> None:
    label_color = {
        "fetal_hr": "tab:green",
        "fetal_hr_alt": "tab:cyan",
        "maternal_hr": "tab:orange",
        "maternal_breathing": "tab:blue",
        "noise": "0.5",
    }

    fig, ax = plt.subplots(figsize=(14, 5))

    if len(maternal_result["bpm"]) > 1:
        ax.plot(maternal_result["times"][1:], maternal_result["bpm"],
                color="tab:orange", lw=1.2, marker="o", ms=3, label="Maternal (chest)")

    for name, beats in all_source_beats.items():
        if len(beats["bpm"]) < 2:
            continue
        label = classification.get(name, "noise")
        color = label_color[label]
        ax.plot(beats["times"][1:], beats["bpm"],
                color=color, lw=1.0, marker="o", ms=3, alpha=0.85,
                label=f"{name} [{label}]")

    ax.axhline(120, color="gray", ls="--", lw=0.8, alpha=0.6)
    ax.axhline(160, color="gray", ls=":", lw=0.8, alpha=0.6)
    ax.set_ylabel("BPM")
    ax.set_xlabel("Time (s)")
    ax.set_title("Instantaneous heart rates")
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig(out / "fetal_hr_bpm.png", dpi=150)
    plt.close()


def classify_sources(
    out_dir: str,
    maternal_band: Tuple[float, float] = MATERNAL_ACOUSTIC_BAND_HZ,
    min_interval_s: float = 0.27,
    threshold_factor: float = 0.40,
    fetal_bpm_range: Tuple[float, float] = FETAL_BPM_RANGE,
    use_bpm_exclusion: bool = False,
    bpm_exclusion_margin: float = 25.0,
    use_source_dedup: bool = False,
    dedup_xcorr_threshold: float = 0.3,
):
    """Pipeline stage factory: classify separated sources and select the fetal one.

    Takes FiberData (chest + multiple sources), scores each source, and returns
    a FiberPair of (chest, selected_fetal_source). Both experimental options
    default off.
    """
    def run_classify_sources(data: FiberData) -> FiberPair:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)

        maternal_result = detect_maternal_beats(data.chest, maternal_band)
        print(f"  Maternal beats detected: {len(maternal_result['times'])}"
              f"  (median {np.median(maternal_result['bpm']):.1f} BPM)" if len(maternal_result['bpm']) else
              f"  Maternal beats detected: 0")

        all_source_beats = {
            name: detect_fetal_beats(audio, min_interval_s, threshold_factor)
            for name, audio in data.abdomen.items()
        }
        for name, beats in all_source_beats.items():
            med = float(np.median(beats["bpm"])) if len(beats["bpm"]) else float("nan")
            print(f"  Source {name}: {len(beats['times'])} beats, median {med:.1f} BPM")

        classification = _classify_sources(
            all_source_beats, maternal_result["times"], data.chest, fetal_bpm_range,
            use_bpm_exclusion=use_bpm_exclusion,
            bpm_exclusion_margin=bpm_exclusion_margin,
            use_source_dedup=use_source_dedup,
            dedup_xcorr_threshold=dedup_xcorr_threshold,
        )
        print(f"  Classification: {classification}")

        fetal_name = next((n for n, l in classification.items() if l == "fetal_hr"), None)
        if fetal_name is None:
            fetal_name = next(iter(data.abdomen))

        _plot_sources(data, all_source_beats, maternal_result, classification, out)
        _plot_bpm(all_source_beats, maternal_result, classification, out)

        return FiberPair(data.chest, data.abdomen[fetal_name])

    run_classify_sources.__name__ = "classify_sources"
    return run_classify_sources
