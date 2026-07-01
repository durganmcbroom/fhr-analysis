import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import hilbert, find_peaks

from analyze.data import Audio
from analyze.util import moving_average


def _shannon_energy(data):
    x = np.clip(data ** 2, 1e-12, None)
    shannon_energy = -1 * x * np.log10(x)

    return shannon_energy


def _envelope(X: np.ndarray) -> np.ndarray:
    X = moving_average(X, 100)
    return np.abs(hilbert(X))


def _local_floor(
        env: np.ndarray,
        floor_k: float = 1.0,
):
    med = float(np.median(env))
    mad = float(np.median(np.abs(env - med))) + 1e-12
    floor = med + floor_k * 1.4826 * mad
    return floor


def v2_beat_detector(
        X: Audio,
        bpm_range,
        out,
        energy_range=0.5,
        tag="",
):
    energy = _shannon_energy(X.data)
    envelope = _envelope(energy)
    min_dist = max(1, int(round(1 / (bpm_range[1] / 60) * X.hz)))

    # window_len = round(bpm_range[0] / 60.0 * X.hz)

    # peaks = []
    # floors = []  # (start, end, floor) per window
    # for start in range(0, len(X.data), window_len):
    # end = min(start + window_len, len(X.data))
    # data = envelope[start:end]

    floor = _local_floor(envelope, floor_k=1e-20)
    # floors.append((start, end, floor))

    peaks, _ = find_peaks(envelope, distance=min_dist,
                                 # prominence=floor
                                 )
    # peaks.extend(peak_idx + start)

    normalized_peaks = []
    beats = []
    n = len(peaks)
    for i, peak_idx in enumerate(peaks):
        energy_bound = energy_range * envelope[peak_idx]

        # Confine the lobe search to this beat's Voronoi cell (the midpoints to
        # the neighbouring peaks). Otherwise a small beat's threshold is so low
        # that the envelope never drops below it in the valley before a louder
        # neighbour, so the right crossing is found past that neighbour and the
        # midpoint gets dragged on top of it.
        left_limit = (peaks[i - 1] + peak_idx) // 2 if i > 0 else 0
        right_limit = (peak_idx + peaks[i + 1]) // 2 if i < n - 1 else len(envelope)

        left = np.where(envelope[left_limit:peak_idx] <= energy_bound)[0]
        right = np.where(envelope[peak_idx + 1:right_limit] <= energy_bound)[0]

        if len(left) and len(right):
            left_idx = left_limit + left[-1]
            right_idx = peak_idx + 1 + right[0]
            middle = (left_idx + right_idx) // 2
        else:
            # Lobe doesn't close inside the cell -> trust the envelope peak.
            middle = peak_idx
            left_idx = peak_idx
            right_idx = peak_idx
        normalized_peaks.append(middle)
        beats.append((left_idx, right_idx))

    normalized_peaks = np.unique(normalized_peaks)  # sorted + de-duped

    if out is not None:
        _plot_stages(X.time, X.data, beats, energy, envelope, peaks, [(0, len(X.time), floor)], normalized_peaks, out, tag)

    beat_times = X.time[normalized_peaks] if len(normalized_peaks) else np.array([], dtype=float)

    return {
        "peaks": normalized_peaks,
        "times": beat_times,
    }


# def detect_beats_v2(
#         out_dir: str,
#         maternal_band: Tuple[float, float] = (40.0, 80.0),
#         min_interval_s: float = 0.27,
#         threshold_factor: float = 0.40,
# ):
#     def run_detect_beats(data: FiberPair) -> FetalHRResult:
#         out = Path(out_dir)
#         out.mkdir(parents=True, exist_ok=True)
#
#         chest = bp_filter(data.chest, maternal_band[0], maternal_band[1])
#         maternal_result = detect_beats_for_v2(chest, (50, 100), out, tag="maternal")
#         fetal_result = detect_beats_for_v2(data.abdomen, (80, 220), out, tag="fetal")
#
#         return FetalHRResult(
#             fetal_source=data.abdomen,
#             fetal_beats=fetal_result["times"],
#             fetal_ibi=fetal_result["ibi"],
#             fetal_bpm=fetal_result["bpm"],
#             maternal_source=data.chest,
#             maternal_beats=maternal_result["times"],
#             maternal_bpm=maternal_result["bpm"],
#         )
#
#     run_detect_beats.__name__ = "detect_beats"
#     return run_detect_beats


# --- per-stage diagnostics: all four stages in one figure -------------------

def _plot_stages(time, data, beats, energy, envelope, peaks, floors, normalized_peaks, out, tag):
    fig, axes = plt.subplots(5, 1, figsize=(14, 10), sharex=True, constrained_layout=True)

    axes[0].plot(time, energy, lw=0.7, color="tab:blue")
    axes[0].set_title(f"{tag} Shannon energy")

    axes[1].plot(time, envelope, lw=0.7, color="tab:purple")
    axes[1].set_title("envelope")

    axes[2].plot(time, envelope, lw=0.7, color="tab:purple")
    for start, end, floor in floors:  # floor reference per window
        axes[2].hlines(floor, time[start], time[end - 1], color="tab:red", lw=0.8)
    if len(peaks):
        axes[2].plot(time[peaks], envelope[peaks], "x", color="k", ms=5)
    axes[2].set_title("peaks + per-window floor")

    axes[3].plot(time, envelope, lw=0.7, color="tab:purple")
    if len(normalized_peaks):
        axes[3].vlines(time[normalized_peaks], 0, float(envelope.max()), color="tab:green", lw=0.8)
    axes[3].set_title("normalized peaks")
    axes[3].set_xlabel("Time (s)")

    axes[4].plot(time, data, lw=0.7, color="tab:blue")
    if len(normalized_peaks):
        axes[4].vlines(time[peaks], float(data.min()), float(data.max()), linestyles="--", color="tab:red", lw=0.8)
    for start, end in beats:
        axes[4].axvspan(time[start], time[end], color="tab:green", alpha=0.2)
        # axes[4].vlines(time[normalized_peaks], 0, float(envelope.max()), color="tab:green", lw=0.8)
    axes[4].set_title("beats on waveform")
    axes[4].set_xlabel("Time (s)")

    fig.savefig(out / f"detect_v2_{tag}.png", dpi=150)
    plt.close(fig)
