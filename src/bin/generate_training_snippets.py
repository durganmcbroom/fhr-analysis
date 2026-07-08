#!/usr/bin/env python3
"""Build a paired fetal/maternal HR + SOT snippet dataset from a yaml window spec.

Yaml format (same as lib/tune-ssnet/training_clips_mono.yaml):

    mode: stereo    # required: 'stereo' (all fibers as channels) or 'mono' (one snippet per fiber)
    length: 10      # snippet length in seconds
    overlap: 3      # overlap in seconds between adjacent snippets within a section
    ppg_col: 0      # column of pvs.npy to use as maternal SOT (defaults to 0)
    time_warp: False  # randomly stretch/compress the gaps between beats (the "random
                      # intervals"); set False to turn it off (default off)
    warp_strength: 12 # max random stretch factor applied to inter-beat gaps (only when
                      # time_warp is on; default 12)
    warp_pad: 300     # samples left untouched around each beat before a gap is warped
                      # (only when time_warp is on; default 300)
    gate_width_ibi_fraction: 0.3  # heartbeat gate half-width as a fraction of the mean
                                  # inter-beat interval (bigger => wider gate; default 0.3)
    data:
      "PT12_1":
        mode: train       # required: 'train' or 'test' -> routes this patient's
        fibers:           # snippets to the <out-dir>/fetal-train/ or fetal-test/ dir
          - "2A"
          - "2B"
        sections:
          - "130-150"
          - "170-178"
      "PT14_1":
        mode: test
        fibers:
          - "2A"
        sections:
          - "20-90"

Each "start-end" section is cut into `length`-second snippets stepping by
`length - overlap`. Every patient declares mode: 'train' or 'test', which routes
all of its snippets to `<out-dir>/fetal-train/` or `<out-dir>/fetal-test/` -- a
whole patient is held out for testing, with no within-patient leakage. Snippet
indices restart from 0 within each split. In stereo mode all listed fibers are
stacked as channels into one snippet per window; in mono mode each fiber is its
own snippet.
"""

import argparse
import sys
from logging import warning
from pathlib import Path

import matplotlib
import numpy as np
import scipy.signal
import torch
import yaml
from scipy.io import wavfile
from scipy.signal import detrend, resample_poly


matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.transforms import blended_transform_factory

SRC_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SRC_DIR))

MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "lib" / "neossnet"
sys.path.insert(0, str(MODEL_DIR))

from analyze.hr.detect import v1_beat_detector
from analyze.hr.detect_v2 import v2_beat_detector
from analyze.sot import _robust_clip, _moving_avg
from analyze.hr.detect_v5 import v5_beat_detector
from analyze.hr.detect_v6 import v6_beat_detector

from analyze.anc import nlms_filter
from analyze.filters import bp_filter
from utils import generate_output

from analyze.data import Audio, load_fibers  # noqa: E402
from analyze.util import normalize_path  # noqa: E402
from constants import (
    DEFAULT_DATA_DIR, ABDOMEN_FIBER_NAMES, FIBER_BUNDLE_B, MIC_FILE, PVS_FILE,
    NEOSSNET_MODEL_HZ, FETAL_ACOUSTIC_BAND_HZ, MATERNAL_ACOUSTIC_BAND_HZ, FETAL_BPM_RANGE,
    NEOSSNET_MODEL_PATH, NEOSSNET_MODEL_CFG,
)  # noqa: E402

WINDOW_IBI_FRACTION = 0.3  # half-window per beat = this * mean IBI
SNAP_TOLERANCE_S = 0.040  # how far a beat may be nudged to the fiber's own energy peak

def chunk_mask(t: np.ndarray, start: float, end: float) -> np.ndarray:
    return (t >= start) & (t < end)


def parse_window(spec: str) -> tuple[float, float]:
    start_s, end_s = spec.split("-")
    return float(start_s), float(end_s)


def snippet_starts(start: float, end: float, length: float, overlap: float) -> list[float]:
    """Start times of fixed-length, possibly-overlapping snippets within [start, end].

    Steps by (length - overlap). Any trailing remainder too short for a full
    snippet is dropped (not padded).
    """
    step = length - overlap
    starts = []
    t = start
    while t + length <= end + 1e-9:
        starts.append(t)
        t += step
    return starts


def load_mic(path: str) -> Audio:
    fs, arr = wavfile.read(path + MIC_FILE)
    arr = arr if arr.ndim == 1 else arr[:, 0]
    t = np.arange(len(arr)) / float(fs)
    return Audio(t, fs, arr)


def load_ppg(path: str, ppg_col: int) -> Audio:
    pvs = np.load(path + PVS_FILE)
    t = pvs[:, 0].astype(float)
    x = pvs[:, ppg_col].astype(float)
    hz = round(1.0 / float(np.median(np.diff(t))))
    return Audio(t, hz, x)


def write_mono(path: Path, hz: int, data: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.asarray(data, dtype=np.float32)
    x = x - x.mean()  # kill DC (critical for PPG)
    peak = np.max(np.abs(x))
    if peak > 0:
        x = x / peak  # into [-1, 1]
    wavfile.write(path, hz, x)


def resample(X, hz, target):
    g = np.gcd(target, int(hz))
    up, down = target // g, int(hz) // g
    return resample_poly(X, up, down) if up != down else X


def lung_sound(X):
    x = np.asarray(X, dtype=float).ravel()

    peak = float(np.max(np.abs(x))) + 1e-12
    xn = x / peak

    tensor = torch.tensor(xn, dtype=torch.float32).unsqueeze(0)  # (1, T)
    _, lung = generate_output(tensor, NEOSSNET_MODEL_PATH, NEOSSNET_MODEL_CFG)

    return lung.numpy()


def noise_sound(X, heart, lung):
    X = nlms_filter(X, heart, round(0.5 * NEOSSNET_MODEL_HZ))
    X = nlms_filter(X, lung, round(0.5 * NEOSSNET_MODEL_HZ))
    return X


class NoBeatException(Exception):
    pass


def window_audio(audio: Audio, start: float, end: float) -> Audio:
    mask = chunk_mask(audio.time, start, end)
    return Audio(audio.time[mask], audio.hz, audio.data[mask])


def beat_centers(beat_times, start_time, sample_rate):
    return [int(round((beat_time - start_time) * sample_rate)) for beat_time in beat_times]


def half_width_samples(mean_ibi_s, sample_rate, ibi_fraction=WINDOW_IBI_FRACTION):
    return int(round(mean_ibi_s * ibi_fraction * sample_rate))


def extract_window(signal, center, half_width):
    start, stop = center - half_width, center + half_width
    if start < 0 or stop > len(signal):
        return None
    return signal[start:stop]


def normalize_window(window):
    # Zero-mean, unit-std so loud beats don't dominate the average.
    std = float(np.std(window))
    if std < 1e-12:
        return None
    return (window - np.mean(window)) / std


def collect_windows(signals, centers, half_width):
    # Every z-scored beat window that fits fully inside its signal, across all fibers.
    windows = []
    for signal in signals:
        for center in centers:
            window = extract_window(signal, center, half_width)
            if window is None:
                continue
            normalized = normalize_window(window)
            if normalized is not None:
                windows.append(normalized)
    return windows


def fiber_energy(signal, sample_rate, window_s=0.02):
    return _moving_avg(np.abs(signal), max(1, int(round(window_s * sample_rate))))


def snap_centers_to_energy(energy, centers, tolerance):
    # Nudge each beat center to the nearest fiber energy peak within +/-tolerance samples.
    snapped = []
    for center in centers:
        lo, hi = max(0, center - tolerance), min(len(energy), center + tolerance)
        snapped.append(lo + int(np.argmax(energy[lo:hi])) if hi > lo else center)
    return snapped


def beat_gate(centers, X, half_width):
    # Smooth (Hann) window around each beat, zero between beats.
    length = len(X)
    window = np.hanning(2 * half_width)
    gate = np.zeros(length)
    for center in centers:
        start, stop = center - half_width, center + half_width
        win_start, win_stop = 0, 2 * half_width
        if start < 0:
            win_start, start = -start, 0
        if stop > length:
            win_stop, stop = win_stop - (stop - length), length
        if stop <= start:
            continue

        amp = window[win_start:win_stop] / (np.max(np.abs(X[start:stop])) + 1e-12)

        gate[start:stop] = np.maximum(gate[start:stop], amp)
    return gate


def gated_heart(filtered, centers, half_width, offset=0):
    # Option 2: keep the real band-passed fiber around each beat, silenced between.
    return filtered * beat_gate(centers, filtered, half_width)


def normalized_template(template):
    peak = np.max(np.abs(template))
    return template / peak if peak > 0 else template


def stamp_template(template, centers, half_width, amplitude_reference, length):
    # Overlap-add the template at each beat, scaled by local fiber amplitude, clipped at edges.
    width = 2 * half_width
    output = np.zeros(length)
    for center in centers:
        start, stop = center - half_width, center + half_width
        template_start, template_stop = 0, width
        if start < 0:
            template_start, start = -start, 0
        if stop > length:
            template_stop, stop = template_stop - (stop - length), length
        if stop <= start:
            continue
        amplitude = np.max(np.abs(amplitude_reference[start:stop]))
        amplitude = 10 * np.log10(amplitude + 10)  # We dont want amp to be too small
        output[start:stop] += template[template_start:template_stop] * amplitude
    return output


def ensemble_heart(filtered, centers, half_width):
    # Fallback: stamp one ensemble-averaged beat template at each beat (no alignment).
    windows = collect_windows([filtered], centers, half_width)
    if not windows:
        raise NoBeatException("no full-width beat windows to build a template")
    template = normalized_template(np.mean(windows, axis=0))
    return stamp_template(template, centers, half_width, filtered, len(filtered))

def heart_target(
        beat_evaluator,
        fibers,
        t,
        sot,
        band,
        gaussian_impulses: bool = False,
        gate_width_ibi_fraction: float = WINDOW_IBI_FRACTION,
):
    # Single-channel clean heartbeat, centered on the reference fiber's own beats.
    evaluation = beat_evaluator(sot)
    beat_times = evaluation["times"]
    if len(beat_times) < 1:
        raise NoBeatException(f"no beats detected in {t[0]}-{t[-1]}")

    reference_fiber = fibers[0]

    sample_rate = round(1 / (t[1] - t[0]))  # reference_fiber.hz
    ibi = np.diff(beat_times)
    half_width = half_width_samples(float(np.mean(ibi)), sample_rate, gate_width_ibi_fraction)
    filtered = bp_filter(Audio(t, sample_rate, reference_fiber), band[0], band[1], filter_type="butter").data

    centers = beat_centers(beat_times, t[0], sample_rate)
    centers = snap_centers_to_energy(fiber_energy(filtered, sample_rate),
                                     centers, int(round(SNAP_TOLERANCE_S * sample_rate)))
    if gaussian_impulses:
        heart = np.zeros_like(reference_fiber)
        sigma = half_width / 4  # tight: pulse is essentially decayed to 0 by the window edge
        for center in centers:
            a = max(0, center - half_width)
            b = min(len(heart), center + half_width)
            x = np.arange(a, b)

            gaussian_template = np.exp(-0.5 * ((x - center) / sigma) ** 2)
            heart[a:b] += gaussian_template
    else:
        heart = gated_heart(filtered, centers, half_width, 0)  # swap to ensemble_heart(...) to revert

    # sos = cheby1(4, rp=1, Wn=band, fs=sample_rate, btype='bandpass', output='sos')
    # heart = sosfiltfilt(sos, heart, axis=0)
    # amplitude = np.mean(np.abs(heart)) * noise_magnitude
    # noise = np.random.normal(0, amplitude, len(heart))

    return heart, beat_times, half_width


def beat_lines(axis, beat_times):
    transform = blended_transform_factory(axis.transData, axis.transAxes)
    axis.vlines(beat_times, 0, 1, transform=transform, color="0.3", lw=0.8, ls="--", alpha=0.7)


def plot_heart(path, sot, raw, heart, heart_hz, start_time, beat_times, title):
    # Two-panel waveform: SOT on top, generated heart below, SOT beats marked on both.
    heart_time = start_time + np.arange(len(heart)) / heart_hz
    figure, (top, middle, bottom) = plt.subplots(3, 1, figsize=(12, 5), sharex=True)

    top.plot(sot.time, sot.data, color="tab:red", lw=0.6)
    top.set_ylabel("SOT")
    top.set_title(title, fontsize=9)

    middle.plot(heart_time, heart, color="tab:green", lw=0.6)
    middle.set_ylabel("generated heart")
    middle.set_xlabel("time (s)")

    bottom.plot(heart_time, raw, color="tab:blue", lw=0.6)
    bottom.set_ylabel("raw")
    bottom.set_xlabel("time (s)")

    for axis in (top, bottom):
        beat_lines(axis, beat_times)
    bottom.set_xlim(float(heart_time[0]), float(heart_time[-1]))

    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)


def stack_resampled(fibers):
    # All fibers resampled to MODEL_HZ and stacked as (channels, time).
    channels = [resample(fiber.data, fiber.hz, NEOSSNET_MODEL_HZ) for fiber in fibers]
    length = min(len(channel) for channel in channels)
    return np.vstack([channel[:length] for channel in channels])


def write_multichannel(path: Path, hz: int, channels: np.ndarray) -> None:
    # Multi-channel wav, DC-removed per channel and globally peak-normalised.
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.asarray(channels, dtype=np.float32)
    data = data - data.mean(axis=1, keepdims=True)
    peak = np.max(np.abs(data))
    if peak > 0:
        data = data / peak
    wavfile.write(path, hz, data.T)


def time_warp(
        X,
        sections,
        resampler=lambda x, n: scipy.signal.resample(x, n) #np.zeros(n)
):
    """
    >>> time_warp(np.array([0, 1, 2, 3, 4, 5]), []).astype(int).tolist()
    [0, 1, 2, 3, 4, 5]
    >>> time_warp(np.array([0, 1, 2, 3, 4, 5]), [(0, 1, 3)]).astype(int).tolist()
    [0, 0, 0, 1, 2, 3, 4, 5]
    >>> time_warp(np.array([0, 1, 2, 3, 4, 5]), [(2, 4, 5)]).astype(int).tolist()
    [0, 1, 2, 2, 2, 2, 2, 4, 5]
    """
    # Each section replaces X[s:e] (length e-s) with new_size samples,
    # so it changes total length by new_size - (e - s).
    length = len(X) + sum(new_size - (end_idx - start_idx) for start_idx, end_idx, new_size in sections)

    data = np.zeros(length)
    in_pos = 0   # how far we've consumed X
    out_pos = 0  # how far we've written data
    for start_idx, end_idx, new_size in sections:
        # Copy the untouched gap before this section verbatim.
        gap = start_idx - in_pos
        data[out_pos:out_pos + gap] = X[in_pos:start_idx]
        out_pos += gap

        # Resample X[start_idx:end_idx] into new_size samples in its place.
        data[out_pos:out_pos + new_size] = resampler(X[start_idx:end_idx], new_size)
        out_pos += new_size
        in_pos = end_idx

    # Trailing remainder after the last section, unchanged.
    data[out_pos:] = X[in_pos:]

    return data


def write_snippet(out_dir, idx, beat_evaluator, fibers, sot, band, gaussian_impulses=False, warp=False, k=12, pad=300,
                  gate_width_ibi_fraction=WINDOW_IBI_FRACTION):
    # One mix (mono or multi-channel) plus its single-channel heart/lung/noise targets, with a plot.
    # `warp` randomly stretches/compresses the gaps between beats (k = max stretch factor,
    # pad = samples kept untouched around each beat); `gate_width_ibi_fraction` sets the
    # heartbeat gate half-width as a fraction of the mean inter-beat interval.
    mix = stack_resampled(fibers)
    t = fibers[0].time[0] + np.arange(len(mix[0])) / NEOSSNET_MODEL_HZ

    try:
        heart, beat_times, half_width = heart_target(beat_evaluator, mix, t, sot, band, gaussian_impulses,
                                                     gate_width_ibi_fraction)
    except NoBeatException as error:
        warning(f"skipping snippet {idx}: {error}")
        return False

    reference = mix.mean(axis=0)
    # heart = resample(h_target, fibers[0].hz, MODEL_HZ)
    # length = min(mix.shape[1], len(reference), len(h_target))
    # mix, reference, heart = mix[:, :length], reference[:length], h_target[:length]
    lung = lung_sound(reference)
    noise = noise_sound(reference, heart, lung)

    if warp:
        rng = np.random.default_rng(42)
        # Use ref fiber to get beat idxs
        beat_idx = np.searchsorted(t, beat_times)
        warp_sections = []
        i = 0
        padded_half_width = half_width + pad

        for bidx in beat_idx:
            start = max(0, bidx - padded_half_width)
            end = min(t.shape[0], bidx + padded_half_width)

            if start > i:
                scale = max((rng.random()) * k, 0.1)
                new_len = round((start - i) * scale)
                warp_sections.append((i, start, new_len))
            i = end

        mix = np.stack([time_warp(channel, warp_sections) for channel in mix])
        heart = time_warp(heart, warp_sections)
        lung = time_warp(lung, warp_sections)
        noise = time_warp(noise, warp_sections)

    write_multichannel(out_dir / f"{idx}_mix.wav", NEOSSNET_MODEL_HZ, mix)
    write_mono(out_dir / f"{idx}_heart.wav", NEOSSNET_MODEL_HZ, heart)
    write_mono(out_dir / f"{idx}_lung.wav", NEOSSNET_MODEL_HZ, lung)
    write_mono(out_dir / f"{idx}_noise.wav", NEOSSNET_MODEL_HZ, noise)
    plot_heart(out_dir / f"{idx}_heart.png", sot, mix[0], heart, NEOSSNET_MODEL_HZ, fibers[0].time[0], beat_times,
               f"{out_dir.name} snippet {idx}")
    return True


def fiber_groups(named_fibers, mode):
    # stereo: one group with every fiber as a channel; mono: one group per fiber.
    if mode == "stereo":
        return [named_fibers]
    return [[named_fiber] for named_fiber in named_fibers]


def require_mode(cfg):
    mode = cfg.get("mode")
    if mode not in ("mono", "stereo"):
        raise ValueError(f"config 'mode' is required and must be 'mono' or 'stereo', got {mode!r}")
    return mode


def require_split_mode(name, dir_spec):
    # Per-patient train/test split -> routes to the fetal-train/ or fetal-test/ output dir.
    mode = dir_spec.get("mode")
    if mode not in ("train", "test"):
        raise ValueError(f"patient {name!r} must declare mode: 'train' or 'test' (got {mode!r})")
    return mode


def fetal_detector(out, idx):
    # Match load_sot's detection signal: detrend + robust-clip before peak finding.
    # v6 is period-locked (one heart sound per cardiac cycle), so the SOT labels
    # don't flip between S1 and S2 the way v2/v5's tallest-lobe picking does;
    # see analyze/hr/detect_v6.py.
    def det(raw_audio):
        prepared = _robust_clip(detrend(raw_audio.data))
        return v6_beat_detector(
            Audio(raw_audio.time, raw_audio.hz, prepared),
            FETAL_BPM_RANGE,
            out,
            tag=f"{idx}_detections",
        )
    return det


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Slice Banner_data directories into paired fetal/maternal HR + SOT wav snippets"
    )
    parser.add_argument(
        "yaml_path", type=Path,
        help="Yaml file with mode ('mono'|'stereo'), length/overlap/ppg_col and data: {dir_name: {fibers: [...], sections: [\"start-end\", ...]}}"
    )
    parser.add_argument(
        "--data-dir", default=DEFAULT_DATA_DIR,
        help=f"Base directory containing the patient subdirectories (default: {DEFAULT_DATA_DIR})"
    )
    parser.add_argument("--out-dir", required=True, type=Path, help="Output base directory")
    args = parser.parse_args()

    with open(args.yaml_path) as f:
        cfg: dict = yaml.safe_load(f)

    mode = require_mode(cfg)
    length = cfg.get("length")  # None => each window becomes a single snippet
    overlap = cfg.get("overlap", 0.0)
    ppg_col = cfg.get("ppg_col", 0)
    time_warp_enabled = cfg.get("time_warp", False)
    gaussian_impulses_enabled = cfg.get("gaussian_impulses", False)
    # Random inter-beat interval warping controls (only used when time_warp is on).
    warp_strength = cfg.get("warp_strength", 12)  # max random stretch factor for inter-beat gaps
    warp_pad = cfg.get("warp_pad", 300)           # samples kept untouched around each beat
    # Heartbeat gate half-width = this fraction of the mean IBI (bigger => wider gate).
    gate_width_ibi_fraction = cfg.get("gate_width_ibi_fraction", WINDOW_IBI_FRACTION)

    spec: dict[str, dict] = cfg["data"]

    if overlap and length is not None and overlap >= length:
        raise ValueError(f"overlap ({overlap}) must be less than length ({length})")

    data_dir = normalize_path(args.data_dir)
    out_dir = args.out_dir

    # Each patient's mode routes its snippets to a held-out train/test split dir.
    fetal_dirs = {"train": out_dir / "fetal-train", "test": out_dir / "fetal-test"}
    idxs = {"train": 0, "test": 0}   # snippet indices restart from 0 within each split
    created: list[tuple[str, int, str, str, str]] = []  # (split, index, dir_name, window_spec, fiber)

    for dir_name, dir_spec in spec.items():
        fiber_names = dir_spec["fibers"]
        windows = dir_spec["sections"]
        split = require_split_mode(dir_name, dir_spec)   # "train" or "test"
        fetal_dir = fetal_dirs[split]

        unknown = [f for f in fiber_names if f not in ABDOMEN_FIBER_NAMES]
        if unknown:
            raise ValueError(
                f"{dir_name}: unknown fiber(s) {unknown}, must be one of {ABDOMEN_FIBER_NAMES}"
            )

        patient_path = normalize_path(f"{data_dir}{dir_name}")
        print(f"\nLoading {dir_name} ... (-> {split})")
        fibers = load_fibers(Path(patient_path))
        mic = load_mic(patient_path)
        ppg = load_ppg(patient_path, ppg_col)

        for window_spec in windows:
            start, end = parse_window(window_spec)
            if end <= start:
                print(f"  WARNING: {dir_name} [{window_spec}] has end <= start; skipping")
                continue

            win_length = length if length is not None else (end - start)
            starts = snippet_starts(start, end, win_length, overlap)
            if not starts:
                print(
                    f"  WARNING: {dir_name} [{window_spec}] is shorter than "
                    f"length {win_length}s; no snippets produced"
                )

            for window_start in starts:
                window_end = window_start + win_length

                # Fetal input is the abdomen fibers; maternal is the chest bundle.
                named_fibers = [(name, window_audio(fibers.abdomen[name], window_start, window_end))
                                for name in fiber_names]
                # chest = window_audio(fibers.chest, window_start, window_end)
                mic_window = window_audio(mic, window_start, window_end)
                # ppg_window = window_audio(ppg, window_start, window_end)

                for group in fiber_groups(named_fibers, mode):
                    idx = idxs[split]
                    print(f"Computing {split} sample {idx} ({dir_name})")
                    group_names = [name for name, _ in group]
                    group_fibers = [fiber for _, fiber in group]

                    fetal_ok = write_snippet(
                        fetal_dir, idx, fetal_detector(fetal_dir, idx), group_fibers, mic_window, FETAL_ACOUSTIC_BAND_HZ,
                        warp=time_warp_enabled,
                        gaussian_impulses=gaussian_impulses_enabled,
                        k=warp_strength,
                        pad=warp_pad,
                        gate_width_ibi_fraction=gate_width_ibi_fraction,
                    )

                    if fetal_ok:
                        created.append((split, idx, dir_name, f"{window_start:.3f}-{window_end:.3f}", "+".join(group_names)))
                        idxs[split] += 1

    print(f"\nCreated {idxs['train']} train + {idxs['test']} test snippets in {out_dir}/")
    print("\nIndex map:")
    for split, i, dir_name, window, fiber_name in created:
        print(f"  [{split}] {i}: {dir_name} [{window}] fiber={fiber_name}")


if __name__ == "__main__":
    main()
