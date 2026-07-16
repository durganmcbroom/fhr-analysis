"""Pipeline for the FUNet beat-activity model (lib/funet).

Unlike the NeoSSNet pipeline -- which separates a heart *waveform* and then runs an
acoustic beat detector on it -- FUNet directly predicts a per-frame beat activity
from the stacked abdomen fibers. So the pipeline is: load fibers, window, run the
model to get a beat-activity envelope, then peak-pick that envelope for beat times
and heart rate. No post-model bandpass: the activity is already a clean beat signal,
not raw acoustics.
"""

import sys
from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy.signal import find_peaks

from analyze.data import Audio, FiberData, load_data, windowed, FiberPair, load_no_chest_data_FULL
from analyze.evaluate_v2 import evaluate_v2
from analyze.filters import abdomen_bp
from analyze.hr import sot_beats, fiber_beats, phase_continuity
from analyze.hr.detect_v2 import v2_beat_detector
from analyze.hr.detect_v5 import v5_beat_detector
from analyze.hr.detect_v7 import v7_beat_detector
from analyze.hr.detect_v8 import v8_beat_detector
from analyze.pipeline import Pipeline
from analyze.plot_hr import plot_hr
from analyze.sot import load_sot, load_sot_no_ppg, plot_mic, SOTResult
from constants import PROJECT_DIR, FUNET_CONFIG, FUNET_MODEL_PATH, FETAL_BPM_RANGE, FETAL_ACOUSTIC_BAND_NARROW_HZ

# lib/funet/src is a flat module dir (bare imports); put it on the path to import from.
sys.path.insert(0, str(Path(PROJECT_DIR) / "lib" / "funet" / "src"))
from config import load_config          # noqa: E402
from inference import load_funet, run_funet  # noqa: E402


def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def use_funet(out, fiber_names, config_path=FUNET_CONFIG, checkpoint=FUNET_MODEL_PATH):
    """Pipeline stage: stack the named abdomen fibers as channels, run FUNet, and
    return the beat-activity as an Audio on the fibers' own time axis.

    ``fiber_names`` selects which abdomen fibers to stack, in order -- it must match
    the count (and ideally the order) the model was trained on (config.model.channels).
    """
    config = load_config(config_path)

    def run_use_funet(data: FiberData) -> Audio:
        out.mkdir(parents=True, exist_ok=True)

        missing = [n for n in fiber_names if n not in data.abdomen]
        if missing:
            raise ValueError(f"use_funet: fibers {missing} not in data (have {list(data.abdomen)})")

        fibers = [data.abdomen[n] for n in fiber_names]
        length = min(f.data.shape[-1] for f in fibers)
        x = np.stack([np.asarray(f.data, dtype=np.float32)[:length] for f in fibers])  # (C, T)
        hz = fibers[0].hz
        time = np.asarray(fibers[0].time)[:length]

        model = load_funet(config, checkpoint, _pick_device())
        activity = run_funet(x, hz, model, config)[:length]

        _plot_activity(out, time, x, activity, fiber_names)
        return FiberPair(data.chest, Audio(time, hz, activity))

    run_use_funet.__name__ = "use_funet"
    return run_use_funet


def funet_beats(out, fetal_bpm=FETAL_BPM_RANGE):
    """Pipeline stage: peak-pick the FUNet beat-activity Audio into beat times + HR."""

    def run_funet_beats(activity: Audio) -> dict:
        out.mkdir(parents=True, exist_ok=True)

        sig = np.asarray(activity.data, dtype=float)
        # Beats can't be closer than the fastest plausible fetal rate allows.
        min_spacing = 60.0 / fetal_bpm[1]
        distance = max(1, int(round(min_spacing * activity.hz)))
        height = sig.mean() + 0.5 * sig.std()

        peaks, _ = find_peaks(sig, distance=distance, height=height)
        beat_times = np.asarray(activity.time)[peaks]

        _plot_beats(out, activity, beat_times)
        print(f"[funet] detected {len(beat_times)} beats "
              f"({_mean_bpm(beat_times):.1f} bpm mean)")
        return {"beats": beat_times, "activity": activity}

    run_funet_beats.__name__ = "funet_beats"
    return run_funet_beats


def _mean_bpm(beat_times: np.ndarray) -> float:
    if len(beat_times) < 2:
        return float("nan")
    return float(60.0 / np.mean(np.diff(beat_times)))


def _plot_activity(out: Path, time, channels, activity, fiber_names) -> None:
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(15, 6), sharex=True)
    for ch, name in zip(channels, fiber_names):
        top.plot(time, ch, lw=0.4, alpha=0.7, label=name)
    top.set_ylabel("Input fibers")
    top.legend(fontsize=7, ncol=len(fiber_names))
    top.set_title("FUNet input (stacked abdomen fibers)", fontsize=9)

    bottom.plot(time, activity, lw=0.6, color="tab:green")
    bottom.set_ylabel("Beat activity")
    bottom.set_xlabel("Time (s)")
    bottom.set_title("FUNet output: beat activity", fontsize=9)

    fig.tight_layout()
    fig.savefig(out / "funet_activity.png", dpi=150)
    plt.close(fig)
    print(f"[funet] saved activity plot -> {out / 'funet_activity.png'}")


def _plot_beats(out: Path, activity: Audio, beat_times) -> None:
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(15, 6))
    top.plot(activity.time, activity.data, lw=0.6, color="tab:green")
    for bt in beat_times:
        top.axvline(bt, color="0.3", lw=0.6, ls="--", alpha=0.6)
    top.set_ylabel("Beat activity")
    top.set_title(f"FUNet beats ({len(beat_times)} detected)", fontsize=9)

    if len(beat_times) >= 2:
        mid = (beat_times[:-1] + beat_times[1:]) / 2
        bpm = 60.0 / np.diff(beat_times)
        bottom.plot(mid, bpm, ".-", color="tab:red", lw=0.8, ms=3)
    bottom.set_ylabel("Instantaneous HR (bpm)")
    bottom.set_xlabel("Time (s)")

    fig.tight_layout()
    fig.savefig(out / "funet_beats.png", dpi=150)
    plt.close(fig)
    print(f"[funet] saved beats plot -> {out / 'funet_beats.png'}")


def run_funet_pipeline(patient, window, datadir):
    """End-to-end: load fibers -> window -> FUNet beat activity -> beats + HR plots.

    ``fiber_names`` are the abdomen fibers to feed the model, in the order it was
    trained on (e.g. ["2D", "2C", "2B"] for a 3-channel model).
    """
    out_path = Path(f"{PROJECT_DIR}.out/{patient}/funet/")
    out_path.mkdir(parents=True, exist_ok=True)

    sot_pipe = Pipeline([
        load_sot(),
        windowed(window[0], window[1]),
        sot_beats(v7_beat_detector, out_path)
    ], f"{PROJECT_DIR}/.out/cache_sot/neossnet/{patient}", play_sound=False)
    sot = sot_pipe.process(datadir)

    pipe = Pipeline([
        load_data,
        windowed(window[0], window[1]),
        use_funet(out_path, ["1B", "2A", "2B", "2C", "2D"]),
        fiber_beats(v2_beat_detector, out_path),
        plot_hr(sot, out_path),
        evaluate_v2(sot, out_path)
    ], f"{PROJECT_DIR}/.out/{patient}/funet/cache/", play_sound=False)

    return pipe.process(datadir)

def run_funet_belly_machine(
        patient,
        window,
        datadir
):
    out_path = Path(f"{PROJECT_DIR}.out/{patient}/funet/")
    out_path.mkdir(parents=True, exist_ok=True)

    # sot_beats prefers a hand-marked mic_beats.npy in datadir over the v8 detector.
    sot_pipe = Pipeline([
        load_sot_no_ppg(),
        windowed(window[0], window[1]),
        plot_mic(out_path),
        sot_beats(v7_beat_detector, out_path, data_dir=datadir)
    ], f"{PROJECT_DIR}/.out/cache_sot/funet/{patient}", play_sound=False)
    sot: SOTResult = sot_pipe.process(datadir)

    pipe = Pipeline([
        load_no_chest_data_FULL,
        windowed(window[0], window[1]),
        use_funet(out_path, ["1A", "1B", "2A", "2B", "2C"]),
        fiber_beats(v2_beat_detector, out_path),
        # phase_continuity(out_path),   # stitch S1<->S2 phase slips so HR doesn't lag/spike
        plot_hr(sot, out_path),
        evaluate_v2(sot, out_path, window_s=(window[1] - window[0])),
    ], f"{PROJECT_DIR}/.out/{patient}/funet/cache/", play_sound=False)

    pipe.process(datadir)