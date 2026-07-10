from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
from numpy import typing as npt

from analyze.data import FiberPair, Audio
from analyze.filters import bp_filter
from analyze.sot import SOTResult, SOTData, detect_ppg_beats
from constants import MATERNAL_ACOUSTIC_BAND_HZ, MATERNAL_BPM_RANGE, FETAL_BPM_RANGE, FETAL_ACOUSTIC_BAND_HZ


def fiber_beats(
        detector,
        out: Path,
        maternal_band: Tuple[float, float] = MATERNAL_ACOUSTIC_BAND_HZ,
        maternal_bpm: Tuple[float, float] = MATERNAL_BPM_RANGE,
        fetal_bpm: Tuple[float, float] = FETAL_BPM_RANGE,
):
    def run_detect_beats(data: FiberPair) -> fHROutput:
        out.mkdir(parents=True, exist_ok=True)

        if data.chest is not None:
            chest = bp_filter(data.chest, maternal_band[0], maternal_band[1])
            maternal_times = detector(chest, maternal_bpm, out, tag="maternal")["times"]
        else:
            maternal_times = None
        fetal_times = detector(data.abdomen, fetal_bpm, out, tag="fetal")["times"]

        return fHROutput(
            fetal_source=data.abdomen,
            fetal_beats=fetal_times,
            maternal_source=data.chest,
            maternal_beats=maternal_times,
        )

    run_detect_beats.__name__ = "detect_fiber_beats"
    return run_detect_beats

def multi_fiber_beats(
        detector,
        out: Path,
        maternal_band: Tuple[float, float] = MATERNAL_ACOUSTIC_BAND_HZ,
        maternal_bpm: Tuple[float, float] = MATERNAL_BPM_RANGE,
        fetal_bpm: Tuple[float, float] = FETAL_BPM_RANGE,
):
    def run_multi_detect_beats(data) -> "fHRMultiOutput":
        out.mkdir(parents=True, exist_ok=True)

        if data.chest is not None:
            chest = bp_filter(data.chest, maternal_band[0], maternal_band[1])
            maternal_times = detector(chest, maternal_bpm, out, tag="maternal")["times"]
        else:
            maternal_times = None

        fetal_beats = {
            name: detector(audio, fetal_bpm, out, tag=f"fetal_{name}")["times"]
            for name, audio in data.abdomen.items()
        }

        return fHRMultiOutput(
            fetal_sources=data.abdomen,
            fetal_beats=fetal_beats,
            maternal_source=data.chest,
            maternal_beats=maternal_times,
        )

    run_multi_detect_beats.__name__ = "multi_fiber_beats"
    return run_multi_detect_beats

def sot_beats(
        detector,
        out:Path,
        maternal_bpm: Tuple[float, float] = MATERNAL_BPM_RANGE,
        fetal_bpm: Tuple[float, float] = FETAL_BPM_RANGE,
):
    def run_detect_beats(data: SOTData) -> SOTResult:
        out.mkdir(parents=True, exist_ok=True)

        mic = bp_filter(data.mic, FETAL_ACOUSTIC_BAND_HZ[0], FETAL_ACOUSTIC_BAND_HZ[1])
        mic = detector(mic, fetal_bpm, out, tag="fetal_mic")

        maternal = None
        if data.ppg is not None:
            maternal = detect_ppg_beats(data.ppg, maternal_bpm)# detector(chest, maternal_band, out, tag="maternal")

        return SOTResult(
            ppg=data.ppg,
            mic=data.mic,
            ppg_beats=maternal["times"] if maternal is not None else None,
            mic_beats=mic["times"],
        )

    run_detect_beats.__name__ = "detect_sot_beats"
    return run_detect_beats

@dataclass
class fHROutput:
    fetal_source: Audio
    fetal_beats: npt.NDArray[np.float64]

    maternal_source: Audio
    maternal_beats: npt.NDArray[np.float64]


def phase_unwrap_beats(beats, systole_frac: float = 0.33, window: int = 9) -> np.ndarray:
    """Remove S1<->S2 phase slips from a (mostly) one-per-cycle beat train.

    A beat train that flips *which* heart sound it marks stays one-per-cycle within a
    run, but at each S1<->S2 transition emits a single inter-beat interval off by ~one
    systole -- long on S1->S2 (HR dips/lags), short on S2->S1 (HR spikes to catch up).
    This walks the train, detects those systole-sized interval jumps against the local
    cardiac period, and re-phases every following beat by the same amount so the
    intervals return to the period. Missed beats (~2x interval) and normal HRV are
    left alone. It does not need to know which sound is S1 -- it only enforces phase
    continuity, which is what fixes the HR trace.
    """
    beats = np.sort(np.asarray(beats, dtype=float))
    if len(beats) < 3:
        return beats
    ibi = np.diff(beats)
    out = np.empty_like(beats)
    out[0] = beats[0]
    offset = 0.0
    for i in range(1, len(beats)):
        a = max(0, i - 1 - window // 2)
        b = min(len(ibi), i - 1 + window // 2 + 1)
        t_local = float(np.median(ibi[a:b]))
        s = systole_frac * t_local
        dev = (beats[i] - beats[i - 1]) - t_local
        if 0.5 * s <= abs(dev) <= 1.5 * s:   # systole-sized jump => phase slip
            offset -= dev
        out[i] = beats[i] + offset
    return out


def phase_continuity(out: Path, systole_frac: float = 0.33):
    """Pipeline stage: re-phase the fetal beat train to remove S1<->S2 flip glitches.

    Runs after a beat detector (e.g. fiber_beats) and before plot_hr/evaluate_v2, so
    the HR is computed from a phase-continuous train. Saves a before/after HR plot.
    """
    def run_phase_continuity(data: fHROutput) -> fHROutput:
        out.mkdir(parents=True, exist_ok=True)
        raw = np.asarray(data.fetal_beats, dtype=float)
        fixed = phase_unwrap_beats(raw, systole_frac=systole_frac)
        _plot_phase_continuity(out, raw, fixed)
        n = int(np.sum(np.abs(np.diff(raw) - np.diff(fixed)) > 1e-6)) if len(raw) > 2 else 0
        print(f"[phase_continuity] re-phased {n} interval(s) (S1<->S2 slips)")
        return fHROutput(
            fetal_source=data.fetal_source,
            fetal_beats=fixed,
            maternal_source=data.maternal_source,
            maternal_beats=data.maternal_beats,
        )

    run_phase_continuity.__name__ = "phase_continuity"
    return run_phase_continuity


def _plot_phase_continuity(out: Path, raw, fixed) -> None:
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots(figsize=(15, 4), constrained_layout=True)
    if len(raw) > 1:
        ax.plot((raw[:-1] + raw[1:]) / 2, 60.0 / np.diff(raw), ".-", color="0.6",
                lw=0.8, ms=4, label="raw (S1/S2 flips -> lag/spike)")
    if len(fixed) > 1:
        ax.plot((fixed[:-1] + fixed[1:]) / 2, 60.0 / np.diff(fixed), ".-",
                color="tab:green", lw=1.0, ms=4, label="phase-continuous")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Instantaneous HR (bpm)")
    ax.set_title("Fetal HR: before vs after phase-continuity correction")
    ax.legend(fontsize=8)
    fig.savefig(out / "phase_continuity.png", dpi=140)
    plt.close(fig)

@dataclass
class fHRMultiOutput:
    fetal_sources: dict[str, Audio]
    fetal_beats: dict[str, npt.NDArray[np.float64]]

    maternal_source: Audio
    maternal_beats: npt.NDArray[np.float64]

    @staticmethod
    def collapse(multi: "fHRMultiOutput", key: str) -> fHROutput:
        return fHROutput(
            fetal_source=multi.fetal_sources[key],
            fetal_beats=multi.fetal_beats[key],
            maternal_source=multi.maternal_source,
            maternal_beats=multi.maternal_beats,
        )
