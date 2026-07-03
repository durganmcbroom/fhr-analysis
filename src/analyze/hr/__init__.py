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
        maternal = detect_ppg_beats(data.ppg, maternal_bpm)# detector(chest, maternal_band, out, tag="maternal")
        mic = detector(mic, fetal_bpm, out, tag="fetal_mic")

        return SOTResult(
            ppg=data.ppg,
            mic=data.mic,
            ppg_beats=maternal["times"],
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
