"""Real-time fetal heart-rate analysis for the recording app's HR panel.

Three independent HR estimates from live buffers. Every estimator returns ABSOLUTE
(system-clock) beat times in seconds, so the panel can plot them on the same
epoch-seconds axis the recording plots now use:

  * SOT      -- the microphone is the source of truth. Band-limit to the fetal
               acoustic band and run the user-selected beat detector (default v7),
               matching the offline SOT path (analyze.sot.load_sot + hr.sot_beats).
  * NeoSSNet -- one abdomen fiber (default 1B): 190-220 Hz bandpass -> NeoSSNet
               source separation (heart output) -> 190-210 Hz narrow band -> the
               same detector. Mirrors analyze.neossnet.run_neossnet_pipeline.
  * FUNet    -- all five abdomen fibers (1B,2A,2B,2C,2D) stacked as channels through
               the FUNet beat-activity model, then peak-picked. Mirrors analyze.funet.

Qt-free and importable on its own. It puts the project's ``src`` and the two
model-lib dirs on sys.path exactly as the offline code does.
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path
from typing import List, Sequence, Tuple

# --- make the project + model libs importable (same layout the offline code uses) ---
_REPO = Path(__file__).resolve().parents[3]
for _p in (_REPO / "src", _REPO / "lib" / "funet" / "src", _REPO / "lib" / "neossnet"):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

# Pin a non-GUI matplotlib backend before analyze.* imports pyplot, so it can't grab a
# Qt backend and collide with the running PyQt5 app.
import matplotlib  # noqa: E402
matplotlib.use("Agg")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.signal import find_peaks  # noqa: E402

from analyze.data import Audio  # noqa: E402
from analyze.filters import bp_filter  # noqa: E402
from analyze.util import run_neossnet, moving_average_v2  # noqa: E402
from beat_app import detectors  # noqa: E402
from constants import (  # noqa: E402
    FETAL_ACOUSTIC_BAND_HZ, FETAL_ACOUSTIC_BAND_NARROW_HZ, FETAL_BPM_RANGE,
    FETAL_MODEL_PATH, FETAL_MODEL_CFG, FUNET_CONFIG, FUNET_MODEL_PATH,
    NEOSSNET_MAX_CHUNK_SECONDS,
)

# Abdomen fibers the FUNet checkpoint was trained on (config.model.channels == 5), in
# order. All five must be live for the FUNet trace to run.
FUNET_FIBERS: List[str] = ["1B", "2A", "2B", "2C", "2D"]
# Every fiber the recording exposes, for the NeoSSNet fiber picker.
ALL_FIBERS: List[str] = ["1A", "1B", "2A", "2B", "2C", "2D"]
NEOSS_DEFAULT_FIBER = "1B"

MIC_HZ_FALLBACK = 8000.0
FIBER_HZ_FALLBACK = 5000.0
# Bandpass filtfilt needs a few periods of the lowest band edge to be stable.
_MIN_CHUNK_SECONDS = 1.0


# ---------------------------------------------------------------------------
# Detector registry (shared with the beat-marking app)
# ---------------------------------------------------------------------------

def list_detector_ids() -> List[Tuple[str, str]]:
    """``[(id, label)]`` for every discovered ``analyze.hr`` detector, version-ordered."""
    return [(d["id"], d["label"]) for d in detectors.list_detectors()]


def default_detector_id() -> str:
    ids = [i for i, _ in list_detector_ids()]
    if "v7_beat_detector" in ids:
        return "v7_beat_detector"
    return ids[-1] if ids else ""


# ---------------------------------------------------------------------------
# HR from a beat train (project convention: 60/IBI at the second beat of each pair)
# ---------------------------------------------------------------------------

def inst_hr(beat_times, band=FETAL_BPM_RANGE, smooth=False, smooth_win=10):
    """Instantaneous HR ``(time, bpm)`` from beat times, clipped to ``band``.

    Matches ``analyze.plot_hr._inst_hr_v2``: 60/IBI plotted at the second beat of each
    pair, out-of-band values dropped, and (optionally) a centred moving average.
    """
    beats = np.sort(np.asarray(beat_times, dtype=float))
    if beats.size < 2:
        return np.array([]), np.array([])
    bpm = 60.0 / np.clip(np.diff(beats), 1e-6, None)
    t = beats[1:]
    keep = (bpm >= band[0]) & (bpm <= band[1])
    bpm, t = bpm[keep], t[keep]
    if smooth and bpm.size:
        bpm = moving_average_v2(bpm, smooth_win)
    return t, bpm


def _hz_of(t: np.ndarray, fallback: float) -> float:
    if t.size >= 2:
        dt = float(np.median(np.diff(t)))
        if dt > 0:
            return 1.0 / dt
    return fallback


def _long_enough(t_abs: np.ndarray, hz: float) -> bool:
    return t_abs.size >= 2 and (t_abs[-1] - t_abs[0]) >= _MIN_CHUNK_SECONDS and t_abs.size > int(hz)


# ---------------------------------------------------------------------------
# The three beat estimators.
#
# Each takes live buffers carrying ABSOLUTE time and returns absolute beat times.
# The chunk is shifted to a local 0 origin before analysis (keeps detectors/models
# away from ~1.75e9 magnitudes and their float precision), then the origin is added
# back. Estimators never raise on empty/short input -- they return an empty array.
# ---------------------------------------------------------------------------

def sot_beats(t_abs, x, detector_id, band=FETAL_BPM_RANGE) -> np.ndarray:
    """Fetal SOT beats from the mic: fetal-band bandpass then the selected detector."""
    t_abs = np.asarray(t_abs, dtype=float)
    hz = _hz_of(t_abs, MIC_HZ_FALLBACK)
    if not _long_enough(t_abs, hz):
        return np.array([])
    t0 = t_abs[0]
    mic = Audio(t_abs - t0, hz, np.asarray(x, dtype=float))
    mic = bp_filter(mic, FETAL_ACOUSTIC_BAND_HZ[0], FETAL_ACOUSTIC_BAND_HZ[1], filter_type="cheby1")
    beats = detectors.run_detector(detector_id, mic, band)
    return np.asarray(beats, dtype=float) + t0


def _neossnet_heart(x, hz) -> np.ndarray:
    """NeoSSNet heart output for a single fiber, split into <= NEOSSNET_MAX_CHUNK_SECONDS
    pieces and run under no_grad.

    NeoSSNet's transformer has a fixed positional-encoding length (~the offline
    NEOSSNET_MAX_CHUNK_SECONDS window); feeding a longer clip makes it allocate large
    O(seq^2) attention tensors and then raise a size-mismatch. Chunking keeps every model
    call within the safe length (so a large panel `chunk` can't blow up memory or crash),
    and no_grad avoids retaining an autograd graph. Mirrors analyze.neossnet._run_neossnet_chunked.
    """
    x = np.asarray(x, dtype=float).ravel()
    hz_i = int(round(hz))
    chunk_len = int(round(NEOSSNET_MAX_CHUNK_SECONDS * hz_i))
    with torch.no_grad():
        if chunk_len <= 0 or len(x) <= chunk_len:
            heart, _lung = run_neossnet(x, hz_i, FETAL_MODEL_PATH, FETAL_MODEL_CFG)
            return heart
        hearts = []
        for start in range(0, len(x), chunk_len):
            heart, _lung = run_neossnet(x[start:start + chunk_len], hz_i,
                                        FETAL_MODEL_PATH, FETAL_MODEL_CFG)
            hearts.append(heart)
    return np.concatenate(hearts)


def neossnet_beats(t_abs, x, detector_id, band=FETAL_BPM_RANGE) -> np.ndarray:
    """Fetal beats from one abdomen fiber via NeoSSNet separation then the detector.

    190-220 Hz bandpass -> NeoSSNet (heart output) -> 190-210 Hz narrow band ->
    selected detector, matching ``analyze.neossnet.run_neossnet_pipeline`` for fiber 1B.
    """
    t_abs = np.asarray(t_abs, dtype=float)
    hz = _hz_of(t_abs, FIBER_HZ_FALLBACK)
    if not _long_enough(t_abs, hz):
        return np.array([])
    t0 = t_abs[0]
    fiber = Audio(t_abs - t0, hz, np.asarray(x, dtype=float))
    fiber = bp_filter(fiber, FETAL_ACOUSTIC_BAND_HZ[0], FETAL_ACOUSTIC_BAND_HZ[1], filter_type="butter")
    heart = _neossnet_heart(fiber.data, hz)
    heart_audio = Audio(fiber.time, hz, heart)
    heart_audio = bp_filter(heart_audio, FETAL_ACOUSTIC_BAND_NARROW_HZ[0],
                            FETAL_ACOUSTIC_BAND_NARROW_HZ[1], filter_type="butter")
    beats = detectors.run_detector(detector_id, heart_audio, band)
    return np.asarray(beats, dtype=float) + t0


class _FunetModel:
    """Lazily loads the FUNet checkpoint once (CPU) and reuses it across chunks.

    run_funet does not reload the model, so a single cached instance is safe to call
    from the analysis worker thread. A lock serialises the (thread-run) inference.
    """

    def __init__(self):
        self._model = None
        self._config = None
        self._device = None
        self._lock = threading.Lock()

    def _ensure(self):
        if self._model is None:
            import torch
            from config import load_config
            from inference import load_funet
            self._config = load_config(FUNET_CONFIG)
            self._device = torch.device("cpu")
            self._model = load_funet(self._config, FUNET_MODEL_PATH, self._device)
        return self._model, self._config, self._device

    def activity(self, stack: np.ndarray, hz: float) -> np.ndarray:
        from inference import run_funet
        with self._lock:
            model, config, device = self._ensure()
            return run_funet(np.asarray(stack, dtype=np.float32), int(round(hz)), model, config, device)


_funet = _FunetModel()


def align_fibers(fiber_series: Sequence[Tuple[np.ndarray, np.ndarray]],
                 hz: float | None = None) -> Tuple[np.ndarray, np.ndarray]:
    """Resample per-fiber ``(t_abs, x)`` series onto one uniform grid over their overlap.

    The abdomen fibers come from two PicoScopes with independent (but equal-rate) time
    bases, so FUNet's stacked channels must be aligned first. Returns
    ``(grid_t_abs, stack)`` with ``stack`` shaped ``(n_fibers, len(grid))``. Returns
    empty arrays if the fibers don't overlap.
    """
    series = [(np.asarray(t, dtype=float), np.asarray(x, dtype=float)) for t, x in fiber_series]
    if any(t.size < 2 for t, _ in series):
        return np.array([]), np.array([])
    lo = max(float(t[0]) for t, _ in series)
    hi = min(float(t[-1]) for t, _ in series)
    if not (hi > lo):
        return np.array([]), np.array([])
    if hz is None:
        hz = min(_hz_of(t, FIBER_HZ_FALLBACK) for t, _ in series)
    grid = np.arange(lo, hi, 1.0 / hz)
    if grid.size < 2:
        return np.array([]), np.array([])
    stack = np.stack([np.interp(grid, t, x) for t, x in series])
    return grid, stack


def funet_beats(fiber_series: Sequence[Tuple[np.ndarray, np.ndarray]],
                fetal_bpm=FETAL_BPM_RANGE) -> np.ndarray:
    """FUNet beat-activity peaks from the five stacked abdomen fibers (absolute times).

    ``fiber_series`` is the five ``(t_abs, x)`` pairs in FUNET_FIBERS order. Peak-picking
    matches ``analyze.funet.funet_beats`` (min spacing from the fast-rate cap, height at
    mean + 0.5 std).
    """
    grid, stack = align_fibers(fiber_series)
    hz = _hz_of(grid, FIBER_HZ_FALLBACK)
    if not _long_enough(grid, hz):
        return np.array([])
    activity = np.asarray(_funet.activity(stack, hz), dtype=float)
    n = min(activity.size, grid.size)
    activity, grid = activity[:n], grid[:n]

    min_spacing = 60.0 / fetal_bpm[1]
    distance = max(1, int(round(min_spacing * hz)))
    height = float(activity.mean() + 0.5 * activity.std())
    peaks, _ = find_peaks(activity, distance=distance, height=height)
    return grid[peaks].astype(float)
