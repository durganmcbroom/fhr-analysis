"""The processing side of the matrix: ways of turning live channels into beat times.

Each processor is a self-describing entry in :data:`PROCESSORS`. The description is
what the setup UI renders -- which channel kinds it will accept, how many, whether it
takes a model version and a detector -- so adding a processor here makes it
configurable in the browser with no frontend change.

Every processor returns absolute (system-clock) beat times, because that is the only
thing the four estimators have in common and the only basis on which they can be
compared. Analysis itself runs on a chunk shifted to a local zero: detectors and
models see 0..30 s rather than 1.75e9, which keeps float precision (and every
``find_peaks`` distance computation) sane.

The pipelines mirror the offline ones deliberately -- ``analyze.sot``,
``analyze.neossnet.run_neossnet_pipeline``, ``analyze.funet_runner``,
``analyze.tslnet_runner`` -- so a number seen live is the number the offline run
would produce for that window, not an approximation of it.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
import torch
from scipy.signal import butter, detrend, find_peaks, resample_poly, sosfiltfilt, welch

from analyze.constants import (
    FETAL_ACOUSTIC_BAND_HZ, FETAL_ACOUSTIC_BAND_NARROW_HZ, FETAL_BPM_RANGE,
    MATERNAL_ACOUSTIC_BAND_HZ, MATERNAL_BPM_RANGE, NEOSSNET_MAX_CHUNK_SECONDS,
    NEOSSNET_MODEL_HZ,
)
from analyze.data import Audio
from analyze.filters import bp_filter
from beat_app import detectors
from rtmon.models import ModelCache, ModelEntry, find as find_model
from rtmon.sources import KIND_AUDIO, KIND_FIBER, KIND_PPG

# Bandpass filtfilt needs several periods of the lowest band edge before it settles.
MIN_CHUNK_SECONDS = 1.0
# How many points of the beat-activity trace get shipped to the browser. Enough to see
# the peak structure the detector is working from; small enough to be free.
ACTIVITY_POINTS = 720

BANDS = {
    "fetal": {"bpm": FETAL_BPM_RANGE, "acoustic": FETAL_ACOUSTIC_BAND_HZ,
              "narrow": FETAL_ACOUSTIC_BAND_NARROW_HZ, "label": "Fetal"},
    "maternal": {"bpm": MATERNAL_BPM_RANGE, "acoustic": MATERNAL_ACOUSTIC_BAND_HZ,
                 "narrow": MATERNAL_ACOUSTIC_BAND_HZ, "label": "Maternal"},
}


def device() -> torch.device:
    """Inference device.

    CPU unless ``RTMON_DEVICE`` says otherwise -- the opposite of the training default
    on purpose. These chunks are small enough that an accelerator saves little, several
    tracks infer concurrently from a thread pool (which MPS in particular does not like),
    and a monitor that occasionally produces NaN is worse than a slightly slower one.
    """
    return torch.device(os.environ.get("RTMON_DEVICE", "cpu"))


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

@dataclass
class Context:
    """One track's inputs for one analysis cycle."""

    series: list[tuple[np.ndarray, np.ndarray]]     # per input channel, (t_abs, x)
    inputs: list[str]                               # the channel ids, same order
    band: str = "fetal"
    detector: str = "v7_beat_detector"
    model: str | None = None
    cache: ModelCache | None = None

    @property
    def bpm(self) -> tuple[float, float]:
        return BANDS[self.band]["bpm"]


@dataclass
class Result:
    beats: np.ndarray                                # absolute seconds
    window: tuple[float, float] | None = None        # the span these beats replace
    activity: tuple[np.ndarray, np.ndarray] | None = None   # (t_abs, y) for display
    note: str = ""


def _hz(t: np.ndarray, fallback: float = 5000.0) -> float:
    if t.size >= 2:
        dt = float(np.median(np.diff(t)))
        if dt > 0:
            return 1.0 / dt
    return fallback


def _usable(t: np.ndarray, hz: float) -> bool:
    return t.size >= 2 and (t[-1] - t[0]) >= MIN_CHUNK_SECONDS and t.size > int(hz)


def _decimate(t: np.ndarray, y: np.ndarray, points: int = ACTIVITY_POINTS):
    """Thin an activity trace for the wire, keeping each bucket's maximum.

    Max rather than mean: the whole point of the trace is where its peaks are, and
    averaging is exactly the operation that removes them.
    """
    n = t.size
    if n <= points:
        return t.astype(np.float64), y.astype(np.float32)
    edges = np.linspace(0, n, points + 1).astype(int)
    starts = edges[:-1]
    keep = starts < edges[1:]
    starts = starts[keep]
    return t[starts].astype(np.float64), np.maximum.reduceat(y, starts).astype(np.float32)


def align(series: Sequence[tuple[np.ndarray, np.ndarray]], hz: float | None = None):
    """Resample per-channel ``(t_abs, x)`` onto one uniform grid over their overlap.

    The abdomen fibers arrive from two PicoScopes with independent time bases, so a
    multi-channel model cannot simply stack the raw arrays -- channel k would be offset
    from channel 0 by however far the two clocks have drifted. Returns
    ``(grid, stack)`` with ``stack`` shaped ``(channels, len(grid))``.
    """
    prepared = [(np.asarray(t, np.float64), np.asarray(x, np.float64)) for t, x in series]
    if any(t.size < 2 for t, _ in prepared):
        return np.array([]), np.array([])
    lo = max(float(t[0]) for t, _ in prepared)
    hi = min(float(t[-1]) for t, _ in prepared)
    if hi <= lo:
        return np.array([]), np.array([])
    if hz is None:
        hz = min(_hz(t) for t, _ in prepared)
    grid = np.arange(lo, hi, 1.0 / hz)
    if grid.size < 2:
        return np.array([]), np.array([])
    return grid, np.stack([np.interp(grid, t, x) for t, x in prepared])


def activity_beats(activity: np.ndarray, grid: np.ndarray, bpm) -> np.ndarray:
    """Peak-pick a beat-activity signal. Matches ``analyze.funet_runner.funet_beats``."""
    distance = max(1, int(round(60.0 / bpm[1] * _hz(grid))))
    height = float(activity.mean() + 0.5 * activity.std())
    peaks, _ = find_peaks(activity, distance=distance, height=height)
    return grid[peaks].astype(float)


# ---------------------------------------------------------------------------
# Acoustic: bandpass + a detector. This is the SOT path.
# ---------------------------------------------------------------------------

def run_acoustic(ctx: Context) -> Result:
    t, x = ctx.series[0]
    hz = _hz(t, 8000.0)
    if not _usable(t, hz):
        return Result(np.array([]))
    t0 = float(t[0])
    band = BANDS[ctx.band]["acoustic"]
    audio = Audio(t - t0, hz, np.asarray(x, dtype=float))
    audio = bp_filter(audio, band[0], band[1], filter_type="cheby1")
    beats = detectors.run_detector(ctx.detector, audio, ctx.bpm)
    return Result(np.asarray(beats, dtype=float) + t0, window=(t0, float(t[-1])))


# ---------------------------------------------------------------------------
# NeoSSNet / tune-ssnet: separate the heart source out of one fiber, then detect.
# ---------------------------------------------------------------------------

_SSNET_LOCK = threading.Lock()


def _load_ssnet(entry: ModelEntry):
    from utils import load_model      # top-level package from the lib/neossnet submodule
    model = load_model(entry.checkpoint, entry.config, device())
    model.eval()
    return model


def _ssnet_heart(model, x: np.ndarray, src_hz: float) -> np.ndarray:
    """Heart output for one fiber.

    Same normalise/resample/restore contract as ``analyze.util.run_neossnet``, but
    against an already-loaded model: that helper reaches ``neossnet.utils.generate_output``,
    which takes checkpoint *paths* and therefore rebuilds the network on every call --
    fine once per offline run, ruinous several times a minute.

    Long inputs are split, because the separator's transformer has a fixed positional
    encoding: past its trained window it allocates an O(seq^2) attention tensor and
    then raises a size mismatch (see ``analyze.neossnet._run_neossnet_chunked``).
    """
    x = np.asarray(x, dtype=float).ravel()
    n = x.size
    peak = float(np.max(np.abs(x))) + 1e-12
    src_hz_i = int(round(src_hz))
    g = np.gcd(NEOSSNET_MODEL_HZ, src_hz_i)
    up, down = NEOSSNET_MODEL_HZ // g, src_hz_i // g

    chunk = int(round(NEOSSNET_MAX_CHUNK_SECONDS * src_hz_i))
    pieces = [x[i:i + chunk] for i in range(0, n, chunk)] if 0 < chunk < n else [x]

    out = []
    with torch.no_grad():
        for piece in pieces:
            scaled = piece / peak
            model_in = resample_poly(scaled, up, down) if up != down else scaled
            tensor = torch.tensor(model_in, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            heart = model(tensor)[0, 0, :].detach().cpu().numpy()
            native = resample_poly(heart, down, up) if up != down else heart
            out.append(native[:piece.size] if native.size >= piece.size
                       else np.pad(native, (0, piece.size - native.size)))
    return np.concatenate(out) * peak


def run_ssnet(ctx: Context) -> Result:
    t, x = ctx.series[0]
    hz = _hz(t)
    if not _usable(t, hz):
        return Result(np.array([]))
    t0 = float(t[0])
    wide = BANDS[ctx.band]["acoustic"]
    narrow = BANDS[ctx.band]["narrow"]

    fiber = Audio(t - t0, hz, np.asarray(x, dtype=float))
    fiber = bp_filter(fiber, wide[0], wide[1], filter_type="butter")

    model = ctx.cache.get("ssnet", ctx.model, _load_ssnet)
    with _SSNET_LOCK:      # MaskNet is not safe to call from two threads at once
        heart = _ssnet_heart(model, fiber.data, hz)

    separated = bp_filter(Audio(fiber.time, hz, heart), narrow[0], narrow[1], filter_type="butter")
    beats = detectors.run_detector(ctx.detector, separated, ctx.bpm)
    at, ay = _decimate(fiber.time + t0, np.abs(separated.data).astype(np.float32))
    return Result(np.asarray(beats, dtype=float) + t0, window=(t0, float(t[-1])),
                  activity=(at, ay))


# ---------------------------------------------------------------------------
# FUNet / TSLNet: stacked fibers straight to a beat-activity signal.
# ---------------------------------------------------------------------------

_ACTIVITY_LOCKS: dict[str, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()


def _lock_for(key: str) -> threading.Lock:
    with _LOCKS_GUARD:
        return _ACTIVITY_LOCKS.setdefault(key, threading.Lock())


def _run_activity_model(ctx: Context, family: str) -> Result:
    grid, stack = align(ctx.series)
    hz = _hz(grid)
    if not _usable(grid, hz):
        return Result(np.array([]))

    if family == "funet":
        from funet.config import load_config
        from funet.inference import load_funet, run_funet
        build = lambda e: load_funet(load_config(e.config), e.checkpoint, device())  # noqa: E731
        infer = run_funet
    else:
        from tslnet.config import load_config
        from tslnet.inference import load_tslnet, run_tslnet
        build = lambda e: load_tslnet(load_config(e.config), e.checkpoint, device())  # noqa: E731
        infer = run_tslnet

    entry = find_model(family, ctx.model)
    if entry is None:
        raise KeyError(f"no {family} model named {ctx.model!r}")
    if stack.shape[0] != entry.channels:
        raise ValueError(
            f"{ctx.model} expects {entry.channels} fiber(s), got {stack.shape[0]} "
            f"({', '.join(ctx.inputs)})")

    model = ctx.cache.get(family, ctx.model, build)
    config = load_config(entry.config)
    with _lock_for(f"{family}:{ctx.model}"):
        activity = np.asarray(
            infer(stack.astype(np.float32), int(round(hz)), model, config, device()), dtype=float)

    n = min(activity.size, grid.size)
    activity, grid = activity[:n], grid[:n]
    beats = activity_beats(activity, grid, ctx.bpm)
    at, ay = _decimate(grid, activity.astype(np.float32))
    return Result(beats, window=(float(grid[0]), float(grid[-1])), activity=(at, ay))


def run_funet(ctx: Context) -> Result:
    return _run_activity_model(ctx, "funet")


def run_tslnet(ctx: Context) -> Result:
    return _run_activity_model(ctx, "tslnet")


# ---------------------------------------------------------------------------
# PPG: the strap's own pulse, as a maternal reference.
# ---------------------------------------------------------------------------

# Minimum band-limited autocorrelation for a PPG trace to count as a real pulse.
# Measured separation on this rig is wide: unworn cases (white noise, DC-offset noise,
# a dark sensor, slow room-light drift, mains flicker) all score <= 0.23, while a worn
# strap scores >= 0.76 even with heavy added noise.
PPG_MIN_PERIODICITY = 0.40
PPG_MIN_BEATS = 5


def ppg_periodicity(x: np.ndarray, hz: float, bpm_range) -> float:
    """Peak normalised autocorrelation of ``x`` within the plausible IBI lag range.

    NOTE: this is a *measurement on a throwaway copy*, not a filter on the signal. The
    PPG that reaches ``detect_ppg_beats`` -- and everything that is recorded -- is the
    raw strap output, unfiltered. Nothing downstream ever sees the band-limited copy
    made here.

    The band-limiting is load-bearing for the measurement, though, and the reason is
    not obvious: any *periodic* interferer autocorrelates near-perfectly at beat-range
    lags. Mains-frequency light flicker at 12 Hz has period 0.083 s, so at a lag of
    0.5 s it lines up with itself exactly. Measured on this rig, an unworn strap under
    flicker scores 0.97 unfiltered, 0.96 with only a high-pass, and 0.89 keeping
    harmonics -- all *above* the worst genuine pulse (0.27 when noisy). Restricted to
    the cardiac fundamental it scores 0.02, and the worst genuine pulse scores 0.77.
    That is the only variant with any separation at all, so that is the one used.
    """
    x = np.asarray(x, dtype=float)
    if x.size < 32 or float(np.ptp(x)) == 0.0:
        return 0.0
    lo_hz = bpm_range[0] / 60.0
    hi_hz = min(bpm_range[1] / 60.0, hz / 2.0 * 0.9)
    if not (0 < lo_hz < hi_hz):
        return 0.0
    sos = butter(2, [lo_hz, hi_hz], fs=hz, btype="bandpass", output="sos")
    band = sosfiltfilt(sos, x - np.mean(x))

    n = band.size
    ac = np.correlate(band, band, mode="full")[n - 1:]
    if ac[0] <= 0:
        return 0.0
    ac = ac / ac[0]
    lo = max(1, int(hz * 60.0 / bpm_range[1]))
    hi = min(ac.size - 1, int(hz * 60.0 / bpm_range[0]))
    return float(np.max(ac[lo:hi + 1])) if hi > lo else 0.0


def ppg_beats(x: np.ndarray, t: np.ndarray, hz: float, bpm_range) -> np.ndarray:
    """Systolic peak times from a PPG trace.

    ``analyze.sot.detect_ppg_beats`` is not used here, for a reason measured on a real
    183 s session whose true rate (spectral, 58.5 bpm) was known independently:

      * fed the RAW trace, it found 103 of ~179 beats. Its prominence test is a fixed
        fraction of the whole window's standard deviation, so the strap's baseline
        wander (~9000 counts of drift here) swamps it and beats vanish wherever the
        baseline sags -- one 30 s block yielded 16 beats of an expected 29.
      * fed a band-limited trace, it found 291 -- it then counts the dicrotic notch as
        a second beat, because its refractory period is derived from the top of the BPM
        range (0.32 s, i.e. 187 bpm) rather than from the rate actually present. At a
        real 58 bpm the notch lands ~0.4 s after systole, well clear of that.

    Both failures come from thresholds fixed against the *range* instead of the signal.
    So: band-limit a copy (the recorded data is untouched), find the rate present, and
    set the refractory from it. On the good-signal blocks of that session this returns
    30/29/31/30 beats at 58.0/59.0/61.8/58.0 bpm against a truth of 58/58/66/58.
    """
    x = np.asarray(x, dtype=float)
    if x.size < 32:
        return np.array([])
    lo_hz = bpm_range[0] / 60.0
    hi_hz = min(bpm_range[1] / 60.0 * 2.5, hz / 2.0 * 0.9)   # keep the upstroke's harmonics
    if not (0 < lo_hz < hi_hz):
        return np.array([])
    sos = butter(2, [lo_hz, hi_hz], fs=hz, btype="bandpass", output="sos")
    band = sosfiltfilt(sos, detrend(x))

    rate = _dominant_rate(band, hz, bpm_range)
    if rate <= 0:
        return np.array([])
    # 0.6 of the expected interval: long enough to reject the dicrotic notch (which
    # falls at ~0.3-0.4 of the cycle), short enough to let the rate genuinely vary.
    distance = max(1, int(round(0.6 / rate * hz)))
    band = band / (np.std(band) + 1e-12)

    # Prefer the most selective prominence that still explains the rate the spectrum
    # found -- an adaptive threshold, since no single value survives both a clean pulse
    # and a weak one.
    best, best_score = None, np.inf
    for prominence in (0.8, 0.6, 0.45, 0.35, 0.25):
        peaks, _ = find_peaks(band, distance=distance, prominence=prominence)
        if peaks.size < 3:
            continue
        ibi = np.diff(t[peaks])
        score = (abs(np.median(60.0 / ibi) - rate * 60.0) / (rate * 60.0)
                 + np.std(ibi) / np.mean(ibi))
        if score < best_score:
            best, best_score = peaks, score
    return t[best] if best is not None else np.array([])


def _dominant_rate(band: np.ndarray, hz: float, bpm_range) -> float:
    """Pulse rate present in ``band``, in Hz, from its power spectrum."""
    nperseg = int(min(band.size, max(hz * 8, 64)))
    freqs, power = welch(band, fs=hz, nperseg=nperseg)
    keep = (freqs >= bpm_range[0] / 60.0) & (freqs <= bpm_range[1] / 60.0)
    return float(freqs[keep][np.argmax(power[keep])]) if keep.any() else 0.0


def run_ppg(ctx: Context) -> Result:
    t, x = ctx.series[0]
    hz = _hz(t, 55.0)
    if t.size < 8:
        return Result(np.array([]))
    t0 = float(t[0])
    x = np.asarray(x, dtype=float)
    window = (t0, float(t[-1]))

    # A strap sitting on the desk still sees ambient light, and a peak picker will
    # dutifully find "beats" in it -- which plotted a confident heart rate for a strap
    # attached to nobody. Its own minimum-spacing constraint even makes those peaks
    # look regular, so inter-beat variability cannot tell the two apart; periodicity
    # of the underlying waveform can. Report nothing rather than a fiction.
    strength = ppg_periodicity(x, hz, ctx.bpm)
    if strength < PPG_MIN_PERIODICITY:
        return Result(np.array([]), window=window,
                      note=f"no pulse detected — strap not worn? (periodicity {strength:.2f})")

    beats = ppg_beats(x, t - t0, hz, ctx.bpm)
    if beats.size < PPG_MIN_BEATS:
        return Result(np.array([]), window=window)
    return Result(beats + t0, window=window, note=f"periodicity {strength:.2f}")


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ProcessorDef:
    id: str
    label: str
    description: str
    run: Callable[[Context], Result]
    kinds: tuple[str, ...]           # channel kinds this will accept as input
    arity: str                       # "one" | "model" (count comes from the checkpoint)
    family: str | None = None        # model family, if it takes one
    detector: bool = False           # takes a beat detector
    default_chunk: float = 10.0
    tags: tuple[str, ...] = field(default_factory=tuple)

    def to_json(self) -> dict:
        return {"id": self.id, "label": self.label, "description": self.description,
                "kinds": list(self.kinds), "arity": self.arity, "family": self.family,
                "detector": self.detector, "default_chunk": self.default_chunk,
                "tags": list(self.tags)}


PROCESSORS: dict[str, ProcessorDef] = {
    p.id: p for p in [
        ProcessorDef(
            id="acoustic", label="Acoustic + detector",
            description="Bandpass to the cardiac band, then a beat detector. The "
                        "microphone path is the source of truth.",
            run=run_acoustic, kinds=(KIND_AUDIO, KIND_FIBER), arity="one",
            detector=True, default_chunk=10.0, tags=("sot",),
        ),
        ProcessorDef(
            id="ssnet", label="NeoSSNet separation + detector",
            description="One fiber through a fine-tuned NeoSSNet to isolate the heart "
                        "source, then a beat detector on the separated signal.",
            run=run_ssnet, kinds=(KIND_FIBER,), arity="one",
            family="ssnet", detector=True, default_chunk=10.0,
        ),
        ProcessorDef(
            id="funet", label="FUNet beat activity",
            description="Stacked fibers through the spectrogram U-Net, then peak-pick "
                        "the beat-activity envelope.",
            run=run_funet, kinds=(KIND_FIBER,), arity="model",
            family="funet", default_chunk=10.0,
        ),
        ProcessorDef(
            id="tslnet", label="TSLNet beat activity",
            description="Stacked fibers through the frozen TimesFM head, then peak-pick "
                        "the beat-activity envelope.",
            run=run_tslnet, kinds=(KIND_FIBER,), arity="model",
            family="tslnet", default_chunk=10.0,
        ),
        ProcessorDef(
            id="ppg", label="PPG pulse peaks",
            description="Peak-pick the Polar strap's PPG. Maternal reference.",
            run=run_ppg, kinds=(KIND_PPG,), arity="one", default_chunk=15.0,
        ),
    ]
}


def describe() -> dict:
    """Everything the setup UI needs to render the processor column of the matrix."""
    return {
        "processors": [p.to_json() for p in PROCESSORS.values()],
        "detectors": detectors.list_detectors(),
        "bands": [{"id": k, "label": v["label"], "bpm": list(v["bpm"])} for k, v in BANDS.items()],
    }
