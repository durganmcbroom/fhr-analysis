"""Simulated devices, so the rig can be exercised with no hardware attached.

These are not toys. They exist because the interesting failure modes of this app --
a five-fiber FUNet track, SOT agreement, chunk scheduling under load, the recorder's
file layout -- are otherwise only reachable in a room with a patient in it. They are
also the answer to a *partial* rig: this laptop has no working ps4000 driver, so
``sim-ps4000`` supplies 1A/1B alongside the real ps3000a and the five-fiber models
still run.

Each simulated device announces the same channel ids as the real one it stands in
for, so a track configured against the simulator runs unchanged against hardware.
The hub refuses to start a simulator whose channels are already claimed, so the two
can never be confused for each other.

All three read the same stateless :class:`Phantom`: beat times are a closed-form
function of absolute time, so independently-scheduled sources agree on where the
beats are without exchanging anything.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import numpy as np

from rtmon.sources.base import (
    KIND_AUDIO, KIND_FIBER, KIND_PPG, Channel, Probe, SampleClock, Sink, Source,
)

FIBER_HZ = 5000.0
MIC_HZ = 8000.0
PPG_HZ = 55.0
BLOCK_SECONDS = 0.1

# Fetal cardiac sounds sit in the 190-220 Hz band the fetal pipeline filters to; the
# maternal beat is an order of magnitude lower. Matching those bands is what makes the
# simulator useful -- a wideband click would be found by any detector.
FETAL_TONE_HZ = 205.0
FETAL_BURST_S = 0.030
MATERNAL_TONE_HZ = 55.0
MATERNAL_BURST_S = 0.060
S2_PHASE = 0.34          # second heart sound, as a fraction of the beat interval


class Phantom:
    """Ground truth for the simulated recording: where the beats are, as a function of
    absolute time.

    Both rates wander slowly, because a synthetic signal at a fixed BPM makes every HR
    trace look perfect and hides exactly the tracking errors this app exists to show.
    """

    FETAL_BPM = (142.0, 16.0, 71.0, 6.0, 13.0)      # base, slow amp, slow period, fast amp, fast period
    MATERNAL_BPM = (74.0, 5.0, 47.0, 2.0, 11.0)

    @staticmethod
    def _phase(t: np.ndarray, p) -> np.ndarray:
        """Cumulative beat count since the epoch -- the analytic integral of rate/60."""
        base, a1, p1, a2, p2 = p
        w1, w2 = 2 * np.pi / p1, 2 * np.pi / p2
        return (base * t - (a1 / w1) * np.cos(w1 * t) - (a2 / w2) * np.cos(w2 * t)) / 60.0

    @classmethod
    def beats(cls, t0: float, t1: float, p) -> np.ndarray:
        """Beat times in ``[t0, t1]``. Stateless, so every source sees the same beats."""
        grid = np.linspace(t0, t1, max(16, int((t1 - t0) * 200)))
        phase = cls._phase(grid, p)
        first = int(np.ceil(phase[0]))
        last = int(np.floor(phase[-1]))
        if last < first:
            return np.empty(0)
        return np.interp(np.arange(first, last + 1, dtype=np.float64), phase, grid)

    @classmethod
    def render(cls, t: np.ndarray, fetal_gain: float, maternal_gain: float,
               rng: np.random.Generator, noise: float) -> np.ndarray:
        """A channel's waveform over the sample times ``t``."""
        out = rng.standard_normal(t.shape[0]).astype(np.float32) * np.float32(noise)
        # Reach back a burst-length before the block so a sound straddling the boundary
        # is rendered in both blocks rather than clipped at the seam.
        lo, hi = float(t[0]) - 0.2, float(t[-1])
        if fetal_gain > 0:
            beats = cls.beats(lo, hi, cls.FETAL_BPM)
            ibi = 60.0 / cls.FETAL_BPM[0]
            _add_bursts(out, t, beats, FETAL_TONE_HZ, FETAL_BURST_S, fetal_gain)
            _add_bursts(out, t, beats + S2_PHASE * ibi, FETAL_TONE_HZ * 1.05,
                        FETAL_BURST_S * 0.8, fetal_gain * 0.6)
        if maternal_gain > 0:
            beats = cls.beats(lo, hi, cls.MATERNAL_BPM)
            _add_bursts(out, t, beats, MATERNAL_TONE_HZ, MATERNAL_BURST_S, maternal_gain)
        return out


def _render_pulse(t: np.ndarray, rng: np.random.Generator, noise: float,
                  ac: float = 1.0) -> np.ndarray:
    """A photoplethysmogram-shaped trace at the maternal rate.

    Not a tone burst: PPG is a slow pulsatile volume change, so the detector that reads
    it (``analyze.sot.detect_ppg_beats``) looks for prominent peaks in a smooth wave,
    not for acoustic energy. Simulating it with a burst would make that detector's job
    artificially easy and the trace unrecognisable to anyone who has seen a real one.
    """
    out = rng.standard_normal(t.shape[0]).astype(np.float32) * np.float32(noise)
    beats = Phantom.beats(float(t[0]) - 1.5, float(t[-1]), Phantom.MATERNAL_BPM)
    if beats.size < 2:
        return out
    ibi = float(np.median(np.diff(beats))) if beats.size > 2 else 60.0 / Phantom.MATERNAL_BPM[0]
    # Phase within the current beat, then a systolic upstroke plus a dicrotic bump.
    idx = np.clip(np.searchsorted(beats, t, side="right") - 1, 0, beats.size - 1)
    phase = np.clip((t - beats[idx]) / max(ibi, 1e-6), 0.0, 1.0)
    wave = (np.exp(-((phase - 0.16) / 0.11) ** 2)
            + 0.42 * np.exp(-((phase - 0.42) / 0.13) ** 2))
    return out + (ac * wave).astype(np.float32)


def _add_bursts(out: np.ndarray, t: np.ndarray, beats: np.ndarray,
                tone_hz: float, width_s: float, gain: float) -> None:
    """Add Gaussian-windowed tone bursts at ``beats`` into ``out`` in place."""
    if beats.size == 0:
        return
    # Only the samples within +/-3 sigma of a beat matter; touching just those keeps
    # this O(beats * burst) rather than O(beats * block).
    sigma = width_s / 2.5
    half = 3.0 * sigma
    hz = 1.0 / (t[1] - t[0]) if t.shape[0] > 1 else 1.0
    span = max(1, int(half * hz))
    idx = np.searchsorted(t, beats)
    for centre, i in zip(beats, idx):
        a, b = max(0, i - span), min(t.shape[0], i + span)
        if b <= a:
            continue
        dt = t[a:b] - centre
        env = np.exp(-0.5 * (dt / sigma) ** 2)
        out[a:b] += (gain * env * np.sin(2 * np.pi * tone_hz * dt)).astype(np.float32)


@dataclass
class SyntheticSource(Source):
    """A simulated device. ``gains`` maps each channel to (fetal, maternal, noise)."""

    gains: dict = field(default_factory=dict)
    pulse: bool = False        # render a PPG pulse waveform rather than sound

    def __post_init__(self):
        super().__post_init__()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._rng = np.random.default_rng(0xF17A1)

    def probe(self, deep: bool = True) -> Probe:
        return Probe(True, "always available (simulated)")

    def start(self, sink: Sink) -> None:
        if self.running:
            return
        self._sink = sink
        self._stop.clear()
        self._error = None
        self._thread = threading.Thread(target=self._run, name=f"rtmon-{self.id}", daemon=True)
        self._running.set()
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout=2.0)
        self._running.clear()

    def _run(self) -> None:
        clock = SampleClock(self.nominal_hz)
        block = max(8, int(self.nominal_hz * BLOCK_SECONDS))
        n_ch = len(self.channels)
        next_due = time.monotonic()
        try:
            while not self._stop.is_set():
                t = clock.stamp(block)
                x = np.empty((block, n_ch), dtype=np.float32)
                for i, ch in enumerate(self.channels):
                    fetal, maternal, noise = self.gains.get(ch.id, (0.5, 0.2, 0.05))
                    x[:, i] = (_render_pulse(t, self._rng, noise, maternal) if self.pulse
                               else Phantom.render(t, fetal, maternal, self._rng, noise))
                self._sink(t, x)
                # Pace against a monotonic deadline rather than sleeping a fixed
                # interval, so generation cost does not make the stream run slow.
                next_due += block / self.nominal_hz
                delay = next_due - time.monotonic()
                if delay > 0:
                    self._stop.wait(delay)
                else:
                    next_due = time.monotonic()
        except Exception as exc:  # noqa: BLE001
            self._error = f"{type(exc).__name__}: {exc}"
        finally:
            self._running.clear()


def sim_ps3000a() -> SyntheticSource:
    src = SyntheticSource(
        id="sim-ps3000a", label="Simulated PicoScope 3000A (abdomen)",
        nominal_hz=FIBER_HZ, history_seconds=90.0,
        # Abdomen fibers: the fetal sound is weak and unevenly distributed, the maternal
        # sound leaks in. 2C is the poor one, which is what makes a fiber picker matter.
        gains={"2A": (0.40, 0.25, 0.08), "2B": (0.55, 0.20, 0.07),
               "2C": (0.18, 0.30, 0.12), "2D": (0.45, 0.22, 0.09)},
    )
    src.channels = tuple(Channel(id=n, label=f"Fiber {n}", kind=KIND_FIBER, unit="V")
                         for n in ("2A", "2B", "2C", "2D"))
    return src


def sim_ps4000() -> SyntheticSource:
    src = SyntheticSource(
        id="sim-ps4000", label="Simulated PicoScope 4000 (chest)",
        nominal_hz=FIBER_HZ, history_seconds=90.0,
        gains={"1A": (0.10, 0.80, 0.08), "1B": (0.50, 0.35, 0.07)},
    )
    src.channels = tuple(Channel(id=n, label=f"Fiber {n}", kind=KIND_FIBER, unit="V")
                         for n in ("1A", "1B"))
    return src


def sim_mic() -> SyntheticSource:
    src = SyntheticSource(
        id="sim-mic", label="Simulated microphone (acoustic SOT)",
        nominal_hz=MIC_HZ, history_seconds=45.0,
        # The SOT is the clean one by definition -- that is what makes it the reference.
        gains={"MIC": (0.90, 0.15, 0.03)},
    )
    src.channels = (Channel(id="MIC", label="Microphone", kind=KIND_AUDIO),)
    return src


def sim_pvs() -> SyntheticSource:
    src = SyntheticSource(
        id="sim-pvs", label="Simulated Polar Verity Sense (PPG)",
        nominal_hz=PPG_HZ, history_seconds=300.0, pulse=True,
        # The strap reads the mother's pulse; the fetal beat does not reach it. The
        # ambient channel is the sensor's own noise reference, so it carries no pulse.
        gains={"PPG0": (0.0, 1.00, 0.04), "PPG1": (0.0, 0.85, 0.05),
               "PPG2": (0.0, 0.70, 0.06), "AMB": (0.0, 0.00, 0.08)},
    )
    from rtmon.sources.polar import PVS_CHANNELS
    src.channels = PVS_CHANNELS      # same ids, labels and notes as the real strap
    return src
