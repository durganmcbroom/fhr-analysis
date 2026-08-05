"""The acquisition-source contract.

Every input the rig can have -- each PicoScope, the microphone, the Polar PPG strap,
the synthetic generator -- is a :class:`Source`. A source is *declared* unconditionally
and *probed* at runtime; nothing is ever commented out to accommodate a machine that
lacks a device.

That is the whole point of this layer. The Qt app had ``# self.ps4000 = PicoScope(...)``
commented out because one laptop has the wrong ps4000 driver, which meant the checked-in
code no longer described the rig, and re-enabling it on a machine that *does* have the
driver was an edit rather than a click. Here ``PS4000Source`` is always registered,
:meth:`Source.probe` reports ``ok=False`` with the driver's own error text, and the UI
shows it greyed out with the reason. On a machine with working drivers the same code
probes ``ok=True`` and can be armed. No edit, no fork, no lost configuration.

A source pushes samples by calling the ``sink`` it was started with:

    sink(t, x)   # t: (m,) float64 absolute epoch seconds, x: (m, n_channels) float32

Absolute epoch time is the contract because the sources have genuinely independent
clocks (two PicoScopes, a sound card, and a BLE strap), and every downstream consumer
-- alignment, plotting, the recorder's relative-time conversion -- needs a common
reference rather than four different notions of "t=0".
"""

from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

Sink = Callable[[np.ndarray, np.ndarray], None]

# What a channel is *for*, which is what decides the processors that will accept it and
# how it is drawn. Not the device it came from -- 1B is a fiber whether it arrives from a
# ps4000 or from the synthetic generator.
KIND_FIBER = "fiber"
KIND_AUDIO = "audio"
KIND_PPG = "ppg"


@dataclass(frozen=True)
class Channel:
    id: str
    label: str
    kind: str
    unit: str = ""
    # One line on what this channel physically is, surfaced as a tooltip. Channel ids
    # are terse by necessity (they are column headers, chips and axis labels), which
    # makes them opaque to anyone who did not wire the rig.
    note: str = ""

    def to_json(self) -> dict:
        return {"id": self.id, "label": self.label, "kind": self.kind,
                "unit": self.unit, "note": self.note}


@dataclass
class Probe:
    """The answer to "can this source run here, right now?"."""

    ok: bool
    detail: str = ""
    hint: str = ""

    def to_json(self) -> dict:
        return {"ok": self.ok, "detail": self.detail, "hint": self.hint}


@dataclass
class Source(ABC):
    """Base class. Subclasses set the class-level metadata and implement the three verbs."""

    id: str = ""
    label: str = ""
    channels: Sequence[Channel] = ()
    nominal_hz: float = 0.0
    # How much raw history the ring keeps. It bounds the longest analysis chunk a
    # processor can ask for, so it is per-source: the mic runs at 44.1 kHz and 60 s of
    # it costs more than 60 s of a 5 kHz fiber.
    history_seconds: float = 60.0
    # Set by subclasses whose sample stream carries no usable device timestamp, so the
    # hub timestamps arrivals against the wall clock instead.
    describe_extra: dict = field(default_factory=dict)

    def __post_init__(self):
        self._sink: Sink | None = None
        self._running = threading.Event()
        self._error: str | None = None

    # ------------------------------------------------------------- lifecycle
    @abstractmethod
    def probe(self, deep: bool = True) -> Probe:
        """Report whether this source can be started.

        ``deep=False`` must be cheap and side-effect free (an import check). ``deep=True``
        may talk to the hardware -- open the unit, enumerate audio devices, scan for the
        strap -- and is what the UI's rescan does. Must never raise: a device that is
        missing, unpowered, or backed by the wrong driver is a normal outcome and is
        reported through the return value.
        """

    @abstractmethod
    def start(self, sink: Sink) -> None:
        """Begin streaming into ``sink``. May raise; the hub reports the failure."""

    @abstractmethod
    def stop(self) -> None:
        """Stop streaming and release the device. Must be safe to call when not running."""

    # ---------------------------------------------------------------- shared
    @property
    def running(self) -> bool:
        return self._running.is_set()

    @property
    def error(self) -> str | None:
        return self._error

    def to_json(self) -> dict:
        return {
            "id": self.id,
            "label": self.label,
            "nominal_hz": self.nominal_hz,
            "channels": [c.to_json() for c in self.channels],
            "running": self.running,
            "error": self._error,
            **self.describe_extra,
        }


class SampleClock:
    """Absolute timestamps for a device that delivers blocks of un-timestamped samples.

    Samples are numbered, so a sample's time is ``t0 + k/hz``: uniform and strictly
    increasing, which the display envelope (bucketing assumes sorted time) and the
    recorder (which derives its sample rate from the time column) both depend on.
    ``t0`` is then slewed by a small fraction of its error against the wall clock on
    every block.

    Both halves matter. A fresh ``time.time()`` per block -- what the Qt app used for
    the mic -- puts callback scheduling jitter straight into the x axis. A pure sample
    count -- what it used for the PicoScopes -- drifts without bound whenever the
    device's true rate differs from its nominal one, which is exactly how the mic and
    the scopes drifted apart there. Slewing gives the uniformity of the first and the
    long-run accuracy of the second.
    """

    __slots__ = ("hz", "slew", "_t0", "_k")

    def __init__(self, hz: float, slew: float = 0.05):
        self.hz = float(hz)
        self.slew = float(slew)
        self._t0: float | None = None
        self._k = 0

    def stamp(self, n: int) -> np.ndarray:
        """Timestamps for the next ``n`` samples."""
        now = time.time()
        if self._t0 is None:
            self._t0 = now - (n - 1) / self.hz
        t = self._t0 + (self._k + np.arange(n, dtype=np.float64)) / self.hz
        err = now - t[-1]
        if err > 1.0:
            # More than a second behind the wall clock is a stall, not drift (the
            # device stopped, or the machine slept). Re-anchor so the trace catches
            # up at once and the gap shows as a gap. Only ever forward.
            t += err
            self._t0 += err
        else:
            # Correcting the origin moves the *next* block, so a backward correction
            # larger than one sample interval makes that block start before this one
            # ended. Clamping it is what keeps time strictly increasing across block
            # boundaries: a PicoScope hands over buffered samples faster than real time
            # after any hiccup, which drives err negative and, unclamped, rewound the
            # clock by up to half a millisecond at every callback.
            self._t0 += max(self.slew * err, -0.5 / self.hz)
        self._k += n
        return t
