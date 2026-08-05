"""The browser wire format: one WebSocket message per rendered frame.

A frame is a JSON header describing a set of arrays, followed by those arrays
concatenated as raw little-endian float32::

    [uint32 header_len][header JSON][payload]

Everything the display needs is float32 with time expressed *relative to the frame's
``now``*: at epoch magnitudes float32 would quantise to ~128 s, but over the few
minutes a view spans it resolves to tens of microseconds, which is far finer than a
pixel. That one change is most of why this is cheap -- the equivalent JSON, at 15
frames a second across seven channels, is roughly two orders of magnitude more bytes
and has to be parsed rather than viewed.

Waveforms are sent as a min/max envelope (two values per bucket) rather than as
samples, so a 10 s window costs the same whether the fiber runs at 5 kHz or 50.
"""

from __future__ import annotations

import json
import struct

import numpy as np


class FrameBuilder:
    """Accumulates arrays into one payload and hands back the framed message."""

    def __init__(self, now: float, kind: str = "frame"):
        self.header: dict = {"type": kind, "now": now}
        self._blocks: list[bytes] = []
        self._offset = 0

    def add(self, values: np.ndarray) -> dict:
        """Append an array; returns ``{"off", "n"}`` for the header to reference."""
        buf = np.ascontiguousarray(values, dtype=np.float32)
        raw = buf.tobytes()
        entry = {"off": self._offset, "n": int(buf.size)}
        self._blocks.append(raw)
        self._offset += len(raw)
        return entry

    def build(self) -> bytes:
        head = json.dumps(self.header, separators=(",", ":")).encode("utf-8")
        # Pad the header to a 4-byte boundary. Every payload offset is then a multiple
        # of 4, which is what lets the browser wrap the payload in a Float32Array view
        # directly -- an unaligned byteOffset makes that constructor throw, and copying
        # to realign would undo the point of sending binary.
        pad = (-len(head)) % 4
        if pad:
            head += b" " * pad
        return b"".join([struct.pack("<I", len(head)), head, *self._blocks])


def wave_entry(builder: FrameBuilder, channel_id: str, envelope, now: float,
               max_buckets: int = 0) -> dict | None:
    """Header entry for one channel's min/max envelope.

    ``envelope`` is what :meth:`rtmon.ring.EnvelopeRing.window` returns. The client
    reconstructs x from ``t0`` and ``dt`` and draws one vertical span per bucket.

    When the envelope holds more buckets than the canvas has pixels (``max_buckets``),
    it is re-bucketed *before* being added to the payload -- collapsing min-of-mins and
    max-of-maxes, so the extremes survive a second reduction just as they survived the
    first. Thinning after the fact would leave the discarded full-resolution arrays in
    the frame, which is the cost this is here to avoid.
    """
    if envelope is None:
        return None
    t0, dt, lo, hi = envelope
    n = lo.size
    if n == 0:
        return None

    if max_buckets and n > max_buckets:
        edges = np.linspace(0, n, max_buckets + 1).astype(int)
        starts = edges[:-1]
        starts = starts[starts < edges[1:]]
        lo = np.fmin.reduceat(lo, starts)
        hi = np.fmax.reduceat(hi, starts)
        dt = dt * n / max(1, starts.size)

    return {
        "ch": channel_id,
        "t0": round(t0 - now, 6),
        "dt": dt,
        "lo": builder.add(lo),
        "hi": builder.add(hi),
    }


def series_entry(builder: FrameBuilder, t: np.ndarray, y: np.ndarray, now: float) -> dict:
    """Header entry for an irregular ``(t, y)`` series (an HR trace, a beat train)."""
    return {"t": builder.add(np.asarray(t, dtype=np.float64) - now),
            "y": builder.add(y)}
