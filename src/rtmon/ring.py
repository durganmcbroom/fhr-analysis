"""Preallocated ring buffers for live acquisition.

Two rings per source, serving the two very different readers:

  * :class:`SignalRing` -- the raw samples, float32, fixed capacity. Read by the
    processors (which need a contiguous chunk) and by the recorder.
  * :class:`EnvelopeRing` -- a min/max envelope at a fixed bucket rate, kept for a
    much longer history. Read by the browser.

The split is what makes rendering cheap. Decimation happens *once per sample*, on
the acquisition push, instead of once per rendered frame over the whole visible
window: a 10 s / 6-fiber view costs a slice of a few thousand floats rather than a
1.2 MB reduction every frame.

Both rings are allocated once and never grow. The Qt app they replace appended with
``np.concatenate`` on every device callback (quadratic over a session) and
preallocated the *entire* capture up front -- ``numBuffersToCapture * sizeOfOneBuffer``
float64 per channel, ~430 MB across the two PicoScopes, which also capped a recording
at 30 minutes. Here the memory is O(history), the recording length is unbounded, and
samples are float32 (the ADC delivers int16, so float64 stored no extra information).
"""

from __future__ import annotations

import math
import threading

import numpy as np


class SignalRing:
    """Fixed-capacity ring of ``(time, channels)`` samples.

    Time is float64 -- these are absolute epoch seconds (~1.75e9), where float32
    would quantise to ~128 s steps. Sample data is float32.
    """

    __slots__ = ("channels", "capacity", "_t", "_x", "_w", "_total", "_hz", "_lock")

    def __init__(self, channels: int, capacity: int):
        self.channels = int(channels)
        self.capacity = int(capacity)
        self._t = np.zeros(self.capacity, dtype=np.float64)
        self._x = np.zeros((self.capacity, self.channels), dtype=np.float32)
        self._w = 0          # write cursor (physical index of the next slot)
        self._total = 0      # samples ever written (saturating at >= capacity for fill checks)
        self._hz = 0.0       # measured sample rate, EWMA over pushes
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ write
    def push(self, t: np.ndarray, x: np.ndarray) -> None:
        """Append ``m`` samples. ``t`` is ``(m,)`` absolute seconds, ``x`` is ``(m, channels)``."""
        t = np.asarray(t, dtype=np.float64)
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x[:, None]
        m = t.shape[0]
        if m == 0:
            return
        if m > self.capacity:
            # A single push larger than the whole ring: only the newest tail can survive.
            t, x, m = t[-self.capacity:], x[-self.capacity:], self.capacity

        if m >= 2:
            dt = float(np.median(np.diff(t)))
            if dt > 0:
                hz = 1.0 / dt
                # EWMA so a momentary scheduling hiccup doesn't reset the rate estimate.
                self._hz = hz if self._hz == 0.0 else 0.9 * self._hz + 0.1 * hz

        with self._lock:
            w = self._w
            end = w + m
            if end <= self.capacity:
                self._t[w:end] = t
                self._x[w:end] = x
                self._w = end % self.capacity
            else:
                head = self.capacity - w
                self._t[w:] = t[:head]
                self._x[w:] = x[:head]
                tail = m - head
                self._t[:tail] = t[head:]
                self._x[:tail] = x[head:]
                self._w = tail
            self._total += m

    # ------------------------------------------------------------------- read
    @property
    def hz(self) -> float:
        return self._hz

    @property
    def filled(self) -> int:
        return min(self._total, self.capacity)

    def latest_time(self) -> float:
        """Absolute time of the newest sample, or 0.0 if empty."""
        with self._lock:
            if self._total == 0:
                return 0.0
            return float(self._t[(self._w - 1) % self.capacity])

    def snapshot(self, seconds: float, cols: slice | int | None = None):
        """Newest ``seconds`` of history as contiguous copies ``(t, x)``.

        Returns ``None`` if fewer than two samples are available. ``cols`` selects a
        subset of channels *before* the copy, so a single-fiber processor never pays
        for the other five.
        """
        with self._lock:
            n = min(self._total, self.capacity)
            if n < 2:
                return None
            # Over-read slightly against the measured rate, then trim by time below.
            # Cheaper than making the ring searchable, and never under-reads.
            hz = self._hz if self._hz > 0 else 1.0
            k = int(min(n, math.ceil(seconds * hz) + 8))
            start = (self._w - k) % self.capacity
            xs = self._x if cols is None else self._x[:, cols]
            if start + k <= self.capacity:
                t = self._t[start:start + k].copy()
                x = np.array(xs[start:start + k], copy=True)
            else:
                head = self.capacity - start
                t = np.concatenate((self._t[start:], self._t[:k - head]))
                x = np.concatenate((xs[start:], xs[:k - head]))

        cut = t[-1] - seconds
        i = int(np.searchsorted(t, cut, side="left"))
        if t.shape[0] - i < 2:
            return None
        return t[i:], x[i:]


# Nothing is drawn finer than this, so nothing finer is kept for drawing. 300 buckets/s
# over a 1200 px canvas stays oversampled down to a 4 s window; below that the renderer
# asks for raw samples instead (see server._wave_frame).
DISPLAY_BUCKET_HZ = 300.0


class EnvelopeRing:
    """Min/max envelope of a multi-channel signal at a fixed bucket rate.

    One bucket holds the extremes of every sample that fell inside it, which is the
    only faithful way to shrink a waveform for the screen: plain subsampling drops
    the peaks that carry the beat. Buckets are indexed off absolute time
    (``floor(t * bucket_hz)``) so two sources with independent clocks land on the
    same grid and gaps are explicit rather than silently closed up.
    """

    __slots__ = ("channels", "bucket_hz", "buckets", "_lo", "_hi", "_last", "_lock")

    def __init__(self, channels: int, seconds: float, bucket_hz: float = DISPLAY_BUCKET_HZ):
        self.channels = int(channels)
        self.bucket_hz = float(bucket_hz)
        self.buckets = max(2, int(round(seconds * bucket_hz)))
        self._lo = np.full((self.buckets, self.channels), np.nan, dtype=np.float32)
        self._hi = np.full((self.buckets, self.channels), np.nan, dtype=np.float32)
        self._last = -1      # newest absolute bucket id written
        self._lock = threading.Lock()

    def push(self, t: np.ndarray, x: np.ndarray) -> None:
        t = np.asarray(t, dtype=np.float64)
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x[:, None]
        if t.shape[0] == 0:
            return

        b = np.floor(t * self.bucket_hz).astype(np.int64)
        # t is non-decreasing, so unique() gives each bucket's first sample index and
        # reduceat can collapse the block in one pass.
        uniq, starts = np.unique(b, return_index=True)
        lo = np.minimum.reduceat(x, starts, axis=0)
        hi = np.maximum.reduceat(x, starts, axis=0)

        with self._lock:
            first = int(uniq[0])
            if self._last >= 0 and first > self._last + 1:
                self._clear_range(self._last + 1, first - 1)
            merge = self._last >= 0 and first == self._last

            slots = np.mod(uniq, self.buckets)
            if merge:
                # The previous push ended mid-bucket; fold the new samples into it.
                s0 = slots[0]
                self._lo[s0] = np.fmin(self._lo[s0], lo[0])
                self._hi[s0] = np.fmax(self._hi[s0], hi[0])
                slots, lo, hi = slots[1:], lo[1:], hi[1:]
            if slots.size:
                self._lo[slots] = lo
                self._hi[slots] = hi
            self._last = max(self._last, int(uniq[-1]))

    def _clear_range(self, lo_id: int, hi_id: int) -> None:
        """Mark buckets ``[lo_id, hi_id]`` as "no data" (a real acquisition gap)."""
        span = hi_id - lo_id + 1
        if span >= self.buckets:
            self._lo[:] = np.nan
            self._hi[:] = np.nan
            return
        a = lo_id % self.buckets
        b = hi_id % self.buckets
        if a <= b:
            self._lo[a:b + 1] = np.nan
            self._hi[a:b + 1] = np.nan
        else:
            self._lo[a:] = np.nan
            self._hi[a:] = np.nan
            self._lo[:b + 1] = np.nan
            self._hi[:b + 1] = np.nan

    def window(self, t_end: float, seconds: float, col: int = 0):
        """``(t0, dt, lo, hi)`` for the ``seconds`` ending at ``t_end``.

        ``lo``/``hi`` are float32 with NaN where the device produced nothing. ``t0`` is
        the centre time of the first bucket and ``dt`` the bucket period, so the caller
        never has to ship an x array.
        """
        with self._lock:
            if self._last < 0:
                return None
            end_id = min(int(math.floor(t_end * self.bucket_hz)), self._last)
            count = max(2, int(round(seconds * self.bucket_hz)))
            count = min(count, self.buckets)
            start_id = end_id - count + 1

            a = start_id % self.buckets
            if a + count <= self.buckets:
                lo = self._lo[a:a + count, col].copy()
                hi = self._hi[a:a + count, col].copy()
            else:
                head = self.buckets - a
                lo = np.concatenate((self._lo[a:, col], self._lo[:count - head, col]))
                hi = np.concatenate((self._hi[a:, col], self._hi[:count - head, col]))

        dt = 1.0 / self.bucket_hz
        return (start_id + 0.5) * dt, dt, lo, hi
