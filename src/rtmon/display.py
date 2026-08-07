"""Bandpassed display envelopes -- the Signals panel's second view.

A scope normally shows what the device delivered. This produces the other thing worth
looking at: what the *processing* sees. The fetal acoustic band is 190-220 Hz and the
maternal one 40-80 Hz, so on a raw 5 kHz trace the content the detectors actually work
from is a few percent of the amplitude and invisible next to everything else. Watching
it filtered is the difference between "the fiber is producing a signal" and "the fiber
is producing a signal *in the band this rig measures*", which is the question being
asked when someone repositions a sensor.

Deliberately the same filter the processors run -- Chebyshev on audio, Butterworth on a
fiber, matching :mod:`rtmon.processors` -- and reduced to the same ``(t0, dt, lo, hi)``
min/max envelope the raw view uses, so the two views are directly comparable and the
client needs no new drawing code.

Not every channel has one. Nothing in the pipeline bandpasses the PPG strap: its pulse
*is* the signal and a 40-80 Hz filter would delete it. Those channels report no band
and the panel keeps showing them raw, rather than inventing a filter to look consistent.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import resample_poly

from analyze.data import Audio
from analyze.filters import bp_filter
from rtmon.sources import KIND_AUDIO, KIND_FIBER

# Rate the filtering runs at. Everything above it is thrown away first, which is most of
# a 5 kHz fiber: filtfilt over a 60 s window at full rate, for every shown channel, on
# every frame, is real CPU on a machine that is also recording and running torch.
# 1 kHz leaves better than 2x headroom over the highest band edge (220 Hz).
WORK_HZ = 1000.0
# Below this there is nothing to gain by decimating first.
MIN_DECIMATE_RATIO = 2

# Which kinds have a band at all, and which filter the processors use on them.
_FILTER_TYPE = {KIND_AUDIO: "cheby1", KIND_FIBER: "butter"}


def has_band(kind: str) -> bool:
    """Whether a channel of this kind is bandpassed anywhere in the pipeline."""
    return kind in _FILTER_TYPE


def band_envelope(t: np.ndarray, x: np.ndarray, kind: str,
                  band: tuple[float, float], bucket_hz: float):
    """``(t0, dt, lo, hi)`` for ``x`` bandpassed to ``band``, or None.

    Same shape and same absolute time base as :meth:`rtmon.ring.EnvelopeRing.window`,
    so a filtered channel and a raw one line up bucket for bucket on the display.
    """
    if not has_band(kind):
        return None
    t = np.asarray(t, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    if t.size < 32:
        return None

    span = float(t[-1] - t[0])
    if span <= 0:
        return None
    hz = t.size / span

    low, high = float(band[0]), float(band[1])
    # A band the channel cannot represent is not an error worth raising -- a 55 Hz
    # input simply has nothing at 190 Hz. Say "no band" and let the caller show raw.
    if high >= hz / 2.0:
        return None

    # --- decimate to WORK_HZ ---------------------------------------------------
    factor = int(hz // WORK_HZ)
    if factor >= MIN_DECIMATE_RATIO:
        # The ACHIEVED factor, not the requested one: the two stages multiply out to a
        # whole number that need not equal what was asked for, and taking it on faith
        # designs the filter for a rate the data is not at. With a 5x request that
        # became 4x, the fetal band landed at 237-275 Hz -- a 205 Hz tone was rejected
        # and a 260 Hz one sailed through.
        x, factor = _decimate(x, factor)
        hz = hz / factor
        t = t[0] + np.arange(x.size) / hz
        if high >= hz / 2.0:
            return None

    try:
        filtered = bp_filter(Audio(t, int(round(hz)), x), low, high,
                             filter_type=_FILTER_TYPE[kind]).data
    except ValueError:
        return None      # window shorter than the filter's padding needs

    return _bucket(t, np.asarray(filtered, dtype=np.float32), bucket_hz)


def _decimate(x: np.ndarray, factor: int) -> tuple[np.ndarray, int]:
    """Reduce the sample rate by ``factor``, cheaply but not naively.

    Two stages, because the cost is dominated by however many samples the first stage
    has to touch and ``resample_poly``'s polyphase FIR costs about the same per *input*
    sample whatever the ratio. Measured on a 60 s window of 44.1 kHz audio, going
    straight to 1 kHz with resample_poly takes 62 ms -- at 15 frames a second that is a
    whole core for one channel. Bulk-reducing with a boxcar first and letting the FIR
    do only the last factor of two takes 2.6 ms for the same result.

    The boxcar's stopband is mediocre (roughly -20 dB where it matters), so a little
    broadband hiss folds into the displayed band. That is a real limitation and the
    reason this lives here rather than anywhere a measurement is taken from: it is an
    aid for judging sensor placement, not a signal path.
    """
    intermediate = max(1, factor // 2)
    if intermediate > 1:
        usable = (x.size // intermediate) * intermediate
        if usable < intermediate:
            return x, 1
        x = x[:usable].reshape(-1, intermediate).mean(axis=1)
    else:
        intermediate = 1
    remainder = max(1, int(round(factor / intermediate)))
    if remainder > 1 and x.size > 32:
        x = resample_poly(x, up=1, down=remainder)
    else:
        remainder = 1
    return x, intermediate * remainder


def _bucket(t: np.ndarray, x: np.ndarray, bucket_hz: float):
    """Reduce ``(t, x)`` to per-bucket min/max on the absolute bucket grid.

    Buckets are indexed off absolute time, not off the start of this window, which is
    what keeps a filtered trace pinned to the same pixels as the raw one it replaced.
    Buckets the window does not cover stay NaN -- the client already skips those.
    """
    ids = np.floor(t * bucket_hz).astype(np.int64)
    first, last = int(ids[0]), int(ids[-1])
    count = last - first + 1
    if count <= 0:
        return None

    # ids is non-decreasing, so each bucket is one contiguous run: reduceat over the
    # run starts is the whole reduction, no scatter-add needed.
    starts = np.flatnonzero(np.diff(ids, prepend=ids[0] - 1))
    lo_runs = np.fmin.reduceat(x, starts)
    hi_runs = np.fmax.reduceat(x, starts)

    lo = np.full(count, np.nan, dtype=np.float32)
    hi = np.full(count, np.nan, dtype=np.float32)
    at = ids[starts] - first
    lo[at] = lo_runs
    hi[at] = hi_runs

    dt = 1.0 / bucket_hz
    return (first + 0.5) * dt, dt, lo, hi
