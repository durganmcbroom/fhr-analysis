"""Tap alignment: put every input on the fiber's clock.

Three devices, three different routes into the machine, three different delays:

    fiber  PicoScope over USB  -- the reference
    mic    sound card, driver + buffer latency
    strap  Polar over BLE, batched, with a free-running clock

**The fiber is ground truth.** Not because its timestamps are perfect, but because it
is the shortest, most direct path -- a USB oscilloscope streaming into a
wall-clock-disciplined :class:`~rtmon.sources.base.SampleClock`, with no radio, no
audio stack and no device clock of its own to drift. Everything else is corrected
onto it. That also makes the corrections meaningful in analysis terms: the fibers are
what the models run on, so aligning the references to the fibers is aligning them to
the measurement.

One knock, struck across all three sensors at once, is recorded three times; the
differences between those three recorded times are the two corrections. The procedure:

    1. wait until every channel is *steady*, and record each one's baseline level
    2. "tap all three now"
    3. locate the impulse in each, and measure each target against the fiber

It is opt-in, and the result is offered for confirmation rather than applied behind
the operator's back.

Two measurement choices, both forced by evidence rather than taste (details at the
functions):

* **Leading edges, not waveform shapes.** A knock is a sharp acoustic burst to a
  microphone and a slow pressure deflection to a photodiode; correlating those shapes
  measures the difference in sensor response time along with the clock offset. Shape
  correlation still runs, as a cross-check that both channels caught the same event.
* **One quantity throughout** -- the peak of the *transient* envelope -- so the gates
  that recognise a tap and the estimator that times it are looking at the same thing.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import numpy as np

from rtmon.sources.base import KIND_FIBER

# Common grid both envelopes are resampled onto before correlating. 200 Hz is far finer
# than the 55 Hz PPG can actually resolve; it just avoids quantising the answer.
GRID_HZ = 200.0
MAX_LAG_S = 2.0             # search range; the expected error is well inside this
# Short, because the leading EDGE is what carries the timing and smoothing blunts it.
ENVELOPE_SMOOTH_S = 0.02
# Fraction of each envelope's rise at which its onset is called. Low, because a
# threshold is crossed later the slower the sensor responds, and low thresholds are
# reached while both responses are still near their true start.
ONSET_FRACTION = 0.10
# Onset timing and correlation timing must agree this closely, or the two detectors
# latched onto different events (e.g. the wearer's own pulse rather than the tap).
ONSET_AGREE_S = 0.30
# Envelope variation slower than this is background (the wearer's pulse, breathing,
# drift), not a tap.
TREND_WINDOW_S = 0.40
# The gates use a shorter one. They judge a single second of history, so a 0.4 s trend
# eats half the window; and unlike the estimator, which wants the tap's whole shape, a
# gate only has to notice that something fast happened. Measured across a real fiber, a
# microphone and a strap, shortening it to 0.15 s roughly halves the force a tap needs
# without producing a single false trigger in 77 s of quiet.
GATE_TREND_S = 0.15
# Each channel's transient envelope must stand this far above its own floor to count as
# containing a tap at all. Measured separation on simulated traces: a tapped PPG scores
# ~8 and an untapped one (wearer's pulse only) ~2.5; the microphone is ~900 vs ~11. The
# PPG is the limiting case, so the threshold sits between 2.5 and 8.
IMPULSE_PROMINENCE_MIN = 4.0

# --- the two-tap calibration ----------------------------------------------------
# Amplitude is measured as the peak of the channel's *transient* envelope inside a
# window -- the largest fast excursion it is currently making (see `activity`).
#
# Rather than guess what counts as a tap, the run asks for TWO. The first is a
# calibration knock: it establishes how big a real tap is on THIS rig, on each channel.
# The second is the one actually measured, recognised against that. Nothing here is a
# tuned constant hoping to fit a millivolt fiber and a photodiode counting in millions.
#
# It also replaces a "wait until everything is steady" gate that, measured on a real
# session, was satisfied by all three channels on 1% of polls -- these signals are
# quasi-periodic and never settle in the sense that gate wanted.
SETTLE_S = 3.0              # a plain wait; the floor is whatever the channel does in it
FLOOR_WINDOW_S = 1.0        # window the floor and the live level are measured over
# The first tap has to beat both the channel's quiet level and whatever the channel was
# doing during the settle. Two thresholds, because each covers the other's failure:
# a multiple of the floor is meaningless on a channel whose floor is near zero, and a
# multiple of the settle level is meaningless on one that was disturbed while settling.
#
# Both are modest, and can afford to be: a false trigger needs EVERY channel to spike
# within COINCIDENCE_S, and a fiber's own noise is not synchronised with an acoustic
# burst and a photodiode deflection. Measured on 77 s of the operator's quiet rig, a
# 1.5x bar fires on some polls of a single channel and on none across all three.
# Simultaneity does the rejecting; these only have to notice.
FIRST_TAP_RATIO = 1.5       # x the quiet floor
FIRST_TAP_MARGIN = 1.35     # x the settle level (see SETTLE_QUANTILE)
# The settle level is a QUANTILE of what was seen, not the maximum. The operator has
# just clicked a button and still has their hands on the rig, so the settle window
# routinely catches one real bump -- and on a live fiber a bump is 15x the quiet floor,
# which as a maximum sets a bar nothing can clear afterwards. Three quarters of the
# polls being below it is enough of a "this is what quiet looks like" without letting
# the loudest quarter define it.
SETTLE_QUANTILE = 0.75
# The second tap is judged against the first: floor + this much of the way to that
# height. Half, not the 80% first tried -- the operator has to reproduce their own
# calibration knock by feel, on three sensors at once, and demanding four fifths of it
# on every one of them is what turned "tap again" into "hit it harder, and again".
TAP_FRACTION = 0.50
# The channels do not have to cross in the same poll, only within this of each other.
# Requiring one 200 ms poll to see all three over threshold is at odds with the entire
# purpose of the exercise: they are misaligned, by up to the several hundred
# milliseconds of BLE batching, which is the quantity being measured. It made the
# operator hit harder and harder until the slowest channel's response happened to
# overlap the fastest one's -- a test of force, not of simultaneity.
COINCIDENCE_S = 1.2
RECOVER_RATIO = 2.0         # back within this multiple of the floor counts as recovered
RECOVER_HOLD_S = 0.6
TAP_WINDOW_S = 1.0
SETTLE_AFTER_TAP_S = 0.8    # let the tail of the tap arrive before measuring
# Only the SECOND tap may be analysed. The two knocks are seconds apart, so a window
# wide enough to be generous also contains the calibration tap -- and the estimator
# then happily locks onto it (measured: leading edges reported -2499 ms, the gap
# between the two taps, rather than the offset). The window is anchored to the moment
# the second tap fired instead.
#
# PRE_TAP_S covers the lag between the knock and a poll noticing it: the level is read
# over a FLOOR_WINDOW_S window, so a tap can be most of a second old by the time the
# crossing is seen. Still far short of the gap between the two taps.
PRE_TAP_S = 1.4             # context kept before the tap, for the baseline
POST_TAP_S = 1.6            # ...and after it, for the decay
MIN_CONFIDENCE = 0.35

# A residual this large is not a transport latency. BLE delivery is a few hundred
# milliseconds; anything approaching a second means the two taps were not simultaneous,
# which the estimator cannot tell from a genuinely huge offset -- it faithfully reports
# the gap between whatever two events it found. Refuse and ask for a cleaner tap rather
# than write a wild calibration.
MAX_PLAUSIBLE_LAG_S = 0.75
ARM_TIMEOUT_S = 45.0
SETTLE_TIMEOUT_S = 45.0
POLL_S = 0.2


def activity(t: np.ndarray, x: np.ndarray) -> float:
    """Peak of the channel's *transient* envelope inside the window.

    The single quantity every gate is built on, and deliberately the same pipeline the
    estimator uses on the tap it finally measures: smooth the deviation, subtract what
    is slower than a tap, take the peak of what is left. It is an absolute level in the
    channel's own units, so the thresholds around it are always *ratios* against that
    channel's own baseline -- never one number shared across a millivolt fiber and a
    photodiode counting in the millions.

    It replaces the plain peak deviation ``max|x - median|``, which was measuring the
    wrong thing on every input. On a PPG strap that number is the wearer's own pulse,
    so a tap had to out-swing a heartbeat to be noticed. On a fiber it is the noise
    tail -- the largest single-sample excursion in five thousand -- so the bar sat near
    an extreme of the noise distribution rather than above the signal. Measured on the
    operator's own live fiber over 24 s, separation between quiet stretches and real
    knocks went from 15x under the old statistic to 33x under this one, which is the
    same tap recognised at less than half the force.
    """
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    if x.size < 8:
        return 0.0
    hz = x.size / max(float(t[-1] - t[0]), 1e-9)

    dev = np.abs(x - np.median(x))
    width = max(1, int(round(ENVELOPE_SMOOTH_S * hz)))
    if width > 1:
        dev = np.convolve(dev, np.ones(width) / width, mode="same")

    trend_width = max(1, int(round(GATE_TREND_S * hz)))
    # Two trend widths, not more: the PPG runs at 55 Hz, so a one-second window holds
    # only 55 samples. Demanding more headroom than that silently skipped the trend
    # removal on exactly the channel that needs it most -- the one where the untreated
    # statistic is the wearer's own pulse.
    if trend_width > 1 and dev.size > 2 * trend_width:
        trend = np.convolve(dev, np.ones(trend_width) / trend_width, mode="same")
        dev = np.maximum(dev - trend, 0.0)
        # A 'same' convolution divides a partial sum by the full width at each end, so
        # the trend is understated there and the residual correspondingly inflated.
        # Trim both ends rather than let an edge artefact masquerade as a tap; the poll
        # interval is short enough that a genuine tap lands inside the kept region
        # within one further poll.
        edge = trend_width // 2
        dev = dev[edge:dev.size - edge]
        if dev.size == 0:
            return 0.0
    return float(np.max(dev))


def envelope(t: np.ndarray, x: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Smoothed deflection envelope of ``x``, resampled onto ``grid``.

    ``|x - median|`` smoothed over ENVELOPE_SMOOTH_S: on a microphone or fiber that is
    the energy burst of the tap, on the PPG the pressure deflection. Different physics,
    same shape of bump, which is all the correlation needs.
    """
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    if t.size < 4:
        return np.zeros(grid.size)
    dev = np.abs(x - np.median(x))

    hz = t.size / max(t[-1] - t[0], 1e-9)
    width = max(1, int(round(ENVELOPE_SMOOTH_S * hz)))
    if width > 1:
        kernel = np.ones(width) / width
        dev = np.convolve(dev, kernel, mode="same")
    return np.interp(grid, t, dev, left=0.0, right=0.0)


def _transient(env_values: np.ndarray) -> np.ndarray:
    """Whatever in an envelope is faster than TREND_WINDOW_S -- i.e. plausibly a tap.

    A tap is over in a fraction of a second; a heartbeat is not. Removing the slow
    trend leaves the impulse and drops the periodic background that would otherwise
    dominate the correlation.
    """
    width = max(1, int(round(TREND_WINDOW_S * GRID_HZ)))
    if width <= 1 or env_values.size <= width:
        return env_values
    trend = np.convolve(env_values, np.ones(width) / width, mode="same")
    return np.maximum(env_values - trend, 0.0)


def _prominence(env_values: np.ndarray) -> float:
    """How far a transient envelope's peak stands above its own floor."""
    positive = env_values[env_values > 0]
    if positive.size < 4:
        return 0.0
    floor = float(np.median(positive))
    if floor <= 0:
        return float("inf")
    return float(np.max(env_values) / floor)


@dataclass
class LagEstimate:
    lag_s: float          # + means the PPG is stamped LATER than the reference
    confidence: float     # normalised correlation peak, 0..1
    ok: bool
    detail: str = ""

    def to_json(self) -> dict:
        return {"lag_s": round(self.lag_s, 4), "confidence": round(self.confidence, 3),
                "ok": self.ok, "detail": self.detail}


def onset(env_values: np.ndarray, grid: np.ndarray) -> float | None:
    """Time at which ``env_values`` first rises through ONSET_FRACTION of its peak.

    Onset rather than the correlation peak, because the two sensors have genuinely
    different response *shapes*: a tap is a sharp acoustic burst to the microphone and
    a slow pressure deflection to the photodiode. Correlating those shapes aligns the
    difference in their response times along with the clock offset -- measured at
    +55 ms here, and worse, an amount that depends on the strap's response time, which
    is unknown. The leading edge is the part of both signals that refers to the same
    physical instant.
    """
    if env_values.size < 4:
        return None
    peak_i = int(np.argmax(env_values))
    if peak_i < 2:
        return None
    baseline = float(np.median(env_values[:max(1, peak_i // 2)]))
    peak = float(env_values[peak_i])
    if peak <= baseline:
        return None

    threshold = baseline + ONSET_FRACTION * (peak - baseline)
    i = peak_i
    while i > 0 and env_values[i] > threshold:
        i -= 1
    if i >= peak_i:
        return None
    y0, y1 = float(env_values[i]), float(env_values[i + 1])
    alpha = 0.0 if y1 == y0 else (threshold - y0) / (y1 - y0)
    return float(grid[i] + alpha / GRID_HZ)


def estimate_lag(ref: tuple[np.ndarray, np.ndarray],
                 ppg: tuple[np.ndarray, np.ndarray]) -> LagEstimate:
    """Lag between two recordings of the same tap, by envelope cross-correlation.

    Both series carry absolute time, so the answer is a real clock offset rather than a
    sample count. Positive means the PPG's copy of the event is stamped later than the
    reference's -- i.e. the strap's samples are being written too late.
    """
    t_ref, x_ref = ref
    t_ppg, x_ppg = ppg
    lo = max(float(t_ref[0]), float(t_ppg[0])) - MAX_LAG_S
    hi = min(float(t_ref[-1]), float(t_ppg[-1])) + MAX_LAG_S
    if hi - lo < 1.0:
        return LagEstimate(0.0, 0.0, False)

    grid = np.arange(lo, hi, 1.0 / GRID_HZ)
    # Isolate transients before doing anything else. The PPG envelope also contains the
    # wearer's own pulse, which is large and periodic, and it breaks both estimators:
    # the correlation happily locks onto a pulse cycle, and the onset threshold (a
    # fraction of the tap's height) ends up SMALLER than the pulse's own swing, so
    # walking back from the tap exits at an arbitrary pulse trough. Removing the slow
    # background leaves only what is fast enough to be a tap, and both behave.
    env_ref = _transient(envelope(t_ref, x_ref, grid))
    env_ppg = _transient(envelope(t_ppg, x_ppg, grid))

    # --- both channels must actually contain an impulse ---
    # Correlation alone is not enough of a guard: with only the reference tapped, the
    # PPG's residual pulse structure still correlated at 0.39, over the confidence
    # floor. Requiring a real impulse on each side is what makes "you only tapped one
    # of them" a refusal instead of a plausible-looking answer.
    for env_values, who in ((env_ref, "reference"), (env_ppg, "PPG")):
        if _prominence(env_values) < IMPULSE_PROMINENCE_MIN:
            return LagEstimate(0.0, 0.0, False, f"no tap seen on the {who}")

    # --- correlation: does the PPG's transient match the reference's at all? ---
    a = env_ref - env_ref.mean()
    b = env_ppg - env_ppg.mean()
    if a.std() <= 0 or b.std() <= 0:
        return LagEstimate(0.0, 0.0, False, "flat signal")

    corr = np.correlate(b, a, mode="full") / (a.std() * b.std() * grid.size)
    lags = (np.arange(corr.size) - (grid.size - 1)) / GRID_HZ
    keep = np.abs(lags) <= MAX_LAG_S
    corr, lags = corr[keep], lags[keep]
    if corr.size == 0:
        return LagEstimate(0.0, 0.0, False, "no overlap")
    confidence = float(np.max(corr))
    corr_lag = float(lags[int(np.argmax(corr))])
    if confidence < MIN_CONFIDENCE:
        return LagEstimate(corr_lag, confidence, False, "transients do not match")

    # --- onset: the unbiased timing (see onset()) ---
    on_ref = onset(env_ref, grid)
    on_ppg = onset(env_ppg, grid)
    if on_ref is None or on_ppg is None:
        missing = "reference" if on_ref is None else "PPG"
        return LagEstimate(corr_lag, confidence, False,
                           f"no clean rising edge on the {missing} — its tap was "
                           f"probably too soft to stand out")
    lag = on_ppg - on_ref

    # Cross-check. The two methods measure the same thing by different means, so a
    # large disagreement means one of them locked onto the wrong event -- typically the
    # PPG onset finding the wearer's own pulse instead of the tap.
    if abs(lag - corr_lag) > ONSET_AGREE_S:
        return LagEstimate(corr_lag, confidence, False,
                           f"the two estimates disagree: leading edges say "
                           f"{lag * 1000:+.0f} ms, waveform shape says "
                           f"{corr_lag * 1000:+.0f} ms ({abs(lag - corr_lag) * 1000:.0f} ms apart)")

    if abs(lag) > MAX_PLAUSIBLE_LAG_S:
        return LagEstimate(lag, confidence, False,
                           f"{abs(lag) * 1000:.0f} ms apart — too far to be transport "
                           f"latency; tap both together")

    return LagEstimate(lag, confidence, True, f"edge-aligned, shape agrees to "
                                              f"{abs(lag - corr_lag) * 1000:.0f} ms")


def _remedy(detail: str) -> str:
    """What to actually do about a given rejection reason."""
    if "disagree" in detail:
        return ("Two different events were matched — usually a double tap, or one sensor "
                "knocked slightly before the other. Tap once, sharply, with both struck "
                "by the same motion.")
    if "no clean rising edge" in detail:
        return "Tap that sensor harder, or pick a fiber that is easier to strike."
    if "too far to be transport latency" in detail:
        return "The two taps were not simultaneous. Strike both together in one motion."
    if "no tap seen" in detail:
        return "That sensor did not register the knock — hit it directly and firmly."
    if "transients do not match" in detail:
        return ("The two impulses do not look like the same event. Make sure one motion "
                "strikes both sensors.")
    return "Tap both sensors at the same instant, firmly, and try again."



@dataclass
class AlignState:
    """What the wizard renders. One calibration run, start to finish."""

    # idle | settling | tap1 | recover | tap2 | measuring | done | failed
    phase: str = "idle"
    message: str = ""
    hint: str = ""
    reference: str = ""                          # the fiber everything is measured against
    channels: list = field(default_factory=list) # reference first, then the targets
    seconds_left: float = 0.0                    # settling countdown, for the UI
    floor: dict = field(default_factory=dict)    # channel -> quiet level
    tap_height: dict = field(default_factory=dict)   # channel -> first tap's peak
    threshold: dict = field(default_factory=dict)    # channel -> level the 2nd tap must beat
    level: dict = field(default_factory=dict)    # channel -> live level, 0..1 of threshold
    results: dict = field(default_factory=dict)  # channel -> {lag_s, confidence, detail, ok}
    applied: bool = False

    def to_json(self) -> dict:
        r = lambda d: {k: round(float(v), 6) for k, v in d.items()}  # noqa: E731
        return {"phase": self.phase, "message": self.message, "hint": self.hint,
                "reference": self.reference, "channels": list(self.channels),
                "seconds_left": round(self.seconds_left, 1),
                "floor": r(self.floor), "tap_height": r(self.tap_height),
                "threshold": r(self.threshold), "level": r(self.level),
                "results": self.results, "applied": self.applied}


class TapAligner:
    """Runs the two-tap calibration on a background thread. Poll :meth:`state`."""

    def __init__(self, hub):
        self.hub = hub
        self._state = AlignState()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._cancel = threading.Event()

    def state(self) -> AlignState:
        with self._lock:
            return AlignState(**dict(self._state.__dict__))

    def _set(self, **fields) -> None:
        with self._lock:
            for key, value in fields.items():
                setattr(self._state, key, value)

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, reference: str, targets: list[str]) -> None:
        if self.running:
            raise RuntimeError("an alignment is already running")
        live = self.hub.channel_map()
        if reference not in live:
            raise RuntimeError(f"{reference} is not streaming — arm the PicoScope first")
        # The reference must be a fiber: the only input arriving over USB, with no radio,
        # no audio stack and no clock of its own. That is what makes it ground truth.
        kind = live[reference].channel.kind
        if kind != KIND_FIBER:
            raise RuntimeError(f"{reference} is a {kind} channel — the timing reference "
                               f"must be a fiber (USB, no radio or audio buffering)")
        targets = [c for c in targets if c != reference and c in live]
        if not targets:
            raise RuntimeError("nothing to align — arm the microphone and/or the strap")

        channels = [reference, *targets]
        self._cancel.clear()
        with self._lock:
            self._state = AlignState(
                phase="settling", reference=reference, channels=channels,
                seconds_left=SETTLE_S,
                message="Settling…",
                hint="Hands off the sensors for a moment.")
        self._thread = threading.Thread(target=self._run, args=(reference, targets),
                                        name="rtmon-align", daemon=True)
        self._thread.start()

    def cancel(self) -> None:
        """Abandon a run, or discard a finished result. Always returns to idle."""
        self._cancel.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=3)
        self._thread = None
        self._set(phase="idle", message="", hint="", results={}, floor={},
                  tap_height={}, threshold={}, level={}, seconds_left=0.0, applied=False)

    def _tail(self, channel: str, seconds: float):
        got = self.hub.snapshot(channel, seconds)
        return got if got is not None and got[0].size >= 8 else None

    def _around(self, channel: str, tap_at: float):
        """The slice of ``channel`` bracketing the second tap, and nothing else."""
        got = self._tail(channel, PRE_TAP_S + POST_TAP_S + SETTLE_AFTER_TAP_S + 1.0)
        if got is None:
            return None
        t, x = got
        keep = (t >= tap_at - PRE_TAP_S) & (t <= tap_at + POST_TAP_S)
        if int(keep.sum()) < 8:
            return None
        return t[keep], x[keep]

    def _levels(self, channels) -> dict:
        """Current amplitude on each channel."""
        out = {}
        for channel in channels:
            tail = self._tail(channel, FLOOR_WINDOW_S)
            out[channel] = activity(*tail) if tail is not None else 0.0
        return out

    @staticmethod
    def _coincident(crossed: dict, channels, now: float):
        """When all of ``channels`` crossed, if they did so close enough together.

        Returns the earliest crossing time, or None. Crossings older than
        COINCIDENCE_S are forgotten in place, so one channel that fired a while ago
        cannot pair with an unrelated knock on another.
        """
        for channel in [c for c, at in crossed.items() if now - at > COINCIDENCE_S]:
            del crossed[channel]
        if len(crossed) < len(channels):
            return None
        return min(crossed.values())

    # ------------------------------------------------------------------- run
    def _run(self, reference: str, targets: list[str]) -> None:
        channels = [reference, *targets]
        try:
            settled = self._settle(channels)
            if settled is None:
                return
            floor, ceiling = settled

            heights = self._first_tap(channels, floor, ceiling)
            if heights is None:
                return

            # Calibrated from the tap the operator actually gave, per channel.
            threshold = {c: floor[c] + TAP_FRACTION * max(heights[c] - floor[c], 0.0)
                         for c in channels}
            self._set(tap_height=heights, threshold=threshold)

            if not self._recover(channels, floor):
                return
            tap_at = self._second_tap(channels, threshold)
            if tap_at is None:
                return

            self._set(phase="measuring", message="Measuring…", hint="")
            if self._cancel.wait(SETTLE_AFTER_TAP_S):
                return
            self._measure(reference, targets, tap_at)
        except Exception as exc:  # noqa: BLE001 - a calibration must never kill the server
            self._set(phase="failed", message=f"{type(exc).__name__}: {exc}", hint="")

    # --------------------------------------------------------------- phase 1
    def _settle(self, channels):
        """Wait SETTLE_S doing nothing, then take each channel's quiet level.

        A plain wait, deliberately. Any cleverer test for "has it settled" has to decide
        what settled means for a signal that is never still, and the previous attempt at
        that opened on 1% of polls. Three seconds of hands-off is something the operator
        can actually satisfy, and the floor it yields is all the next phase needs.
        """
        samples: dict[str, list[float]] = {c: [] for c in channels}
        end = time.time() + SETTLE_S
        while not self._cancel.is_set():
            remaining = end - time.time()
            self._set(seconds_left=max(0.0, remaining))
            if remaining <= 0:
                break
            for channel, level in self._levels(channels).items():
                samples[channel].append(level)
            self._cancel.wait(POLL_S)
        if self._cancel.is_set():
            return None

        floor, ceiling = {}, {}
        for channel, values in samples.items():
            usable = [v for v in values if v > 0]
            if not usable:
                self._set(phase="failed", message=f"No signal on {channel}.",
                          hint="Check it is streaming, then retry.")
                return None
            floor[channel] = float(np.median(usable))
            # What the channel was doing while holding still, robustly -- see
            # SETTLE_QUANTILE for why this is not the maximum.
            ceiling[channel] = float(np.quantile(usable, SETTLE_QUANTILE))
        self._set(floor=floor, seconds_left=0.0)
        return floor, ceiling

    # --------------------------------------------------------------- phase 2
    def _first_tap(self, channels, floor, ceiling) -> dict | None:
        """Wait for a calibration tap and return its height on each channel."""
        self._set(phase="tap1", message="Tap all sensors together — once, firmly.",
                  hint="This one just calibrates; the next one is measured.")
        thresholds = {c: max(ceiling[c] * FIRST_TAP_MARGIN, floor[c] * FIRST_TAP_RATIO)
                      for c in channels}
        self._set(threshold=thresholds)
        peaks = {c: 0.0 for c in channels}
        deadline = time.time() + ARM_TIMEOUT_S
        crossed: dict[str, float] = {}
        seen_at = None
        while not self._cancel.is_set():
            if time.time() > deadline:
                self._set(phase="failed", message="No tap detected.",
                          hint="Strike the sensors together firmly and retry.")
                return None
            now = time.time()
            levels = self._levels(channels)
            for c, v in levels.items():
                peaks[c] = max(peaks[c], v)
                if v >= thresholds[c]:
                    crossed[c] = now
            self._set(level={c: min(1.0, levels[c] / thresholds[c]) if thresholds[c] > 0 else 0.0
                             for c in channels})
            if seen_at is None and self._coincident(crossed, channels, now) is not None:
                seen_at = now
            # Once seen, keep watching for a moment so the recorded height is the tap's
            # true peak rather than whatever the first crossing caught -- but time that
            # out UNCONDITIONALLY. Nesting it inside the threshold test deadlocked: a
            # tap decays out of the 1 s window well within the wait, the condition goes
            # false again, and the phase never completes.
            if seen_at is not None and time.time() - seen_at >= TAP_WINDOW_S:
                self._set(tap_height=peaks)
                return peaks
            self._cancel.wait(POLL_S)
        return None

    # --------------------------------------------------------------- phase 3
    def _recover(self, channels, floor) -> bool:
        """Wait for every channel to fall back near its floor before asking again."""
        self._set(phase="recover", message="Good — now let it settle again.",
                  hint="Hands off until it asks for the second tap.")
        limits = {c: floor[c] * RECOVER_RATIO for c in channels}
        deadline = time.time() + ARM_TIMEOUT_S
        quiet_since = None
        while not self._cancel.is_set():
            if time.time() > deadline:
                self._set(phase="failed", message="The signals never came back down.",
                          hint="Something is still moving; stop touching the sensors and retry.")
                return False
            levels = self._levels(channels)
            self._set(level={c: min(1.0, levels[c] / limits[c]) if limits[c] > 0 else 0.0
                             for c in channels})
            if all(levels[c] <= limits[c] for c in channels):
                quiet_since = quiet_since or time.time()
                if time.time() - quiet_since >= RECOVER_HOLD_S:
                    return True
            else:
                quiet_since = None
            self._cancel.wait(POLL_S)
        return False

    # --------------------------------------------------------------- phase 4
    def _second_tap(self, channels, threshold) -> float | None:
        self._set(phase="tap2", message="Now tap again — this one is measured.",
                  hint="Same force, all sensors struck by one motion.")
        deadline = time.time() + ARM_TIMEOUT_S
        crossed: dict[str, float] = {}
        while not self._cancel.is_set():
            if time.time() > deadline:
                self._set(phase="failed", message="No second tap detected.",
                          hint="Retry, striking about as hard as the first time.")
                return None
            now = time.time()
            levels = self._levels(channels)
            for c, v in levels.items():
                if v >= threshold[c]:
                    crossed[c] = now
            self._set(level={c: min(1.0, levels[c] / threshold[c]) if threshold[c] > 0 else 0.0
                             for c in channels})
            # Every channel must see it, within COINCIDENCE_S of the others: one alone
            # is a sensor being bumped. The earliest crossing anchors the analysis
            # window, since that is the channel that noticed the knock first.
            tap_at = self._coincident(crossed, channels, now)
            if tap_at is not None:
                return tap_at
            self._cancel.wait(POLL_S)
        return None

    # --------------------------------------------------------------- measure
    def _measure(self, reference: str, targets: list[str], tap_at: float) -> None:
        ref = self._around(reference, tap_at)
        if ref is None:
            self._set(phase="failed", message=f"Lost {reference} while measuring.", hint="")
            return
        results, good = {}, []
        for channel in targets:
            tail = self._around(channel, tap_at)
            if tail is None:
                results[channel] = {"ok": False, "detail": "signal lost",
                                    "lag_s": None, "confidence": None}
                continue
            est = estimate_lag(ref, tail)
            results[channel] = {"ok": est.ok, "lag_s": est.lag_s,
                                "confidence": est.confidence, "detail": est.detail}
            if est.ok:
                good.append(channel)
        self._set(results=results)

        if not good:
            detail = " · ".join(f"{c}: {r['detail']}" for c, r in results.items())
            self._set(phase="failed", message=detail,
                      hint=_remedy(" ".join(r["detail"] for r in results.values())))
            return
        parts = [f"{c} {abs(results[c]['lag_s']) * 1000:.0f} ms "
                 f"{'late' if results[c]['lag_s'] > 0 else 'early'}"
                 for c in targets if results[c]["ok"]]
        self._set(phase="done",
                  message=f"Relative to {reference}: " + ", ".join(parts),
                  hint="Apply adds these to the corrections already in force.")
