"""Tap alignment: measure the PPG strap's timing offset against the rest of the rig.

The fibers and the microphone are all timestamped by a wall-clock-disciplined
:class:`~rtmon.sources.base.SampleClock`, so they share one time base. The BLE strap
does not -- its samples arrive already stale, by an amount that depends on the strap,
the host stack and the connection interval. :data:`~rtmon.sources.polar.PPG_PIPELINE_LATENCY_S`
carries a default inherited from the Qt app's measurement, which is a reasonable guess
and nothing more.

This measures it instead. Tap the strap and a *fiber* together: one physical
event, recorded twice, and the difference between the two recorded times *is* the
error. A fiber rather than the microphone, for two reasons -- it is the channel the
strap is actually compared against (a maternal fiber estimate is scored against the PPG
source of truth), and both are contact sensors struck by the same knock, where a
microphone would hear it through the air and add an acoustic path delay unrelated to
the strap. The procedure is deliberately small --

    1. wait until both channels are quiet, so a tap will stand out
    2. "tap both now"
    3. cross-correlate the two impulse envelopes and report the lag

-- and it is opt-in. Nothing here runs unless asked, and the result is offered for
confirmation rather than applied behind the operator's back.

Cross-correlation of envelopes, rather than picking each impulse's onset: a tap excites
a microphone and a photodiode quite differently (a sharp acoustic burst versus a slower
pressure deflection), so their onsets are not directly comparable, while the position of
best overlap is. The PPG's 55 Hz sampling puts a floor of roughly +/-20 ms on the
result, which is ample against an error measured in hundreds of milliseconds.
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
# Each channel's transient envelope must stand this far above its own floor to count as
# containing a tap at all. Measured separation on simulated traces: a tapped PPG scores
# ~8 and an untapped one (wearer's pulse only) ~2.5; the microphone is ~900 vs ~11. The
# PPG is the limiting case, so the threshold sits between 2.5 and 8.
IMPULSE_PROMINENCE_MIN = 4.0

QUIET_WINDOW_S = 1.5        # window each channel's activity level is measured over
# "Settled" means STEADY, not silent. Silence is the wrong requirement: a microphone on
# a belly is full of heart sounds and measures a crest of ~34, an abdomen fiber ~8.8, so
# a threshold low enough to call either of those quiet is a threshold alignment can
# never arm under. What the baseline actually has to be is *representative*, because the
# tap is recognised relative to it -- and those same signals are remarkably steady
# (34.2, 33.7, 34.5, 35.4, 34.3, 34.9 over consecutive windows). So the gate is on the
# spread of recent measurements rather than their magnitude.
QUIET_STABILITY = 0.35      # (max-min)/median across recent windows
QUIET_SAMPLES = 6           # consecutive windows that must agree
QUIET_HOLD_S = 1.2
# A tap is recognised RELATIVE to each channel's own settled level, not against one
# absolute number. The sensors differ far too much for a single threshold: a struck
# microphone measures ~120 while a struck strap measures ~7, and an unstruck microphone
# measures ~5.4 -- so any fixed value that catches the strap's tap also fires on the
# mic's silence, and any value above the mic's silence misses the strap entirely. The
# earlier fixed 9.0 did exactly that: a genuine strap tap scored 7.0 and the run simply
# timed out. Each channel's quiet crest is captured during the settle phase and the tap
# must beat it by this ratio (with a floor, so a pathologically flat channel still
# needs a real impulse).
TAP_CREST_RATIO = 2.0
TAP_CREST_FLOOR = 4.0
TAP_WINDOW_S = 1.0
SETTLE_AFTER_TAP_S = 0.8    # let the tail of the tap arrive before measuring
ANALYSIS_WINDOW_S = 4.0
# Normalised correlation peak below which the two transients clearly are not the same
# event. Only a coarse sanity check: genuine dual taps measured 0.46-0.80 while a
# single-sensor tap correlated against the other channel's noise reached 0.46, so the
# two overlap and tightening this only starts rejecting real measurements (0.50 threw
# away 12% of them). The guards that actually work are layered instead --
#
#   * TapAligner._wait_tap requires BOTH channels to cross TAP_CREST_MIN live, which is
#     what establishes that both sensors were struck (measured: a struck mic ~140, an
#     unstruck one ~6; a struck strap ~6.2, an unstruck one ~1.7);
#   * _prominence rejects an analysis window with no impulse in it;
#   * MAX_PLAUSIBLE_LAG_S rejects two taps that were not simultaneous;
#   * the onset/correlation cross-check rejects locking onto different events.
MIN_CONFIDENCE = 0.35

# A residual this large is not a transport latency. BLE delivery is a few hundred
# milliseconds; anything approaching a second means the two taps were not simultaneous,
# which the estimator cannot tell from a genuinely huge offset -- it faithfully reports
# the gap between whatever two events it found. Refuse and ask for a cleaner tap rather
# than write a wild calibration.
MAX_PLAUSIBLE_LAG_S = 0.75
ARM_TIMEOUT_S = 45.0
QUIET_TIMEOUT_S = 45.0
POLL_S = 0.2


def _crest(x: np.ndarray) -> float:
    """Peak-to-median ratio of ``|x - median|`` -- large when a transient is present.

    Deliberately scale-free: the fibers are millivolts, the PPG is raw photodiode
    counts in the millions, and the same threshold has to mean the same thing on both.
    """
    x = np.asarray(x, dtype=float)
    if x.size < 8:
        return 0.0
    dev = np.abs(x - np.median(x))
    typical = np.median(dev)
    if typical <= 0:
        # A perfectly flat trace is quiet, not infinitely spiky.
        return 0.0 if float(np.max(dev)) <= 0 else float("inf")
    return float(np.max(dev) / typical)


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
    """What the UI renders. One tap-alignment run, start to finish."""

    phase: str = "idle"          # idle|waiting_quiet|armed|measuring|done|failed
    message: str = ""
    reference: str = ""
    lag_s: float | None = None
    confidence: float | None = None
    detail: str = ""
    applied: bool = False
    started_at: float = 0.0
    quiet: dict = field(default_factory=dict)   # channel -> is it currently settled

    def to_json(self) -> dict:
        return {"phase": self.phase, "message": self.message, "reference": self.reference,
                "lag_s": None if self.lag_s is None else round(self.lag_s, 4),
                "confidence": None if self.confidence is None else round(self.confidence, 3),
                "detail": self.detail, "applied": self.applied, "quiet": self.quiet}


class TapAligner:
    """Drives one alignment run on a background thread. Poll :meth:`state` to render."""

    def __init__(self, hub, ppg_channel: str = "PPG0"):
        self.hub = hub
        self.ppg_channel = ppg_channel
        self._state = AlignState()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._cancel = threading.Event()

    def state(self) -> AlignState:
        with self._lock:
            return AlignState(**{**self._state.__dict__})

    def _set(self, **fields) -> None:
        with self._lock:
            for key, value in fields.items():
                setattr(self._state, key, value)

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, reference: str) -> None:
        if self.running:
            raise RuntimeError("an alignment is already running")
        live = self.hub.channel_map()
        for channel in (reference, self.ppg_channel):
            if channel not in live:
                raise RuntimeError(f"{channel} is not streaming — arm it first")
        # The reference must be a FIBER. It is what the strap is actually compared
        # against (a maternal fiber estimate is scored against the PPG source of
        # truth), so the fiber's time base is the one that has to line up. It is also
        # the honest tap: both are contact sensors struck by the same knock, whereas a
        # microphone hears it through the air and adds a path delay that has nothing to
        # do with the strap.
        kind = live[reference].channel.kind
        if kind != KIND_FIBER:
            raise RuntimeError(
                f"{reference} is a {kind} channel — align the PPG against a fiber, "
                f"which is what it is compared against")
        self._cancel.clear()
        with self._lock:
            self._state = AlignState(phase="waiting_quiet", reference=reference,
                                     started_at=time.time(),
                                     message="Hold still — waiting for both signals to settle…")
        self._thread = threading.Thread(target=self._run, args=(reference,),
                                        name="rtmon-align", daemon=True)
        self._thread.start()

    def cancel(self) -> None:
        """Abandon a run in progress, or discard a finished result.

        Always returns to idle. It used to leave a completed measurement in place, so
        the UI's Discard button -- which is this same call -- left the result and its
        Apply button on screen, discarding nothing.
        """
        self._cancel.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=3)
        self._thread = None
        self._set(phase="idle", message="", lag_s=None, confidence=None,
                  applied=False, quiet={})

    def _tail(self, channel: str, seconds: float):
        got = self.hub.snapshot(channel, seconds)
        return got if got is not None and got[0].size >= 8 else None

    def _run(self, reference: str) -> None:
        channels = (reference, self.ppg_channel)
        try:
            quiet_crests = self._wait_quiet(channels)
            if quiet_crests is None:
                return
            tap_at = self._wait_tap(channels, quiet_crests)
            if tap_at is None:
                return

            self._set(phase="measuring", message="Measuring…")
            # Let the tail of the tap land in both rings before reading them.
            if self._cancel.wait(SETTLE_AFTER_TAP_S):
                return
            ref = self._tail(reference, ANALYSIS_WINDOW_S)
            ppg = self._tail(self.ppg_channel, ANALYSIS_WINDOW_S)
            if ref is None or ppg is None:
                self._set(phase="failed", message="Lost one of the signals while measuring.")
                return

            est = estimate_lag(ref, ppg)
            if not est.ok:
                # Report the reason it ACTUALLY rejected for. This used to print
                # "could not match the two taps (confidence 0.59)", naming the one
                # quantity that was fine -- 0.59 clears the 0.35 confidence floor
                # comfortably -- while the real cause sat unused in est.detail.
                self._set(phase="failed", confidence=est.confidence, detail=est.detail,
                          message=f"{est.detail.capitalize()}. {_remedy(est.detail)}")
                return
            direction = "late" if est.lag_s > 0 else "early"
            self._set(phase="done", lag_s=est.lag_s, confidence=est.confidence,
                      detail=est.detail,
                      message=(f"PPG is {abs(est.lag_s) * 1000:.0f} ms {direction} "
                               f"relative to {reference} (confidence {est.confidence:.2f}). "
                               f"Apply adds this to the correction already in force."))
        except Exception as exc:  # noqa: BLE001 - a calibration must never kill the server
            self._set(phase="failed", message=f"{type(exc).__name__}: {exc}")

    def _wait_quiet(self, channels) -> dict | None:
        """Block until both channels are STEADY (see QUIET_STABILITY).

        Returns each channel's baseline activity level, which is what the tap then has
        to beat (see TAP_CREST_RATIO) -- so the point of this phase is to establish a
        representative baseline, not to wait for silence that may never come.
        """
        deadline = time.time() + QUIET_TIMEOUT_S
        history: dict[str, list[float]] = {c: [] for c in channels}
        steady_since = None
        while not self._cancel.is_set():
            if time.time() > deadline:
                self._set(phase="failed",
                          message="Signals never steadied. Stop moving the sensors and retry.")
                return None

            baselines, settled = {}, {}
            for channel in channels:
                tail = self._tail(channel, QUIET_WINDOW_S)
                samples = history[channel]
                samples.append(_crest(tail[1]) if tail is not None else float("inf"))
                del samples[:-QUIET_SAMPLES]
                if len(samples) < QUIET_SAMPLES or not all(np.isfinite(samples)):
                    settled[channel] = False
                    continue
                median = float(np.median(samples))
                spread = (max(samples) - min(samples)) / median if median > 0 else float("inf")
                settled[channel] = bool(spread <= QUIET_STABILITY)
                baselines[channel] = median
            self._set(quiet=settled)

            if all(settled.values()):
                steady_since = steady_since or time.time()
                if time.time() - steady_since >= QUIET_HOLD_S:
                    self._set(phase="armed",
                              message="Ready — tap both sensors together, once, firmly.")
                    return baselines
            else:
                steady_since = None
            self._cancel.wait(POLL_S)
        return None

    def _wait_tap(self, channels, quiet_crests: dict) -> float | None:
        thresholds = {c: max(TAP_CREST_FLOOR, quiet_crests.get(c, 0.0) * TAP_CREST_RATIO)
                      for c in channels}
        deadline = time.time() + ARM_TIMEOUT_S
        while not self._cancel.is_set():
            if time.time() > deadline:
                self._set(phase="failed", message="No tap detected. Retry when ready.")
                return None
            hits = 0
            for channel in channels:
                tail = self._tail(channel, TAP_WINDOW_S)
                if tail is not None and _crest(tail[1]) >= thresholds[channel]:
                    hits += 1
            # Both must see it. One alone is someone bumping a single sensor, and
            # correlating that against the other channel's noise is exactly the case
            # the estimator's own confidence cannot reliably reject.
            if hits == len(channels):
                return time.time()
            self._cancel.wait(POLL_S)
        return None
