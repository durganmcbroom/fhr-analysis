"""Tracks: the configurable matrix of "what is computing what, from which inputs".

A :class:`Track` is one row of the setup table and one trace on the chart. It names a
processor, the channels feeding it, the model version and detector it should use, how
much history to analyse and how often. Nothing about the estimators is fixed in code:
which FUNet, which NeoSSNet, which fibers go into each, and which track is the source
of truth are all runtime state, editable while the recording runs.

The engine schedules them. Every track has its own period and its own in-flight guard,
so a slow five-fiber FUNet cannot delay the cheap acoustic SOT beside it -- the Qt
panel ran all three estimators in one serial pass behind one shared ``_busy`` flag,
which meant the whole panel updated at the speed of its slowest model.

Results merge by *window*: a cycle re-analyses the last ``chunk`` seconds and replaces
exactly that span of the accumulated beat train, so overlapping chunks refine recent
beats instead of duplicating them.
"""

from __future__ import annotations

import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace

import numpy as np

from analyze.util import moving_average_v2
from rtmon import models as model_registry
from rtmon import processors as proc
from rtmon.hub import Hub

# How much HR history a track keeps. Bounds both memory and the cost of redrawing;
# raise it to see a longer trend.
HISTORY_SECONDS = 900.0
# Window the live agreement statistics are computed over.
AGREEMENT_SECONDS = 180.0
SCHEDULER_TICK = 0.2

# Categorical slots from the project's validated palette, assigned in fixed order and
# never cycled -- a track keeps its colour when other tracks are added or removed.
TRACK_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                "#e87ba4", "#008300", "#4a3aa7", "#e34948"]


@dataclass
class Track:
    id: str = ""
    name: str = "Track"
    enabled: bool = True
    processor: str = "acoustic"
    inputs: list[str] = field(default_factory=list)
    model: str | None = None
    detector: str = "v7_beat_detector"
    band: str = "fetal"
    chunk_s: float = 10.0
    period_s: float = 5.0
    role: str = "estimate"          # "sot" marks the reference; at most one track has it
    color: str = TRACK_COLORS[0]
    smooth: int = 0                 # moving-average window in beats; 0 disables
    show_activity: bool = False

    def to_json(self) -> dict:
        return {k: getattr(self, k) for k in (
            "id", "name", "enabled", "processor", "inputs", "model", "detector",
            "band", "chunk_s", "period_s", "role", "color", "smooth", "show_activity")}

    @staticmethod
    def from_json(raw: dict) -> "Track":
        known = {f for f in Track.__dataclass_fields__}
        clean = {k: v for k, v in raw.items() if k in known}
        clean.setdefault("id", uuid.uuid4().hex[:8])
        if clean.get("inputs") is None:
            clean["inputs"] = []
        return Track(**clean)


@dataclass
class TrackState:
    """Live results for one track. Owned by the engine, read under its lock."""

    beats: np.ndarray = field(default_factory=lambda: np.empty(0))
    activity: tuple[np.ndarray, np.ndarray] | None = None
    busy: bool = False
    due_at: float = 0.0
    last_run: float = 0.0
    last_ms: float = 0.0
    runs: int = 0
    skipped: int = 0
    error: str | None = None
    warming: str | None = None      # channel whose ring has not filled yet
    note: str = ""                  # processor's own commentary, e.g. why it found no pulse
    # Bumped whenever this track's results are reset (config change, clear). A run
    # launched before the bump carries the old generation and its result is discarded,
    # so beats computed under the previous model/fibers can never leak into the fresh
    # trace through the completion path.
    gen: int = 0


def instantaneous_hr(beats: np.ndarray, bpm_range, smooth: int = 0):
    """``(t, bpm)`` from beat times: 60/IBI plotted at the second beat of each pair.

    Matches ``analyze.plot_hr._inst_hr_v2``, including dropping out-of-band values
    rather than clamping them -- a clamped value is an invented beat.
    """
    beats = np.sort(np.asarray(beats, dtype=float))
    if beats.size < 2:
        return np.empty(0), np.empty(0)
    bpm = 60.0 / np.clip(np.diff(beats), 1e-6, None)
    t = beats[1:]
    keep = (bpm >= bpm_range[0]) & (bpm <= bpm_range[1])
    bpm, t = bpm[keep], t[keep]
    if smooth > 1 and bpm.size:
        # Clamp the window to the series: np.convolve(..., mode="same") returns
        # max(len(signal), len(kernel)) points, so a 10-beat window over a 3-beat
        # series hands back MORE smoothed values than there are times. The length
        # mismatch then blows up np.interp in the agreement stats -- which sat inside
        # state(), so one sparsely-beating track took down /api/state and the
        # WebSocket hello, i.e. the whole page.
        bpm = moving_average_v2(bpm, min(smooth, bpm.size))
    return t, bpm


class Engine:
    def __init__(self, hub: Hub, max_workers: int | None = None):
        self.hub = hub
        self.cache = model_registry.ModelCache()
        self._tracks: dict[str, Track] = {}
        self._order: list[str] = []
        self._state: dict[str, TrackState] = {}
        self._lock = threading.RLock()
        self._running = threading.Event()
        self._thread: threading.Thread | None = None
        # Deliberately small. These are torch forward passes; oversubscribing turns
        # every track's latency into every other track's latency.
        workers = max_workers or 3
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="rtmon-proc")

    # ------------------------------------------------------------ scheduling
    def start(self) -> None:
        if self._running.is_set():
            return
        self._running.set()
        self._thread = threading.Thread(target=self._loop, name="rtmon-engine", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running.clear()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout=2.0)
        self._pool.shutdown(wait=False, cancel_futures=True)

    def _loop(self) -> None:
        while self._running.is_set():
            try:
                self._tick()
            except Exception:
                traceback.print_exc()
            time.sleep(SCHEDULER_TICK)

    def _tick(self) -> None:
        now = time.monotonic()
        with self._lock:
            due = []
            for track_id in self._order:
                track = self._tracks[track_id]
                state = self._state[track_id]
                if not track.enabled or now < state.due_at:
                    continue
                if state.busy:
                    # Push the deadline out by one period rather than leaving it in the
                    # past. The pass is dropped either way -- never queued, or a track
                    # slower than its period would build an unbounded backlog -- but this
                    # makes `skipped` count *missed passes* instead of scheduler ticks,
                    # which is the number that means something next to `period_s`.
                    state.skipped += 1
                    state.due_at = now + track.period_s
                    continue
                if self.validate(track):
                    continue
                state.busy = True
                state.due_at = now + track.period_s
                due.append((replace(track), state.gen))

        for track, gen in due:
            self._pool.submit(self._run_track, track, gen)

    # --------------------------------------------------------------- running
    def _run_track(self, track: Track, gen: int = 0) -> None:
        started = time.perf_counter()
        error = None
        result = None
        warming = None
        try:
            definition = proc.PROCESSORS[track.processor]
            series = []
            for channel_id in track.inputs:
                got = self.hub.snapshot(channel_id, track.chunk_s)
                if got is None:
                    # Not an error: the device armed a moment ago and its ring has not
                    # filled yet. validate() passes as soon as the channel exists, so
                    # there is always a window of a second or two where this is simply
                    # the normal warm-up state. It used to raise, which printed a
                    # traceback per track per cycle on every arm.
                    warming = channel_id
                    break
                series.append(got)
            if warming is None:
                ctx = proc.Context(series=series, inputs=list(track.inputs), band=track.band,
                                   detector=track.detector, model=track.model, cache=self.cache)
                result = definition.run(ctx)
        except Exception as exc:  # noqa: BLE001 - one bad track must not stop the others
            error = f"{type(exc).__name__}: {exc}"
            traceback.print_exc()

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        with self._lock:
            state = self._state.get(track.id)
            if state is None or state.gen != gen:
                return   # track removed or reconfigured mid-run; result is stale
            state.busy = False
            state.error = error
            state.warming = warming
            if warming is not None:
                # Retry soon and do not count it as a run; the UI shows "waiting for
                # <channel>" rather than an error.
                state.due_at = time.monotonic() + 1.0
                return
            state.last_run = time.time()
            state.last_ms = elapsed_ms
            state.runs += 1
            if result is not None:
                state.note = result.note
                state.beats = _merge_beats(state.beats, result.beats, result.window)
                state.beats = _trim(state.beats, HISTORY_SECONDS)
                if result.activity is not None:
                    state.activity = result.activity
            # A run that took longer than its period would otherwise be permanently
            # due; push the next deadline out so the pool is not saturated by one track.
            state.due_at = max(state.due_at, time.monotonic() + max(0.5, track.period_s * 0.25))

    # ------------------------------------------------------------ validation
    def validate(self, track: Track) -> list[str]:
        """Why this track cannot run right now. Empty means it can."""
        problems = []
        definition = proc.PROCESSORS.get(track.processor)
        if definition is None:
            return [f"unknown processor {track.processor!r}"]

        channels = self.hub.channel_map()
        for channel_id in track.inputs:
            ref = channels.get(channel_id)
            if ref is None:
                problems.append(f"{channel_id} is not streaming")
            elif ref.channel.kind not in definition.kinds:
                problems.append(f"{channel_id} is a {ref.channel.kind} channel; "
                                f"{definition.label} takes {'/'.join(definition.kinds)}")

        if definition.family:
            if not track.model:
                problems.append(f"pick a {definition.family} version")
            else:
                entry = model_registry.find(definition.family, track.model)
                if entry is None:
                    problems.append(f"{track.model} not found under lib/{definition.family}")
                elif definition.arity == "model" and len(track.inputs) != entry.channels:
                    problems.append(f"{track.model} takes {entry.channels} fiber(s), "
                                    f"{len(track.inputs)} selected")
        if definition.arity == "one" and len(track.inputs) != 1:
            problems.append("select exactly one input channel")
        if not track.inputs:
            problems.append("no inputs selected")
        return problems

    # ------------------------------------------------------------- accessors
    def tracks(self) -> list[Track]:
        with self._lock:
            return [replace(self._tracks[t]) for t in self._order]

    def set_tracks(self, tracks: list[Track]) -> None:
        """Replace the whole matrix, preserving results for tracks that survive.

        Changing a track's *configuration* invalidates its beats -- a trace from FUNet
        v21 must not silently continue as v35 -- so any edit beyond cosmetics clears
        that track's history. Reordering or renaming does not.
        """
        with self._lock:
            old_tracks, old_state = self._tracks, self._state
            self._tracks = {t.id: t for t in tracks}
            self._order = [t.id for t in tracks]
            self._state = {}
            for track in tracks:
                previous = old_tracks.get(track.id)
                state = old_state.get(track.id)
                if state is not None and previous is not None and _same_computation(previous, track):
                    self._state[track.id] = state
                else:
                    self._state[track.id] = TrackState(gen=(state.gen + 1) if state else 0)

    def clear_track(self, track_id: str | None = None) -> None:
        with self._lock:
            targets = [track_id] if track_id else list(self._order)
            for tid in targets:
                if tid in self._state:
                    self._state[tid] = TrackState(gen=self._state[tid].gen + 1)

    def sot_ids(self) -> dict[str, str]:
        """``{band: track_id}`` -- the reference for each band, independently.

        There is one source of truth *per band*, not one overall, because the rig
        measures two different hearts with two different references: the microphone is
        the fetal reference, the PPG strap is the maternal one. A single global SOT
        forced every track to be scored against whichever was chosen, which produced
        confident nonsense across bands (a fetal FUNet trace "disagreeing" with a
        maternal pulse by 70 bpm says nothing about either).
        """
        found: dict[str, str] = {}
        with self._lock:
            for tid in self._order:
                track = self._tracks[tid]
                if track.role == "sot" and track.enabled:
                    found.setdefault(track.band, tid)
        return found

    def snapshot(self) -> dict:
        """Everything the browser needs to draw the HR chart and the status table."""
        with self._lock:
            tracks = [(self._tracks[t], self._state[t]) for t in self._order]
        sot_by_band = self.sot_ids()

        series: dict[str, dict] = {}
        for track, state in tracks:
            bpm_range = proc.BANDS[track.band]["bpm"]
            try:
                t, y = instantaneous_hr(state.beats, bpm_range, track.smooth)
            except Exception:  # noqa: BLE001 - one track's math must never take down state()
                traceback.print_exc()
                t, y = np.empty(0), np.empty(0)
            series[track.id] = {
                "t": t, "y": y, "beats": state.beats, "track": track, "state": state,
            }

        out = []
        for track, state in tracks:
            entry = series[track.id]
            t, y = entry["t"], entry["y"]
            item = {
                "id": track.id,
                "name": track.name,
                "color": track.color,
                "role": track.role,
                "band": track.band,
                "enabled": track.enabled,
                "problems": self.validate(track),
                "error": state.error,
                "warming": state.warming,
                "note": state.note,
                "runs": state.runs,
                "skipped": state.skipped,
                "last_ms": round(state.last_ms, 1),
                "last_run": state.last_run,
                "busy": state.busy,
                # A pass that takes longer than the gap between passes is not tracking
                # in real time -- it is showing whatever it last managed to finish.
                # Worth saying out loud: on CPU the TimesFM-backed TSLNet runs tens of
                # seconds per chunk, and the trace looks plausible while being minutes old.
                "slow": bool(state.last_ms > track.period_s * 1000.0),
                "beats": int(state.beats.size),
                "bpm": _latest(y),
                "median_bpm": float(np.median(y)) if y.size else None,
                "t": t.tolist(),
                "y": [round(float(v), 2) for v in y],
            }
            if track.show_activity and state.activity is not None:
                at, ay = state.activity
                item["activity"] = {"t": at.tolist(),
                                    "y": [round(float(v), 4) for v in ay]}
            # Scored against the reference for its OWN band, and never across bands.
            sot_id = sot_by_band.get(track.band)
            sot = series.get(sot_id) if sot_id else None
            if sot is not None and track.id != sot_id:
                try:
                    item["agreement"] = _agreement(sot["t"], sot["y"], t, y)
                    item["agreement_vs"] = series[sot_id]["track"].name
                except Exception:  # noqa: BLE001 - a statistic, never worth an outage
                    item["agreement"] = None
            out.append(item)
        return {"tracks": out, "sot": dict(sot_by_band)}


def _latest(y: np.ndarray) -> float | None:
    return float(y[-1]) if y.size else None


def _merge_beats(existing: np.ndarray, new: np.ndarray, window) -> np.ndarray:
    """Replace the just-reanalysed span with fresh beats, keeping everything older."""
    if new is None or window is None:
        return existing
    w0 = window[0]
    kept = existing[existing < w0] if existing.size else existing
    return np.sort(np.concatenate([kept, np.asarray(new, dtype=float)]))


def _trim(beats: np.ndarray, seconds: float) -> np.ndarray:
    """Trim against the track's *own* newest beat, not a global clock -- a faster-
    clocked source must not be able to trim a slower one away."""
    if beats.size == 0:
        return beats
    return beats[beats >= beats[-1] - seconds]


def _agreement(sot_t, sot_y, t, y) -> dict | None:
    """Live comparison against the source of truth over the recent window.

    The track's HR is sampled at the SOT's own time points, so the two are compared
    where the reference actually has an opinion. Reported as median absolute error
    (in bpm, the unit the operator thinks in) plus Pearson r for trend agreement.
    """
    if sot_t.size < 2 or t.size < 2 or sot_t.size != sot_y.size or t.size != y.size:
        return None
    lo = max(float(sot_t[0]), float(t[0]), float(sot_t[-1]) - AGREEMENT_SECONDS)
    hi = min(float(sot_t[-1]), float(t[-1]))
    if hi - lo < 5.0:
        return None
    mask = (sot_t >= lo) & (sot_t <= hi)
    ref_t, ref_y = sot_t[mask], sot_y[mask]
    if ref_t.size < 3:
        return None
    est = np.interp(ref_t, t, y)
    error = np.abs(est - ref_y)
    out = {"mae": round(float(np.median(error)), 1),
           "within5": round(float(np.mean(error <= 5.0) * 100.0)),
           "span": round(hi - lo, 1)}
    if ref_y.std() > 1e-6 and est.std() > 1e-6:
        out["r"] = round(float(np.corrcoef(ref_y, est)[0, 1]), 3)
    return out


# Fields that change what a track *computes*. Editing any of them invalidates the
# accumulated beats; name, colour, smoothing and activity display do not.
_COMPUTATION_FIELDS = ("processor", "inputs", "model", "detector", "band", "chunk_s")


def _same_computation(a: Track, b: Track) -> bool:
    return all(getattr(a, f) == getattr(b, f) for f in _COMPUTATION_FIELDS)
