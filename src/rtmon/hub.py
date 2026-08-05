"""The acquisition hub: sources in, buffers and files out.

One place owns the answer to "what is streaming, and where can I read it". Sources
push blocks; the hub fans each block out to three consumers and nothing else in the
app talks to a device:

    source.start(sink) --> sink --> SignalRing    (processors read contiguous chunks)
                                --> EnvelopeRing  (the browser reads a decimated view)
                                --> Recorder      (only while a session is open)

Channel ids are global, not per-source: ``2A`` means the same fiber whichever box
produced it. That is what lets a simulator stand in for a missing device without
rewriting any track, and it is enforced -- starting a source whose channels are
already claimed by a running one fails loudly rather than silently shadowing them.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rtmon.recorder import Recorder, StreamSpec
from rtmon.ring import EnvelopeRing, SignalRing
from rtmon.sources import RECORD_AS, Channel, Probe, Source, build_sources

# How long the display keeps a scrollback, independent of the raw ring. Cheap: at 300
# buckets/s a channel-minute of min/max envelope is 144 kB.
DISPLAY_HISTORY_SECONDS = 300.0
SILENCE_EPS = 1e-7           # below the LSB of any 16-bit ADC on the rig


@dataclass(frozen=True)
class ChannelRef:
    channel: Channel
    source_id: str
    col: int


class _Runtime:
    """Per-source buffers, allocated when the source starts and dropped when it stops."""

    __slots__ = ("raw", "display")

    def __init__(self, source: Source):
        n = len(source.channels)
        capacity = max(1024, int(source.history_seconds * max(source.nominal_hz, 1.0)))
        self.raw = SignalRing(n, capacity)
        self.display = EnvelopeRing(n, DISPLAY_HISTORY_SECONDS)


class Hub:
    def __init__(self, session_root: Path):
        self.sources: dict[str, Source] = {s.id: s for s in build_sources()}
        self.recorder = Recorder(session_root)
        self._runtimes: dict[str, _Runtime] = {}
        self._probes: dict[str, Probe] = {}
        self._lock = threading.RLock()

    # ----------------------------------------------------------------- probe
    def probe_all(self, deep: bool = True) -> dict[str, Probe]:
        """Probe every source concurrently.

        Concurrently because a deep probe talks to hardware: a BLE scan alone is
        several seconds, and serialising four of them would make the app feel broken
        at startup.
        """
        def one(source: Source) -> tuple[str, Probe]:
            if source.running:
                return source.id, Probe(True, "streaming")
            try:
                return source.id, source.probe(deep=deep)
            except Exception as exc:  # noqa: BLE001 - probe promises not to raise; enforce it
                return source.id, Probe(False, f"probe raised {type(exc).__name__}: {exc}")

        with ThreadPoolExecutor(max_workers=len(self.sources) or 1) as pool:
            results = dict(pool.map(one, self.sources.values()))
        with self._lock:
            self._probes = results
        return results

    def probes(self) -> dict[str, Probe]:
        return dict(self._probes)

    # ------------------------------------------------------------- lifecycle
    def start_source(self, source_id: str) -> None:
        with self._lock:
            source = self._require(source_id)
            if source.running:
                return
            clash = self._channel_clash(source)
            if clash:
                owner, ids = clash
                raise RuntimeError(
                    f"{', '.join(ids)} already streaming from {self.sources[owner].label}. "
                    f"Stop it first — a channel id names one physical fiber, so two "
                    f"sources cannot both provide it.")
            runtime = _Runtime(source)
            self._runtimes[source_id] = runtime

        # start() outside the lock: it opens hardware and can block for seconds.
        try:
            source.start(self._make_sink(source_id, runtime))
        except Exception:
            with self._lock:
                self._runtimes.pop(source_id, None)
            raise

        if self.recorder.active:
            # Late-joining a live recording is normal (the strap reconnects, a scope is
            # re-plugged); the stream just starts partway through the session.
            self._open_stream_for(source)

    def stop_source(self, source_id: str) -> None:
        source = self._require(source_id)
        try:
            source.stop()
        finally:
            with self._lock:
                self._runtimes.pop(source_id, None)

    def stop_all(self) -> None:
        for source_id in list(self._runtimes):
            try:
                self.stop_source(source_id)
            except Exception:
                pass

    def _require(self, source_id: str) -> Source:
        source = self.sources.get(source_id)
        if source is None:
            raise KeyError(f"unknown source {source_id!r}")
        return source

    def _channel_clash(self, source: Source):
        wanted = {c.id for c in source.channels}
        for other_id, other in self.sources.items():
            if other_id == source.id or not other.running:
                continue
            overlap = wanted & {c.id for c in other.channels}
            if overlap:
                return other_id, sorted(overlap)
        return None

    def _make_sink(self, source_id: str, runtime: _Runtime):
        recorder = self.recorder

        def sink(t: np.ndarray, x: np.ndarray) -> None:
            runtime.raw.push(t, x)
            runtime.display.push(t, x)
            if recorder.active:
                recorder.write(source_id, t, x)

        return sink

    # ------------------------------------------------------------- channels
    def channel_map(self) -> dict[str, ChannelRef]:
        """Every channel currently streaming, keyed by channel id."""
        out: dict[str, ChannelRef] = {}
        with self._lock:
            for source_id in self._runtimes:
                source = self.sources[source_id]
                for col, channel in enumerate(source.channels):
                    out[channel.id] = ChannelRef(channel, source_id, col)
        return out

    def rate_of(self, channel_id: str) -> float:
        ref = self.channel_map().get(channel_id)
        if ref is None:
            return 0.0
        runtime = self._runtimes.get(ref.source_id)
        if runtime is None:
            return 0.0
        return runtime.raw.hz or self.sources[ref.source_id].nominal_hz

    def latest_time(self, channel_id: str) -> float:
        ref = self.channel_map().get(channel_id)
        runtime = self._runtimes.get(ref.source_id) if ref else None
        return runtime.raw.latest_time() if runtime else 0.0

    def newest_time(self) -> float:
        """Newest sample time across every running source (the display's 'now')."""
        with self._lock:
            times = [r.raw.latest_time() for r in self._runtimes.values()]
        times = [t for t in times if t > 0]
        return max(times) if times else 0.0

    # ----------------------------------------------------------------- read
    def snapshot(self, channel_id: str, seconds: float):
        """Newest ``seconds`` of one channel as ``(t, x)`` 1-D arrays, or ``None``."""
        ref = self.channel_map().get(channel_id)
        if ref is None:
            return None
        runtime = self._runtimes.get(ref.source_id)
        if runtime is None:
            return None
        got = runtime.raw.snapshot(seconds, cols=slice(ref.col, ref.col + 1))
        if got is None:
            return None
        t, x = got
        return t, x[:, 0]

    def is_silent(self, channel_id: str, seconds: float = 4.0) -> bool:
        """True if the channel has delivered nothing but zeros for ``seconds``.

        Worth its own signal because the common causes leave a device looking
        perfectly healthy: macOS hands an app digital silence when microphone
        permission has not been granted, and a disconnected fiber reads as a flat line
        at exactly the right sample rate. Both record a full-length session of nothing.
        """
        envelope = self.envelope(channel_id, self.newest_time() or 0.0, seconds)
        if envelope is None:
            return False
        _t0, _dt, lo, hi = envelope
        if lo.size == 0 or not np.isfinite(lo).any():
            return False
        return bool(max(abs(np.nanmin(lo)), abs(np.nanmax(hi))) < SILENCE_EPS)

    def envelope(self, channel_id: str, t_end: float, seconds: float):
        ref = self.channel_map().get(channel_id)
        if ref is None:
            return None
        runtime = self._runtimes.get(ref.source_id)
        if runtime is None:
            return None
        return runtime.display.window(t_end, seconds, col=ref.col)

    # ------------------------------------------------------------ recording
    def start_recording(self, setup: dict | None = None):
        specs: dict[str, StreamSpec] = {}
        with self._lock:
            running = list(self._runtimes)
        for source_id in running:
            source = self.sources[source_id]
            spec = self._spec_for(source)
            if spec is not None:
                specs[source_id] = spec
        if not specs:
            raise RuntimeError("nothing is streaming — start a device before recording")
        return self.recorder.start(specs, setup=setup)

    def _spec_for(self, source: Source) -> StreamSpec | None:
        name = RECORD_AS.get(source.id)
        if name is None:
            return None
        is_audio = name == "microphone"
        return StreamSpec(
            name=name,
            columns=[c.id for c in source.channels],
            audio=is_audio,
            audio_hz=int(source.nominal_hz) if is_audio else 0,
        )

    def _open_stream_for(self, source: Source) -> None:
        """Attach a source that started after the recording did."""
        spec = self._spec_for(source)
        if spec is None:
            return
        try:
            self.recorder.add_stream(source.id, spec)
        except Exception:
            pass

    def stop_recording(self):
        return self.recorder.stop()

    # ------------------------------------------------------------- describe
    def describe(self) -> dict:
        probes = self.probes()
        return {
            "sources": [
                {**source.to_json(), "probe": probes.get(sid, Probe(False, "not probed")).to_json()}
                for sid, source in self.sources.items()
            ],
            "channels": [
                {**ref.channel.to_json(), "source": ref.source_id,
                 "hz": round(self.rate_of(cid), 2), "silent": self.is_silent(cid)}
                for cid, ref in sorted(self.channel_map().items())
            ],
            "recording": self.recorder.status(),
        }
