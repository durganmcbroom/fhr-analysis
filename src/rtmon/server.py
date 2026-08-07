"""HTTP + WebSocket server for the real-time monitor.

Control is ordinary JSON over HTTP (arm a device, edit the matrix, start a recording);
the live data goes one way down a WebSocket as binary frames. Splitting them that way
keeps the streaming path free of request/response bookkeeping and means the UI's
control surface is inspectable with ``curl``.

Stdlib only, matching ``beat_app.server``: this serves one page to one operator on
loopback, and a framework would add a dependency and an event loop without changing
anything the user sees.

Frame budget: waveform envelopes go out at :data:`WAVE_FPS`; the HR traces and the
status block, which change on the order of seconds, ride along once a second. Each
client declares its own viewport, so a narrow window costs proportionally less.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import queue
import socket
import struct
import threading
import time
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

from analyze.constants import PROJECT_DIR
from rtmon import models as model_registry
from rtmon import processors as proc
from rtmon import setups as setup_store
from rtmon import display, ring, wire, wsproto
from rtmon.align import TapAligner
from rtmon.engine import Engine, Track
from rtmon.sources import polar
from rtmon.hub import Hub
from rtmon.recorder import recover
from rtmon.setups import Setup

STATIC_DIR = Path(__file__).resolve().parent / "static"
DEFAULT_SESSION_ROOT = Path(PROJECT_DIR) / ".out" / "rtmon" / "sessions"

WAVE_FPS = 15.0
SLOW_INTERVAL = 1.0          # seconds between HR-series + status updates
MAX_BUCKETS = 4000           # per channel per frame; a 4K-wide canvas cannot use more
MAX_WINDOW_S = 120.0
MAX_HR_WINDOW_S = 900.0
SILENCE_WINDOW_S = 4.0       # how long a channel must read zero before it is called silent
# How long a bandpassed display envelope is reused before being recomputed. Filtering is
# the only part of a frame that is not nearly free (~5 ms per channel for a 60 s window),
# and a diagnostic view of a filtered band does not need the full WAVE_FPS -- the render
# clock keeps sliding it in between, so it scrolls smoothly at a fifth of the work.
FILTER_TTL_S = 0.2

# The channel each correctable source is aligned through. Only one channel per source
# needs measuring — they share a clock — so PPG0 stands for the whole strap.
ALIGN_TARGET_CHANNEL = {"pvs": "PPG0", "mic": "MIC"}


class Client:
    """One connected browser tab: its viewport, plus a private outbound queue.

    The queue is the reload fix. ``send`` on a socket whose reader is gone -- a
    reloaded tab, a sleeping laptop, a half-closed connection -- blocks as soon as the
    kernel buffer fills, and the broadcaster used to write to every client serially
    from one thread. One wedged socket therefore froze frames for *every* tab: reload
    during that window and the new page came up armed but never drew anything, which
    is exactly the "works every few reloads" behaviour. Now the broadcaster only ever
    enqueues (never blocks); each client has its own writer thread; a slow client
    gets the newest frame instead of a backlog; a dead one costs a bounded queue and
    one thread that exits when its send times out.
    """

    MAX_PENDING = 4

    def __init__(self, ws: wsproto.WebSocket):
        self.ws = ws
        self.window_s = 10.0
        self.buckets = 1200
        self.hr_window_s = 300.0
        self.channels: list[str] = []
        self.signal_view = "raw"     # "raw", or a band name from proc.BANDS
        self.paused = False
        self.lock = threading.Lock()
        self.dead = False
        self._outbox: queue.Queue = queue.Queue(maxsize=Client.MAX_PENDING)
        self._writer: threading.Thread | None = None

    def start_writer(self, on_dead) -> None:
        self._writer = threading.Thread(target=self._write_loop, args=(on_dead,),
                                        name="rtmon-ws-send", daemon=True)
        self._writer.start()

    def _write_loop(self, on_dead) -> None:
        try:
            while True:
                item = self._outbox.get()
                if item is None:
                    return
                kind, payload = item
                if kind == "text":
                    self.ws.send_text(payload)
                else:
                    self.ws.send_binary(payload)
        except wsproto.WebSocketClosed:
            pass
        finally:
            self.dead = True
            on_dead()

    def offer(self, kind: str, payload) -> None:
        """Enqueue without ever blocking. A client that is behind loses its oldest
        pending frame and keeps the newest -- for a live display, fresher is truer."""
        if self.dead:
            return
        try:
            self._outbox.put_nowait((kind, payload))
        except queue.Full:
            try:
                self._outbox.get_nowait()
            except queue.Empty:
                pass
            try:
                self._outbox.put_nowait((kind, payload))
            except queue.Full:
                pass

    def close(self) -> None:
        self.dead = True
        try:
            self._outbox.put_nowait(None)
        except queue.Full:
            try:
                self._outbox.get_nowait()
                self._outbox.put_nowait(None)
            except Exception:
                pass

    def update(self, message: dict) -> None:
        with self.lock:
            if "paused" in message:
                self.paused = bool(message["paused"])
            if "window_s" in message:
                self.window_s = float(min(MAX_WINDOW_S, max(1.0, message["window_s"])))
            if "buckets" in message:
                self.buckets = int(min(MAX_BUCKETS, max(80, message["buckets"])))
            if "hr_window_s" in message:
                self.hr_window_s = float(min(MAX_HR_WINDOW_S, max(30.0, message["hr_window_s"])))
            if "channels" in message:
                self.channels = [str(c) for c in message["channels"]][:16]
            if "signal_view" in message:
                want = str(message["signal_view"])
                self.signal_view = want if want in proc.BANDS else "raw"

    def view(self):
        with self.lock:
            return (self.window_s, self.buckets, self.hr_window_s,
                    list(self.channels), self.paused, self.signal_view)


class App:
    def __init__(self, session_root: Path):
        self.hub = Hub(session_root)
        self.engine = Engine(self.hub)
        self.setup: Setup = setup_store.load_last() or setup_store.default_setup()
        self.engine.set_tracks(self.setup.tracks)
        self.aligner = TapAligner(self.hub)
        self._apply_ppg_latency()
        self.clients: set[Client] = set()
        self._clients_lock = threading.Lock()
        self._stop = threading.Event()
        self._broadcaster: threading.Thread | None = None
        self._filter_cache: dict[tuple, tuple[float, object]] = {}
        self._filter_lock = threading.Lock()

    # ---------------------------------------------------------------- setup
    def apply_setup(self, raw: dict) -> Setup:
        setup = Setup.from_json(raw)
        # Colour follows the entity, never its rank: a track that already has a slot
        # keeps it, and only the blanks are filled. Hence two passes — reserving every
        # existing colour first is what stops inserting a row above another from
        # repainting the one below it, which would silently re-label every trace on
        # the chart and in every screenshot taken so far.
        used = {t.color for t in setup.tracks if t.color}
        for track in setup.tracks:
            if track.color:
                continue
            track.color = next((c for c in setup_store.TRACK_COLORS if c not in used),
                               setup_store.TRACK_COLORS[len(used) % len(setup_store.TRACK_COLORS)])
            used.add(track.color)
        # One source of truth per BAND, not one overall: the rig measures two hearts
        # with two references (mic = fetal, PPG strap = maternal), and each band's
        # estimates are scored against its own.
        claimed: set[str] = set()
        for track in setup.tracks:
            if track.role == "sot":
                if track.band in claimed:
                    track.role = "estimate"
                else:
                    claimed.add(track.band)
        self.setup = setup
        self.engine.set_tracks(setup.tracks)
        self._apply_ppg_latency()
        setup_store.save_last(setup)
        return setup

    # Which setup field carries each source's timing correction. Both are corrections
    # ONTO the fiber, which is the reference (see rtmon.align).
    LATENCY_FIELDS = {"pvs": "ppg_latency_s", "mic": "mic_latency_s"}

    def latency_of(self, source_id: str) -> float:
        """The correction in force for a source, falling back to its default."""
        value = getattr(self.setup, self.LATENCY_FIELDS[source_id], None)
        if value is not None:
            return value
        return polar.PPG_PIPELINE_LATENCY_S if source_id == "pvs" else 0.0

    def _apply_ppg_latency(self) -> None:
        """Push the setup's measured corrections onto the sources.

        Applied live: both sources subtract their correction at push time, so a new
        value takes effect on a stream that is already running. That is required, not
        cosmetic -- tap alignment measures the residual under whatever is currently in
        force, so if the value did not take effect until the next arm, re-measuring
        would keep reading the old residual and repeated applies would diverge.
        """
        for source_id in self.LATENCY_FIELDS:
            source = self.hub.sources.get(source_id)
            if source is not None and hasattr(source, "latency_s"):
                source.latency_s = self.latency_of(source_id)

    def state(self) -> dict:
        return {
            "hub": self.hub.describe(),
            "setup": self.setup.to_json(),
            "tracks": self.engine.snapshot()["tracks"],
            "catalog": {
                **proc.describe(),
                "models": {family: [e.to_json() for e in entries]
                           for family, entries in model_registry.discover_all().items()},
                "colors": setup_store.TRACK_COLORS,
                # Every channel any source *could* provide, not just the streaming ones,
                # so a matrix can be built before the rig is armed (and so a track that
                # references a stopped device still shows what it is waiting for).
                "all_channels": self._all_channels(),
            },
            "presets": setup_store.list_presets(),
            # Included so a control response carries the current alignment phase and
            # corrections rather than leaving the page to wait for the next status frame.
            "align": self.aligner.state().to_json(),
            "latency": self._latency_block(),
        }

    def _latency_block(self) -> dict:
        """Per-channel timing correction in force, and whether it was measured."""
        return {ALIGN_TARGET_CHANNEL[s]: {"s": self.latency_of(s),
                                          "measured": getattr(self.setup, f) is not None}
                for s, f in App.LATENCY_FIELDS.items()}

    def _all_channels(self) -> list[dict]:
        live = self.hub.channel_map()
        seen: dict[str, dict] = {}
        for source in self.hub.sources.values():
            for channel in source.channels:
                entry = seen.setdefault(channel.id, {**channel.to_json(), "sources": []})
                entry["sources"].append(source.id)
                if channel.id in live:
                    entry["live"] = True
                    entry["source"] = live[channel.id].source_id
        return [seen[k] for k in sorted(seen)]

    # ----------------------------------------------------------- broadcast
    def start(self) -> None:
        self.engine.start()
        self._broadcaster = threading.Thread(target=self._broadcast_loop,
                                             name="rtmon-broadcast", daemon=True)
        self._broadcaster.start()

    def shutdown(self) -> None:
        self._stop.set()
        try:
            self.engine.stop()
        finally:
            if self.hub.recorder.active:
                self.hub.stop_recording()
            self.hub.stop_all()

    def add_client(self, client: Client) -> None:
        with self._clients_lock:
            self.clients.add(client)

    def drop_client(self, client: Client) -> None:
        with self._clients_lock:
            self.clients.discard(client)
        client.close()

    def _broadcast_loop(self) -> None:
        period = 1.0 / WAVE_FPS
        next_slow = 0.0
        while not self._stop.is_set():
            started = time.monotonic()
            with self._clients_lock:
                clients = list(self.clients)
            if clients:
                slow = started >= next_slow
                if slow:
                    next_slow = started + SLOW_INTERVAL
                try:
                    self._send_frames(clients, slow)
                except Exception:
                    traceback.print_exc()
            # Pace on elapsed time, so a heavy frame lowers the rate instead of
            # queueing frames the client will never catch up on.
            time.sleep(max(0.0, period - (time.monotonic() - started)))

    def _send_frames(self, clients: list[Client], slow: bool) -> None:
        now = self.hub.newest_time() or time.time()
        snapshot = self.engine.snapshot() if slow else None
        status = self._status(snapshot) if slow else None

        for client in clients:
            if client.dead:
                continue
            window_s, buckets, hr_window_s, channels, paused, signal_view = client.view()
            if paused and snapshot is None:
                continue     # backgrounded tab, and this is a waveform-only frame
            builder = wire.FrameBuilder(now)
            waves = []
            if not paused:
                for channel_id in channels:
                    envelope = self._envelope(channel_id, now, window_s, signal_view)
                    entry = wire.wave_entry(builder, channel_id, envelope, now, max_buckets=buckets)
                    if entry is not None:
                        waves.append(entry)
            builder.header["waves"] = waves

            if snapshot is not None:
                builder.header["tracks"] = [
                    self._track_entry(builder, item, now, hr_window_s)
                    for item in snapshot["tracks"]
                ]
                builder.header["status"] = status

            client.offer("bin", builder.build())

    def _envelope(self, channel_id: str, now: float, window_s: float, signal_view: str):
        """The min/max envelope for one channel, raw or bandpassed.

        Falls back to raw whenever there is no band to apply -- the PPG strap, or a
        channel whose sample rate cannot represent the band. That fallback is the
        behaviour, not a failure: the panel shows every channel either way, and only
        the ones the pipeline actually filters change when the view is switched.
        """
        if signal_view == "raw":
            return self.hub.envelope(channel_id, now, window_s)

        # Shared across clients and across frames for FILTER_TTL_S: two tabs on the
        # same view must not pay twice, and neither must consecutive frames.
        key = (channel_id, signal_view, round(window_s, 3))
        with self._filter_lock:
            hit = self._filter_cache.get(key)
            if hit is not None and now - hit[0] < FILTER_TTL_S:
                return hit[1]

        band = proc.BANDS[signal_view]["acoustic"]
        ref = self.hub.channel_map().get(channel_id)
        result = None
        if ref is not None and display.has_band(ref.channel.kind):
            got = self.hub.snapshot(channel_id, window_s)
            if got is not None:
                try:
                    result = display.band_envelope(got[0], got[1], ref.channel.kind,
                                                   band, ring.DISPLAY_BUCKET_HZ)
                except Exception:  # noqa: BLE001 - a display filter must never kill a frame
                    traceback.print_exc()
        if result is None:
            result = self.hub.envelope(channel_id, now, window_s)
        with self._filter_lock:
            self._filter_cache[key] = (now, result)
            # Bounded: window and band both vary, and a page left open for a shift
            # should not accumulate an entry per combination ever visited.
            if len(self._filter_cache) > 64:
                oldest = min(self._filter_cache, key=lambda k: self._filter_cache[k][0])
                del self._filter_cache[oldest]
        return result

    @staticmethod
    def _track_entry(builder: wire.FrameBuilder, item: dict, now: float, hr_window_s: float) -> dict:
        t = np.asarray(item["t"], dtype=np.float64)
        y = np.asarray(item["y"], dtype=np.float32)
        if t.size:
            keep = t >= now - hr_window_s
            t, y = t[keep], y[keep]
        return {"id": item["id"], **wire.series_entry(builder, t, y, now)}

    def _status(self, snapshot: dict | None) -> dict:
        """The block the page renders its chrome from.

        Deliberately the same shape the HTTP ``/api/state`` response can be folded into,
        so a control action and the live stream update the UI through one code path
        instead of the click waiting on the next frame.
        """
        tracks = snapshot["tracks"] if snapshot else []
        return {
            "align": self.aligner.state().to_json(),
            "latency": self._latency_block(),
            "recording": self.hub.recorder.status(),
            "sources": {sid: {"running": s.running, "error": s.error}
                        for sid, s in self.hub.sources.items()},
            "channels": [{"id": cid, "hz": round(self.hub.rate_of(cid), 1),
                          "silent": self.hub.is_silent(cid, SILENCE_WINDOW_S)}
                         for cid in sorted(self.hub.channel_map())],
            "tracks": [{k: v for k, v in item.items() if k not in ("t", "y")}
                       for item in tracks],
        }


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    server_version = "rtmon/1.0"
    protocol_version = "HTTP/1.1"
    app: App = None  # type: ignore[assignment]

    def log_message(self, fmt, *args):
        pass

    # -- helpers ------------------------------------------------------------
    def _json(self, obj, status=200):
        body = json.dumps(obj, default=_jsonable).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0) or 0)
        if length <= 0:
            return {}
        if length > 8 << 20:
            raise ValueError("request too large")
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def _static(self, rel: str):
        target = (STATIC_DIR / rel).resolve()
        if not str(target).startswith(str(STATIC_DIR.resolve())) or not target.is_file():
            return self._json({"error": "not found"}, status=404)
        ctype = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        body = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    # -- routing ------------------------------------------------------------
    def do_GET(self):
        parsed = urlparse(self.path)
        route = parsed.path
        try:
            if route == "/ws":
                return self._websocket()
            if route in ("/", "/index.html"):
                return self._static("index.html")
            if route == "/api/state":
                return self._json(self.app.state())
            if route == "/api/align":
                # A dedicated, cheap poll for the calibration wizard. The 1 Hz status
                # frame is far too slow to drive a "tap NOW" prompt, and relying on it
                # is why the panel could sit showing a stale phase until the page was
                # reloaded. The wizard polls this several times a second while open.
                return self._json({"align": self.app.aligner.state().to_json(),
                                   "latency": self.app._latency_block()})

            if route == "/api/presets":
                return self._json({"presets": setup_store.list_presets()})
            if route.startswith("/static/"):
                return self._static(route[len("/static/"):])
            if route.lstrip("/") in ("app.js", "style.css"):
                return self._static(route.lstrip("/"))
            return self._json({"error": "not found"}, status=404)
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            return self._json({"error": f"{type(exc).__name__}: {exc}"}, status=500)

    def do_POST(self):
        parsed = urlparse(self.path)
        route = parsed.path
        query = parse_qs(parsed.query)
        try:
            body = self._body()
            app = self.app

            if route == "/api/probe":
                app.hub.probe_all(deep=bool(body.get("deep", True)))
                return self._json(app.state())

            if route == "/api/source/start":
                source = app.hub.sources.get(body.get("id", ""))
                # Optional per-source config rides along with the arm request; today
                # that is the microphone's input device (a name, so it survives
                # replugging — see MicrophoneSource.device).
                if (source is not None and not source.running
                        and body.get("device") not in (None, "")
                        and hasattr(source, "device")):
                    source.device = body["device"]
                app.hub.start_source(body["id"])
                return self._json(app.state())

            if route == "/api/source/stop":
                app.hub.stop_source(body["id"])
                return self._json(app.state())

            if route == "/api/setup":
                app.apply_setup(body)
                return self._json(app.state())

            if route == "/api/tracks/clear":
                app.engine.clear_track(body.get("id"))
                return self._json({"ok": True})

            # Both return the full state, not just the session: the page derives the
            # Record button and its timer from state, and a response that omitted it
            # would leave the button stale until the next status frame.
            if route == "/api/record/start":
                session = app.hub.start_recording(setup=app.setup.to_json())
                return self._json({**app.state(), "session": session.to_json()})

            if route == "/api/record/stop":
                session = app.hub.stop_recording()
                return self._json({**app.state(), "session": session.to_json() if session else None})

            if route == "/api/presets/save":
                name = setup_store.save_preset(app.setup, body.get("name", "preset"))
                return self._json({"saved": name, "presets": setup_store.list_presets()})

            if route == "/api/presets/load":
                loaded = setup_store.load_preset(body["name"])
                app.apply_setup(loaded.to_json())
                return self._json(app.state())

            if route == "/api/presets/delete":
                setup_store.delete_preset(body["name"])
                return self._json({"presets": setup_store.list_presets()})

            # These return the FULL state, like every other control endpoint. Returning
            # a bare {"align": ...} broke the page: the client folds control responses
            # into its state, so a partial one dereferenced a missing `hub` and threw --
            # which is what made Discard error out.
            if route == "/api/align/start":
                app.aligner.start(body["reference"], body.get("targets") or [])
                return self._json(app.state())

            if route == "/api/align/cancel":
                app.aligner.cancel()
                return self._json(app.state())

            if route == "/api/align/apply":
                result = app.aligner.state()
                if result.phase != "done":
                    return self._json({"error": "no completed measurement to apply"}, status=400)
                # Each tap measures the RESIDUAL under the correction currently in force,
                # so it ADJUSTS that value rather than replacing it. Positive lag means
                # the channel is stamped late, which needs more subtracted.
                low, high = polar.LATENCY_RANGE_S
                changes, rejected = {}, []
                for source_id, field in App.LATENCY_FIELDS.items():
                    channel = ALIGN_TARGET_CHANNEL[source_id]
                    entry = result.results.get(channel)
                    if not entry or not entry.get("ok"):
                        continue
                    current = app.latency_of(source_id)
                    proposed = round(current + entry["lag_s"], 4)
                    if not (low <= proposed <= high):
                        rejected.append(f"{channel} would need {proposed * 1000:.0f} ms")
                        continue
                    setattr(app.setup, field, proposed)
                    changes[channel] = {"previous_s": round(current, 4),
                                        "measured_lag_s": round(entry["lag_s"], 4),
                                        "latency_s": proposed}
                if not changes:
                    detail = ("; ".join(rejected) + ", outside the sane range "
                              f"({low * 1000:.0f}…{high * 1000:.0f} ms)") if rejected \
                             else "no channel produced a usable measurement"
                    return self._json({"error": f"nothing applied — {detail}."}, status=400)
                app.apply_setup(app.setup.to_json())
                app.aligner._set(applied=True)
                # Return the arithmetic per channel: the measured lag and the resulting
                # correction are different numbers, and showing only the second looks
                # like the app reporting one figure and saving another.
                return self._json({**app.state(), "changes": changes,
                                   "rejected": rejected})

            if route == "/api/align/reset":
                app.setup.ppg_latency_s = None
                app.setup.mic_latency_s = None
                app.apply_setup(app.setup.to_json())
                return self._json(app.state())

            if route == "/api/recover":
                written = recover(Path(body["directory"]))
                return self._json({"written": [str(p) for p in written]})

            return self._json({"error": "not found"}, status=404)
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            return self._json({"error": f"{type(exc).__name__}: {exc}"}, status=400)

    # -- websocket ----------------------------------------------------------
    def _websocket(self):
        if not wsproto.is_upgrade(self.headers):
            return self._json({"error": "expected a websocket upgrade"}, status=400)
        if not wsproto.origin_is_local(self.headers):
            return self._json({"error": "cross-origin websocket refused"}, status=403)

        self.send_response(101, "Switching Protocols")
        for key, value in wsproto.handshake_headers(self.headers["Sec-WebSocket-Key"]):
            self.send_header(key, value)
        self.end_headers()
        try:
            self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except Exception:
            pass
        try:
            # Best-effort send timeout: a writer wedged on a dead socket exits in
            # seconds instead of waiting out the TCP retransmission timer.
            self.connection.setsockopt(socket.SOL_SOCKET, socket.SO_SNDTIMEO,
                                       struct.pack("ll", 10, 0))
        except Exception:
            pass

        ws = wsproto.WebSocket(self.rfile, self.wfile)
        client = Client(ws)
        # Writer first, hello second, THEN visible to the broadcaster — so the hello
        # (which carries full state) is always the first thing on the wire and cannot
        # be displaced by frames racing in ahead of it.
        client.start_writer(lambda: self.app.drop_client(client))
        client.offer("text", json.dumps({"type": "hello", "state": self.app.state()},
                                        default=_jsonable))
        self.app.add_client(client)
        try:
            while True:
                opcode, payload = ws.recv()
                if opcode != wsproto.OP_TEXT:
                    continue
                try:
                    message = json.loads(payload.decode("utf-8"))
                except Exception:
                    continue
                if message.get("type") == "view":
                    client.update(message)
                elif message.get("type") == "state":
                    client.offer("text", json.dumps(
                        {"type": "state", "state": self.app.state()}, default=_jsonable))
        except wsproto.WebSocketClosed:
            pass
        except Exception:
            pass
        finally:
            self.app.drop_client(client)
            self.close_connection = True


def _jsonable(obj):
    """Fallback encoder for the numpy scalars that leak out of the engine snapshot."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Track):
        return obj.to_json()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"not JSON serialisable: {type(obj).__name__}")


def main():
    parser = argparse.ArgumentParser(description="Real-time fiber recording + fetal HR monitor")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8720)
    parser.add_argument("--sessions", default=str(DEFAULT_SESSION_ROOT),
                        help="directory recordings are written under")
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--no-probe", action="store_true",
                        help="skip the startup hardware probe (the UI can rescan)")
    args = parser.parse_args()

    app = App(Path(args.sessions))
    Handler.app = app

    found = {family: len(entries) for family, entries in model_registry.discover_all().items()}
    print("[rtmon] models: " + ", ".join(f"{k} x{v}" for k, v in found.items()))

    app.start()
    try:
        httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    except OSError as exc:
        app.shutdown()
        raise SystemExit(
            f"[rtmon] cannot bind {args.host}:{args.port} — {exc}\n"
            f"        Another rtmon is probably already running; use --port to pick "
            f"another, or stop it with:  lsof -ti :{args.port} | xargs kill") from None
    httpd.daemon_threads = True

    if not args.no_probe:
        # Probe AFTER binding, in the background. A deep probe holds a BLE scan and a
        # unit-open for several seconds, and running it before serve_forever left the
        # auto-opened browser staring at a refused connection — the page then had to
        # be reloaded by hand once the server finally came up. The UI shows probe
        # results as they land (the Setup drawer refetches state when opened).
        def _startup_probe():
            print("[rtmon] probing devices…")
            for source_id, probe in app.hub.probe_all(deep=True).items():
                mark = "ok  " if probe.ok else "--  "
                print(f"[rtmon] {mark}{source_id:<12} {probe.detail}")
        threading.Thread(target=_startup_probe, name="rtmon-probe", daemon=True).start()
    url = f"http://{args.host}:{args.port}"
    print(f"[rtmon] serving on {url}  (Ctrl-C to stop)")
    if not args.no_browser and not os.environ.get("RTMON_NO_BROWSER"):
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[rtmon] shutting down")
    finally:
        app.shutdown()
        httpd.shutdown()


if __name__ == "__main__":
    main()
