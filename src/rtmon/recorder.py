"""Session recording, in the layout ``analyze.data`` already knows how to load.

A finished session directory is exactly what the offline pipelines expect::

    session-01/
      ps4000.npy       (N, 3)  float64: [t, 1A, 1B]
      ps3000a.npy      (N, 5)  float64: [t, 2A, 2B, 2C, 2D]
      pvs.npy          (N, 5)  float64: [t, PPG0, PPG1, PPG2, ambient]
      microphone.wav
      session.json     what was running, and the track setup it was recorded under

Column 0 is seconds *relative to the session start*, which is what
``analyze.data.load_data`` divides to recover the sample rate -- and the reason time
is written relative rather than absolute: at epoch magnitudes a float64 difference
of 200 us is still exact, but nothing downstream expects 1.7e9 on its x axis.

Acquisition threads never touch the filesystem. They hand blocks to a queue and one
writer thread appends them to a flat ``.raw`` file whose bytes are already in the
final array's layout, so closing a session is a header write plus a
``copyfileobj`` -- constant memory and disk-speed, no matter how long the recording
ran. The ``.raw`` and its ``.json`` sidecar are also what makes a crashed session
recoverable (``rtmon-recover``); the Qt app's periodic ``tmp/*.npy`` dumps existed
for the same reason but cost a full re-slice of a multi-hundred-MB array every
300 s.
"""

from __future__ import annotations

import json
import queue
import threading
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

RAW_SUFFIX = ".raw"
META_SUFFIX = ".raw.json"
# Bounds how much a hard kill can cost: samples sitting in the stream's 1 MB userspace
# buffer are gone, samples handed to the kernel survive. Half a second of six fibers is
# ~120 kB, so flushing this often costs a write() per block and nothing else.
FLUSH_INTERVAL = 0.5
INT16_FULL_SCALE = 32768.0


@dataclass
class StreamSpec:
    """How one source's samples become files."""

    name: str                      # base filename ("ps3000a", "microphone", ...)
    columns: list[str]             # channel ids, in column order
    audio: bool = False            # write a WAV (mono int16) instead of an .npy
    audio_hz: int = 0


@dataclass
class _Stream:
    spec: StreamSpec
    raw: object | None
    wav: object | None = None
    rows: int = 0
    dropped: int = 0
    t0: float | None = None      # offset of the first sample from the session start


@dataclass
class Session:
    directory: Path
    started_at: float
    stopped_at: float | None = None
    streams: dict = field(default_factory=dict)

    @property
    def elapsed(self) -> float:
        return (self.stopped_at or time.time()) - self.started_at

    def to_json(self) -> dict:
        return {
            "directory": str(self.directory),
            "name": self.directory.name,
            "started_at": self.started_at,
            "stopped_at": self.stopped_at,
            "elapsed": self.elapsed,
            # ``t0`` is each stream's first sample relative to the session start. The
            # .npy files carry it in their own time column, but a WAV has no time
            # axis, so for audio this is the only record that it began a few tens of
            # milliseconds off zero.
            "streams": {k: {"rows": v.rows, "columns": v.spec.columns,
                            "dropped": v.dropped, "t0": v.t0}
                        for k, v in self.streams.items()},
        }


class Recorder:
    """Owns at most one open session."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self._session: Session | None = None
        self._queue: queue.SimpleQueue | None = None
        self._writer: threading.Thread | None = None
        self._lock = threading.Lock()
        self._error: str | None = None

    # ------------------------------------------------------------------ state
    @property
    def active(self) -> bool:
        return self._session is not None and self._session.stopped_at is None

    def status(self) -> dict:
        with self._lock:
            if self._session is None:
                return {"active": False, "error": self._error}
            return {"active": self.active, "error": self._error,
                    "queued": self._queue.qsize() if self._queue else 0,
                    **self._session.to_json()}

    # ------------------------------------------------------------------ start
    def start(self, specs: dict[str, StreamSpec], setup: dict | None = None) -> Session:
        """Open ``session-NN/`` and a stream per source id in ``specs``."""
        with self._lock:
            if self.active:
                raise RuntimeError("a recording is already running")
            directory = _next_session_dir(self.root)
            directory.mkdir(parents=True)
            session = Session(directory=directory, started_at=time.time())

            for source_id, spec in specs.items():
                session.streams[source_id] = _open_stream(directory, spec, session.started_at)

            (directory / "session.json").write_text(json.dumps({
                "started_at": session.started_at,
                "started_at_iso": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(session.started_at)),
                "setup": setup or {},
            }, indent=2))

            self._error = None
            self._session = session
            self._queue = queue.SimpleQueue()
            self._writer = threading.Thread(target=self._drain, name="rtmon-recorder", daemon=True)
            self._writer.start()
            return session

    def add_stream(self, source_id: str, spec: StreamSpec) -> None:
        """Attach a source that came up after the session opened.

        A device reconnecting mid-session is routine (the BLE strap sleeps, a scope
        gets re-plugged). Its file simply starts partway in; because time is recorded
        relative to the *session* start, not the stream's, the offline loaders still
        line it up against everything else.
        """
        with self._lock:
            if not self.active or self._session is None:
                return
            if source_id in self._session.streams:
                return
            self._session.streams[source_id] = _open_stream(
                self._session.directory, spec, self._session.started_at)

    # ------------------------------------------------------------------ write
    def write(self, source_id: str, t: np.ndarray, x: np.ndarray) -> None:
        """Queue a block. Called from acquisition threads; never blocks on I/O."""
        q = self._queue
        if q is None:
            return
        q.put((source_id, t, x))

    def _drain(self) -> None:
        session = self._session
        assert session is not None
        t0 = session.started_at
        last_flush = time.monotonic()
        while True:
            item = self._queue.get()
            if item is None:
                break
            source_id, t, x = item
            stream = session.streams.get(source_id)
            if stream is None:
                continue
            try:
                self._append(stream, t0, t, x)
            except Exception as exc:  # noqa: BLE001 - a failing disk must not kill acquisition
                self._error = f"{type(exc).__name__}: {exc}"
                stream.dropped += int(t.shape[0])
            now = time.monotonic()
            if now - last_flush > FLUSH_INTERVAL:
                last_flush = now
                for s in session.streams.values():
                    try:
                        s.raw.flush()
                    except Exception:
                        pass

    @staticmethod
    def _append(stream: _Stream, t0: float, t: np.ndarray, x: np.ndarray) -> None:
        n = t.shape[0]
        if n == 0:
            return
        if stream.t0 is None:
            stream.t0 = float(t[0] - t0)
        stream.rows += n

        if stream.wav is not None:
            # Audio gets a WAV and nothing else: that is the file the offline loaders
            # read, and a second float64 copy of the same samples would be several
            # times larger for no additional information (the device delivers int16,
            # and the /32768 scaling round-trips exactly).
            pcm = np.clip(x[:, 0].astype(np.float32) * INT16_FULL_SCALE,
                          -INT16_FULL_SCALE, INT16_FULL_SCALE - 1).astype("<i2")
            stream.wav.writeframes(pcm.tobytes())
            return

        # The row layout written here IS the final .npy body: float64, C-order,
        # [t_rel, ch...]. That is what lets close() finish with a byte copy.
        row = np.empty((n, x.shape[1] + 1), dtype=np.float64)
        row[:, 0] = t - t0
        row[:, 1:] = x
        stream.raw.write(row.tobytes())

    # ------------------------------------------------------------------- stop
    def stop(self) -> Session | None:
        with self._lock:
            session, self._session = self._session, None
            q, self._queue = self._queue, None
            writer, self._writer = self._writer, None
        if session is None:
            return None

        if q is not None:
            q.put(None)
        if writer is not None:
            writer.join(timeout=30.0)

        for stream in session.streams.values():
            if stream.wav is not None:
                try:
                    stream.wav.close()
                except Exception:
                    pass
                continue
            try:
                stream.raw.flush()
                stream.raw.close()
            except Exception:
                pass
            raw_path = session.directory / f"{stream.spec.name}{RAW_SUFFIX}"
            try:
                finalize(raw_path, len(stream.spec.columns) + 1)
            except Exception as exc:  # noqa: BLE001
                self._error = f"finalize {stream.spec.name}: {type(exc).__name__}: {exc}"

        session.stopped_at = time.time()
        meta_path = session.directory / "session.json"
        try:
            meta = json.loads(meta_path.read_text())
        except Exception:
            meta = {}
        meta.update({"stopped_at": session.stopped_at, "elapsed": session.elapsed,
                     "streams": session.to_json()["streams"]})
        meta_path.write_text(json.dumps(meta, indent=2))
        self._session = session
        return session


def _open_stream(directory: Path, spec: StreamSpec, started_at: float) -> _Stream:
    """Open the output file(s) for one source: a WAV for audio, an append stream
    that becomes a ``.npy`` for everything else."""
    if spec.audio:
        wav = wave.open(str(directory / f"{spec.name}.wav"), "wb")
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(int(spec.audio_hz))
        return _Stream(spec=spec, raw=None, wav=wav)

    raw_path = directory / f"{spec.name}{RAW_SUFFIX}"
    # The sidecar is what rtmon-recover needs to finalize this file if the process
    # dies: the row width is not recoverable from the raw bytes alone.
    (directory / f"{spec.name}{META_SUFFIX}").write_text(json.dumps({
        "name": spec.name, "columns": spec.columns,
        "dtype": "<f8", "n_cols": len(spec.columns) + 1,
        "started_at": started_at,
    }, indent=2))
    return _Stream(spec=spec, raw=open(raw_path, "wb", buffering=1 << 20))


def finalize(raw_path: Path, n_cols: int, keep_raw: bool = False) -> Path:
    """Turn ``<name>.raw`` into ``<name>.npy`` without loading it.

    The raw bytes are already the array body, so this writes a ``.npy`` header and
    copies. A 3-hour six-fiber session is ~7 GB and still finalises in constant memory.
    """
    raw_path = Path(raw_path)
    npy_path = raw_path.with_suffix("")           # strips ".raw"
    npy_path = npy_path.with_suffix(".npy") if npy_path.suffix else Path(str(npy_path) + ".npy")

    size = raw_path.stat().st_size
    row_bytes = 8 * n_cols
    rows = size // row_bytes
    if rows == 0:
        raw_path.unlink(missing_ok=True)
        Path(str(raw_path) + ".json").unlink(missing_ok=True)
        return npy_path

    with open(npy_path, "wb") as out:
        np.lib.format.write_array_header_2_0(
            out, {"descr": "<f8", "fortran_order": False, "shape": (int(rows), int(n_cols))})
        with open(raw_path, "rb") as src:
            # A trailing partial row (killed mid-write) is excluded by the header's
            # row count, so copy exactly the whole rows and no more.
            _copy_n(src, out, rows * row_bytes)

    if not keep_raw:
        raw_path.unlink(missing_ok=True)
        Path(str(raw_path) + ".json").unlink(missing_ok=True)
    return npy_path


def _copy_n(src, dst, nbytes: int, chunk: int = 1 << 22) -> None:
    remaining = nbytes
    while remaining > 0:
        buf = src.read(min(chunk, remaining))
        if not buf:
            break
        dst.write(buf)
        remaining -= len(buf)


def recover(directory: Path, keep_raw: bool = False) -> list[Path]:
    """Finalize every leftover ``.raw`` in ``directory`` (after a crash or kill)."""
    directory = Path(directory)
    written = []
    for meta_path in sorted(directory.glob(f"*{META_SUFFIX}")):
        meta = json.loads(meta_path.read_text())
        raw_path = directory / f"{meta['name']}{RAW_SUFFIX}"
        if raw_path.exists():
            written.append(finalize(raw_path, int(meta["n_cols"]), keep_raw=keep_raw))
    return written


def _next_session_dir(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    i = 1
    while True:
        candidate = root / f"session-{i:02d}"
        if not candidate.exists():
            return candidate
        i += 1
