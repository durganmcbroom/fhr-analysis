"""Polar Verity Sense PPG source (BLE).

Two halves live here:

* :func:`decode_frame` and :func:`_decode_deltas` -- the PMD wire format, rewritten in
  numpy. The Qt app decoded each frame by building a bit *string*
  (``''.join(format(b, '08b')[::-1] ...)``) and then constructing a ``BitArray`` per
  delta, four per sample; one ``np.unpackbits`` plus a weighted sum now decodes the
  whole frame at once. The two agree bit for bit at every legal delta width.
* :class:`PolarSource` -- which runs *no Bluetooth at all*. It supervises
  ``rtmon.sources.polar_worker`` in a subprocess and reads decoded samples off its
  stdout. See that module for why: on macOS, bleak from any non-main thread aborts the
  process outright, and the server's main thread belongs to the HTTP server.

The practical consequence is worth stating plainly: the strap can fail in every way a
Bluetooth device fails -- unpaired, asleep, out of range, permission denied, stack
wedged -- and the worst outcome is this source reporting an error while the recording
continues.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass

import numpy as np

from rtmon.sources.base import KIND_PPG, Channel, Probe, Sink, Source

BATTERY_UUID = "00002a19-0000-1000-8000-00805f9b34fb"
PMD_CONTROL = "fb005c81-02e7-f387-1cad-8acd2d8df0c8"
PMD_DATA = "fb005c82-02e7-f387-1cad-8acd2d8df0c8"

PPG_START = bytearray([0x02, 0x01, 0x00, 0x01, 0x37, 0x00, 0x01, 0x01, 0x16, 0x00, 0x04, 0x01, 0x04])
PPG_STOP = bytearray([0x03, 0x01])

DEFAULT_NAME = "Polar Sense DE957E2E"
NOMINAL_HZ = 55.0
SCAN_TIMEOUT = 6.0
N_CHANNELS = 4

# Timing correction applied to every PPG sample: how much further back the samples have
# to be moved, on top of what the measured clock offset already accounts for. Positive
# moves them earlier (the usual direction -- a PMD sample is already stale when its
# notification arrives, because the strap batches, the BLE connection interval adds
# more, and the host stack adds the rest); negative moves them later, which is a real
# and reachable case (see LATENCY_RANGE_S).
#
# This is the alignment that matters -- pvs.npy is consumed by analyze.sot with its time
# column taken at face value, so whatever the recorder writes IS the PPG's alignment
# against the fibers.
#
# 0.5 s is the value the Qt app measured. It had a hardcoded
#     timestamp = <device ns>/1e9 + 1211010636.1   # empirically determined
# whose constant maps device-time 0 to 2008-05-17, not to Polar's 2000-01-01 epoch --
# it is a calibration of one strap's free-running clock, not an epoch conversion, and
# is wrong for any strap that has since been reset or re-paired. Beside it sat
#     # print(time.time() - timestamp)  # approx. 0.5
# which is the part worth keeping: with that calibration applied, samples landed ~0.5 s
# behind arrival. So the constant was doing two jobs, and only the second generalises.
# The clock offset is now measured per strap (see decode_frame); this is the latency.
#
# Override with RTMON_PPG_LATENCY_S once you can measure it on your own rig.
PPG_PIPELINE_LATENCY_S = float(os.environ.get("RTMON_PPG_LATENCY_S", "0.5"))

# Bounds on the CORRECTION, which is a different quantity from the raw transport delay
# and unlike it may be negative.
#
# The raw delay is of course non-negative -- samples cannot arrive before they were
# taken. But this value is not that. Timestamps are built as
#
#     stamp = device_ts + min_delay - latency_s
#
# where ``min_delay`` (the smallest arrival delay seen) has ALREADY absorbed some of the
# transport delay. What remains to subtract is a residual, and it goes negative whenever
# min_delay overshoots -- which happens routinely: the strap's crystal is not matched to
# the host's, so on a long session a fast-running device clock keeps pushing the observed
# minimum down past the true latency. The reference chain has its own residual too. Above
# all, the tap measurement is empirical: if it says the PPG sits 30 ms EARLY against the
# fiber, the correction has to be able to express that, and a floor at zero would make
# the calibration unable to record what was actually measured.
#
# So the bound here is only a sanity check against a corrupt or hand-edited file, not a
# physical claim. Two seconds either way is far outside anything BLE or clock drift
# produces.
LATENCY_RANGE_S = (-2.0, 2.0)


def valid_latency(value) -> float | None:
    """A stored timing correction if it is within :data:`LATENCY_RANGE_S`, else ``None``.

    Discarded rather than clamped: a value outside the range is a corrupt calibration,
    not a large one, and pinning it to a boundary would leave a wrong answer wearing the
    look of a deliberate setting. Falling back to the documented default is the honest
    recovery.
    """
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    low, high = LATENCY_RANGE_S
    return value if low <= value <= high else None

# Ignore an improved clock-offset estimate unless it beats the best so far by this
# much, so ordinary jitter does not keep nudging the time base.
_OFFSET_IMPROVE_S = 0.005

# The strap's PMD stream is four channels in a fixed order: three photodiodes and one
# ambient-light reference. AMB is not a pulse signal -- it is what the sensor sees with
# the LEDs contributing nothing, i.e. room light leaking under the band. It exists to be
# *subtracted* from PPG0-2 (and to show when a movement artefact is really a light
# artefact); on its own it carries no heartbeat, which is why nothing plots it by default.
PVS_CHANNELS = (
    Channel(id="PPG0", label="PPG 0", kind=KIND_PPG,
            note="Photodiode 0 — the main pulse channel; this is the one to feed a PPG track."),
    Channel(id="PPG1", label="PPG 1", kind=KIND_PPG,
            note="Photodiode 1 — same pulse from a second detector, useful when 0 is noisy."),
    Channel(id="PPG2", label="PPG 2", kind=KIND_PPG,
            note="Photodiode 2 — third detector."),
    Channel(id="AMB", label="Ambient light", kind=KIND_PPG,
            note="Ambient-light reference, not a pulse: room light reaching the sensor with "
                 "the LEDs off. Used to cancel lighting artefacts from PPG0–2; has no "
                 "heartbeat of its own."),
)


# ---------------------------------------------------------------------------
# PMD frame decoding (pure, and unit-tested against the original implementation)
# ---------------------------------------------------------------------------

def _decode_deltas(payload: bytes, offset: int, delta_size: int, sample_count: int):
    """``(sample_count, 4)`` signed deltas from a packed PMD delta block.

    Bits are little-endian within each byte and each delta is ``delta_size`` bits,
    least-significant bit first, four channels per sample.
    """
    n_bytes = delta_size * sample_count // 2      # 4 channels * delta_size bits / 8
    chunk = np.frombuffer(payload, dtype=np.uint8, count=n_bytes, offset=offset)
    bits = np.unpackbits(chunk, bitorder="little")
    n_deltas = sample_count * 4
    fields = bits[:n_deltas * delta_size].reshape(n_deltas, delta_size).astype(np.int64)
    weights = (1 << np.arange(delta_size, dtype=np.int64))
    raw = fields @ weights
    sign = np.int64(1) << (delta_size - 1)        # two's-complement sign extension
    raw = (raw ^ sign) - sign
    return raw.reshape(sample_count, 4), offset + n_bytes


def decode_frame(data: bytes, state: dict, hz: float = NOMINAL_HZ,
                 latency_s: float | None = None):
    """One PMD notification -> ``(t_abs, x)``, or ``None`` if it carries no samples.

    ``state`` is a mutable dict carrying ``last_ts``, ``offset`` and ``last_emitted``
    between calls (the frame format is differential in both value and time).
    """
    if len(data) < 22 or data[0] != 0x01 or data[9] != 0x80:
        return None                                # not a PPG delta frame

    device_ts = int.from_bytes(data[1:9], "little", signed=False) / 1e9

    ref = np.array([int.from_bytes(data[10 + 3 * i: 13 + 3 * i], "little", signed=True)
                    for i in range(4)], dtype=np.int64)
    blocks = [ref[None, :]]
    offset = 22
    while offset + 2 <= len(data):
        delta_size, sample_count = data[offset], data[offset + 1]
        offset += 2
        if delta_size == 0 or sample_count == 0:
            break
        deltas, offset = _decode_deltas(data, offset, delta_size, sample_count)
        blocks.append(deltas)

    # Deltas are relative to the running value, so one cumulative sum over the
    # reference row plus every delta row reconstructs the whole frame at once.
    samples = np.cumsum(np.concatenate(blocks, axis=0), axis=0)
    n = samples.shape[0]

    # Map the strap's free-running clock onto the wall clock, then back the samples up
    # by the transport latency so they land where they were actually taken.
    #
    #   arrival = capture + latency  =>  arrival - device_ts = offset + latency
    #
    # Latency is one-sided (BLE only ever delays), so the SMALLEST delay seen is the
    # closest look at the true offset -- a single first-frame reading bakes in whatever
    # jitter that one frame happened to carry. Track the minimum, then subtract the
    # pipeline latency, which is what the Qt app's hardcoded constant was silently
    # providing (see PPG_PIPELINE_LATENCY_S).
    delay = time.time() - device_ts
    best = state.get("min_delay")
    if best is None or delay < best - _OFFSET_IMPROVE_S:
        state["min_delay"] = best = delay
    if latency_s is None:
        latency_s = PPG_PIPELINE_LATENCY_S
    stamp = device_ts + best - latency_s

    last_ts = state.get("last_ts")
    if last_ts is None:
        t = stamp - (n - 1 - np.arange(n, dtype=np.float64)) / hz
    else:
        # Spread the samples evenly between the previous frame's timestamp and this
        # one -- the strap batches a variable number of samples per notification.
        step = (stamp - last_ts) / n
        t = last_ts + step * np.arange(1, n + 1, dtype=np.float64)
    state["last_ts"] = stamp

    last_emitted = state.get("last_emitted", 0.0)
    if t[0] <= last_emitted:
        keep = t > last_emitted
        if not keep.any():
            return None
        t, samples = t[keep], samples[keep]
    state["last_emitted"] = float(t[-1])

    return t, samples.astype(np.float32)


# ---------------------------------------------------------------------------
# Source
# ---------------------------------------------------------------------------

@dataclass
class PolarSource(Source):
    id: str = "pvs"
    label: str = "Polar Verity Sense (PPG)"
    nominal_hz: float = NOMINAL_HZ
    history_seconds: float = 300.0     # 55 Hz is cheap; keep a long trend
    device_name: str = DEFAULT_NAME
    # Transport latency actually in force. Starts at the inherited default and is
    # replaced by a tap measurement (see rtmon.align) once one has been taken.
    latency_s: float = PPG_PIPELINE_LATENCY_S

    def __post_init__(self):
        super().__post_init__()
        self.channels = PVS_CHANNELS
        self._proc: subprocess.Popen | None = None
        self._readers: list[threading.Thread] = []
        self._stop = threading.Event()
        self._ready_seen = False       # worker reported "ready" (notify stream is up)
        self._last_status = ""         # last stderr line, kept for error context
        self._last_out: float | None = None   # newest corrected time emitted

    def _worker_argv(self, *extra: str) -> list[str]:
        return [sys.executable, "-m", "rtmon.sources.polar_worker",
                "--name", self.device_name, *extra]

    # ---------------------------------------------------------------- probe
    def probe(self, deep: bool = True) -> Probe:
        try:
            import bleak  # noqa: F401
        except Exception as exc:
            return Probe(False, f"bleak not installed: {exc}",
                         "poetry install --with record")
        if not deep:
            return Probe(True, "bleak importable (not scanned)")

        try:
            done = subprocess.run(self._worker_argv("--scan"), capture_output=True,
                                  timeout=SCAN_TIMEOUT + 15)
        except subprocess.TimeoutExpired:
            return Probe(False, "BLE scan timed out",
                         "The Bluetooth stack may be wedged; toggling Bluetooth usually clears it.")
        if done.returncode != 0 or not done.stdout.strip():
            # Includes the native-abort case, which is exactly why this runs out of
            # process: here it is a return code instead of a dead server.
            return Probe(False, *_worker_failure(done))
        try:
            result = json.loads(done.stdout.decode())
        except Exception:
            return Probe(False, "BLE scan produced no usable result")
        return Probe(bool(result.get("ok")), result.get("detail", ""),
                     "" if result.get("ok") else
                     "Wake the strap (it sleeps when idle) and keep it in range.")

    # ------------------------------------------------------------ lifecycle
    def start(self, sink: Sink) -> None:
        if self.running:
            return
        self._sink = sink
        self._error = None
        self._ready_seen = False
        self._last_status = ""
        self._last_out = None
        self._stop.clear()

        self._proc = subprocess.Popen(
            self._worker_argv(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)

        ready = threading.Event()
        self._readers = [
            threading.Thread(target=self._read_samples, name="rtmon-pvs-data", daemon=True),
            threading.Thread(target=self._read_status, args=(ready,),
                             name="rtmon-pvs-status", daemon=True),
        ]
        for thread in self._readers:
            thread.start()

        ready.wait(timeout=SCAN_TIMEOUT + 20)
        if not self._ready_seen:
            # The worker never reached "ready": it died (permission SIGABRT, missing
            # bleak), reported an error, or hung. The old check passed whenever the
            # status reader unblocked -- which its finally-clause does on worker death
            # too -- so a strap that aborted instantly still armed as "streaming" and
            # then just never produced a sample. Require the explicit ready event.
            proc = self._proc
            code = proc.poll() if proc else None
            message = self._error or _worker_exit_message(code, self._last_status)
            self.stop()
            self._error = message      # after stop(), so the card keeps the reason
            raise RuntimeError(message)
        self._running.set()

    def _read_samples(self) -> None:
        """Pull ``[u32 n][f64 t*n][f32 x*4n]`` frames off the worker's stdout."""
        stream = self._proc.stdout
        try:
            while not self._stop.is_set():
                header = _read_exact(stream, 4)
                if header is None:
                    break
                (n,) = np.frombuffer(header, dtype="<u4")
                if n == 0 or n > 4096:
                    break                       # framing lost; let the supervisor restart it
                body = _read_exact(stream, int(n) * 8 + int(n) * N_CHANNELS * 4)
                if body is None:
                    break
                t = np.frombuffer(body, dtype="<f8", count=int(n))
                x = np.frombuffer(body, dtype="<f4", count=int(n) * N_CHANNELS,
                                  offset=int(n) * 8).reshape(int(n), N_CHANNELS)
                got = self._correct(t, x)
                if got is not None:
                    self._sink(*got)
        except Exception as exc:  # noqa: BLE001
            self._error = f"{type(exc).__name__}: {exc}"
        finally:
            # EOF while nobody asked it to stop means the worker died mid-stream.
            # Leave a reason behind: without it the card silently flips back to
            # "available" and the strap looks like it never really connected.
            if not self._stop.is_set() and self._error is None and self._ready_seen:
                proc = self._proc
                self._error = _worker_exit_message(proc.poll() if proc else None,
                                                   self._last_status)
            self._running.clear()

    def _correct(self, t: np.ndarray, x: np.ndarray):
        """Apply the timing correction, live, keeping emitted time strictly increasing.

        Read once per block so tap alignment takes effect immediately on a streaming
        strap. Raising the correction shifts samples earlier, which can overlap what has
        already been emitted; the overlap is dropped once (at most |change| seconds) so
        the rings never see time run backwards.
        """
        t = t - self.latency_s
        last = self._last_out
        if last is not None and t[0] <= last:
            keep = t > last
            if not keep.any():
                return None
            t, x = t[keep], x[keep]
        self._last_out = float(t[-1])
        return t, x

    def _read_status(self, ready: threading.Event) -> None:
        """Consume the worker's JSON status lines (and unblock start() on 'ready')."""
        stream = self._proc.stderr
        try:
            for raw in iter(stream.readline, b""):
                if self._stop.is_set():
                    break
                line = raw.decode(errors="replace").strip()
                if line:
                    self._last_status = line
                try:
                    message = json.loads(line)
                except Exception:
                    continue      # pyobjc / bleak noise on stderr, kept in _last_status
                event = message.get("event")
                if event == "ready":
                    self._ready_seen = True
                    ready.set()
                elif event == "error":
                    self._error = message.get("message", "unknown BLE error")
                    ready.set()
                elif event == "info" and "battery" in message:
                    self.describe_extra = {"battery": int(message["battery"])}
        except Exception:
            pass
        finally:
            ready.set()          # covers the worker dying without saying anything

    def stop(self) -> None:
        self._stop.set()
        proc, self._proc = self._proc, None
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=6)
            except subprocess.TimeoutExpired:
                proc.kill()
        for thread in self._readers:
            thread.join(timeout=2)
        self._readers = []
        self._running.clear()


def _worker_exit_message(code, last_line: str = "") -> str:
    """Human-usable reason for a worker that died, from its exit code.

    ``-6`` gets special treatment because it is the one non-obvious case: macOS
    SIGABRTs any process that touches CoreBluetooth without the hosting app having
    Bluetooth permission, and nothing in the abort names the fix.
    """
    # Only attach the last stderr line when it is real diagnostic noise (a native
    # log, a traceback tail) — an ordinary JSON status event adds nothing.
    tail = f" — {last_line}" if last_line and not last_line.startswith("{") else ""
    if code == -6:
        return ("Bluetooth worker aborted (SIGABRT): macOS has not granted Bluetooth "
                "access to the app running rtmon. System Settings → Privacy & "
                "Security → Bluetooth, enable the terminal (or IDE) you launch rtmon "
                "from, then restart it.")
    if code is None:
        return f"BLE worker stopped responding{tail}"
    if code < 0:
        return f"BLE worker killed by signal {-code}{tail}"
    return f"BLE worker exited with code {code}{tail}"


def _worker_failure(done: subprocess.CompletedProcess) -> tuple[str, str]:
    """``(detail, hint)`` for a worker that exited badly.

    A bare "exit -6" tells the operator nothing. On macOS a SIGABRT out of CoreBluetooth
    is almost always the permission prompt never having been answered for whichever
    binary is hosting Python, so name that specifically rather than making them search.
    """
    stderr = done.stderr.decode(errors="replace").strip()
    last = (stderr.splitlines() or [""])[-1][:160]
    code = done.returncode
    if code == -6:
        return ("Bluetooth stack aborted the scan (SIGABRT)",
                "macOS has not granted Bluetooth access to the process running rtmon. "
                "Check System Settings -> Privacy & Security -> Bluetooth for your "
                "terminal (or Python), enable it, and restart rtmon.")
    if code and code < 0:
        return (f"BLE worker killed by signal {-code}. {last}".strip(),
                "Toggling Bluetooth off and on usually clears a wedged stack.")
    return (f"BLE scan failed (exit {code}). {last}".strip(),
            "Wake the strap (it sleeps when idle) and keep it in range.")


def _read_exact(stream, n: int) -> bytes | None:
    buf = bytearray()
    while len(buf) < n:
        chunk = stream.read(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return bytes(buf)
