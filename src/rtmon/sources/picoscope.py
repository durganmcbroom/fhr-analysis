"""PicoScope streaming sources (ps3000a: 4 fibers, ps4000: 2 fibers).

Both units are always registered. Whether either can actually run is decided by
:meth:`PicoSource.probe`, which imports the driver wrapper and opens the unit -- the
one operation that distinguishes "SDK installed and this box is plugged in" from
"wrong driver" or "nothing there". A failure is reported, not raised, and the UI
shows the driver's own message.

Differences from the Qt implementation this replaces, all of them about not paying
for the whole capture up front:

* ``autoStop`` is off and no ``bufferComplete`` array is allocated. The old code
  preallocated ``numBuffersToCapture * sizeOfOneBuffer`` float64 *per channel* --
  ~290 MB for the ps3000a alone -- purely to hold a session it also capped at 30
  minutes. Samples now go straight from the driver's overview buffer into the
  bounded ring and the recorder's stream, so memory is O(history) and a recording
  runs until it is stopped.
* Conversion to volts is one vectorised multiply into a float32 block, instead of
  four in-place scalings of million-element float64 slices per callback.
* The streaming poll loop no longer stacks and emits the whole visible window on
  every callback; it hands over only the new samples.
"""

from __future__ import annotations

import ctypes
import threading
import time
from dataclasses import dataclass, field

import numpy as np

from rtmon.sources import picosdk_loader, ps4000_bridge
from rtmon.sources.base import KIND_FIBER, Channel, Probe, SampleClock, Sink, Source

# Index into the driver's range table; 8 == +/-5 V, which is what the rig has always used.
CHANNEL_RANGE = 8
VOLT_RANGES = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200]

SAMPLE_INTERVAL_US = 200          # 5 kHz
OVERVIEW_BUFFER = 5000            # samples per channel in the driver's overview buffer
POLL_IDLE_SLEEP = 0.002           # driver had nothing ready; yield briefly


@dataclass
class PicoSource(Source):
    """Shared streaming logic. Subclasses bind the driver module and its call shapes."""

    driver_module: str = ""
    _fiber_names: tuple[str, ...] = ()

    def __post_init__(self):
        super().__post_init__()
        self.channels = tuple(
            Channel(id=name, label=f"Fiber {name}", kind=KIND_FIBER, unit="V")
            for name in self._fiber_names
        )
        self.nominal_hz = 1e6 / SAMPLE_INTERVAL_US
        self._handle = ctypes.c_int16()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._buffers: list[np.ndarray] = []
        self._cfunc = None
        self._clock: SampleClock | None = None
        self._sink = None
        self._bridged = False
        self._start_error: Exception | None = None

    # ----------------------------------------------------------------- driver
    def _import(self):
        """Import the driver wrapper. Raises with a usable message.

        The loader shim goes in first: the wrapper resolves its shared library at
        *import* time, and on macOS the stock lookup cannot see a framework install.
        See ``picosdk_loader`` for why this is a lookup fix rather than an env var.
        """
        from importlib import import_module

        picosdk_loader.install()
        module = import_module(f"picosdk.{self.driver_module}")
        # The callable driver is an *instance* inside the module, named after it
        # (picosdk.ps3000a.ps3000a) -- the module itself has no ps3000aOpenUnit.
        return getattr(module, self.driver_module), import_module("picosdk.functions")

    def probe(self, deep: bool = True) -> Probe:
        # Three failures that read alike and have different fixes: the Python wrapper
        # missing (a poetry problem), the native driver library missing (a PicoSDK
        # install problem), and the unit not being plugged in. Telling them apart is
        # most of the value of probing at all.
        try:
            ps, _fn = self._import()
        except ImportError as exc:
            return Probe(False, f"{type(exc).__name__}: {exc}",
                         "The picosdk Python wrapper is not installed: "
                         "poetry install --with record")
        except Exception as exc:
            return Probe(False, f"{type(exc).__name__}: {exc}",
                         picosdk_loader.describe_failure(self.driver_module))
        if not deep:
            found = picosdk_loader.locate(self.driver_module)
            return Probe(True, f"driver library at {found}" if found
                                else "wrapper importable (unit not opened)")

        handle = ctypes.c_int16()
        try:
            status = self._open_unit(ps, handle)
        except Exception as exc:
            return Probe(False, f"{type(exc).__name__}: {exc}",
                         picosdk_loader.describe_failure(self.driver_module))
        if status != 0:
            return Probe(False, f"{self.driver_module}OpenUnit returned {status} "
                                f"({_PICO_STATUS.get(status, 'see PicoStatus.h')})",
                         "Unit not connected, in use by PicoScope 6, or served by a "
                         "different driver than this wrapper expects.")
        try:
            info = self._unit_info(ps, handle)
        finally:
            self._close_unit(ps, handle)
        return Probe(True, info or "unit opened")

    # Implemented per driver -----------------------------------------------
    def _open_unit(self, ps, handle) -> int:
        raise NotImplementedError

    def _close_unit(self, ps, handle) -> None:
        raise NotImplementedError

    def _unit_info(self, ps, handle) -> str:
        return ""

    def _configure(self, ps, fn) -> None:
        raise NotImplementedError

    def _run_streaming(self, ps, fn, interval) -> None:
        raise NotImplementedError

    def _get_latest(self, ps) -> None:
        raise NotImplementedError

    def _max_adc(self, ps) -> float:
        raise NotImplementedError

    # -------------------------------------------------------------- lifecycle
    def start(self, sink: Sink) -> None:
        if self.running:
            return
        self._sink = sink
        self._stop.clear()
        self._error = None
        ready = threading.Event()
        self._start_error: Exception | None = None
        self._thread = threading.Thread(target=self._run, args=(ready,),
                                        name=f"rtmon-{self.id}", daemon=True)
        self._thread.start()
        # Open + configure happen on the streaming thread (the driver is happiest when
        # one thread owns the handle), so block here until it either streams or fails --
        # the caller needs a real answer, not a thread that dies quietly.
        ready.wait(timeout=20.0)
        if self._start_error is not None:
            raise self._start_error
        if not self.running:
            raise RuntimeError(f"{self.label} did not start within 20 s")

    def stop(self) -> None:
        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is not None and thread.is_alive():
            thread.join(timeout=5.0)
        self._running.clear()

    # ------------------------------------------------------------------- run
    def _run(self, ready: threading.Event) -> None:
        ps = fn = None
        opened = False
        try:
            ps, fn = self._import()
            status = self._open_unit(ps, self._handle)
            if status != 0:
                raise RuntimeError(
                    f"{self.driver_module}OpenUnit returned {status} "
                    f"({_PICO_STATUS.get(status, 'see PicoStatus.h')})")
            opened = True
            self._configure(ps, fn)

            scale = np.float32(VOLT_RANGES[CHANNEL_RANGE] / self._max_adc(ps))
            self._clock = SampleClock(self.nominal_hz)
            n_ch = len(self.channels)

            def callback(handle, n_samples, start_index, overflow,
                         trigger_at, triggered, auto_stop, param):
                if n_samples <= 0:
                    return
                sl = slice(start_index, start_index + n_samples)
                block = np.empty((n_samples, n_ch), dtype=np.float32)
                for i, buf in enumerate(self._buffers):
                    block[:, i] = buf[sl]
                block *= scale
                self._sink(self._clock.stamp(n_samples), block)

            self._cfunc = ps.StreamingReadyType(callback)

            interval = ctypes.c_int32(SAMPLE_INTERVAL_US)
            self._run_streaming(ps, fn, interval)
            # The driver may not honour the requested interval exactly; it writes back
            # what it actually used, and the clock must be told or the trace drifts.
            actual_hz = 1e6 / interval.value
            self.nominal_hz = actual_hz
            self._clock.hz = actual_hz

            self._running.set()
            ready.set()

            while not self._stop.is_set():
                self._get_latest(ps)
                # GetStreamingLatestValues returns immediately whether or not it had
                # data; without a yield on the empty path this is a busy spin on a core.
                time.sleep(POLL_IDLE_SLEEP)
        except Exception as exc:  # noqa: BLE001 - surfaced through .error and the UI
            self._error = f"{type(exc).__name__}: {exc}"
            self._start_error = exc
            ready.set()
        finally:
            self._running.clear()
            if opened and ps is not None:
                try:
                    self._stop_unit(ps)
                except Exception:
                    pass
                try:
                    self._close_unit(ps, self._handle)
                except Exception:
                    pass
            self._buffers = []
            self._cfunc = None
            ready.set()

    def _stop_unit(self, ps) -> None:
        raise NotImplementedError


@dataclass
class PS3000ASource(PicoSource):
    id: str = "ps3000a"
    label: str = "PicoScope 3000A (abdomen)"
    driver_module: str = "ps3000a"
    history_seconds: float = 90.0
    _fiber_names: tuple[str, ...] = ("2A", "2B", "2C", "2D")

    def _open_unit(self, ps, handle) -> int:
        status = ps.ps3000aOpenUnit(ctypes.byref(handle), None)
        # 282/286 are "powered from USB only" advisories, not failures: the unit is
        # there and works, it just told us how it is powered. Acknowledge and continue.
        if status in (282, 286):
            status = ps.ps3000aChangePowerSource(handle, status)
        return status

    def _close_unit(self, ps, handle) -> None:
        ps.ps3000aCloseUnit(handle)

    def _unit_info(self, ps, handle) -> str:
        return "4 channels @ 5 kHz"

    def _configure(self, ps, fn) -> None:
        self._buffers = []
        for i, name in enumerate("ABCD"):
            ch = ps.PS3000A_CHANNEL[f"PS3000A_CHANNEL_{name}"]
            fn.assert_pico_ok(ps.ps3000aSetChannel(self._handle, ch, 1, 1, CHANNEL_RANGE, 0.0))
            buf = np.zeros(OVERVIEW_BUFFER, dtype=np.int16)
            fn.assert_pico_ok(ps.ps3000aSetDataBuffers(
                self._handle, ch,
                buf.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)), None,
                OVERVIEW_BUFFER, 0, ps.PS3000A_RATIO_MODE['PS3000A_RATIO_MODE_NONE']))
            self._buffers.append(buf)

    def _run_streaming(self, ps, fn, interval) -> None:
        fn.assert_pico_ok(ps.ps3000aRunStreaming(
            self._handle, ctypes.byref(interval), 3,   # 3 == PS3000A_US
            0, 0,                                      # pre/post trigger: unused
            0,                                         # autoStop off -> stream forever
            1, ps.PS3000A_RATIO_MODE['PS3000A_RATIO_MODE_NONE'], OVERVIEW_BUFFER))

    def _get_latest(self, ps) -> None:
        ps.ps3000aGetStreamingLatestValues(self._handle, self._cfunc, None)

    def _stop_unit(self, ps) -> None:
        ps.ps3000aStop(self._handle)

    def _max_adc(self, ps) -> float:
        max_adc = ctypes.c_int16()
        ps.ps3000aMaximumValue(self._handle, ctypes.byref(max_adc))
        return float(max_adc.value)


@dataclass
class PS4000Source(PicoSource):
    """The legacy 4000-series unit, in-process where possible and bridged where not.

    Pico's arm64 macOS SDK has no classic ps4000 driver, so on Apple silicon the
    in-process route is not merely unconfigured -- it cannot exist. Rather than make
    that the operator's problem, the source falls back to running the Intel driver in a
    subprocess (:mod:`rtmon.sources.ps4000_bridge`) and carries on presenting the same
    two channels. The device card says which route is in use; nothing else in the
    server can tell the difference.
    """

    id: str = "ps4000"
    label: str = "PicoScope 4000 (chest)"
    driver_module: str = "ps4000"
    history_seconds: float = 90.0
    _fiber_names: tuple[str, ...] = ("1A", "1B")

    # ------------------------------------------------------------- routing
    def _native_error(self) -> str | None:
        """Why the driver cannot be used in-process, or None if it can."""
        try:
            self._import()
        except Exception as exc:  # noqa: BLE001 - the text is the whole point
            return f"{type(exc).__name__}: {exc}"
        return None

    def probe(self, deep: bool = True) -> Probe:
        native = self._native_error()
        if native is None:
            return super().probe(deep)

        # No in-process driver. Say so once, then report on the bridge instead -- the
        # native failure is the *reason* for the bridge, not a competing diagnosis.
        blocked = ps4000_bridge.describe()
        if blocked:
            return Probe(False, native, blocked)
        if not deep:
            return Probe(True, f"Intel driver at {ps4000_bridge.find_library()} "
                               f"(the arm64 SDK has no ps4000; running it in a "
                               f"subprocess)")
        helper = ps4000_bridge.Helper(SAMPLE_INTERVAL_US, OVERVIEW_BUFFER, CHANNEL_RANGE)
        try:
            answer = helper.open(probe=True)
        except Exception as exc:  # noqa: BLE001
            # The Intel side is working -- it got far enough to run the driver and be
            # told no. Say that, so a missing cable is not mistaken for a broken setup.
            found = "PICO_NOT_FOUND" in str(exc)
            return Probe(False, f"{type(exc).__name__}: {exc}",
                         "Plug the unit in and rescan." if found else
                         "The Intel driver loaded but the unit did not open. Check it "
                         "is not held open by PicoScope.")
        finally:
            helper.close()
        return Probe(True, f"{answer.get('info') or 'unit opened'} · via Intel subprocess")

    def start(self, sink: Sink) -> None:
        self._bridged = self._native_error() is not None
        if not self._bridged:
            super().start(sink)
            return

        self._sink = sink
        self._stop.clear()
        self._error = None
        ready = threading.Event()
        self._start_error = None
        self._thread = threading.Thread(target=self._run_bridged, args=(ready,),
                                        name=f"rtmon-{self.id}", daemon=True)
        self._thread.start()
        ready.wait(timeout=40.0)          # a cold Rosetta start is not instant
        if self._start_error is not None:
            raise self._start_error
        if not self.running:
            raise RuntimeError(f"{self.label} did not start within 40 s")

    def _run_bridged(self, ready: threading.Event) -> None:
        """Same contract as PicoSource._run: stamp blocks into the sink until stopped.

        The conversion is the same one the in-process path does -- counts to volts,
        one vectorised multiply into a float32 block -- so downstream nothing knows
        which side of the process boundary the samples came from.
        """
        helper = ps4000_bridge.Helper(SAMPLE_INTERVAL_US, OVERVIEW_BUFFER, CHANNEL_RANGE)
        try:
            helper.open()
            actual_us = helper.release()
            self.nominal_hz = 1e6 / actual_us
            clock = SampleClock(self.nominal_hz)
            scale = np.float32(VOLT_RANGES[CHANNEL_RANGE] / helper.max_adc)

            self._running.set()
            ready.set()

            stopping = self._stop.is_set
            while not self._stop.is_set():
                payload = helper.block(stopping)
                if payload is None:
                    break                       # helper finished or is being stopped
                if not payload:
                    continue
                counts = np.frombuffer(payload, dtype="<i2")
                n = counts.size // 2
                block = np.empty((n, 2), dtype=np.float32)
                block[:, 0] = counts[:n]
                block[:, 1] = counts[n:]
                block *= scale
                self._sink(clock.stamp(n), block)
        except Exception as exc:  # noqa: BLE001 - surfaced through .error and the UI
            self._error = f"{type(exc).__name__}: {exc}"
            self._start_error = exc
        finally:
            helper.close()
            self._running.clear()
            ready.set()

    # --------------------------------------------------- in-process driver
    def _open_unit(self, ps, handle) -> int:
        return ps.ps4000OpenUnit(ctypes.byref(handle))

    def _close_unit(self, ps, handle) -> None:
        ps.ps4000CloseUnit(handle)

    def _unit_info(self, ps, handle) -> str:
        return "2 channels @ 5 kHz"

    def _configure(self, ps, fn) -> None:
        self._buffers = []
        for name in "AB":
            ch = ps.PS4000_CHANNEL[f"PS4000_CHANNEL_{name}"]
            fn.assert_pico_ok(ps.ps4000SetChannel(self._handle, ch, 1, 1, CHANNEL_RANGE))
            buf = np.zeros(OVERVIEW_BUFFER, dtype=np.int16)
            fn.assert_pico_ok(ps.ps4000SetDataBuffers(
                self._handle, ch,
                buf.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)), None, OVERVIEW_BUFFER))
            self._buffers.append(buf)

    def _run_streaming(self, ps, fn, interval) -> None:
        fn.assert_pico_ok(ps.ps4000RunStreaming(
            self._handle, ctypes.byref(interval), 3,   # 3 == PS4000_US
            0, 0, 0, 1, OVERVIEW_BUFFER))

    def _get_latest(self, ps) -> None:
        ps.ps4000GetStreamingLatestValues(self._handle, self._cfunc, None)

    def _stop_unit(self, ps) -> None:
        ps.ps4000Stop(self._handle)

    def _max_adc(self, ps) -> float:
        return 32767.0


# The handful of PICO_STATUS codes worth naming in a UI. Anything else is shown as a
# number with a pointer to the header, which beats a bare integer.
_PICO_STATUS = {
    3: "PICO_NOT_FOUND (no unit connected)",
    4: "PICO_FW_FAIL",
    12: "PICO_OS_NOT_SUPPORTED",
    13: "PICO_PICOPP_TOO_OLD (driver older than this wrapper expects)",
    14: "PICO_INVALID_HANDLE",
    269: "PICO_NOT_RESPONDING",
    282: "PICO_POWER_SUPPLY_NOT_CONNECTED",
    286: "PICO_USB3_0_DEVICE_NON_USB3_0_PORT",
}
