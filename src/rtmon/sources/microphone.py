"""Microphone source -- the acoustic reference (SOT) channel.

Prefers ``sounddevice`` (hands back a numpy block directly, and can enumerate devices
without opening one, which makes a useful probe) and falls back to ``pyaudio``, which
is what the Qt app used. Either way samples reach the sink as float32 in [-1, 1]; the
recorder scales back to int16 for the WAV, and because the scale factor is a power of
two that round-trip is exact.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np

from rtmon.sources.base import KIND_AUDIO, Channel, Probe, SampleClock, Sink, Source

DEFAULT_RATE = 44100
BLOCK_FRAMES = 2048
INT16_SCALE = np.float32(1.0 / 32768.0)


@dataclass
class MicrophoneSource(Source):
    id: str = "mic"
    label: str = "Microphone (NST / acoustic SOT)"
    nominal_hz: float = DEFAULT_RATE
    # 45 s rather than the fibers' 90: at 44.1 kHz each second costs nine times as
    # much, and no processor asks the mic for more than a ~30 s chunk.
    history_seconds: float = 45.0
    # None -> the system default input. A device NAME (string) rather than an index:
    # names survive replugging and reboots, indices do not. sounddevice accepts the
    # name directly; the pyaudio path resolves it to an index at start.
    device: int | str | None = None
    # Timing correction, in seconds, subtracted from every sample's timestamp — the
    # audio stack's capture latency (device buffer + driver + the block the callback
    # hands over), which SampleClock cannot see because it only ever observes arrival
    # times. Measured by tap alignment against a fiber; see rtmon.align. Applied at
    # push, so changing it takes effect on a stream that is already running.
    latency_s: float = 0.0

    def __post_init__(self):
        super().__post_init__()
        self.channels = (Channel(id="MIC", label="Microphone", kind=KIND_AUDIO, unit=""),)
        self._backend = None
        self._stream = None
        self._pa = None
        self._clock: SampleClock | None = None
        self._last_out: float | None = None
        self._lock = threading.Lock()

    # ---------------------------------------------------------------- probe
    def probe(self, deep: bool = True) -> Probe:
        try:
            import sounddevice as sd
        except Exception:
            sd = None
        if sd is not None:
            try:
                devices = sd.query_devices()
                inputs = [(i, d) for i, d in enumerate(devices)
                          if d.get("max_input_channels", 0) > 0]
                if not inputs:
                    return Probe(False, "sounddevice found no input devices",
                                 "Check system input permissions and that a mic is connected.")
                default = sd.query_devices(kind="input")
                # The enumeration rides along in describe_extra so the UI can offer a
                # device picker -- "the default input" is wrong exactly when it matters
                # (AirPods steal the default while the NST mic sits on another input).
                self.describe_extra = {
                    **self.describe_extra,
                    "devices": [{"index": i, "name": d["name"]} for i, d in inputs],
                    "default_device": default["name"],
                }
                return Probe(True, f"sounddevice: {default['name']} "
                                   f"({len(inputs)} input device(s))")
            except Exception as exc:
                return Probe(False, f"sounddevice: {type(exc).__name__}: {exc}")

        try:
            import pyaudio
        except Exception as exc:
            return Probe(False, f"no audio backend: {exc}",
                         "pip install sounddevice (preferred) or pyaudio.")
        if not deep:
            return Probe(True, "pyaudio importable (not opened)")
        pa = None
        try:
            pa = pyaudio.PyAudio()
            inputs = [(i, pa.get_device_info_by_index(i)) for i in range(pa.get_device_count())]
            inputs = [(i, d) for i, d in inputs if d["maxInputChannels"] > 0]
            if not inputs:
                return Probe(False, "pyaudio found no input devices")
            name = pa.get_default_input_device_info()["name"]
            self.describe_extra = {
                **self.describe_extra,
                "devices": [{"index": i, "name": d["name"]} for i, d in inputs],
                "default_device": name,
            }
            return Probe(True, f"pyaudio: {name} ({len(inputs)} input device(s))")
        except Exception as exc:
            return Probe(False, f"pyaudio: {type(exc).__name__}: {exc}")
        finally:
            if pa is not None:
                pa.terminate()

    # ------------------------------------------------------------ lifecycle
    def start(self, sink: Sink) -> None:
        if self.running:
            return
        self._sink = sink
        self._clock = SampleClock(self.nominal_hz)
        self._last_out = None
        self._error = None
        try:
            self._start_sounddevice()
            self._backend = "sounddevice"
        except ImportError:
            self._start_pyaudio()
            self._backend = "pyaudio"
        self.describe_extra = {"backend": self._backend}
        self._running.set()

    def _emit(self, mono: np.ndarray) -> None:
        n = mono.shape[0]
        if not n:
            return
        t = self._clock.stamp(n) - self.latency_s
        # Keep emitted time strictly increasing if the correction changes mid-stream:
        # raising it shifts samples earlier and can overlap what was already sent. The
        # overlap is dropped once, at most |change| seconds of it.
        last = self._last_out
        if last is not None and t[0] <= last:
            keep = t > last
            if not keep.any():
                return
            t, mono = t[keep], mono[keep]
        self._last_out = float(t[-1])
        self._sink(t, mono.reshape(t.size, 1))

    def _start_sounddevice(self) -> None:
        import sounddevice as sd

        def callback(indata, frames, time_info, status):
            # Mono-mix rather than taking channel 0: on a stereo interface the fetal
            # sound may be on either input, and averaging costs nothing here.
            block = indata if indata.ndim == 1 else indata.mean(axis=1)
            self._emit(np.asarray(block, dtype=np.float32))

        self._stream = sd.InputStream(
            samplerate=self.nominal_hz, blocksize=BLOCK_FRAMES, dtype="float32",
            channels=1, device=self.device, callback=callback,
        )
        self._stream.start()
        # The device may refuse the requested rate and pick its own.
        actual = float(self._stream.samplerate)
        self.nominal_hz = actual
        self._clock.hz = actual

    def _start_pyaudio(self) -> None:
        import pyaudio

        self._pa = pyaudio.PyAudio()
        channels = 1
        device = self.device
        if isinstance(device, str):
            device = next((i for i in range(self._pa.get_device_count())
                           if self._pa.get_device_info_by_index(i)["name"] == device), None)

        def callback(in_data, frame_count, time_info, status):
            raw = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) * INT16_SCALE
            self._emit(raw)
            return (in_data, pyaudio.paContinue)

        self._stream = self._pa.open(
            format=pyaudio.paInt16, channels=channels, rate=int(self.nominal_hz),
            input=True, frames_per_buffer=BLOCK_FRAMES,
            input_device_index=device, stream_callback=callback,
        )

    def stop(self) -> None:
        with self._lock:
            stream, self._stream = self._stream, None
            pa, self._pa = self._pa, None
        if stream is not None:
            try:
                stream.stop()
            except Exception:
                pass
            try:
                stream.close()
            except Exception:
                pass
        if pa is not None:
            try:
                pa.terminate()
            except Exception:
                pass
        self._running.clear()
