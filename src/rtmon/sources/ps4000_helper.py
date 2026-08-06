"""Standalone x86_64 capture process for the classic ps4000 driver.

Pico ships no arm64 macOS build of the *classic* ps4000 driver -- the arm64 PicoSDK
carries ps4000a and nothing for legacy 4000-series units -- so on Apple silicon it
cannot be loaded into the server's own process. Running the whole application under
Rosetta to work around that costs every other part of it: torch loses MPS, the arm64
ps3000a driver becomes unloadable, and startup slows down. So only this file runs
x86_64. It streams raw ADC counts down a pipe and the server, still native, does
everything else. See ``ps4000_bridge`` for the parent side.

Deliberately stdlib-only, with the driver bound by hand through ctypes rather than
through the picosdk wrapper. The wrapper imports numpy at module scope, and numpy is a
compiled extension -- requiring an x86_64 build of it turns "have any Intel-capable
Python" into "maintain a second scientific stack under Rosetta", which is the problem
this is meant to remove. The eight entry points below are the whole of what streaming
needs, and their signatures are copied from picosdk's own declarations.

Protocol -- one JSON line per handshake step, then binary frames:

    stdout  {"ok": true, "max_adc": 32767, "info": "..."}   unit open and configured
                                                            (--probe stops here)
    stdin   b"G"                                            release; EOF aborts
    stdout  {"ok": true, "interval_us": 200}                streaming, at this actual
                                                            interval -- the driver is
                                                            free to round the request
    stdout  <int32 n><n*int16 A><n*int16 B>  repeated       raw counts
    stdout  EOF                                             stopped, unit closed

Any step may instead emit ``{"ok": false, "error": "..."}`` and exit; the parent shows
that text verbatim, since the driver's own words beat anything paraphrased.

Must stay compatible with Python 3.9: the most convenient Intel interpreter on a Mac is
often whatever happens to be installed, not a current one.
"""

import argparse
import ctypes
import json
import os
import select
import signal
import struct
import sys
import time

# ---------------------------------------------------------------------------
# Driver binding. Signatures from picosdk/ps4000.py, which is generated from
# PicoStatus.h and ps4000Api.h.
# ---------------------------------------------------------------------------
PS4000_US = 3                      # PS4000_TIME_UNITS
CHANNEL_A, CHANNEL_B = 0, 1
DC_COUPLING = 1
MAX_ADC = 32767                    # fixed for this unit family

StreamingReady = ctypes.CFUNCTYPE(
    None,
    ctypes.c_int16,     # handle
    ctypes.c_int32,     # noOfSamples
    ctypes.c_uint32,    # startIndex
    ctypes.c_int16,     # overflow
    ctypes.c_uint32,    # triggerAt
    ctypes.c_int16,     # triggered
    ctypes.c_int16,     # autoStop
    ctypes.c_void_p,    # pParameter
)

_SIGNATURES = {
    "ps4000OpenUnit": (ctypes.c_uint32, [ctypes.c_void_p]),
    "ps4000CloseUnit": (ctypes.c_uint32, [ctypes.c_int16]),
    "ps4000GetUnitInfo": (ctypes.c_uint32, [ctypes.c_int16, ctypes.c_char_p,
                                            ctypes.c_int16, ctypes.c_void_p,
                                            ctypes.c_uint32]),
    "ps4000SetChannel": (ctypes.c_uint32, [ctypes.c_int16, ctypes.c_int32, ctypes.c_int16,
                                           ctypes.c_int16, ctypes.c_int32]),
    "ps4000SetDataBuffers": (ctypes.c_uint32, [ctypes.c_int16, ctypes.c_int32,
                                               ctypes.c_void_p, ctypes.c_void_p,
                                               ctypes.c_int32]),
    "ps4000RunStreaming": (ctypes.c_uint32, [ctypes.c_int16, ctypes.c_void_p, ctypes.c_int32,
                                             ctypes.c_uint32, ctypes.c_uint32, ctypes.c_int16,
                                             ctypes.c_uint32, ctypes.c_uint32]),
    "ps4000GetStreamingLatestValues": (ctypes.c_uint32, [ctypes.c_int16, ctypes.c_void_p,
                                                         ctypes.c_void_p]),
    "ps4000Stop": (ctypes.c_uint32, [ctypes.c_int16]),
}

# Named in the same terms the server uses, so one vocabulary covers both routes.
STATUS = {
    3: "PICO_NOT_FOUND (no unit connected)",
    4: "PICO_FW_FAIL",
    12: "PICO_OS_NOT_SUPPORTED",
    13: "PICO_PICOPP_TOO_OLD (driver older than this wrapper expects)",
    14: "PICO_INVALID_HANDLE",
    268435457: "the driver could not load its own dependencies "
               "(libpicoipp / libiomp5 -- see DYLD_LIBRARY_PATH)",
    269: "PICO_NOT_RESPONDING",
}


def load(path):
    lib = ctypes.CDLL(path)
    for name, (restype, argtypes) in _SIGNATURES.items():
        fn = getattr(lib, name)
        fn.restype, fn.argtypes = restype, argtypes
    return lib


def say(**message):
    sys.stdout.write(json.dumps(message) + "\n")
    sys.stdout.flush()


def die(error):
    say(ok=False, error=error)
    sys.exit(1)


def check(status, what):
    if status != 0:
        die("%s returned %s (%s)" % (what, status, STATUS.get(status, "see PicoStatus.h")))


def unit_info(lib, handle):
    """Variant and serial, for the device card. Never fatal -- it is a label."""
    out = []
    buf, used = ctypes.create_string_buffer(64), ctypes.c_int16()
    for code in (6, 3):                     # PICO_VARIANT_INFO, PICO_BATCH_AND_SERIAL
        if lib.ps4000GetUnitInfo(handle, buf, 64, ctypes.byref(used), code) == 0:
            text = buf.value.decode("utf-8", "replace").strip()
            if text:
                out.append(text)
    return " · ".join(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lib", required=True, help="absolute path to libps4000")
    parser.add_argument("--probe", action="store_true",
                        help="open the unit, report, close, exit")
    parser.add_argument("--interval-us", type=int, default=200)
    parser.add_argument("--buffer", type=int, default=5000)
    parser.add_argument("--range", type=int, default=8, dest="channel_range")
    args = parser.parse_args()

    stop = {"now": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__("now", True))
    signal.signal(signal.SIGINT, lambda *_: stop.__setitem__("now", True))

    try:
        lib = load(args.lib)
    except OSError as exc:
        die("cannot load %s: %s" % (args.lib, exc))

    handle = ctypes.c_int16()
    status = lib.ps4000OpenUnit(ctypes.byref(handle))
    if status != 0:
        die("ps4000OpenUnit returned %s (%s)"
            % (status, STATUS.get(status, "see PicoStatus.h")))

    try:
        if args.probe:
            say(ok=True, max_adc=MAX_ADC, info=unit_info(lib, handle) or "unit opened")
            return

        buffers = []
        for channel in (CHANNEL_A, CHANNEL_B):
            check(lib.ps4000SetChannel(handle, channel, 1, DC_COUPLING, args.channel_range),
                  "ps4000SetChannel")
            buf = (ctypes.c_int16 * args.buffer)()
            check(lib.ps4000SetDataBuffers(handle, channel, ctypes.byref(buf), None,
                                           args.buffer),
                  "ps4000SetDataBuffers")
            buffers.append(buf)

        say(ok=True, max_adc=MAX_ADC, info=unit_info(lib, handle) or "unit opened")

        # Rendezvous. The unit is open and configured but not yet sampling, so the
        # parent can hold every scope here and release them together -- which is what
        # keeps two units' time bases within a sample of each other at the start of a
        # recording. EOF here means the parent gave up; close cleanly.
        released = False
        while not released and not stop["now"]:
            if select.select([sys.stdin], [], [], 0.1)[0]:
                if sys.stdin.buffer.read(1) != b"G":
                    return
                released = True
        if stop["now"]:
            return

        interval = ctypes.c_uint32(args.interval_us)
        check(lib.ps4000RunStreaming(
            handle, ctypes.byref(interval), PS4000_US,
            0, 0,                    # pre/post trigger samples: unused
            0,                       # autoStop off -- the recording ends when it is told to
            1,                       # no downsampling
            args.buffer), "ps4000RunStreaming")
        say(ok=True, interval_us=int(interval.value))

        out = sys.stdout.buffer
        pending = {"start": 0, "n": 0}

        def on_ready(_handle, n_samples, start_index, _overflow,
                     _trigger_at, _triggered, _auto_stop, _param):
            if n_samples > 0:
                pending["start"], pending["n"] = start_index, n_samples

        callback = StreamingReady(on_ready)

        while not stop["now"]:
            pending["n"] = 0
            lib.ps4000GetStreamingLatestValues(handle, callback, None)
            n = pending["n"]
            if not n:
                # The call returns immediately whether or not it had data; without a
                # yield on the empty path this is a busy spin on a core.
                time.sleep(0.002)
                continue
            at = pending["start"]
            out.write(struct.pack("<i", n))
            for buf in buffers:
                out.write(bytes(memoryview(buf)[at:at + n]))
            out.flush()
    finally:
        try:
            lib.ps4000Stop(handle)
        finally:
            lib.ps4000CloseUnit(handle)


if __name__ == "__main__":
    # A broken pipe means the parent is gone; that is a normal shutdown, not a crash.
    try:
        main()
    except (BrokenPipeError, KeyboardInterrupt):
        os._exit(0)
