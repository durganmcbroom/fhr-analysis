"""Out-of-process BLE worker for the Polar Verity Sense.

``python -m rtmon.sources.polar_worker [--scan]``

This exists because of a hard platform constraint, not a preference. On macOS,
CoreBluetooth delivers its callbacks on the *process main thread's* run loop; driving
bleak from any other thread aborts the process — ``SIGABRT``, native, uncatchable, no
Python traceback. The server's main thread is already the HTTP server, so the BLE
stack cannot have it.

Running the strap in its own process gives CoreBluetooth the main thread it requires
and, just as importantly, contains the blast radius: a Bluetooth stack that aborts,
hangs, or is denied permission can no longer take down a recording in progress. That
matters more here than the cost of a pipe — a session is a patient visit, and the
strap is the least essential thing on the rig.

Protocol, deliberately split so binary and text never interleave:

    stdout  binary sample frames: [u32 n][f64 t * n][f32 x * 4n]
    stderr  one JSON object per line: {"event": "ready"|"error"|"info", ...}

With ``--scan`` it instead writes a single JSON line to stdout and exits, which is
what :meth:`PolarSource.probe` reads.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import struct
import sys

import numpy as np

from rtmon.sources.polar import (
    BATTERY_UUID, DEFAULT_NAME, NOMINAL_HZ, PMD_CONTROL, PMD_DATA,
    PPG_START, PPG_STOP, SCAN_TIMEOUT, decode_frame,
)


def _say(event: str, **fields) -> None:
    sys.stderr.write(json.dumps({"event": event, **fields}) + "\n")
    sys.stderr.flush()


def _emit(t: np.ndarray, x: np.ndarray) -> None:
    payload = (struct.pack("<I", t.size)
               + np.ascontiguousarray(t, dtype=np.float64).tobytes()
               + np.ascontiguousarray(x, dtype=np.float32).tobytes())
    sys.stdout.buffer.write(payload)
    sys.stdout.buffer.flush()


async def scan(name: str) -> dict:
    from bleak import BleakScanner
    device = await BleakScanner.find_device_by_name(name, timeout=SCAN_TIMEOUT)
    if device is None:
        return {"ok": False, "detail": f"'{name}' not advertising"}
    return {"ok": True, "detail": f"found {name}"}


async def stream(name: str) -> None:
    from bleak import BleakClient, BleakScanner

    device = await BleakScanner.find_device_by_name(name, timeout=SCAN_TIMEOUT)
    if device is None:
        _say("error", message=f"'{name}' not found")
        return

    state = {"last_ts": None, "min_delay": None, "last_emitted": 0.0}

    def on_data(_sender, data: bytearray) -> None:
        try:
            # No timing correction here: the worker emits stamps carrying only the
            # measured clock offset, and the parent applies the correction. That is what
            # lets tap alignment change it on a strap that is already streaming --
            # applying it here meant a new value only took effect at the next arm, so
            # each re-measurement measured the residual under the OLD value while the
            # apply step added it to the NEW one, and repeated applies ran away.
            got = decode_frame(bytes(data), state, NOMINAL_HZ, 0.0)
        except Exception as exc:  # noqa: BLE001 - one bad frame must not end the stream
            _say("error", message=f"decode: {type(exc).__name__}: {exc}")
            return
        if got is not None:
            _emit(*got)

    async with BleakClient(device) as client:
        try:
            battery = (await client.read_gatt_char(BATTERY_UUID))[0]
            _say("info", battery=int(battery))
        except Exception:
            pass
        await client.write_gatt_char(PMD_CONTROL, PPG_START)
        await client.start_notify(PMD_DATA, on_data)
        _say("ready")
        try:
            # Held open until the parent closes the pipe or kills us; the notification
            # callback does all the work from here.
            while client.is_connected:
                await asyncio.sleep(0.5)
        finally:
            try:
                await client.stop_notify(PMD_DATA)
                await client.write_gatt_char(PMD_CONTROL, PPG_STOP)
            except Exception:
                pass
    _say("info", message="disconnected")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", default=DEFAULT_NAME)
    parser.add_argument("--scan", action="store_true", help="probe and exit")
    args = parser.parse_args()

    # asyncio.run on the process main thread is the whole point: this is the one run
    # loop CoreBluetooth will accept.
    if args.scan:
        try:
            result = asyncio.run(scan(args.name))
        except Exception as exc:  # noqa: BLE001
            result = {"ok": False, "detail": f"{type(exc).__name__}: {exc}"}
        sys.stdout.write(json.dumps(result))
        sys.stdout.flush()
        return

    try:
        asyncio.run(stream(args.name))
    except KeyboardInterrupt:
        pass
    except Exception as exc:  # noqa: BLE001
        _say("error", message=f"{type(exc).__name__}: {exc}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
