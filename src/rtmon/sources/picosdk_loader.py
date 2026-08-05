"""Make the picosdk wrapper able to find the PicoSDK that is actually installed.

``picosdk.library.Library._load`` resolves its driver with
``ctypes.util.find_library(name)`` and raises ``CannotFindPicoSDKError`` when that
returns ``None``. On macOS it always returns ``None`` for these drivers: the installer
puts them in a framework —

    /Library/Frameworks/PicoSDK.framework/Libraries/libps3000a/libps3000a.dylib

— and ``find_library`` searches ``DYLD_*`` paths, ``/usr/lib`` and the dyld shared
cache, none of which is a framework's ``Libraries`` directory. So a correctly installed
PicoSDK with a scope plugged in reports "PicoSDK (ps3000a) not found, check
LD_LIBRARY_PATH", naming an environment variable that macOS strips from child processes
anyway. Exporting ``DYLD_LIBRARY_PATH`` is not a fix either — System Integrity
Protection drops it, and it must be set before the process starts, which a Python-level
assignment cannot do.

The fix is to teach the *lookup* about the real install locations rather than to teach
the *environment* about them: :func:`install` wraps ``find_library`` so it falls back to
the paths the vendor installers actually use. The wrapper is never modified, the real
``find_library`` still runs first, and anything it finds still wins.

``RTMON_PICOSDK_DIR`` overrides the search for a non-standard install.
"""

from __future__ import annotations

import ctypes.util
import os
import sys
from pathlib import Path

# Where each platform's official installer puts the driver shared libraries. Templated
# on the driver name ("ps3000a", "ps4000", ...) as picosdk asks for it.
_SEARCH: dict[str, tuple[str, ...]] = {
    "darwin": (
        "/Library/Frameworks/PicoSDK.framework/Libraries/lib{name}/lib{name}.dylib",
        "/Library/Frameworks/PicoSDK.framework/Libraries/lib{name}.dylib",
        "/usr/local/lib/lib{name}.dylib",
        "/opt/homebrew/lib/lib{name}.dylib",
    ),
    "linux": (
        "/opt/picoscope/lib/lib{name}.so",
        "/usr/local/lib/lib{name}.so",
        "/usr/lib/lib{name}.so",
    ),
    "win32": (
        r"C:\Program Files\Pico Technology\SDK\lib\{name}.dll",
        r"C:\Program Files (x86)\Pico Technology\SDK\lib\{name}.dll",
    ),
}

_installed = False
_original = None


def candidates(name: str) -> list[str]:
    """Every path that would be tried for driver ``name``, in order."""
    patterns: list[str] = []
    override = os.environ.get("RTMON_PICOSDK_DIR")
    if override:
        suffix = {"darwin": "dylib", "win32": "dll"}.get(sys.platform, "so")
        prefix = "" if sys.platform == "win32" else "lib"
        patterns += [
            str(Path(override) / f"{prefix}{{name}}.{suffix}"),
            str(Path(override) / f"lib{{name}}" / f"{prefix}{{name}}.{suffix}"),
        ]
    patterns += list(_SEARCH.get(sys.platform, ()))
    return [p.format(name=name) for p in patterns]


def locate(name: str) -> str | None:
    """First existing path for driver ``name``, or ``None``."""
    return next((p for p in candidates(name) if os.path.isfile(p)), None)


def install() -> None:
    """Extend ``ctypes.util.find_library`` with the PicoSDK locations. Idempotent."""
    global _installed, _original
    if _installed:
        return
    _original = ctypes.util.find_library

    def find_library(name):
        # The stock lookup first: if the system can already resolve it (a Linux box with
        # the SDK on the loader path, say), that answer is the right one.
        found = _original(name)
        if found:
            return found
        return locate(name) if name.startswith("ps") else None

    ctypes.util.find_library = find_library
    _installed = True


def installed_drivers() -> list[str]:
    """Driver names whose library is present, e.g. ``['ps2000a', 'ps3000a', 'ps4000a']``.

    Read off the same directories :func:`locate` searches, by asking about every driver
    the wrapper knows rather than by globbing — so the answer is always in the vocabulary
    the rest of the app uses.
    """
    known = ("ps2000", "ps2000a", "ps3000", "ps3000a", "ps4000", "ps4000a",
             "ps5000", "ps5000a", "ps6000", "ps6000a")
    return [name for name in known if locate(name)]


def describe_failure(name: str) -> str:
    """A hint for a probe that could not load driver ``name``.

    Names what *is* installed, because the useful question is almost never "is the
    PicoSDK installed" (it usually is) but "does this install include the driver this
    box needs". Recent PicoSDK releases ship ps4000a and no ps4000, so a legacy
    4000-series unit reports a missing driver on a machine that looks fully set up.
    """
    have = installed_drivers()
    if have and name not in have:
        return (f"This PicoSDK install has no {name} driver. It does have: "
                f"{', '.join(have)}. If the unit is a {name}a-series scope use that "
                f"driver instead; if it really is a legacy {name}, install a PicoSDK "
                f"release that still ships it, or point RTMON_PICOSDK_DIR at one.")
    return ("No PicoSDK driver libraries found. Install the PicoSDK from picotech.com, "
            "or set RTMON_PICOSDK_DIR to the directory holding them. Looked in: "
            + ", ".join(candidates(name)[:3]))
