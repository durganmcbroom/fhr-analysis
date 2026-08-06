"""Run the classic ps4000 driver in an x86_64 subprocess, from a native server.

The problem this solves, precisely: Pico's arm64 macOS SDK carries ``ps4000a`` and no
``ps4000``. A legacy 4000-series unit therefore needs an Intel process, and the obvious
answer -- run everything under Rosetta -- is a bad trade. It costs torch its MPS
backend, makes the arm64 ps3000a driver unloadable in the same process, and slows the
whole server down, all to accommodate two channels. So the boundary is drawn around the
driver alone: :mod:`rtmon.sources.ps4000_helper` runs x86_64 and does nothing but
stream raw counts; everything else stays native.

Two things have to be found for that to work, and both are found rather than
configured, because a rig that needs a setup guide is a rig that stops working when
somebody reinstalls something:

**An Intel-capable Python.** Any will do -- the helper imports nothing outside the
standard library. Not ``/usr/bin/python3``, though, despite being universal: System
Integrity Protection strips ``DYLD_*`` from Apple-signed binaries, and without it the
driver cannot find the libraries it dlopens by bare name at open time. That failure is
detected and reported rather than guessed at.

**An x86_64 build of libps4000.** The PicoSDK framework is the first place to look, but
on a machine with the arm64 SDK it holds arm64 libraries, which are useless here -- so
each candidate's Mach-O header is read and non-Intel ones are skipped. PicoScope 6 and
PicoScope 7 both ship an Intel ``libps4000.dylib`` inside the application bundle, next
to the ``libpicoipp`` and ``libiomp5`` it needs, which is what usually ends up being
used.

``RTMON_PS4000_PYTHON`` and ``RTMON_PS4000_LIB_DIR`` override either search.
"""

from __future__ import annotations

import glob
import json
import os
import select
import shutil
import struct
import subprocess
import sys
import time
from pathlib import Path

HELPER = Path(__file__).with_name("ps4000_helper.py")

# Interpreters worth trying, best first. A dedicated Intel binary beats a universal one
# run through `arch`, and anything beats the system Python -- see the module docstring.
_PYTHON_GLOBS = (
    "/usr/local/bin/python3*-intel64",
    "/usr/local/bin/python3*",
    "/Library/Frameworks/Python.framework/Versions/*/bin/python3",
    "/opt/homebrew/bin/python3*",
)

# Directories that hold an x86_64 libps4000 on a Mac. The framework paths are where the
# PicoSDK installer puts drivers; the application bundles are where an Intel build
# survives on a machine whose SDK has moved on to arm64.
_LIB_DIRS = (
    "/Library/Frameworks/PicoSDK.framework/Libraries/libps4000",
    "/Library/Frameworks/PicoSDK.framework/Libraries",
    "/Applications/PicoScope 7 T&M.app/Contents/Resources",
    "/Applications/PicoScope 6.app/Contents/Resources",
    "~/lib",
    "/usr/local/lib",
)

# Sibling directories added to the helper's DYLD_LIBRARY_PATH: the driver dlopens
# libpicoipp and libiomp5 by bare name once the unit is opened, and in the framework
# layout each library lives in its own directory.
_DEP_DIRS = (
    "/Library/Frameworks/PicoSDK.framework/Libraries/libpicoipp",
    "/Applications/PicoScope 7 T&M.app/Contents/Resources",
    "/Applications/PicoScope 6.app/Contents/Resources",
)

MACHO_X86_64 = 0x01000007
MACHO_ARM64 = 0x0100000C


class BridgeUnavailable(RuntimeError):
    """No way to run the helper. The message says which half is missing."""


# ------------------------------------------------------------------ discovery
def architectures(path: str) -> set[int]:
    """CPU types present in a Mach-O file, fat or thin. Header only.

    Reading the header rather than shelling out to ``lipo``: this runs during a device
    probe, on a machine that may not have the command line tools installed at all.
    """
    try:
        with open(path, "rb") as fh:
            head = fh.read(8)
            if len(head) < 8:
                return set()
            magic = struct.unpack(">I", head[:4])[0]
            if magic in (0xCAFEBABE, 0xCAFEBABF):          # fat; counts are big-endian
                count = struct.unpack(">I", head[4:8])[0]
                entry = 20 if magic == 0xCAFEBABE else 32
                blob = fh.read(count * entry)
                return {struct.unpack(">i", blob[i * entry:i * entry + 4])[0]
                        for i in range(count) if len(blob) >= (i + 1) * entry}
            if magic in (0xFEEDFACF, 0xFEEDFACE):          # thin, big-endian magic
                return {struct.unpack(">i", head[4:8])[0]}
            if magic in (0xCFFAEDFE, 0xCEFAEDFE):          # thin, little-endian magic
                return {struct.unpack("<i", head[4:8])[0]}
    except OSError:
        return set()
    return set()


def _is_x86_64(path: str) -> bool:
    return MACHO_X86_64 in architectures(path)


def find_library() -> str | None:
    """An x86_64 ``libps4000.dylib``, or None."""
    override = os.environ.get("RTMON_PS4000_LIB_DIR")
    roots = ([override] if override else []) + list(_LIB_DIRS)
    for root in roots:
        candidate = os.path.join(os.path.expanduser(root), "libps4000.dylib")
        if os.path.isfile(candidate) and _is_x86_64(candidate):
            return candidate
    return None


def find_python() -> list[str] | None:
    """Argv prefix that runs a script as x86_64, or None.

    A universal interpreter is invoked through ``arch -x86_64``; a thin Intel one
    directly. ``/usr/bin/python3`` is excluded deliberately -- it is universal and would
    pass every test here, and then fail at ``ps4000OpenUnit`` with a dependency error,
    because SIP strips the loader path it needs.
    """
    override = os.environ.get("RTMON_PS4000_PYTHON")
    if override:
        return _invocation(override)
    if sys.platform != "darwin":
        return None
    # Intel-only binaries first: they need no `arch` wrapper, which is one fewer
    # Apple-signed process between here and the driver (see _invocation).
    for wants_arch in (False, True):
        for pattern in _PYTHON_GLOBS:
            for path in sorted(glob.glob(pattern), reverse=True):
                if os.path.isdir(path) or not os.access(path, os.X_OK):
                    continue
                if path.startswith("/usr/bin/") or not _is_x86_64(path):
                    continue
                if _needs_arch(path) != wants_arch:
                    continue
                argv = _invocation(path)
                if argv is not None:
                    return argv
    return None


def _needs_arch(path: str) -> bool:
    """Does this binary also contain arm64, so the slice has to be chosen explicitly?

    A file whose only slice is x86_64 runs as x86_64 when executed directly, fat header
    or not. Only a *universal* binary is ambiguous.
    """
    return MACHO_ARM64 in architectures(path)


def _invocation(path: str) -> list[str] | None:
    if not _needs_arch(path):
        return [path]
    return ["arch", "-x86_64", path] if shutil.which("arch") else None


def describe() -> str:
    """Why the bridge cannot run, in terms of what to install. Empty when it can."""
    if find_library() is None:
        return ("No x86_64 libps4000 found. The arm64 PicoSDK ships ps4000a only, so a "
                "legacy 4000-series unit needs the Intel driver: install PicoScope 6 or "
                "7 (its app bundle contains one), or set RTMON_PS4000_LIB_DIR to a "
                "directory holding libps4000.dylib.")
    if find_python() is None:
        return ("No Intel-capable Python found to run the ps4000 helper in. Any will do "
                "— it needs only the standard library — but not /usr/bin/python3, whose "
                "loader path macOS strips. Install python.org's universal build, or set "
                "RTMON_PS4000_PYTHON.")
    return ""


def loader_path(library: str) -> str:
    """DYLD_LIBRARY_PATH the driver needs to resolve what it dlopens by bare name."""
    dirs = [os.path.dirname(library)]
    dirs += [os.path.expanduser(d) for d in _DEP_DIRS if os.path.isdir(os.path.expanduser(d))]
    existing = os.environ.get("DYLD_LIBRARY_PATH")
    if existing:
        dirs.append(existing)
    # Deduplicated, order preserved: the directory the driver came from wins.
    seen, ordered = set(), []
    for d in dirs:
        if d not in seen:
            seen.add(d)
            ordered.append(d)
    return os.pathsep.join(ordered)


def _command(python: list[str], library: str, tail: list[str]) -> tuple[list[str], dict]:
    """The argv and environment that get DYLD_LIBRARY_PATH all the way to the driver.

    Setting it in the child's environment is not enough when ``arch`` is in the way:
    ``arch`` is Apple-signed, so System Integrity Protection strips every ``DYLD_*``
    variable as it execs, and the driver then fails to load libpicoipp with a status
    code that looks nothing like "your loader path was deleted". ``arch -e`` sets the
    variable on the far side of that boundary, which is the only place it survives.
    """
    path = loader_path(library)
    env = dict(os.environ)
    env["DYLD_LIBRARY_PATH"] = path
    if python and os.path.basename(python[0]) == "arch":
        return [*python[:-1], "-e", f"DYLD_LIBRARY_PATH={path}", python[-1], *tail], env
    return [*python, *tail], env


# -------------------------------------------------------------------- process
class Helper:
    """The helper subprocess, and the framing on its stdout."""

    def __init__(self, interval_us: int, buffer_size: int, channel_range: int):
        self.interval_us = interval_us
        self.buffer_size = buffer_size
        self.channel_range = channel_range
        self.proc: subprocess.Popen | None = None
        self.max_adc = 32767.0
        self.info = ""

    # ------------------------------------------------------------ lifecycle
    def open(self, probe: bool = False) -> dict:
        library, python = find_library(), find_python()
        if library is None or python is None:
            raise BridgeUnavailable(describe())

        tail = [str(HELPER), "--lib", library,
                "--interval-us", str(self.interval_us),
                "--buffer", str(self.buffer_size),
                "--range", str(self.channel_range)]
        if probe:
            tail.append("--probe")
        argv, env = _command(python, library, tail)
        self.proc = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE, env=env, bufsize=0)
        answer = self._line(timeout=25.0)
        self.max_adc = float(answer.get("max_adc", 32767))
        self.info = answer.get("info", "")
        return answer

    def release(self) -> int:
        """Start streaming. Returns the interval the driver actually used, in µs."""
        self.proc.stdin.write(b"G")
        self.proc.stdin.flush()
        return int(self._line(timeout=15.0).get("interval_us", self.interval_us))

    def close(self) -> None:
        proc, self.proc = self.proc, None
        if proc is None:
            return
        try:
            # Closing stdin is the polite stop: the helper sees EOF, closes the unit and
            # exits. Terminate only if it does not take the hint.
            if proc.stdin and not proc.stdin.closed:
                proc.stdin.close()
            proc.terminate()
            proc.wait(timeout=5.0)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    # --------------------------------------------------------------- reading
    def _stderr(self) -> str:
        """Whatever the helper complained about, for an error message."""
        try:
            if self.proc.stderr is not None:
                return self.proc.stderr.read().decode("utf-8", "replace").strip()[-400:]
        except Exception:
            pass
        return ""

    def _line(self, timeout: float) -> dict:
        """One JSON handshake line, or raise with the helper's own words."""
        raw = b""
        deadline = time.monotonic() + timeout
        while not raw.endswith(b"\n"):
            if time.monotonic() > deadline:
                self.close()
                raise RuntimeError(f"ps4000 helper did not answer within {timeout:.0f} s")
            if not select.select([self.proc.stdout], [], [], 0.2)[0]:
                if self.proc.poll() is not None:
                    raise RuntimeError(self._stderr()
                                       or f"ps4000 helper exited ({self.proc.returncode})")
                continue
            chunk = self.proc.stdout.read(1)
            if not chunk:
                raise RuntimeError(self._stderr() or "ps4000 helper closed its pipe")
            raw += chunk
        try:
            answer = json.loads(raw.decode("utf-8"))
        except ValueError:
            raise RuntimeError(f"ps4000 helper said {raw!r}") from None
        if not answer.get("ok"):
            raise RuntimeError(answer.get("error", "ps4000 helper failed"))
        return answer

    def block(self, stopping) -> bytes | None:
        """The next sample block, or None when the helper has finished.

        ``stopping`` is polled while waiting so a stop request is not held up by a
        driver that has gone quiet.
        """
        header = self._exact(4, stopping)
        if header is None:
            return None
        count = struct.unpack("<i", header)[0]
        if count <= 0:
            return b""
        return self._exact(4 * count, stopping)      # int16 x 2 channels

    def _exact(self, nbytes: int, stopping) -> bytes | None:
        buf = b""
        while len(buf) < nbytes:
            if stopping():
                return None
            if not select.select([self.proc.stdout], [], [], 0.2)[0]:
                if self.proc.poll() is not None:
                    return None
                continue
            chunk = self.proc.stdout.read(nbytes - len(buf))
            if not chunk:
                return None
            buf += chunk
        return buf
