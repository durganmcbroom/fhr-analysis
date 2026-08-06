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

# Interpreters worth trying. Order is a preference, not a filter: every one of these is
# actually *run* before it is believed (see _usable), because the properties that matter
# cannot be read off the file. python.org's `-intel64` binaries come first only because
# they need no `arch` wrapper.
_PYTHON_GLOBS = (
    "/usr/local/bin/python3*-intel64",
    "/usr/local/bin/python3*",
    "/Library/Frameworks/Python.framework/Versions/*/bin/python3*",
    # Shipped with the Xcode command line tools, universal, and -- unlike
    # /usr/bin/python3 -- not SIP-restricted, so it keeps the loader path the driver
    # needs. On a developer's Mac with no other Intel Python this is usually the one
    # that works.
    "/Library/Developer/CommandLineTools/usr/bin/python3*",
    "/Applications/Xcode.app/Contents/Developer/usr/bin/python3*",
    "/opt/homebrew/bin/python3*",
    "/opt/local/bin/python3*",                       # MacPorts
    "~/miniconda3*/bin/python3*", "~/miniforge3*/bin/python3*",
    "~/anaconda3*/bin/python3*", "~/opt/anaconda3*/bin/python3*",
    "~/miniconda3*/envs/*/bin/python3*", "~/miniforge3*/envs/*/bin/python3*",
    "~/anaconda3*/envs/*/bin/python3*", "~/opt/anaconda3*/envs/*/bin/python3*",
    "~/.pyenv/versions/*/bin/python3*",
    "~/.local/share/uv/python/*/bin/python3*",
)

# Directories that hold an x86_64 libps4000 on a Mac. The framework paths are where the
# PicoSDK installer puts drivers; the application bundles are where an Intel build
# survives on a machine whose SDK has moved on to arm64. Globbed, because the
# PicoScope 7 bundle is named after its edition ("T&M", "Automotive", "Early Access").
_LIB_GLOBS = (
    "/Library/Frameworks/PicoSDK.framework/Libraries/libps4000",
    "/Library/Frameworks/PicoSDK.framework/Libraries",
    "/Applications/PicoScope*.app/Contents/Resources",
    "/Applications/PicoScope*.app/Contents/MonoBundle",
    "/Applications/Pico*/PicoScope*.app/Contents/Resources",
    "~/Applications/PicoScope*.app/Contents/Resources",
    "~/lib",
    "/usr/local/lib",
    "/opt/picoscope/lib",
)

# Directories added to the helper's DYLD_LIBRARY_PATH beyond the driver's own. Once the
# unit is opened the driver dlopens libpicoipp and libiomp5 *by bare name*, and in the
# framework layout each library lives in its own directory. Preloading them by absolute
# path does not work as a substitute: libpicoipp.dylib's install name is
# libpicoipp.1.dylib, so an already-loaded image never matches the leaf name the driver
# asks for. The search path is the only lever.
_DEP_GLOBS = (
    "/Library/Frameworks/PicoSDK.framework/Libraries/libpicoipp",
    "/Library/Frameworks/PicoSDK.framework/Libraries/libiomp5",
    "/Applications/PicoScope*.app/Contents/Resources",
    "/Applications/PicoScope*.app/Contents/MonoBundle",
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


def _expand(patterns) -> list[str]:
    found = []
    for pattern in patterns:
        found.extend(sorted(glob.glob(os.path.expanduser(pattern)), reverse=True))
    return found


def library_candidates() -> list[str]:
    """Every ``libps4000.dylib`` looked at, whatever its architecture."""
    override = os.environ.get("RTMON_PS4000_LIB_DIR")
    roots = ([override] if override else []) + _expand(_LIB_GLOBS)
    seen, out = set(), []
    for root in roots:
        path = os.path.join(os.path.expanduser(root), "libps4000.dylib")
        if path not in seen and os.path.isfile(path):
            seen.add(path)
            out.append(path)
    return out


def find_library() -> str | None:
    """An x86_64 ``libps4000.dylib``, or None."""
    return next((p for p in library_candidates() if _is_x86_64(p)), None)


def python_candidates() -> list[str]:
    override = os.environ.get("RTMON_PS4000_PYTHON")
    if override:
        return [override]
    if sys.platform != "darwin":
        return []
    seen, out = set(), []
    for path in _expand(_PYTHON_GLOBS):
        real = os.path.realpath(path)
        if real in seen or os.path.isdir(path) or not os.access(path, os.X_OK):
            continue
        if path.endswith(("-config", ".pyc")):
            continue
        seen.add(real)
        out.append(path)
    return out


# Printed by a candidate that is genuinely usable. Both halves matter: the interpreter
# has to be running as x86_64, *and* the loader path we set has to have survived the
# exec -- macOS strips DYLD_* from Apple-signed binaries, and a driver that cannot find
# libpicoipp fails much later with a status code that says nothing about why.
_PROBE = ("import os,platform;"
          "print('rtmon-ps4000', platform.machine(), os.environ.get('DYLD_LIBRARY_PATH'))")
_SENTINEL = "/rtmon-ps4000-probe"
_usable_cache: dict[tuple, bool] = {}


def _has_intel_slice(path: str) -> bool:
    """False only when the file is a Mach-O we could read that has no x86_64 in it.

    A pre-filter, not a decision: skipping an arm64-only interpreter saves launching it
    to be told what its header already said. Anything that is not a readable Mach-O --
    a pyenv shim, a wrapper script -- is still tried, because it may exec something
    else entirely.
    """
    archs = architectures(path)
    return not archs or MACHO_X86_64 in archs


def _usable(python: list[str]) -> bool:
    """Run the candidate and see. Cached, because probing repeats on every rescan.

    Testing rather than inferring is the point. The previous version read Mach-O headers
    and applied path rules, and got it wrong in the field: it skipped every Apple-signed
    interpreter to dodge the SIP problem, and that also skipped the command line tools'
    Python -- universal, unrestricted, on every developer's Mac, and on a machine with
    no python.org install the only candidate that would have worked.
    """
    key = tuple(python)
    if key in _usable_cache:
        return _usable_cache[key]
    if not _has_intel_slice(python[-1]):
        _usable_cache[key] = False
        return False
    argv, env = _command(python, _SENTINEL, ["-c", _PROBE])
    try:
        done = subprocess.run(argv, env=env, capture_output=True, timeout=25)
        answer = done.stdout.decode("utf-8", "replace").split()
        ok = (len(answer) == 3 and answer[0] == "rtmon-ps4000"
              and answer[1] == "x86_64" and _SENTINEL in answer[2])
    except Exception:  # noqa: BLE001 - an unusable candidate, whatever went wrong
        ok = False
    _usable_cache[key] = ok
    return ok


def find_python() -> list[str] | None:
    """Argv prefix that runs a script as x86_64 with a loader path, or None.

    Each candidate is tried directly first and then through ``arch -x86_64``: a binary
    whose only slice is Intel needs no wrapper, a universal one does, and rather than
    work out which is which from the header, both are simply attempted.
    """
    have_arch = bool(shutil.which("arch"))
    for path in python_candidates():
        for python in ([path], ["arch", "-x86_64", path] if have_arch else None):
            if python and _usable(python):
                return python
    return None


def describe() -> str:
    """Why the bridge cannot run, in terms of what to install. Empty when it can."""
    if find_library() is None:
        seen = library_candidates()
        wrong = ", ".join(seen[:2])
        return ((f"Found libps4000 at {wrong}, but not built for Intel — the arm64 "
                 f"PicoSDK ships ps4000a only. " if seen else "No libps4000 found at all. ")
                + "Install PicoScope 6 or 7 (its app bundle contains an Intel build), or "
                  "set RTMON_PS4000_LIB_DIR to a directory holding one. Run "
                  "`python -m rtmon.sources.ps4000_bridge` for the full search.")
    if find_python() is None:
        tried = len(python_candidates())
        return (f"No Intel-capable Python found to run the ps4000 helper in ({tried} "
                f"candidate{'s' if tried != 1 else ''} tried). Any will do — it needs "
                f"only the standard library — but not /usr/bin/python3, whose loader "
                f"path macOS strips. Installing the Xcode command line tools "
                f"(`xcode-select --install`) or python.org's universal build is enough; "
                f"or set RTMON_PS4000_PYTHON. Run "
                f"`python -m rtmon.sources.ps4000_bridge` for the full search.")
    return ""


def report() -> str:
    """Everything the search looked at and what it concluded. For `python -m`.

    This exists because the bridge fails on someone else's machine, not yours, and
    "it says no" is not enough to act on -- the answer is always one of two specific
    missing pieces, and which one decides what to install.
    """
    import platform

    lines = [f"rtmon ps4000 bridge — {platform.platform()} / {platform.machine()}", ""]

    lines.append("Intel libps4000:")
    candidates = library_candidates()
    if not candidates:
        lines.append("  (none found)")
    for path in candidates:
        archs = architectures(path)
        names = ",".join(sorted({0x01000007: "x86_64", 0x0100000C: "arm64"}.get(a, hex(a))
                                for a in archs)) or "unreadable"
        lines.append(f"  [{'ok ' if MACHO_X86_64 in archs else 'no '}] {path}  ({names})")
    chosen_lib = find_library()
    lines += [f"  -> using {chosen_lib}" if chosen_lib else "  -> NONE USABLE", ""]

    lines.append("Intel-capable Python (each is run, not guessed at):")
    pythons = python_candidates()
    if not pythons:
        lines.append("  (none found)")
    have_arch = bool(shutil.which("arch"))
    chosen_py = None
    for path in pythons:
        marks = []
        for python in ([path], ["arch", "-x86_64", path] if have_arch else None):
            if not python:
                continue
            ok = _usable(python)
            marks.append(("direct" if len(python) == 1 else "arch") + ("=ok" if ok else "=no"))
            if ok and chosen_py is None:
                chosen_py = python
        lines.append(f"  [{'ok ' if 'ok' in ' '.join(marks) else 'no '}] {path}  "
                     f"({', '.join(marks)})")
    lines += [f"  -> using {' '.join(chosen_py)}" if chosen_py else "  -> NONE USABLE", ""]

    if chosen_lib:
        lines += ["Loader path handed to the driver:",
                  *[f"  {d}" for d in loader_path(chosen_lib).split(os.pathsep)], ""]
    problem = describe()
    lines.append(problem if problem else "Bridge is ready. Plug the unit in and rescan.")
    return "\n".join(lines)


def loader_path(library: str) -> str:
    """DYLD_LIBRARY_PATH the driver needs to resolve what it dlopens by bare name."""
    dirs = [os.path.dirname(library)]
    dirs += [d for d in _expand(_DEP_GLOBS) if os.path.isdir(d)]
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


def _command(python: list[str], dyld_path: str, tail: list[str]) -> tuple[list[str], dict]:
    """The argv and environment that get DYLD_LIBRARY_PATH all the way to the driver.

    Setting it in the child's environment is not enough when ``arch`` is in the way:
    ``arch`` is Apple-signed, so System Integrity Protection strips every ``DYLD_*``
    variable as it execs, and the driver then fails to load libpicoipp with a status
    code that looks nothing like "your loader path was deleted". ``arch -e`` sets the
    variable on the far side of that boundary, which is the only place it survives.
    """
    env = dict(os.environ)
    env["DYLD_LIBRARY_PATH"] = dyld_path
    if python and os.path.basename(python[0]) == "arch":
        return [*python[:-1], "-e", f"DYLD_LIBRARY_PATH={dyld_path}", python[-1], *tail], env
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
        argv, env = _command(python, loader_path(library), tail)
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


if __name__ == "__main__":       # python -m rtmon.sources.ps4000_bridge
    print(report())
