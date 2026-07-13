"""Detector discovery + invocation for the beat-marking app.

Auto-discovers every ``*_beat_detector`` function in the ``analyze.hr`` package
(``detect.py``, ``detect_v2.py`` … ``detect_v8.py``, and anything added later) and
exposes them behind a stable id so the frontend can list them in a dropdown and
run one against a loaded waveform.

Every detector shares the same contract:

    detector(X: Audio, bpm_range, out=None, tag="") -> {"times": np.ndarray, ...}

so running one is uniform regardless of its internals.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
import re
from typing import Callable, Dict, List, Tuple

import numpy as np

from analyze.data import Audio


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

_VERSION_RE = re.compile(r"v(\d+)_beat_detector")


def discover_detectors() -> Dict[str, Callable]:
    """Import every ``analyze.hr.detect*`` module and collect the callables whose
    name ends in ``_beat_detector``. Keyed by the bare function name (unique across
    the package, e.g. ``v8_beat_detector``)."""
    import analyze.hr as hr_pkg

    registry: Dict[str, Callable] = {}
    for _finder, modname, _ispkg in pkgutil.iter_modules(hr_pkg.__path__):
        if not modname.startswith("detect"):
            continue
        try:
            mod = importlib.import_module(f"analyze.hr.{modname}")
        except Exception as exc:  # a broken/optional detector module shouldn't kill the app
            print(f"[detectors] skipping analyze.hr.{modname}: {exc}")
            continue
        for name, fn in inspect.getmembers(mod, inspect.isfunction):
            # Only functions actually defined in this module (not re-imported).
            if fn.__module__ != mod.__name__:
                continue
            if name.endswith("_beat_detector"):
                registry[name] = fn
    return registry


def _sort_key(name: str) -> Tuple[int, str]:
    m = _VERSION_RE.search(name)
    return (int(m.group(1)) if m else 9999, name)


def list_detectors() -> List[dict]:
    """Return ``[{"id", "label", "doc"}]`` for the UI dropdown, version-ordered."""
    reg = discover_detectors()
    items = []
    for name in sorted(reg, key=_sort_key):
        fn = reg[name]
        m = _VERSION_RE.search(name)
        label = f"v{m.group(1)}" if m else name.replace("_beat_detector", "")
        doc = (inspect.getdoc(fn) or "").strip().split("\n", 1)[0]
        items.append({"id": name, "label": f"{label}  ({name})", "doc": doc})
    return items


# ---------------------------------------------------------------------------
# Invocation
# ---------------------------------------------------------------------------

def run_detector(detector_id: str, audio: Audio, bpm_range: Tuple[float, float]) -> np.ndarray:
    """Run the detector named ``detector_id`` on ``audio`` and return sorted beat
    times (seconds). Plotting is disabled by passing ``out=None``."""
    reg = discover_detectors()
    if detector_id not in reg:
        raise KeyError(f"unknown detector {detector_id!r}; have {sorted(reg)}")
    fn = reg[detector_id]

    result = fn(audio, tuple(bpm_range), None, tag="beat_app")

    times = np.asarray(result["times"], dtype=float).ravel()
    times = times[np.isfinite(times)]
    return np.sort(times)
