"""Saved setups -- the matrix, the armed devices, and the view, as one named thing.

A rig is not reconfigured once. It is reconfigured per patient, per room, per
question ("does v35 beat v21 on this belly?"), and the configuration is worth more
than the five minutes it takes to retype. Setups are plain JSON in ``.out/rtmon/``
so they can be diffed, copied between machines, and pasted into a lab notebook.

``last.json`` is written on every change, so the app comes back up as it was left.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from analyze.constants import ABDOMEN_FIBER_NAMES, PROJECT_DIR
from rtmon import models as model_registry
from rtmon.engine import TRACK_COLORS, Track
from rtmon.sources import polar

STORE_DIR = Path(PROJECT_DIR) / ".out" / "rtmon"
PRESET_DIR = STORE_DIR / "presets"
LAST_PATH = STORE_DIR / "last.json"

_SAFE_NAME = re.compile(r"[^A-Za-z0-9 _-]+")


@dataclass
class Setup:
    tracks: list[Track] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)      # source ids to arm
    window_s: float = 10.0                                 # waveform view span
    channels: list[str] = field(default_factory=list)      # channels shown as scopes
    mic_device: str | None = None                          # input device NAME; None = default
    # Measured PPG transport latency (rtmon.align). None means "use the default";
    # saved with the setup because it is a property of this strap and this host.
    ppg_latency_s: float | None = None
    # Same, for the microphone: the audio stack's capture latency. Both are corrections
    # ONTO the fiber, which is the timing reference (see rtmon.align).
    mic_latency_s: float | None = None
    # Which fiber everything is tap-aligned against. Remembered because it should be the
    # fiber physically nearest the sensors, which is a property of how the rig is laid out.
    align_fiber: str | None = None
    # Which view the Signals panel is on: "raw", or a band name whose bandpass is drawn
    # instead. A view setting like window_s, saved for the same reason -- the operator
    # who works in the fetal band wants it that way when they come back.
    signal_view: str = "raw"
    name: str = "unsaved"

    def to_json(self) -> dict:
        return {"name": self.name, "tracks": [t.to_json() for t in self.tracks],
                "sources": self.sources, "window_s": self.window_s,
                "channels": self.channels, "mic_device": self.mic_device,
                "ppg_latency_s": self.ppg_latency_s, "mic_latency_s": self.mic_latency_s,
                "align_fiber": self.align_fiber, "signal_view": self.signal_view}

    @staticmethod
    def from_json(raw: dict) -> "Setup":
        return Setup(
            tracks=[Track.from_json(t) for t in raw.get("tracks", [])],
            sources=list(raw.get("sources", [])),
            window_s=float(raw.get("window_s", 10.0)),
            channels=list(raw.get("channels", [])),
            mic_device=raw.get("mic_device") or None,
            # Sanitised on the way in: a setup file is editable, may predate a fix, and
            # a nonsense latency here would silently skew every PPG timestamp.
            ppg_latency_s=polar.valid_latency(raw.get("ppg_latency_s")),
            mic_latency_s=polar.valid_latency(raw.get("mic_latency_s")),
            align_fiber=raw.get("align_fiber") or None,
            signal_view=str(raw.get("signal_view") or "raw"),
            name=raw.get("name", "unsaved"),
        )


def default_setup() -> Setup:
    """A setup that makes sense on a full rig, and degrades honestly on a partial one.

    Tracks are created for the pipelines the project actually compares -- acoustic SOT,
    NeoSSNet on one fiber, FUNet on the abdomen stack -- pointed at the newest checkpoint
    of each family that fits. Anything whose inputs are not streaming shows up disabled
    with the reason, rather than being silently omitted; that is the point of the
    validation column.
    """
    funet = _newest("funet", channels=len(ABDOMEN_FIBER_NAMES))
    ssnet = _newest("ssnet")
    tslnet = _newest("tslnet")

    # Two references, because two hearts are being measured: the microphone is the
    # fetal source of truth, the PPG strap the maternal one. Each band's estimates are
    # scored against its own; nothing is ever compared across bands.
    tracks = [
        Track(id="sot-fetal", name="SOT — microphone", processor="acoustic", inputs=["MIC"],
              detector="v7_beat_detector", band="fetal", role="sot",
              chunk_s=10.0, period_s=5.0, color=TRACK_COLORS[0]),
        Track(id="sot-maternal", name="SOT — PPG strap", processor="ppg", inputs=["PPG0"],
              band="maternal", role="sot",
              chunk_s=15.0, period_s=5.0, color=TRACK_COLORS[4]),
    ]
    if funet is not None:
        tracks.append(Track(
            id="funet", name=f"FUNet — {funet.version}", processor="funet",
            inputs=list(ABDOMEN_FIBER_NAMES), model=funet.version, band="fetal",
            chunk_s=10.0, period_s=5.0, color=TRACK_COLORS[1]))
    if ssnet is not None:
        tracks.append(Track(
            id="ssnet", name=f"NeoSSNet — 1B", processor="ssnet", inputs=["1B"],
            model=ssnet.version, detector="v7_beat_detector", band="fetal",
            chunk_s=10.0, period_s=10.0, color=TRACK_COLORS[2]))
    if tslnet is not None:
        tracks.append(Track(
            id="tslnet", name=f"TSLNet — {tslnet.version}", processor="tslnet",
            inputs=_first_n(tslnet.channels), model=tslnet.version, band="fetal",
            enabled=False, chunk_s=10.0, period_s=10.0, color=TRACK_COLORS[3]))

    return Setup(
        tracks=tracks,
        sources=[],
        channels=["MIC", "1A", "1B", "2A", "2B", "2C", "2D"],
        name="default",
    )


def _newest(family: str, channels: int | None = None):
    entries = model_registry.discover(family)
    if channels is not None:
        entries = [e for e in entries if e.channels == channels] or entries
    return entries[-1] if entries else None


def _first_n(n: int) -> list[str]:
    pool = ABDOMEN_FIBER_NAMES + ["1A"]
    return pool[:n]


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_last(setup: Setup) -> None:
    STORE_DIR.mkdir(parents=True, exist_ok=True)
    _atomic_write(LAST_PATH, setup.to_json())


def load_last() -> Setup | None:
    if not LAST_PATH.is_file():
        return None
    try:
        return Setup.from_json(json.loads(LAST_PATH.read_text()))
    except Exception:
        return None


def list_presets() -> list[dict]:
    if not PRESET_DIR.is_dir():
        return []
    out = []
    for path in sorted(PRESET_DIR.glob("*.json")):
        try:
            raw = json.loads(path.read_text())
            out.append({"name": raw.get("name", path.stem), "file": path.name,
                        "tracks": len(raw.get("tracks", [])),
                        "saved_at": raw.get("saved_at")})
        except Exception:
            continue
    return out


def save_preset(setup: Setup, name: str) -> str:
    clean = _SAFE_NAME.sub("", name).strip() or "preset"
    PRESET_DIR.mkdir(parents=True, exist_ok=True)
    payload = setup.to_json()
    payload["name"] = clean
    payload["saved_at"] = time.time()
    _atomic_write(PRESET_DIR / f"{clean}.json", payload)
    return clean


def load_preset(name: str) -> Setup:
    clean = _SAFE_NAME.sub("", name).strip()
    path = PRESET_DIR / f"{clean}.json"
    if not path.is_file():
        raise FileNotFoundError(f"no preset named {name!r}")
    return Setup.from_json(json.loads(path.read_text()))


def delete_preset(name: str) -> None:
    clean = _SAFE_NAME.sub("", name).strip()
    (PRESET_DIR / f"{clean}.json").unlink(missing_ok=True)


def _atomic_write(path: Path, payload: dict) -> None:
    """Write via a temp file and rename, so a crash mid-save cannot leave a truncated
    setup that the next start would silently fall back from."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(path)
