"""Discovery and caching of the trained checkpoints under ``lib/``.

The Qt panel hard-wired one FUNet and one NeoSSNet -- whichever
``analyze.constants`` happened to point at -- so comparing two model versions live
meant editing a constant and restarting. Every version on disk is listed here
instead, with the facts the UI needs to stop you configuring an impossible track:
how many input channels the checkpoint expects, and what it was trained on.

The channel count is read straight out of the run's own YAML rather than from the
model object, so listing thirty-odd versions costs thirty small file reads and no
torch imports.

:class:`ModelCache` then keeps loaded models resident. This matters most for the
NeoSSNet path: ``neossnet.utils.generate_output`` takes *paths*, so the offline code
re-reads and re-builds the checkpoint on every call -- which, at one call per chunk,
is a disk read and a full model construction several times a minute.
"""

from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import yaml

from analyze.constants import PROJECT_DIR

_ROOT = Path(PROJECT_DIR)

# family -> (directory of versions, config filenames in preference order)
FAMILIES = {
    "funet": (_ROOT / "lib/funet/models", ("fetal-config.yaml", "config.yaml")),
    "tslnet": (_ROOT / "lib/tslnet/models", ("config.yaml",)),
    "ssnet": (_ROOT / "lib/tune-ssnet/models", ("model.yaml",)),
}
CHECKPOINTS = ("model_best.pt", "model_last.pt")

_VERSION_RE = re.compile(r"(\d+)")


@dataclass(frozen=True)
class ModelEntry:
    family: str
    version: str            # directory name, e.g. "funet-v21"
    checkpoint: str
    config: str
    channels: int           # input channels the checkpoint expects (1 for ssnet)
    note: str = ""

    def to_json(self) -> dict:
        return {"family": self.family, "version": self.version, "channels": self.channels,
                "note": self.note, "checkpoint": self.checkpoint, "config": self.config}


def _sort_key(name: str):
    """Numeric-aware ordering: v9 before v21, and unnumbered runs (``funet-v(CONTROL)``)
    ahead of every numbered one, so "the newest version" is the highest number rather
    than whatever sorts last alphabetically."""
    m = _VERSION_RE.search(name)
    return (1 if m else 0, int(m.group(1)) if m else 0, name)


def _channels_and_note(family: str, cfg: dict) -> tuple[int, str]:
    model = cfg.get("model") or {}
    train = cfg.get("train") or {}
    if family == "ssnet":
        # tune-ssnet's model.yaml is the MaskNet constructor's kwargs -- it separates one
        # mixed channel into heart and lung, so its input is always single-channel.
        return 1, f"{model.get('N', '')}x{model.get('B', '')}" if model else ""
    bits = []
    if train.get("loss"):
        bits.append(str(train["loss"]))
    if train.get("crop_len"):
        bits.append(f"{train['crop_len']}s crop")
    if family == "tslnet" and model.get("model_hz"):
        bits.append(f"{model['model_hz']} Hz")
    return int(model.get("channels", 1)), " · ".join(bits)


# Short-lived discovery cache. Track validation runs inside every engine snapshot --
# once a second while a page is open -- and uncached it walked every version directory
# and re-parsed every YAML each time. A few seconds of staleness only delays a *new*
# checkpoint's first appearance in the picker, which is a fine trade for not hitting
# the disk ~50 times a second.
_CACHE_TTL_S = 5.0
_cache: dict[str, tuple[float, list["ModelEntry"]]] = {}
_cache_lock = threading.Lock()


def discover(family: str) -> list[ModelEntry]:
    """Every usable version of ``family``, newest-numbered last. Cached briefly."""
    now = time.monotonic()
    with _cache_lock:
        hit = _cache.get(family)
        if hit is not None and now - hit[0] < _CACHE_TTL_S:
            return hit[1]
    entries = _discover_uncached(family)
    with _cache_lock:
        _cache[family] = (now, entries)
    return entries


def _discover_uncached(family: str) -> list[ModelEntry]:
    directory, config_names = FAMILIES[family]
    if not directory.is_dir():
        return []
    entries = []
    for version_dir in sorted(directory.iterdir(), key=lambda p: _sort_key(p.name)):
        if not version_dir.is_dir():
            continue
        config = next((version_dir / n for n in config_names if (version_dir / n).is_file()), None)
        checkpoint = next((version_dir / n for n in CHECKPOINTS if (version_dir / n).is_file()), None)
        if config is None or checkpoint is None:
            continue  # a half-finished training run, not a model
        try:
            cfg = yaml.safe_load(config.read_text()) or {}
            channels, note = _channels_and_note(family, cfg)
        except Exception as exc:  # noqa: BLE001 - one unreadable config must not hide the rest
            channels, note = 1, f"unreadable config: {exc}"
        entries.append(ModelEntry(family, version_dir.name, str(checkpoint), str(config),
                                  channels, note))
    return entries


def discover_all() -> dict[str, list[ModelEntry]]:
    return {family: discover(family) for family in FAMILIES}


def find(family: str, version: str) -> ModelEntry | None:
    return next((e for e in discover(family) if e.version == version), None)


class ModelCache:
    """Loaded models, keyed by ``(family, version)``.

    Bounded because the checkpoints are not small and a curious user can click
    through thirty versions in a minute; least-recently-used is evicted. The lock is
    held across a load (two threads asking for the same cold model should not both
    build it) but not across inference -- the per-entry lock in the processors covers
    that, since these models are not re-entrant.
    """

    def __init__(self, max_entries: int = 4):
        self.max_entries = max_entries
        self._models: dict[tuple[str, str], object] = {}
        self._order: list[tuple[str, str]] = []
        self._lock = threading.Lock()

    def get(self, family: str, version: str, build):
        """Cached model for ``(family, version)``, calling ``build(entry)`` on a miss."""
        key = (family, version)
        with self._lock:
            if key in self._models:
                self._order.remove(key)
                self._order.append(key)
                return self._models[key]

            entry = find(family, version)
            if entry is None:
                raise KeyError(f"no {family} model named {version!r}")
            model = build(entry)
            self._models[key] = model
            self._order.append(key)
            while len(self._order) > self.max_entries:
                self._models.pop(self._order.pop(0), None)
            return model

    def clear(self) -> None:
        with self._lock:
            self._models.clear()
            self._order.clear()
