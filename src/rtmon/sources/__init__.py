"""Every acquisition source the rig can have, declared unconditionally.

Registration is not a claim that the device is present -- that is what
:meth:`~rtmon.sources.base.Source.probe` answers, per machine, at runtime. See
``base.py`` for why this is a registry rather than a block of imports guarded by
try/except or comments.
"""

from __future__ import annotations

import os

from rtmon.sources.base import (
    KIND_AUDIO, KIND_FIBER, KIND_PPG, Channel, Probe, SampleClock, Sink, Source,
)
from rtmon.sources.microphone import MicrophoneSource
from rtmon.sources.picoscope import PS3000ASource, PS4000Source
from rtmon.sources.polar import PolarSource
from rtmon.sources.synthetic import sim_mic, sim_ps3000a, sim_ps4000, sim_pvs

# The file each source's samples are recorded to, and therefore which offline loader
# picks them up. A simulator records under the same name as the device it replaces, so
# a session captured with a partly-simulated rig loads exactly like a real one.
RECORD_AS = {
    "ps4000": "ps4000",
    "ps3000a": "ps3000a",
    "mic": "microphone",
    "pvs": "pvs",
    "sim-ps4000": "ps4000",
    "sim-ps3000a": "ps3000a",
    "sim-mic": "microphone",
    "sim-pvs": "pvs",
}


def build_sources() -> list[Source]:
    """Fresh instances of every source, in the order the UI should list them.

    The device list is the real rig only. The simulators still exist as a bench
    harness -- they are how the whole pipeline gets exercised on a machine with no
    hardware at all -- but they are opt-in via ``RTMON_SIM=1`` rather than four
    permanent extra cards in the panel (and a standing chance of a simulator
    claiming 2A-2D before the real ps3000a is armed).
    """
    sources: list[Source] = [
        PS3000ASource(),
        PS4000Source(),
        MicrophoneSource(),
        PolarSource(),
    ]
    if os.environ.get("RTMON_SIM"):
        sources += [sim_ps3000a(), sim_ps4000(), sim_mic(), sim_pvs()]
    return sources


__all__ = [
    "Channel", "KIND_AUDIO", "KIND_FIBER", "KIND_PPG", "Probe", "RECORD_AS",
    "SampleClock", "Sink", "Source", "build_sources",
]
