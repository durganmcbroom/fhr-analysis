"""WAV loading + timestamp export for the beat-marking app.

The backend loads the uploaded audio into the project's :class:`analyze.data.Audio`
so the existing detectors can run against it unchanged. The browser decodes the
same file independently for display/playback, so everything is keyed on *seconds*
(the time axis both sides agree on) rather than sample indices.
"""

from __future__ import annotations

import io
from typing import Tuple

import numpy as np

from analyze.data import Audio


def load_audio_from_bytes(raw: bytes) -> Audio:
    """Decode WAV bytes into a mono, float :class:`Audio` with a 0-based time axis.

    Tries :func:`scipy.io.wavfile.read` first (covers PCM8/16/32 and float WAV);
    falls back to ``torchaudio`` for anything scipy can't parse. Stereo is mixed to
    mono; integer PCM is normalised to roughly [-1, 1] so detector thresholds see
    sane magnitudes (this is monotonic, so peak locations are unaffected)."""
    try:
        from scipy.io import wavfile

        rate, data = wavfile.read(io.BytesIO(raw))
        data = np.asarray(data)
        was_integer = np.issubdtype(data.dtype, np.integer)
        if data.ndim > 1:
            data = data.mean(axis=1)
        data = data.astype(np.float64)
        # Normalise integer PCM by its full-scale range (monotonic; peaks unchanged).
        if was_integer:
            peak = float(np.max(np.abs(data))) or 1.0
            data = data / peak
        rate = int(rate)
    except Exception:
        import torchaudio

        wav, rate = torchaudio.load(io.BytesIO(raw))
        data = wav.mean(dim=0).cpu().numpy().astype(np.float64)
        rate = int(rate)

    n = data.shape[0]
    time = np.arange(n, dtype=np.float64) / float(rate)
    return Audio(time=time, hz=rate, data=data)


def times_to_yaml(times: np.ndarray, name: str = "") -> str:
    """Serialise beat timestamps (seconds) to a small, human-editable YAML doc."""
    times = np.asarray(times, dtype=float).ravel()
    try:
        import yaml

        payload = {
            "source": name,
            "unit": "seconds",
            "count": int(times.size),
            "beats": [float(t) for t in times],
        }
        return yaml.safe_dump(payload, sort_keys=False, default_flow_style=False)
    except Exception:
        # Zero-dependency fallback so export never fails.
        lines = [f"source: {name}", "unit: seconds", f"count: {times.size}", "beats:"]
        lines += [f"  - {float(t):.9g}" for t in times]
        return "\n".join(lines) + "\n"


def times_to_npy_bytes(times: np.ndarray) -> bytes:
    """Serialise beat timestamps to ``.npy`` bytes (a 1-D float64 array)."""
    times = np.asarray(times, dtype=np.float64).ravel()
    buf = io.BytesIO()
    np.save(buf, times, allow_pickle=False)
    return buf.getvalue()


def npy_bytes_to_times(raw: bytes) -> np.ndarray:
    """Load timestamps from an uploaded ``.npy`` file.

    Accepts a plain 1-D array of times, or a 2-D array where one column is time
    (the first column is used, matching the project's ``[:, 0]`` time convention)."""
    arr = np.load(io.BytesIO(raw), allow_pickle=False)
    arr = np.asarray(arr, dtype=float)
    if arr.ndim == 2 and arr.shape[1] >= 1:
        arr = arr[:, 0]
    arr = arr.ravel()
    arr = arr[np.isfinite(arr)]
    return np.sort(arr)
