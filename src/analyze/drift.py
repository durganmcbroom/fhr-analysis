"""Correct NST/mic beat timestamps for dropped-sample clock drift.

The NST recording's nominal clock (sample_index / mic_fs) falls behind real
elapsed time whenever samples are dropped -- fewer samples were actually
captured than the recording's nominal rate assumes for that stretch of real
time. A drift log (one row per detected dropout, giving the nominal NST time
it happened at and how much real time it cost) lets any downstream timestamp
-- e.g. detect_v7's beat times -- be corrected without re-deriving the raw
waveform.
"""
import csv as csv_module
from pathlib import Path

import numpy as np
import numpy.typing as npt


def load_drift_log(path) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Load a dropout log CSV with columns ``time_s`` (nominal NST time the
    dropout was detected at) and ``seconds_lost`` (how much real time that
    dropout cost). Returns ``(event_times, seconds_lost)``, sorted by time."""
    times, losses = [], []
    with open(path, newline="") as f:
        for row in csv_module.DictReader(f):
            times.append(float(row["time_s"]))
            losses.append(float(row["seconds_lost"]))

    times = np.asarray(times, dtype=float)
    losses = np.asarray(losses, dtype=float)
    order = np.argsort(times)
    return times[order], losses[order]


def correct_drift(beat_times, drift_log_path) -> npt.NDArray[np.float64]:
    """Shift each of ``beat_times`` later by however much cumulative drift the
    log says had already accrued by that nominal time, undoing the NST clock
    falling behind. Returns ``beat_times`` unchanged if the log is missing or
    empty."""
    beat_times = np.asarray(beat_times, dtype=float)
    if not Path(drift_log_path).exists():
        return beat_times

    event_times, seconds_lost = load_drift_log(drift_log_path)
    if len(event_times) == 0:
        return beat_times

    cumulative = np.concatenate(([0.0], np.cumsum(seconds_lost)))
    idx = np.searchsorted(event_times, beat_times, side="right")
    return beat_times + cumulative[idx]
