"""Heart-rate trace agreement between a model's beat activity and the target comb.

The training losses score a model frame by frame; this scores the thing the model is actually
for -- the heart-rate plot it produces. Beats are picked out of the predicted activity and out
of the target comb *with the same detector*, each beat train becomes an instantaneous-BPM
trace, and the two traces are compared with a plain Pearson r.

Two deliberate departures from ``analyze.evaluate_v3``, which scores whole recordings:

* **No lag search.** Validation snippets are time-aligned with their targets by construction,
  so the correlation is taken at zero lag. (evaluate_v3 searches +/- 5 s because a real
  recording's SOT has an unknown offset -- and on a 7 s snippet that search would slide the
  prediction almost entirely off the reference.)
* **No beat matching.** Nothing here measures whether individual beats land within a tolerance;
  only whether the two BPM traces move together.

Scale-invariance is the useful property: BPM traces live in physical units (bpm against
seconds), so unlike a per-frame loss this number is comparable across models whose frame rates
differ. A per-frame MSE is not -- halving ``hop_length`` spreads the same beats over twice the
frames and mechanically lowers it (see FUNetTask.frozen_fields).

Caveat worth remembering when reading the number: over a window as short as a training crop,
essentially all the variance in a BPM trace comes from beat-to-beat interval jitter, so at that
length this is closer to a measure of beat-timing precision than of slow accel/decel tracking.
"""

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
from scipy.signal import find_peaks

#: Plausible fetal range, mirroring analyze.constants.FETAL_BPM_RANGE. Duplicated rather than
#: imported: common is a model library and must not depend on the analysis stack.
FETAL_BPM_RANGE = (90.0, 280.0)


@dataclass
class HRScore:
    """Result of scoring a whole validation split."""

    mean: float           #: mean per-snippet Pearson r -- the selection metric, higher is better
    std: float            #: spread across snippets; two configs closer than this are tied
    n: int                #: snippets scored
    n_degenerate: int     #: snippets that produced no usable BPM trace, counted as r = 0

    def __str__(self) -> str:
        degenerate = f", {self.n_degenerate}/{self.n} degenerate" if self.n_degenerate else ""
        return f"{self.mean:.4f} +/- {self.std:.4f}{degenerate}"


def detect_beats(activity: np.ndarray, frame_hz: float, max_bpm: float) -> np.ndarray:
    """Beat times (seconds) as peaks of a non-negative activity envelope.

    The same detector runs on the prediction and on the target, which is what makes the two
    BPM traces comparable: any systematic quirk of the peak picker applies to both sides.
    """
    if activity.size < 3:
        return np.empty(0)
    # A beat cannot follow another faster than max_bpm allows; height keeps the picker off the
    # noise floor without pinning an absolute scale (the envelope's units are arbitrary).
    distance = max(1, int(round(60.0 / max_bpm * frame_hz)))
    height = activity.mean() + 0.5 * activity.std()
    peaks, _ = find_peaks(activity, distance=distance, height=height)
    return peaks / frame_hz


def bpm_trace(beat_times: np.ndarray,
              bpm_range: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray]:
    """Instantaneous rate as ``(t, bpm)``: one point per inter-beat interval, timestamped at
    the interval's midpoint. Intervals implying an out-of-band rate are dropped as detector
    errors (a missed beat halves the apparent rate, a doubled one doubles it)."""
    if beat_times.size < 2:
        return np.empty(0), np.empty(0)
    ibi = np.diff(beat_times)
    ibi = np.where(ibi > 0, ibi, np.nan)          # guard a zero interval -> inf bpm
    bpm = 60.0 / ibi
    t = beat_times[:-1] + ibi / 2.0
    keep = np.isfinite(bpm) & (bpm >= bpm_range[0]) & (bpm <= bpm_range[1])
    return t[keep], bpm[keep]


def snippet_correlation(
        pred_activity: np.ndarray,
        target: np.ndarray,
        frame_hz: float,
        bpm_range: Tuple[float, float] = FETAL_BPM_RANGE,
        grid_hz: float = 10.0,
        min_span_s: float = 1.0,
) -> Optional[float]:
    """Pearson r between the predicted and target BPM traces for one snippet.

    The two traces carry one sample per detected interval, so they are neither the same length
    nor on the same timestamps; both are resampled onto a uniform grid over the span they both
    cover before correlating. Returns ``None`` when no meaningful correlation exists -- too few
    beats on either side, too little overlap, or a trace with no variance (a perfectly constant
    rate correlates with nothing). Callers decide what a degenerate snippet scores.
    """
    tp, bp = bpm_trace(detect_beats(pred_activity, frame_hz, bpm_range[1]), bpm_range)
    tr, br = bpm_trace(detect_beats(target, frame_hz, bpm_range[1]), bpm_range)
    if tp.size < 2 or tr.size < 2:
        return None

    start, end = max(tp[0], tr[0]), min(tp[-1], tr[-1])
    if end - start < min_span_s:
        return None
    n = int((end - start) * grid_hz) + 1
    if n < 3:
        return None

    grid = np.linspace(start, end, n)
    p = np.interp(grid, tp, bp)
    r = np.interp(grid, tr, br)
    if p.std() == 0 or r.std() == 0:
        return None

    corr = float(np.corrcoef(p, r)[0, 1])
    return None if not np.isfinite(corr) else corr


class HRCorrelation:
    """Accumulates :func:`snippet_correlation` over a validation split.

    One instance scores one pass over the split: construct, ``update`` per batch, read
    ``result``. Degenerate snippets score 0 rather than being dropped -- dropping them would
    score a model only on the snippets where it happened to fire, which rewards a model that
    emits almost nothing.

    The per-snippet r values are averaged plainly. Fisher-z averaging is the textbook
    correction for the fact that r is bounded and its sampling distribution skewed, but
    measured at this sample size (~15 intervals per snippet) the raw bias is only ~0.01 and the
    z-transform overcorrects, because an occasional r near +/-1 sends arctanh to infinity.
    """

    def __init__(
            self,
            frame_hz: float,
            bpm_range: Tuple[float, float] = FETAL_BPM_RANGE,
            postprocess: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ):
        self.frame_hz = frame_hz
        self.bpm_range = bpm_range
        # Maps raw model output to the non-negative envelope the beat detector expects, exactly
        # as inference does (common.phases.inference.activity_postprocess): a log-prob head has
        # to be exp'd before its peaks mean anything.
        self.postprocess = postprocess
        self._values: List[float] = []
        self._degenerate = 0

    def update(self, output: torch.Tensor, target: torch.Tensor) -> None:
        """Score every item in a ``(batch, time)`` output/target pair."""
        if self.postprocess is not None:
            output = self.postprocess(output)
        out = output.detach().float().cpu().numpy()
        tgt = target.detach().float().cpu().numpy()

        for pred_i, target_i in zip(out, tgt):
            value = snippet_correlation(pred_i, target_i, self.frame_hz, self.bpm_range)
            if value is None:
                self._degenerate += 1
                value = 0.0
            self._values.append(value)

    def result(self) -> HRScore:
        if not self._values:
            return HRScore(mean=0.0, std=0.0, n=0, n_degenerate=0)
        values = np.asarray(self._values)
        return HRScore(mean=float(values.mean()), std=float(values.std()),
                       n=values.size, n_degenerate=self._degenerate)
