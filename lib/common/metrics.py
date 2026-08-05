"""Heart-rate trace agreement between a model's beat activity and the target comb.

The training losses score a model frame by frame; this scores the thing the model is for --
the heart-rate plot it produces. Beats are picked out of the predicted activity and out of the
target comb *by the same detector the inference pipeline uses*, each beat train becomes an
instantaneous-BPM trace, and the two traces are compared with a plain Pearson r.

Going through the real inference path matters, and not only for fidelity. A model emits one
frame per ``hop_length`` samples, so picking peaks straight off that grid quantises every beat
to a frame. At hop 256 (64 ms) a fetal IBI of ~0.35 s is 5 or 6 whole frames and nothing in
between, so ``60/ibi`` can only land on 187.5 or 156.2 bpm -- a 31 bpm step, against real
beat-to-beat variability of a few bpm. The BPM trace becomes rounding pattern rather than
physiology, and the correlation measures noise. Inference avoids this: ``frames_to_native``
lifts the activity onto the input's own sample grid and the detector's Shannon-energy/Hilbert
envelope is *nonlinear*, so a lobe midpoint lands between frames. Beats come back at sample
resolution and the quantisation step disappears.

Two deliberate departures from ``analyze.evaluate_v3``, which scores whole recordings:

* **No lag search.** Validation snippets are time-aligned with their targets by construction,
  so the correlation is taken at zero lag. (evaluate_v3 searches +/- 5 s because a real
  recording's SOT has an unknown offset -- and on a short snippet that search would slide the
  prediction almost entirely off the reference.)
* **No beat matching.** Nothing here measures whether individual beats land within a tolerance;
  only whether the two BPM traces move together.

Caveat worth remembering when reading the number: over a window as short as a training crop
there is little slow accel/decel for two traces to agree *about*, so most of the signal is
beat-to-beat interval agreement. Pearson r over ~15 intervals is inherently high-variance;
compare trials using the spread, not the mean alone.
"""

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch

from common.phases.inference import frames_to_native

#: Plausible fetal range, mirroring analyze.constants.FETAL_BPM_RANGE. Duplicated rather than
#: imported: common is a model library and must not depend on the analysis stack.
FETAL_BPM_RANGE = (90.0, 280.0)

#: ``(activity, hz) -> beat times in seconds``. Supplied by the task so that the detector is
#: literally the one inference runs, without common importing the analysis package.
BeatDetector = Callable[[np.ndarray, float], np.ndarray]


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


def trace_correlation(
        pred_beats: np.ndarray,
        ref_beats: np.ndarray,
        bpm_range: Tuple[float, float] = FETAL_BPM_RANGE,
        grid_hz: float = 10.0,
        min_span_s: float = 1.0,
) -> Optional[float]:
    """Pearson r between the BPM traces of two beat trains, at zero lag.

    The traces carry one sample per detected interval, so they are neither the same length nor
    on the same timestamps; both are resampled onto a uniform grid over the span they both
    cover before correlating. Returns ``None`` when no meaningful correlation exists -- too few
    beats on either side, too little overlap, or a trace with no variance (a perfectly constant
    rate correlates with nothing). Callers decide what a degenerate snippet scores.
    """
    tp, bp = bpm_trace(pred_beats, bpm_range)
    tr, br = bpm_trace(ref_beats, bpm_range)
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
    """Accumulates :func:`trace_correlation` over a validation split.

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
            detect: BeatDetector,
            hop_length: int,
            sample_rate: int,
            bpm_range: Tuple[float, float] = FETAL_BPM_RANGE,
            postprocess: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
            reference_beats: Optional[dict] = None,
    ):
        self.detect = detect
        self.hop_length = hop_length
        self.sample_rate = sample_rate
        self.bpm_range = bpm_range
        # Maps raw model output to the non-negative envelope the beat detector expects, exactly
        # as inference does (common.phases.inference.activity_postprocess): a log-prob head has
        # to be exp'd before its peaks mean anything.
        self.postprocess = postprocess
        # Targets are identical every epoch (the validation loader is deterministic and
        # un-augmented), so their beats are detected once and reused. Keyed by the target's own
        # bytes, which makes the cache correct regardless of batch order. Pass a dict owned by
        # the scorer *factory* to share it across epochs; None disables caching.
        self.reference_beats = reference_beats
        self._values: List[float] = []
        self._degenerate = 0

    def _beats(self, frames: np.ndarray) -> np.ndarray:
        """Frame-rate activity -> beat times, along the inference path."""
        native = frames_to_native(
            frames,
            hop_length=self.hop_length,
            model_hz=self.sample_rate,
            n_native=frames.size * self.hop_length,
            src_hz=self.sample_rate,
        )
        return self.detect(native, float(self.sample_rate))

    def _ref_beats(self, target: np.ndarray) -> np.ndarray:
        if self.reference_beats is None:
            return self._beats(target)
        key = target.tobytes()
        cached = self.reference_beats.get(key)
        if cached is None:
            cached = self._beats(target)
            self.reference_beats[key] = cached
        return cached

    def update(self, output: torch.Tensor, target: torch.Tensor) -> None:
        """Score every item in a ``(batch, frames)`` output/target pair."""
        if self.postprocess is not None:
            output = self.postprocess(output)
        out = output.detach().float().cpu().numpy()
        tgt = target.detach().float().cpu().numpy()

        for pred_i, target_i in zip(out, tgt):
            value = trace_correlation(
                self._beats(pred_i), self._ref_beats(target_i), self.bpm_range)
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
