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

**Why the headline metric is agreement, not correlation.** Over a window as short as a
training crop the reference rate barely moves -- a real trace runs ~152 bpm with a standard
deviation near 3.5. Pearson divides by that standard deviation, so it does not ask "is this
the right heart rate", it asks "do the small wiggles around 152 wiggle together", and those
wiggles are detector jitter. Worse, one mistimed beat throws a ~95 bpm excursion, tens of
standard deviations wide, which then dominates the covariance: measured on a flat reference, a
model tracking tightly with a single bad beat scores r = +0.04 while a uniformly sloppy model
scores +0.17, and deleting that one beat of sixteen moves r from +0.04 to +0.84. So r is
decided by the worst one or two beats and is blind to the other fifteen.

Agreement asks the question the plot actually poses -- is the predicted rate the right number
-- and needs no variance to do it. ``median_delta`` is the reading (in bpm) and ``within_tol``
the score. Pearson stays available in ``corr`` because earlier runs were scored with it, and
because on a multi-minute recording, where the rate really does accelerate and decelerate, it
is the right statistic and is what ``analyze.evaluate_v3`` uses.
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


#: A predicted rate this far from the reference still counts as "the plot is right here".
#: 10 bpm is a few percent of a fetal rate -- tight enough that a mistimed beat fails it,
#: loose enough that frame quantisation alone does not.
TOLERANCE_BPM = 10.0


@dataclass
class SnippetHR:
    """The three ways one snippet's predicted BPM trace can be compared to the reference."""

    within_tol: float          #: fraction of the trace within TOLERANCE_BPM -- higher is better
    median_delta: float        #: median |predicted - reference| in bpm -- lower is better
    corr: Optional[float]      #: Pearson r, or None when either trace is perfectly flat


@dataclass
class HRScore:
    """Result of scoring a whole validation split.

    ``within_tol`` is the selection metric. ``median_delta`` is the number to read: it is in
    bpm, so "4.2" means the typical point of the predicted heart-rate plot sits 4.2 bpm from
    the truth. ``corr`` is kept because it was the original metric, but it is close to
    meaningless on a single snippet -- see the module docstring.
    """

    within_tol: float          #: mean per-snippet fraction within tolerance; the score
    median_delta: float        #: median per-snippet median |delta bpm|; nan if all degenerate
    corr: float                #: mean per-snippet Pearson r, for continuity with older runs
    n: int                     #: snippets scored
    n_degenerate: int          #: snippets with no usable trace, scored within_tol = 0
    tolerance_bpm: float = TOLERANCE_BPM

    def __str__(self) -> str:
        degenerate = f", {self.n_degenerate}/{self.n} degenerate" if self.n_degenerate else ""
        delta = "n/a" if np.isnan(self.median_delta) else f"{self.median_delta:.1f}bpm"
        return (f"within{self.tolerance_bpm:.0f} {self.within_tol:.3f}, "
                f"median|d| {delta}, r {self.corr:+.3f}{degenerate}")


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


def aligned_traces(
        pred_beats: np.ndarray,
        ref_beats: np.ndarray,
        bpm_range: Tuple[float, float] = FETAL_BPM_RANGE,
        grid_hz: float = 10.0,
        min_span_s: float = 1.0,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """The two BPM traces resampled onto one shared time grid, or None if they cannot be
    compared (too few beats on either side, or too little overlap).

    A beat train yields one BPM sample per interval, at that interval's midpoint, so the two
    traces are neither the same length nor on the same timestamps; both are interpolated onto a
    uniform grid spanning the range they both cover.
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
    return np.interp(grid, tp, bp), np.interp(grid, tr, br)


def snippet_hr(
        pred_beats: np.ndarray,
        ref_beats: np.ndarray,
        bpm_range: Tuple[float, float] = FETAL_BPM_RANGE,
        tolerance_bpm: float = TOLERANCE_BPM,
        grid_hz: float = 10.0,
        min_span_s: float = 1.0,
) -> Optional[SnippetHR]:
    """Compare one snippet's predicted and reference BPM traces. None if incomparable.

    Agreement (``within_tol``, ``median_delta``) is the useful part; ``corr`` is reported for
    continuity but is unreliable here for the reason in the module docstring: it is None
    whenever a trace is perfectly flat, and near-meaningless when it is nearly flat.
    """
    traces = aligned_traces(pred_beats, ref_beats, bpm_range, grid_hz, min_span_s)
    if traces is None:
        return None
    pred, ref = traces

    delta = np.abs(pred - ref)
    corr: Optional[float] = None
    if pred.std() > 0 and ref.std() > 0:
        value = float(np.corrcoef(pred, ref)[0, 1])
        corr = value if np.isfinite(value) else None

    return SnippetHR(within_tol=float((delta <= tolerance_bpm).mean()),
                     median_delta=float(np.median(delta)),
                     corr=corr)


def trace_correlation(
        pred_beats: np.ndarray,
        ref_beats: np.ndarray,
        bpm_range: Tuple[float, float] = FETAL_BPM_RANGE,
        grid_hz: float = 10.0,
        min_span_s: float = 1.0,
) -> Optional[float]:
    """Pearson r alone, for callers that only want the legacy number."""
    result = snippet_hr(pred_beats, ref_beats, bpm_range,
                        grid_hz=grid_hz, min_span_s=min_span_s)
    return None if result is None else result.corr


class HRMetrics:
    """Accumulates :func:`snippet_hr` over a validation split.

    One instance scores one pass over the split: construct, ``update`` per batch, read
    ``result``. A degenerate snippet scores ``within_tol = 0`` rather than being dropped --
    dropping it would score a model only where it happened to fire, which rewards a model that
    emits almost nothing.

    Aggregation differs per metric on purpose. ``within_tol`` is a mean, because it is already
    a bounded per-snippet fraction. ``median_delta`` is a median of medians, so one snippet
    where beat detection collapsed cannot drag the headline number. ``corr`` is a plain mean;
    Fisher-z averaging is the textbook correction, but at ~15 intervals per snippet the raw
    bias is ~0.01 and the z-transform overcorrects, since an r near +/-1 sends arctanh to
    infinity.
    """

    def __init__(
            self,
            detect: BeatDetector,
            hop_length: int,
            sample_rate: int,
            bpm_range: Tuple[float, float] = FETAL_BPM_RANGE,
            postprocess: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
            reference_beats: Optional[dict] = None,
            interpolation: str = "linear",
            tolerance_bpm: float = TOLERANCE_BPM,
            target_postprocess: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ):
        self.detect = detect
        self.hop_length = hop_length
        self.sample_rate = sample_rate
        self.bpm_range = bpm_range
        # Must match what inference uses, or the score describes a readout nobody deploys.
        self.interpolation = interpolation
        # Maps raw model output to the envelope the beat detector expects, exactly as inference
        # does (common.phases.inference.activity_postprocess): a log-prob head has to be exp'd
        # before its peaks mean anything.
        self.postprocess = postprocess
        # The same for the target, needed when a task's target is not already a single
        # per-frame trace. A separation model emits and is trained against (batch, sources,
        # time), so both sides have to be narrowed to the source carrying the heartbeat before
        # either can become a beat train. None leaves the target as the loader produced it.
        self.target_postprocess = target_postprocess
        # Targets are identical every epoch (the validation loader is deterministic and
        # un-augmented), so their beats are detected once and reused. Keyed by the target's own
        # bytes, which makes the cache correct regardless of batch order. Pass a dict owned by
        # the scorer *factory* to share it across epochs; None disables caching.
        self.reference_beats = reference_beats
        self.tolerance_bpm = tolerance_bpm
        self._within: List[float] = []
        self._deltas: List[float] = []
        self._corrs: List[float] = []
        self._degenerate = 0

    def beats(self, frames: np.ndarray) -> np.ndarray:
        """Frame-rate activity -> beat times, along the inference path.

        Public so a diagnostic can reuse the exact path the score is built from instead of
        reimplementing it (see common.diagnostics)."""
        native = frames_to_native(
            frames,
            hop_length=self.hop_length,
            model_hz=self.sample_rate,
            n_native=frames.size * self.hop_length,
            src_hz=self.sample_rate,
            interpolation=self.interpolation,
        )
        return self.detect(native, float(self.sample_rate))

    def _ref_beats(self, target: np.ndarray) -> np.ndarray:
        if self.reference_beats is None:
            return self.beats(target)
        key = target.tobytes()
        cached = self.reference_beats.get(key)
        if cached is None:
            cached = self.beats(target)
            self.reference_beats[key] = cached
        return cached

    def update(self, output: torch.Tensor, target: torch.Tensor) -> None:
        """Score every item in an output/target pair.

        Both are ``(batch, frames)`` once the postprocess hooks have run; a task whose raw
        output carries more than that (a separation model's source axis, say) narrows it there.
        """
        if self.postprocess is not None:
            output = self.postprocess(output)
        if self.target_postprocess is not None:
            target = self.target_postprocess(target)
        out = output.detach().float().cpu().numpy()
        tgt = target.detach().float().cpu().numpy()

        for pred_i, target_i in zip(out, tgt):
            result = snippet_hr(self.beats(pred_i), self._ref_beats(target_i),
                                self.bpm_range, self.tolerance_bpm)
            if result is None:
                # No comparable trace at all: the model gets no credit, but there is no
                # meaningful bpm error to fold into the median, so only the count records it.
                self._degenerate += 1
                self._within.append(0.0)
                continue
            self._within.append(result.within_tol)
            self._deltas.append(result.median_delta)
            if result.corr is not None:
                self._corrs.append(result.corr)

    def result(self) -> HRScore:
        if not self._within:
            return HRScore(within_tol=0.0, median_delta=float("nan"), corr=0.0,
                           n=0, n_degenerate=0, tolerance_bpm=self.tolerance_bpm)
        return HRScore(
            within_tol=float(np.mean(self._within)),
            median_delta=float(np.median(self._deltas)) if self._deltas else float("nan"),
            corr=float(np.mean(self._corrs)) if self._corrs else 0.0,
            n=len(self._within),
            n_degenerate=self._degenerate,
            tolerance_bpm=self.tolerance_bpm,
        )
