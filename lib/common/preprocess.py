"""Deterministic input preprocessing, shared by every model.

The contrast with ``common.augment`` is the whole reason this is a separate module. Those are
random and train-only: each epoch sees a different version of a snippet, and the validation
loader deliberately gets none of them. Everything here is the opposite -- fixed, and applied
to **every split and at inference**.

That distinction is not stylistic. These transforms change the input distribution, so:

* running them on train but not validation makes the two losses incomparable -- the val curve
  would be measuring a different kind of input than the one being fitted;
* running them on train but not inference runs the trained model on data it has never seen.

So they are wired in next to the augmenter in each dataset (after it, so that augmentation
noise is band-limited and the peak normalisation has the last word) and again in each model's
inference path. ``config.data.preprocess`` toggles them by name for both.

Like ``augment``, strengths live at module level and the config only picks names -- there is
one right passband for this signal, not a per-run tuning knob.

Note the snippets on disk are already globally peak-normalised by
``fhr_bin.generate_training_snippets.write_multichannel``. ``normalize`` here is still not a
no-op: a crop is a *window* of a snippet and does not inherit its peak, and gain/noise
augmentation moves the scale again before the model sees it.
"""

import functools
from typing import Iterable

import numpy as np
import torch
from scipy.signal import butter, sosfiltfilt

from common.audio import SAMPLE_RATE

# Fetal heart sound sits well inside this band; it is the same passband
# analyze.funet_runner already applies to fibers before running a model on them
# (`bp(100, 300, "butter")`), so training on it matches how the pipeline feeds it.
BANDPASS_HZ = (100.0, 300.0)
BANDPASS_ORDER = 3            # matches analyze.filters.bp_filter's default order


@functools.cache
def _bandpass_sos(low: float, high: float, order: int, sample_rate: int) -> np.ndarray:
    """Second-order sections for the passband. Cached: the coefficients depend only on
    constants, and re-deriving them per __getitem__ would run once per snippet per epoch."""
    return butter(order, [low, high], fs=sample_rate, btype="bandpass", output="sos")


def bandpass(mix: torch.Tensor, band=BANDPASS_HZ, order: int = BANDPASS_ORDER) -> torch.Tensor:
    """Zero-phase band-limit each channel of a ``(channels, time)`` waveform.

    ``sosfiltfilt`` runs the filter forwards and backwards, which squares the magnitude
    response but cancels the phase entirely. That matters more here than the extra
    attenuation: a phase shift would move beats in time, and beat *timing* is the label.
    Squaring also means the passband edges sit at -6 dB rather than -3, so 100 and 300 Hz are
    half-amplitude and the band is effectively ~120-260 Hz at full strength.

    Filtering a *crop* rather than the whole recording leaves a transient at each end. On real
    snippets it is unmeasurable against ordinary signal variation (edge RMS runs ~0.8x the
    interior), because a crop boundary is continuing signal, not the abrupt onset that makes
    filtfilt ring.
    """
    sos = _bandpass_sos(float(band[0]), float(band[1]), order, SAMPLE_RATE)

    # filtfilt needs enough samples either side to build its edge padding; a crop shorter than
    # that is not something to silently half-filter.
    padlen = 3 * (2 * len(sos) + 1)
    if mix.shape[-1] <= padlen:
        raise ValueError(
            f"bandpass needs more than {padlen} samples to filter without edge artefacts, got "
            f"{mix.shape[-1]}; raise crop_len or drop 'bandpass' from data.preprocess")

    filtered = sosfiltfilt(sos, mix.numpy(), axis=-1)
    # sosfiltfilt promotes to float64 and returns a reversed view; ascontiguousarray makes it
    # something torch can wrap without copying twice.
    return torch.from_numpy(np.ascontiguousarray(filtered, dtype=np.float32))


def peak_normalize(mix: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Scale a ``(channels, time)`` waveform so its largest absolute sample is 1.

    One peak across all channels, not one per channel: the relative loudness of the fibers is
    signal, not nuisance -- it is roughly where the sensor sat relative to the heart -- and
    per-channel scaling would flatten exactly that. A silent (all-zero) input passes through
    unchanged rather than being amplified into noise by a near-zero divisor.
    """
    peak = mix.abs().max()
    if peak < eps:
        return mix
    return mix / peak


# name -> function, for the toggle list in config.data.preprocess
PREPROCESSORS = {
    "bandpass": bandpass,
    "normalize": peak_normalize,
}
# Applied in this order regardless of how the list is written, so the config is a set of
# on/off toggles rather than an ordering. normalize goes last on purpose: filtering changes
# the peak, so normalising first would not leave the output in [-1, 1].
_ORDER = ["bandpass", "normalize"]


class Preprocessor:
    """Applies the enabled preprocessors (by name) to a mix waveform. An empty list is a no-op;
    unknown names raise. Mirrors ``common.augment.Augmenter`` so the two compose readably in a
    dataset's ``__getitem__``."""

    def __init__(self, enabled: Iterable[str] = ()):
        enabled = list(enabled)
        unknown = [n for n in enabled if n not in PREPROCESSORS]
        if unknown:
            raise ValueError(f"unknown preprocessor(s) {unknown}; valid: {list(PREPROCESSORS)}")
        self.enabled = [n for n in _ORDER if n in enabled]

    def __call__(self, mix: torch.Tensor) -> torch.Tensor:
        for name in self.enabled:
            mix = PREPROCESSORS[name](mix)
        return mix
