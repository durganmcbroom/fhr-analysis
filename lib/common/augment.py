"""On-the-fly (online) waveform augmentation.

Applied per-sample inside a Dataset's ``__getitem__`` and re-sampled every epoch, so the
model sees a different version of each snippet each time -- the anti-memorization property
that fights small-patient-count overfitting. The dataset size is unchanged; the
augmentations are layered on at load time.

All of these alter only the mix waveform of shape ``(channels, time)``; targets are never
touched. Which ones run is a list of names (``config.train.augment``); they are applied in a
fixed sensible order regardless of list order, and only for the training loader (the
validation loader passes an empty list -> no-op).

Randomness uses torch's RNG so DataLoader worker seeding is handled correctly.

For deterministic transforms that must apply to every split and to inference -- bandpass,
peak normalisation -- see ``common.preprocess``. Putting one of those in this registry would
silently make it train-only, which is a distribution mismatch rather than regularisation.
"""

from typing import Iterable

import torch

# Default strengths, kept module-level so the config just toggles which augs are on.
CHANNEL_DROPOUT_P = 0.3     # probability of dropping each channel (never drops all)
NOISE_STD = 0.05            # additive gaussian noise std, as a fraction of the mix RMS
GAIN_DB = 6.0               # per-channel gain jitter range, +/- this many dB


def channel_dropout(mix: torch.Tensor, p: float = CHANNEL_DROPOUT_P) -> torch.Tensor:
    """Randomly zero whole fiber channels (never all of them). Forces the model to read beats
    from any fiber rather than a patient-specific one -- fiber placement differs between
    patients, which is a main cause of cross-patient failure."""
    c = mix.shape[0]
    if c <= 1:
        return mix
    keep = torch.rand(c) >= p
    if not keep.any():                       # never drop every channel
        keep[torch.randint(c, (1,))] = True
    return mix * keep.to(mix.dtype).unsqueeze(-1)


def gain_jitter(mix: torch.Tensor, gain_db: float = GAIN_DB) -> torch.Tensor:
    """Scale each channel by an independent random gain in +/- gain_db decibels. Varies the
    relative fiber balance -> robustness to per-fiber coupling/loudness differences between
    patients."""
    db = (torch.rand(mix.shape[0]) * 2 - 1) * gain_db
    gain = (10.0 ** (db / 20.0)).to(mix.dtype).unsqueeze(-1)
    return mix * gain


def additive_noise(mix: torch.Tensor, std: float = NOISE_STD) -> torch.Tensor:
    """Add gaussian noise scaled to the mix RMS (signal-proportional, so the effective SNR is
    consistent regardless of gain) -> robustness to per-recording noise."""
    rms = mix.pow(2).mean().sqrt()
    return mix + torch.randn_like(mix) * (std * rms)


# name -> function, for the toggle list in config.train.augment
AUGMENTATIONS = {
    "channel_dropout": channel_dropout,
    "gain": gain_jitter,
    "noise": additive_noise,
}
# Applied in this order (dropout -> gain -> noise) regardless of how the list is written, so
# the config list is a set of on/off toggles, not an ordering.
_ORDER = ["channel_dropout", "gain", "noise"]


class Augmenter:
    """Applies the enabled augmentations (by name) to a mix waveform. An empty list is a
    no-op (used for the validation loader); unknown names raise."""

    def __init__(self, enabled: Iterable[str] = ()):
        enabled = list(enabled)
        unknown = [n for n in enabled if n not in AUGMENTATIONS]
        if unknown:
            raise ValueError(f"unknown augmentation(s) {unknown}; valid: {list(AUGMENTATIONS)}")
        self.enabled = [n for n in _ORDER if n in enabled]

    def __call__(self, mix: torch.Tensor) -> torch.Tensor:
        for name in self.enabled:
            mix = AUGMENTATIONS[name](mix)
        return mix
