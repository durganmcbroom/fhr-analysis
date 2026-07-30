"""The envelope front-end and paired dataset for TSLNet.

TimesFM consumes a low-rate univariate series, so the 4 kHz fiber audio is reduced to a
per-fiber band-energy envelope first: STFT, sum power across the passband, square-root back to
amplitude. At the default hop of 16 that is a 250 Hz series in which a beat is a bump a few
frames wide -- a shape the backbone's pretraining corpus is full of, unlike raw acoustics.

The target is built exactly as FUNet builds it, on the same frame grid as the envelope, so the
two stay time-aligned and the same losses apply to both models.
"""

import math

import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset

from common.audio import SAMPLE_RATE, crop_time, holdout_split, load_wav, pad_time, snippet_indices
from common.augment import Augmenter


def envelope_frames(config) -> int:
    """Frames the envelope has for one training crop, before the patch-length floor.

    Matches torchaudio's default (onesided, center=True) Spectrogram: 1 + samples // hop.
    ``TSLNetPairs`` then floors this to a multiple of patch_length; this is the pre-floor
    count, which is what a feasibility check wants -- a crop yielding fewer than one patch
    floors to 0. Lets the tuner reject an unusable crop_len/hop_length pair without
    downloading the backbone.
    """
    return 1 + config.train.crop_len * SAMPLE_RATE // config.model.hop_length


def band_bins(n_fft: int, band, sample_rate: int = SAMPLE_RATE) -> tuple[int, int]:
    """Half-open ``[lo, hi)`` STFT bin range covering ``band`` (a ``[low_hz, high_hz]`` pair),
    or every bin when ``band`` is None.

    Bin k is centred at ``k * sample_rate / n_fft``. The range is widened outward (floor the
    low edge, ceil the high) so the requested band is fully covered rather than shaved.
    """
    n_bins = n_fft // 2 + 1
    if band is None:
        return 0, n_bins

    low, high = band
    if not 0 <= low < high:
        raise ValueError(f"band must be [low, high] with 0 <= low < high, got {band}")

    hz_per_bin = sample_rate / n_fft
    lo = max(0, math.floor(low / hz_per_bin))
    hi = min(n_bins, math.ceil(high / hz_per_bin) + 1)   # +1: the range is half-open
    if hi <= lo:
        raise ValueError(
            f"band {band} Hz selects no STFT bins at n_fft={n_fft} (bin width "
            f"{hz_per_bin:.1f} Hz, Nyquist {sample_rate // 2} Hz)")
    return lo, hi


class Envelope:
    """Waveform ``(channels, samples)`` -> band-energy envelope ``(channels, frames)``."""

    def __init__(self, n_fft: int, hop_length: int, band=None, log: bool = True):
        self.spectrogram = torchaudio.transforms.Spectrogram(n_fft=n_fft, hop_length=hop_length)
        self.lo, self.hi = band_bins(n_fft, band)
        self.log = log

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        power = self.spectrogram(waveform)[:, self.lo:self.hi, :]   # (channels, bins, frames)
        # Sum power (not magnitude) across the band, then sqrt: that is the band-limited
        # amplitude envelope by Parseval, and it keeps a beat's energy from being diluted by
        # however many bins happen to be in the passband.
        envelope = power.sum(dim=1).clamp_min(0).sqrt()
        return torch.log1p(envelope) if self.log else envelope


class TSLNetPairs(Dataset):
    """Paired snippet dataset in the shared training layout: ``{i}_mix.wav`` (multi-channel)
    plus mono ``{i}_heart.wav``.

    mix    -> (channels, frames): per-fiber band-energy envelope
    target -> (frames,): per-frame heart-beat activity, normalised to sum to 1

    mix and heart are cropped in the time domain with a single shared offset so they stay
    aligned, and to a fixed length so the frame count (and the default collate) is consistent
    across the batch. Short clips are zero-padded.
    """

    def __init__(self, snippet_dir: str, indices: list, crop_samples: int, train: bool,
                 n_fft: int, hop_length: int, band=None, log_envelope: bool = True,
                 patch_length: int = 1, augment=()):
        self.dir = snippet_dir
        self.indices = indices
        self.crop_samples = crop_samples
        self.train = train  # train => random crop offset + augmentation; eval => deterministic
        self.hop_length = hop_length
        self.patch_length = patch_length
        self.augmenter = Augmenter(augment)
        self.envelope = Envelope(n_fft, hop_length, band, log=log_envelope)

    def __len__(self):
        return len(self.indices)

    def _load(self, name: str) -> torch.Tensor:
        return load_wav(f"{self.dir}/{name}")

    def __getitem__(self, i: int):
        idx = self.indices[i]
        mix = self._load(f"{idx}_mix.wav")
        heart = self._load(f"{idx}_heart.wav")

        # Crop/pad to a fixed length (random offset when training) and layer on the enabled
        # input augmentations (train-only; empty Augmenter is a no-op for validation).
        mix, heart = crop_time([mix, heart], self.crop_samples, random_offset=self.train)
        mix = self.augmenter(mix)

        envelope = self.envelope(mix)

        # Floor the frame count to a whole number of patches: the backbone views the series as
        # (batch, frames/patch_length, patch_length), so a partial trailing patch is not a
        # shape it can take at all (see TSLNet.forward).
        frames = envelope.shape[-1]
        frames -= frames % self.patch_length
        envelope = envelope[:, :frames]

        # Build the target on the SAME frame grid so beats stay aligned with the input. A
        # torchaudio frame t sits at sample t*hop_length, so the kept frames span the first
        # frames*hop_length samples; pool the heart into those hop-sized bins. clamp_min(0)
        # drops the gated/negative lobes; normalising to sum 1 makes it a valid KLDivLoss
        # target for the log-softmax head.
        covered = frames * self.hop_length
        heart_flat = pad_time(heart, covered)[0, :covered]           # (covered,)
        heart_target = heart_flat.reshape(frames, self.hop_length).clamp_min(0).mean(dim=-1)
        heart_target = heart_target / (heart_target.sum() + 1e-12)

        return envelope.float(), heart_target.float()


def make_dataloader(config, snippet_dir: str, *, train: bool) -> DataLoader:
    """Build a loader over ``snippet_dir``.

    Two split strategies, as in the other models: a separate ``val_dir`` holds out a whole
    patient (preferred -- no within-patient leakage), and ``val_fraction`` carves the tail off
    ``train_dir`` by index for datasets that ship as one directory. ``val_fraction`` is used
    only when it is set and no ``val_dir`` is given.
    """
    m = config.model
    crop_samples = config.train.crop_len * SAMPLE_RATE

    indices = snippet_indices(snippet_dir)
    split_note = ""
    if not config.data.val_dir and config.data.val_fraction > 0:
        train_idx, val_idx = holdout_split(indices, config.data.val_fraction)
        chosen = train_idx if train else val_idx
        split_note = f" -> {len(chosen)}"
    else:
        chosen = indices

    ds = TSLNetPairs(snippet_dir, chosen, crop_samples, train=train,
                     n_fft=m.n_fft, hop_length=m.hop_length, band=m.band,
                     log_envelope=m.log_envelope, patch_length=m.patch_length,
                     augment=config.train.augment if train else ())

    print(f"Loaded {len(indices)} snippets from {snippet_dir}{split_note} "
          f"({'train' if train else 'validation'})")
    return DataLoader(ds, batch_size=config.train.batch_size, shuffle=train,
                      num_workers=config.data.num_workers, pin_memory=True)
