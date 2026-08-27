"""PALNet's front-end and paired dataset: FUNet's spectrogram, deliberately unchanged.

This is a near-copy of ``funet.data``, and the sameness is the point. PALNet exists to test one
thing -- whether a frozen AudioSet trunk beats a U-Net learned from ~350 snippets -- and that
question is only answerable if both models are handed the identical tensor. Every difference
between the two front-ends would be a difference the comparison could not attribute.

What PALNet's earlier front-end did instead, and why it was abandoned: it used the STFT and mel
filterbank stored inside the AudioSet checkpoint. Those resolved the 100-300 Hz fetal band with
~16 of 64 perceptually-spaced mel bins and spent the other 48 on frequencies a fetus does not
emit. A linear probe over it reached train 0.0845 and a 1.6M-parameter head 0.0790, against
FUNet's 0.041. Neither stored tensor was ever *learned* -- one is a windowed DFT basis, the
other a triangular filterbank -- so replacing them costs the transfer bet nothing.

Two differences from ``funet.data`` remain, both forced by the backbone:

* **Rows are floored to a multiple of 32**, not ``2**depth``. The trunk halves frequency five
  times (see ``palnet.panns``), and flooring makes each halving exact instead of letting
  ``avg_pool2d`` silently drop an odd row at every stage. FUNet's shipped passband,
  ``[80, 350]`` at n_fft 1024, yields 71 rows and floors to 64 -- which is why that band is the
  default here too.
* **Time is not floored at all.** The trunk's pools are frequency-only, so there is no
  requirement on the frame count and the model emits one value per spectrogram frame.

The target is built by pooling the heart comb into ``hop_length``-sized bins, on the same frame
grid the spectrogram produces, rather than resampling it: a comb put through an anti-alias
lowpass rings and smears, and beat *timing* is the label.
"""

import math
from typing import Optional

import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset

from common.audio import SAMPLE_RATE, crop_time, holdout_split, load_wav, pad_time, snippet_indices
from common.augment import Augmenter
from common.preprocess import Preprocessor

from palnet.panns import FREQ_DOWNSAMPLE


def freq_crop_bins(config) -> Optional[tuple[int, int]]:
    """Half-open ``[lo, hi)`` spectrogram row range for ``config.model.freq_crop_hz``.

    None when no band is configured, which keeps the full height.

    Row k of a onesided STFT is centred at ``k * SAMPLE_RATE / n_fft`` Hz. The low edge floors
    and the high edge ceils, so the kept rows always *bracket* the requested band rather than
    landing inside it -- no row whose centre falls in [low, high] is ever dropped, and the
    bandpass's transition skirt keeps a row of margin on each side.

    One definition, called by the dataset, by inference and by ``spectrogram_shape``, so
    training and inference cannot drift onto different row ranges (which would silently feed a
    checkpoint a band it never saw).
    """
    band = config.model.freq_crop_hz
    if band is None:
        return None

    if len(band) != 2:
        raise ValueError(f"model.freq_crop_hz must be [low_hz, high_hz], got {band!r}")
    low, high = float(band[0]), float(band[1])
    if not 0 <= low < high:
        raise ValueError(
            f"model.freq_crop_hz must satisfy 0 <= low < high, got [{low}, {high}]")

    n_bins = config.model.n_fft // 2 + 1
    bin_hz = SAMPLE_RATE / config.model.n_fft
    lo = int(math.floor(low / bin_hz))
    hi = min(int(math.ceil(high / bin_hz)) + 1, n_bins)   # +1: half-open, so `high` is kept
    if lo >= hi:
        raise ValueError(
            f"model.freq_crop_hz [{low}, {high}] keeps no spectrogram rows at n_fft="
            f"{config.model.n_fft} ({bin_hz:.3f} Hz/bin, Nyquist {SAMPLE_RATE / 2:g} Hz)")
    return lo, hi


def spectrogram_shape(config) -> tuple[int, int]:
    """(freq_rows, frames) of the spectrogram PALNet is fed, *before* the row floor.

    ``freq_rows`` is ``n_fft // 2 + 1``, or the rows kept by ``freq_crop_hz``; ``frames`` is
    ``1 + crop_samples // hop_length``, matching torchaudio's default (onesided, center=True)
    Spectrogram. Pre-floor because that is what a feasibility check wants -- a band too narrow
    for the trunk's five halvings floors to 0, and the tuner should be able to reject it
    without building anything.
    """
    freq_rows = config.model.n_fft // 2 + 1
    crop = freq_crop_bins(config)
    if crop is not None:
        freq_rows = crop[1] - crop[0]
    frames = 1 + config.train.crop_len * SAMPLE_RATE // config.model.hop_length
    return freq_rows, frames


def model_rows(config) -> int:
    """Rows the model actually sees: ``spectrogram_shape`` floored to a whole number of
    halvings. This is the width ``PALNet.input_norm`` is built for."""
    rows, _ = spectrogram_shape(config)
    return rows - rows % FREQ_DOWNSAMPLE


class PALNetPairs(Dataset):
    """Paired snippet dataset in the shared training layout: ``{i}_mix.wav`` (multi-channel)
    plus mono ``{i}_heart.wav``.

    mix    -> (channels, freq, frames): log-power spectrogram, passband-cropped
    target -> (frames,): per-frame heart-beat activity, normalised to sum to 1

    mix and heart are cropped in the time domain with a single shared offset so they stay
    aligned, and to a fixed length so the frame count (and the default collate) is consistent
    across the batch. Short clips are zero-padded.
    """

    def __init__(self, snippet_dir: str, indices: list, crop_samples: int, train: bool,
                 n_fft: int, hop_length: int, freq_crop: Optional[tuple[int, int]] = None,
                 augment=(), preprocess=(), freq_mask: int = 0, time_mask: int = 0):
        self.dir = snippet_dir
        self.indices = indices
        self.crop_samples = crop_samples
        self.train = train  # train => random crop offset + augmentation; eval => deterministic
        self.hop_length = hop_length
        self.freq_crop = freq_crop
        self.augmenter = Augmenter(augment)
        # Unlike the augmenter, this is passed for validation too -- see common.preprocess.
        self.preprocessor = Preprocessor(preprocess)
        self.spectrogram = torchaudio.transforms.Spectrogram(n_fft=n_fft, hop_length=hop_length)
        # SpecAugment-style masking (train-only): one random-width band of freq bins / time
        # frames zeroed, width uniform in [0, param). 0 disables. Applied after log1p, where
        # 0 == log1p(0) == silence. Reachable at all only because PALNet owns its spectrogram
        # now -- the AudioSet checkpoint's own spec_augmenter sat inside a frozen, eval-pinned
        # module and could never fire.
        self.freq_masking = torchaudio.transforms.FrequencyMasking(freq_mask) if train and freq_mask > 0 else None
        self.time_masking = torchaudio.transforms.TimeMasking(time_mask) if train and time_mask > 0 else None

    def __len__(self):
        return len(self.indices)

    def _load(self, name: str) -> torch.Tensor:
        return load_wav(f"{self.dir}/{name}")

    def __getitem__(self, i: int):
        idx = self.indices[i]
        mix = self._load(f"{idx}_mix.wav")
        heart = self._load(f"{idx}_heart.wav")

        # Crop/pad to a fixed length (random offset when training) and layer on the enabled
        # input augmentations (train-only; an empty Augmenter is a no-op for validation).
        # Augment first, then preprocess: it band-limits the augmentation noise the same way
        # real in-band noise arrives, and leaves peak normalisation with the last word.
        mix, heart = crop_time([mix, heart], self.crop_samples, random_offset=self.train)
        mix = self.preprocessor(self.augmenter(mix))

        # Power spectrogram magnitudes span orders of magnitude; log1p compresses that to a
        # learnable range. Same transform FUNet applies, for the same reason.
        spectrogram = torch.log1p(self.spectrogram(mix))

        # Drop the rows the bandpass already emptied, before anything else looks at the axis.
        # Ahead of the SpecAugment masks on purpose: a freq_mask is a width in rows, so masking
        # first would spend part of its budget on rows about to be discarded.
        if self.freq_crop is not None:
            spectrogram = spectrogram[:, self.freq_crop[0]:self.freq_crop[1], :]

        # Masks go on the input only -- the target still expects beats inside a time-masked
        # span, deliberately: the model must interpolate through the gap from rhythm context.
        if self.freq_masking is not None:
            spectrogram = self.freq_masking(spectrogram)
        if self.time_masking is not None:
            spectrogram = self.time_masking(spectrogram)

        # Floor the frequency axis to a whole number of the trunk's five halvings. The time
        # axis is left alone: PALNet's pools are frequency-only, so any frame count works.
        freq = spectrogram.shape[-2]
        freq -= freq % FREQ_DOWNSAMPLE
        spectrogram = spectrogram[:, :freq, :]

        # Build the target on the SAME frame grid as the spectrogram so beats stay time-aligned
        # with the input. A torchaudio frame t sits at sample t*hop_length, so the frames span
        # the first frames*hop_length samples; pool the heart into those hop-sized bins.
        # clamp_min(0) drops the gated/negative lobes; normalising to sum 1 keeps the target on
        # the same footing as FUNet's and TSLNet's, which the shared losses assume.
        frames = spectrogram.shape[-1]
        covered = frames * self.hop_length
        heart_flat = pad_time(heart, covered)[0, :covered]           # (covered,)
        target = heart_flat.reshape(frames, self.hop_length).clamp_min(0).mean(dim=-1)
        target = target / (target.sum() + 1e-12)

        return spectrogram, target.float()


def make_dataloader(config, snippet_dir: str, *, train: bool) -> DataLoader:
    """Build a loader over ``snippet_dir``.

    Two split strategies, as in the other models: a separate ``val_dir`` holds out a whole
    patient (preferred -- no within-patient leakage), and ``val_fraction`` carves the tail off
    ``train_dir`` by index for datasets that ship as one directory. ``val_fraction`` is used
    only when it is set and no ``val_dir`` is given.
    """
    m = config.model

    indices = snippet_indices(snippet_dir)
    split_note = ""
    if not config.data.val_dir and config.data.val_fraction > 0:
        train_idx, val_idx = holdout_split(indices, config.data.val_fraction)
        chosen = train_idx if train else val_idx
        split_note = f" -> {len(chosen)}"
    else:
        chosen = indices

    ds = PALNetPairs(snippet_dir, chosen, config.train.crop_len * SAMPLE_RATE, train=train,
                     n_fft=m.n_fft, hop_length=m.hop_length, freq_crop=freq_crop_bins(config),
                     augment=config.train.augment if train else (),
                     preprocess=config.data.preprocess,   # every split, not just train
                     freq_mask=config.train.freq_mask, time_mask=config.train.time_mask)

    print(f"Loaded {len(indices)} snippets from {snippet_dir}{split_note} "
          f"({'train' if train else 'validation'})")
    return DataLoader(ds, batch_size=config.train.batch_size, shuffle=train,
                      num_workers=config.data.num_workers, pin_memory=True)
