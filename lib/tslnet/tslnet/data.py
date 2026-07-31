"""The waveform front-end and paired dataset for TSLNet.

TimesFM consumes a low-rate univariate series, so the 4 kHz fiber audio is decimated to
``model_hz`` and handed over as-is -- no spectrogram, no derived features.

The rate is set by Nyquist, not by convenience. Fetal heart sound runs to ~300 Hz, so the
model rate has to keep 600 Hz underneath it, plus headroom: an anti-alias filter is not a
brick wall, and at 600 Hz a 295 Hz component survives at 0.59 amplitude. 800 Hz is the lowest
*clean divisor of 4 kHz* that passes the whole band untouched (Nyquist 400), which also makes
decimation an exact 1/5 and lets the target pool into whole 5-sample bins.

At 800 Hz the 2048-step context holds 2.56 s -- about six beats, enough for the model to see
rhythm -- and one 32-step patch is 40 ms, under a tenth of a beat.

Note this front-end does no band-limiting of its own; decimation only removes content above
Nyquist. Everything below the fetal band (maternal sounds, motion) still arrives unless
``data.preprocess`` includes ``bandpass``, which is why the shipped config enables it and
``check_feasible`` warns when it is off.

The target is built by pooling the heart comb into the same ``SAMPLE_RATE // model_hz`` bins,
rather than resampling it: a comb put through an anti-alias lowpass rings and smears, and beat
*timing* is the label.
"""

import math

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from common.audio import (
    SAMPLE_RATE, crop_time, holdout_split, load_wav, pad_time, resample, snippet_indices,
)
from common.augment import Augmenter
from common.preprocess import Preprocessor


def decimation(model_hz: int) -> int:
    """Samples of 4 kHz audio per model step. Exact only when model_hz divides SAMPLE_RATE,
    which ``check_feasible`` enforces -- the target pooling below reshapes by this number."""
    return SAMPLE_RATE // model_hz


def crop_samples(config) -> int:
    """Length of one training crop in 4 kHz samples. crop_len is a float for TSLNet (see
    TSLNetTrainConfig) so a crop can line up exactly with the backbone's context: 2.56 s at
    800 Hz is 2048 steps, and rounding that to whole seconds would waste a fifth of it."""
    return int(round(config.train.crop_len * SAMPLE_RATE))


def model_steps(config) -> int:
    """Steps one crop becomes at the model rate, before the patch-length floor.

    ``TSLNetPairs`` then floors this to a multiple of patch_length; this is the pre-floor
    count, which is what a feasibility check wants -- a crop yielding fewer than one patch
    floors to 0. Lets the tuner reject an unusable crop_len/model_hz pair without downloading
    the backbone.
    """
    return crop_samples(config) // decimation(config.model.model_hz)


def to_model_rate(waveform: torch.Tensor, model_hz: int) -> torch.Tensor:
    """Decimate a ``(channels, samples)`` waveform from SAMPLE_RATE down to ``model_hz``.

    ``resample`` is polyphase (scipy ``resample_poly``), so the anti-alias filter comes with
    it; at 800 Hz that filter has nothing left to remove if the passband is already 100-300 Hz.
    """
    if model_hz == SAMPLE_RATE:
        return waveform
    decimated = resample(waveform.numpy(), SAMPLE_RATE, model_hz)
    return torch.from_numpy(np.ascontiguousarray(decimated, dtype=np.float32))


class TSLNetPairs(Dataset):
    """Paired snippet dataset in the shared training layout: ``{i}_mix.wav`` (multi-channel)
    plus mono ``{i}_heart.wav``.

    mix    -> (channels, steps): per-fiber waveform at the model rate
    target -> (steps,): per-step heart-beat activity, normalised to sum to 1

    mix and heart are cropped in the time domain with a single shared offset so they stay
    aligned, and to a fixed length so the step count (and the default collate) is consistent
    across the batch. Short clips are zero-padded.
    """

    def __init__(self, snippet_dir: str, indices: list, crop_length: int, train: bool,
                 model_hz: int, patch_length: int = 1, augment=(), preprocess=()):
        self.dir = snippet_dir
        self.indices = indices
        self.crop_length = crop_length
        self.train = train  # train => random crop offset + augmentation; eval => deterministic
        self.model_hz = model_hz
        self.decimation = decimation(model_hz)
        self.patch_length = patch_length
        self.augmenter = Augmenter(augment)
        # Unlike the augmenter, this is passed for validation too -- see common.preprocess.
        self.preprocessor = Preprocessor(preprocess)

    def __len__(self):
        return len(self.indices)

    def _load(self, name: str) -> torch.Tensor:
        return load_wav(f"{self.dir}/{name}")

    def __getitem__(self, i: int):
        idx = self.indices[i]
        mix = self._load(f"{idx}_mix.wav")
        heart = self._load(f"{idx}_heart.wav")

        # Crop/pad to a fixed length (random offset when training), then augment and
        # preprocess at 4 kHz -- the rate common.preprocess designs its bandpass for.
        mix, heart = crop_time([mix, heart], self.crop_length, random_offset=self.train)
        mix = self.preprocessor(self.augmenter(mix))

        series = to_model_rate(mix, self.model_hz)

        # Floor to a whole number of patches: the backbone views the series as
        # (batch, steps/patch_length, patch_length), so a partial trailing patch is not a
        # shape it can take at all (see TSLNet.forward).
        steps = series.shape[-1]
        steps -= steps % self.patch_length
        series = series[:, :steps]

        # Build the target on the SAME grid so beats stay aligned with the input. Step t covers
        # samples [t*decimation, (t+1)*decimation) of the 4 kHz heart comb; pool it into those
        # bins. clamp_min(0) drops the gated/negative lobes; normalising to sum 1 makes it a
        # valid KLDivLoss target for the log-softmax head.
        covered = steps * self.decimation
        heart_flat = pad_time(heart, covered)[0, :covered]           # (covered,)
        heart_target = heart_flat.reshape(steps, self.decimation).clamp_min(0).mean(dim=-1)
        heart_target = heart_target / (heart_target.sum() + 1e-12)

        return series.float(), heart_target.float()


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

    ds = TSLNetPairs(snippet_dir, chosen, crop_samples(config), train=train,
                     model_hz=m.model_hz, patch_length=m.patch_length,
                     augment=config.train.augment if train else (),
                     preprocess=config.data.preprocess)   # every split, not just train

    print(f"Loaded {len(indices)} snippets from {snippet_dir}{split_note} "
          f"({'train' if train else 'validation'})")
    return DataLoader(ds, batch_size=config.train.batch_size, shuffle=train,
                      num_workers=config.data.num_workers, pin_memory=True)
