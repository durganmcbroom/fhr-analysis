"""The waveform front-end and paired dataset for PALNet.

PALNet does not build a spectrogram: the backbone owns one, and its STFT basis and mel
filterbank are tensors in the published checkpoint. All this module does is put the waveform on
the rate the backbone should see it at, and build the target on the frame grid the backbone
will emit.

**Why the rate is a design decision.** The mel filterbank maps FFT *bin index* to mel bin, so a
bin's effective frequency is whatever the feeding rate claims. Resampling the 4 kHz snippets to
the checkpoint's nominal 32 kHz -- the "honest" mapping, where Hz means Hz -- is the worst
option available: AudioSet's mel scale spends its resolution on speech and music, and 100-300 Hz
lands on 5 of its 64 bins. Feeding at 8 kHz instead deliberately pitch-shifts the fetal band up
to a pretend 400-1200 Hz, where 16 bins cover it, and simultaneously makes the fixed 1024-tap
window 128 ms rather than 32 ms. 4 kHz (no resampling at all) covers the band with 22 bins but
stretches the window to 256 ms, over half the fastest plausible beat interval. See
``PALNetTask.check_feasible``, which reports all of this for whatever rate is configured.

**Why the hop looks absurd.** ``hop 8`` at 8 kHz is a 99.2%-overlap STFT. It is not a choice:
``n_fft`` is frozen at 1024 by the conv kernel, and the network reduces time by exactly 32, so
the input frame rate has to be 32x the output frame rate whatever the window length is. The
output is 32 ms frames, which is the number that matters.

**Why crops are aligned.** Every pooling stage floors, so a crop that is not a whole number of
output frames silently drops frames off the tail and slides the target out of alignment with
the input. ``crop_samples`` floors to the nearest aligned length and everything downstream
derives from it.

The target is built by pooling the heart comb into frame-sized bins at 4 kHz, rather than
resampling it: a comb put through an anti-alias lowpass rings and smears, and beat *timing* is
the label.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from common.audio import (
    SAMPLE_RATE, crop_time, holdout_split, load_wav, pad_time, resample, snippet_indices,
)
from common.augment import Augmenter
from common.preprocess import Preprocessor

from palnet.panns import TAPS


def time_downsample(config) -> int:
    """Spectrogram frames per output frame, fixed by where the features are tapped."""
    return TAPS[config.model.feature_layer][1]


def frame_stride(config) -> int:
    """Samples *at model_hz* per output frame."""
    return time_downsample(config) * config.model.hop


def native_stride(config) -> int:
    """Samples *at 4 kHz* per output frame -- the target's bin width, and the alignment unit.

    Exact only when ``model_hz`` divides ``frame_stride * SAMPLE_RATE``, which check_feasible
    enforces; the crop and the target both reshape by this number.
    """
    return frame_stride(config) * SAMPLE_RATE // config.model.model_hz


def crop_samples(config) -> int:
    """Length of one training crop in 4 kHz samples, floored to a whole number of frames."""
    n = int(round(config.train.crop_len * SAMPLE_RATE))
    return n - n % native_stride(config)


def model_frames(config) -> int:
    """Output frames one crop becomes.

    Zero for a crop shorter than one frame, which is what makes this usable as a feasibility
    check: the tuner can reject an unusable crop_len/hop/model_hz combination without
    downloading the backbone.
    """
    return crop_samples(config) // native_stride(config)


def to_model_rate(waveform: torch.Tensor, model_hz: int) -> torch.Tensor:
    """Resample a ``(channels, samples)`` waveform from SAMPLE_RATE to ``model_hz``.

    Usually an *up*sample (4 kHz -> 8 kHz), which is lossless -- nothing above 2 kHz exists in
    the source to alias -- and exact in length, since ``resample_poly`` at an integer ratio
    returns exactly ``L * up`` samples.
    """
    if model_hz == SAMPLE_RATE:
        return waveform
    resampled = resample(waveform.numpy(), SAMPLE_RATE, model_hz)
    return torch.from_numpy(np.ascontiguousarray(resampled, dtype=np.float32))


class PALNetPairs(Dataset):
    """Paired snippet dataset in the shared training layout: ``{i}_mix.wav`` (multi-channel)
    plus mono ``{i}_heart.wav``.

    mix    -> (channels, samples): per-fiber waveform at ``model_hz``
    target -> (frames,): per-frame heart-beat activity, normalised to sum to 1

    mix and heart are cropped in the time domain with a single shared offset so they stay
    aligned, and to a fixed aligned length so the frame count (and the default collate) is
    consistent across the batch. Short clips are zero-padded.
    """

    def __init__(self, snippet_dir: str, indices: list, crop_length: int, frames: int,
                 stride: int, train: bool, model_hz: int, augment=(), preprocess=()):
        self.dir = snippet_dir
        self.indices = indices
        self.crop_length = crop_length   # 4 kHz samples, already aligned
        self.frames = frames
        self.stride = stride             # 4 kHz samples per output frame
        self.train = train  # train => random crop offset + augmentation; eval => deterministic
        self.model_hz = model_hz
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

        # Crop/pad to the aligned length (random offset when training), then augment and
        # preprocess at 4 kHz -- the rate common.preprocess designs its bandpass for.
        mix, heart = crop_time([mix, heart], self.crop_length, random_offset=self.train)
        mix = self.preprocessor(self.augmenter(mix))

        series = to_model_rate(mix, self.model_hz)

        # Build the target on the SAME grid the backbone will emit on, so beats stay aligned
        # with the input. Frame t covers 4 kHz samples [t*stride, (t+1)*stride). clamp_min(0)
        # drops the gated/negative lobes; normalising to sum 1 keeps the target on the same
        # footing as FUNet's and TSLNet's, which the shared losses assume.
        covered = self.frames * self.stride
        heart_flat = pad_time(heart, covered)[0, :covered]           # (covered,)
        target = heart_flat.reshape(self.frames, self.stride).clamp_min(0).mean(dim=-1)
        target = target / (target.sum() + 1e-12)

        return series.float(), target.float()


def make_dataloader(config, snippet_dir: str, *, train: bool, shuffle=None) -> DataLoader:
    """Build a loader over ``snippet_dir``.

    Two split strategies, as in the other models: a separate ``val_dir`` holds out a whole
    patient (preferred -- no within-patient leakage), and ``val_fraction`` carves the tail off
    ``train_dir`` by index for datasets that ship as one directory. ``val_fraction`` is used
    only when it is set and no ``val_dir`` is given.

    ``shuffle`` defaults to ``train``. The BatchNorm recalibration pass overrides it: it wants
    the deterministic, un-augmented view of the *training* directory, which is ``train=False``
    over ``train_dir`` -- but with shuffling on, so a 32-batch sample is not just the first 32
    snippets in index order.
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

    ds = PALNetPairs(snippet_dir, chosen, crop_samples(config), model_frames(config),
                     native_stride(config), train=train, model_hz=m.model_hz,
                     augment=config.train.augment if train else (),
                     preprocess=config.data.preprocess)   # every split, not just train

    print(f"Loaded {len(indices)} snippets from {snippet_dir}{split_note} "
          f"({'train' if train else 'validation'})")
    return DataLoader(ds, batch_size=config.train.batch_size,
                      shuffle=train if shuffle is None else shuffle,
                      num_workers=config.data.num_workers, pin_memory=True)
