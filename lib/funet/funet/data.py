import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset

from common.audio import SAMPLE_RATE, crop_time, load_wav, pad_time, snippet_indices
from common.augment import Augmenter
from common.preprocess import Preprocessor


def stft_output_shape(config) -> tuple[int, int]:
    """(freq_bins, time_frames) of the spectrogram FUNet sees for this config.

    freq_bins = n_fft // 2 + 1 and time_frames = 1 + crop_samples // hop_length, matching
    torchaudio's default (onesided, center=True) Spectrogram. __getitem__ then floors each
    axis to a multiple of 2**depth; these are the pre-floor counts, which is exactly what a
    feasibility check wants -- a level fits only if 2**depth <= the count (a smaller count
    floors to 0). Lets the tuner reject too-deep networks without building the model first.
    """
    freq_bins = config.model.n_fft // 2 + 1
    time_frames = 1 + config.train.crop_len * SAMPLE_RATE // config.model.hop_length
    return freq_bins, time_frames


class FetalPairs(Dataset):
    """Paired snippet dataset in the tune-ssnet .../training/.../fetal layout:
    {i}_mix.wav (mono or multi-channel) plus mono {i}_heart.wav, {i}_lung.wav,
    and optionally {i}_noise.wav.

    mix    -> (mix_channels, freq, frames)
    target -> (frames,): per-frame heart-beat activity, normalised to sum to 1

    mix and heart are cropped in the time domain with a single shared offset so they stay
    time-aligned, and to a fixed length so the spectrogram frame count (and the default
    collate) is consistent across the batch. If a clip is shorter than crop_samples it is
    zero-padded. Cropped waveforms are then converted to power spectrograms.
    """

    def __init__(self, snippet_dir: str, indices: list, crop_samples: int, train: bool,
                 n_fft: int, hop_length: int, divisor: int = 1, augment=(), preprocess=(),
                 freq_mask: int = 0, time_mask: int = 0):
        self.dir = snippet_dir
        self.indices = indices
        self.crop_samples = crop_samples
        self.train = train  # train => random crop offset + augmentation; eval => deterministic
        self.hop_length = hop_length
        self.divisor = divisor  # FUNet needs freq/time both divisible by 2**depth
        self.augmenter = Augmenter(augment)
        # Unlike the augmenter, this is passed for validation too -- see common.preprocess.
        self.preprocessor = Preprocessor(preprocess)
        self.spectrogram = torchaudio.transforms.Spectrogram(n_fft=n_fft, hop_length=hop_length)
        # SpecAugment-style masking (train-only): each sample gets one random-width band of
        # freq bins / time frames zeroed, width uniform in [0, param). 0 disables. Applied
        # after log1p, where 0 == log1p(0) == silence.
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
        # input augmentations (train-only; empty Augmenter is a no-op for validation).
        mix, heart = crop_time([mix, heart], self.crop_samples, random_offset=self.train)
        # Augment first, then preprocess: it band-limits the augmentation noise the same way
        # real in-band noise arrives, and leaves peak normalisation with the last word.
        mix = self.preprocessor(self.augmenter(mix))

        # Power spectrogram magnitudes span orders of magnitude (raw audio, unnormalized);
        # log1p compresses that to a learnable range before it hits the first conv.
        spectrogram = torch.log1p(self.spectrogram(mix))

        # SpecAugment masks go on the input only -- the target still expects beats inside a
        # time-masked span, deliberately: the model must interpolate through the gap from
        # rhythm context instead of memorizing band- or frame-specific cues.
        if self.freq_masking is not None:
            spectrogram = self.freq_masking(spectrogram)
        if self.time_masking is not None:
            spectrogram = self.time_masking(spectrogram)

        # Crop freq/time down to a multiple of `divisor` so FUNet's pool/transpose-conv
        # levels exactly invert each other (see the ValueError raised in FUNet.forward).
        freq, time = spectrogram.shape[-2], spectrogram.shape[-1]
        freq -= freq % self.divisor
        time -= time % self.divisor
        spectrogram = spectrogram[:, :freq, :time]

        # Build the target on the SAME frame grid as the spectrogram so beats stay
        # time-aligned with the input. A torchaudio frame t sits at sample t*hop_length,
        # so the `time` kept frames span the first time*hop_length samples; pool the heart
        # into those hop-sized bins. clamp_min(0) drops the gated/negative lobes; normalize
        # to a distribution (sums to 1) so it's a valid KLDivLoss target for the model's
        # log-softmax output.
        covered = time * self.hop_length
        heart_flat = pad_time(heart, covered)[0, :covered]      # (covered,)
        heart_target = heart_flat.reshape(time, self.hop_length).clamp_min(0).mean(dim=-1)
        heart_target = heart_target / (heart_target.sum() + 1e-12)

        return spectrogram, heart_target.float()


def make_dataloader(config, snippet_dir: str, *, train: bool) -> DataLoader:
    """Build a loader over ``snippet_dir``.

    Train and validation come from separate directories -- a whole patient is held out for
    validation (see generate_training_snippets.py per-patient mode), so there's no
    within-patient leakage between them. Train shuffles, random-crops and augments;
    validation is deterministic.
    """
    crop_samples = config.train.crop_len * SAMPLE_RATE
    divisor = 2 ** len(config.model.dilations)

    indices = snippet_indices(snippet_dir)
    ds = FetalPairs(snippet_dir, indices, crop_samples, train=train,
                    n_fft=config.model.n_fft, hop_length=config.model.hop_length,
                    divisor=divisor,
                    augment=config.train.augment if train else (),
                    preprocess=config.data.preprocess,   # every split, not just train
                    freq_mask=config.train.freq_mask, time_mask=config.train.time_mask)

    print(f"Loaded {len(indices)} {'train' if train else 'validation'} snippets from {snippet_dir}")
    return DataLoader(ds, batch_size=config.train.batch_size, shuffle=train,
                      num_workers=config.data.num_workers, pin_memory=True)
