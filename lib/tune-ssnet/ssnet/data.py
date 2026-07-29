import math

import torch
from torch.utils.data import DataLoader, Dataset

from common.audio import SAMPLE_RATE, crop_time, holdout_split, load_wav, snippet_indices
from common.augment import Augmenter


class PreprocessedPairs(Dataset):
    """Paired snippet dataset: returns (mix, target).

    mix    -> (1, crop_samples)
    target -> (num_sources, crop_samples): [heart, lung] or [heart, lung, noise]

    mix and all sources are cropped with a single shared offset so they stay time-aligned,
    and to a fixed length so the default collate can stack them. If a clip is shorter than
    crop_samples it is zero-padded.
    """

    def __init__(self, snippet_dir, indices, num_sources, crop_samples, train, augment=()):
        self.dir = snippet_dir
        self.indices = indices
        self.num_sources = num_sources
        self.crop_samples = crop_samples
        self.train = train  # train => random crop offset + augmentation; eval => deterministic
        self.augmenter = Augmenter(augment)

    def __len__(self):
        return len(self.indices)

    def _load(self, name):
        return load_wav(f"{self.dir}/{name}")

    def __getitem__(self, i):
        idx = self.indices[i]
        mix = self._load(f"{idx}_mix.wav")
        sources = [self._load(f"{idx}_heart.wav"), self._load(f"{idx}_lung.wav")]
        if self.num_sources == 3:
            sources.append(self._load(f"{idx}_noise.wav"))

        mix, *sources = crop_time([mix, *sources], self.crop_samples, random_offset=self.train)
        mix = self.augmenter(mix)

        target = torch.cat(sources, dim=0)  # (num_sources, crop_samples)
        return mix, target


def make_dataloader(config, *, train: bool) -> DataLoader:
    """Build a loader over the train/validation half of ``config.data.train_dir``.

    ssnet's snippets ship as a single directory, so validation is the tail fraction by index
    (config.data.val_fraction). Both loaders derive their split from the same deterministic
    sorted index list, so they never overlap.
    """
    snippet_dir = config.data.train_dir
    crop_samples = config.train.crop_len * SAMPLE_RATE

    indices = snippet_indices(snippet_dir)
    train_idx, val_idx = holdout_split(indices, config.data.val_fraction)
    chosen = train_idx if train else val_idx

    if train and math.ceil(len(train_idx) / config.train.batch_size) < 2:
        print(f"WARNING: only {len(train_idx)} training snippets "
              f"({math.ceil(len(train_idx)/config.train.batch_size)} batches); "
              f"add more snippets or lower batch_size.")

    ds = PreprocessedPairs(snippet_dir, chosen, config.model.num_sources, crop_samples,
                           train=train, augment=config.train.augment if train else ())

    print(f"Loaded {len(indices)} snippets from {snippet_dir} -> "
          f"{len(chosen)} {'train' if train else 'validation'}")
    return DataLoader(ds, batch_size=config.train.batch_size, shuffle=train,
                      num_workers=config.data.num_workers, pin_memory=True)
