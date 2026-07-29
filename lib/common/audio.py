"""Snippet I/O and waveform helpers shared by the models in lib/.

The snippet layout is the one produced by tune-ssnet's generate_training_snippets: a
directory of ``{i}_mix.wav`` plus per-source ``{i}_heart.wav`` / ``{i}_lung.wav`` /
``{i}_noise.wav``, 4 kHz.
"""

from pathlib import Path
from typing import List, Sequence

import numpy as np
import torch
from scipy.io import wavfile
from scipy.signal import resample_poly

# Snippets are generated at 4 kHz; both models train at this rate.
SAMPLE_RATE = 4000


def load_wav(path) -> torch.Tensor:
    """Read a wav as a float32 ``(channels, time)`` tensor.

    scipy instead of torchaudio.load: avoids the torchcodec backend dependency. Integer PCM
    is normalised to [-1, 1]; mono becomes ``(1, T)``.
    """
    _, data = wavfile.read(str(path))
    if np.issubdtype(data.dtype, np.integer):
        data = data.astype(np.float32) / np.iinfo(data.dtype).max
    else:
        data = data.astype(np.float32)
    if data.ndim == 1:
        return torch.from_numpy(data).reshape(1, -1)          # mono -> (1, T)
    return torch.from_numpy(np.ascontiguousarray(data.T))     # (T, C) wav -> (C, T)


def pad_time(x: torch.Tensor, n: int) -> torch.Tensor:
    """Right-pad the time axis of a ``(channels, time)`` tensor with zeros up to length n."""
    if x.shape[-1] >= n:
        return x
    pad = torch.zeros(x.shape[0], n - x.shape[-1], dtype=x.dtype)
    return torch.cat([x, pad], dim=-1)


def crop_time(tensors: Sequence[torch.Tensor], crop_samples: int,
              random_offset: bool) -> List[torch.Tensor]:
    """Crop every tensor to ``crop_samples`` using a single shared offset, or zero-pad if
    shorter. The shared offset is what keeps the mix and its sources time-aligned; a random
    offset (training only) is itself an augmentation -- a different window each epoch.

    Snippets can differ by a sample, so the crop window is sized against the shortest input.
    """
    n = crop_samples
    avail = min(t.shape[-1] for t in tensors)
    if avail >= n:
        start = int(torch.randint(0, avail - n + 1, (1,))) if random_offset else 0
        sl = slice(start, start + n)
        return [t[:, sl] for t in tensors]
    return [pad_time(t, n) for t in tensors]


def snippet_indices(snippet_dir) -> List[int]:
    """Sorted snippet indices in ``snippet_dir``, discovered from its ``*_mix.wav`` files."""
    mix_files = sorted(Path(snippet_dir).glob("*_mix.wav"),
                       key=lambda p: int(p.stem.split("_")[0]))
    indices = [int(p.stem.split("_")[0]) for p in mix_files]
    if not indices:
        raise FileNotFoundError(f"No *_mix.wav snippets found in {Path(snippet_dir).absolute()}")
    return indices


def holdout_split(indices: Sequence[int], val_fraction: float):
    """Split ``indices`` into (train, validation), holding out the tail.

    The alternative strategy -- a separate validation directory -- is preferable when it is
    available, because a whole patient can be held out; this index split is for datasets that
    ship as a single directory, and it leaks within-patient information if a patient's
    snippets straddle the boundary.
    """
    if not 0 < val_fraction < 1:
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")
    n_val = max(1, int(len(indices) * val_fraction))
    train_idx, val_idx = list(indices[:-n_val]), list(indices[-n_val:])
    if not train_idx:
        raise ValueError(
            f"Only {len(indices)} snippet(s); not enough to hold out a validation set.")
    return train_idx, val_idx


def resample(x: np.ndarray, src_hz: int, target_hz: int) -> np.ndarray:
    """Polyphase-resample along the last axis. A no-op when the rates already match."""
    if src_hz == target_hz:
        return x
    g = np.gcd(int(target_hz), int(src_hz))
    return resample_poly(x, target_hz // g, src_hz // g, axis=-1)
