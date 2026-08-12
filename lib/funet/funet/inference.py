"""Run a trained FUNet on a raw waveform and get a beat-activity signal over time.

FUNet consumes a (channels, time) waveform -- the same abdomen fibers stacked as channels
that produced the training mix -- turns it into a log1p power spectrogram (exactly as
lib/funet/src/data.py does), and predicts a per-frame beat activity.

The model was trained on fixed crop_len-second crops, so inference processes the spectrogram
in equal frame windows (matching the training frame count, so GroupNorm sees the same spatial
extent it trained on) and stitches the per-window activity back together -- see
common.phases.inference.run_windowed. The frame-rate activity is then mapped onto the input's
own time axis so it lines up sample-for-sample with the source waveform.
"""

import numpy as np
import torch
import torchaudio

from common.audio import SAMPLE_RATE, resample
from common.phases.inference import (
    activity_postprocess, frames_to_native, load_model, normalize_blocks, run_windowed,
)
from common.preprocess import Preprocessor

from funet.data import freq_crop_bins
from funet.model import FUNet
from funet.task import FUNetTask


def load_funet(config, checkpoint: str, device: torch.device = None) -> FUNet:
    """Build a FUNet matching ``config`` and load weights from ``checkpoint``."""
    return load_model(FUNetTask(), config, checkpoint, device)


@torch.no_grad()
def run_funet(
        x: np.ndarray,
        src_hz: int,
        model: FUNet,
        config,
        device: torch.device = None,
) -> np.ndarray:
    """Beat-activity over time for waveform ``x`` (``(T,)`` or ``(channels, T)``).

    Returns a non-negative activity signal the same length as ``x`` and sampled at ``src_hz``:
    high where the model thinks a fetal beat occurs. For the log-prob head this is the exp'd
    probability; for the signal head it is the softmax'd output. Relative peaks (not absolute
    scale) are what carry the beats.
    """
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x[None, :]                      # (1, T)
    n_native = x.shape[-1]

    channels = x.shape[0]
    if channels != config.model.channels:
        raise ValueError(
            f"waveform has {channels} channel(s) but the model expects "
            f"{config.model.channels} (config.model.channels)"
        )

    # Match training preprocessing: resample to the model rate, then peak-normalise on the
    # same time scale a training snippet was normalised on -- not across the whole recording,
    # which lets one loud transient rescale everything else (see normalize_blocks).
    x = resample(x, src_hz, SAMPLE_RATE)
    x = normalize_blocks(x, config.train.crop_len * SAMPLE_RATE)

    hop = config.model.hop_length
    divisor = 2 ** len(config.model.dilations)

    # The same deterministic transforms the dataset applied, or the model meets an input
    # distribution it never trained on (see common.preprocess).
    waveform = Preprocessor(config.data.preprocess)(torch.from_numpy(np.ascontiguousarray(x)))

    spec = torchaudio.transforms.Spectrogram(n_fft=config.model.n_fft, hop_length=hop)
    S = torch.log1p(spec(waveform))                       # (channels, freq, frames)

    # Passband crop, from the same helper the dataset uses -- a checkpoint trained on a band
    # must be run on that band, and two copies of the arithmetic would eventually disagree.
    crop = freq_crop_bins(config)
    if crop is not None:
        S = S[:, crop[0]:crop[1], :]

    freq = S.shape[-2] - S.shape[-2] % divisor            # crop freq to a multiple of divisor
    S = S[:, :freq, :]

    # Window the time axis to a training-sized crop, rounded down to a multiple of divisor.
    window = max(divisor, ((config.train.crop_len * SAMPLE_RATE) // hop) // divisor * divisor)

    # How a window of raw output becomes activity depends on what the loss pinned down, so the
    # readout is chosen from the loss rather than fixed here -- see ACTIVITY_POSTPROCESS. This
    # matters for FUNet in the opposite direction to TSLNet: trained with 'mse' its output is
    # already calibrated to a unit-peak comb, and the old blanket softmax over ~100+ frames
    # flattened those calibrated peaks toward uniform -- a plausible contributor to the
    # weak-peak problem, which is now just a clamp.
    # NOTE: inference-only; training optimizes the raw signal-head output.
    postprocess = activity_postprocess(config.train.loss)

    activity = run_windowed(model, S, window, device=device, postprocess=postprocess)

    return frames_to_native(activity, hop, SAMPLE_RATE, n_native, src_hz,
                            interpolation=config.model.interpolation)
