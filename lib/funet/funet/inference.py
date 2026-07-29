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
from common.phases.inference import frames_to_native, load_model, run_windowed

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
    is_logprob = model.head == "logprob"

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

    # Match training preprocessing: peak-normalise, then resample to the model rate.
    peak = float(np.max(np.abs(x))) + 1e-12
    x = resample(x / peak, src_hz, SAMPLE_RATE)

    hop = config.model.hop_length
    divisor = 2 ** len(config.model.dilations)

    spec = torchaudio.transforms.Spectrogram(n_fft=config.model.n_fft, hop_length=hop)
    S = torch.log1p(spec(torch.from_numpy(x)))            # (channels, freq, frames)

    freq = S.shape[-2] - S.shape[-2] % divisor            # crop freq to a multiple of divisor
    S = S[:, :freq, :]

    # Window the time axis to a training-sized crop, rounded down to a multiple of divisor.
    window = max(divisor, ((config.train.crop_len * SAMPLE_RATE) // hop) // divisor * divisor)

    # Both heads become a per-window softmax activity envelope: logprob already applied
    # log_softmax in forward (exp -> softmax); the signal head (corr/snr) is affine-invariant
    # -- corr never pins a baseline or scale -- so softmax normalizes its arbitrary offset
    # away into a clean positive envelope. Full-size windows keep the normalization consistent
    # across them. NOTE: inference-only; training optimizes the raw signal-head output.
    def postprocess(out: torch.Tensor) -> torch.Tensor:
        return out.exp() if is_logprob else out.softmax(dim=-1)

    activity = run_windowed(model, S, window, device=device, postprocess=postprocess)

    return frames_to_native(activity, hop, SAMPLE_RATE, n_native, src_hz)
