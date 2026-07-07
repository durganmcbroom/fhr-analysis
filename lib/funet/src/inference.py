"""Run a trained FUNet on a raw waveform and get a beat-activity signal over time.

FUNet consumes a (channels, time) waveform -- the same abdomen fibers stacked as
channels that produced the training mix -- turns it into a log1p power spectrogram
(exactly as lib/funet/src/data.py does), and predicts a per-frame beat activity.

The model was trained on fixed crop_len-second crops, so inference processes the
spectrogram in equal frame windows (matching the training frame count, so GroupNorm
sees the same spatial extent it trained on) and stitches the per-window activity
back together. The frame-rate activity is then mapped onto the input's own time
axis so it lines up sample-for-sample with the source waveform.
"""

import numpy as np
import torch
import torchaudio
from scipy.signal import resample_poly

from lib.funet.src.config import Config
from lib.funet.src.data import SAMPLE_RATE
from lib.funet.src.model import FUNet


def _head_for(config: Config) -> str:
    # Mirrors the LOSSES table in main.py: SNR uses a raw-signal head, else log-probs.
    return "signal" if config.train.loss == "snr" else "logprob"


def load_funet(config: Config, checkpoint: str, device: torch.device = None) -> FUNet:
    """Build a FUNet matching ``config`` and load weights from ``checkpoint``."""
    device = device or torch.device("cpu")
    model = FUNet(
        channels=config.model.channels,
        dilations=config.model.dilations,
        bottleneck_dilation=config.model.bottleneck_dilation,
        base_channels=config.model.base_channels,
        head=_head_for(config),
    )
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.to(device).eval()
    return model


def _resample(x: np.ndarray, src_hz: int, target_hz: int) -> np.ndarray:
    if src_hz == target_hz:
        return x
    g = np.gcd(int(target_hz), int(src_hz))
    return resample_poly(x, target_hz // g, src_hz // g, axis=-1)


@torch.no_grad()
def run_funet(
        x: np.ndarray,
        src_hz: int,
        model: FUNet,
        config: Config,
        device: torch.device = None,
) -> np.ndarray:
    """Beat-activity over time for waveform ``x`` (``(T,)`` or ``(channels, T)``).

    Returns a non-negative activity signal the same length as ``x`` and sampled at
    ``src_hz``: high where the model thinks a fetal beat occurs. For the log-prob
    head this is the exp'd probability; for the SNR head it is the raw signal
    clamped at 0. Relative peaks (not absolute scale) are what carry the beats.
    """
    device = device or next(model.parameters()).device
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
    x = _resample(x / peak, src_hz, SAMPLE_RATE)

    hop = config.data.hop_length
    divisor = 2 ** len(config.model.dilations)

    spec = torchaudio.transforms.Spectrogram(n_fft=config.data.n_fft, hop_length=hop)
    S = torch.log1p(spec(torch.from_numpy(x)))            # (channels, freq, frames)

    freq = S.shape[-2] - S.shape[-2] % divisor            # crop freq to a multiple of divisor
    S = S[:, :freq, :]
    total_frames = S.shape[-1]

    # Process in windows the size of a training crop so GroupNorm sees a familiar extent.
    window = max(divisor, ((config.train.crop_len * SAMPLE_RATE) // hop) // divisor * divisor)

    activity = np.zeros(total_frames, dtype=np.float32)
    S = S.to(device)
    for start in range(0, total_frames, window):
        chunk = S[:, :, start:start + window]
        w = chunk.shape[-1] - chunk.shape[-1] % divisor   # last window may be short; crop to divisor
        if w == 0:
            break
        chunk = chunk[:, :, :w]
        out = model(chunk.unsqueeze(0))[0]                # (w,)
        # logprob head -> exp to probabilities; signal head -> beats are positive peaks
        # (the 'corr' loss enforces the sign, so no flip is needed here).
        out = out.exp() if is_logprob else out.clamp_min(0)
        activity[start:start + w] = out.cpu().numpy()

    # Map frame activity (frame t centred at sample t*hop of the 4kHz signal) onto the
    # native time axis so it aligns with the input waveform.
    frame_times = np.arange(total_frames) * hop / SAMPLE_RATE
    native_times = np.arange(n_native) / src_hz
    return np.interp(native_times, frame_times, activity).astype(np.float32)
