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

from config import Config
from data import SAMPLE_RATE
from model import FUNet


# Which output head each loss trains (mirrors the LOSSES table in main.py). kldiv is a
# distribution (log_softmax); snr, corr and corr_amp are signal-regression (raw output).
_LOSS_HEADS = {"kldiv": "logprob", "snr": "signal", "corr": "signal", "corr_amp": "signal"}


def _head_for(config: Config) -> str:
    try:
        return _LOSS_HEADS[config.train.loss]
    except KeyError:
        raise ValueError(
            f"unknown loss {config.train.loss!r}; expected one of {list(_LOSS_HEADS)}"
        ) from None


def load_funet(config: Config, checkpoint: str, device: torch.device = None) -> FUNet:
    """Build a FUNet matching ``config`` and load weights from ``checkpoint``."""
    device = device or torch.device("cpu")
    model = FUNet(
        channels=config.model.channels,
        dilations=config.model.dilations,
        bottleneck_dilation=config.model.bottleneck_dilation,
        bottleneck_convs=config.model.bottleneck_convs,
        base_channels=config.model.base_channels,
        head=_head_for(config),
        # Inactive under eval(), but dropout>0 shifts Sequential state_dict keys, so the
        # architecture must match the checkpoint's training config to load it.
        dropout=config.model.dropout,
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

    # Process in windows the size of a training crop so GroupNorm sees the same spatial
    # extent it trained on. Pad the time axis up to a whole number of windows first, so
    # EVERY window is exactly `window` frames: a partial final window is out-of-
    # distribution (fewer frames shift GroupNorm's per-sample statistics), which inflated
    # the tail of the output. The padded frames are trimmed off at the end.
    window = max(divisor, ((config.train.crop_len * SAMPLE_RATE) // hop) // divisor * divisor)

    pad_frames = (-total_frames) % window
    if pad_frames:
        # reflect so the boundary looks like signal continuing (falls back to zeros if the
        # clip is too short to reflect that many frames).
        mode = "reflect" if total_frames > pad_frames else "constant"
        S = torch.nn.functional.pad(S, (0, pad_frames), mode=mode)
    padded_frames = total_frames + pad_frames

    activity = np.zeros(padded_frames, dtype=np.float32)
    S = S.to(device)
    for start in range(0, padded_frames, window):
        chunk = S[:, :, start:start + window]             # always exactly `window` frames
        out = model(chunk.unsqueeze(0))[0]                # (window,)
        # Both heads become a per-window softmax activity envelope: logprob already applied
        # log_softmax in forward (exp -> softmax); the signal head (corr/snr) is affine-
        # invariant -- corr never pins a baseline or scale -- so softmax normalizes its
        # arbitrary offset away into a clean positive envelope. Full-size windows keep the
        # normalization consistent across them. NOTE: inference-only; training optimizes the
        # raw signal-head output, not this softmax.
        out = out.exp() if is_logprob else out.softmax(dim=-1)
        activity[start:start + window] = out.cpu().numpy()

    activity = activity[:total_frames]                    # drop the padded tail

    # Map frame activity (frame t centred at sample t*hop of the 4kHz signal) onto the
    # native time axis so it aligns with the input waveform.
    frame_times = np.arange(total_frames) * hop / SAMPLE_RATE
    native_times = np.arange(n_native) / src_hz
    return np.interp(native_times, frame_times, activity).astype(np.float32)
