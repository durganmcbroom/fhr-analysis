"""Run a trained PALNet on a raw waveform and get a beat-activity signal over time.

Mirrors ``funet.inference`` almost line for line -- same signature, same return contract, same
front-end -- so the analyze pipeline and rtmon can swap one model for the other. The difference
is only what sits between the spectrogram and the frame-rate output: a frozen AudioSet trunk
rather than a learned U-Net.

The model trains on fixed crop_len-second crops, so inference processes the spectrogram in
equal frame windows matching the training extent and stitches the per-window activity back
together -- see common.phases.inference.run_windowed. The frame-rate activity is then mapped
onto the input's own time axis so it lines up sample-for-sample with the source waveform.
"""

import numpy as np
import torch
import torchaudio

from common.audio import SAMPLE_RATE, resample
from common.phases.inference import (
    activity_postprocess, frames_to_native, load_model, normalize_blocks, run_windowed,
)
from common.preprocess import Preprocessor

from palnet.data import freq_crop_bins
from palnet.model import PALNet
from palnet.panns import FREQ_DOWNSAMPLE
from palnet.task import PALNetTask


def load_palnet(config, checkpoint: str, device: torch.device = None) -> PALNet:
    """Build a PALNet matching ``config`` and load head weights from ``checkpoint``.

    ``checkpoint`` is a head-only file (see PALNet.state_dict); the frozen trunk comes from the
    Hugging Face cache.
    """
    return load_model(PALNetTask(), config, checkpoint, device)


def spectrogram_input(config, x: np.ndarray, src_hz: int) -> torch.Tensor:
    """The exact tensor PALNet is fed for waveform ``x``: ``(channels, rows, frames)``.

    Factored out of ``run_palnet`` so that anything needing to *see* the model's input -- the
    diagnostic's first column, via ``PALNetTask.make_input`` -- gets the same tensor the model
    gets, rather than a reconstruction that could drift from it.
    """
    # Match training preprocessing: resample to the model rate, then peak-normalise on the same
    # time scale a training snippet was normalised on -- not across the whole recording, which
    # lets one loud transient rescale everything else (see normalize_blocks).
    x = resample(x, src_hz, SAMPLE_RATE)
    x = normalize_blocks(x, config.train.crop_len * SAMPLE_RATE)

    # The same deterministic transforms the dataset applied, or the model meets an input
    # distribution it never trained on (see common.preprocess).
    waveform = Preprocessor(config.data.preprocess)(torch.from_numpy(np.ascontiguousarray(x)))

    spec = torchaudio.transforms.Spectrogram(n_fft=config.model.n_fft,
                                             hop_length=config.model.hop_length)
    S = torch.log1p(spec(waveform))                       # (channels, freq, frames)

    # Passband crop, from the same helper the dataset uses -- a checkpoint trained on a band
    # must be run on that band, and two copies of the arithmetic would eventually disagree.
    crop = freq_crop_bins(config)
    if crop is not None:
        S = S[:, crop[0]:crop[1], :]

    # Floor the rows the same way the dataset does; the time axis needs no flooring because the
    # trunk's pools are frequency-only.
    freq = S.shape[-2] - S.shape[-2] % FREQ_DOWNSAMPLE
    return S[:, :freq, :]


@torch.no_grad()
def run_palnet(
        x: np.ndarray,
        src_hz: int,
        model: PALNet,
        config,
        device: torch.device = None,
) -> np.ndarray:
    """Beat-activity over time for waveform ``x`` (``(T,)`` or ``(channels, T)``).

    Returns a non-negative activity signal the same length as ``x`` and sampled at ``src_hz``:
    high where the model thinks a fetal beat occurs. Relative peaks (not absolute scale) are
    what carry the beats -- under the affine-invariant losses the units are standard deviations
    above the window's own floor.
    """
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x[None, :]                      # (1, T)
    n_native = x.shape[-1]

    channels = x.shape[0]
    if channels != config.model.channels:
        raise ValueError(
            f"waveform has {channels} channel(s) but the model expects "
            f"{config.model.channels} (config.model.channels)")

    S = spectrogram_input(config, x, src_hz)

    # Window the time axis to a training-sized crop, so the model sees the extent it trained
    # on. No rounding is needed: the trunk does not downsample time.
    hop = config.model.hop_length
    window = max(1, config.train.crop_len * SAMPLE_RATE // hop)

    # How a window of raw output becomes activity depends on what the loss pinned down, so the
    # readout is chosen from the loss rather than fixed here -- see ACTIVITY_POSTPROCESS.
    # NOTE: inference-only; training optimizes the raw output.
    postprocess = activity_postprocess(config.train.loss)

    activity = run_windowed(model, S, window, device=device, postprocess=postprocess)

    return frames_to_native(activity, hop, SAMPLE_RATE, n_native, src_hz,
                            interpolation=config.model.interpolation)
