"""Run a trained TSLNet on a raw waveform and get a beat-activity signal over time.

Mirrors ``funet.inference`` -- same signature, same return contract -- so the analyze pipeline
can swap one model for the other. The difference is only the front-end: TSLNet turns the
stacked fibers into a band-energy envelope (``tslnet.data.Envelope``) rather than a
spectrogram.

The model trains on fixed crop_len-second crops, so inference processes the envelope in equal
frame windows (matching the training frame count) and stitches the per-window activity back
together -- see common.phases.inference.run_windowed. The frame-rate activity is then mapped
onto the input's own time axis so it lines up sample-for-sample with the source waveform.
"""

import numpy as np
import torch

from common.audio import SAMPLE_RATE, resample
from common.phases.inference import (
    activity_postprocess, frames_to_native, load_model, normalize_blocks, run_windowed,
)
from common.preprocess import Preprocessor

from tslnet.data import Envelope
from tslnet.model import TSLNet
from tslnet.task import TSLNetTask


def load_tslnet(config, checkpoint: str, device: torch.device = None) -> TSLNet:
    """Build a TSLNet matching ``config`` and load head weights from ``checkpoint``.

    ``checkpoint`` is a head-only file (see TSLNet.state_dict); the frozen backbone comes from
    ``config.model.checkpoint`` via the Hugging Face cache.
    """
    return load_model(TSLNetTask(), config, checkpoint, device)


@torch.no_grad()
def run_tslnet(
        x: np.ndarray,
        src_hz: int,
        model: TSLNet,
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

    # Match training preprocessing: resample to the model rate, then peak-normalise on the
    # same time scale a training snippet was normalised on -- not across the whole recording,
    # which lets one loud transient rescale everything else (see normalize_blocks).
    x = resample(x, src_hz, SAMPLE_RATE)
    x = normalize_blocks(x, config.train.crop_len * SAMPLE_RATE)

    m = config.model
    # The same deterministic transforms the dataset applied, or the model meets an input
    # distribution it never trained on (see common.preprocess).
    waveform = Preprocessor(config.data.preprocess)(torch.from_numpy(np.ascontiguousarray(x)))

    envelope = Envelope(m.n_fft, m.hop_length, m.band, log=m.log_envelope)
    series = envelope(waveform)                                    # (channels, frames)

    # Window the frame axis to a training-sized crop, rounded down to a whole number of
    # patches -- the only frame count the backbone can take.
    frames_per_crop = config.train.crop_len * SAMPLE_RATE // m.hop_length
    window = max(m.patch_length, frames_per_crop // m.patch_length * m.patch_length)

    # How a window of raw output becomes activity depends on what the loss pinned down, so the
    # readout is chosen from the loss rather than fixed here -- see ACTIVITY_POSTPROCESS.
    # NOTE: inference-only; training optimizes the raw signal-head output.
    postprocess = activity_postprocess(config.train.loss)

    activity = run_windowed(model, series, window, device=device, postprocess=postprocess)

    return frames_to_native(activity, m.hop_length, SAMPLE_RATE, n_native, src_hz)
