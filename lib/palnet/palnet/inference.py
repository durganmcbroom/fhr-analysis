"""Run a trained PALNet on a raw waveform and get a beat-activity signal over time.

Mirrors ``funet.inference`` and ``tslnet.inference`` -- same signature, same return contract --
so the analyze pipeline and rtmon can swap one model for another. The difference is only the
front-end: PALNet resamples the stacked fibers to ``model_hz`` and hands the waveform to the
backbone, which owns the STFT and mel filterbank (see ``palnet.data`` for why the rate is a
design decision rather than a formality).

The model trains on fixed crop_len-second crops, so inference processes the series in equal
windows matching the training extent -- BatchNorm then sees the spatial statistics it trained
on -- and stitches the per-window activity back together. PALNet is the reason
``common.phases.inference.run_windowed`` takes a ``stride``: it windows *samples* and emits
*frames*, at a fixed 32:1 (or 16:1, or 8:1) ratio, where FUNet and TSLNet emit one value per
input position.
"""

import numpy as np
import torch

from common.audio import SAMPLE_RATE, resample
from common.phases.inference import (
    activity_postprocess, frames_to_native, load_model, normalize_blocks, run_windowed,
)
from common.preprocess import Preprocessor

from palnet.data import crop_samples, frame_stride, to_model_rate
from palnet.model import PALNet
from palnet.task import PALNetTask


def load_palnet(config, checkpoint: str, device: torch.device = None) -> PALNet:
    """Build a PALNet matching ``config`` and load ``checkpoint`` into it.

    ``checkpoint`` is head-only when the run left the backbone pristine, and a full state dict
    otherwise (fine-tuned, a trainable bn0, or recalibrated BatchNorm statistics) -- see
    ``PALNet.state_dict``. Either way the frozen part comes from ``config.model.checkpoint``
    via the Hugging Face cache.
    """
    return load_model(PALNetTask(), config, checkpoint, device)


def waveform_input(config, x: np.ndarray, src_hz: int) -> torch.Tensor:
    """The exact tensor PALNet is fed for waveform ``x``: ``(channels, samples)`` at model_hz.

    Factored out of ``run_palnet`` so that anything needing to *see* the model's input -- the
    diagnostic, via ``PALNetTask.make_input`` -- gets the same tensor the model gets, rather
    than a reconstruction that could drift from it.

    The order matters and matches the dataset exactly: everything deterministic happens at
    4 kHz (where ``common.preprocess`` designs its bandpass) and the resample to ``model_hz``
    comes last.
    """
    # Peak-normalise on the same time scale a training snippet was normalised on -- not across
    # the whole recording, which lets one loud transient rescale everything else.
    x = resample(x, src_hz, SAMPLE_RATE)
    x = normalize_blocks(x, crop_samples(config))

    # The same deterministic transforms the dataset applied, at the same 4 kHz rate, or the
    # model meets an input distribution it never trained on (see common.preprocess).
    waveform = Preprocessor(config.data.preprocess)(torch.from_numpy(np.ascontiguousarray(x)))

    return to_model_rate(waveform, config.model.model_hz)


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

    series = waveform_input(config, x, src_hz)                 # (channels, samples)

    stride = frame_stride(config)
    # A training-sized crop, in model_hz samples. crop_samples is already aligned to a whole
    # number of frames, so this is an exact multiple of stride -- the only window the pooling
    # stages can take without dropping a partial frame off the end.
    window = max(stride, crop_samples(config) * config.model.model_hz // SAMPLE_RATE)

    # How a window of raw output becomes activity depends on what the loss pinned down, so the
    # readout is chosen from the loss rather than fixed here -- see ACTIVITY_POSTPROCESS.
    # NOTE: inference-only; training optimizes the raw output.
    postprocess = activity_postprocess(config.train.loss)

    activity = run_windowed(model, series, window, device=device, postprocess=postprocess,
                            stride=stride)

    # Frame t is placed at model sample t*stride, which is the convention funet and tslnet use
    # and the one HRMetrics scores both sides through. (A frame actually *aggregates* samples
    # [t*stride, (t+1)*stride), so its centre of mass sits half a frame later -- 16 ms at the
    # default geometry, against FUNet's 32 ms. Correcting it here alone would put PALNet on a
    # different timing convention from every other model in the repo, so it is left as a
    # measured question rather than a unilateral shift.)
    return frames_to_native(activity, stride, config.model.model_hz, n_native, src_hz,
                            interpolation=config.model.interpolation)
