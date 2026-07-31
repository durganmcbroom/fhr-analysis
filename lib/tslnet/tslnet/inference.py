"""Run a trained TSLNet on a raw waveform and get a beat-activity signal over time.

Mirrors ``funet.inference`` -- same signature, same return contract -- so the analyze pipeline
can swap one model for the other. The difference is only the front-end: TSLNet decimates the
stacked fibers to ``model_hz`` (``tslnet.data.to_model_rate``) rather than building a
spectrogram.

The model trains on fixed crop_len-second crops, so inference processes the series in equal
step windows (matching the training step count) and stitches the per-window activity back
together -- see common.phases.inference.run_windowed. The step-rate activity is then mapped
onto the input's own time axis so it lines up sample-for-sample with the source waveform.
"""

import numpy as np
import torch

from common.audio import SAMPLE_RATE, resample
from common.phases.inference import (
    activity_postprocess, frames_to_native, load_model, normalize_blocks, run_windowed,
)
from common.preprocess import Preprocessor

from tslnet.data import crop_samples, decimation, model_steps, to_model_rate
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
    # crop_samples(), not crop_len * SAMPLE_RATE: TSLNet's crop_len is a float (2.56 s), so the
    # product is a float, and this is the exact same block size the dataset cropped to.
    x = normalize_blocks(x, crop_samples(config))

    m = config.model
    # The same deterministic transforms the dataset applied, at the same 4 kHz rate, or the
    # model meets an input distribution it never trained on (see common.preprocess). This is
    # also where band-limiting happens: the waveform front-end does none of its own.
    waveform = Preprocessor(config.data.preprocess)(torch.from_numpy(np.ascontiguousarray(x)))

    series = to_model_rate(waveform, m.model_hz)                   # (channels, steps)

    # Window the step axis to a training-sized crop, rounded down to a whole number of
    # patches -- the only step count the backbone can take.
    steps_per_crop = model_steps(config)
    window = max(m.patch_length, steps_per_crop // m.patch_length * m.patch_length)

    # How a window of raw output becomes activity depends on what the loss pinned down, so the
    # readout is chosen from the loss rather than fixed here -- see ACTIVITY_POSTPROCESS.
    # NOTE: inference-only; training optimizes the raw signal-head output.
    postprocess = activity_postprocess(config.train.loss)

    activity = run_windowed(model, series, window, device=device, postprocess=postprocess)

    # Step t sits at sample t*decimation of the 4 kHz signal, so that is the hop that maps the
    # result back onto the source waveform's own time axis.
    return frames_to_native(activity, decimation(m.model_hz), SAMPLE_RATE, n_native, src_hz)
