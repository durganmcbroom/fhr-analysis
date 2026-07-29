"""The inference phase: load a trained checkpoint and run it over a long input.

Models here train on fixed-length crops, so inference processes a long recording in equal
windows matching the training extent (GroupNorm and friends then see the statistics they
trained on) and stitches the per-window output back together.

What stays in the task: how a waveform becomes model input, and how a window of raw output
becomes an activity value. What lives here: loading, windowing, and mapping a frame-rate
result back onto the input's own time axis. This module must never import optuna.
"""

from typing import Callable, Optional

import numpy as np
import torch
from torch import nn


def load_model(task, config, checkpoint: str, device: Optional[torch.device] = None) -> nn.Module:
    """Build the model ``config`` describes and load ``checkpoint`` into it, in eval mode."""
    device = device or torch.device("cpu")
    model = task.build_model(config)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.to(device).eval()
    return model


@torch.no_grad()
def run_windowed(
        model: nn.Module,
        x: torch.Tensor,
        window: int,
        *,
        device: Optional[torch.device] = None,
        postprocess: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> np.ndarray:
    """Run ``model`` over ``x`` in consecutive ``window``-sized slices of its last axis.

    ``x`` is a single unbatched example, e.g. ``(channels, freq, frames)`` or
    ``(channels, samples)``; the model is expected to map a batch of those to
    ``(batch, window)``. Returns the stitched ``(len,)`` result, trimmed back to ``x``'s
    original length.

    The last axis is padded up to a whole number of windows first, so EVERY window is exactly
    ``window`` wide: a short final window is out-of-distribution (fewer frames shift a
    normalisation layer's per-sample statistics), which visibly inflates the tail of the
    output. Padding reflects so the boundary looks like signal continuing, falling back to
    zeros when the input is too short to reflect that far.
    """
    device = device or next(model.parameters()).device
    total = x.shape[-1]

    pad = (-total) % window
    if pad:
        mode = "reflect" if total > pad else "constant"
        x = torch.nn.functional.pad(x, (0, pad), mode=mode)
    padded = total + pad

    out = np.zeros(padded, dtype=np.float32)
    x = x.to(device)
    for start in range(0, padded, window):
        chunk = x[..., start:start + window]          # always exactly `window` wide
        y = model(chunk.unsqueeze(0))[0]              # (window,)
        if postprocess is not None:
            y = postprocess(y)
        out[start:start + window] = y.cpu().numpy()

    return out[:total]


def frames_to_native(
        activity: np.ndarray,
        hop_length: int,
        model_hz: int,
        n_native: int,
        src_hz: int,
) -> np.ndarray:
    """Map a frame-rate signal onto the input's own sample grid.

    Frame ``t`` is centred at sample ``t * hop_length`` of the ``model_hz`` signal; the result
    is ``n_native`` samples at ``src_hz``, so it lines up with the source waveform.
    """
    frame_times = np.arange(len(activity)) * hop_length / model_hz
    native_times = np.arange(n_native) / src_hz
    return np.interp(native_times, frame_times, activity).astype(np.float32)
