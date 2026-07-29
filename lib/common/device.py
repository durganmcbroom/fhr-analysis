"""Device selection shared by every model."""

import os

import torch


def pick_device(*env_vars: str) -> torch.device:
    """CUDA if present, else Apple MPS, else CPU.

    Any of ``env_vars`` (checked in order, then ``COMMON_DEVICE``) overrides the choice
    when set, e.g. ``FUNET_DEVICE=cpu``. Tasks pass their own historical variable names so
    existing habits keep working.
    """
    for var in (*env_vars, "COMMON_DEVICE"):
        forced = os.environ.get(var)
        if forced:
            return torch.device(forced)
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
