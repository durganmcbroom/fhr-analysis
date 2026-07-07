import math
import os
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.io import wavfile
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset

# Make the model src dir importable without modifying it.
MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "neossnet"
sys.path.insert(0, str(MODEL_DIR))

from models import MaskNet          # noqa: E402
from train import get_optimizer, get_loss_fn, fit  # noqa: E402

MODEL_HZ = 4000
VAL_FRACTION = 0.1   # fraction of snippets (by index, held out at the end) used for validation

# Config keys whose values are filesystem paths. They are resolved relative to the
# directory the config yaml lives in (see resolve_config_paths), so the same yaml
# works regardless of where python is invoked from.
PATH_CONFIG_KEYS = ("train_dir", "model_dir", "train_resume")


def resolve_config_paths(config, base_dir):
    """Resolve the config's path fields relative to ``base_dir`` (the config yaml's
    own directory), in place, and return ``config``.

    Absolute paths are left as-is; ``..`` segments are normalised. Missing or null
    keys are skipped. This makes every path inside the yaml relative to the yaml,
    not to the process's current working directory.
    """
    base_dir = Path(base_dir).resolve()
    for key in PATH_CONFIG_KEYS:
        value = config.get(key)
        if value is None:
            continue
        # `base_dir / value` yields `value` unchanged when it is already absolute.
        config[key] = str((base_dir / value).resolve())
    return config


def pick_device() -> torch.device:
    """CUDA if present, else Apple MPS, else CPU. Avoids train.py's hardcoded cuda:0.

    Set TUNE_DEVICE (e.g. "cpu") to override. MPS works now that the in-place
    positional-encoding add in models/transformer.py was made out-of-place; before
    that fix it raised an autograd "inplace operation" error on the backward pass.
    """
    forced = os.environ.get("TUNE_DEVICE")
    if forced:
        return torch.device(forced)
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class PreprocessedPairs(Dataset):
    """Paired snippet dataset: returns (mix, target).

    mix    -> (1, crop_samples)
    target -> (num_sources, crop_samples): [heart, lung] or [heart, lung, noise]

    mix and all sources are cropped with a single shared offset so they stay
    time-aligned, and to a fixed length so the default collate can stack them.
    If a clip is shorter than crop_samples it is zero-padded.
    """

    def __init__(self, snippet_dir, indices, num_sources, crop_samples, random_crop):
        self.dir = Path(snippet_dir)
        self.indices = indices
        self.num_sources = num_sources
        self.crop_samples = crop_samples
        self.random_crop = random_crop

    def __len__(self):
        return len(self.indices)

    def _load(self, name):
        # scipy instead of torchaudio.load: avoids the torchcodec backend dependency.
        _, data = wavfile.read(str(self.dir / name))
        if np.issubdtype(data.dtype, np.integer):  # normalize int PCM to [-1, 1]
            data = data.astype(np.float32) / np.iinfo(data.dtype).max
        else:
            data = data.astype(np.float32)
        if data.ndim == 1:
            return torch.from_numpy(data).reshape(1, -1)          # mono -> (1, T)
        return torch.from_numpy(np.ascontiguousarray(data.T))     # (T, C) wav -> (C, T)

    @staticmethod
    def _pad(x, n):
        if x.shape[-1] >= n:
            return x
        pad = torch.zeros(x.shape[0], n - x.shape[-1], dtype=x.dtype)
        return torch.cat([x, pad], dim=-1)

    def __getitem__(self, i):
        idx = self.indices[i]
        mix = self._load(f"{idx}_mix.wav")
        sources = [self._load(f"{idx}_heart.wav"), self._load(f"{idx}_lung.wav")]
        if self.num_sources == 3:
            sources.append(self._load(f"{idx}_noise.wav"))

        # Align lengths (snippets may differ by a sample), then crop/pad to a fixed size.
        n = self.crop_samples
        avail = min([mix.shape[-1]] + [s.shape[-1] for s in sources])
        if avail >= n:
            start = random.randint(0, avail - n) if self.random_crop else 0
            sl = slice(start, start + n)
            mix = mix[:, sl]
            sources = [s[:, sl] for s in sources]
        else:
            mix = self._pad(mix, n)
            sources = [self._pad(s, n) for s in sources]

        target = torch.cat(sources, dim=0)  # (num_sources, n)
        return mix, target


def make_loaders(config):
    snippet_dir = config["train_dir"]
    num_sources = config["hyperparam"]["model_config"]["num_sources"]
    batch_size = config["hyperparam"]["batch_size"]
    num_workers = config["num_workers"]
    crop_samples = config["hyperparam"]["crop_len"] * MODEL_HZ

    mix_files = sorted(Path(snippet_dir).glob("*_mix.wav"),
                       key=lambda p: int(p.stem.split("_")[0]))
    indices = [int(p.stem.split("_")[0]) for p in mix_files]
    if not indices:
        raise FileNotFoundError(f"No *_mix.wav snippets found in {snippet_dir!r}")

    n_val = max(1, int(len(indices) * VAL_FRACTION))
    train_idx, val_idx = indices[:-n_val], indices[-n_val:]
    if not train_idx:
        raise ValueError(f"Only {len(indices)} snippet(s); not enough to hold out a validation set.")

    # train.py's inner loop divides by len(dataloader)//5, so <5 train batches => ZeroDivisionError.
    if math.ceil(len(train_idx) / batch_size) < 5:
        print(f"WARNING: only {len(train_idx)} training snippets ({math.ceil(len(train_idx)/batch_size)} "
              f"batches). train.py needs >=5 batches; add more snippets or lower batch_size.")

    train_ds = PreprocessedPairs(snippet_dir, train_idx, num_sources, crop_samples, random_crop=True)
    val_ds = PreprocessedPairs(snippet_dir, val_idx, num_sources, crop_samples, random_crop=False)

    print(f"Loaded {len(indices)} snippets from {snippet_dir} -> {len(train_idx)} train / {len(val_idx)} val")

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
    return train_dl, val_dl


def tile_encoder_for_channels(state_dict, target_channels=1):
    # Warm-start a multi-channel encoder from a 1-channel checkpoint by tiling the
    # single input-channel weight across the new channels (divided to keep scale).
    weight = state_dict.get("encoder.weight")
    if weight is None or weight.shape[1] == target_channels:
        return state_dict
    adapted = dict(state_dict)
    adapted["encoder.weight"] = weight.repeat(1, target_channels, 1) / target_channels
    return adapted


def build_model(config, device):
    """Like train.get_model but loads the checkpoint onto the chosen device, not cuda:0."""
    model = MaskNet(**config["hyperparam"]["model_config"])
    resume = config.get("train_resume")
    if resume is not None:
        state_dict = torch.load(resume, map_location=device)
        state_dict = tile_encoder_for_channels(state_dict, 1)
        model.load_state_dict(state_dict)
    return model


def main(config):
    device = pick_device()
    print(f"Using device: {device}")

    model = build_model(config, device)
    train_dl, val_dl = make_loaders(config)
    optimiser = get_optimizer(config, model)
    loss_fn = get_loss_fn(config)
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer=optimiser,
        factor=config["hyperparam"]["factor"],
        patience=config["hyperparam"]["patience"],
    )
    earlystop = config["hyperparam"]["earlystop_patient"]

    model_dir = config["model_dir"]
    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)
    os.makedirs(model_dir)
    with open(os.path.join(model_dir, "model.yaml"), "w") as f:
        yaml.dump(config["hyperparam"]["model_config"], f)

    print("-------- Start of Training --------")
    fit(model, train_dl, val_dl, loss_fn, optimiser, scheduler, device, config, earlystop)

    with open(os.path.join(model_dir, "config.yaml"), "w") as f:
        yaml.dump(config, f)


if __name__ == "__main__":
    config_path = Path(sys.argv[1] if len(sys.argv) > 1 else "fetal-tune-config.yaml").resolve()
    with open(config_path, "r") as f:
        print(f"Loaded Config: '{config_path}'")
        config = yaml.safe_load(f)
    # Resolve every path in the yaml relative to the yaml's own directory.
    resolve_config_paths(config, config_path.parent)
    main(config)
