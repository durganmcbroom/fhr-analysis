import os
import sys
from typing import Callable, Optional

import torch
from torch import nn, optim

from config import Config, load_config
from data import make_train_dataloader, make_test_dataloader
from loss import SNRLoss, CorrelationLoss, CorrAmpLoss, MSELoss
from model import FUNet
from train import fit

OPTIMIZERS = {
    "SGD": optim.SGD,
    "Adam": optim.Adam,
    "AdamW": optim.AdamW,
}

# loss name -> (config -> loss module, matching model output head). Factories take the
# config so amplitude-aware losses can read their hyperparameters from it.
LOSSES = {
    "kldiv": (lambda cfg: nn.KLDivLoss(reduction="batchmean"), "logprob"),
    "snr": (lambda cfg: SNRLoss(), "signal"),
    "corr": (lambda cfg: CorrelationLoss(), "signal"),   # sign-sensitive; fixes the SI-SNR sign-flip
    "corr_amp": (lambda cfg: CorrAmpLoss(amp_weight=cfg.train.amp_weight,   # corr + d' peak-contrast
                                         beat_threshold=cfg.train.amp_beat_threshold), "signal"),
    "mse": (lambda cfg: MSELoss(), "signal"),   # per-frame regression to a unit-peak comb
}


def pick_device() -> torch.device:
    """CUDA if present, else Apple MPS, else CPU. Set FUNET_DEVICE (e.g. "cpu") to override."""
    forced = os.environ.get("FUNET_DEVICE")
    if forced:
        return torch.device(forced)
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_optimiser(config: Config, model: nn.Module) -> optim.Optimizer:
    try:
        cls = OPTIMIZERS[config.train.optimizer]
    except KeyError:
        raise ValueError(f"Unknown optimizer: {config.train.optimizer!r}") from None
    return cls(model.parameters(), lr=config.train.learning_rate, weight_decay=config.train.weight_decay)


def build_scheduler(config: Config, optimiser: optim.Optimizer):
    """None for a constant LR, or cosine annealing learning_rate -> min_lr over the run."""
    if config.train.lr_schedule == "none":
        return None
    if config.train.lr_schedule == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(
            optimiser, T_max=config.train.epochs, eta_min=config.train.min_lr)
    raise ValueError(f"Unknown lr_schedule: {config.train.lr_schedule!r} (expected 'none' or 'cosine')")


def build_model(config: Config, head: str) -> FUNet:
    return FUNet(
        channels=config.model.channels,
        dilations=config.model.dilations,
        bottleneck_dilation=config.model.bottleneck_dilation,
        bottleneck_convs=config.model.bottleneck_convs,
        codec_convolutions=config.model.codec_convolutions,
        base_channels=config.model.base_channels,
        head=head,
        dropout=config.model.dropout,
    )


def run_training(
        config: Config,
        *,
        on_epoch: Optional[Callable[[int, float], None]] = None,
        save_artifacts: bool = True,
) -> float:
    """Build the model/data/optimiser from ``config``, train, and return the best val loss.

    This is the whole training pipeline in one call, shared by the CLI (``main``) and the
    Optuna search (``tune.objective``). The search passes ``save_artifacts=False`` (it keeps
    only the score, not the model) and an ``on_epoch`` callback (to report/prune trials).
    """
    device = pick_device()
    print(f"Using device: {device}")

    try:
        loss_factory, head = LOSSES[config.train.loss]
    except KeyError:
        raise ValueError(f"Unknown loss: {config.train.loss!r} (expected one of {list(LOSSES)})") from None
    loss_fn = loss_factory(config)
    print(f"Loss: {config.train.loss} (model head: {head})")

    model = build_model(config, head)

    if config.resume is not None:
        print(f"Resuming from checkpoint: '{config.resume}'")
        model.load_state_dict(torch.load(config.resume, map_location=device))

    train_dl = make_train_dataloader(config)
    val_dl = make_test_dataloader(config)
    optimiser = build_optimiser(config, model)
    scheduler = build_scheduler(config, optimiser)

    if save_artifacts:
        os.makedirs(config.model_dir, exist_ok=True)

    print("-------- Start of Training --------")
    return fit(
        model=model,
        train_data=train_dl,
        val_data=val_dl,
        optimiser=optimiser,
        loss_fn=loss_fn,
        epochs=config.train.epochs,
        device=device,
        model_dir=config.model_dir,
        clip=config.train.clip,
        scheduler=scheduler,
        save_artifacts=save_artifacts,
        on_epoch=on_epoch,
    )


def main(config: Config):
    run_training(config)


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "fetal-config.yaml"
    loaded = load_config(config_path)
    print(f"Loaded config: '{config_path}'")
    main(loaded)
