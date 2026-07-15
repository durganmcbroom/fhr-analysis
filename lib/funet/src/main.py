import os
import sys

import torch
from torch import nn, optim

from config import Config, load_config
from data import make_train_dataloader, make_test_dataloader
from loss import SNRLoss, CorrelationLoss
from model import FUNet
from train import fit

OPTIMIZERS = {
    "SGD": optim.SGD,
    "Adam": optim.Adam,
    "AdamW": optim.AdamW,
}

# loss name -> (loss module factory, matching model output head)
LOSSES = {
    "kldiv": (lambda: nn.KLDivLoss(reduction="batchmean"), "logprob"),
    "snr": (SNRLoss, "signal"),
    "corr": (CorrelationLoss, "signal"),   # sign-sensitive; fixes the SI-SNR sign-flip
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


def main(config: Config):
    device = pick_device()
    print(f"Using device: {device}")

    try:
        loss_factory, head = LOSSES[config.train.loss]
    except KeyError:
        raise ValueError(f"Unknown loss: {config.train.loss!r} (expected one of {list(LOSSES)})") from None
    loss_fn = loss_factory()
    print(f"Loss: {config.train.loss} (model head: {head})")

    model = FUNet(
        channels=config.model.channels,
        dilations=config.model.dilations,
        bottleneck_dilation=config.model.bottleneck_dilation,
        base_channels=config.model.base_channels,
        head=head,
        dropout=config.model.dropout,
    )

    if config.resume is not None:
        print(f"Resuming from checkpoint: '{config.resume}'")
        model.load_state_dict(torch.load(config.resume, map_location=device))

    train_dl = make_train_dataloader(config)
    val_dl = make_test_dataloader(config)
    optimiser = build_optimiser(config, model)
    scheduler = build_scheduler(config, optimiser)

    os.makedirs(config.model_dir, exist_ok=True)

    print("-------- Start of Training --------")
    fit(
        model=model,
        train_data=train_dl,
        val_data=val_dl,
        optimiser=optimiser,
        loss_fn=loss_fn,
        epochs=config.train.epochs,
        device=device,
        config={"model_dir": config.model_dir},
        clip=config.train.clip,
        scheduler=scheduler,
    )


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "fetal-config.yaml"
    loaded = load_config(config_path)
    print(f"Loaded config: '{config_path}'")
    main(loaded)
