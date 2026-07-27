import os
from typing import List, Optional, Union

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from model import FUNet


def plot_training_curves(
        train_losses: List[Optional[float]],
        val_losses: List[Optional[float]],
        out_path: str,
):
    """Save a train/test loss-vs-epoch plot to ``out_path`` (PNG).

    Imports matplotlib lazily and forces the headless Agg backend, so it works when
    training runs without a display (remote box, CI, nohup, ...).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = list(range(1, len(train_losses) + 1))
    # A None loss (e.g. an epoch that produced no batch loss) becomes nan so the line
    # simply breaks there instead of erroring.
    train = [float("nan") if v is None else v for v in train_losses]
    val = [float("nan") if v is None else v for v in val_losses]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train, label="Train", color="#1f77b4")
    ax.plot(epochs, val, label="Test", color="#d62728")

    # Mark the lowest test loss (the checkpoint saved as model_best.pt).
    finite = [(e, v) for e, v in zip(epochs, val) if v == v]  # v == v drops nan
    if finite:
        best_e, best_v = min(finite, key=lambda ev: ev[1])
        ax.scatter([best_e], [best_v], color="#d62728", zorder=5)
        ax.annotate(f"best: {best_v:.4f} @ epoch {best_e}", (best_e, best_v),
                    textcoords="offset points", xytext=(0, 9), ha="center", fontsize=8)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training curves")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)



def train(
        model: nn.Module,
        data: DataLoader,
        optimiser: Union[optim.SGD, optim.Adam, optim.AdamW],
        loss_fn: Union[nn.MSELoss],
        device: torch.device,
        clip: float = None,
):
    model.to(device)
    model.train()

    total_loss = 0.0
    max_grad_norm = 0.0

    for i, (inpt, target) in enumerate(data):
        # transfer to GPU
        inpt = inpt.to(device)
        target = target.to(device)

        optimiser.zero_grad()

        # forward pass
        output = model(inpt)
        loss = loss_fn(output, target)

        # backward pass
        loss.backward()
        if clip is not None:
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), clip).item()
            max_grad_norm = max(max_grad_norm, grad_norm)
        optimiser.step()

        total_loss += loss.item()

    # average over batches so it's comparable to test() and not dominated by one noisy batch
    return total_loss / len(data), max_grad_norm

def test(
        model: nn.Module,
        device: torch.device,
        dataloader: DataLoader,
        loss_fn: Union[nn.MSELoss]
):
    model.to(device)
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for inpt, target in dataloader:
            # transfer to GPU
            inpt = inpt.to(device)
            target = target.to(device)

            # get output
            output = model(inpt)

            loss = loss_fn(output, target)
            total_loss += loss.item()

    return total_loss / len(dataloader)


def fit(
        model: nn.Module,
        train_data: DataLoader,
        val_data: DataLoader,
        optimiser: Union[optim.SGD, optim.Adam, optim.AdamW],
        loss_fn: Union[nn.MSELoss],
        epochs: int,
        device: torch.device,
        config: dict,
        clip: float = None,
        scheduler=None,   # optional per-epoch LR scheduler (e.g. CosineAnnealingLR)
):
    lowest_loss = float("inf")
    train_losses, test_losses = [], []

    for epoch in range(epochs):
        train_loss, max_grad_norm = train(model, train_data, optimiser, loss_fn, device, clip)
        test_loss = test(model, device, val_data, loss_fn)

        train_losses.append(train_loss)
        test_losses.append(test_loss)

        if lowest_loss > test_loss:
            lowest_loss = test_loss
            torch.save(model.state_dict(), os.path.join(config['model_dir'], 'model_best.pt'))

        # Report the LR this epoch actually ran at, then step the schedule for the next one.
        lr_note = f', LR: {optimiser.param_groups[0]["lr"]:.2e}' if scheduler is not None else ''
        if scheduler is not None:
            scheduler.step()

        print(f'[{epoch+1}|{epochs}] Train loss: {train_loss:.6f}, Test loss: {test_loss:.6f}, '
              f'Max grad norm (pre-clip): {max_grad_norm:.4f}{lr_note}')

    torch.save(model.state_dict(), os.path.join(config['model_dir'], 'model_last.pt'))

    plot_path = os.path.join(config['model_dir'], 'training_curves.png')
    plot_training_curves(train_losses, test_losses, plot_path)
    print(f'Saved training curves to {plot_path}')