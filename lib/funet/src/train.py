import os
from typing import Union

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from model import FUNet



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
):
    lowest_loss = float("inf")

    for epoch in range(epochs):
        train_loss, max_grad_norm = train(model, train_data, optimiser, loss_fn, device, clip)
        test_loss = test(model, device, val_data, loss_fn)

        if lowest_loss > test_loss:
            lowest_loss = test_loss
            torch.save(model.state_dict(), os.path.join(config['model_dir'], 'model_best.pt'))

        print(f'[{epoch+1}|{epochs}] Train loss: {train_loss:.6f}, Test loss: {test_loss:.6f}, '
              f'Max grad norm (pre-clip): {max_grad_norm:.4f}')

    torch.save(model.state_dict(), os.path.join(config['model_dir'], 'model_last.pt'))