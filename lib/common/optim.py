"""Optimiser and LR-schedule construction, shared by every model.

Both are chosen by name from the config, so any model gets all three schedules
(``none``/``cosine``/``plateau``) without writing code.
"""

from typing import Optional

from torch import nn, optim

OPTIMIZERS = {
    "SGD": optim.SGD,
    "Adam": optim.Adam,
    "AdamW": optim.AdamW,
}


def build_optimizer(config, model: nn.Module) -> optim.Optimizer:
    name = config.train.optimizer
    try:
        cls = OPTIMIZERS[name]
    except KeyError:
        raise ValueError(f"Unknown optimizer: {name!r} (expected one of {list(OPTIMIZERS)})") from None

    kwargs = dict(lr=config.train.learning_rate, weight_decay=config.train.weight_decay)
    if name == "SGD":
        kwargs["momentum"] = config.train.momentum
    else:
        kwargs["amsgrad"] = config.train.amsgrad

    # Frozen parameters are filtered out rather than passed and ignored. For funet/ssnet this
    # is every parameter, so nothing changes; for tslnet, whose backbone is a frozen 500M-param
    # foundation model, handing them to the optimiser would allocate optimiser state for
    # tensors that never receive a gradient. Erroring on an all-frozen model is deliberate --
    # an optimiser over nothing trains silently and forever.
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise ValueError(
            f"{type(model).__name__} has no trainable parameters (every one has "
            "requires_grad=False); there is nothing for the optimiser to update.")
    return cls(trainable, **kwargs)


class Scheduler:
    """Uniform ``step(val_loss)`` over schedules that do and don't consume a metric.

    ReduceLROnPlateau steps on the validation loss; cosine steps blind. Wrapping the
    difference here means ``engine.fit`` never has to type-check the scheduler -- calling
    ``step()`` unconditionally on a plateau scheduler is a silent no-op-shaped bug.
    """

    def __init__(self, inner, needs_metric: bool):
        self.inner = inner
        self.needs_metric = needs_metric

    def step(self, val_loss: float) -> None:
        if self.needs_metric:
            self.inner.step(val_loss)
        else:
            self.inner.step()


def build_scheduler(config, optimiser: optim.Optimizer) -> Optional[Scheduler]:
    """None for a constant LR, else a wrapped per-epoch scheduler."""
    kind = config.train.lr_schedule

    if kind == "none":
        return None

    if kind == "cosine":
        # NOTE: T_max is the config's epoch count, so a short search trial (--epochs 10)
        # anneals fully within those 10 epochs while the final run anneals over all 80. The
        # LR *trajectory* therefore differs between search and final training -- a trial's
        # result does not perfectly predict the full run.
        if config.train.early_stop_patience is not None:
            print("WARNING: early_stop_patience with lr_schedule 'cosine' -- stopping early "
                  "ends the run mid-anneal, so min_lr is never reached.")
        return Scheduler(
            optim.lr_scheduler.CosineAnnealingLR(
                optimiser, T_max=config.train.epochs, eta_min=config.train.min_lr),
            needs_metric=False,
        )

    if kind == "plateau":
        return Scheduler(
            optim.lr_scheduler.ReduceLROnPlateau(
                optimiser,
                factor=config.train.plateau_factor,
                patience=config.train.plateau_patience,
            ),
            needs_metric=True,
        )

    raise ValueError(
        f"Unknown lr_schedule: {kind!r} (expected 'none', 'cosine' or 'plateau')")
