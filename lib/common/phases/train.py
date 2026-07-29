"""The train phase: build everything from a config and run the loop.

Shared by the CLI entry points and by the Optuna search, so a config the search likes trains
identically when you run it for real. This module must never import optuna.
"""

import os
from typing import Callable, Optional

import torch

from common.device import pick_device
from common.engine import fit
from common.io import write_config

BEST_MODEL = "model_best.pt"
LAST_MODEL = "model_last.pt"
CURVES = "training_curves.png"
CONFIG = "config.yaml"


def run_training(
        task,
        config,
        *,
        on_epoch: Optional[Callable[[int, float], None]] = None,
        save_artifacts: bool = True,
        best_model_path: Optional[str] = None,
) -> float:
    """Build the model/data/optimiser from ``config``, train, and return the best val loss.

    ``save_artifacts`` writes the usual best/last checkpoints, the archived config and the
    curves plot under ``config.model_dir``; the search turns it off and instead passes
    ``best_model_path`` to capture just that trial's best-epoch checkpoint, plus an
    ``on_epoch`` callback to report and prune trials.
    """
    device = pick_device(*task.device_env_vars)
    print(f"Using device: {device}")

    # Fail fast, before building anything, on a config whose geometry cannot work.
    task.check_feasible(config)

    loss_fn = task.build_loss(config)
    model = task.build_model(config)

    if config.resume is not None:
        print(f"Resuming from checkpoint: '{config.resume}'")
        state_dict = torch.load(config.resume, map_location=device)
        model.load_state_dict(task.adapt_state_dict(state_dict, config))

    train_dl = task.make_train_loader(config)
    val_dl = task.make_val_loader(config)
    optimiser = task.build_optimizer(config, model)
    scheduler = task.build_scheduler(config, optimiser)

    # A full run saves best/last/config/curves under model_dir; the search saves only the
    # best-epoch checkpoint at the path it hands us. atomic_save creates parent dirs.
    last_model_path = curves_path = None
    save_config = None
    if save_artifacts:
        best_model_path = best_model_path or os.path.join(config.model_dir, BEST_MODEL)
        last_model_path = os.path.join(config.model_dir, LAST_MODEL)
        curves_path = os.path.join(config.model_dir, CURVES)
        save_config = _config_saver(task, config, os.path.join(config.model_dir, CONFIG))

    print("-------- Start of Training --------")
    return fit(
        model=model,
        train_data=train_dl,
        val_data=val_dl,
        optimiser=optimiser,
        loss_fn=loss_fn,
        epochs=config.train.epochs,
        device=device,
        clip=config.train.clip,
        scheduler=scheduler,
        best_model_path=best_model_path,
        last_model_path=last_model_path,
        curves_path=curves_path,
        save_config=save_config,
        early_stop_patience=config.train.early_stop_patience,
        on_epoch=on_epoch,
    )


def _config_saver(task, config, out_path: str) -> Optional[Callable[[], None]]:
    """A callable that archives ``config`` next to the checkpoints, or None if we can't.

    Re-reads the source YAML rather than dumping the loaded object: loading resolves paths to
    absolutes, so a dumped config would bake this machine's layout into the archive.
    """
    if not config.source_path:
        return None
    overrides = task.config_overrides(config)
    return lambda: write_config(config.source_path, out_path, overrides)
