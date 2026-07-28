"""Optuna hyperparameter search for FUNet.

Runs many short training trials, each with a different set of hyperparameters that Optuna
picks, and keeps the ones reaching the lowest validation loss. Everything that is *not*
searched (data dirs, batch size, loss, crop_len, channels, augment, ...) is inherited from a
base YAML config -- the same file main.py takes -- so the search sweeps only the knobs in
``suggest_config`` below and leaves the rest of your setup untouched.

Usage:
    python src/tune.py [base-config.yaml] [--trials N] [--epochs N] [--storage URL] [--seed S]

After the search the best hyperparameters are written to a runnable config
(``<model_dir>/best-config.yaml`` by default) that trains a final model with the full epoch
budget:
    python src/main.py <model_dir>/best-config.yaml

Design: every trial goes through the exact same ``run_training`` the CLI uses, so a config the
search likes trains identically when you run it for real -- the tuner only chooses the numbers
and asks not to save per-trial checkpoints. Objective = validation loss, minimised; that holds
for every supported loss here (they are all lower-is-better).
"""

import argparse
import copy
import os
import sys
from typing import Optional

import optuna
import torch
import yaml

from config import Config, load_config
from data import stft_output_shape
from main import OPTIMIZERS, run_training


# --------------------------------------------------------------------------------------
# Search space
# --------------------------------------------------------------------------------------

def suggest_config(trial: optuna.Trial, base: Config) -> Config:
    """Return a copy of ``base`` with the searched hyperparameters replaced for this trial.

    Only the fields set here are touched; every other field on ``base`` is inherited as-is.
    Keep this in sync with ``searched_fields`` (which serialises the same set back to YAML).
    """
    config = copy.deepcopy(base)
    model, train, data = config.model, config.train, config.data

    # -- Architecture --
    # `dilations` encodes both the network depth (its length) and the per-level dilation, so we
    # sample a depth and then one dilation per level. Depth drives the 2**depth downsampling
    # that must divide the spectrogram (enforced in `objective`); it is capped at 6 because
    # deeper nets stop fitting the smaller n_fft / larger hops in the space below.
    depth = trial.suggest_int("depth", 3, 6)
    model.dilations = [
        trial.suggest_categorical(f"dilation_l{i}", [1, 2, 4, 8]) for i in range(depth)
    ]
    model.bottleneck_dilation = trial.suggest_categorical("bottleneck_dilation", [1, 2, 4, 8])
    model.bottleneck_convs = trial.suggest_int("bottleneck_convs", 1, 4)
    model.codec_convolutions = trial.suggest_int("codec_convolutions", 1, 4)
    model.base_channels = trial.suggest_categorical("base_channels", [8, 12, 16, 24, 32])
    model.dropout = trial.suggest_float("dropout", 0.0, 0.5)

    # -- Optimisation --
    train.optimizer = trial.suggest_categorical("optimizer", list(OPTIMIZERS))
    train.learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
    train.weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-1, log=True)
    # min_lr is the cosine floor, so it is only meaningful below the peak LR. Sampling it as a
    # fraction of learning_rate guarantees min_lr < learning_rate for every trial (sampling the
    # two independently could put the floor above the peak). Only used when the base config's
    # lr_schedule is 'cosine'; harmless otherwise.
    train.min_lr = train.learning_rate * trial.suggest_float("min_lr_frac", 1e-3, 1e-1, log=True)

    # -- Spectrogram / SpecAugment --
    data.n_fft = trial.suggest_categorical("n_fft", [512, 1024, 2048])
    data.hop_length = trial.suggest_categorical("hop_length", [64, 128, 256, 512])
    data.freq_mask = trial.suggest_int("freq_mask", 0, 64)   # max freq bins zeroed; 0 = off
    data.time_mask = trial.suggest_int("time_mask", 0, 8)    # max time frames zeroed; 0 = off

    return config


def searched_fields(config: Config) -> dict:
    """The searched fields of ``config``, shaped like the config YAML (model/train/data).

    Used to write the winning trial back out as a runnable config. Mirrors the set of fields
    ``suggest_config`` assigns -- keep the two together.
    """
    return {
        "model": {
            "dilations": config.model.dilations,
            "bottleneck_dilation": config.model.bottleneck_dilation,
            "bottleneck_convs": config.model.bottleneck_convs,
            "codec_convolutions": config.model.codec_convolutions,
            "base_channels": config.model.base_channels,
            "dropout": config.model.dropout,
        },
        "train": {
            "optimizer": config.train.optimizer,
            "learning_rate": config.train.learning_rate,
            "weight_decay": config.train.weight_decay,
            "min_lr": config.train.min_lr,
        },
        "data": {
            "n_fft": config.data.n_fft,
            "hop_length": config.data.hop_length,
            "freq_mask": config.data.freq_mask,
            "time_mask": config.data.time_mask,
        },
    }


# --------------------------------------------------------------------------------------
# Objective
# --------------------------------------------------------------------------------------

def _prune_if_geometry_unfit(config: Config) -> None:
    """Prune trials whose network is deeper than the spectrogram can support.

    FUNet halves freq and time once per level, so it needs 2**depth <= both the freq-bin count
    and the frame count (see FUNet.forward / data.__getitem__). A deep net combined with a
    small n_fft or a large hop violates that; skipping the trial is cleaner than letting the
    model constructor / forward pass raise mid-run.
    """
    freq_bins, time_frames = stft_output_shape(config)
    divisor = 2 ** len(config.model.dilations)
    if divisor > freq_bins or divisor > time_frames:
        raise optuna.TrialPruned(
            f"depth {len(config.model.dilations)} needs freq and time >= {divisor}, but this "
            f"config yields freq={freq_bins}, time={time_frames}"
        )


def objective(trial: optuna.Trial, base: Config, epochs: Optional[int] = None) -> float:
    """Train one sampled config and return its best validation loss (lower is better)."""
    config = suggest_config(trial, base)
    _prune_if_geometry_unfit(config)
    if epochs is not None:
        config.train.epochs = epochs   # short trials; the emitted best config keeps base epochs

    def on_epoch(epoch: int, test_loss: float) -> None:
        trial.report(test_loss, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    try:
        return run_training(config, on_epoch=on_epoch, save_artifacts=False)
    except RuntimeError as e:
        # One oversized model (base_channels * 2**depth channels at the bottleneck can get big)
        # shouldn't sink the whole study -- treat OOM as an unfit trial and move on.
        if "out of memory" in str(e).lower():
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise optuna.TrialPruned(f"out of memory: {e}") from None
        raise


# --------------------------------------------------------------------------------------
# Study driver / CLI
# --------------------------------------------------------------------------------------

def write_best_config(base_config_path: str, best_params: dict, out_path: str) -> None:
    """Write a runnable config = the base YAML with the searched fields set to ``best_params``.

    Rebuilding the config from the params through ``suggest_config`` (via a FixedTrial) rather
    than reading them off the trial guarantees the emitted values are exactly what was evaluated
    -- including the derived ones (min_lr from min_lr_frac, dilations from depth + per-level).
    """
    base = load_config(base_config_path)
    best = suggest_config(optuna.trial.FixedTrial(best_params), base)

    with open(base_config_path) as f:
        raw = yaml.safe_load(f)
    for section, fields in searched_fields(best).items():
        raw.setdefault(section, {}).update(fields)

    with open(out_path, "w") as f:
        yaml.safe_dump(raw, f, sort_keys=False)


def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config", nargs="?", default="fetal-config.yaml",
                   help="base FUNet config; fixed fields inherited, searched fields overridden")
    p.add_argument("--trials", type=int, default=50, help="number of trials to run (default 50)")
    p.add_argument("--timeout", type=float, default=None,
                   help="stop the search after this many seconds (default: no limit)")
    p.add_argument("--epochs", type=int, default=None,
                   help="epochs per trial (default: the config's epochs); fewer = faster search")
    p.add_argument("--storage", default=None,
                   help="Optuna storage URL, e.g. sqlite:///funet-tuning.db, to persist/resume")
    p.add_argument("--study-name", default="funet", help="study name (used with --storage)")
    p.add_argument("--seed", type=int, default=None, help="sampler seed for reproducibility")
    p.add_argument("--out", default=None,
                   help="path for the best config (default: <model_dir>/best-config.yaml)")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    base = load_config(args.config)
    print(f"Base config: '{args.config}'")
    print(f"Trials: {args.trials}, epochs/trial: {args.epochs or base.train.epochs}")

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=args.seed),
        pruner=optuna.pruners.MedianPruner(),   # stops trials tracking worse than the median
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=True,
    )
    study.optimize(
        lambda trial: objective(trial, base, epochs=args.epochs),
        n_trials=args.trials,
        timeout=args.timeout,
    )

    print("\n===== Search complete =====")
    print(f"Completed trials: {len(study.get_trials(states=(optuna.trial.TrialState.COMPLETE,)))}"
          f" / {len(study.trials)}")
    print(f"Best validation loss: {study.best_value:.6f}")
    print("Best hyperparameters:")
    for key, value in sorted(study.best_params.items()):
        print(f"  {key}: {value}")

    out_path = args.out or os.path.join(base.model_dir, "best-config.yaml")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    write_best_config(args.config, study.best_params, out_path)
    print(f"\nWrote best config to '{out_path}'. Train the final model with:\n"
          f"  python src/main.py {out_path}")


if __name__ == "__main__":
    main(sys.argv[1:])
