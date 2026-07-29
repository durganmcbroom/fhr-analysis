"""The optimize phase: an Optuna hyperparameter search over any Task.

Runs many short training trials, each with a different set of hyperparameters that Optuna
picks, and keeps the ones reaching the lowest validation loss. Everything that is *not*
searched (data dirs, batch size, loss, crop_len, augment, ...) is inherited from a base YAML
config -- the same file the train phase takes -- so the search sweeps only the knobs the
task's ``suggest`` touches and leaves the rest of the setup alone.

Usage (from a model's tune.py shim):
    python src/tune.py [base-config.yaml] [--trials N] [--epochs N] [--storage URL] [--seed S]

Preemption-safe: as the search runs it keeps four files current in the output directory
(``model_dir`` by default) -- ``best-config.yaml`` + ``best-model.pt`` for the best trial so
far, and ``latest-config.yaml`` + ``latest-model.pt`` for the most recent one -- rewriting
them atomically after each trial. The study itself is persisted to ``<out-dir>/study.db``, so
if the task is killed and restarted it resumes where it left off (which is also what keeps
the saved "best" honest: a fresh study would otherwise overwrite it with its first trial).

Train the final model from the best config with the full epoch budget:
    python src/main.py <out-dir>/best-config.yaml

Design: every trial goes through the exact same ``run_training`` the train phase uses, so a
config the search likes trains identically when you run it for real -- the tuner only chooses
the numbers. Objective = validation loss, minimised.

This is the ONLY module in common that imports optuna. Tasks never do: their ``suggest``
receives a trial object structurally, and ``check_feasible`` raises ``InfeasibleConfig``,
which this module translates into a pruned trial.
"""

import argparse
import os
from typing import Optional

import optuna
import torch

from common.config import load_config
from common.errors import InfeasibleConfig
from common.io import atomic_copy, write_config
from common.phases.train import run_training

# Filenames kept current in the output directory (see module docstring).
BEST_CONFIG, BEST_MODEL = "best-config.yaml", "best-model.pt"
LATEST_CONFIG, LATEST_MODEL = "latest-config.yaml", "latest-model.pt"
STUDY_DB = "study.db"


# --------------------------------------------------------------------------------------
# Objective
# --------------------------------------------------------------------------------------

def _best_value_so_far(study: optuna.Study) -> float:
    """Best value among trials already recorded complete. Called from inside a running trial,
    which Optuna has not recorded yet, so this is the bar the current trial must beat.
    Infinity until the first trial completes."""
    try:
        return study.best_value
    except ValueError:   # no completed trials yet
        return float("inf")


def objective(trial: optuna.Trial, task, base, base_config_path: str,
              out_dir: str, epochs: Optional[int] = None) -> float:
    """Train one sampled config, persist it as the latest (and best, if it wins), and return
    its best validation loss (lower is better)."""
    config = task.suggest(trial, base)

    try:
        task.check_feasible(config)
    except InfeasibleConfig as e:
        # Skipping is cleaner than letting the model constructor or forward pass raise mid-run.
        raise optuna.TrialPruned(str(e)) from None

    if epochs is not None:
        config.train.epochs = epochs   # short trials; the emitted best config keeps base epochs

    def on_epoch(epoch: int, val_loss: float) -> None:
        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    # The trial's best-epoch weights are written here first; only a fully completed trial
    # promotes them into the shared latest/best slots, so a pruned or crashed trial can never
    # leave latest-model.pt out of sync with latest-config.yaml. The finally cleans up the
    # temp file for any trial that doesn't promote it.
    trial_model = os.path.join(out_dir, f".trial-{trial.number}.pt")
    searched = task.searched_fields(config)
    try:
        loss = run_training(task, config, on_epoch=on_epoch, save_artifacts=False,
                            best_model_path=trial_model)

        os.replace(trial_model, os.path.join(out_dir, LATEST_MODEL))
        write_config(base_config_path, os.path.join(out_dir, LATEST_CONFIG), searched)
        if loss < _best_value_so_far(trial.study):
            atomic_copy(os.path.join(out_dir, LATEST_MODEL), os.path.join(out_dir, BEST_MODEL))
            write_config(base_config_path, os.path.join(out_dir, BEST_CONFIG), searched)
            print(f"  new best: {loss:.6f} -> saved {BEST_CONFIG} + {BEST_MODEL}")
        return loss
    except RuntimeError as e:
        # One oversized model shouldn't sink the whole study -- treat OOM as an unfit trial.
        if "out of memory" in str(e).lower():
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise optuna.TrialPruned(f"out of memory: {e}") from None
        raise
    finally:
        if os.path.exists(trial_model):   # survives only when the trial didn't promote it
            os.remove(trial_model)


def config_from_params(task, base, params: dict):
    """Reconstruct the exact config a trial evaluated from its recorded Optuna params. Routing
    them back through ``suggest`` (via a FixedTrial) reproduces the derived values too."""
    return task.suggest(optuna.trial.FixedTrial(params), base)


# --------------------------------------------------------------------------------------
# Study driver / CLI
# --------------------------------------------------------------------------------------

def parse_args(task, argv):
    p = argparse.ArgumentParser(
        prog=f"{task.name} optimize",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config", nargs="?", default="fetal-config.yaml",
                   help="base config; fixed fields inherited, searched fields overridden")
    p.add_argument("--trials", type=int, default=50, help="number of trials to run (default 50)")
    p.add_argument("--timeout", type=float, default=None,
                   help="stop the search after this many seconds (default: no limit)")
    p.add_argument("--epochs", type=int, default=None,
                   help="epochs per trial (default: the config's epochs); fewer = faster search")
    p.add_argument("--out-dir", default=None,
                   help="dir for best/latest config+model and the study db (default: model_dir)")
    p.add_argument("--storage", default=None,
                   help="Optuna storage URL (default: sqlite:///<out-dir>/study.db, which "
                        "resumes the search if it already exists)")
    p.add_argument("--study-name", default=None,
                   help="study name for resuming a study (default: the task name)")
    p.add_argument("--seed", type=int, default=None, help="sampler seed for reproducibility")
    return p.parse_args(argv)


def main(task, argv=None) -> None:
    if not task.supports_optimize:
        raise SystemExit(
            f"task {task.name!r} defines no search space (no suggest()/searched_fields()); "
            "nothing to optimize.")

    args = parse_args(task, argv)
    base = load_config(args.config, task)

    out_dir = args.out_dir or base.model_dir
    os.makedirs(out_dir, exist_ok=True)
    # Default to a SQLite study inside out_dir so a preempted search resumes on restart (and
    # the saved "best" isn't clobbered by a fresh study's first trial). f"sqlite:///{path}" is
    # right for both relative and absolute paths (absolute yields the required four slashes).
    storage = args.storage or f"sqlite:///{os.path.join(out_dir, STUDY_DB)}"

    print(f"Base config: '{args.config}'")
    print(f"Output dir:  '{out_dir}' (best/latest config+model, study db)")
    print(f"Storage:     {storage} (resumes if it already exists)")
    print(f"Trials: {args.trials}, epochs/trial: {args.epochs or base.train.epochs}")

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=args.seed),
        pruner=optuna.pruners.MedianPruner(),   # stops trials tracking worse than the median
        study_name=args.study_name or task.name,
        storage=storage,
        load_if_exists=True,
    )

    study.optimize(
        lambda trial: objective(trial, task, base, args.config, out_dir, epochs=args.epochs),
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

    # best-config.yaml / best-model.pt were already saved the moment the best trial ran;
    # re-emit the config from the recorded best params so it's present and correct after a resume.
    best_config_path = os.path.join(out_dir, BEST_CONFIG)
    best_config = config_from_params(task, base, study.best_params)
    write_config(args.config, best_config_path, task.searched_fields(best_config))
    print(f"\nBest   config + model: '{best_config_path}' + '{os.path.join(out_dir, BEST_MODEL)}'")
    print(f"Latest config + model: '{os.path.join(out_dir, LATEST_CONFIG)}' + "
          f"'{os.path.join(out_dir, LATEST_MODEL)}'")
    print(f"Train the final model (full epochs) with:\n  python src/main.py {best_config_path}")
