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
the numbers. What trials are ranked by is chosen with ``--objective`` (see OBJECTIVES); the
study always minimises, and any higher-is-better objective is negated in one place,
``_trial_value``.

This is the ONLY module in common that imports optuna. Tasks never do: their ``suggest``
receives a trial object structurally, and ``check_feasible`` raises ``InfeasibleConfig``,
which this module translates into a pruned trial.
"""

import argparse
import os
from typing import Optional

import numpy as np
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

# A trial is scored by the mean validation loss over its final SCORE_WINDOW epochs, and that
# same trailing mean is what is reported to the pruner. The raw per-epoch val loss is noisy, so
# the single best epoch is an over-optimistic, high-variance objective (a lucky one-epoch dip
# could win the search); a trailing mean is a stabler estimate of where a config settled.
SCORE_WINDOW = 5

#: What a trial can be scored by (``--objective``), all evaluated at the epoch validation loss
#: selected -- the checkpoint the trial would actually hand you:
#:   'loss'      trailing-mean validation loss
#:   'hr_delta'  median |predicted - reference| in bpm. The one to use: it is the error in the
#:               heart-rate plot, in the units you read it in, and lower is better.
#:   'hr_agree'  fraction of the predicted HR trace within tolerance. The same question phrased
#:               as a bounded 0-1 score; useful when a hard tolerance is what you care about.
#:   'hr_corr'   Pearson r between the traces. Kept so earlier runs stay reproducible, but on a
#:               single short snippet the rate is nearly constant, so r ends up decided by the
#:               worst one or two beats -- see common.metrics.
#: A search-phase choice, so it lives on the CLI next to --trials/--epochs/--seed rather than
#: in the config: it changes how trials are ranked, never how any individual model trains.
OBJECTIVES = ("loss", "hr_delta", "hr_agree", "hr_corr")

#: Objectives needing the HR metric measured.
HR_OBJECTIVES = ("hr_delta", "hr_agree", "hr_corr")

#: Of those, the higher-is-better ones. The study always minimises, so these are negated on the
#: way in and un-negated when reported. 'hr_delta' is absent: it is an error in bpm, already
#: lower-is-better, so it is handed to optuna as-is.
MAXIMISED_OBJECTIVES = ("hr_agree", "hr_corr")


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


def _dotted(config, key: str):
    """Read a dotted config key, e.g. ``"model.freq_crop_hz"``."""
    obj = config
    for name in key.split("."):
        obj = getattr(obj, name)
    return obj


def check_frozen(task, base, config, searched: dict) -> None:
    """Raise if this trial reached a field the task declared frozen (``Task.frozen_fields``).

    A search has exactly two ways to move such a field, and both are checked here:
    ``suggest`` assigning it, and ``searched_fields`` listing it -- the latter matters because
    ``write_config`` overlays that dict onto the emitted YAML, so a frozen field named there
    would be written into best-config.yaml even if the trial itself ran with the base value.

    This is a programming error in the task's search space, not an unlucky sample, so it
    raises rather than pruning: pruning would bury it as a run of skipped trials.
    """
    for key in task.frozen_fields:
        want, got = _dotted(base, key), _dotted(config, key)
        if got != want:
            raise RuntimeError(
                f"{task.name}.suggest changed frozen field {key!r}: base has {want!r}, the "
                f"trial config has {got!r}. Frozen fields are inherited from the base config; "
                f"remove the assignment or drop {key!r} from {type(task).__name__}.frozen_fields.")
        section, _, field = key.rpartition(".")
        if field in (searched.get(section) or {}):
            raise RuntimeError(
                f"{task.name}.searched_fields lists frozen field {key!r}, which would be "
                f"written into the emitted config. Remove it, or drop {key!r} from "
                f"{type(task).__name__}.frozen_fields.")


def _trial_value(objective_name: str, result, val_history: list) -> tuple:
    """The number Optuna minimises for a finished trial, plus a line describing it.

    The study is always ``direction="minimize"``, so every objective is turned into a
    lower-is-better number here and nowhere else. 'hr_delta' already is one -- it is a bpm
    error -- so it passes through untouched; the higher-is-better ones are negated.

    The HR objectives read the score *at the epoch validation loss selected*, i.e. the
    checkpoint the trial would actually hand you. 'loss' uses the trailing mean of the
    validation loss, stabler than its single best epoch on a noisy curve (see SCORE_WINDOW).
    """
    if objective_name in HR_OBJECTIVES:
        score = result.best_score
        if score is None:
            raise optuna.TrialPruned(
                f"objective is {objective_name!r} but no HR metric was measured -- this task "
                "provides no scorer (Task.make_val_scorer), or every epoch was cut short")

        if objective_name == "hr_delta":
            # nan when every snippet was degenerate: there is no bpm error to speak of, and
            # letting nan reach optuna would silently never compare greater than anything.
            if not np.isfinite(score.median_delta):
                raise optuna.TrialPruned(
                    "every validation snippet was degenerate, so there is no bpm error to score")
            return score.median_delta, f"HR {score} @ epoch {result.best_epoch + 1}"

        value = score.within_tol if objective_name == "hr_agree" else score.corr
        return -value, f"HR {score} @ epoch {result.best_epoch + 1}"

    window = val_history[-SCORE_WINDOW:]
    value = sum(window) / len(window) if window else float("inf")
    return value, f"val loss {value:.6f} (mean of last {len(window)} epochs)"


def check_loss_comparable(task, base, config, objective_name: str) -> None:
    """Refuse to rank trials by loss when the search moves a loss-scale field.

    ``Task.loss_scale_fields`` names the settings that change what the loss measures rather
    than how well the model does -- FUNet's spectrogram geometry, which sets the frame grid the
    target comb is built on. Comparing losses across trials that differ there ranks the units,
    not the models, and it is not a subtle effect: an all-zeros FUNet scores 0.116 at hop 64
    against 0.139 at hop 256.

    Raised rather than pruned, and on the first trial that moves the field, because it is a
    misconfigured run: every trial after it would be just as meaningless, and the failure it
    prevents is silent -- a search that finishes cleanly and reports a winner that is worse.
    """
    if objective_name != "loss":
        return
    moved = [key for key in task.loss_scale_fields
             if _dotted(config, key) != _dotted(base, key)]
    if moved:
        raise RuntimeError(
            f"{task.name}.suggest varies {moved}, which change the scale of the loss, but this "
            f"search is ranking trials by loss -- so it would reward whichever trial picked the "
            f"cheapest units rather than the best model. Re-run with '--objective hr_corr' "
            f"(scale-free: it scores BPM traces in bpm against seconds), or stop searching "
            f"{moved} by adding them to {type(task).__name__}.frozen_fields.")


def objective(trial: optuna.Trial, task, base, base_config_path: str, out_dir: str,
              epochs: Optional[int] = None, objective_name: str = "loss") -> float:
    """Train one sampled config, persist it as the latest (and best, if it wins), and return
    its best validation loss (lower is better)."""
    config = task.suggest(trial, base)
    # Before anything is built or trained, so a search space that reaches a frozen field fails
    # on trial 1 rather than after a night of runs that quietly swept it.
    searched = task.searched_fields(config)
    check_frozen(task, base, config, searched)
    check_loss_comparable(task, base, config, objective_name)

    try:
        task.check_feasible(config)
    except InfeasibleConfig as e:
        # Skipping is cleaner than letting the model constructor or forward pass raise mid-run.
        raise optuna.TrialPruned(str(e)) from None

    if epochs is not None:
        config.train.epochs = epochs   # short trials; the emitted best config keeps base epochs

    val_history: list[float] = []

    def on_epoch(epoch: int, val_loss: float) -> None:
        val_history.append(val_loss)
        # Pruning always watches the validation loss, even when the trial is *scored* on HR
        # correlation: loss is dense and informative from epoch 1, whereas a model that has not
        # yet learned to produce beats scores exactly 0 correlation for many epochs, which
        # gives the pruner a flat line it cannot rank. Report the trailing mean, not the raw
        # epoch loss, so a single noisy epoch does not trip it.
        window = val_history[-SCORE_WINDOW:]
        trial.report(sum(window) / len(window), epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    # The trial's best-epoch weights are written here first; only a fully completed trial
    # promotes them into the shared latest/best slots, so a pruned or crashed trial can never
    # leave latest-model.pt out of sync with latest-config.yaml. The finally cleans up the
    # temp file for any trial that doesn't promote it.
    trial_model = os.path.join(out_dir, f".trial-{trial.number}.pt")
    try:
        # Force the metric on when it is the objective, so a search never depends on the config
        # remembering to enable it; None otherwise, leaving train.measure_hr to decide.
        result = run_training(task, config, on_epoch=on_epoch, save_artifacts=False,
                              best_model_path=trial_model,
                              measure_hr=True if objective_name in HR_OBJECTIVES else None)
        value, note = _trial_value(objective_name, result, val_history)

        os.replace(trial_model, os.path.join(out_dir, LATEST_MODEL))
        write_config(base_config_path, os.path.join(out_dir, LATEST_CONFIG), searched)
        if value < _best_value_so_far(trial.study):
            atomic_copy(os.path.join(out_dir, LATEST_MODEL), os.path.join(out_dir, BEST_MODEL))
            write_config(base_config_path, os.path.join(out_dir, BEST_CONFIG), searched)
            print(f"  new best: {note} -> saved {BEST_CONFIG} + {BEST_MODEL}")
        return value
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

def _enqueue_baseline(study: optuna.Study, task, base) -> None:
    """Queue the base config itself as a trial, so the search has an anchor to beat.

    Without this the study reports whichever sampled config won *among the sampled configs*,
    with no evidence it beats the hand-tuned one it started from -- exactly how funet-v33
    returned a "best" model that was worse in practice. Running the base config under the same
    objective makes the comparison direct and puts a known-good point in TPE's model.

    ``skip_if_exists`` keeps a resumed study from re-running the anchor on every restart.
    """
    params = task.baseline_params(base)
    if params is None:
        return

    # Rebuild the config from these params before queueing: FixedTrial raises on any parameter
    # `suggest` asks for that `baseline_params` omits, so drift between the two surfaces here
    # rather than as an "anchor" trial that silently sampled the missing dimensions at random.
    try:
        config_from_params(task, base, params)
    except Exception as e:
        raise RuntimeError(
            f"{type(task).__name__}.baseline_params does not reproduce the base config under "
            f"suggest(): {e}. Every parameter suggest() requests must appear there.") from None

    study.enqueue_trial(params, skip_if_exists=True)
    print("Base config queued as the anchor trial -- the score every other trial must beat "
          "(skipped if this study already ran it).")


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
    p.add_argument("--objective", choices=OBJECTIVES, default="loss",
                   help="what to rank trials by: 'loss' = trailing-mean validation loss "
                        "(default); 'hr_delta' = median bpm error of the HR trace, minimised; "
                        "'hr_agree' = fraction of it within tolerance, maximised; 'hr_corr' = "
                        "Pearson r, maximised. The HR ones are all read at the loss-selected "
                        "epoch. Does not change how any model trains.")
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

    # Before any training: a search that cannot compute its own objective should say so now,
    # not after the first trial has run to completion.
    if args.objective in HR_OBJECTIVES and task.make_val_scorer(base) is None:
        raise SystemExit(
            f"--objective {args.objective} needs a validation scorer, but task {task.name!r} "
            f"provides "
            f"none (Task.make_val_scorer returned None). Use '--objective loss', or implement "
            f"the hook for this task.")

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
    direction = "maximised" if args.objective in MAXIMISED_OBJECTIVES else "minimised"
    print(f"Objective: {args.objective} ({direction})"
          " -- within a trial the checkpoint is still selected by validation loss")

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=args.seed),
        # Prune trials tracking worse than the median, but only after each has had a fair chance:
        # the val curve sits on a near-flat plateau for its first ~20-25 epochs before it breaks
        # through, and pruning during that plateau kills good-but-slow-starting configs on epoch-1
        # noise (the funet-v33 run pruned 65/75 trials this way). n_warmup_steps holds pruning off
        # until epoch 30; n_startup_trials leaves TPE's random-exploration trials whole.
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=10, n_warmup_steps=20, interval_steps=5),
        study_name=args.study_name or task.name,
        storage=storage,
        load_if_exists=True,
    )

    _enqueue_baseline(study, task, base)

    study.optimize(
        lambda trial: objective(trial, task, base, args.config, out_dir,
                                epochs=args.epochs, objective_name=args.objective),
        n_trials=args.trials,
        timeout=args.timeout,
    )

    print("\n===== Search complete =====")
    print(f"Completed trials: {len(study.get_trials(states=(optuna.trial.TrialState.COMPLETE,)))}"
          f" / {len(study.trials)}")
    # study.best_value is what optuna minimised: the objective itself for lower-is-better ones
    # (loss, hr_delta), and its negation for the maximised ones.
    if args.objective in MAXIMISED_OBJECTIVES:
        print(f"Best {args.objective}: {-study.best_value:.4f}")
    elif args.objective == "hr_delta":
        print(f"Best {args.objective}: {study.best_value:.2f} bpm")
    else:
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
