"""Shared training infrastructure for the models in lib/.

Every model under lib/ (funet, tune-ssnet, ...) implements a small ``Task`` (see
``common.task``) and gets the rest -- config loading, the training loop, atomic
checkpointing, curves, inference windowing and the Optuna search -- from here.

Three phases live in ``common.phases``:

    train      python <model>/src/main.py <config.yaml>
    optimize   python <model>/src/tune.py <config.yaml> [--trials N]
    inference  load_model(...) + run_windowed(...)

``optimize`` is optional. Nothing outside ``common.phases.optimize`` imports optuna,
so a model that never defines a search space still trains and runs inference; the
pruning seam is the plain ``on_epoch(epoch, val_loss)`` callback that ``engine.fit``
accepts, and feasibility checks raise ``common.errors.InfeasibleConfig`` rather than
``optuna.TrialPruned``.

Terminology: the split used to select checkpoints and drive the search is the
*validation* set (``val_*``) throughout. ``test`` is reserved for a genuinely
held-out split that is scored once, which no model here currently has.
"""
