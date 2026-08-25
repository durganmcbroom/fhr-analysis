"""Optuna hyperparameter search for PALNet.

    palnet-optimize [fetal-config.yaml] [--trials N] [--epochs N] [--storage URL] [--seed S]

The search space lives in PALNetTask.suggest / PALNetTask.searched_fields; the driver (trials,
pruning, study resume, best/latest config+model slots) lives in common.phases.optimize, which
is the only place optuna is imported.

Note that ``model_hz``, ``hop`` and ``feature_layer`` are declared in
``PALNetTask.loss_scale_fields``: they change how many frames the loss averages over, so the
optimize phase will refuse to rank trials by loss while they are being searched. Rank by
``--objective hr_agree`` instead, which is measured in bpm against seconds and does not move
with the frame rate.
"""

import sys

from common.phases.optimize import main as optimize_main

from palnet.task import PALNetTask


def main(argv=None) -> None:
    optimize_main(PALNetTask(), sys.argv[1:] if argv is None else argv)


if __name__ == "__main__":
    main()
