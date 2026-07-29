"""Optuna hyperparameter search for FUNet.

    funet-optimize [fetal-config.yaml] [--trials N] [--epochs N] [--storage URL] [--seed S]

The search space lives in FUNetTask.suggest / FUNetTask.searched_fields; the driver (trials,
pruning, study resume, best/latest config+model slots) lives in common.phases.optimize, which
is the only place optuna is imported.
"""

import sys

from common.phases.optimize import main as optimize_main

from funet.task import FUNetTask


def main(argv=None) -> None:
    optimize_main(FUNetTask(), sys.argv[1:] if argv is None else argv)


if __name__ == "__main__":
    main()
