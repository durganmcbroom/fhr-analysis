"""Optuna hyperparameter search for SSNet.

    ssnet-tune [fetal-tune-config.yaml] [--trials N] [--epochs N] [--seed S]

The search space lives in SSNetTask.suggest / SSNetTask.searched_fields; the driver (trials,
pruning, study resume, best/latest config+model slots) lives in common.phases.optimize.
"""

import sys

from common.phases.optimize import main as optimize_main

from ssnet.task import SSNetTask


def main(argv=None) -> None:
    optimize_main(SSNetTask(), sys.argv[1:] if argv is None else argv)


if __name__ == "__main__":
    main()
