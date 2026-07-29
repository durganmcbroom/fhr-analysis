"""Train FUNet.

    funet-train [fetal-config.yaml]

Everything below the config lives in common: see common.phases.train.run_training.
"""

import sys

from common.config import load_config
from common.phases.train import run_training

from funet.task import FUNetTask


def main(argv=None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] in ("-h", "--help"):
        print(__doc__)
        return
    config_path = argv[0] if argv else "fetal-config.yaml"

    task = FUNetTask()
    config = load_config(config_path, task)
    print(f"Loaded config: '{config_path}'")
    run_training(task, config)


if __name__ == "__main__":
    main()
