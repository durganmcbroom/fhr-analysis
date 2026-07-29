"""Train (fine-tune) SSNet.

    ssnet-train [fetal-tune-config.yaml]

Everything below the config lives in common: see common.phases.train.run_training.
"""

import sys

from common.config import load_config
from common.phases.train import run_training

from ssnet.task import SSNetTask


def main(argv=None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] in ("-h", "--help"):
        print(__doc__)
        return
    config_path = argv[0] if argv else "fetal-tune-config.yaml"

    task = SSNetTask()
    config = load_config(config_path, task)
    print(f"Loaded config: '{config_path}'")
    run_training(task, config)


if __name__ == "__main__":
    main()
