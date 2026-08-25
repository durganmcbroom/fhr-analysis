"""Train PALNet.

    palnet-train [fetal-config.yaml]

By default only the head trains; the PANNs ResNet22 backbone is frozen (see palnet.model, and
``model.freeze`` to change that). The first run downloads the checkpoint named in
``config.model.checkpoint`` -- about 259 MB -- into the Hugging Face cache; later runs reuse it.

Everything below the config lives in common: see common.phases.train.run_training.
"""

import sys

from common.config import load_config
from common.phases.train import run_training

from palnet.task import PALNetTask


def main(argv=None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] in ("-h", "--help"):
        print(__doc__)
        return
    config_path = argv[0] if argv else "fetal-config.yaml"

    task = PALNetTask()
    config = load_config(config_path, task)
    print(f"Loaded config: '{config_path}'")
    run_training(task, config)


if __name__ == "__main__":
    main()
