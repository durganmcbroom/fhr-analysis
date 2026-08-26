"""Train PALNet.

    palnet-train [fetal-config.yaml] [--diagnostics]

By default only the head trains; the PANNs ResNet22 backbone is frozen (see palnet.model, and
``model.freeze`` to change that). The first run downloads the checkpoint named in
``config.model.checkpoint`` -- about 259 MB -- into the Hugging Face cache; later runs reuse it.

Everything below the config lives in common: see common.phases.train.run_training.
"""

import argparse
import os
import sys

from common.config import load_config
from common.phases.train import BEST_MODEL, run_training

from palnet.task import PALNetTask

DIAGNOSTICS_PLOT = "snippet_diagnostics.png"


def parse_args(argv):
    p = argparse.ArgumentParser(prog="palnet-train", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config", nargs="?", default="fetal-config.yaml",
                   help="config to train (default: fetal-config.yaml)")
    # --- diagnostics (optional; delete this argument and the block in main to remove) ---
    p.add_argument("--diagnostics", action="store_true",
                   help="after training, plot the best model against the validation snippets: "
                        "the log-mel the frozen backbone actually sees, the activity against "
                        "the target, the BPM traces those produce, and the cardiac period the "
                        "beat detector locked onto. Writes " + DIAGNOSTICS_PLOT + " next to "
                        "the checkpoints. Costs a forward pass per snippet, so it wants the "
                        "GPU the training run already has -- see the note in main().")
    # --- end diagnostics ---
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(sys.argv[1:] if argv is None else argv)

    task = PALNetTask()
    config = load_config(args.config, task)
    print(f"Loaded config: '{args.config}'")
    run_training(task, config)

    # --- diagnostics (optional; delete this block and the CLI argument to remove) ---
    # Runs the validation split a batch at a time through the real inference path, which for
    # PALNet means the 99.2%-overlap STFT and 59.5M frozen params per snippet. That is fine on
    # the GPU this job already holds and is not fine on a laptop: a 4.096 s crop peaks near
    # 3 GB on CPU, so `PALNET_DEVICE=cpu palnet-train --diagnostics` will thrash or be killed.
    # Use `fhr-diagnose --task palnet ... --window 8 --snippets N` there instead.
    if args.diagnostics:
        from common.diagnostics import plot_snippet_diagnostics
        plot_snippet_diagnostics(
            task, config,
            checkpoint_path=os.path.join(config.model_dir, BEST_MODEL),
            out_path=os.path.join(config.model_dir, DIAGNOSTICS_PLOT),
        )
    # --- end diagnostics ---


if __name__ == "__main__":
    main()
