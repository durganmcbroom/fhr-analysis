#!/bin/bash

# Job Flags
#SBATCH -p mit_normal_gpu
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -G 1
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.out

module load miniforge

chmod a+x setup.sh
./setup.sh

chmod a+x lib/funet/generate_training_snippets.sh
chmod a+x lib/funet/failing/generate_failing_snippets.sh
#lib/funet/generate_training_snippets.sh
lib/funet/failing/generate_failing_snippets.sh

# --diagnostics writes snippet_diagnostics.png next to the checkpoints: the best model's
# activity against the target, and the BPM traces those produce, for a few validation
# snippets. Drop the flag to skip it (see common/diagnostics.py).
poetry run funet-train lib/funet/failing/failing-config.yaml --diagnostics
#python3 lib/funet/src/tune.py lib/funet/fetal-config.yaml --trials=75
