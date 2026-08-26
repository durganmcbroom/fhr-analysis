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

chmod a+x lib/palnet/generate_training_snippets.sh
lib/palnet/generate_training_snippets.sh

# The PANNs ResNet22 checkpoint is ~259 MB and is fetched once, then reused. Keep the cache
# beside the repo rather than in the default ~/.cache: a compute node whose home is not shared
# (or is wiped between allocations) otherwise re-downloads it on every single job. An HF_HOME
# already set in the environment wins, so this only supplies a default.
export HF_HOME="${HF_HOME:-$PWD/.hf-cache}"
mkdir -p "$HF_HOME"

# --diagnostics writes snippet_diagnostics.png next to the checkpoints, one row per validation
# snippet: the 64 log-mel bins the frozen backbone actually sees, the model's activity against
# the target, the BPM traces those produce, and the cardiac period the beat detector locked
# onto. Drop the flag to skip it (see common/diagnostics.py).
#
# The mel column is the one to look at first for this model. PALNet does not own its front-end
# -- the mel filterbank is a tensor in the AudioSet checkpoint -- so whether a fetal beat is
# visible in those 64 bins at all is the question the whole approach rests on.
poetry run palnet-train lib/palnet/fetal-config.yaml --diagnostics
