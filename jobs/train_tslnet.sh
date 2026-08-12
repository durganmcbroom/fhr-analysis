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

chmod a+x lib/tslnet/generate_training_snippets.sh
#lib/tslnet/generate_training_snippets.sh

# The TimesFM checkpoint is ~1.9 GB and is fetched once, then reused. Keep the cache beside
# the repo rather than in the default ~/.cache: a compute node whose home is not shared (or is
# wiped between allocations) otherwise re-downloads it on every single job. An HF_HOME already
# set in the environment wins, so this only supplies a default.
export HF_HOME="${HF_HOME:-$PWD/.hf-cache}"
mkdir -p "$HF_HOME"

poetry run tslnet-train lib/tslnet/fetal-config.yaml
