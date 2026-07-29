#!/bin/bash

# Job Flags
#SBATCH -p mit_normal_gpu
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -G 1

module load miniforge

chmod a+x setup.sh
./setup.sh

source .venv/bin/activate


chmod a+x lib/funet/generate_training_snippets.sh
#lib/funet/generate_training_snippets.sh

python3 lib/funet/src/tune.py lib/funet/fetal-config.yaml --trials=75
