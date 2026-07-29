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
#lib/funet/generate_training_snippets.sh

poetry run funet-train lib/funet/fetal-config.yaml
#python3 lib/funet/src/tune.py lib/funet/fetal-config.yaml --trials=75
