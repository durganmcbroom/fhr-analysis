#!/bin/bash

# Job Flags
#SBATCH -p mit_normal_gpu
#SBATCH -c 12
#SBATCH --mem=32G
#SBATCH -G 1
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.out

module load miniforge

chmod a+x setup.sh
./setup.sh

chmod a+x lib/tune-ssnet/generate_training_snippets.sh
lib/tune-ssnet/generate_training_snippets.sh

poetry run ssnet-train lib/tune-ssnet/fetal-tune-config.yaml
