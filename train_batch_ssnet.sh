#!/bin/bash

# Job Flags
#SBATCH -p mit_normal_gpu
#SBATCH -c 12
#SBATCH --mem=32G
#SBATCH -G 1

module load miniforge

chmod a+x setup.sh
./setup.sh

source .venv/bin/activate

#lib/tune-ssnet/generate_training_snippets.sh

python3 lib/tune-ssnet/src/main.py lib/tune-ssnet/fetal-tune-config.yaml
