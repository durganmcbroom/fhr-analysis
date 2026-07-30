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

# The first run pulls the TimesFM checkpoint (~2 GB) into the Hugging Face cache. Point
# HF_HOME at shared storage if the compute node's home is small or not persisted.
poetry run tslnet-train lib/tslnet/fetal-config.yaml
