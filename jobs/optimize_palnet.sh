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

export HF_HOME="${HF_HOME:-$PWD/.hf-cache}"
mkdir -p "$HF_HOME"

# model_hz / hop / feature_layer are declared in PALNetTask.loss_scale_fields -- they change
# how many frames the loss averages over, so the optimize phase refuses to rank trials by loss
# while they move. hr_agree is measured in bpm against seconds and does not.
poetry run palnet-optimize lib/palnet/fetal-config.yaml --trials 60 --epochs 20 --objective hr_agree
