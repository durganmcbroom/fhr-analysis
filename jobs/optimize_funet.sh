#!/bin/bash

# Job Flags
#SBATCH -p mit_normal_gpu
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -G 1
#SBATCH -t 05:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.out

module load miniforge

chmod a+x setup.sh
./setup.sh

chmod a+x lib/funet/generate_training_snippets.sh
#lib/funet/generate_training_snippets.sh

# --objective hr_corr ranks trials by the Pearson r between the predicted and target BPM
# traces at the epoch validation loss selected, instead of by the loss itself. Without it the
# search ranks on validation loss (the default). Either way each trial still picks its own
# checkpoint by validation loss, and the per-epoch "HR r" is logged regardless -- so the log
# shows whether loss and r agree even on a plain loss-ranked run.
poetry run funet-optimize lib/funet/fetal-config.yaml --trials=100 --objective hr_corr
