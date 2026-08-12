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

# --objective hr_delta ranks trials by the median bpm error between the predicted and target
# HR traces, read at the epoch validation loss selected, and minimises it. Without the flag the
# search ranks on validation loss instead. Either way each trial still picks its own checkpoint
# by validation loss. ('hr_agree' scores the same comparison as a within-tolerance fraction;
# 'hr_corr' is the old Pearson r, kept only so earlier runs stay reproducible -- over a single
# snippet the rate is nearly flat, so r ends up decided by the worst beat or two.)
poetry run funet-optimize lib/funet/fetal-config.yaml --trials=100 --objective hr_delta
