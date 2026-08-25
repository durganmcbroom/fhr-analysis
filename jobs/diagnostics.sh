#!/bin/bash

# Job Flags
#SBATCH -p mit_normal_gpu
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -G 1
#SBATCH -t 01:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.out

# Draw the diagnostic figure for already-trained models, one per patient. No training happens
# here -- each run loads a checkpoint and redraws -- so this is minutes, not hours.
#
# Scoring is against the microphone SOT: hand-marked mic_beats.npy when the patient has one,
# otherwise the v7 detector on the band-limited mic. See lib/common/diagnostics.py.

module load miniforge

chmod a+x setup.sh
./setup.sh

OUT=".out/diagnostics"
PATIENTS="Banner_data/Banner_test_20251220"

# Rows per figure. Unset (or empty) draws the whole recording as a single row; a value splits
# it into rows that long, which is what makes a long recording readable.
WINDOW=60

FUNET_CONFIG="lib/funet/models/funet-v36/config.yaml"
FUNET_MODEL_DIR="lib/funet/models/funet-v36"
FUNET_PATIENTS=(PT13_1 PT14_1)

SSNET_CONFIG="lib/tune-ssnet/fetal-tune-config.yaml"
SSNET_MODEL_DIR="lib/tune-ssnet/models/tuned-model-v14"
SSNET_PATIENTS=(PT12_1 PT13_1 PT14_1)

# Each figure is independent, so one unreadable patient or missing checkpoint should not take
# the rest of the sweep down -- report and carry on.
diagnose() {
  local task=$1 config=$2 model_dir=$3 patient=$4
  local out="$OUT/$task-$patient.png"
  echo "---- $task: $patient -> $out"
  poetry run fhr-diagnose --task "$task" "$config" \
      --model-dir "$model_dir" --patient-dir "$PATIENTS/$patient" --out "$out" \
      ${WINDOW:+--window "$WINDOW"} \
      || echo "!! $task diagnostics failed for $patient (continuing)"
}

for p in "${FUNET_PATIENTS[@]}"; do
  diagnose funet "$FUNET_CONFIG" "$FUNET_MODEL_DIR" "$p"
done

for p in "${SSNET_PATIENTS[@]}"; do
  diagnose ssnet "$SSNET_CONFIG" "$SSNET_MODEL_DIR" "$p"
done

echo "Figures in $OUT"
