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
chmod a+x lib/funet/failing/generate_failing_snippets.sh
#lib/funet/generate_training_snippets.sh

# Five rounds of mine-then-train, back to back. Round 1 mines with v36 (the script's own
# default); every later round mines with the model the previous round just trained, so each
# pass re-asks "what does the current model still get wrong" rather than re-mining the same
# snippets five times. Both the snippet dir and the model dir are rewritten each round, so only
# the final model survives -- copy models/funet-v37(poor snippets) between rounds if you want
# to keep the intermediates.
ROUNDS=5
TRAINED="lib/funet/models/funet-v37(poor snippets)"

for round in $(seq 1 $ROUNDS); do
  echo "==================== round $round of $ROUNDS ===================="

  # Round 1 leaves MODEL_DIR unset so the generate script falls back to v36.
  if [ "$round" -gt 1 ]; then
    export MODEL_DIR="$TRAINED"
  fi

  # Stop rather than train on the previous round's snippets if mining fails.
  lib/funet/failing/generate_failing_snippets.sh || exit 1

  # --diagnostics writes snippet_diagnostics.png next to the checkpoints: the best model's
  # activity against the target, and the BPM traces those produce, for a few validation
  # snippets. Drop the flag to skip it (see common/diagnostics.py).
  poetry run funet-train lib/funet/failing/failing-config.yaml --diagnostics || exit 1
done
#python3 lib/funet/src/tune.py lib/funet/fetal-config.yaml --trials=75
