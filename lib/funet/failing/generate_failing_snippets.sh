set -e

BASEDIR=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$BASEDIR/../../.." && pwd)

# Which model decides what "failing" means. Defaults to v36 so running this by hand behaves as
# before; the training job overrides it each round so later rounds mine with the model that
# round just produced.
MODEL_DIR="${MODEL_DIR:-$BASEDIR/../models/funet-v36}"

OUT="$BASEDIR/../training/stereo_v14(failed)"

# Wipe the previous round's snippets first. fhr-mine-failures copies into the directory and
# never clears it, so without this a snippet that failed last round but passes now would linger
# and the "failing set" would only ever grow.
rm -rf "$OUT/fetal-train" "$OUT/fetal-test"

poetry -P "$ROOT" run fhr-mine-failures "$MODEL_DIR/config.yaml" --model-dir="$MODEL_DIR"  --snippet-dir="$BASEDIR/../training/stereo_v13/fetal-train" --out-dir="$OUT/fetal-train" --metric hr_agree --threshold 0.7
poetry -P "$ROOT" run fhr-mine-failures "$MODEL_DIR/config.yaml" --model-dir="$MODEL_DIR"  --snippet-dir="$BASEDIR/../training/stereo_v13/fetal-test" --out-dir="$OUT/fetal-test" --metric hr_agree --threshold 0.7
