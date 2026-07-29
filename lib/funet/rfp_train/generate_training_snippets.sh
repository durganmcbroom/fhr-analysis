BASEDIR=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$BASEDIR/../../.." && pwd)

poetry -P "$ROOT" run fhr-snippets "$BASEDIR/rough_pass_training_clips.yaml" --out-dir="$BASEDIR/training/rough_v1/"
poetry -P "$ROOT" run fhr-snippets "$BASEDIR/fine_pass_training_clips.yaml" --out-dir="$BASEDIR/training/fine_v1/"
