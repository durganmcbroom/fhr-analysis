BASEDIR=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$BASEDIR/../.." && pwd)

poetry -P "$ROOT" run fhr-snippets "$BASEDIR/training_clips.yaml" --out-dir="$BASEDIR/training/stereo_v1/" --no-plots
