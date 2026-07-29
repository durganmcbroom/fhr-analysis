BASEDIR=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$BASEDIR/../../.." && pwd)

poetry -P "$ROOT" run fhr-plot-clips "$ROOT/lib/tune-ssnet/training_clips.yaml" --out-dir="$BASEDIR/out/plotted-clips/"
