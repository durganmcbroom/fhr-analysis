BASEDIR=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$BASEDIR/../../.." && pwd)

poetry -P "$ROOT" run fhr-mine-failures "$BASEDIR/../models/funet-v36/config.yaml" --model-dir="$BASEDIR/../models/funet-v36"  --snippet-dir="$BASEDIR/../training/stereo_v13/" --out-dir="$BASEDIR/../training/stereo_v14(failed)/" --metric hr_agree --threshold 0.7
