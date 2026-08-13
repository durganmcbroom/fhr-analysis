BASEDIR=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$BASEDIR/../../.." && pwd)

poetry -P "$ROOT" run fhr-mine-failures "$BASEDIR/../models/funet-v36/config.yaml" --model-dir="$BASEDIR/../models/funet-v36"  --snippet-dir="$BASEDIR/../training/stereo_v13/fetal-train" --out-dir="$BASEDIR/../training/stereo_v14(failed)/fetal-train" --metric hr_agree --threshold 0.7
poetry -P "$ROOT" run fhr-mine-failures "$BASEDIR/../models/funet-v36/config.yaml" --model-dir="$BASEDIR/../models/funet-v36"  --snippet-dir="$BASEDIR/../training/stereo_v13/fetal-test" --out-dir="$BASEDIR/../training/stereo_v14(failed)/fetal-test" --metric hr_agree --threshold 0.7
