BASEDIR=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$BASEDIR/../../.." && pwd)
BANNER_DATA="$ROOT/Banner_data/Banner_test_20251220"

poetry -P "$ROOT" run fhr-plot-waveforms "$BANNER_DATA/Patient 6" "$BANNER_DATA/Patient 7" "$BANNER_DATA/patient8-session1" --out-dir="$BASEDIR/out"
#poetry -P "$ROOT" run fhr-plot-waveforms "$BANNER_DATA/patient8-session1" "$BANNER_DATA/Patient 7" --out-dir="$BASEDIR/out"
