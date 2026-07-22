BASEDIR=$(dirname $0)

source "$BASEDIR/../../.venv/bin/activate"

python3 "$BASEDIR/../../src/bin/generate_training_snippets.py" "$BASEDIR/training_clips.yaml" --out-dir="$BASEDIR/training/stereo_v10/" --no-plots