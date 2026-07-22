BASEDIR=$(dirname $0)

source "$BASEDIR/../../../.venv/bin/activate"

python3 "$BASEDIR/../../../src/bin/generate_training_snippets.py" "$BASEDIR/rough_pass_training_clips.yaml" --out-dir="$BASEDIR/training/rough_v1/"
python3 "$BASEDIR/../../../src/bin/generate_training_snippets.py" "$BASEDIR/fine_pass_training_clips.yaml" --out-dir="$BASEDIR/training/fine_v1/"
