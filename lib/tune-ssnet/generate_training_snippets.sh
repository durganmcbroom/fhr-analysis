BASEDIR=$(dirname $0)
echo $BASEDIR

python3 "$BASEDIR/../../src/bin/generate_training_snippets.py" "$BASEDIR/training_clips.yaml" --out-dir="$BASEDIR/training/training_clips_mono_v6/"