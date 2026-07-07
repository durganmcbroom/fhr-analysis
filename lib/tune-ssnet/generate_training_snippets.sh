BASEDIR=$(dirname $0)
echo $BASEDIR
python3 "$BASEDIR/../../src/bin/generate_training_snippets.py" training_clips.yaml --out-dir="training/training_clips_mono_v6/"