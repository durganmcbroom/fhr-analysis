VENV_DIR=".venv"
BANNER_DATA_DIR="$PWD/Banner_data/Banner_test_20251220/"

if [[ ! -d "$VENV_DIR" ]]; then
  echo "Creating virtual environment in ./$VENV_DIR"
  python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

pip install --upgrade pip
pip install -r requirements.txt

git submodule update --init --recursive