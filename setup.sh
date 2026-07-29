#!/usr/bin/env bash
set -e
set -o pipefail

ROOT=$(cd "$(dirname "$0")" && pwd)

# Both pipx and Poetry's official installer put the launcher here, and it is not
# on PATH by default on a fresh machine.
LOCAL_BIN="$HOME/.local/bin"

have_poetry() {
  command -v poetry >/dev/null 2>&1
}

# pyproject requires >=3.14; poetry needs to be able to find one to build the env.
require_python() {
  local p
  for p in python3.14 python3 python; do
    if command -v "$p" >/dev/null 2>&1 && "$p" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 14) else 1)' 2>/dev/null; then
      return 0
    fi
  done
  echo "No Python >= 3.14 found, which pyproject.toml requires." >&2
  echo "  macOS:  brew install python@3.14" >&2
  echo "  else:   https://www.python.org/downloads/" >&2
  exit 1
}

install_poetry() {
  echo "poetry not found -- installing it"
  if command -v pipx >/dev/null 2>&1; then
    # Preferred: pipx keeps poetry in its own venv and manages the launcher.
    pipx install poetry
  else
    # Poetry's official installer. Same thing pipx does (isolated venv under
    # ~/.local), without needing pipx present first.
    curl -sSL https://install.python-poetry.org | python3 -
  fi
}

# A poetry installed by a previous run of this script won't be on PATH in a
# fresh shell, so look in ~/.local/bin before deciding it is missing.
if ! have_poetry && [[ -x "$LOCAL_BIN/poetry" ]]; then
  export PATH="$LOCAL_BIN:$PATH"
fi

if ! have_poetry; then
  require_python
  install_poetry
  export PATH="$LOCAL_BIN:$PATH"
  if ! have_poetry; then
    echo "poetry was installed but is still not on PATH." >&2
    echo "Add this to your shell profile and re-run:  export PATH=\"$LOCAL_BIN:\$PATH\"" >&2
    exit 1
  fi
  echo "Installed $(poetry --version)"
  echo "Add this to your shell profile so poetry stays on PATH:"
  echo "  export PATH=\"$LOCAL_BIN:\$PATH\""
  echo
fi

require_python

# Submodules first: pyproject.toml declares lib/neossnet's models/utils/loss_fn
# as packages of this project, so the install below fails if it isn't checked out.
git -C "$ROOT" submodule update --init --recursive

# Installs the locked dependency set and the project itself, editable.
# It has to stay editable: analyze.constants.PROJECT_DIR derives the repo root
# from __file__, and every model checkpoint and data path hangs off it.
poetry -P "$ROOT" install

echo
echo "Done. Run things with:"
echo "  poetry run main                      # the analysis pipeline"
echo "  poetry run funet-train <config.yaml> # train FUNet"
echo "  eval \$(poetry env activate)          # or activate the env in this shell"
