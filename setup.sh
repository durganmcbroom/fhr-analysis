#!/usr/bin/env bash
set -e
set -o pipefail

ROOT=$(cd "$(dirname "$0")" && pwd)

if ! command -v poetry >/dev/null 2>&1; then
  echo "poetry not found -- install it first: https://python-poetry.org/docs/#installation" >&2
  exit 1
fi

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
