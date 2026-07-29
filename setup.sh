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

# One "<name> <version> <path>" line per interpreter, using "-" for a name that
# resolves to nothing. Versioned names are scanned off PATH rather than
# hardcoded, so a future python3.15 counts for just as much as python3.14 does.
survey_pythons() {
  local names p path ver dir f
  names=$(
    printf '%s\n' "$PATH" | tr ':' '\n' | while IFS= read -r dir; do
      [ -d "$dir" ] || continue
      for f in "$dir"/python3.[0-9] "$dir"/python3.[0-9][0-9]; do
        [ -x "$f" ] && basename "$f"
      done
    done | sort -u
    printf 'python3\npython\n'
  )
  for p in $names; do
    if ! path=$(command -v "$p" 2>/dev/null); then
      printf '%s - -\n' "$p"
      continue
    fi
    ver=$("$p" -c 'import sys; print("%d.%d.%d" % sys.version_info[:3])' 2>/dev/null) || ver="?"
    printf '%s %s %s\n' "$p" "$ver" "$path"
  done
}

# pyproject requires >=3.14; poetry needs to be able to find one to build the env.
PYTHON_OK=0
require_python() {
  local survey name ver path
  if [ "$PYTHON_OK" -eq 1 ]; then return 0; fi

  survey=$(survey_pythons)

  # The version is what matters, not the name -- python3 may well be the 3.14
  # while python3.14 does not exist. A non-numeric version ("-", "?") compares
  # false here, so unreadable interpreters are simply skipped.
  if printf '%s\n' "$survey" | awk '
       { split($2, v, ".")
         if (v[1] + 0 > 3 || (v[1] + 0 == 3 && v[2] + 0 >= 14)) { found = 1; exit } }
       END { exit(found ? 0 : 1) }'; then
    PYTHON_OK=1
    return 0
  fi

  {
    echo "No Python >= 3.14 found, which pyproject.toml requires."
    echo
    echo "Current Python environment:"
    # An active env is the usual reason a new enough python is installed but not
    # the one being picked up. On the cluster `module load miniforge` alone does it.
    if [ -n "${CONDA_PREFIX:-}" ]; then
      printf '  conda env    %s (%s)\n' "${CONDA_DEFAULT_ENV:-unnamed}" "$CONDA_PREFIX"
    fi
    if [ -n "${VIRTUAL_ENV:-}" ]; then
      printf '  virtualenv   %s\n' "$VIRTUAL_ENV"
    fi
    if [ -z "${CONDA_PREFIX:-}" ] && [ -z "${VIRTUAL_ENV:-}" ]; then
      printf '  no virtualenv or conda env active\n'
    fi
    echo
    printf '  %-13s %-9s %s\n' "INTERPRETER" "VERSION" "PATH"
    printf '%s\n' "$survey" | while read -r name ver path; do
      if [ "$path" = "-" ]; then
        printf '  %-13s %s\n' "$name" "not on PATH"
      else
        printf '  %-13s %-9s %s\n' "$name" "$ver" "$path"
      fi
    done
    echo
    echo "Install one:"
    echo "  macOS:    brew install python@3.14"
    echo "  cluster:  module avail python   (then load one that is >= 3.14)"
    echo "  else:     https://www.python.org/downloads/"
  } >&2
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
