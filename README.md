# fhr-analysis

Fetal heart rate (FHR) extraction and evaluation from Banner chest/abdomen
fiber sensor recordings. Several separation methods (ICA, MNMF, MLCMED, NMCF,
NeoSSNet-based source separation) are implemented as composable pipeline
stages and scored against microphone/PPG "sources of truth."

## Setup

```bash
./setup.sh
```

That installs Poetry if you don't have it (via `pipx` if present, otherwise the
official installer), initializes the git submodules, and runs `poetry install`,
which creates the virtualenv, installs the locked dependency set, and installs
this project into it in editable mode. It needs a Python >= 3.12 on `PATH`; if
it can't find one it prints every interpreter it did find, with versions and
paths, plus whatever virtualenv or conda env is active — usually the reason the
wrong one is being picked up.

3.12 is the real floor, not a preference: `numpy` and `scipy` are the strictest
direct dependencies, and no source file uses syntax newer than 3.10. It is kept
at the floor deliberately so the cluster's `module load miniforge` python
(3.12.x) satisfies it without a separate conda env. `setup.sh` reads the version
straight out of `requires-python`, so raising the floor there is the only change
needed — the check and its error messages follow.

If Poetry gets installed, it lands in `~/.local/bin`, which may not be on your
`PATH` — the script prints the line to add to your shell profile.

Everything runs through `poetry run`:

```bash
poetry run main                  # the analysis pipeline
poetry run funet-train lib/funet/fetal-config.yaml
```

or activate the environment once and drop the prefix:

```bash
eval $(poetry env activate)
```

The install must stay editable (it is, by default): `analyze.constants.PROJECT_DIR`
derives the repo root from `__file__`, and every model checkpoint and data path
hangs off it.

`pyproject.toml` declares the direct dependencies; `poetry.lock` pins the exact
resolution and is what `poetry install` reads. (`requirements.txt` is a leftover
pip lockfile from before the Poetry migration — nothing reads it anymore.)

## Layout

Source lives in `src/` and `lib/`, and the import name is always the directory
name. `poetry install` writes a `fhr_analysis.pth` listing the roots below, so
every package is importable by name from any working directory — no
`PYTHONPATH`, no `sys.path` juggling.

| Import name | On disk | What it is |
|---|---|---|
| `analyze` | `src/analyze/` | Core library: pipeline stages, detectors, scoring |
| `beat_app` | `src/beat_app/` | Local beat-marking web app |
| `rtmon` | `src/rtmon/` | Live rig: recording + real-time HR, served to a browser ([README](src/rtmon/README.md)) |
| `fhr_bin` | `src/fhr_bin/` | Standalone CLI utilities |
| `common` | `lib/common/` | Shared training engine: config, phases, optim, losses, io |
| `funet` | `lib/funet/funet/` | FUNet beat-activity model |
| `ssnet` | `lib/tune-ssnet/ssnet/` | NeoSSNet fine-tuning |
| `tslnet` | `lib/tslnet/tslnet/` | TSLNet: frozen TimesFM backbone + trainable head |
| `models`, `utils`, `loss_fn` | `lib/neossnet/` | Submodule, vendored unmodified — it uses bare imports internally, so these stay top-level |

```
src/
  analyze/           -> package `analyze`
    constants.py      Shared paths, sample rates, acoustic bands, BPM ranges.
    data.py           Audio / FiberData / FiberPair containers + raw loaders.
    filters.py        Bandpass / notch filter stages.
    pipeline.py       Pipeline class: stage chaining with content-hash caching.
    main.py           Entry point wiring stages into full analysis runs.
    hr/               Beat detectors (v1-v8) and source classification.
    ica.py, mnmf.py, mlcmed.py, nmcf.py, neossnet.py, funet_pipeline.py
                      Source separation / beat-activity methods. funet_pipeline
                      is deliberately not named funet.py -- see below.
    evaluate*.py, plot_hr.py
                      Scoring against the SOT and result plots.
  fhr_bin/           -> package `fhr_bin`
    generate_training_snippets.py   Build fine-tuning snippet sets.
    pico2data.py                    Convert PicoScope CSV exports.
    plots/            Waveform/clip plotting scripts.
    snr/, peak_det/   SNR and peak-detector comparison scripts.
  beat_app/          -> package `beat_app`
  rtmon/             -> package `rtmon`; the live rig — probed device sources, the
                        configurable processing matrix, streamed recording, and the
                        browser UI it all serves. Replaces the old PyQt5 app in
                        bin/record_with_realtime_tracking/.

lib/
  common/            -> package `common`; training loop + losses shared by every model.
  funet/             -> package `funet` (funet/), plus configs and checkpoints.
  tune-ssnet/        -> package `ssnet` (ssnet/), plus configs and checkpoints.
  tslnet/            -> package `tslnet` (tslnet/), plus its config and clips spec.
                      No backbone checkpoints of its own: the frozen TimesFM weights
                      come from the Hugging Face cache, and only the head is saved.
  neossnet/           Git submodule: base pretrained NeoSSNet model + code.
  beats/              Hand-marked mic beat times.

bin/record*/          Standalone recording apps. Separate venvs and separate,
                      conflicting dependency pins — not part of this install.
Banner_data/          Patient recording data (gitignored).
.out/                 Pipeline run outputs/cache (gitignored).
```

## Running

Every entry point is a console script, so it works from any directory:

| Command | Does |
|---|---|
| `poetry run main` | Run the analysis pipeline (`analyze/main.py`) |
| `poetry run funet-train [config.yaml]` | Train FUNet |
| `poetry run funet-optimize [config.yaml] [--trials N]` | Optuna search for FUNet |
| `poetry run ssnet-train [config.yaml]` | Fine-tune NeoSSNet |
| `poetry run ssnet-optimize [config.yaml] [--trials N]` | Optuna search for SSNet |
| `poetry run tslnet-train [config.yaml]` | Train TSLNet's head (backbone stays frozen) |
| `poetry run tslnet-optimize [config.yaml] [--trials N]` | Optuna search for TSLNet |
| `poetry run beat-app` | Serve the beat-marking web app |
| `poetry run rtmon` | Live rig: record the fibers and track fetal HR in real time |
| `poetry run rtmon-recover <session-dir>` | Finalize a recording whose server was killed |
| `poetry run fhr-snippets <clips.yaml> --out-dir out/` | Build training snippet sets |
| `poetry run fhr-pico2data` | Convert PicoScope CSV exports |
| `poetry run fhr-concat` | Concatenate two `.npy` recordings |
| `poetry run fhr-change-start` | Re-zero a recording's start time |
| `poetry run fhr-snr` | SNR plots from CSV data |
| `poetry run fhr-plot-clips <clips.yaml> --out-dir out/` | Plot the sections a clips yaml selects |
| `poetry run fhr-plot-waveforms <dir>... --out-dir out/` | Plot every channel of a recording |
| `poetry run fhr-peak-det` | Compare peak detectors |
| `poetry run fhr-beat-buildup [patient] [--source mic\|fiber] [--variant v2\|v4\|v7\|all]` | Build a detector up one step at a time |

Paths inside a config are resolved relative to that config file, not to the
working directory, so `poetry run funet-train lib/funet/fetal-config.yaml`
behaves the same from anywhere.

`analyze/main.py` selects which pipeline runs via the `PATIENT`/`WINDOW`
constants and the (mostly commented-out) calls in `main()`.

Anything without a console script runs as a module:

```bash
poetry run python -m analyze.evaluate_v3
```

### Cluster jobs

The long training and tuning runs live in `jobs/` as Slurm batch scripts.
`./batch.sh` lists them with the resources each one requests and the command it
runs, and submits the ones you pick:

```bash
./batch.sh                 # pick from the menu
./batch.sh train_funet     # submit by name, no prompt
./batch.sh -n all          # print the sbatch commands without submitting
```

It creates `logs/` before submitting — Slurm will not create a missing output
directory, and a job whose `-o` path is unwritable dies with nowhere to report
why. It also passes `--chdir` and `--job-name` on the command line, which take
precedence over the `#SBATCH` directives in the script: `--chdir` because the
scripts hardcode `~/dev/fhr-analysis` and sbatch does no tilde expansion, and
`--job-name` because the default is the filename *with* `.sh`, which would turn
the `%x` in the log pattern into `train_funet.sh_123.out`.

To submit one by hand instead, fix those two things yourself:

```bash
mkdir -p logs && sbatch --chdir="$PWD" --job-name=train_funet jobs/train_funet.sh
```

### One naming rule

`poetry install` puts the five source roots on `sys.path` (via
`fhr_analysis.pth`), but running a file as a script rather than with `-m` puts
that file's *own* directory at `sys.path[0]` — ahead of all of them. So a local
filename always wins over a package of the same name. A module in `src/analyze/`
named `funet.py` shadows the `funet` package and breaks
`from funet.config import ...` with a confusing `'funet' is not a package`.

So: **don't name a module or subpackage after one of the packages in the table
above** (`analyze`, `beat_app`, `rtmon`, `fhr_bin`, `common`, `funet`, `ssnet`,
`tslnet`, `models`, `utils`, `loss_fn`). Two renames exist for exactly this reason —
`analyze/funet_pipeline.py` (not `funet.py`) and `fhr_bin/plots/` (not
`fhr_bin/analyze/`). One dormant case is left, `src/analyze/hr/utils.py`, which
would shadow `utils` if anything in that directory were ever run as a script.
