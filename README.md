# fhr-analysis

Fetal heart rate (FHR) extraction and evaluation from Banner chest/abdomen
fiber sensor recordings. Several separation methods (ICA, MNMF, MLCMED, NMCF,
NeoSSNet-based source separation) are implemented as composable pipeline
stages and scored against microphone/PPG "sources of truth."

## Setup

```bash
./setup.sh
```

This creates a `.venv`, installs `requirements.txt`, and initializes the git
submodule (`lib/neossnet`). Activate the environment with:

```bash
source .venv/bin/activate
```

## Layout

```
src/
  analyze/          Core library (package `analyze`), imported as
                     `from analyze.X import ...` with `src/` on PYTHONPATH.
    constants.py     Shared paths, sample rates, acoustic bands, BPM ranges.
    data.py          Audio / FiberData / FiberPair containers + raw loaders.
    filters.py       Bandpass / notch filter stages.
    pipeline.py      Pipeline class: stage chaining with content-hash caching.
    main.py          Entry point wiring stages into full analysis runs.
    hr/              Beat detectors (v1-v4) and source classification.
    ica.py, mnmf.py, mlcmed.py, nmcf.py, neossnet.py
                     Source separation methods.
    evaluate.py, evaluate_v2.py, plot_hr.py
                     Scoring against the SOT and result plots.
  bin/               Standalone CLI utilities (add `src/` to sys.path
                     themselves, then import from `analyze`).
    generate_training_snippets.py   Build NeoSSNet fine-tuning snippet sets.
    pico2data.py                    Convert PicoScope CSV exports.
    analyze/         Waveform/clip plotting scripts.
    snr/             SNR calculation scripts.

lib/
  neossnet/          Git submodule: base pretrained NeoSSNet model + code.
  tune-ssnet/        Fine-tuning configs/scripts and tuned model checkpoints.

Banner_data/         Patient recording data (gitignored).
.out/                Pipeline run outputs/cache (gitignored).
```

## Running

Most entry points expect to be run with `src/` on `PYTHONPATH` (or run
directly, since each script inserts it itself):

```bash
.venv/bin/python src/analyze/main.py
.venv/bin/python src/bin/generate_training_snippets.py <clips.yaml> --out-dir out/
```

`src/analyze/main.py` selects which pipeline runs via the `PATIENT`/`WINDOW`
constants and the (mostly commented-out) call at the bottom of the file.
