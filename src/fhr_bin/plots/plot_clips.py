#!/usr/bin/env python3
"""Plot the fibers + SOTs requested in a build_snippets yaml, with the yaml's
sections shaded green.

Same figure format as analyze_waveforms.py, but for each directory only the
channels named in the yaml are shown -- the listed abdomen fibers plus the
microphone (fetal SOT) and the ppg column (maternal SOT, chosen by ppg_col) --
and every "start-end" section is highlighted in green.

    python3 plot_clips.py lib/tune-ssnet/training_clips_mono.yaml --out-dir out/
"""

import argparse
from pathlib import Path

import yaml

from analyze.data import load_fibers
from analyze.util import normalize_path
from analyze.constants import ABDOMEN_FIBER_NAMES, DEFAULT_DATA_DIR
from fhr_bin.generate_training_snippets import load_mic, load_ppg
from fhr_bin.plots.analyze_waveforms import plot_directory


def parse_sections(specs):
    """build_snippets "start-end" strings -> [(start, end), ...], skipping any
    where end <= start (same leniency build_snippets has)."""
    out = []
    for spec in specs:
        a, b = spec.split("-")
        s, e = float(a), float(b)
        if e > s:
            out.append((s, e))
        else:
            print(f"  skipping invalid section '{spec}' (end <= start)")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("yaml_path", type=Path, help="build_snippets-style yaml")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR,
                        help=f"Base directory of patient subdirectories (default: {DEFAULT_DATA_DIR})")
    parser.add_argument("--out-dir", required=True, type=Path, help="Directory for the figures")
    parser.add_argument("--inches-per-sec", type=float, default=0.5,
                        help="Figure width in inches per second of recording (default: 0.5)")
    parser.add_argument("--dpi", type=int, default=100, help="Output resolution (default: 100)")
    args = parser.parse_args()

    with open(args.yaml_path) as f:
        cfg = yaml.safe_load(f)

    # Column 0 of pvs.npy is the time axis, so the first signal column is 1 --
    # same default as analyze.ica.load_ica_data.
    ppg_col = cfg.get("ppg_col", 1)
    spec = cfg["data"]
    data_dir = normalize_path(args.data_dir)

    for dir_name, dir_spec in spec.items():
        fiber_names = dir_spec["fibers"]
        unknown = [f for f in fiber_names if f not in ABDOMEN_FIBER_NAMES]
        if unknown:
            raise ValueError(
                f"{dir_name}: unknown fiber(s) {unknown}, must be one of {ABDOMEN_FIBER_NAMES}"
            )

        path = normalize_path(f"{data_dir}{dir_name}")
        print(f"\nLoading {dir_name} ...")
        fibers = load_fibers(Path(path))
        mic = load_mic(path)
        ppg = load_ppg(path, ppg_col)

        signals = [(f"{fn} (abdomen, fetal HR)", fibers.abdomen[fn]) for fn in fiber_names]
        signals.append(("microphone (fetal SOT)", mic))
        signals.append((f"pvs[{ppg_col}] (maternal SOT)", ppg))

        sections = parse_sections(dir_spec["sections"])
        plot_directory(dir_name, signals, args.out_dir / f"{dir_name}.png",
                       args.inches_per_sec, args.dpi, highlights=sections)

    print("\n---- All Done ----")


if __name__ == "__main__":
    main()
