#!/usr/bin/env python3
"""Plot every waveform in one or more BannerData directories.

For each directory a single figure is produced with one subplot per signal
(chest fiber, the abdomen fibers 2A-2D, the microphone, and the pvs/ppg
channels). Each subplot is decimated with a min/max envelope so peaks stay
visible across the full ~30 min recording.
"""

import argparse
import sys
from pathlib import Path

import math

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
from scipy.io import wavfile

SRC_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # for build_snippets

from analyze.data import Audio, load_fibers  # noqa: E402
from analyze.util import normalize_path  # noqa: E402
from constants import ABDOMEN_FIBER_NAMES, MIC_FILE, PVS_FILE  # noqa: E402


def nice_interval(target):
    """Largest 'nice' number (1, 2, 2.5, 5 x 10^k) <= target."""
    if target <= 0:
        return 1.0
    base = 10 ** math.floor(math.log10(target))
    for m in (5, 2.5, 2, 1):
        if m * base <= target:
            return m * base
    return base


def plot_directory(name, signals, out_path, inches_per_sec, dpi, highlights=None):
    n = len(signals)
    duration = max(s.time[-1] for _, s in signals) if signals else 1.0

    # Horizontal detail is width_inches * dpi pixels, capped by matplotlib's
    # 2^16 px limit. Push width as wide as inches_per_sec asks, then clamp.
    max_width = 64000.0 / dpi
    width = max(20.0, duration * inches_per_sec)
    if width > max_width:
        print(f"  clamping width {width:.0f}in -> {max_width:.0f}in (PNG pixel limit; "
              f"lower --dpi or window the recording for finer detail)")
        width = max_width

    fig, axes = plt.subplots(
        n, 1, figsize=(width, 2.4 * n), sharex=True, constrained_layout=True
    )
    if n == 1:
        axes = [axes]

    # Aim for ~one timestamp per inch, but no coarser than every 50s.
    sec_per_inch = duration / width
    tick = nice_interval(min(50.0, sec_per_inch))

    for ax, (label, audio) in zip(axes, signals):
        t, x = audio.time, audio.data
        ax.plot(t, x, color="#1f6feb", linewidth=0.6)
        ax.set_title(label, loc="left", fontsize=10)
        ax.set_ylabel("Amplitude")
        ax.set_xlabel("Time (s)")
        # sharex hides inner tick labels by default; force timestamps on every plot.
        ax.tick_params(axis="x", labelbottom=True)
        ax.xaxis.set_major_locator(MultipleLocator(tick))
        for s, e in (highlights or []):
            ax.axvspan(s, e, color="#2ca02c", alpha=0.15, linewidth=0)
        ax.margins(x=0)
        ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.5)

    axes[-1].set_xlim(0, duration)
    fig.suptitle(name, fontsize=13)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)

def load_mic(path: str) -> Audio:
    fs, arr = wavfile.read(path + MIC_FILE)
    arr = arr if arr.ndim == 1 else arr[:, 0]
    t = np.arange(len(arr)) / float(fs)
    return Audio(t, fs, arr)

def process_directory(dir_path, out_dir, inches_per_sec, dpi):
    dir_path = normalize_path(str(dir_path))
    name = Path(dir_path.rstrip("/")).name
    print(f"\nLoading {name} ...")

    fibers = load_fibers(Path(dir_path))
    mic = load_mic(dir_path)

    signals = [("chest (maternal HR)", fibers.chest)]
    signals += [(f"{fn} (abdomen, fetal HR)", fibers.abdomen[fn])
                for fn in ABDOMEN_FIBER_NAMES if fn in fibers.abdomen]
    signals.append(("microphone (fetal SOT)", mic))

    # pvs / ppg channels (maternal SOT).
    pvs_path = Path(dir_path) / PVS_FILE
    if pvs_path.exists():
        pvs = np.load(pvs_path)
        pt = pvs[:, 0].astype(float)
        phz = round(1.0 / float(np.median(np.diff(pt))))
        for i in range(1, pvs.shape[1]):
            signals.append((f"pvs[{i}] (maternal SOT)", Audio(pt, phz, pvs[:, i].astype(float))))

    plot_directory(name, signals, Path(out_dir) / f"{name}.png", inches_per_sec, dpi)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dirs", nargs="+", type=Path, help="BannerData-style directories")
    parser.add_argument("--out-dir", required=True, type=Path, help="Directory for the figures")
    parser.add_argument("--inches-per-sec", type=float, default=0.5,
                        help="Figure width in inches per second of recording (default: 0.5)")
    parser.add_argument("--dpi", type=int, default=100, help="Output resolution (default: 100)")
    args = parser.parse_args()

    for d in args.dirs:
        process_directory(d, args.out_dir, args.inches_per_sec, args.dpi)

    print("\n---- All Done ----")


if __name__ == "__main__":
    main()
