#!/usr/bin/env python3
"""Convert PicoScope CSV exports to .npy (Banner_data format) or .wav files."""

import argparse
import sys
from pathlib import Path

import numpy as np


def parse_pico_csv(csv_path: Path) -> tuple[np.ndarray, int]:
    """Return (data, fs) where data is shape (N, 1+num_channels): [time, ch_a, ...]."""
    with open(csv_path) as f:
        header = f.readline().strip()   # "Time,Channel A,Channel B,..."
        f.readline()                     # units row "(s),(V),..."
        f.readline()                     # blank row

    col_names = [c.strip() for c in header.split(",")]
    num_cols = len(col_names)

    data = np.genfromtxt(csv_path, delimiter=",", skip_header=3, usecols=range(num_cols))
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    dt = float(np.median(np.diff(data[:, 0])))
    fs = round(1.0 / dt)
    return data, fs


def to_npy(data: np.ndarray, fs: int, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, data.astype(np.float64))
    print(f"Saved {data.shape} float64 array -> {out_path}  (fs={fs} Hz)")


def to_wav(data: np.ndarray, fs: int, out_path: Path, channel: int = 1) -> None:
    from scipy.io import wavfile

    sig = data[:, channel].astype(np.float64)
    peak = np.max(np.abs(sig))
    if peak > 0:
        sig /= peak
    pcm = (sig * 32767).astype(np.int16)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(out_path, fs, pcm)
    ch_name = f"column {channel}"
    print(f"Saved {len(pcm)} samples @ {fs} Hz -> {out_path}  ({ch_name})")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert PicoScope CSV to .npy or .wav"
    )
    parser.add_argument("csv", type=Path, help="PicoScope CSV file")
    parser.add_argument(
        "output", type=Path, nargs="?",
        help="Output file path (default: <csv stem>.npy or <csv stem>.wav beside the CSV)"
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--npy", action="store_true", help="Write Banner_data-style .npy array")
    mode.add_argument("--wav", action="store_true", help="Write .wav audio file")

    parser.add_argument(
        "--channel", type=int, default=1,
        help="Column index to use for --wav (1=Channel A, 2=Channel B, …; default: 1)"
    )

    args = parser.parse_args()

    if not args.csv.exists():
        sys.exit(f"Error: {args.csv} not found")

    data, fs = parse_pico_csv(args.csv)
    print(f"Loaded {args.csv.name}: {data.shape[0]} samples, {data.shape[1]-1} channel(s), fs={fs} Hz")

    if args.wav:
        if args.channel < 1 or args.channel >= data.shape[1]:
            sys.exit(f"Error: --channel {args.channel} out of range (1..{data.shape[1]-1})")
        out = args.output or args.csv.with_suffix(".wav")
        to_wav(data, fs, out, channel=args.channel)
    else:
        out = args.output or args.csv.with_suffix(".npy")
        to_npy(data, fs, out)


if __name__ == "__main__":
    main()
