#!/usr/bin/env python3
"""Concatenate two ps3000a.npy-format recordings ([time, ch1, ch2, ...]) into one.

Timestamps are ignored; a fresh time column is generated from the first file's
sample rate so the output starts at 0s and runs to the combined length.
"""

import argparse

import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Concatenate two ps3000a.npy recordings")
    parser.add_argument("first", help="First .npy file (comes first in the output)")
    parser.add_argument("second", help="Second .npy file (appended after the first)")
    parser.add_argument("out", help="Output .npy path")
    args = parser.parse_args()

    a = np.load(args.first)
    b = np.load(args.second)

    hz = round(1 / (a[1, 0] - a[0, 0]))

    data = np.vstack([a[:, 1:], b[:, 1:]])
    time = np.arange(len(data)) / hz

    out = np.column_stack([time, data])
    np.save(args.out, out)
    print(f"Wrote {len(out)} samples ({len(out) / hz:.1f}s) at {hz}Hz to {args.out}")


if __name__ == "__main__":
    main()
