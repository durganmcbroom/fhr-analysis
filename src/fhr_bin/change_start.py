import argparse

import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="Prepend `start` seconds of zero-valued samples (at the recording's own "
                     "sample rate) to a ps3000a.npy-style [time, ch1, ch2, ...] array. Used when "
                     "a recording actually began `start` seconds into the session, so consumers "
                     "that window by absolute time see real (silent) samples for that gap instead "
                     "of an empty/truncated array."
    )
    parser.add_argument("file", help=".npy file (comes first in the output)")
    parser.add_argument("start", help="Seconds of silence to prepend")
    parser.add_argument("out", help="Output .npy path")
    args = parser.parse_args()

    data = np.load(args.file)
    start = float(args.start)
    if start < 0:
        raise ValueError(f"start must be >= 0, got {start}")

    dt = float(np.median(np.diff(data[:, 0])))
    n_pad = round(start / dt) + 1

    pad = np.zeros((n_pad, data.shape[1]), dtype=data.dtype)
    pad[:, 0] = np.arange(n_pad) * dt

    # Shift the original samples by exactly the padding's duration (n_pad * dt,
    # not the raw `start`) so the time column stays on one continuous, evenly
    # spaced grid across the pad/data boundary.
    shifted = data.copy()
    shifted[:, 0] += n_pad * dt

    out_data = np.vstack([pad, shifted])

    np.save(args.out, out_data)
    np.savetxt(f"{args.out}.csv", out_data, delimiter=',')


if __name__ == "__main__":
    main()
