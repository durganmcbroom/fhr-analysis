"""``rtmon-recover`` — finalize a session whose server died mid-recording.

The recorder appends to ``<name>.raw`` and only writes the ``.npy`` header on a clean
stop, so a crash leaves the samples on disk but not in a form ``np.load`` accepts.
This turns them into the normal session files. Nothing is lost but the partial row at
the very end, which is excluded by the row count.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from rtmon.recorder import recover


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("directory", nargs="+", help="session directory (or directories)")
    parser.add_argument("--keep-raw", action="store_true",
                        help="leave the .raw files in place after converting")
    args = parser.parse_args()

    for directory in args.directory:
        path = Path(directory)
        if not path.is_dir():
            print(f"[recover] {path}: not a directory")
            continue
        written = recover(path, keep_raw=args.keep_raw)
        if not written:
            print(f"[recover] {path}: nothing to finalize")
        for out in written:
            print(f"[recover] wrote {out}")


if __name__ == "__main__":
    main()
