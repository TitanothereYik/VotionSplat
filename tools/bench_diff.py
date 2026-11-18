"""Compare two bench frame files. Prints the mean and largest difference on a 0-255 scale."""

from __future__ import annotations

import sys

import numpy as np


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 2:
        raise SystemExit("usage: python -m tools.bench_diff A.npy B.npy")
    first = np.load(args[0]).astype(np.int16)
    second = np.load(args[1]).astype(np.int16)
    if first.shape != second.shape:
        raise SystemExit(f"shapes differ: {first.shape} {second.shape}")
    diff = np.abs(first - second)
    print(f"frames={first.shape[0]} mean={diff.mean():.4f} max={int(diff.max())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
