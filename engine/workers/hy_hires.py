"""Upscale expanded HY-World clips with SeedVR2 3B. Run with .venvs/seedvr2."""

from __future__ import annotations

import argparse
import sys

from engine.hyworld.hires import ALL_CHOICES, SCALE, SCALES


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="HiRes expanded clips with SeedVR2 3B")
    parser.add_argument("--scene", required=True)
    parser.add_argument("--only", default="")
    parser.add_argument("--restart", action="store_true")
    parser.add_argument("--level", default="auto", choices=ALL_CHOICES)
    parser.add_argument("--scale", type=int, default=SCALE, choices=SCALES)
    parser.add_argument("--vram-cap-gb", type=float, default=0.0)
    parser.add_argument("--bench", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    from engine.hyworld.hires_run import run

    return run(
        args.scene,
        only=args.only.strip(),
        resume=not args.restart,
        level=args.level,
        cap_gb=float(args.vram_cap_gb),
        bench=bool(args.bench),
        verbose=bool(args.verbose),
        scale=int(args.scale),
    )


if __name__ == "__main__":
    sys.exit(main())
