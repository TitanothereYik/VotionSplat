"""One WorldStereo trajectory at a time over the HY-World guides."""

from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29531")
os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from engine.job import apply_huggingface_cache_env


def main(argv: list[str] | None = None) -> int:
    apply_huggingface_cache_env(os.environ)
    parser = argparse.ArgumentParser(description="HY-World expansion with the FP8 DiT")
    parser.add_argument("--scene", required=True)
    parser.add_argument("--only", default="")
    parser.add_argument("--restart", action="store_true")
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument(
        "--level",
        default="auto",
        choices=("auto", "card", "blocks", "states", "controlnet"),
        help="How much of the DiT stays on the card. auto reads this machine and decides.",
    )
    parser.add_argument(
        "--vram-cap-gb",
        type=float,
        default=0.0,
        help="Treat the card as having this many GB for the DiT. 0 uses what is free.",
    )
    parser.add_argument(
        "--bench",
        action="store_true",
        help="With --only: sample that path into _votion/bench and leave the real result alone.",
    )
    args = parser.parse_args(argv)
    from engine.worldstereo.hy_expand import generate

    return generate(
        args.scene,
        only=args.only.strip(),
        resume=not args.restart,
        seed=int(args.seed),
        level=args.level,
        cap_gb=float(args.vram_cap_gb),
        bench=bool(args.bench),
    )


if __name__ == "__main__":
    sys.exit(main())
