"""In-app LiteGS trainer. Qt never imports this. ``python -m engine.workers.splat``."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from engine.pano.scene_io import scene_dir
from engine.splat import defaults as D
from engine.splat.trainer import SplatTrainer, log


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Votion splat trainer (LiteGS | MCMC Compact)")
    parser.add_argument("--scene", required=True)
    parser.add_argument("--strategy", default="litegs", choices=sorted(D.STRATEGY_KEYS))
    parser.add_argument(
        "--mode",
        default="train",
        choices=["train", "continue", "retrain", "export", "export_compact", "export_compact_dc", "view"],
    )
    parser.add_argument("--max-steps", type=int, default=D.MAX_STEPS)
    parser.add_argument("--budget", type=int, default=D.GAUSSIAN_BUDGET)
    parser.add_argument("--res-scale", type=float, default=1.0)
    parser.add_argument("--holdout", action="store_true")
    parser.add_argument("--headless", action="store_true", help="no stdin control thread (CLI runs)")
    parser.add_argument(
        "--color-space",
        default=D.COLOR_SPACE_DEFAULT,
        choices=list(D.COLOR_SPACES),
        help="PLY export: wallgs (WallGS Pro, default), wallgs_acescg (WallGS Pro, linear ACEScg), luma, srgb, linear",
    )
    args = parser.parse_args(argv)

    import torch

    strategy = D.strategy_from_key(args.strategy)
    idx = int(os.environ.get("CUDA_DEVICE_INDEX", "0") or 0)
    if not torch.cuda.is_available():
        log("STATUS // torch.cuda.is_available() is False")
        return 3
    device = torch.device(f"cuda:{idx}")
    root = Path(scene_dir(args.scene))
    steps = max(1, int(args.max_steps))
    log(
        f"STATUS // splat {args.mode} scene={args.scene} controller={strategy} steps={steps} "
        f"budget={int(args.budget)} res={args.res_scale} holdout={bool(args.holdout)} "
        f"color={D.color_space_from_key(args.color_space)} device={device}"
    )
    trainer = SplatTrainer(
        scene_dir=root,
        strategy=strategy,
        mode=args.mode,
        max_steps=steps,
        device=device,
        budget=int(args.budget),
        resolution_scale=float(args.res_scale),
        holdout=bool(args.holdout),
        headless=bool(args.headless),
        color_space=args.color_space,
    )
    try:
        return trainer.run()
    except Exception as exc:
        log(f"STATUS // splat failed: {type(exc).__name__}: {exc}")
        import traceback

        traceback.print_exc()
        return 4


if __name__ == "__main__":
    sys.exit(main())
