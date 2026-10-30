"""Known-pose COLMAP dataset. Qt never imports this.

``python -m engine.workers.known_pose --scene alpine_02``

Writes ``<scene>/worldstereo/dataset``. Does not modify ``<scene>/colmap``.
"""

from __future__ import annotations

import argparse
import os
import traceback


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="WorldStereo known-pose COLMAP dataset")
    parser.add_argument("--scene", required=True)
    args = parser.parse_args(argv)
    idx = int(os.environ.get("CUDA_DEVICE_INDEX", "0") or 0)
    device = f"cuda:{idx}"
    try:
        from engine.sfm.known_pose import build

        build(args.scene, device=device)
    except Exception as exc:
        print(f"STATUS // known-pose failed: {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
