"""Write outputs/<scene>/hyworld/splat.ply from the newest HY-World checkpoint."""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--color-space", default="unreal")
    args = parser.parse_args(argv)
    import numpy as np
    import torch

    from engine.hyworld.gs_export import to_ply_arrays
    from engine.hyworld.paths import hy_root, latest_checkpoint
    from engine.job import job_progress
    from engine.splat.ply_export import write_splat_ply

    ckpt = latest_checkpoint(args.scene)
    if ckpt is None:
        print("STATUS // no checkpoint to export. Train first.", flush=True)
        return 2
    job_progress(0, 3, "load checkpoint")
    data = torch.load(ckpt, map_location="cpu", weights_only=False)
    splats = {key: value.detach().float().cpu().numpy() for key, value in data["splats"].items()}
    job_progress(1, 3, "convert to the panorama frame")
    try:
        arrays = to_ply_arrays(splats, np.asarray(data["transform"], dtype=np.float64))
    except ValueError as exc:
        print(f"STATUS // export failed: {exc}", flush=True)
        return 2
    job_progress(2, 3, "write ply")
    path, count = write_splat_ply(hy_root(args.scene) / "splat.ply", arrays, color_space=args.color_space)
    job_progress(3, 3, "done")
    print(f"SPLAT\tply\t{path}", flush=True)
    print(f"STATUS // exported {count} gaussians to {path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
