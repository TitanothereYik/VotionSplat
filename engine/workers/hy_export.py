"""Write outputs/<scene>/hyworld/splat.ply from the newest HY-World checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from engine.splat import defaults as D


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--color-space", default=D.COLOR_SPACE_DEFAULT)
    parser.add_argument("--no-soften", action="store_true",
                        help="keep the trained scales and opacities instead of baking the viewport's softness")
    parser.add_argument("--out", type=Path, default=None,
                        help="write here instead of outputs/<scene>/hyworld/splat.ply")
    args = parser.parse_args(argv)
    import numpy as np
    import torch

    from engine.hyworld.gs_export import load_training_cameras, similarity, soften_splats, to_ply_arrays
    from engine.hyworld.paths import gs_result, hy_root, latest_checkpoint
    from engine.job import job_progress
    from engine.splat.ply_export import write_splat_ply

    ckpt = latest_checkpoint(args.scene)
    if ckpt is None:
        print("STATUS // no checkpoint to export. Train first.", flush=True)
        return 2
    space = D.color_space_from_key(args.color_space)
    is_votion = space == D.COLOR_SPACE_VOTION
    job_progress(0, 100, "load checkpoint")
    data = torch.load(ckpt, map_location="cpu", weights_only=False)
    splats = {key: value.detach().float().cpu().numpy() for key, value in data["splats"].items()}
    soften = None
    if is_votion:
        print("STATUS // votion preset: soften skipped (plugin renders the viewport rule)", flush=True)
    elif not args.no_soften:
        cams = load_training_cameras(gs_result(args.scene))
        if cams is None:
            print("STATUS // soften skipped: run not antialiased or training cameras missing", flush=True)
        else:
            shown = [-1]

            def _progress(done: int, total: int) -> None:
                pct = 5 + int(75 * done / max(total, 1))
                if pct != shown[0]:
                    shown[0] = pct
                    job_progress(pct, 100, f"soften from {len(cams[0])} training views")

            soften = soften_splats(
                splats, *cams, focal=D.SOFTEN_FOCAL, extra_px2=D.SOFTEN_EXTRA_PX2,
                total_px2=D.SOFTEN_EXTRA_PX2 + D.SOFTEN_UE_EPS_PX2, progress=_progress)
            factor = soften[1]
            print(f"STATUS // soften: {float((factor < 0.9).mean()) * 100:.1f}% of splats faded below 0.9, "
                  f"median opacity factor {float(np.median(factor)):.2f}", flush=True)
    job_progress(82, 100, "convert to the panorama frame")
    try:
        arrays = to_ply_arrays(splats, np.asarray(data["transform"], dtype=np.float64), soften)
    except ValueError as exc:
        print(f"STATUS // export failed: {exc}", flush=True)
        return 2
    job_progress(88, 100, "write ply")
    s, _, _ = similarity(np.asarray(data["transform"], dtype=np.float64))
    dest = args.out if args.out is not None else hy_root(args.scene) / "splat.ply"
    path, count = write_splat_ply(
        dest, arrays, color_space=args.color_space,
        votion_raster="gsplat_antialiased", votion_near_plane=D.GSPLAT_NEAR_PLANE / s, votion_sh_degree=0,
    )
    job_progress(100, 100, "done")
    print(f"SPLAT\tply\t{path}", flush=True)
    label = dict(D.COLOR_SPACE_CHOICES).get(space, space)
    print(f"STATUS // exported {count} gaussians to {path} as {label}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
