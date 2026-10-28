"""Compute geometry worker: MoGe depth -> sparse cloud + FLOOR/SIDE orthos.

Qt never imports this module. Run as ``python -m engine.workers.moge``.
Uses SplatKit ``core.matrix3d_pipeline`` at the pinned vendor SHA — no ComfyUI.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SPLATKIT = REPO_ROOT / "vendor" / "splatkit"
if str(SPLATKIT) not in sys.path:
    sys.path.insert(0, str(SPLATKIT))

from engine.geo import defaults as D
from engine.geo.ply import write_xyz_rgb_ply
from engine.geo.scene_ref import equirect_to_cloud, equirect_to_ortho_views, write_png_rgba
from engine.job import job_progress
from engine.pano.scene_io import pano_path, scene_dir


def log(msg: str) -> None:
    print(msg, flush=True)


def moge_ckpt() -> Path:
    return REPO_ROOT / "models" / "MoGe" / "model.pt"


def moge_url() -> str:
    return "https://huggingface.co/Ruicheng/moge-vitl/resolve/main/model.pt"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Votion Compute geometry (MoGe)")
    parser.add_argument("--scene", required=True)
    parser.add_argument("--moge-level", type=int, default=D.MOGE_LEVEL)
    parser.add_argument("--point-budget", type=int, default=D.POINT_BUDGET)
    args = parser.parse_args(argv)

    pano_file = pano_path(args.scene)
    if not pano_file.is_file():
        log("STATUS // missing pano.png — run Phase 1 first")
        log(f"MISSING_pano {pano_file}")
        return 2

    ckpt = moge_ckpt()
    if not ckpt.is_file():
        log(f"MISSING_moge {moge_url()}")
        log("STATUS // MoGe model.pt not in models/MoGe — Download models (P2 subset)")
        return 3

    idx = int(os.environ.get("CUDA_DEVICE_INDEX", "0") or 0)
    device = f"cuda:{idx}"
    level = max(0, min(9, int(args.moge_level)))
    budget = max(D.POINT_BUDGET_MIN, min(D.POINT_BUDGET_MAX, int(args.point_budget)))

    log(f"STATUS // moge start scene={args.scene} device={device} moge_level={level} budget={budget}")
    job_progress(1, 6, "read pano")

    import cv2
    import numpy as np
    from core import matrix3d_pipeline as mp

    bgr = cv2.imread(str(pano_file), cv2.IMREAD_COLOR)
    if bgr is None:
        log(f"STATUS // failed to read {pano_file}")
        return 2
    pano = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    pano = cv2.resize(pano, (2048, 1024), interpolation=cv2.INTER_AREA)

    job_progress(2, 6, "load MoGe")
    model = mp.get_moge_model(ckpt=str(ckpt), device=device)
    job_progress(3, 6, "depth")
    depth, mask = mp.moge_panorama_depth(
        pano,
        model=model,
        ckpt=str(ckpt),
        device=device,
        resolution_level=level,
        merge_long=1440,
        merge_short=720,
    )
    job_progress(4, 6, "cloud + orthos")
    pts, cols = equirect_to_cloud(depth, mask, pano, budget)
    views = equirect_to_ortho_views(depth, mask, pano)

    job_progress(5, 6, "write FLOOR / SIDE")
    out = scene_dir(args.scene) / "moge"
    out.mkdir(parents=True, exist_ok=True)
    ply = write_xyz_rgb_ply(out / "pointcloud.ply", pts, cols)

    meta = {
        "moge_level": level,
        "budget": budget,
        "count": int(pts.shape[0]),
        "device": device,
        "pano": str(pano_file),
        "views": {},
        "points": np.round(pts, 3).tolist(),
        "colors": cols.tolist(),
    }

    floor_path = out / "floor.png"
    side_path = out / "side.png"
    if views and "top" in views:
        write_png_rgba(floor_path, views["top"]["rgba"])
        meta["views"]["floor"] = {
            "lo": views["top"]["lo"],
            "hi": views["top"]["hi"],
            "w": views["top"]["w"],
            "h": views["top"]["h"],
            "file": "floor.png",
        }
        log(f"GEO\tfloor\t{floor_path}")
    if views and "side" in views:
        write_png_rgba(side_path, views["side"]["rgba"])
        meta["views"]["side"] = {
            "lo": views["side"]["lo"],
            "hi": views["side"]["hi"],
            "w": views["side"]["w"],
            "h": views["side"]["h"],
            "file": "side.png",
        }
        log(f"GEO\tside\t{side_path}")

    # Drop bulky arrays from the sidecar the UI reloads (points stay in ply).
    slim = {k: v for k, v in meta.items() if k not in ("points", "colors")}
    slim["ply"] = "pointcloud.ply"
    json_path = out / "scene_ref.json"
    json_path.write_text(json.dumps(slim, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(out / "cloud.npz", points=pts, colors=cols)

    log(f"GEO\tply\t{ply}")
    log(f"GEO\tjson\t{json_path}")
    log(f"STATUS // moge done {int(pts.shape[0])} pts -> {out}")
    job_progress(6, 6, "moge done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
