"""Short GPU compare of the dotted guide and the solid panorama mesh.

Ask before running. One pose, no WorldStereo sample.

    python tools/spike_world_guide.py --scene alpine_01 --rail 2 --pose 80
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from engine.pano.scene_io import pano_path, scene_dir
from engine.world.defaults import GUIDE_SUPERSAMPLE, HIRES_SUPERSAMPLE
from engine.world.pano_mesh import ensure_pano_mesh, load_ground, world_dir
from engine.world.render import WorldRenderer
from engine.worldstereo.adapter import pin_K
from engine.worldstereo.defaults import SIZE_H, SIZE_W


def _read_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"missing {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _video_frame(path: Path, index: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open {path}")
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = cap.read()
    finally:
        cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"{path} has no frame {index}")
    return frame


def main(argv: list[str] | None = None) -> int:
    import torch

    parser = argparse.ArgumentParser(description="Compare the old dot guide with the panorama mesh")
    parser.add_argument("--scene", default="alpine_01")
    parser.add_argument("--rail", type=int, default=2)
    parser.add_argument("--pose", type=int, default=80)
    args = parser.parse_args(argv)

    root = scene_dir(args.scene)
    rail_path = root / "rails" / f"rail_{int(args.rail):02d}.json"
    poses = np.asarray(json.loads(rail_path.read_text(encoding="utf-8")), dtype=np.float64)
    if int(args.pose) < 0 or int(args.pose) >= poses.shape[0]:
        raise IndexError(f"{rail_path} has {poses.shape[0]} poses, no index {args.pose}")
    w2c = poses[int(args.pose)]
    erp = _read_rgb(pano_path(args.scene))

    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    piece = ensure_pano_mesh(args.scene, erp, "cuda:0")
    mesh_s = time.perf_counter() - started
    stats_path = world_dir(args.scene) / "pano_mesh.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8")) if stats_path.is_file() else {}

    view_dir = root / "worldstereo" / "rails" / f"rail_{int(args.rail):02d}" / "front"
    meta = json.loads((view_dir / "meta.json").read_text(encoding="utf-8"))
    positions = [int(v) for v in meta["positions"]]
    frame_i = positions.index(int(args.pose))
    old = _video_frame(view_dir / "render" / "render.mp4", frame_i)

    k_mat = pin_K()
    renderer = WorldRenderer([piece], {"pano": erp}, "cuda:0", ground=load_ground(args.scene))
    try:
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        guide = renderer.render(w2c, k_mat, SIZE_H, SIZE_W, supersample=GUIDE_SUPERSAMPLE)
        guide_s = time.perf_counter() - started
        guide_vram = int(torch.cuda.max_memory_allocated())

        from engine.hires.pinhole import HIRES_H, HIRES_W, scaled_k

        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        hires = renderer.render(w2c, scaled_k(k_mat), HIRES_H, HIRES_W, supersample=HIRES_SUPERSAMPLE)
        hires_s = time.perf_counter() - started
        hires_vram = int(torch.cuda.max_memory_allocated())
    finally:
        renderer.close()

    out = root / "worldstereo" / "spike"
    out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out / "old_guide.png"), old)
    cv2.imwrite(str(out / "new_guide.png"), cv2.cvtColor(guide.rgb, cv2.COLOR_RGB2BGR))
    mask = np.where(guide.empty, 255, 0).astype(np.uint8)
    cv2.imwrite(str(out / "new_mask.png"), mask)
    cv2.imwrite(str(out / "hires_world.png"), cv2.cvtColor(hires.rgb, cv2.COLOR_RGB2BGR))
    payload = {
        "scene": args.scene,
        "rail": int(args.rail),
        "pose": int(args.pose),
        "mesh_seconds": mesh_s,
        "guide_seconds": guide_s,
        "hires_seconds": hires_s,
        "guide_vram_bytes": guide_vram,
        "hires_vram_bytes": hires_vram,
        "faces": stats.get("faces"),
        "faces_dropped": stats.get("faces_dropped"),
        "ground": stats.get("ground"),
        "empty_fraction": float(guide.empty.mean()),
        "guide_supersample": GUIDE_SUPERSAMPLE,
        "hires_supersample": HIRES_SUPERSAMPLE,
    }
    (out / "spike.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        f"STATUS // spike mesh={mesh_s:.2f}s guide={guide_s:.2f}s/{guide_vram / 1e9:.2f}GB "
        f"hires={hires_s:.2f}s/{hires_vram / 1e9:.2f}GB empty={payload['empty_fraction']:.3f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
