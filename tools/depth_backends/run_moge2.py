"""MoGe-2 panorama depth. Runs in .venvs/depth_moge2, never in the app venv."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import save

FOV = 90.0
RES = 1024
BATCH = 4
CHECKPOINT = "Ruicheng/moge-2-vitl-normal"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pano", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    try:
        import torch
        from moge.model.v2 import MoGeModel
        from moge.utils.panorama import merge_panorama_depth, split_panorama_image

        try:
            import utils3d_moge as utils3d
        except ImportError:
            import utils3d
    except Exception as exc:
        print(f"STATUS // moge2 import failed: {exc}", file=sys.stderr)
        return 1
    bgr = cv2.imread(args.pano, cv2.IMREAD_COLOR)
    if bgr is None:
        print(f"STATUS // failed to read {args.pano}", file=sys.stderr)
        return 1
    image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (int(args.width), int(args.height)), interpolation=cv2.INTER_AREA)
    # Same 12 icosahedron cameras as upstream infer_panorama, at 1024 px instead of 512.
    vertices, _ = utils3d.np.create_icosahedron_mesh()
    intrinsics = utils3d.np.intrinsics_from_fov(fov_x=np.deg2rad(FOV), fov_y=np.deg2rad(FOV))
    extrinsics = utils3d.np.extrinsics_look_at([0, 0, 0], vertices, [0, 0, 1]).astype(np.float32)
    intr_list = [intrinsics] * len(vertices)
    views = split_panorama_image(image, extrinsics, intr_list, RES)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = MoGeModel.from_pretrained(CHECKPOINT).to(device).eval()
    distances: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(views), BATCH):
            batch = np.stack(views[start : start + BATCH]).astype(np.float32) / 255.0
            tensor = torch.from_numpy(batch).permute(0, 3, 1, 2).to(device)
            fov = torch.full((tensor.shape[0],), FOV, dtype=torch.float32, device=device)
            output = model.infer(tensor, fov_x=fov, apply_mask=False)
            distances.extend(list(output["points"].norm(dim=-1).float().cpu().numpy()))
            masks.extend(list(output["mask"].float().cpu().numpy() > 0.5))
    model.to("cpu")
    print(f"STATUS // moge2 views {len(views)} in {time.perf_counter() - started:.1f}s", flush=True)
    merge_w = min(3072, int(args.width))
    merge_h = min(1536, int(args.height))
    if time.perf_counter() - started > 300.0:
        merge_w, merge_h = min(2048, int(args.width)), min(1024, int(args.height))
        print("STATUS // moge2 merge fallback 2048x1024", flush=True)
    depth, mask = merge_panorama_depth(merge_w, merge_h, distances, masks, extrinsics, intr_list)
    depth = cv2.resize(np.asarray(depth, np.float32), (int(args.width), int(args.height)), interpolation=cv2.INTER_LINEAR)
    valid = cv2.resize(np.asarray(mask, np.uint8), (int(args.width), int(args.height)), interpolation=cv2.INTER_NEAREST) > 0
    save(args.out, depth, valid)
    print(f"STATUS // moge2 wrote {args.out} in {time.perf_counter() - started:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
