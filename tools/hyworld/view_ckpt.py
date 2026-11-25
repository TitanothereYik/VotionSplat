"""Idle viewer: the newest HY-World checkpoint in the live viewport. Stage ``view_ckpt``."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", required=True)
    args = parser.parse_args()
    import torch
    from gsplat.rendering import rasterization

    import votion_live

    result = Path(args.result_dir)
    found = sorted((result / "ckpts").glob("ckpt_*_rank0.pt"), key=lambda path: path.stat().st_mtime)
    if not found:
        print("STATUS // view: no checkpoint", flush=True)
        return 2
    device = torch.device("cuda:0")
    data = torch.load(found[-1], map_location=device, weights_only=False)
    splats = data["splats"]
    keep = torch.ones(splats["means"].shape[0], dtype=torch.bool, device=device)
    if "mask_score" in splats:
        keep = splats["mask_score"][:, 0] > splats["mask_score"][:, 1]
    means = splats["means"][keep].float()
    quats = splats["quats"][keep].float()
    scales = torch.exp(splats["scales"][keep].float())
    opacities = torch.sigmoid(splats["opacities"][keep].float())
    colors = torch.cat([splats["sh0"][keep], splats["shN"][keep]], 1).float()
    degree = int(round(math.sqrt(colors.shape[1]))) - 1
    background = torch.tensor([votion_live.BG], device=device, dtype=torch.float32) / 255.0

    def render(c2w, k, w, h):
        with torch.no_grad():
            out, _alphas, _info = rasterization(
                means=means,
                quats=quats,
                scales=scales,
                opacities=opacities,
                colors=colors,
                viewmats=torch.linalg.inv(torch.from_numpy(c2w).float().to(device))[None],
                Ks=torch.from_numpy(k).float().to(device)[None],
                width=w,
                height=h,
                sh_degree=degree,
                backgrounds=background,
                packed=False,
                rasterize_mode="antialiased",
                render_mode="RGB",
            )
        return (out[0, ..., :3].clamp(0, 1) * 255 + 0.5).to(torch.uint8).cpu().numpy()

    count = int(means.shape[0])
    pick = torch.linspace(0, max(count - 1, 0), min(count, 8192), device=device).long()
    unit = votion_live.scene_unit(means[pick].cpu().numpy(), data["center_point"])
    names, c2ws = [], []
    cams = result / votion_live.CAMERAS_FILE
    if cams.is_file():
        saved = json.loads(cams.read_text(encoding="utf-8"))
        names = list(saved.get("names") or [])
        c2ws = [np.asarray(m, dtype=np.float64) for m in saved.get("c2w") or []]
    view = votion_live.LiveView(render, data["up_direction"], data["facing_direction"],
                                data["center_point"], unit, c2ws)
    votion_live.emit("cameras", json.dumps(votion_live.camera_rows(names)))
    votion_live.emit("step", int(data.get("step", 0)) + 1, int(data.get("step", 0)) + 1)
    votion_live.emit("gaussians", count)
    votion_live.emit("state", "VIEW")
    print(f"STATUS // view: {found[-1].name} {count} gaussians", flush=True)
    view.render(force=True)
    while True:
        words = view.poll()
        if "STOP" in words or view.control.closed:
            break
        if not view.render():
            time.sleep(0.01)
    view.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
