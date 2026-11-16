"""PaGeR panorama depth. Runs in .venvs/depth_pager. Weights are CC BY-NC."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import save

REPO = Path(__file__).resolve().parents[2] / ".venvs" / "src" / "pager"
CHECKPOINT = "prs-eth/PaGeR"
INPUT_SIZE = (4032, 2016)
SKY_PROB = 0.3


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pano", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    args = parser.parse_args()
    pano = str(Path(args.pano).resolve())
    out = str(Path(args.out).resolve())
    # PaGeR imports its helpers as ``src.*`` from the repo root.
    for path in (REPO / "src", REPO):
        sys.path.insert(0, str(path))
    os.chdir(REPO)
    try:
        import torch
        from huggingface_hub import snapshot_download
        from omegaconf import OmegaConf
        from src.pager import Pager
        from src.utils.geometry_utils import cubemap_to_erp, erp_to_cubemap
    except Exception as exc:
        print(f"STATUS // pager import failed: {exc}", file=sys.stderr)
        return 1
    bgr = cv2.imread(pano, cv2.IMREAD_COLOR)
    if bgr is None:
        print(f"STATUS // failed to read {pano}", file=sys.stderr)
        return 1
    image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, INPUT_SIZE, interpolation=cv2.INTER_AREA)
    # A hub id inside pathlib turns into backslashes on Windows, so load from a local snapshot.
    local = Path(snapshot_download(CHECKPOINT, allow_patterns=["config.yaml", "model.safetensors"]))
    cfg = OmegaConf.load(local / "config.yaml")
    modalities = set(cfg.modalities)
    face = int(cfg.face_size)
    cube_fov = float(getattr(cfg, "cube_fov", 90.0))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pager = Pager(local, cfg=cfg, device=device)
    pager.get_intrinsics_extrinsics(image_size=face, fov=cube_fov)
    pager.model.to(device, dtype=pager.weight_dtype)
    pager.model.eval()
    img = torch.from_numpy(image).permute(2, 0, 1).to(torch.float32) / 255.0
    img = img * 2.0 - 1.0
    cube = erp_to_cubemap(img, face_w=face, fov=cube_fov).unsqueeze(0).to(device)
    # Scale is normalised to camera height downstream; outdoor head, no CLIP router.
    skip = {name for name in ("scale_indoor", "normals") if name in modalities}
    with torch.inference_mode():
        pred = pager(cube, dtype=torch.float16, skip_heads=skip)
        sky_logits = pred["sky"][0] if "sky" in pred else None
        height, width = int(args.height), int(args.width)
        distance, _log = pager.process_depth_output(
            pred["depth"][0],
            orig_size=(height, width),
            sky_mask=sky_logits,
            log_scale=pred.get("scale"),
        )
        distance = np.squeeze(distance.float().cpu().numpy()).astype(np.float32)
        sky = None
        if sky_logits is not None:
            sky_prob = cubemap_to_erp(torch.sigmoid(sky_logits.float()), height, width, fov=cube_fov)
            sky = np.squeeze(sky_prob.cpu().numpy()) >= SKY_PROB
    if distance.shape != (height, width):
        distance = cv2.resize(distance, (width, height), interpolation=cv2.INTER_LINEAR)
    valid = np.isfinite(distance) & (distance > 1e-6)
    if sky is not None:
        valid &= ~sky
    save(out, distance, valid, sky=sky)
    print(f"STATUS // pager wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
