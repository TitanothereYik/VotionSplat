"""DAP panorama depth. Runs in .venvs/depth_dap. Weights follow the DINOv3 licence."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import save

REPO = Path(__file__).resolve().parents[2] / ".venvs" / "src" / "dap"
WEIGHTS = ("Insta360-Research/DAP-weights", "model.pth")
INPUT_SIZE = (1024, 512)
SKY_PROB = 0.5


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pano", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    args = parser.parse_args()
    pano = str(Path(args.pano).resolve())
    out = str(Path(args.out).resolve())
    # DAP loads DINOv3 through torch.hub from a path relative to the repo root.
    sys.path.insert(0, str(REPO))
    os.chdir(REPO)
    try:
        import torch
        import yaml
        from huggingface_hub import hf_hub_download
        from networks.models import make
    except Exception as exc:
        print(f"STATUS // dap import failed: {exc}", file=sys.stderr)
        return 1
    bgr = cv2.imread(pano, cv2.IMREAD_COLOR)
    if bgr is None:
        print(f"STATUS // failed to read {pano}", file=sys.stderr)
        return 1
    image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, INPUT_SIZE, interpolation=cv2.INTER_AREA)
    config = yaml.safe_load((REPO / "config" / "infer.yaml").read_text(encoding="utf-8"))
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    state = torch.load(hf_hub_download(*WEIGHTS), map_location=device)
    if any(key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    model = make(config["model"]).to(device)
    own = model.state_dict()
    model.load_state_dict({k: v for k, v in state.items() if k in own}, strict=False)
    missing = [k for k in own if k not in state]
    if missing:
        print(f"STATUS // dap missing {len(missing)} weights, first {missing[0]}", file=sys.stderr)
        return 1
    model.eval()
    tensor = torch.from_numpy(image.astype(np.float32).transpose(2, 0, 1) / 255.0).unsqueeze(0).to(device)
    with torch.inference_mode():
        outputs = model(tensor)
    distance = np.squeeze(outputs["pred_depth"][0].float().cpu().numpy()).astype(np.float32)
    sky = np.squeeze(outputs["pred_mask"][0].float().cpu().numpy()) > SKY_PROB
    width, height = int(args.width), int(args.height)
    distance = cv2.resize(distance, (width, height), interpolation=cv2.INTER_LINEAR)
    sky = cv2.resize(sky.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST) > 0
    valid = np.isfinite(distance) & (distance > 1e-6) & ~sky
    save(out, distance, valid, sky=sky)
    print(f"STATUS // dap wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
