"""One-off: render a training view next to its GT for the current checkpoint.

Run as ``python -m tools.render_gt_compare --scene NAME [--cam N] [--out PATH]``.
Not part of the product CLI — a visual sanity check while iterating on the
splat trainer / dataset pipeline.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from engine.pano.scene_io import scene_dir
from engine.splat import defaults as D
from engine.splat.backend.litegs_backend import LiteGSBackend, ensure_litegs


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--cam", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    root = Path(scene_dir(args.scene))
    ckpt_path = root / "checkpoints" / "splat_latest.pt"
    for line in ensure_litegs():
        print(line)
    device = torch.device("cuda:0")
    backend = LiteGSBackend(root, device, sh_degree=D.SH_DEGREE, resolution_scale=1.0, holdout=True, log=print)
    backend.load_colmap()
    meta, _ctl = backend.load_checkpoint(ckpt_path)
    print(f"loaded step={meta.get('step')} controller={meta.get('controller')} N={backend.n_gaussians}")

    view = backend.views[int(args.cam)]
    print(f"cam {args.cam}: {view.name} {view.width}x{view.height} holdout={view.holdout}")
    frame = backend.render_camera(view, D.SH_DEGREE, with_gt=True)
    if frame is None:
        print("render_camera returned None")
        return 1
    arr = frame.contiguous().cpu().numpy().astype(np.uint8)

    out_path = args.out or (root / "_work" / f"gt_compare_cam{args.cam}.png")
    from PIL import Image

    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(out_path)
    print(f"wrote {out_path} ({arr.shape[1]}x{arr.shape[0]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
