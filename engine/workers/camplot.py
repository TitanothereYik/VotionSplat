"""Plot Camera worker: spline one rail, mesh-render control_video + control_mask.

Qt never imports this. Run as ``python -m engine.workers.camplot``.

Video container: mp4 + fourcc ``mp4v``, fps 10 — copied from SplatKit
``vendored/utils_3dscene/pipeline_utils_3dscene.write_video``. Mask is white=valid,
black=hole (SplatKit IMAGE convention). Phase 3 paths are per-rail so four rails
cannot clobber a shared condition_dir:

``control/rail_00/control_video.mp4``, ``rails/rail_00.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SPLATKIT = REPO_ROOT / "vendor" / "splatkit"
if str(SPLATKIT) not in sys.path:
    sys.path.insert(0, str(SPLATKIT))

from engine.geo import defaults as D
from engine.geo.rail import build_w2c, snap_length
from engine.geo.rails import (
    camplot_meta_rel,
    condition_dir_rel,
    control_mask_rel,
    control_video_rel,
    rail_json_rel,
)
from engine.job import job_progress
from engine.pano.scene_io import pano_path, scene_dir


def log(msg: str) -> None:
    print(msg, flush=True)


def moge_ckpt() -> Path:
    return REPO_ROOT / "models" / "MoGe" / "model.pt"


def write_mp4(frames_bgr: list[np.ndarray], path: Path, fps: int = 10) -> None:
    """Match SplatKit ``write_video``: mp4v, fps default 10."""
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames_bgr[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(path), fourcc, float(fps), (width, height))
    if not out.isOpened():
        raise RuntimeError(f"VideoWriter failed to open {path}")
    for frame in frames_bgr:
        out.write(frame)
    out.release()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Votion Plot Camera (mesh control video)")
    parser.add_argument("--scene", required=True)
    parser.add_argument("--request", type=Path, default=None)
    parser.add_argument("--rail", type=int, default=None)
    args = parser.parse_args(argv)

    root = scene_dir(args.scene)
    req_path = args.request or (root / "_work" / "camplot_request.json")
    if not req_path.is_file():
        log(f"STATUS // missing camplot request {req_path}")
        return 2
    req = json.loads(req_path.read_text(encoding="utf-8"))
    orientation = str(req.get("orientation") or D.LOOK_DEFAULT)
    if orientation not in D.LOOK_MODES:
        orientation = D.LOOK_DEFAULT
    length = snap_length(int(req.get("length") or D.LENGTH_DEFAULT))
    moge_level = max(0, min(9, int(req.get("moge_level") or D.MOGE_LEVEL)))
    anchors_text = str(req.get("anchors") or D.ANCHORS_DEFAULT)
    look_at = str(req.get("look_at") or D.LOOK_AT_DEFAULT)
    if args.rail is not None:
        rail_index = int(args.rail)
    else:
        rail_index = int(req.get("rail_index") or 0)

    pano_file = pano_path(args.scene)
    if not pano_file.is_file():
        log("STATUS // missing pano.png — run Phase 1 first")
        return 2
    ckpt = moge_ckpt()
    if not ckpt.is_file():
        log("MISSING_moge https://huggingface.co/Ruicheng/moge-vitl/resolve/main/model.pt")
        return 3

    idx = int(os.environ.get("CUDA_DEVICE_INDEX", "0") or 0)
    device = f"cuda:{idx}"
    log(
        f"STATUS // camplot start scene={args.scene} rail={rail_index} {orientation} "
        f"length={length} moge_level={moge_level} device={device}"
    )
    job_progress(1, 5, "build rail")

    w2c = build_w2c(anchors_text, orientation, look_at, length)
    work = root / "_work"
    rail_rel = rail_json_rel(rail_index)
    video_rel = control_video_rel(rail_index)
    mask_rel = control_mask_rel(rail_index)
    meta_rel = camplot_meta_rel(rail_index)
    cond_rel = condition_dir_rel(rail_index)
    rail_path = root / rail_rel
    video_path = root / video_rel
    mask_path = root / mask_rel
    cond = root / cond_rel
    for folder in (work, rail_path.parent, video_path.parent, cond):
        folder.mkdir(parents=True, exist_ok=True)
    payload = w2c.tolist()
    rail_path.write_text(json.dumps(payload), encoding="utf-8")
    log(f"CAMPLOT\trail\t{rail_path}")

    bgr = cv2.imread(str(pano_file), cv2.IMREAD_COLOR)
    if bgr is None:
        log(f"STATUS // failed to read {pano_file}")
        return 2
    pano = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    from core import matrix3d_pipeline as mp

    job_progress(2, 5, "load MoGe")
    model = mp.get_moge_model(ckpt=str(ckpt), device=device)
    job_progress(3, 5, "render mesh flight")
    res = mp.render_control(
        pano,
        movement_mode="straight",
        movement_range=1.0,
        angle=0.0,
        frame_size=int(length),
        json_path=str(rail_path),
        scale_mode="absolute_literal",
        moge_ckpt=str(ckpt),
        model=model,
        device=device,
        moge_level=moge_level,
    )
    rgb = np.clip(res["rendered_rgb"] * 255.0, 0, 255).astype(np.uint8)
    mask = res["rendered_mask"].astype(np.uint8) * 255
    frames_rgb = [cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) for frame in rgb]
    frames_mask = [cv2.cvtColor(m, cv2.COLOR_GRAY2BGR) for m in mask]
    job_progress(4, 5, "write control video")
    write_mp4(frames_rgb, video_path)
    write_mp4(frames_mask, mask_path)
    np.savez(cond / "cameras.npz", res["cameras"])
    ff_depth = res["firstframe_depth"].astype(np.float32)
    # Same persist as SplatKit camera_plot: try EXR, else float32 npy.
    try:
        exr_ok = cv2.imwrite(str(cond / "firstframe_depth.exr"), ff_depth)
    except cv2.error:
        exr_ok = False
    if not exr_ok:
        np.save(cond / "firstframe_depth.npy", ff_depth)
    ff_mask = (ff_depth < 0.9 * float(ff_depth.max())).astype(np.uint8) * 255
    cv2.imwrite(str(cond / "firstframe_mask.png"), ff_mask)

    meta = {
        "rail_index": rail_index,
        "orientation": orientation,
        "length": length,
        "moge_level": moge_level,
        "fps": 10,
        "fourcc": "mp4v",
        "rail": rail_rel,
        "video": video_rel,
        "mask": mask_rel,
        "anchors": anchors_text,
        "look_at": look_at,
    }
    (root / meta_rel).write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    log(f"CAMPLOT\tvideo\t{video_path}")
    log(f"CAMPLOT\tmask\t{mask_path}")
    log(f"STATUS // camplot done {length} frames -> {video_path}")
    job_progress(5, 5, "camplot done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
