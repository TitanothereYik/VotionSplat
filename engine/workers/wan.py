"""engine.workers.wan — one rail per process. Load 14B, sample, unload, exit.

Qt never imports this. Run as ``python -m engine.workers.wan --scene NAME --rail N``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from engine.geo.rails import (
    confirmed_path,
    control_mask_rel,
    control_video_rel,
    set_dataset_rail,
    wan_video_rel,
)
from engine.job import apply_huggingface_cache_env, job_progress, worker_env
from engine.pano.defaults import wan_prompt_from_stored
from engine.pano.scene_io import pano_path, prompt_path, scene_dir
from engine.video import defaults as V
from engine.video.wan_runtime import log, run_rail


def _read_mp4(path: Path) -> torch.Tensor:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {path}")
    frames = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        frames.append(rgb.astype(np.float32) / 255.0)
    cap.release()
    if not frames:
        raise RuntimeError(f"no frames in {path}")
    return torch.from_numpy(np.stack(frames, axis=0))


def _pano_rgb(path: Path) -> np.ndarray:
    from PIL import Image

    im = Image.open(path).convert("RGB")
    return np.asarray(im, dtype=np.uint8)


def _control_paths(root: Path, scene: str, rail: int) -> tuple[Path, Path]:
    video = root / control_video_rel(rail)
    mask = root / control_mask_rel(rail)
    gate = confirmed_path(scene)
    if gate.is_file():
        data = json.loads(gate.read_text(encoding="utf-8"))
        rows = data.get("rails")
        if isinstance(rows, list):
            for row in rows:
                if int(row.get("index", -1)) == int(rail):
                    video = root / str(row.get("video") or control_video_rel(rail))
                    mask = root / str(row.get("mask") or control_mask_rel(rail))
                    break
    if not video.is_file():
        legacy = root / "control" / "control_video.mp4"
        if int(rail) == 0 and legacy.is_file():
            video = legacy
    if not mask.is_file():
        legacy_m = root / "control" / "control_mask.mp4"
        if int(rail) == 0 and legacy_m.is_file():
            mask = legacy_m
    return video, mask


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Votion WAN I2V (one rail)")
    parser.add_argument("--scene", required=True)
    parser.add_argument("--rail", type=int, required=True)
    parser.add_argument("--request", type=Path, default=None)
    args = parser.parse_args(argv)

    os.environ.update(worker_env(int(os.environ.get("CUDA_DEVICE_INDEX", "0"))))
    apply_huggingface_cache_env(os.environ)

    root = scene_dir(args.scene)
    req_path = args.request or (root / "_work" / "generate_request.json")
    if not req_path.is_file():
        log(f"STATUS // missing generate request {req_path}")
        return 2
    req = json.loads(req_path.read_text(encoding="utf-8"))
    raw = str(req.get("prompt") or "").strip()
    if not raw:
        raw = prompt_path(args.scene).read_text(encoding="utf-8").strip() if prompt_path(args.scene).is_file() else ""
    prompt_body = wan_prompt_from_stored(raw)
    if not prompt_body:
        log("STATUS // empty WAN prompt blocks Generate")
        return 2
    style = str(req.get("style_suffix") or "").strip()
    prompt = f"{prompt_body} {style}".strip() if style else prompt_body
    negative = str(req.get("negative") or V.NEGATIVE)
    seed = int(req.get("seed") if req.get("seed") is not None else V.WAN_SEED)
    length = int(req.get("wan_length") or V.WAN_LENGTH)
    width = int(req.get("wan_width") or V.WAN_WIDTH)
    height = int(req.get("wan_height") or V.WAN_HEIGHT)
    steps = int(req.get("wan_steps") or V.WAN_STEPS)

    pano = pano_path(args.scene)
    if not pano.is_file():
        log(f"STATUS // missing pano {pano}")
        return 2
    video_path, mask_path = _control_paths(root, args.scene, int(args.rail))
    if not video_path.is_file():
        log(f"STATUS // missing control video {video_path} — Play this rail on Geometry first")
        return 2
    if not mask_path.is_file():
        log(f"STATUS // missing control mask {mask_path} — Play this rail on Geometry first")
        return 2

    job_progress(0, 10, f"wan rail {int(args.rail)}")
    log(f"WAN\trail\t{int(args.rail)}\t{mask_path}\t{video_path}")
    log(f"STATUS // wan rail={int(args.rail)} {video_path.name} seed={seed} steps={steps}")
    try:
        rgb = _read_mp4(video_path)
        mask = _read_mp4(mask_path)
        out = root / wan_video_rel(int(args.rail))
        preview_dir = root / "_work" / "wan_preview" / f"rail_{int(args.rail):02d}"
        w, h = run_rail(
            prompt=prompt,
            negative=negative,
            pano_rgb=_pano_rgb(pano),
            control_rgb=rgb,
            control_mask=mask,
            out_mp4=out,
            seed=seed,
            width=width,
            height=height,
            length=length,
            steps=steps,
            rail_index=int(args.rail),
            preview_dir=preview_dir,
        )
        log(f"WAN\tvideo\t{out}\t{w}x{h}")
        set_dataset_rail(args.scene, int(args.rail), status="wan", hires=False)
        log("STATUS // wan done")
        return 0
    except Exception as exc:
        log(f"STATUS // wan failed {type(exc).__name__}: {exc}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
