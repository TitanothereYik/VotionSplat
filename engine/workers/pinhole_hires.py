"""HiRes worker. One process upscales every WorldStereo keyframe.

Qt never imports this. 4x-UltraSharp loads once. The panorama is not reprojected.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

from engine.geo.rails import kept_rail_indices
from engine.hires.pinhole import HIRES_H, HIRES_SOURCE, HIRES_W, ULTRASHARP_FILE, hires_complete
from engine.job import REPO_ROOT, apply_huggingface_cache_env, job_progress
from engine.pano.scene_io import scene_dir
from engine.worldstereo.views import VIEWS


def _log(msg: str) -> None:
    print(msg, flush=True)


def _upscaler(path: Path):
    if not path.is_file() or path.stat().st_size <= 0:
        _log("STATUS // HiRes: no 4x-UltraSharp — bicubic")
        return None
    import torch
    from spandrel import ModelLoader

    device = torch.device("cuda")
    model = ModelLoader().load_from_file(str(path))
    model.to(device).eval()
    _log(f"STATUS // HiRes upscaler {path.name} scale={float(getattr(model, 'scale', 4) or 4)}")

    def run(img_u8: np.ndarray) -> np.ndarray:
        tensor = torch.from_numpy(img_u8.astype(np.float32) / 255.0).permute(2, 0, 1)[None].to(device)
        with torch.no_grad():
            out = model(tensor.float())
        image = (out[0].permute(1, 2, 0).clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
        del tensor, out
        return image

    return run


def _fill(keyframe: np.ndarray, upscale) -> np.ndarray:
    if upscale is None:
        return cv2.resize(keyframe, (HIRES_W, HIRES_H), interpolation=cv2.INTER_CUBIC)
    big = upscale(keyframe)
    return cv2.resize(big, (HIRES_W, HIRES_H), interpolation=cv2.INTER_AREA)


def _view_dirs(scene: str) -> list[tuple[int, str, Path]]:
    root = scene_dir(scene) / "worldstereo" / "rails"
    kept = set(kept_rail_indices(scene))
    found: list[tuple[int, str, Path]] = []
    for index in sorted(kept):
        rail = root / f"rail_{index:02d}"
        if not rail.is_dir():
            continue
        for view in VIEWS:
            folder = rail / view
            frames = folder / "keyframes"
            if (folder / "camera.json").is_file() and frames.is_dir() and any(frames.glob("[0-9][0-9][0-9].png")):
                found.append((index, view, folder))
    return found


def _keyframe_shots(folder: Path) -> list[Path]:
    frames = folder / "keyframes"
    if not frames.is_dir():
        return []
    return sorted(frames.glob("[0-9][0-9][0-9].png"))


def run_scene(scene: str) -> int:
    jobs = _view_dirs(scene)
    if not jobs:
        _log("STATUS // no generated views to HiRes")
        return 2
    pending = []
    for job in jobs:
        shots = _keyframe_shots(job[2])
        if not shots or not hires_complete(job[2], len(shots)):
            pending.append(job)
    _log(f"STATUS // HiRes views={len(jobs)} pending={len(pending)} source={HIRES_SOURCE}")
    if not pending:
        _log("STATUS // HiRes already complete")
        return 0
    upscale = _upscaler(REPO_ROOT / "models" / "upscale_models" / ULTRASHARP_FILE)
    total = len(pending)
    for step, (index, view, folder) in enumerate(pending, start=1):
        shots = _keyframe_shots(folder)
        hires = folder / "hires"
        hires.mkdir(parents=True, exist_ok=True)
        stamp = hires / "source.txt"
        if stamp.is_file():
            stamp.unlink()
        gates = folder / "gate_masks"
        if gates.is_dir():
            shutil.rmtree(gates)
        for shot in shots:
            key_bgr = cv2.imread(str(shot), cv2.IMREAD_COLOR)
            if key_bgr is None:
                _log(f"STATUS // missing keyframe {shot}")
                return 2
            key = cv2.cvtColor(key_bgr, cv2.COLOR_BGR2RGB)
            filled = _fill(key, upscale)
            cv2.imwrite(str(hires / shot.name), cv2.cvtColor(filled, cv2.COLOR_RGB2BGR))
        stamp.write_text(HIRES_SOURCE + "\n", encoding="utf-8")
        _log(f"STATUS // HiRes rail {index} {view} frames={len(shots)} size={HIRES_W}x{HIRES_H}")
        print(f"WS\thires\t{index}\t{view}\t{len(shots)}", flush=True)
        job_progress(step, total, f"hires rail {index} {view}")
    _log("STATUS // HiRes wrote the WorldStereo frames")
    return 0


def main(argv: list[str] | None = None) -> int:
    import argparse

    apply_huggingface_cache_env(os.environ)
    parser = argparse.ArgumentParser(description="Votion HiRes — upscale WorldStereo keyframes")
    parser.add_argument("--scene", required=True)
    args = parser.parse_args(argv)
    return run_scene(args.scene)


if __name__ == "__main__":
    sys.exit(main())
