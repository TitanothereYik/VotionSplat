"""HiRes worker. One process composites every WorldStereo keyframe.

The 8K panorama where that mesh is visible, then this frame's own pixels in
the holes. The shared layer is a guide-only triangle cloud and is not drawn.
Qt never imports this.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

from engine.geo.rails import kept_rail_indices
from engine.hires.pinhole import (
    HIRES_COMPOSITE,
    HIRES_FRAME,
    HIRES_H,
    HIRES_SOURCES,
    HIRES_W,
    ULTRASHARP_FILE,
    composite_world,
    hires_ready,
    scaled_k,
    use_hires,
)
from engine.job import REPO_ROOT, apply_huggingface_cache_env, job_progress
from engine.pano.scene_io import pano_path, scene_dir
from engine.world.defaults import FEATHER_PX, GATE_LAYER, GATE_OWN, GATE_PANO, HIRES_SUPERSAMPLE
from engine.world.pano_mesh import ensure_pano_mesh, load_ground
from engine.world.render import WorldRenderer
from engine.worldstereo.adapter import pin_K
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


def _read_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"missing {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _run_frames(scene: str, pending: list[tuple[int, str, Path]]) -> int:
    upscale = _upscaler(REPO_ROOT / "models" / "upscale_models" / ULTRASHARP_FILE)
    total = len(pending)
    for step, (index, view, folder) in enumerate(pending, start=1):
        shots = _keyframe_shots(folder)
        out_dir = folder / "hires" / HIRES_FRAME
        out_dir.mkdir(parents=True, exist_ok=True)
        for shot in shots:
            image = _fill(_read_rgb(shot), upscale)
            cv2.imwrite(str(out_dir / shot.name), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        use_hires(folder, HIRES_FRAME)
        _log(f"STATUS // HiRes rail {index} {view} upscale frames={len(shots)}")
        print(f"WS\thires\t{index}\t{view}\t1.0000", flush=True)
        job_progress(step, total, f"hires rail {index} {view}")
    _log(f"STATUS // HiRes wrote upscaled WorldStereo frames for {scene}")
    return 0


def run_scene(scene: str, source: str = HIRES_COMPOSITE) -> int:
    if source not in HIRES_SOURCES:
        _log(f"STATUS // HiRes unknown source {source}")
        return 2
    jobs = _view_dirs(scene)
    if not jobs:
        _log("STATUS // no generated views to HiRes")
        return 2
    pending = []
    for job in jobs:
        shots = _keyframe_shots(job[2])
        if not shots:
            continue
        if hires_ready(job[2], source, len(shots)):
            use_hires(job[2], source)
            continue
        pending.append(job)
    _log(f"STATUS // HiRes views={len(jobs)} pending={len(pending)} source={source}")
    if not pending:
        _log("STATUS // HiRes already complete")
        return 0
    if source == HIRES_FRAME:
        return _run_frames(scene, pending)
    erp = _read_rgb(pano_path(scene))
    piece = ensure_pano_mesh(scene, erp, "cuda:0")
    upscale = _upscaler(REPO_ROOT / "models" / "upscale_models" / ULTRASHARP_FILE)
    pano_renderer = WorldRenderer([piece], {"pano": erp}, "cuda:0", ground=load_ground(scene))
    k_hi = scaled_k(pin_K())
    try:
        total = len(pending)
        for step, (index, view, folder) in enumerate(pending, start=1):
            shots = _keyframe_shots(folder)
            camera = json.loads((folder / "camera.json").read_text(encoding="utf-8"))
            w2cs = np.asarray(camera["extrinsic"], dtype=np.float64)
            if int(w2cs.shape[0]) != len(shots):
                _log(f"STATUS // rail {index} {view} has {len(shots)} frames and {int(w2cs.shape[0])} poses")
                return 2
            hires = folder / "hires"
            hires.mkdir(parents=True, exist_ok=True)
            stamp = hires / "source.txt"
            if stamp.is_file():
                stamp.unlink()
            gates = folder / "gate_masks"
            gates.mkdir(parents=True, exist_ok=True)
            totals = np.zeros(3, dtype=np.float64)
            for enum_i, shot in enumerate(shots):
                own = _fill(_read_rgb(shot), upscale)
                pano = pano_renderer.render(
                    w2cs[enum_i], k_hi, HIRES_H, HIRES_W, supersample=HIRES_SUPERSAMPLE
                )
                out, gate = composite_world(
                    pano.rgb,
                    pano.empty,
                    None,
                    None,
                    own,
                    FEATHER_PX,
                    GATE_PANO,
                    GATE_LAYER,
                    GATE_OWN,
                )
                cv2.imwrite(str(hires / shot.name), cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(gates / shot.name), gate)
                totals[0] += float(np.mean(gate == GATE_PANO))
                totals[1] += float(np.mean(gate == GATE_LAYER))
                totals[2] += float(np.mean(gate == GATE_OWN))
            use_hires(folder, HIRES_COMPOSITE)
            count = max(1, len(shots))
            pano_f, layer_f, own_f = (totals / count).tolist()
            _log(
                f"STATUS // HiRes rail {index} {view} pano={pano_f:.3f} layer={layer_f:.3f} "
                f"own={own_f:.3f} frames={len(shots)}"
            )
            print(f"WS\thires\t{index}\t{view}\t{pano_f:.4f}", flush=True)
            job_progress(step, total, f"hires rail {index} {view}")
    finally:
        pano_renderer.close()
    _log("STATUS // HiRes wrote world frames")
    return 0


def main(argv: list[str] | None = None) -> int:
    import argparse

    apply_huggingface_cache_env(os.environ)
    parser = argparse.ArgumentParser(description="Votion HiRes — panorama, then the WorldStereo frame in the holes")
    parser.add_argument("--scene", required=True)
    parser.add_argument("--source", choices=HIRES_SOURCES, default=HIRES_COMPOSITE)
    args = parser.parse_args(argv)
    return run_scene(args.scene, args.source)


if __name__ == "__main__":
    sys.exit(main())
