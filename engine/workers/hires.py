"""engine.workers.hires — one rail per process. 8K PNGs to disk, never one GPU tensor.

Qt never imports this. Run as ``python -m engine.workers.hires --scene NAME --rail N``.
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

REPO_ROOT = Path(__file__).resolve().parents[2]
SPLATKIT = REPO_ROOT / "vendor" / "splatkit"
if str(SPLATKIT) not in sys.path:
    sys.path.insert(0, str(SPLATKIT))

from engine.geo.rails import (
    rail_json_rel,
    set_dataset_rail,
    wan_video_rel,
)
from engine.job import apply_huggingface_cache_env, job_progress, worker_env
from engine.pano.scene_io import pano_path, scene_dir
from engine.video import defaults as V


def log(msg: str) -> None:
    print(msg, flush=True)


def moge_ckpt() -> Path:
    return REPO_ROOT / "models" / "MoGe" / "model.pt"


def ultrasharp_path() -> Path:
    return REPO_ROOT / "models" / "upscale_models" / V.HOLE_UPSCALER


def _read_mp4_u8(path: Path) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {path}")
    frames = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise RuntimeError(f"no frames in {path}")
    return np.stack(frames, axis=0)


def _load_pano_u8(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"failed to read {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _load_rail(path: Path) -> np.ndarray:
    rail = np.asarray(json.loads(path.read_text(encoding="utf-8")), dtype=np.float32)
    if rail.ndim != 3 or rail.shape[1:] != (4, 4):
        raise RuntimeError(f"{path} is not a stack of 4x4 matrices (got {rail.shape})")
    return rail


def _upscaler(path: Path):
    if not path.is_file() or path.stat().st_size <= 0:
        log("STATUS // HiRes hole fill: no 4x-UltraSharp — bicubic")
        return None
    from spandrel import ModelLoader

    device = torch.device("cuda")
    model = ModelLoader().load_from_file(str(path))
    model.to(device).eval()
    scale = float(getattr(model, "scale", 4) or 4)
    log(f"STATUS // HiRes upscaler {path.name} scale={scale}")

    def run(img_u8: np.ndarray) -> np.ndarray:
        t = torch.from_numpy(img_u8.astype(np.float32) / 255.0).permute(2, 0, 1)[None].to(device)
        with torch.no_grad(), torch.autocast("cuda", torch.float16):
            out = model(t)
        if not torch.isfinite(out).all():
            with torch.no_grad():
                out = model(t.float())
        o = (out[0].permute(1, 2, 0).clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
        del t, out
        return o

    return run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Votion HiRes Composite (one rail)")
    parser.add_argument("--scene", required=True)
    parser.add_argument("--rail", type=int, required=True)
    parser.add_argument("--request", type=Path, default=None)
    args = parser.parse_args(argv)

    os.environ.update(worker_env(int(os.environ.get("CUDA_DEVICE_INDEX", "0"))))
    apply_huggingface_cache_env(os.environ)

    root = scene_dir(args.scene)
    req_path = args.request or (root / "_work" / "generate_request.json")
    req = json.loads(req_path.read_text(encoding="utf-8")) if req_path.is_file() else {}
    rail_index = int(args.rail)
    frames_spec = str(req.get("frames") or V.HIRES_FRAMES)
    debug_save = str(req.get("debug_save") or V.HIRES_DEBUG_SAVE_PRODUCT)
    out_w = int(req.get("hires_output_width") or V.HIRES_OUTPUT_WIDTH)
    base_mode = str(req.get("hires_base_mode") or V.HIRES_BASE_MODE)
    moge_level = int(req.get("hires_moge_level") or V.HIRES_MOGE_LEVEL)
    gate_edge = float(req.get("hires_gate_edge", V.HIRES_GATE_EDGE))
    edge_erode = int(req.get("hires_edge_erode", V.HIRES_EDGE_ERODE))
    tone_mode = str(req.get("hires_tone_mode") or V.HIRES_TONE_MODE)

    pano_file = pano_path(args.scene)
    if not pano_file.is_file():
        log(f"STATUS // missing pano {pano_file}")
        return 2
    src_hi = _load_pano_u8(pano_file)
    if src_hi.shape[1] < int(V.HIRES_OUTPUT_WIDTH):
        log(
            f"STATUS // refuse Quality HiRes — pano is {src_hi.shape[1]} wide, need "
            f"{V.HIRES_OUTPUT_WIDTH}"
        )
        return 2
    rail_path = root / rail_json_rel(rail_index)
    if not rail_path.is_file():
        log(f"STATUS // missing rail json {rail_path} — Play this rail on Geometry first")
        return 2
    wan_path = root / wan_video_rel(rail_index)
    if not wan_path.is_file():
        log(f"STATUS // missing WAN clip {wan_path} — Generate WAN and confirm first")
        return 2
    ckpt = moge_ckpt()
    if not ckpt.is_file():
        log("MISSING_moge https://huggingface.co/Ruicheng/moge-vitl/resolve/main/model.pt")
        return 3

    from core import hires_composite as hc
    from core import matrix3d_pipeline as mp

    idx = int(os.environ.get("CUDA_DEVICE_INDEX", "0") or 0)
    device = f"cuda:{idx}"
    name_prefix = f"traj{rail_index:02d}_frame_"
    out_dir = root / "hires"
    out_dir.mkdir(parents=True, exist_ok=True)
    log(f"HIRES\trail\t{rail_index}\t{wan_path}")
    log(
        f"STATUS // hires rail={rail_index} {frames_spec} {out_w} {base_mode} "
        f"gate_edge={gate_edge:g} edge_erode={edge_erode} tone={tone_mode}"
    )
    job_progress(0, 1, f"hires rail {rail_index}")

    wan = _read_mp4_u8(wan_path)
    rail = _load_rail(rail_path)
    model = mp.get_moge_model(ckpt=str(ckpt), device=device)
    moge_kwargs = dict(
        resolution_level=int(moge_level),
        merge_long=int(V.HIRES_MERGE_LONG),
        merge_short=int(V.HIRES_MERGE_LONG) // 2,
        model=model,
        ckpt=str(ckpt),
    )
    last = {"n": 0}

    def progress(done, total):
        last["n"] = int(done)
        job_progress(int(done), max(int(total), 1), f"hires {done}/{total}")
        proxy = out_dir / "proxies" / f"{name_prefix}{max(int(done) - 1, 0):04d}.png"
        if proxy.is_file():
            log(f"HIRES\tframe\t{rail_index}\t{int(done) - 1}\t\t{proxy}")

    add_path = rail_index >= 4
    try:
        res = hc.run_composite(
            src_hi,
            src_hi,
            rail,
            str(out_dir),
            wan=wan,
            out_w=int(out_w if not add_path else 4096),
            base_mode=base_mode,
            frames_spec=frames_spec,
            upscale=_upscaler(ultrasharp_path()),
            proxy_width=int(V.HIRES_PROXY_WIDTH),
            name_prefix=name_prefix,
            device=device,
            moge_kwargs=moge_kwargs,
            depth_grid=V.HIRES_DEPTH_GRID,
            prefetch=bool(V.HIRES_PREFETCH),
            params=dict(
                geom_scale=int(V.HIRES_GEOM_SCALE),
                rho_hi=float(V.HIRES_RHO_HI),
                tone_work=int(V.HIRES_TONE_WORK),
                tone_mode=tone_mode,
                gate_mode=V.HIRES_GATE_MODE,
                gate_edge=float(gate_edge),
                edge_erode=int(edge_erode),
                disc_erode=int(V.HIRES_DISC_ERODE),
                bg_fill=V.HIRES_BG_FILL,
            ),
            debug_save=debug_save,
            progress=progress,
            save_proxies=bool(V.HIRES_SAVE_PROXIES),
        )
    except Exception as exc:
        log(f"STATUS // hires failed {type(exc).__name__}: {exc}")
        import traceback

        traceback.print_exc()
        return 1

    cov = float(res.get("coverage_mean") or 0.0)
    log(f"HIRES\tcoverage\t{rail_index}\t{cov:.4f}")
    log(f"HIRES\tdone\t{rail_index}\t{res.get('frames_dir')}\t{cov:.4f}")
    log(f"STATUS // hires done rail={rail_index} coverage={cov:.3f} -> {res.get('frames_dir')}")
    set_dataset_rail(args.scene, rail_index, status="hires", hires=True)
    job_progress(1, 1, "hires done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
