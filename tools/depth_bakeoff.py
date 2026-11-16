"""Compare panorama depth backends. Writes under outputs/<scene>/_work/depth_bakeoff.

Does not touch outputs/<scene>/depth. moge1 runs in this process. The others need
tools/depth_backends/install.ps1. A backend that fails is recorded and skipped.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np

from engine.depth.defaults import BACKENDS, VENV_ROOT
from engine.depth.scene_depth import _guard_convention, _moge1, _normalise_scale
from engine.geo.scene_ref import equirect_to_ortho_views
from engine.job import REPO_ROOT, apply_huggingface_cache_env
from engine.pano.scene_io import pano_path, scene_dir
from engine.world.defaults import MESH_CANVAS
from engine.world.pano_mesh import build_pano_mesh, erp_dirs, fit_ground, fit_planes
from engine.world.render import WorldRenderer
from engine.worldstereo.adapter import pin_K


def _log(msg: str) -> None:
    print(msg, flush=True)


def _vram() -> int:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
            timeout=5,
        )
        return int(out.strip().splitlines()[0])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return 0


def _load_pose(root: Path, spec: str) -> np.ndarray | None:
    if spec == "joint10":
        path = root / "worldstereo" / "input" / "joint_c00" / "traj0" / "camera.json"
        if not path.is_file():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return np.asarray(data["extrinsic"][10], dtype=np.float64)
    rail_s, pose_s = spec.split(":")
    path = root / "rails" / f"rail_{int(rail_s):02d}.json"
    if not path.is_file():
        return None
    poses = np.asarray(json.loads(path.read_text(encoding="utf-8")), dtype=np.float64)
    index = int(pose_s)
    if poses.ndim != 3 or index >= poses.shape[0]:
        return None
    return poses[index]


def _run_other(scene: str, name: str, out: Path) -> None:
    exe = VENV_ROOT / f"depth_{name}" / "Scripts" / "python.exe"
    script = REPO_ROOT / "tools" / "depth_backends" / f"run_{name}.py"
    if not exe.is_file():
        raise RuntimeError(f"depth backend {name} not installed: run tools/depth_backends/install.ps1 {name}")
    env = dict(**{k: v for k, v in __import__("os").environ.items()})
    apply_huggingface_cache_env(env)
    proc = subprocess.run(
        [str(exe), str(script), "--pano", str(pano_path(scene)), "--out", str(out), "--width", "2048", "--height", "1024"],
        env=env,
        timeout=3600,
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0 or not out.is_file():
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()
        raise RuntimeError(tail[-1] if tail else f"exit {proc.returncode}")


def _metrics(distance: np.ndarray, valid: np.ndarray, pano: np.ndarray) -> dict:
    small = cv2.resize(distance, (2048, 1024), interpolation=cv2.INTER_NEAREST)
    mask = cv2.resize(valid.astype(np.uint8), (2048, 1024), interpolation=cv2.INTER_NEAREST) > 0
    finite = mask & np.isfinite(small) & (small > 1e-6)
    dirs = erp_dirs(1024, 2048)[:, :2048]
    below = finite & (dirs[..., 1] > math.sin(math.radians(10.0)))
    scale = float(np.median(small[finite])) if finite.any() else 1.0
    ground = fit_ground((dirs * small[..., None])[below], scale) if int(below.sum()) >= 3 else None
    points = dirs * small[..., None]
    from engine.world.pano_mesh import erp_normals

    normals = erp_normals(np.concatenate([points, points[:, :1]], axis=1))[:, :2048]
    planes = fit_planes(points, normals, finite, np.zeros(finite.shape, dtype=bool), scale)
    rms = []
    for normal, plane_d, plane_mask in planes:
        resid = np.abs(points[plane_mask] @ normal + plane_d)
        if resid.size:
            rms.append(float(np.sqrt(np.mean(resid**2))) / max(scale, 1e-6))
    gray = cv2.cvtColor(cv2.resize(pano, (2048, 1024), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 80, 160) > 0
    log_d = np.log(np.maximum(small, 1e-6))
    grad = np.zeros(log_d.shape, dtype=bool)
    grad[:, 1:] |= np.abs(np.diff(log_d, axis=1)) > 0.03
    grad[1:, :] |= np.abs(np.diff(log_d, axis=0)) > 0.03
    grad &= finite
    dil = cv2.dilate(edges.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    hit = int((grad & dil).sum())
    precision = hit / max(int(grad.sum()), 1)
    recall = hit / max(int(edges.sum()), 1)
    fscore = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {
        "ground_inliers": None if ground is None else float(ground[2]),
        "facade_rms": None if not rms else float(np.median(rms)),
        "edge_f": float(fscore),
    }


def _render_row(distance: np.ndarray, valid: np.ndarray, pano: np.ndarray, poses: list[np.ndarray]) -> tuple[list[np.ndarray], dict]:
    depth = cv2.resize(distance, MESH_CANVAS, interpolation=cv2.INTER_NEAREST)
    mask = cv2.resize(valid.astype(np.uint8), MESH_CANVAS, interpolation=cv2.INTER_NEAREST) > 0
    if bool(mask.any()):
        filled = np.array(depth, copy=True)
        filled[~mask] = 2.0 * float(depth[mask].max())
    else:
        filled = depth
    piece, _stats = build_pano_mesh(filled, mask)
    tex = cv2.resize(pano, MESH_CANVAS, interpolation=cv2.INTER_AREA)
    times = {}
    frames = []
    renderer = WorldRenderer([piece], {"pano": tex}, "cuda:0")
    try:
        k_mat = pin_K()
        for label, pose in zip(("joint10", "a", "b"), poses):
            if pose is None:
                frames.append(np.zeros((480, 832, 3), np.uint8))
                continue
            t0 = time.perf_counter()
            view = renderer.render(pose, k_mat, 480, 832, supersample=1)
            times[label] = time.perf_counter() - t0
            frames.append(view.rgb)
        big = cv2.resize(distance, (4096, 2048), interpolation=cv2.INTER_NEAREST)
        big_mask = cv2.resize(valid.astype(np.uint8), (4096, 2048), interpolation=cv2.INTER_NEAREST) > 0
        t0 = time.perf_counter()
        piece_big, _stats = build_pano_mesh(big, big_mask)
        renderer.close()
        renderer = WorldRenderer([piece_big], {"pano": cv2.resize(pano, (4096, 2048), interpolation=cv2.INTER_AREA)}, "cuda:0")
        if poses[0] is not None:
            renderer.render(poses[0], pin_K(), 480, 832, supersample=1)
        times["canvas_4096"] = time.perf_counter() - t0
    finally:
        renderer.close()
    views = equirect_to_ortho_views(filled, mask, tex, res_max=256)
    floor = np.zeros((256, 256, 3), np.uint8)
    if views and "top" in views:
        rgba = views["top"]["rgba"]
        floor = cv2.resize(rgba[..., :3], (256, 256), interpolation=cv2.INTER_AREA)
    frames.append(floor)
    return frames, times


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default="alpine_01")
    parser.add_argument("--backends", default=",".join(BACKENDS))
    parser.add_argument("--poses", default="joint10,0:40,2:40")
    args = parser.parse_args()
    root = scene_dir(args.scene)
    bgr = cv2.imread(str(pano_path(args.scene)), cv2.IMREAD_COLOR)
    if bgr is None:
        _log(f"STATUS // missing pano {args.scene}")
        return 2
    pano = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    specs = [part.strip() for part in args.poses.split(",") if part.strip()]
    poses = [_load_pose(root, spec) for spec in specs]
    out_root = root / "_work" / "depth_bakeoff"
    out_root.mkdir(parents=True, exist_ok=True)
    rows = []
    summary = []
    for name in [part.strip() for part in args.backends.split(",") if part.strip()]:
        folder = out_root / name
        folder.mkdir(parents=True, exist_ok=True)
        peak = _vram()
        t0 = time.perf_counter()
        try:
            if name == "moge1":
                distance, valid, _normals = _moge1(args.scene, "cuda:0")
            else:
                raw = folder / "raw.npz"
                _run_other(args.scene, name, raw)
                with np.load(raw) as blob:
                    distance = np.asarray(blob["distance"], np.float32)
                    valid = np.asarray(blob["valid"], bool)
                distance, valid, _normals, convention = _guard_convention(distance, valid, None, args.scene)
                if convention != "identity":
                    _log(f"STATUS // bakeoff {name} convention {convention}")
            distance, scale = _normalise_scale(distance, valid)
            peak = max(peak, _vram())
            np.savez_compressed(folder / "depth.npz", distance=distance, valid=valid)
            stats = _metrics(distance, valid, pano)
            frames, times = _render_row(distance, valid, pano, poses)
            stats.update({"seconds": time.perf_counter() - t0, "vram_mib": peak, "scale": scale, "frames_s": times, "error": ""})
            rows.append(frames)
        except Exception as exc:
            stats = {"seconds": time.perf_counter() - t0, "vram_mib": peak, "error": f"{type(exc).__name__}: {exc}"}
            rows.append([np.zeros((256, 256, 3), np.uint8)])
            _log(f"STATUS // bakeoff {name} failed: {exc}")
        (folder / "stats.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
        summary.append({"backend": name, **{k: stats.get(k) for k in ("ground_inliers", "facade_rms", "edge_f", "seconds", "vram_mib", "error")}})
        _log(f"STATUS // bakeoff {name} {stats.get('error') or 'ok'}")
    thumb_h = 240
    tiled = []
    for frames in rows:
        sized = [cv2.resize(frame, (int(frame.shape[1] * thumb_h / max(frame.shape[0], 1)), thumb_h)) for frame in frames]
        tiled.append(np.concatenate(sized, axis=1) if sized else np.zeros((thumb_h, 320, 3), np.uint8))
    width = max(row.shape[1] for row in tiled)
    sheet = np.concatenate([cv2.copyMakeBorder(row, 0, 0, 0, width - row.shape[1], cv2.BORDER_CONSTANT) for row in tiled], axis=0)
    cv2.imwrite(str(out_root / "compare.png"), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    _log(f"STATUS // bakeoff sheet {out_root / 'compare.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
