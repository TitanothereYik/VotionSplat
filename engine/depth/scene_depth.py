"""One cached panorama depth for Geometry, Plot Camera, the guide, and HiRes."""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from engine.depth.defaults import (
    CONVENTION_EDGE_MARGIN,
    DEPTH_BACKEND,
    DEPTH_CANVAS,
    MOGE1_CANVAS,
    MOGE1_LEVEL,
    MOGE1_MERGE,
    RUNNER_TIMEOUT_S,
    SCALE_CAMERA_HEIGHT,
    SCALE_MEDIAN_FALLBACK,
    VENV_ROOT,
)
from engine.depth.keys import depth_dir, scene_dir
from engine.job import REPO_ROOT, apply_huggingface_cache_env
from engine.pano.scene_io import pano_path


@dataclass
class SceneDepth:
    distance: np.ndarray
    valid: np.ndarray
    normals: np.ndarray | None
    backend: str
    key: str
    scale: float


def pano_sha1(scene: str) -> str:
    digest = hashlib.sha1()
    with pano_path(scene).open("rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def ensure_scene_depth(scene: str, device: str) -> SceneDepth:
    folder = depth_dir(scene)
    folder.mkdir(parents=True, exist_ok=True)
    sha = pano_sha1(scene)
    backend = str(DEPTH_BACKEND)
    cached = _load_cache(folder, backend, sha)
    if cached is not None:
        print(f"STATUS // depth cache {cached.key}", flush=True)
        return cached
    started = time.perf_counter()
    if backend == "moge1":
        distance, valid, normals = _moge1(scene, device)
        convention = "identity"
    else:
        distance, valid, normals = _subprocess_backend(scene, backend)
        distance, valid, normals, convention = _guard_convention(distance, valid, normals, scene)
    distance, scale = _normalise_scale(distance, valid)
    _check_axes(distance, valid)
    seconds = time.perf_counter() - started
    key = f"{backend}:{sha}"
    _write_cache(folder, distance, valid, normals, backend, sha, scale, seconds, convention, key)
    print(
        f"STATUS // depth {backend} {distance.shape[1]}x{distance.shape[0]} "
        f"scale={scale:.4f} {seconds:.1f}s",
        flush=True,
    )
    return SceneDepth(distance, valid, normals, backend, key, scale)


def depth_at(sd: SceneDepth, size: tuple[int, int], fill_sky: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Resize to ``(width, height)``. Nearest, so edges do not grow flying points."""
    width, height = int(size[0]), int(size[1])
    distance = cv2.resize(np.asarray(sd.distance, dtype=np.float32), (width, height), interpolation=cv2.INTER_NEAREST)
    valid = cv2.resize(sd.valid.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST) > 0
    if fill_sky:
        distance = np.array(distance, copy=True)
        if bool(valid.any()):
            distance[~valid] = 2.0 * float(distance[valid].max())
        else:
            distance[:] = 1.0
    return distance.astype(np.float32), valid


def _load_cache(folder: Path, backend: str, sha: str) -> SceneDepth | None:
    meta_path = folder / "depth.json"
    data_path = folder / "pano_depth.npz"
    if not meta_path.is_file() or not data_path.is_file():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(meta, dict):
        return None
    if str(meta.get("backend") or "") != backend or str(meta.get("pano_sha1") or "") != sha:
        return None
    with np.load(data_path) as blob:
        distance = np.asarray(blob["distance"], dtype=np.float32)
        valid = np.asarray(blob["valid"], dtype=bool)
        normals = np.asarray(blob["normals"], dtype=np.float32) if "normals" in blob.files else None
    return SceneDepth(
        distance,
        valid,
        normals,
        backend,
        f"{backend}:{sha}",
        float(meta.get("scale") or 1.0),
    )


def _write_cache(
    folder: Path,
    distance: np.ndarray,
    valid: np.ndarray,
    normals: np.ndarray | None,
    backend: str,
    sha: str,
    scale: float,
    seconds: float,
    convention: str,
    key: str,
) -> None:
    payload = {
        "distance": np.asarray(distance, dtype=np.float32),
        "valid": np.asarray(valid, dtype=bool),
    }
    if normals is not None:
        payload["normals"] = np.asarray(normals, dtype=np.float32)
    np.savez_compressed(folder / "pano_depth.npz", **payload)
    meta = {
        "backend": backend,
        "pano_sha1": sha,
        "shape": [int(distance.shape[0]), int(distance.shape[1])],
        "scale": float(scale),
        "seconds": float(seconds),
        "convention": convention,
        "key": key,
    }
    (folder / "depth.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


def _moge1(scene: str, device: str) -> tuple[np.ndarray, np.ndarray, None]:
    import sys

    splatkit = REPO_ROOT / "vendor" / "splatkit"
    if str(splatkit) not in sys.path:
        sys.path.insert(0, str(splatkit))
    import cv2 as cv

    from core import matrix3d_pipeline as mp

    bgr = cv.imread(str(pano_path(scene)), cv.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"missing {pano_path(scene)}")
    rgb = cv.cvtColor(bgr, cv.COLOR_BGR2RGB)
    small = cv.resize(rgb, MOGE1_CANVAS, interpolation=cv.INTER_AREA)
    ckpt = REPO_ROOT / "models" / "MoGe" / "model.pt"
    model = mp.get_moge_model(ckpt=str(ckpt), device=device)
    depth, mask = mp.moge_panorama_depth(
        small,
        model=model,
        ckpt=str(ckpt),
        device=device,
        resolution_level=MOGE1_LEVEL,
        merge_long=int(MOGE1_MERGE[0]),
        merge_short=int(MOGE1_MERGE[1]),
    )
    model.to("cpu")
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return np.asarray(depth, dtype=np.float32), np.asarray(mask, dtype=bool), None


def _subprocess_backend(scene: str, backend: str) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    exe = VENV_ROOT / f"depth_{backend}" / "Scripts" / "python.exe"
    script = REPO_ROOT / "tools" / "depth_backends" / f"run_{backend}.py"
    if not exe.is_file() or not script.is_file():
        raise RuntimeError(f"depth backend {backend} not installed: run tools/depth_backends/install.ps1 {backend}")
    folder = depth_dir(scene)
    out = folder / f"_{backend}_raw.npz"
    env = os.environ.copy()
    apply_huggingface_cache_env(env)
    width, height = DEPTH_CANVAS
    proc = subprocess.run(
        [
            str(exe),
            str(script),
            "--pano",
            str(pano_path(scene)),
            "--out",
            str(out),
            "--width",
            str(width),
            "--height",
            str(height),
        ],
        env=env,
        timeout=RUNNER_TIMEOUT_S,
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0 or not out.is_file():
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()
        detail = tail[-1] if tail else f"exit {proc.returncode}"
        raise RuntimeError(f"depth backend {backend} failed: {detail}")
    with np.load(out) as blob:
        distance = np.asarray(blob["distance"], dtype=np.float32)
        valid = np.asarray(blob["valid"], dtype=bool)
        normals = np.asarray(blob["normals"], dtype=np.float32) if "normals" in blob.files else None
    return distance, valid, normals


def _guard_convention(
    distance: np.ndarray,
    valid: np.ndarray,
    normals: np.ndarray | None,
    scene: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, str]:
    ref_path = scene_dir(scene) / "_work" / "depth_bakeoff" / "moge1" / "depth.npz"
    if not ref_path.is_file():
        bgr = cv2.imread(str(pano_path(scene)), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"missing {pano_path(scene)}")
        name, score = _edge_convention(distance, valid, bgr)
        if name != "identity":
            distance, valid, normals = _apply_convention(distance, valid, normals, name)
            print(f"STATUS // depth convention {name} edge gain={score:.3f}", flush=True)
        return distance, valid, normals, name
    with np.load(ref_path) as blob:
        reference = cv2.resize(np.asarray(blob["distance"], dtype=np.float32), (1024, 512), interpolation=cv2.INTER_NEAREST)
        ref_valid = cv2.resize(np.asarray(blob["valid"], dtype=np.uint8), (1024, 512), interpolation=cv2.INTER_NEAREST) > 0
    small = cv2.resize(distance, (1024, 512), interpolation=cv2.INTER_NEAREST)
    small_valid = cv2.resize(valid.astype(np.uint8), (1024, 512), interpolation=cv2.INTER_NEAREST) > 0
    name, score = _best_convention(small, small_valid, reference, ref_valid)
    if name != "identity":
        distance, valid, normals = _apply_convention(distance, valid, normals, name)
        print(f"STATUS // depth convention {name} corr={score:.3f}", flush=True)
    return distance, valid, normals, name


def _best_convention(
    distance: np.ndarray,
    valid: np.ndarray,
    reference: np.ndarray,
    ref_valid: np.ndarray,
) -> tuple[str, float]:
    from scipy.stats import spearmanr

    names = ("identity", "flip", "roll", "flip_roll")
    best_name = "identity"
    best_score = -2.0
    for name in names:
        trial, trial_valid, _normals = _apply_convention(distance, valid, None, name)
        both = trial_valid & ref_valid & np.isfinite(trial) & np.isfinite(reference)
        if int(both.sum()) < 1000:
            continue
        score = float(spearmanr(trial[both], reference[both]).statistic)
        if math.isfinite(score) and score > best_score:
            best_score = score
            best_name = name
    return best_name, best_score


def _edge_magnitude(image: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(image.astype(np.float32), (0, 0), 1.0)
    gx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3, borderType=cv2.BORDER_REFLECT)
    gy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3, borderType=cv2.BORDER_REFLECT)
    return np.sqrt(gx * gx + gy * gy)


def _edge_convention(distance: np.ndarray, valid: np.ndarray, pano_bgr: np.ndarray) -> tuple[str, float]:
    """Pick the convention whose depth edges line up with the panorama's colour edges.

    Returns ``identity`` unless another convention beats it by ``CONVENTION_EDGE_MARGIN``.
    """
    from scipy.stats import spearmanr

    size = (1024, 512)
    gray = cv2.cvtColor(cv2.resize(pano_bgr, size, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    image_edges = _edge_magnitude(gray)
    scores: dict[str, float] = {}
    for name in ("identity", "flip", "roll", "flip_roll"):
        trial, trial_valid, _normals = _apply_convention(distance, valid, None, name)
        small = cv2.resize(np.asarray(trial, dtype=np.float32), size, interpolation=cv2.INTER_AREA)
        small_valid = cv2.resize(trial_valid.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) > 0
        finite = small_valid & np.isfinite(small) & (small > 1e-6)
        if int(finite.sum()) < 1000:
            continue
        log_depth = np.log(np.where(finite, small, float(np.nanmax(small[finite]))))
        depth_edges = _edge_magnitude(log_depth)
        inner = cv2.erode(finite.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        score = float(spearmanr(depth_edges[inner], image_edges[inner]).statistic)
        if math.isfinite(score):
            scores[name] = score
    if "identity" not in scores:
        return "identity", 0.0
    best = max(scores, key=scores.get)
    gain = scores[best] - scores["identity"]
    print(
        "STATUS // depth edge alignment " + " ".join(f"{k}={v:.3f}" for k, v in scores.items()),
        flush=True,
    )
    if best == "identity" or gain < CONVENTION_EDGE_MARGIN:
        return "identity", 0.0
    return best, gain


def _apply_convention(
    distance: np.ndarray,
    valid: np.ndarray,
    normals: np.ndarray | None,
    name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    if name == "identity":
        return distance, valid, normals
    if name == "flip":
        return _flip(distance), _flip(valid), None if normals is None else _flip(normals)
    if name == "roll":
        return _roll(distance), _roll(valid), None if normals is None else _roll(normals)
    if name == "flip_roll":
        return _roll(_flip(distance)), _roll(_flip(valid)), None if normals is None else _roll(_flip(normals))
    raise ValueError(name)


def _flip(array: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(array[:, ::-1])


def _roll(array: np.ndarray) -> np.ndarray:
    return np.roll(array, array.shape[1] // 2, axis=1)


def _normalise_scale(distance: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, float]:
    from engine.world.defaults import GROUND_MIN_BELOW_DEG
    from engine.world.pano_mesh import erp_dirs, fit_ground

    finite = valid & np.isfinite(distance) & (distance > 1e-6)
    if not bool(finite.any()):
        raise RuntimeError("scene depth has no valid pixels")
    height, width = distance.shape
    dirs = erp_dirs(height, width)[:, :width]
    below = finite & (dirs[..., 1] > math.sin(math.radians(GROUND_MIN_BELOW_DEG)))
    scale = float(np.median(distance[finite]))
    factor = SCALE_MEDIAN_FALLBACK / max(scale, 1e-6)
    if int(below.sum()) >= 3:
        plane = fit_ground((dirs * distance[..., None])[below], scale)
        if plane is not None:
            _normal, plane_d, _fraction = plane
            height_now = -float(plane_d)
            if height_now > 1e-4:
                factor = SCALE_CAMERA_HEIGHT / height_now
    out = np.asarray(distance, dtype=np.float32) * np.float32(factor)
    return out, float(factor)


def _check_axes(distance: np.ndarray, valid: np.ndarray) -> None:
    from engine.world.defaults import GROUND_MIN_BELOW_DEG
    from engine.world.pano_mesh import erp_dirs, fit_ground

    height, width = distance.shape
    dirs = erp_dirs(height, width)[:, :width]
    finite = valid & np.isfinite(distance) & (distance > 1e-6)
    below = finite & (dirs[..., 1] > math.sin(math.radians(GROUND_MIN_BELOW_DEG)))
    if int(below.sum()) < 3:
        print("STATUS // depth has no ground sample; axis check skipped", flush=True)
        return
    scale = float(np.median(distance[finite]))
    plane = fit_ground((dirs * distance[..., None])[below], scale)
    if plane is None:
        print("STATUS // depth ground plane not found; axis check skipped", flush=True)
        return
    normal, _plane_d, _fraction = plane
    if float(normal[1]) <= 0.9:
        raise RuntimeError("depth backend axes do not match OpenCV ERP")
