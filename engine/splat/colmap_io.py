"""Load SphereSfM COLMAP sparse/0 for the gsplat trainer.

Uses SplatKit ``tools/colmap_read_model.py`` at pin f59de25 (OpenCV w2c, no
WorldFM conversion). Cube faces are SIMPLE_PINHOLE.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from engine.job import REPO_ROOT

SPLATKIT = REPO_ROOT / "vendor" / "splatkit"
if str(SPLATKIT) not in sys.path:
    sys.path.insert(0, str(SPLATKIT))

from tools.colmap_read_model import (  # noqa: E402
    qvec2rotmat,
    read_cameras_binary,
    read_images_binary,
    read_points3D_binary,
)


@dataclass
class SplatView:
    name: str
    image_path: Path
    w2c: np.ndarray  # 4x4 OpenCV
    K: np.ndarray  # 3x3
    width: int
    height: int


def _k_from_camera(cam) -> np.ndarray:
    p = np.asarray(cam.params, dtype=np.float64)
    if cam.model == "SIMPLE_PINHOLE":
        f, cx, cy = p[0], p[1], p[2]
        fx = fy = f
    elif cam.model == "PINHOLE":
        fx, fy, cx, cy = p[0], p[1], p[2], p[3]
    elif cam.model in ("SIMPLE_RADIAL",):
        f, cx, cy = p[0], p[1], p[2]
        fx = fy = f
    else:
        raise RuntimeError(f"unsupported COLMAP camera model {cam.model}")
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def _w2c(qvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = qvec2rotmat(qvec)
    mat[:3, 3] = tvec
    return mat


def load_colmap_views(colmap_root: Path) -> list[SplatView]:
    sparse = colmap_root / "sparse" / "0"
    image_dir = colmap_root / "images"
    cams = read_cameras_binary(str(sparse / "cameras.bin"))
    images = read_images_binary(str(sparse / "images.bin"))
    views: list[SplatView] = []
    for img in images.values():
        cam = cams[img.camera_id]
        path = image_dir / img.name
        if not path.is_file():
            path = image_dir / Path(img.name).name
        if not path.is_file():
            continue
        views.append(
            SplatView(
                name=img.name,
                image_path=path,
                w2c=_w2c(img.qvec, img.tvec),
                K=_k_from_camera(cam),
                width=int(cam.width),
                height=int(cam.height),
            )
        )
    views.sort(key=lambda v: v.name)
    if len(views) < 2:
        raise RuntimeError(f"need at least 2 COLMAP images under {image_dir}")
    return views


def resolve_sparse0(path: Path) -> Path:
    """Accept colmap root, sparse/, or sparse/0."""
    path = Path(path)
    for cand in (path, path / "0", path / "sparse" / "0"):
        if (cand / "points3D.bin").is_file():
            return cand
    raise FileNotFoundError(f"no points3D.bin under {path}")


def load_points3d(colmap_root: Path) -> tuple[np.ndarray, np.ndarray]:
    sparse = resolve_sparse0(colmap_root)
    pts = read_points3D_binary(str(sparse / "points3D.bin"))
    if not pts:
        raise RuntimeError(f"no points in {sparse / 'points3D.bin'}")
    xyz = np.stack([p.xyz for p in pts.values()], axis=0).astype(np.float32)
    rgb = np.stack([p.rgb for p in pts.values()], axis=0).astype(np.float32) / 255.0
    return xyz, rgb


def load_sparse_display(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sparse cloud + camera centers for the Reconstruct viewport.

    COLMAP OpenCV +Y down is flipped to editor +Y up (same as Geometry).
    """
    sparse = resolve_sparse0(path)
    xyz, rgb = load_points3d(sparse)
    xyz = xyz.copy()
    xyz[:, 1] *= -1.0
    cam_xyz = np.zeros((0, 3), dtype=np.float32)
    images_bin = sparse / "images.bin"
    if images_bin.is_file():
        images = read_images_binary(str(images_bin))
        centers = []
        for img in images.values():
            rot = qvec2rotmat(img.qvec)
            center = -rot.T @ np.asarray(img.tvec, dtype=np.float64)
            center[1] *= -1.0
            centers.append(center)
        if centers:
            cam_xyz = np.stack(centers, axis=0).astype(np.float32)
    return xyz, rgb, cam_xyz


def scene_scale_from_views(views: list[SplatView]) -> float:
    centers = []
    for v in views:
        r = v.w2c[:3, :3]
        t = v.w2c[:3, 3]
        centers.append(-r.T @ t)
    arr = np.stack(centers, axis=0)
    extent = float(np.linalg.norm(arr.max(axis=0) - arr.min(axis=0)))
    return max(extent, 1e-3)


def scene_scale_from_points(xyz: np.ndarray) -> float:
    """Robust point-cloud diameter (p90 radius × 2), ignoring far SfM outliers."""
    pts = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    if pts.shape[0] < 2:
        return 1e-3
    med = np.median(pts, axis=0)
    d = np.linalg.norm(pts - med, axis=1)
    return max(float(np.quantile(d, 0.90)) * 2.0, 1e-3)


def camera_centroid_from_views(views: list[SplatView]) -> np.ndarray:
    """Capture origin: SphereSfM cameras sit inside the splat shell."""
    centers = []
    for v in views:
        r = v.w2c[:3, :3]
        t = v.w2c[:3, 3]
        centers.append(-r.T @ t)
    return np.mean(np.stack(centers), axis=0).astype(np.float64)


def read_rgb(path: Path, width: int, height: int) -> np.ndarray:
    with Image.open(path) as im:
        rgb = im.convert("RGB")
        if rgb.size != (width, height):
            rgb = rgb.resize((width, height), Image.Resampling.BILINEAR)
        return np.asarray(rgb, dtype=np.float32) / 255.0
