"""HiRes images are the WorldStereo keyframes, upscaled.

``hires/source.txt`` says ``worldstereo`` when those files are the upscale.
Older folders were a pinhole reprojection of the panorama and are not complete.
The reprojection helpers stay for the z-buffer check. HiRes does not call them.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from engine.worldstereo import defaults as D

SCALE = 2
HIRES_H = D.SIZE_H * SCALE
HIRES_W = D.SIZE_W * SCALE
MOGE_LEVEL = 6
DEPTH_LONG_SIDE = 4096
ULTRASHARP_FILE = "4x-UltraSharp.pth"
HIRES_SOURCE = "worldstereo"


def scaled_k(K: np.ndarray, width: int = HIRES_W, height: int = HIRES_H) -> np.ndarray:
    out = np.array(K, dtype=np.float64, copy=True)
    out[0, 0] *= width / float(D.SIZE_W)
    out[0, 2] *= width / float(D.SIZE_W)
    out[1, 1] *= height / float(D.SIZE_H)
    out[1, 2] *= height / float(D.SIZE_H)
    return out


def depth_cloud(depth: np.ndarray, mask: np.ndarray, pano: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """OpenCV points and 8K colours. Invalid MoGe pixels are dropped."""
    depth = np.asarray(depth, dtype=np.float32)
    valid = np.asarray(mask, dtype=bool) & np.isfinite(depth) & (depth > 0)
    height, width = depth.shape
    step = 1
    while (height // step) * (width // step) > 2_500_000:
        step *= 2
    if step > 1:
        depth = depth[::step, ::step]
        valid = valid[::step, ::step]
        height, width = depth.shape
    ys, xs = np.nonzero(valid)
    if xs.size == 0:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint8)
    # Pixel centres on the depth grid, mapped back onto the full ERP.
    full_h, full_w = int(pano.shape[0]), int(pano.shape[1])
    uu = (xs.astype(np.float64) + 0.5) / width
    vv = (ys.astype(np.float64) + 0.5) / height
    lon = uu * (2.0 * np.pi) - np.pi
    lat = (np.pi / 2.0) - vv * np.pi
    cos_lat = np.cos(lat)
    direc = np.stack([cos_lat * np.sin(lon), -np.sin(lat), cos_lat * np.cos(lon)], axis=1)
    points = (direc * depth[ys, xs, None]).astype(np.float32)
    pu = np.clip(np.rint(uu * full_w - 0.5), 0, full_w - 1).astype(np.int64)
    pv = np.clip(np.rint(vv * full_h - 0.5), 0, full_h - 1).astype(np.int64)
    colors = np.asarray(pano)[pv, pu].astype(np.uint8)
    return points, colors


def reproject(
    points: np.ndarray,
    colors: np.ndarray,
    w2c: np.ndarray,
    K: np.ndarray,
    height: int = HIRES_H,
    width: int = HIRES_W,
) -> tuple[np.ndarray, np.ndarray]:
    """Nearest-point splat. Returns RGB and a bool gate (True = 8K pixel)."""
    rgb = np.zeros((height, width, 3), np.uint8)
    gate = np.zeros((height, width), dtype=bool)
    if points.shape[0] == 0:
        return rgb, gate
    pose = np.asarray(w2c, dtype=np.float64)
    cam = np.asarray(points, dtype=np.float64) @ pose[:3, :3].T + pose[:3, 3]
    z = cam[:, 2]
    keep = np.isfinite(z) & (z > 1e-3)
    if not np.any(keep):
        return rgb, gate
    cam = cam[keep]
    z = z[keep]
    cols = np.asarray(colors)[keep]
    u = K[0, 0] * cam[:, 0] / z + K[0, 2]
    v = K[1, 1] * cam[:, 1] / z + K[1, 2]
    ui = np.rint(u).astype(np.int64)
    vi = np.rint(v).astype(np.int64)
    inside = (ui >= 0) & (ui < width) & (vi >= 0) & (vi < height) & np.isfinite(u) & np.isfinite(v)
    if not np.any(inside):
        return rgb, gate
    ui, vi, z, cols = ui[inside], vi[inside], z[inside], cols[inside]
    order = np.argsort(-z, kind="stable")
    ui, vi, cols = ui[order], vi[order], cols[order]
    # Far points are written first. A nearer point overwrites the same pixel.
    flat = vi * width + ui
    rgb.reshape(-1, 3)[flat] = cols
    gate.reshape(-1)[flat] = True
    return rgb, gate


def composite(pano_rgb: np.ndarray, gate: np.ndarray, fill_rgb: np.ndarray) -> np.ndarray:
    out = np.array(fill_rgb, copy=True)
    out[gate] = pano_rgb[gate]
    return out


def hires_complete(folder: Path, nframe: int) -> bool:
    hires = folder / "hires"
    stamp = hires / "source.txt"
    if not hires.is_dir() or not stamp.is_file():
        return False
    if stamp.read_text(encoding="utf-8").strip() != HIRES_SOURCE:
        return False
    shots = list(hires.glob("[0-9][0-9][0-9].png"))
    return len(shots) >= int(nframe)


def prepare_depth_image(pano: np.ndarray, long_side: int = DEPTH_LONG_SIDE) -> np.ndarray:
    height, width = pano.shape[:2]
    long_side = int(long_side)
    if max(height, width) <= long_side:
        return pano
    scale = long_side / float(max(height, width))
    return cv2.resize(pano, (max(2, int(round(width * scale))), max(2, int(round(height * scale)))), interpolation=cv2.INTER_AREA)
