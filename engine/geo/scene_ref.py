"""MoGe scene-ref cloud + FLOOR/SIDE orthos.

Adapted from SplatKit ``nodes/camera_plot.py`` at pin f59de25
(``_equirect_to_cloud``, ``_ortho_view``, ``_equirect_to_ortho_views``).
No ComfyUI / folder_paths imports.
"""

from __future__ import annotations

import os

import cv2
import numpy as np


def voxel_downsample(pts: np.ndarray, cols: np.ndarray, budget: int) -> tuple[np.ndarray, np.ndarray]:
    n = pts.shape[0]
    if n <= budget:
        return pts, cols
    lo = np.percentile(pts, 1, axis=0)
    hi = np.percentile(pts, 99, axis=0)
    extent = float(np.max(hi - lo))
    if not np.isfinite(extent) or extent <= 0:
        idx = np.linspace(0, n - 1, budget).astype(np.int64)
        return pts[idx], cols[idx]
    origin = pts.min(axis=0)
    lo_v, hi_v = extent / 4096.0, extent
    tol = max(1, budget // 40)
    best = None
    for _ in range(20):
        v = (lo_v * hi_v) ** 0.5
        keys = np.floor((pts - origin) / v).astype(np.int64)
        _, first = np.unique(keys, axis=0, return_index=True)
        cnt = first.size
        if best is None or abs(cnt - budget) < abs(best[0] - budget):
            best = (cnt, first)
        if abs(cnt - budget) <= tol:
            break
        if cnt > budget:
            lo_v = v
        else:
            hi_v = v
    keep = np.sort(best[1])
    if keep.size > budget:
        keep = keep[np.linspace(0, keep.size - 1, budget).astype(np.int64)]
    return pts[keep], cols[keep]


def equirect_to_cloud(
    depth: np.ndarray,
    mask: np.ndarray,
    pano_rgb: np.ndarray,
    budget: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Equirect depth -> sparse cloud. +Z forward, +X right, +Y up, origin=camera."""
    H, W = depth.shape[:2]
    cols = np.arange(W)
    rows = np.arange(H)
    lon = (cols / W) * 2.0 * np.pi - np.pi
    lat = (np.pi / 2.0) - (rows / max(H - 1, 1)) * np.pi
    lon_g, lat_g = np.meshgrid(lon, lat)
    cl = np.cos(lat_g)
    dirx = cl * np.sin(lon_g)
    diry = np.sin(lat_g)
    dirz = cl * np.cos(lon_g)
    valid = mask.astype(bool) & np.isfinite(depth) & (depth > 0)
    idx = np.flatnonzero(valid.reshape(-1))
    if idx.size == 0:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint8)
    budget = max(int(budget), 1)
    pool = max(budget * 6, 60000)
    if idx.size > pool:
        idx = idx[np.linspace(0, idx.size - 1, pool).astype(np.int64)]
    d = depth.reshape(-1)[idx].astype(np.float32)
    pts = np.stack(
        [dirx.reshape(-1)[idx], diry.reshape(-1)[idx], dirz.reshape(-1)[idx]],
        axis=1,
    ) * d[:, None]
    cols_rgb = pano_rgb.reshape(-1, 3)[idx].astype(np.uint8)
    pts, cols_rgb = voxel_downsample(pts.astype(np.float32), cols_rgb, budget)
    return pts.astype(np.float32), cols_rgb


def ortho_view(
    pts: np.ndarray,
    cols: np.ndarray,
    ai: int,
    bi: int,
    proj_axis: int,
    sign: float,
    res_max: int | None = None,
    pct: tuple[float, float] = (2.0, 98.0),
    fill: int | None = None,
) -> dict | None:
    """Rasterise a colored point set to an orthographic RGBA image.

    Same painter's z-buffer as SplatKit. Returns png ndarray + lo/hi extents.
    """
    if res_max is None:
        res_max = int(os.environ.get("P2S_OVERLAY_RES", "768"))
    if fill is None:
        fill = int(os.environ.get("P2S_OVERLAY_FILL", "2"))
    if pts.shape[0] == 0:
        return None
    a = pts[:, ai].astype(np.float32)
    b = pts[:, bi].astype(np.float32)
    c = pts[:, proj_axis].astype(np.float32) * float(sign)
    lo_a, hi_a = np.percentile(a, pct)
    lo_b, hi_b = np.percentile(b, pct)
    ext_a, ext_b = float(hi_a - lo_a), float(hi_b - lo_b)
    if not (ext_a > 1e-6 and ext_b > 1e-6):
        return None
    if ext_a >= ext_b:
        W = int(res_max)
        H = max(1, int(round(res_max * ext_b / ext_a)))
    else:
        H = int(res_max)
        W = max(1, int(round(res_max * ext_a / ext_b)))
    px = np.round((a - lo_a) / ext_a * (W - 1)).astype(np.int64)
    py = np.round((hi_b - b) / ext_b * (H - 1)).astype(np.int64)
    inb = (px >= 0) & (px < W) & (py >= 0) & (py < H)
    px, py, c = px[inb], py[inb], c[inb]
    cc = cols[inb]
    if px.size == 0:
        return None
    order = np.argsort(c, kind="stable")
    flat = (py * W + px)[order]
    img = np.zeros((H * W, 3), np.uint8)
    alpha = np.zeros(H * W, np.uint8)
    img[flat] = cc[order]
    alpha[flat] = 255
    img = img.reshape(H, W, 3)
    alpha = alpha.reshape(H, W)
    if fill > 0:
        r = int(fill)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        closed = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, k)
        holes = ((closed > 0) & (alpha == 0)).astype(np.uint8)
        if holes.any():
            img = cv2.inpaint(img, holes * 255, 3, cv2.INPAINT_TELEA)
        alpha = closed
    rgba = np.dstack([img, alpha])
    return {
        "rgba": rgba,
        "lo": [float(lo_a), float(lo_b)],
        "hi": [float(hi_a), float(hi_b)],
        "w": int(W),
        "h": int(H),
    }


def equirect_to_ortho_views(
    depth: np.ndarray,
    mask: np.ndarray,
    pano_rgb: np.ndarray,
    cap: int = 2_500_000,
    res_max: int | None = None,
) -> dict | None:
    """Dense FLOOR (top-down X/Z) + SIDE (elevation Z/Y) from MoGe depth."""
    H, W = depth.shape[:2]
    cols_i = np.arange(W)
    rows_i = np.arange(H)
    lon = (cols_i / W) * 2.0 * np.pi - np.pi
    lat = (np.pi / 2.0) - (rows_i / max(H - 1, 1)) * np.pi
    lon_g, lat_g = np.meshgrid(lon, lat)
    cl = np.cos(lat_g)
    dirx = cl * np.sin(lon_g)
    diry = np.sin(lat_g)
    dirz = cl * np.cos(lon_g)
    valid = mask.astype(bool) & np.isfinite(depth) & (depth > 0)
    idx = np.flatnonzero(valid.reshape(-1))
    if idx.size == 0:
        return None
    if idx.size > cap:
        idx = idx[np.linspace(0, idx.size - 1, cap).astype(np.int64)]
    d = depth.reshape(-1)[idx].astype(np.float32)
    pts = np.stack(
        [dirx.reshape(-1)[idx], diry.reshape(-1)[idx], dirz.reshape(-1)[idx]],
        axis=1,
    ) * d[:, None]
    cols_rgb = pano_rgb.reshape(-1, 3)[idx].astype(np.uint8)
    out: dict = {}
    top = ortho_view(pts, cols_rgb, 0, 2, 1, -1, res_max)
    side = ortho_view(pts, cols_rgb, 2, 1, 0, +1, res_max)
    if top:
        out["top"] = top
    if side:
        out["side"] = side
    return out or None


def write_png_rgba(path, rgba: np.ndarray) -> None:
    bgra = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)
    cv2.imwrite(str(path), bgra)
