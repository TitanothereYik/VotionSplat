"""CPU warp preview for the Qt process. Same pinhole math as the ported perspective-to-ERP warp
(pin 4d5ff7c; see THIRD_PARTY_NOTICES.md). No torch — the Qt process must not import torch for inference.
Fill for the viewport is green #00ff00 (dummy / PRD canvas), not the graph control fill.
"""

from __future__ import annotations

import math

import numpy as np
from PIL import Image

from engine.pano.defaults import (
    GENERATE_HEIGHT,
    GENERATE_WIDTH,
    GREEN_ERP,
    H_FOV_DEG_DEFAULT,
    WARP_PITCH_DEG,
    WARP_SUPERSAMPLE,
    WARP_YAW_DEG,
)


def _parse_color(s: str, default: tuple[float, float, float] = (0.0, 1.0, 0.0)) -> np.ndarray:
    t = str(s).strip().lstrip("#")
    if len(t) == 6:
        try:
            return np.array([int(t[i : i + 2], 16) / 255.0 for i in (0, 2, 4)], dtype=np.float32)
        except ValueError:
            pass
    return np.array(default, dtype=np.float32)


def warp_photo_to_green_erp(
    photo: Image.Image,
    h_fov_deg: float = H_FOV_DEG_DEFAULT,
    width: int = GENERATE_WIDTH,
    height: int = GENERATE_HEIGHT,
    yaw_deg: float = WARP_YAW_DEG,
    pitch_deg: float = WARP_PITCH_DEG,
    supersample: float = WARP_SUPERSAMPLE,
    fill_hex: str = GREEN_ERP,
) -> Image.Image:
    """Place a perspective photo on a 2:1 ERP canvas. Returns RGB uint8 PIL image."""
    src = np.asarray(photo.convert("RGB"), dtype=np.float32) / 255.0
    sh, sw = src.shape[0], src.shape[1]
    W, H = int(width), int(height)

    f = (sw / 2.0) / math.tan(math.radians(float(h_fov_deg)) / 2.0)
    cx, cy = (sw - 1) / 2.0, (sh - 1) / 2.0

    xs = (np.arange(W, dtype=np.float32) + 0.5) / W * (2 * math.pi) - math.pi
    ys = (math.pi / 2.0) - (np.arange(H, dtype=np.float32) + 0.5) / H * math.pi
    lon_g, lat_g = np.meshgrid(xs, ys)
    cl = np.cos(lat_g)
    dx = cl * np.sin(lon_g)
    dy = np.sin(lat_g)
    dz = cl * np.cos(lon_g)

    ay, ap = math.radians(float(yaw_deg)), math.radians(float(pitch_deg))
    if ay:
        ca, sa = math.cos(ay), math.sin(ay)
        dx, dz = ca * dx - sa * dz, sa * dx + ca * dz
    if ap:
        cp, sp = math.cos(ap), math.sin(ap)
        dy, dz = cp * dy - sp * dz, sp * dy + cp * dz

    front = dz > 1e-6
    dzc = np.where(front, dz, np.ones_like(dz))
    xf = f * (dx / dzc) + cx
    yf = cy - f * (dy / dzc)
    valid = front & (xf >= 0) & (xf <= sw - 1) & (yf >= 0) & (yf <= sh - 1)

    foot_w = max(1.0, width * (float(h_fov_deg) / 360.0))
    tgt_w = int(max(1, round(foot_w * max(1.0, float(supersample)))))
    work = src
    if tgt_w < sw:
        tgt_h = int(max(1, round(tgt_w * sh / sw)))
        work = np.asarray(
            Image.fromarray((src * 255).astype(np.uint8)).resize((tgt_w, tgt_h), Image.Resampling.BOX),
            dtype=np.float32,
        ) / 255.0
        yf = yf * (tgt_h / max(sh, 1))
        xf = xf * (tgt_w / max(sw, 1))
        valid = front & (xf >= 0) & (xf <= work.shape[1] - 1) & (yf >= 0) & (yf <= work.shape[0] - 1)

    # Bilinear sample (align_corners-style via (size-1) already in xf/yf).
    h2, w2 = work.shape[0], work.shape[1]
    x0 = np.floor(xf).astype(np.int32)
    y0 = np.floor(yf).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, w2 - 1)
    y1 = np.clip(y0 + 1, 0, h2 - 1)
    x0 = np.clip(x0, 0, w2 - 1)
    y0 = np.clip(y0, 0, h2 - 1)
    wx = (xf - np.floor(xf)).astype(np.float32)[..., None]
    wy = (yf - np.floor(yf)).astype(np.float32)[..., None]
    ia = work[y0, x0]
    ib = work[y0, x1]
    ic = work[y1, x0]
    id_ = work[y1, x1]
    sampled = ia * (1 - wx) * (1 - wy) + ib * wx * (1 - wy) + ic * (1 - wx) * wy + id_ * wx * wy

    fill = _parse_color(fill_hex)
    a = valid.astype(np.float32)[..., None]
    canvas = sampled * a + fill.reshape(1, 1, 3) * (1.0 - a)
    out = np.clip(canvas * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(out, mode="RGB")


LOOK_HFOV_DEG = 90.0
LOOK_PITCH_MAX_DEG = 85.0
LOOK_ERP_MAX_WIDTH = 2048


def erp_to_rectilinear(
    erp: Image.Image | np.ndarray,
    yaw_deg: float = 0.0,
    pitch_deg: float = 0.0,
    h_fov_deg: float = LOOK_HFOV_DEG,
    width: int = 640,
    height: int = 360,
) -> Image.Image:
    """Pinhole look from inside an equirectangular sphere (HDRI-style). No torch.

    Camera at the origin, +Z forward / +Y up, same lon/lat as the ERP warp.
    yaw: look right is positive. pitch: look up is positive. Horizontal wrap.
    """
    if isinstance(erp, Image.Image):
        src = np.asarray(erp.convert("RGB"), dtype=np.float32) / 255.0
    else:
        src = np.asarray(erp, dtype=np.float32)
        if src.max() > 1.5:
            src = src / 255.0
        if src.ndim == 2:
            src = np.repeat(src[..., None], 3, axis=2)
        elif src.shape[-1] > 3:
            src = src[..., :3]
    sh, sw = int(src.shape[0]), int(src.shape[1])
    W, H = max(1, int(width)), max(1, int(height))
    hfov = math.radians(float(h_fov_deg))
    f = (W / 2.0) / math.tan(hfov / 2.0)

    us = (np.arange(W, dtype=np.float32) + 0.5) - (W / 2.0)
    vs = (H / 2.0) - (np.arange(H, dtype=np.float32) + 0.5)
    xx, yy = np.meshgrid(us, vs)
    dx = xx / f
    dy = yy / f
    dz = np.ones_like(dx)

    ap = math.radians(float(pitch_deg))
    ay = math.radians(float(yaw_deg))
    if ap:
        cp, sp = math.cos(ap), math.sin(ap)
        dy, dz = cp * dy + sp * dz, -sp * dy + cp * dz
    if ay:
        ca, sa = math.cos(ay), math.sin(ay)
        dx, dz = ca * dx + sa * dz, -sa * dx + ca * dz

    hyp = np.sqrt(dx * dx + dy * dy + dz * dz).clip(min=1e-8)
    dx, dy, dz = dx / hyp, dy / hyp, dz / hyp
    lon = np.arctan2(dx, dz)
    lat = np.arcsin(np.clip(dy, -1.0, 1.0))
    xf = (lon + math.pi) / (2.0 * math.pi) * sw
    yf = (math.pi / 2.0 - lat) / math.pi * sh

    x0 = np.floor(xf).astype(np.int32)
    y0 = np.floor(yf).astype(np.int32)
    wx = (xf - np.floor(xf)).astype(np.float32)[..., None]
    wy = (yf - np.floor(yf)).astype(np.float32)[..., None]
    x1 = np.mod(x0 + 1, sw)
    x0 = np.mod(x0, sw)
    y0 = np.clip(y0, 0, sh - 1)
    y1 = np.clip(y0 + 1, 0, sh - 1)
    ia = src[y0, x0]
    ib = src[y0, x1]
    ic = src[y1, x0]
    id_ = src[y1, x1]
    sampled = ia * (1 - wx) * (1 - wy) + ib * wx * (1 - wy) + ic * (1 - wx) * wy + id_ * wx * wy
    out = np.clip(sampled * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(out, mode="RGB")
