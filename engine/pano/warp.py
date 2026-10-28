"""Mickmumpitz PerspToErpWarp.build at 4d5ff7c (worker only). No erp_guide / cv2."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from engine.pano.defaults import (
    GENERATE_HEIGHT,
    GENERATE_WIDTH,
    H_FOV_DEG_DEFAULT,
    WARP_FILL_COLOR,
    WARP_MASK_FEATHER,
    WARP_PITCH_DEG,
    WARP_SUPERSAMPLE,
    WARP_YAW_DEG,
)


def _parse_color(s: str, default: tuple[float, float, float] = (0.0, 0.0, 0.0)):
    t = str(s).strip()
    if "," in t:
        try:
            parts = [float(p) for p in t.split(",")]
        except ValueError:
            parts = []
        if len(parts) == 3:
            return tuple(min(max(p, 0.0), 255.0) / 255.0 for p in parts)
    h = t.lstrip("#")
    if len(h) == 6:
        try:
            return tuple(int(h[i : i + 2], 16) / 255.0 for i in (0, 2, 4))
        except ValueError:
            pass
    return default


def persp_to_erp(
    image: torch.Tensor,
    width: int = GENERATE_WIDTH,
    height: int = GENERATE_HEIGHT,
    h_fov_deg: float = H_FOV_DEG_DEFAULT,
    yaw_deg: float = WARP_YAW_DEG,
    pitch_deg: float = WARP_PITCH_DEG,
    mask_feather: int = WARP_MASK_FEATHER,
    supersample: float = WARP_SUPERSAMPLE,
    fill_color: str = WARP_FILL_COLOR,
) -> tuple[torch.Tensor, torch.Tensor]:
    """IMAGE [B,H,W,C] -> control_video [1,H,W,C], inpaint_mask [1,H,W] (white = hole)."""
    W, H = int(width), int(height)
    src = image[0].permute(2, 0, 1).unsqueeze(0).float().clamp(0, 1)
    sh, sw = int(src.shape[2]), int(src.shape[3])

    f = (sw / 2.0) / math.tan(math.radians(float(h_fov_deg)) / 2.0)
    cx, cy = (sw - 1) / 2.0, (sh - 1) / 2.0

    xs = torch.arange(W, dtype=torch.float32)
    ys = torch.arange(H, dtype=torch.float32)
    lon = (xs + 0.5) / W * (2 * math.pi) - math.pi
    lat = (math.pi / 2.0) - (ys + 0.5) / H * math.pi
    lat_g, lon_g = torch.meshgrid(lat, lon, indexing="ij")
    cl = torch.cos(lat_g)
    dx = cl * torch.sin(lon_g)
    dy = torch.sin(lat_g)
    dz = cl * torch.cos(lon_g)

    ay, ap = math.radians(float(yaw_deg)), math.radians(float(pitch_deg))
    if ay:
        ca, sa = math.cos(ay), math.sin(ay)
        dx, dz = ca * dx - sa * dz, sa * dx + ca * dz
    if ap:
        cp, sp = math.cos(ap), math.sin(ap)
        dy, dz = cp * dy - sp * dz, sp * dy + cp * dz

    front = dz > 1e-6
    dzc = torch.where(front, dz, torch.ones_like(dz))
    xf = f * (dx / dzc) + cx
    yf = cy - f * (dy / dzc)
    valid = front & (xf >= 0) & (xf <= sw - 1) & (yf >= 0) & (yf <= sh - 1)

    gx = 2.0 * xf / max(sw - 1, 1) - 1.0
    gy = 2.0 * yf / max(sh - 1, 1) - 1.0
    grid = torch.stack([gx, gy], dim=-1).unsqueeze(0)

    foot_w = max(1.0, width * (float(h_fov_deg) / 360.0))
    tgt_w = int(max(1, round(foot_w * max(1.0, float(supersample)))))
    src_s = src
    if tgt_w < sw:
        tgt_h = int(max(1, round(tgt_w * sh / sw)))
        src_s = F.interpolate(src, size=(tgt_h, tgt_w), mode="area")
    sampled = F.grid_sample(
        src_s, grid, mode="bicubic", padding_mode="border", align_corners=True
    )[0].clamp(0, 1)

    validf = valid.float()
    fill = torch.tensor(_parse_color(fill_color), dtype=torch.float32)
    a = validf.unsqueeze(-1)
    canvas = sampled.permute(1, 2, 0) * a + fill.view(1, 1, 3) * (1.0 - a)

    hole_bin = 1.0 - validf
    hole = hole_bin
    r = int(mask_feather)
    if r > 0:
        k = 2 * r + 1
        m = hole_bin.view(1, 1, H, W)
        kh = torch.ones(1, 1, 1, k) / k
        kv = torch.ones(1, 1, k, 1) / k
        m = F.conv2d(F.pad(m, (r, r, 0, 0), mode="reflect"), kh)
        m = F.conv2d(F.pad(m, (0, 0, r, r), mode="reflect"), kv)
        hole = torch.maximum(m.view(H, W).clamp(0, 1), hole_bin)
    inpaint_mask = hole.unsqueeze(0)
    return canvas.unsqueeze(0), inpaint_mask
