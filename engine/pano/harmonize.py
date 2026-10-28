"""Mickmumpitz HarmonizeBoundary.run at 4d5ff7c. Worker only."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from engine.pano.defaults import (
    HARMONIZE_ITERATIONS,
    HARMONIZE_SOLVE_SCALE,
    HARMONIZE_STRENGTH,
    HARMONIZE_WRAP_HORIZONTAL,
)


def harmonize_boundary(
    image: torch.Tensor,
    plate: torch.Tensor,
    inpaint_mask: torch.Tensor,
    strength: float = HARMONIZE_STRENGTH,
    solve_scale: int = HARMONIZE_SOLVE_SCALE,
    iterations: int = HARMONIZE_ITERATIONS,
    wrap_horizontal: bool = HARMONIZE_WRAP_HORIZONTAL,
) -> torch.Tensor:
    """IMAGE [B,H,W,C], plate same, MASK [1,H,W] white=hole. Returns IMAGE [1,H,W,C]."""
    img = image[0].float()
    plt = plate[0].float()
    if plt.shape[:2] != img.shape[:2]:
        plt = F.interpolate(
            plt.permute(2, 0, 1)[None],
            size=img.shape[:2],
            mode="bilinear",
            align_corners=False,
        )[0].permute(1, 2, 0)
    m = inpaint_mask[0].float() if inpaint_mask.ndim == 3 else inpaint_mask.float()
    if m.shape != img.shape[:2]:
        m = F.interpolate(m[None, None], size=img.shape[:2], mode="bilinear", align_corners=False)[
            0, 0
        ]

    known = (m < 0.5).float()
    if known.sum() < 16:
        print("STATUS // HarmonizeBoundary: no known region, passing through", flush=True)
        return image

    diff = (plt - img) * known.unsqueeze(-1)
    s = max(1, int(solve_scale))
    h, w = max(4, img.shape[0] // s), max(8, img.shape[1] // s)
    kd = F.interpolate(known[None, None], size=(h, w), mode="area")[0, 0]
    dd = F.interpolate(diff.permute(2, 0, 1)[None], size=(h, w), mode="area")[0]
    fixed = kd > 0.5
    vals = torch.where(fixed.unsqueeze(0), dd / kd.clamp(min=1e-6), torch.zeros_like(dd))

    c = torch.where(fixed.unsqueeze(0), vals, torch.zeros_like(vals))
    fx = fixed.unsqueeze(0)
    for _ in range(int(iterations)):
        if wrap_horizontal:
            left = torch.roll(c, 1, dims=2)
            right = torch.roll(c, -1, dims=2)
        else:
            left = torch.cat([c[:, :, :1], c[:, :, :-1]], dim=2)
            right = torch.cat([c[:, :, 1:], c[:, :, -1:]], dim=2)
        up = torch.cat([c[:, :1, :], c[:, :-1, :]], dim=1)
        down = torch.cat([c[:, 1:, :], c[:, -1:, :]], dim=1)
        c = 0.25 * (left + right + up + down)
        c = torch.where(fx, vals, c)

    cf = F.interpolate(c[None], size=img.shape[:2], mode="bilinear", align_corners=False)[0]
    cf = cf.permute(1, 2, 0)
    out = (img + cf * float(strength)).clamp(0, 1)
    return out.unsqueeze(0)
