"""MickmumpitzPanoRollHorizontal + PanoSeamMask (4d5ff7c). Torch tensors, worker only."""

from __future__ import annotations

import torch

from engine.pano.defaults import ROLL_FRACTION, SEAM_STRIP_FEATHER, SEAM_STRIP_WIDTH


def roll_horizontal(image: torch.Tensor, fraction: float = ROLL_FRACTION) -> torch.Tensor:
    """IMAGE [B,H,W,C] wrap-around roll. Same as PanoRollHorizontal.run."""
    w = int(image.shape[2])
    shift = int(round(w * float(fraction))) % w if w else 0
    if shift == 0:
        return image
    return torch.roll(image, shifts=shift, dims=2)


def seam_mask(
    width: int,
    height: int,
    seam_width: int = SEAM_STRIP_WIDTH,
    feather: int = SEAM_STRIP_FEATHER,
) -> torch.Tensor:
    """MASK [1,H,W] centred strip. Same as PanoSeamMask.run."""
    w, h = int(width), int(height)
    s = max(min(int(seam_width), w), 1)
    f = max(int(feather), 0)
    x0 = (w - s) // 2
    col = torch.zeros(w, dtype=torch.float32)
    col[x0 : x0 + s] = 1.0
    if f > 0:
        ramp = torch.linspace(0.0, 1.0, f + 2, dtype=torch.float32)[1:-1]
        n = min(f, s // 2)
        if n > 0:
            col[x0 : x0 + n] = ramp[:n]
            col[x0 + s - n : x0 + s] = ramp[:n].flip(0)
    return col.view(1, 1, w).expand(1, h, w).clone()
