"""ImageCASharpening+ amount from ComfyUI_essentials ImageCAS (graph 0.5)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from engine.pano.defaults import CAS_AMOUNT


def _min(tensor_list: tuple[torch.Tensor, ...]) -> torch.Tensor:
    stacked = torch.stack(list(tensor_list))
    return torch.clamp(stacked.min(axis=0)[0], min=0)


def _max(tensor_list: tuple[torch.Tensor, ...]) -> torch.Tensor:
    stacked = torch.stack(list(tensor_list))
    return torch.clamp(stacked.max(axis=0)[0], max=1)


def contrast_adaptive_sharpen(image: torch.Tensor, amount: float = CAS_AMOUNT) -> torch.Tensor:
    """IMAGE [B,H,W,C] in [0,1]. Copied from essentials ImageCAS.execute."""
    epsilon = 1e-5
    img = F.pad(image.detach().permute([0, 3, 1, 2]), pad=(1, 1, 1, 1))
    a = img[..., :-2, :-2]
    b = img[..., :-2, 1:-1]
    c = img[..., :-2, 2:]
    d = img[..., 1:-1, :-2]
    e = img[..., 1:-1, 1:-1]
    f = img[..., 1:-1, 2:]
    g = img[..., 2:, :-2]
    h = img[..., 2:, 1:-1]
    i = img[..., 2:, 2:]
    cross = (b, d, e, f, h)
    mn = _min(cross)
    mx = _max(cross)
    diag = (a, c, g, i)
    mn2 = _min(diag)
    mx2 = _max(diag)
    mx = mx + mx2
    mn = mn + mn2
    inv_mx = torch.reciprocal(mx + epsilon)
    amp = inv_mx * torch.minimum(mn, (2 - mx))
    amp = torch.sqrt(amp)
    w = -amp * (amount * (1 / 5 - 1 / 8) + 1 / 8)
    div = torch.reciprocal(1 + 4 * w)
    output = ((b + d + f + h) * w + e) * div
    output = output.clamp(0, 1)
    return output.permute([0, 2, 3, 1]).detach()
