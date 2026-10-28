"""SplatKit WanI2VMaskedConditioning pixel stage + Diffusers 36-ch pack.

Pixel hole-fill copied from ``vendor/splatkit/nodes/wan.py`` (native torch, no Comfy).
Channel layout copied from Diffusers ``WanImageToVideoPipeline.prepare_latents``:
4-ch packed mask + 16-ch VAE latent, concatenated with 16-ch noise = 36.
Per-frame validity replaces first-frame-only (product holes are not only frame 0).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def splatkit_masked_pixels(
    control_rgb: torch.Tensor,
    control_mask: torch.Tensor,
    width: int,
    height: int,
    length: int,
    hole_fill: str = "black",
    invert_mask: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return filled RGB [T,H,W,3] in [0,1] and validity [T,H,W] (1=known)."""
    fill_val = 0.0 if hole_fill == "black" else 0.5
    vid = control_rgb[:length]
    if vid.ndim != 4 or vid.shape[-1] != 3:
        raise ValueError(f"control_rgb expected [T,H,W,3], got {tuple(vid.shape)}")
    vid = F.interpolate(vid.movedim(-1, 1), size=(height, width), mode="bilinear", align_corners=False)
    vid = vid.movedim(1, -1)
    image = torch.ones((length, height, width, 3), device=vid.device, dtype=vid.dtype) * fill_val
    image[: vid.shape[0]] = vid[..., :3]

    m = control_mask[:length]
    if m.ndim == 4:
        m = m[..., 0]
    m = F.interpolate(m.unsqueeze(1).float(), size=(height, width), mode="bilinear", align_corners=False).squeeze(1)
    m = (m > 0.5).float()
    if invert_mask:
        m = 1.0 - m
    mask_valid = torch.zeros((length, height, width), device=m.device, dtype=torch.float32)
    mask_valid[: m.shape[0]] = m
    image[mask_valid < 0.5] = fill_val
    return image, mask_valid


def pack_mask_diffusers(mask_valid: torch.Tensor, latent_h: int, latent_w: int, temporal: int = 4) -> torch.Tensor:
    """Diffusers I2V mask pack. 1 = known, 0 = generate. Out [1, 4, T_lat, H, W]."""
    length, _, _ = mask_valid.shape
    mv = F.interpolate(mask_valid.unsqueeze(1), size=(latent_h, latent_w), mode="area").squeeze(1)
    known = (mv > 0.5).float()
    mask_lat_size = known.unsqueeze(0).unsqueeze(0)
    first = torch.repeat_interleave(mask_lat_size[:, :, 0:1], dim=2, repeats=temporal)
    packed = torch.concat([first, mask_lat_size[:, :, 1:]], dim=2)
    packed = packed.view(1, -1, temporal, latent_h, latent_w)
    return packed.transpose(1, 2)


def hidden_36(noise: torch.Tensor, packed_mask: torch.Tensor, concat_latent: torch.Tensor) -> torch.Tensor:
    condition = torch.concat([packed_mask.to(concat_latent.dtype), concat_latent], dim=1)
    return torch.concat([noise, condition], dim=1)
