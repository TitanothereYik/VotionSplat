"""Mickmumpitz Krea2FullResReference.run at 4d5ff7c. Worker only; needs Comfy on path."""

from __future__ import annotations

import math

import torch

from engine.pano.defaults import KREA2_REFERENCE_MAX_MEGAPIXELS

SNAP = 16


def krea2_full_res_reference(conditioning, vae, image: torch.Tensor, max_megapixels: float = KREA2_REFERENCE_MAX_MEGAPIXELS):
    import comfy.utils
    import node_helpers

    s = image.movedim(-1, 1)
    h, w = int(s.shape[2]), int(s.shape[3])
    cap = int(float(max_megapixels) * 1024 * 1024)
    scale = min(1.0, math.sqrt(cap / float(w * h)))
    nw = max(int(round(w * scale / SNAP)) * SNAP, SNAP)
    nh = max(int(round(h * scale / SNAP)) * SNAP, SNAP)
    if (nh, nw) != (h, w):
        s = comfy.utils.common_upscale(s, nw, nh, "area", "disabled")
    latent = vae.encode(s.movedim(1, -1)[:, :, :, :3])
    print(
        f"STATUS // Krea2FullResReference {w}x{h} -> {nw}x{nh}",
        flush=True,
    )
    return node_helpers.conditioning_set_values(conditioning, {"reference_latents": [latent]})
