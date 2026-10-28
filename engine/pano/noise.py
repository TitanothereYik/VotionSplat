"""ImageAddNoise from ComfyUI v0.32.0 comfy_extras/nodes_images.py execute()."""

from __future__ import annotations

import torch

from engine.pano.defaults import NOISE_SEED, NOISE_STRENGTH


def add_noise(image: torch.Tensor, seed: int = NOISE_SEED, strength: float = NOISE_STRENGTH) -> torch.Tensor:
    generator = torch.manual_seed(int(seed))
    image = image.detach()
    return torch.clip(
        image + strength * torch.randn(image.size(), generator=generator, device="cpu").to(image),
        min=0.0,
        max=1.0,
    )
