"""Paste the source photo back onto the finished 8K panorama.

The upscale tail repaints the whole frame, including the photo. This puts the
original pixels back inside the h_fov footprint and blends the seam.
"""

from __future__ import annotations

import math

import cv2
import numpy as np
import torch

from engine.pano.defaults import PHOTO_FEATHER_PX, PHOTO_PYRAMID_LEVELS, WARP_SUPERSAMPLE
from engine.pano.warp import persp_to_erp


def paste_photo(
    pano: np.ndarray,
    photo: np.ndarray,
    h_fov: float,
    feather_px: int | None = None,
    levels: int | None = None,
) -> np.ndarray:
    """``pano`` and ``photo`` are uint8 RGB. The footprint outside the feather is unchanged."""
    base = np.asarray(pano)
    if base.ndim != 3 or base.shape[2] < 3:
        raise ValueError("pano must be HxWx3")
    height, width = int(base.shape[0]), int(base.shape[1])
    src = np.asarray(photo)
    if src.ndim != 3 or src.shape[2] < 3:
        raise ValueError("photo must be HxWx3")
    feather = PHOTO_FEATHER_PX if feather_px is None else int(feather_px)
    pyramid_levels = PHOTO_PYRAMID_LEVELS if levels is None else int(levels)
    photo_t = torch.from_numpy(np.ascontiguousarray(src[..., :3].astype(np.float32) / 255.0)).unsqueeze(0)
    canvas, inpaint = persp_to_erp(
        photo_t,
        width=width,
        height=height,
        h_fov_deg=float(h_fov),
        supersample=WARP_SUPERSAMPLE,
        mask_feather=0,
    )
    warp = canvas[0].detach().cpu().numpy().astype(np.float32)
    valid = inpaint[0].detach().cpu().numpy() < 0.5
    if not bool(valid.any()):
        return np.ascontiguousarray(base[..., :3])
    dist = cv2.distanceTransform(valid.astype(np.uint8), cv2.DIST_L2, 5)
    span = max(int(feather), 1)
    ramp = np.clip(dist / float(span), 0.0, 1.0)
    weight = np.zeros(valid.shape, dtype=np.float32)
    weight[valid] = (0.5 - 0.5 * np.cos(np.pi * ramp[valid])).astype(np.float32)
    ys, xs = np.nonzero(valid)
    pad = (2 ** max(pyramid_levels, 0)) * 4
    y0 = max(int(ys.min()) - pad, 0)
    y1 = min(int(ys.max()) + 1 + pad, height)
    x0 = max(int(xs.min()) - pad, 0)
    x1 = min(int(xs.max()) + 1 + pad, width)
    step = 2 ** max(pyramid_levels, 0)
    y1 = y0 + max(step, ((y1 - y0) // step) * step)
    x1 = x0 + max(step, ((x1 - x0) // step) * step)
    y1 = min(y1, height)
    x1 = min(x1, width)
    y0 = max(y1 - ((y1 - y0) // step) * step, 0)
    x0 = max(x1 - ((x1 - x0) // step) * step, 0)
    if y1 <= y0 or x1 <= x0:
        return np.ascontiguousarray(base[..., :3])
    pano_crop = base[y0:y1, x0:x1, :3].astype(np.float32) / 255.0
    photo_crop = warp[y0:y1, x0:x1]
    weight_crop = weight[y0:y1, x0:x1]
    blended = _laplacian_blend(photo_crop, pano_crop, weight_crop, pyramid_levels)
    keep = weight_crop < 1e-6
    blended[keep] = pano_crop[keep]
    out = np.array(base[..., :3], copy=True)
    out[y0:y1, x0:x1] = np.clip(np.rint(blended * 255.0), 0, 255).astype(np.uint8)
    return out


def _down(image: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(image, (5, 5), 1.0)
    return blurred[::2, ::2]


def _up(image: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    return cv2.resize(image, (int(shape[1]), int(shape[0])), interpolation=cv2.INTER_LINEAR)


def _laplacian_blend(photo: np.ndarray, pano: np.ndarray, weight: np.ndarray, levels: int) -> np.ndarray:
    photo_g = [photo]
    pano_g = [pano]
    weight_g = [weight]
    for _ in range(max(0, int(levels))):
        if min(photo_g[-1].shape[0], photo_g[-1].shape[1]) < 4:
            break
        photo_g.append(_down(photo_g[-1]))
        pano_g.append(_down(pano_g[-1]))
        weight_g.append(_down(weight_g[-1]))
    photo_l = [photo_g[-1]]
    pano_l = [pano_g[-1]]
    for index in range(len(photo_g) - 2, -1, -1):
        target = photo_g[index].shape[:2]
        photo_l.append(photo_g[index] - _up(photo_g[index + 1], target))
        pano_l.append(pano_g[index] - _up(pano_g[index + 1], target))
    mixed: list[np.ndarray] = []
    for index, (photo_level, pano_level) in enumerate(zip(photo_l, pano_l)):
        mask = weight_g[len(weight_g) - 1 - index]
        if mask.shape[:2] != photo_level.shape[:2]:
            mask = _up(mask, photo_level.shape[:2])
        mask = mask[..., None]
        mixed.append(mask * photo_level + (1.0 - mask) * pano_level)
    image = mixed[0]
    for index in range(1, len(mixed)):
        image = _up(image, mixed[index].shape[:2]) + mixed[index]
    return image.astype(np.float32)
