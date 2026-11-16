"""Align a frame depth map to a rendered depth map.

The fit is the same least-squares scale and shift ``known_pose`` used, moved
here so the world layer and the COLMAP writer share one implementation.
"""

from __future__ import annotations

import numpy as np

MIN_INLIERS = 500


def render_z(points: np.ndarray, w2c: np.ndarray, K: np.ndarray, height: int, width: int) -> np.ndarray:
    """Nearest OpenCV camera-z of ``points``. NaN where the cloud missed."""
    rot = np.asarray(w2c[:3, :3], dtype=np.float64)
    t = np.asarray(w2c[:3, 3], dtype=np.float64)
    cam = np.asarray(points, dtype=np.float64) @ rot.T + t
    z = cam[:, 2]
    ok = z > 1e-4
    inv_z = 1.0 / np.maximum(z, 1e-8)
    u = K[0, 0] * cam[:, 0] * inv_z + K[0, 2]
    v = K[1, 1] * cam[:, 1] * inv_z + K[1, 2]
    # Sky-fill and points behind the camera make u/v non-finite or huge.
    # Cast only the pixels that can land in the frame.
    in_frame = (
        np.isfinite(u)
        & np.isfinite(v)
        & (u >= -1.0)
        & (u < width + 1)
        & (v >= -1.0)
        & (v < height + 1)
    )
    ui = np.zeros(u.shape[0], dtype=np.int32)
    vi = np.zeros(v.shape[0], dtype=np.int32)
    if np.any(in_frame):
        ui[in_frame] = np.rint(u[in_frame]).astype(np.int32)
        vi[in_frame] = np.rint(v[in_frame]).astype(np.int32)
    ok &= in_frame & (ui >= 0) & (ui < width) & (vi >= 0) & (vi < height)
    depth = np.full(height * width, np.nan, dtype=np.float32)
    if not np.any(ok):
        return depth.reshape(height, width)
    ui = ui[ok]
    vi = vi[ok]
    z = z[ok]
    order = np.argsort(z, kind="mergesort")
    lin = vi[order].astype(np.int64) * width + ui[order].astype(np.int64)
    z = z[order]
    _uniq, first = np.unique(lin, return_index=True)
    depth[lin[first]] = z[first].astype(np.float32)
    return depth.reshape(height, width)


def align_depth(z_render: np.ndarray, z_pred: np.ndarray) -> tuple[float, float, np.ndarray, float] | None:
    """``z_render ≈ scale * z_pred + shift`` on pixels the cloud hit.

    Returns scale, shift, inlier mask, and the median relative residual.
    """
    valid = np.isfinite(z_render) & np.isfinite(z_pred) & (z_render > 1e-3) & (z_pred > 1e-3)
    idx = np.flatnonzero(valid.ravel())
    if idx.size < MIN_INLIERS:
        return None
    zr = z_render.ravel()[idx].astype(np.float64)
    zp = z_pred.ravel()[idx].astype(np.float64)

    def fit(sel: np.ndarray) -> tuple[float, float]:
        design = np.stack([zp[sel], np.ones(sel.size)], axis=1)
        scale, shift = np.linalg.lstsq(design, zr[sel], rcond=None)[0]
        return float(scale), float(shift)

    chosen = np.arange(idx.size)
    scale, shift = fit(chosen)
    if not np.isfinite(scale) or scale <= 0.0:
        return None
    resid = np.abs(scale * zp + shift - zr)
    med = float(np.median(resid))
    mad = float(np.median(np.abs(resid - med)))
    thresh = med + 3.0 * 1.4826 * max(mad, 1e-8)
    keep = resid <= thresh
    if int(keep.sum()) < MIN_INLIERS:
        return None
    scale, shift = fit(np.flatnonzero(keep))
    if not np.isfinite(scale) or scale <= 0.0:
        return None
    resid = np.abs(scale * zp + shift - zr)
    keep = resid <= thresh
    if int(keep.sum()) < MIN_INLIERS:
        return None
    inlier = np.zeros(z_render.size, dtype=bool)
    inlier[idx[keep]] = True
    rel = float(np.median(resid[keep] / np.maximum(zr[keep], 1e-8)))
    return scale, shift, inlier.reshape(z_render.shape), rel


def moge_frame_depth(model, rgb: np.ndarray, fov_x_deg: float):
    import torch

    image = torch.tensor(rgb, dtype=torch.float32, device=next(model.parameters()).device).permute(2, 0, 1) / 255.0
    out = model.infer(image, fov_x=float(fov_x_deg), resolution_level=9, apply_mask=True)
    depth = out["depth"].detach().float().cpu().numpy()
    return depth
