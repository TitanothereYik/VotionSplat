"""Turn an HY-World checkpoint into an Inria splat ply in the panorama frame. Numpy only.

The trainer's ``transform`` T maps the HY world to its normalised frame: p_n = s R p_hy + t.
HY's world is Z-up. ``M`` maps our OpenCV y-down points into it: p_hy = M p_ours.
So p_ours = Q (p_n - t) / s with Q = M^T R^T.
"""

from __future__ import annotations

import math

import numpy as np

M = np.array([[0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]], dtype=np.float64)


def similarity(transform: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    t = np.asarray(transform, dtype=np.float64)
    sr = t[:3, :3]
    det = float(np.linalg.det(sr))
    if not math.isfinite(det) or det <= 0.0:
        raise ValueError("trainer transform is not a similarity (determinant <= 0)")
    s = det ** (1.0 / 3.0)
    r = sr / s
    if float(np.abs(r @ r.T - np.eye(3)).max()) > 1e-3:
        raise ValueError("trainer transform is not a similarity (rotation is not orthonormal)")
    return s, r, t[:3, 3].copy()


def world_points(means_n: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Undo the trainer normalisation, then convert HY world to OpenCV y-down. Rows are points."""
    s, r, t = similarity(transform)
    hy = ((np.asarray(means_n, dtype=np.float64) - t) / s) @ r
    return hy @ M


def quat_from_matrix(m: np.ndarray) -> np.ndarray:
    """Rotation matrix to wxyz."""
    m = np.asarray(m, dtype=np.float64)
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0.0:
        k = math.sqrt(tr + 1.0) * 2.0
        q = [0.25 * k, (m[2, 1] - m[1, 2]) / k, (m[0, 2] - m[2, 0]) / k, (m[1, 0] - m[0, 1]) / k]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        k = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        q = [(m[2, 1] - m[1, 2]) / k, 0.25 * k, (m[0, 1] + m[1, 0]) / k, (m[0, 2] + m[2, 0]) / k]
    elif m[1, 1] > m[2, 2]:
        k = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        q = [(m[0, 2] - m[2, 0]) / k, (m[0, 1] + m[1, 0]) / k, 0.25 * k, (m[1, 2] + m[2, 1]) / k]
    else:
        k = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        q = [(m[1, 0] - m[0, 1]) / k, (m[0, 2] + m[2, 0]) / k, (m[1, 2] + m[2, 1]) / k, 0.25 * k]
    return np.asarray(q, dtype=np.float64)


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product a * b, wxyz. ``a`` is [4], ``b`` is [N, 4]."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    return np.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        axis=1,
    )


def to_ply_arrays(splats: dict, transform: np.ndarray) -> dict:
    """gsplat state dict (numpy) -> the layout ``engine.splat.ply_export.write_splat_ply`` reads."""
    s, r, _t = similarity(transform)
    means = np.asarray(splats["means"], dtype=np.float64)
    keep = np.ones(means.shape[0], dtype=bool)
    if "mask_score" in splats:
        score = np.asarray(splats["mask_score"])
        keep = score[:, 0] > score[:, 1]
    shn = np.asarray(splats.get("shN", np.zeros((means.shape[0], 0, 3))))
    if shn.shape[1] > 0:
        raise ValueError("this checkpoint has SH bands above degree 0; the export rotates degree 0 only")
    quats = np.asarray(splats["quats"], dtype=np.float64)[keep]
    quats = quats / np.clip(np.linalg.norm(quats, axis=1, keepdims=True), 1e-12, None)
    rot = quat_mul(quat_from_matrix(M.T @ r.T), quats)
    count = int(keep.sum())
    return {
        "xyz": world_points(means[keep], transform).T,
        "scale": (np.asarray(splats["scales"], dtype=np.float64)[keep] - math.log(s)).T,
        "rot": rot.T,
        "sh_0": np.asarray(splats["sh0"], dtype=np.float64)[keep].transpose(1, 2, 0),
        "sh_rest": np.zeros((0, 3, count), dtype=np.float32),
        "opacity": np.asarray(splats["opacities"], dtype=np.float64)[keep].reshape(-1),
    }
