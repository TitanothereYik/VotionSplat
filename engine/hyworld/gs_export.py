"""Turn an HY-World checkpoint into an Inria splat ply in the panorama frame. Numpy only.

The trainer's ``transform`` T maps the HY world to its normalised frame: p_n = s R p_hy + t.
HY's world is Z-up. ``M`` maps our OpenCV y-down points into it: p_hy = M p_ours.
So p_ours = Q (p_n - t) / s with Q = M^T R^T.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

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


def keep_mask(splats: dict) -> np.ndarray:
    """Gaussians the trainer renders at inference (``mask_score`` argmax)."""
    keep = np.ones(np.asarray(splats["means"]).shape[0], dtype=bool)
    if "mask_score" in splats:
        score = np.asarray(splats["mask_score"])
        keep = score[:, 0] > score[:, 1]
    return keep


def load_training_cameras(result_dir: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """(c2w [C,4,4], K [C,3,3]) of the training views in the checkpoint frame, or None when the
    run was not trained antialiased or its camera files are missing."""
    cfg_path = result_dir / "cfg.yml"
    cams_path = result_dir / "votion_cameras.json"
    if not cfg_path.is_file() or not cams_path.is_file():
        return None
    cfg: dict[str, str] = {}
    for line in cfg_path.read_text(encoding="utf-8").splitlines():
        found = re.match(r"^(\w+):\s*(.*?)\s*$", line)
        if found:
            cfg[found.group(1)] = found.group(2).strip("'\"")
    if cfg.get("antialiased", "").lower() != "true":
        return None
    intr_path = Path(cfg.get("data_dir", "")) / "cameras.json"
    if not intr_path.is_file():
        return None
    factor = float(cfg.get("data_factor", "1") or 1)
    intrinsics = json.loads(intr_path.read_text(encoding="utf-8"))
    cams = json.loads(cams_path.read_text(encoding="utf-8"))
    c2ws, ks = [], []
    for name, c2w in zip(cams.get("names") or [], cams.get("c2w") or []):
        entry = intrinsics.get(name)
        if not entry:
            continue
        k = np.asarray(entry["intrinsic"], dtype=np.float64).copy()
        k[:2] /= factor
        c2ws.append(np.asarray(c2w, dtype=np.float64))
        ks.append(k)
    if not c2ws:
        return None
    return np.stack(c2ws), np.stack(ks)


def _quat_rotations(quats: np.ndarray) -> np.ndarray:
    q = quats / np.clip(np.linalg.norm(quats, axis=1, keepdims=True), 1e-12, None)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], 1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], 1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], 1),
    ], 1)


def soften_splats(
    splats: dict,
    c2ws: np.ndarray,
    ks: np.ndarray,
    *,
    focal: float,
    extra_px2: float,
    total_px2: float,
    chunk: int = 16384,
    progress=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Bake a screen-space low-pass into each kept Gaussian for classic renderers (WallGS, Luma,
    LiteGS: +0.3 px^2, full opacity). The reference view sits at the nearest training camera that
    frames the Gaussian (any camera in front if none does), with pinhole ``focal``. Returns
    (isotropic world variance to add, opacity factor): the variance widens the footprint there by
    ``extra_px2``, and the factor sqrt(det(S) / det(S + total_px2)) keeps the footprint's integral,
    like gsplat's antialiased mode with ``total_px2`` in place of 0.3. The factor comes from the 2D
    footprint, not the 3D volume, so flat surface splats seen face-on keep their opacity.
    Gaussians behind every camera are left alone."""
    keep = keep_mask(splats)
    means = np.asarray(splats["means"], dtype=np.float64)[keep]
    quats = np.asarray(splats["quats"], dtype=np.float64)[keep]
    scales = np.exp(np.asarray(splats["scales"], dtype=np.float64)[keep])
    w2c = np.linalg.inv(np.asarray(c2ws, dtype=np.float64))
    rot = w2c[:, :3, :3]
    trans = w2c[:, :3, 3]
    tan_x = (ks[:, 0, 2] / ks[:, 0, 0]).astype(np.float32)[None]
    tan_y = (ks[:, 1, 2] / ks[:, 1, 1]).astype(np.float32)[None]
    cams = rot.shape[0]
    project = rot.transpose(2, 0, 1).reshape(3, cams * 3).astype(np.float32)
    trans32 = trans.astype(np.float32)[None]
    total = means.shape[0]
    var_add = np.zeros(total, dtype=np.float64)
    factor = np.ones(total, dtype=np.float64)
    f = float(focal)
    for start in range(0, total, chunk):
        stop = min(start + chunk, total)
        m = means[start:stop]
        p = (m.astype(np.float32) @ project).reshape(-1, cams, 3) + trans32
        z = p[..., 2]
        front = z > 0.01
        near = np.where(front, -z, -np.inf)
        framed = front & (np.abs(p[..., 0]) <= tan_x * z) & (np.abs(p[..., 1]) <= tan_y * z)
        best = np.where(framed.any(1), np.argmax(np.where(framed, near, -np.inf), 1), np.argmax(near, 1))
        seen = front.any(1)
        rb = rot[best]
        pb = np.einsum("bij,bj->bi", rb, m) + trans[best]
        zb = np.maximum(pb[:, 2], 0.01)
        jac = np.zeros((m.shape[0], 2, 3))
        jac[:, 0, 0] = f / zb
        jac[:, 0, 2] = -f * pb[:, 0] / zb**2
        jac[:, 1, 1] = f / zb
        jac[:, 1, 2] = -f * pb[:, 1] / zb**2
        rq = rb @ _quat_rotations(quats[start:stop])
        cov = (rq * (scales[start:stop] ** 2)[:, None, :]) @ np.transpose(rq, (0, 2, 1))
        cov2 = jac @ cov @ np.transpose(jac, (0, 2, 1))
        a, b, c = cov2[:, 0, 0], cov2[:, 0, 1], cov2[:, 1, 1]
        det0 = np.maximum(a * c - b * b, 0.0)
        det = (a + total_px2) * (c + total_px2) - b * b
        factor[start:stop] = np.where(seen, np.sqrt(det0 / det), 1.0)
        var_add[start:stop] = np.where(seen, extra_px2 * (zb / f) ** 2, 0.0)
        if progress is not None:
            progress(stop, total)
    return var_add, factor


def to_ply_arrays(
    splats: dict, transform: np.ndarray, soften: tuple[np.ndarray, np.ndarray] | None = None
) -> dict:
    """gsplat state dict (numpy) -> the layout ``engine.splat.ply_export.write_splat_ply`` reads.
    ``soften`` is ``soften_splats`` output: (variance added to every scale axis, opacity factor)."""
    s, r, _t = similarity(transform)
    means = np.asarray(splats["means"], dtype=np.float64)
    keep = keep_mask(splats)
    shn = np.asarray(splats.get("shN", np.zeros((means.shape[0], 0, 3))))
    if shn.shape[1] > 0:
        raise ValueError("this checkpoint has SH bands above degree 0; the export rotates degree 0 only")
    quats = np.asarray(splats["quats"], dtype=np.float64)[keep]
    quats = quats / np.clip(np.linalg.norm(quats, axis=1, keepdims=True), 1e-12, None)
    rot = quat_mul(quat_from_matrix(M.T @ r.T), quats)
    count = int(keep.sum())
    opacity = np.asarray(splats["opacities"], dtype=np.float64)[keep].reshape(-1)
    log_scales = np.asarray(splats["scales"], dtype=np.float64)[keep]
    if soften is not None:
        var_add, factor = (np.asarray(v, dtype=np.float64).reshape(-1) for v in soften)
        log_scales = 0.5 * np.log(np.exp(2.0 * log_scales) + var_add[:, None])
        alpha = np.clip(1.0 / (1.0 + np.exp(-opacity)) * factor, 1e-7, 1.0 - 1e-7)
        opacity = np.where(factor < 1.0, np.log(alpha) - np.log1p(-alpha), opacity)
    return {
        "xyz": world_points(means[keep], transform).T,
        "scale": (log_scales - math.log(s)).T,
        "rot": rot.T,
        "sh_0": np.asarray(splats["sh0"], dtype=np.float64)[keep].transpose(1, 2, 0),
        "sh_rest": np.zeros((0, 3, count), dtype=np.float32),
        "opacity": opacity,
    }
