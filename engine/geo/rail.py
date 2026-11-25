"""Catmull-Rom rail + camera-to-world stack.

Ported from ComfyUI-SplatKit (MIT) ``nodes/camera_plot.py``, pin f59de25. See THIRD_PARTY_NOTICES.md.
(``_camplot_catmull_rom``, ``_camplot_catmull_rom_tangent``, ``_camplot_c2w_stack``,
``_camplot_fill_targets``). No ComfyUI imports.

Editor frame is +Y up. Render/OpenCV frame is +Y down: ``P_render = [x, -y, z]``.
``nvrender.load_rail`` expects a JSON list of 4x4 **world-to-camera** matrices.
"""

from __future__ import annotations

import numpy as np

from engine.geo.anchors import parse_anchors_ext, parse_point


def fill_targets(positions: np.ndarray, targets: np.ndarray) -> np.ndarray:
    n = positions.shape[0]
    out = np.array(targets, dtype=np.float64, copy=True)
    for i in range(n):
        if not np.all(np.isfinite(out[i])):
            if i < n - 1:
                out[i] = positions[i] + (positions[i + 1] - positions[i])
            elif n >= 2:
                out[i] = positions[i] + (positions[i] - positions[i - 1])
            else:
                out[i] = positions[i] + np.array([0.0, 0.0, 1.0])
    return out


def catmull_rom(anchors: np.ndarray, n_samples: int) -> np.ndarray:
    pts = np.asarray(anchors, dtype=np.float64)
    n = pts.shape[0]
    if n == 2:
        u = np.linspace(0.0, 1.0, n_samples)[:, None]
        return (1.0 - u) * pts[0][None] + u * pts[1][None]
    p0 = 2.0 * pts[0] - pts[1]
    pn = 2.0 * pts[-1] - pts[-2]
    ext = np.vstack([p0, pts, pn])
    us = np.linspace(0.0, n - 1, n_samples)
    out = np.empty((n_samples, 3), dtype=np.float64)
    for j, u in enumerate(us):
        k = min(int(np.floor(u)), n - 2)
        t = u - k
        p0_, p1, p2, p3 = ext[k], ext[k + 1], ext[k + 2], ext[k + 3]
        t2, t3 = t * t, t * t * t
        out[j] = 0.5 * (
            (2.0 * p1)
            + (-p0_ + p2) * t
            + (2.0 * p0_ - 5.0 * p1 + 4.0 * p2 - p3) * t2
            + (-p0_ + 3.0 * p1 - 3.0 * p2 + p3) * t3
        )
    return out


def catmull_rom_tangent(anchors: np.ndarray, n_samples: int) -> np.ndarray:
    pts = np.asarray(anchors, dtype=np.float64)
    n = pts.shape[0]
    n_samples = max(int(n_samples), 1)
    if n == 2:
        return np.tile(pts[1] - pts[0], (n_samples, 1))
    p0 = 2.0 * pts[0] - pts[1]
    pn = 2.0 * pts[-1] - pts[-2]
    ext = np.vstack([p0, pts, pn])
    us = np.linspace(0.0, n - 1, n_samples)
    out = np.empty((n_samples, 3), dtype=np.float64)
    for j, u in enumerate(us):
        k = min(int(np.floor(u)), n - 2)
        t = u - k
        p0_, p1, p2, p3 = ext[k], ext[k + 1], ext[k + 2], ext[k + 3]
        t2 = t * t
        out[j] = 0.5 * (
            (-p0_ + p2)
            + 2.0 * (2.0 * p0_ - 5.0 * p1 + 4.0 * p2 - p3) * t
            + 3.0 * (-p0_ + 3.0 * p1 - 3.0 * p2 + p3) * t2
        )
    return out


def c2w_stack(positions: np.ndarray, mode: str, target=None, anchors=None) -> np.ndarray:
    t_count = positions.shape[0]
    c2w = np.tile(np.eye(4, dtype=np.float64), (t_count, 1, 1))
    c2w[:, :3, 3] = positions
    if mode == "fixed_forward":
        return c2w
    if mode == "per_point_look":
        fwd = np.asarray(target, dtype=np.float64) - positions
    elif mode == "look_at_target":
        if target is None:
            target = np.array([0.0, 0.0, 1.0])
        fwd = np.asarray(target, dtype=np.float64)[None, :] - positions
    else:
        if anchors is not None and np.asarray(anchors).shape[0] >= 2:
            fwd = catmull_rom_tangent(anchors, t_count)
        elif t_count > 1:
            fwd = np.gradient(positions, axis=0)
        else:
            fwd = np.tile(np.array([0.0, 0.0, 1.0]), (t_count, 1))
    world_up = np.array([0.0, 1.0, 0.0])
    prev_z = np.array([0.0, 0.0, 1.0])
    prev_x = np.array([1.0, 0.0, 0.0])
    for i in range(t_count):
        f = fwd[i]
        nrm = np.linalg.norm(f)
        z = f / nrm if nrm > 1e-8 else prev_z
        x = np.cross(world_up, z)
        xn = np.linalg.norm(x)
        if xn > 5e-2:
            x = x / xn
        else:
            x = prev_x - z * np.dot(prev_x, z)
            xn2 = np.linalg.norm(x)
            x = x / xn2 if xn2 > 1e-8 else prev_x
        y = np.cross(z, x)
        c2w[i, :3, 0] = x
        c2w[i, :3, 1] = y
        c2w[i, :3, 2] = z
        prev_z, prev_x = z, x
    return c2w


def snap_length(n: int) -> int:
    n = max(9, min(257, int(n)))
    k = int(round((n - 9) / 4.0))
    return 9 + 4 * k


def build_w2c(
    anchors_text: str,
    orientation: str,
    look_at_text: str,
    length: int,
) -> np.ndarray:
    """Editor +Y up anchors → render-frame w2c stack (T,4,4)."""
    anchor_pts, anchor_tgts = parse_anchors_ext(anchors_text)
    anchor_pts[0] = 0.0
    length = snap_length(length)
    render_pts = anchor_pts.copy()
    render_pts[:, 1] *= -1.0
    positions = catmull_rom(render_pts, length)
    if orientation == "per_point_look":
        tgt_anchors = fill_targets(anchor_pts, anchor_tgts)
        tgt_render = tgt_anchors.copy()
        tgt_render[:, 1] *= -1.0
        per_frame = catmull_rom(tgt_render, length)
        c2w = c2w_stack(positions, "per_point_look", per_frame)
    elif orientation == "look_at_target":
        target_editor = parse_point(look_at_text, (0.0, 0.0, 1.0))
        render_target = target_editor.copy()
        render_target[1] *= -1.0
        c2w = c2w_stack(positions, "look_at_target", render_target, anchors=render_pts)
    else:
        c2w = c2w_stack(positions, "look_forward", None, anchors=render_pts)
    return np.linalg.inv(c2w)
