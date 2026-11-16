"""Solid panorama mesh from the 2048x1024 MoGe depth.

One vertex per depth pixel, plus a column that closes the longitude seam.
Triangles that stretch across an occlusion edge are dropped. Floor vertices
that already lie near one plane are pulled onto that plane along their ray,
so the texture coordinate does not move.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from engine.pano.scene_io import scene_dir
from engine.world.defaults import (
    DEPTH_RATIO,
    EDGE_LOG_JUMP,
    GRAZE_COS,
    GROUND_MAX_TILT_DEG,
    GROUND_MIN_BELOW_DEG,
    GROUND_MIN_INLIER_FRAC,
    GROUND_NORMAL_DEG,
    GROUND_RANSAC_ITERS,
    GROUND_SEED,
    GROUND_SNAP_REL,
    GROUND_TOL_REL,
    PLANE_MAX,
    PLANE_MIN_COMPONENT,
    PLANE_MIN_POINTS,
    PLANE_NORMAL_DEG,
    PLANE_RANSAC_ITERS,
    PLANE_TOL_REL,
)


@dataclass
class GroundPlane:
    normal: np.ndarray  # (3,) unit, points down (+Y)
    d: float
    erp_dist: np.ndarray | None = None  # (H, W) distance from the origin; NaN = not seen


@dataclass
class Piece:
    verts: np.ndarray  # (N, 3) float32 world, OpenCV axes
    faces: np.ndarray  # (M, 3) int32 indices into verts
    uvs: np.ndarray  # (N, 2) float32, (0, 0) = top-left of the texture, (1, 1) = bottom-right
    tex: str  # "pano", or a scene-relative PNG path


def _log(msg: str) -> None:
    print(msg, flush=True)


def world_dir(scene: str) -> Path:
    return scene_dir(scene) / "worldstereo" / "world"


def erp_dirs(height: int, width: int) -> np.ndarray:
    """Unit directions (H, W+1, 3). Column ``width`` repeats longitude 0."""
    xs = (np.arange(width + 1, dtype=np.float64) + 0.5) / float(width)
    ys = (np.arange(height, dtype=np.float64) + 0.5) / float(height)
    uu, vv = np.meshgrid(xs, ys)
    lon = uu * (2.0 * np.pi) - np.pi
    lat = (np.pi / 2.0) - vv * np.pi
    cos_lat = np.cos(lat)
    return np.stack([cos_lat * np.sin(lon), -np.sin(lat), cos_lat * np.cos(lon)], axis=-1)


def fit_ground(candidates: np.ndarray, depth_scale: float) -> tuple[np.ndarray, float, float] | None:
    """RANSAC plane through points below the horizon. ``n`` points down (+Y)."""
    pts = np.asarray(candidates, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] < 3 or pts.shape[1] != 3:
        return None
    rng = np.random.default_rng(GROUND_SEED)
    if pts.shape[0] > 200_000:
        pts = pts[rng.choice(pts.shape[0], 200_000, replace=False)]
    count = int(pts.shape[0])
    tol = float(GROUND_TOL_REL) * float(depth_scale)
    min_down = float(np.cos(np.radians(GROUND_MAX_TILT_DEG)))
    best_count = 0
    best_inliers: np.ndarray | None = None
    for _ in range(int(GROUND_RANSAC_ITERS)):
        choice = rng.choice(count, 3, replace=False)
        a, b, c = pts[choice]
        normal = np.cross(b - a, c - a)
        length = float(np.linalg.norm(normal))
        if length < 1e-8:
            continue
        normal = normal / length
        if normal[1] < 0.0:
            normal = -normal
        if float(normal[1]) < min_down:
            continue
        dist = np.abs(pts @ normal + (-float(np.dot(normal, a))))
        inliers = dist < tol
        hits = int(inliers.sum())
        if hits > best_count:
            best_count = hits
            best_inliers = inliers
    if best_inliers is None or best_count / count < GROUND_MIN_INLIER_FRAC:
        return None
    subset = pts[best_inliers]
    mean = subset.mean(axis=0)
    _u, _s, vh = np.linalg.svd(subset - mean, full_matrices=False)
    normal = vh[-1].astype(np.float64)
    length = float(np.linalg.norm(normal))
    if length < 1e-8:
        return None
    normal = normal / length
    if normal[1] < 0.0:
        normal = -normal
    plane_d = -float(np.dot(normal, mean))
    fraction = float(np.mean(np.abs(pts @ normal + plane_d) < tol))
    if fraction < GROUND_MIN_INLIER_FRAC:
        return None
    return normal, plane_d, fraction


def snap_to_ground(
    depth_grid: np.ndarray,
    dirs: np.ndarray,
    below_grid: np.ndarray,
    normal: np.ndarray,
    plane_d: float,
    normals: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Pull near-floor vertices onto the plane along their own ray."""
    depth = np.array(depth_grid, dtype=np.float64, copy=True)
    align = dirs @ np.asarray(normal, dtype=np.float64)
    snapped = np.zeros(depth.shape, dtype=bool)
    usable = np.asarray(below_grid, dtype=bool) & (align > 1e-6) & (depth > 1e-6)
    if normals is not None:
        agree = np.abs(np.sum(normals * np.asarray(normal, dtype=np.float64), axis=-1))
        usable = usable & (agree > math.cos(math.radians(GROUND_NORMAL_DEG)))
    if not np.any(usable):
        return depth, snapped
    hit = np.zeros_like(depth)
    hit[usable] = -float(plane_d) / align[usable]
    close = usable & (np.abs(hit - depth) / depth < GROUND_SNAP_REL) & np.isfinite(hit) & (hit > 1e-6)
    depth[close] = hit[close]
    snapped[close] = True
    return depth, snapped


def erp_normals(points: np.ndarray) -> np.ndarray:
    """Outward-from-surface normals. ``points`` is (H, W, 3), seam already closed or not."""
    right = np.roll(points, -1, axis=1)
    left = np.roll(points, 1, axis=1)
    tangent_u = right - left
    tangent_v = np.empty_like(points)
    tangent_v[1:-1] = points[2:] - points[:-2]
    tangent_v[0] = points[1] - points[0]
    tangent_v[-1] = points[-1] - points[-2]
    normal = np.cross(tangent_u, tangent_v)
    length = np.linalg.norm(normal, axis=-1, keepdims=True)
    normal = normal / np.maximum(length, 1e-12)
    flip = np.sum(normal * -points, axis=-1) < 0.0
    normal[flip] *= -1.0
    return normal.astype(np.float64)


def fit_planes(
    points: np.ndarray,
    normals: np.ndarray,
    valid: np.ndarray,
    exclude: np.ndarray,
    scale: float,
) -> list[tuple[np.ndarray, float, np.ndarray]]:
    """Up to ``PLANE_MAX`` facade planes. Each mask is the pixels that may snap."""
    candidate = np.asarray(valid, dtype=bool) & ~np.asarray(exclude, dtype=bool)
    candidate &= np.abs(normals[..., 1]) < math.sin(math.radians(20.0))
    remaining = candidate.copy()
    flat_points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    flat_normals = np.asarray(normals, dtype=np.float64).reshape(-1, 3)
    rng = np.random.default_rng(GROUND_SEED)
    tol = float(PLANE_TOL_REL) * max(float(scale), 1e-6)
    agree_cos = math.cos(math.radians(PLANE_NORMAL_DEG))
    found: list[tuple[np.ndarray, float, np.ndarray]] = []
    for _plane_index in range(int(PLANE_MAX)):
        index = np.flatnonzero(remaining.reshape(-1))
        if index.size < int(PLANE_MIN_POINTS):
            break
        pts = flat_points[index]
        nrm = flat_normals[index]
        best_count = 0
        best_normal = None
        best_d = 0.0
        best_hits = None
        count = int(pts.shape[0])
        for _ in range(int(PLANE_RANSAC_ITERS)):
            choice = rng.choice(count, 3, replace=False)
            a, b, c = pts[choice]
            normal = np.cross(b - a, c - a)
            length = float(np.linalg.norm(normal))
            if length < 1e-8:
                continue
            normal = normal / length
            if float(np.dot(normal, -a)) < 0.0:
                normal = -normal
            dist = np.abs(pts @ normal - float(np.dot(normal, a)))
            hits = (dist < tol) & (np.abs(nrm @ normal) > agree_cos)
            hits_n = int(hits.sum())
            if hits_n > best_count:
                best_count = hits_n
                best_normal = normal
                best_d = -float(np.dot(normal, a))
                best_hits = hits
        if best_normal is None or best_count < int(PLANE_MIN_POINTS) or best_hits is None:
            break
        subset = pts[best_hits]
        mean = subset.mean(axis=0)
        _u, _s, vh = np.linalg.svd(subset - mean, full_matrices=False)
        normal = vh[-1].astype(np.float64)
        length = float(np.linalg.norm(normal))
        if length < 1e-8:
            break
        normal = normal / length
        if float(np.dot(normal, -mean)) < 0.0:
            normal = -normal
        plane_d = -float(np.dot(normal, mean))
        mask = np.zeros(remaining.size, dtype=bool)
        mask[index[best_hits]] = True
        mask = mask.reshape(remaining.shape)
        _count, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
        keep = np.zeros(mask.shape, dtype=bool)
        for label in range(1, int(_count)):
            component = labels == label
            if int(component.sum()) >= int(PLANE_MIN_COMPONENT):
                keep |= component
        if not bool(keep.any()):
            remaining[mask] = False
            continue
        found.append((normal, plane_d, keep))
        remaining[keep] = False
    return found


def depth_edge_mask(depth: np.ndarray, valid: np.ndarray, exempt: np.ndarray) -> np.ndarray:
    """Pixels whose log-distance jumps to a neighbour, grown by one pixel."""
    log_depth = np.log(np.maximum(np.asarray(depth, dtype=np.float64), 1e-6))
    log_depth = np.where(valid, log_depth, np.nan)
    jump = np.zeros(log_depth.shape, dtype=bool)
    for shift, axis in ((1, 0), (-1, 0), (1, 1), (-1, 1)):
        moved = np.roll(log_depth, shift, axis=axis)
        if axis == 0:
            if shift > 0:
                moved[:shift] = np.nan
            else:
                moved[shift:] = np.nan
        diff = np.abs(log_depth - moved)
        jump |= np.isfinite(diff) & (diff > float(EDGE_LOG_JUMP))
    jump &= np.asarray(valid, dtype=bool) & ~np.asarray(exempt, dtype=bool)
    grown = cv2.dilate(jump.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1) > 0
    return grown & np.asarray(valid, dtype=bool)


def grid_faces(valid_grid: np.ndarray) -> np.ndarray:
    """Two triangles per quad whose four corners are valid."""
    valid = np.asarray(valid_grid, dtype=bool)
    height, width_p = valid.shape
    if height < 2 or width_p < 2:
        return np.zeros((0, 3), np.int32)
    width = width_p - 1
    yy, xx = np.mgrid[0 : height - 1, 0:width]
    v00 = yy * width_p + xx
    v10 = v00 + 1
    v01 = v00 + width_p
    v11 = v01 + 1
    ok = valid[yy, xx] & valid[yy, xx + 1] & valid[yy + 1, xx] & valid[yy + 1, xx + 1]
    if not np.any(ok):
        return np.zeros((0, 3), np.int32)
    first = np.stack([v00[ok], v10[ok], v01[ok]], axis=1)
    second = np.stack([v10[ok], v11[ok], v01[ok]], axis=1)
    return np.concatenate([first, second], axis=0).astype(np.int32)


def cull_faces(verts: np.ndarray, faces: np.ndarray, center: np.ndarray) -> np.ndarray:
    """Drop grazing triangles and triangles that jump in depth."""
    faces = np.asarray(faces, dtype=np.int64)
    if faces.size == 0:
        return np.zeros((0, 3), np.int32)
    points = np.asarray(verts, dtype=np.float64)
    origin = np.asarray(center, dtype=np.float64).reshape(3)
    kept: list[np.ndarray] = []
    step = 1_000_000
    for start in range(0, faces.shape[0], step):
        chunk = faces[start : start + step]
        tri = points[chunk]
        finite = np.isfinite(tri).all(axis=(1, 2))
        a = tri[:, 0]
        b = tri[:, 1]
        c = tri[:, 2]
        normal = np.cross(b - a, c - a)
        nlen = np.linalg.norm(normal, axis=1)
        unit_n = normal / np.maximum(nlen[:, None], 1e-12)
        centroid = tri.mean(axis=1)
        ray = centroid - origin
        rlen = np.linalg.norm(ray, axis=1)
        unit_r = ray / np.maximum(rlen[:, None], 1e-12)
        graze = np.abs(np.sum(unit_n * unit_r, axis=1)) < GRAZE_COS
        dist = np.linalg.norm(tri - origin, axis=2)
        ratio = dist.max(axis=1) / np.maximum(dist.min(axis=1), 1e-8)
        stretch = ratio > DEPTH_RATIO
        drop = (~finite) | graze | stretch | (nlen < 1e-12)
        if np.any(~drop):
            kept.append(chunk[~drop])
    if not kept:
        return np.zeros((0, 3), np.int32)
    return np.concatenate(kept, axis=0).astype(np.int32)


def _snap_mask(depth_grid: np.ndarray, dirs: np.ndarray, mask: np.ndarray, normal: np.ndarray, plane_d: float) -> np.ndarray:
    depth = np.array(depth_grid, copy=True)
    align = dirs @ np.asarray(normal, dtype=np.float64)
    usable = np.asarray(mask, dtype=bool) & (align > 0.05)
    if not np.any(usable):
        return depth
    depth[usable] = -float(plane_d) / align[usable]
    return depth


def build_pano_mesh(
    depth: np.ndarray,
    valid: np.ndarray,
    snap_ground: bool = True,
    normals: np.ndarray | None = None,
) -> tuple[Piece, dict]:
    depth = np.asarray(depth, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    height, width = depth.shape
    depth_grid = np.concatenate([depth, depth[:, :1]], axis=1)
    valid_grid = np.concatenate([valid, valid[:, :1]], axis=1)
    dirs = erp_dirs(height, width)
    below = valid_grid & (dirs[..., 1] > np.sin(np.radians(GROUND_MIN_BELOW_DEG)))
    if normals is None:
        normals_grid = erp_normals(dirs * depth_grid[..., None])
    else:
        supplied = np.asarray(normals, dtype=np.float64)
        if supplied.shape[:2] != (height, width):
            raise ValueError("normals must match the depth image")
        normals_grid = np.concatenate([supplied, supplied[:, :1]], axis=1)
    snapped = np.zeros(valid_grid.shape, dtype=bool)
    ground: dict | None = None
    if snap_ground and np.any(valid):
        scale = float(np.median(depth[valid])) if np.any(valid) else 1.0
        candidates = (dirs * depth_grid[..., None])[below]
        plane = fit_ground(candidates, scale) if candidates.shape[0] >= 3 else None
        if plane is not None:
            normal, plane_d, fraction = plane
            depth_grid, snapped = snap_to_ground(
                depth_grid, dirs, below, normal, plane_d, normals_grid
            )
            ground = {
                "normal": [float(v) for v in normal],
                "d": float(plane_d),
                "inliers": float(fraction),
                "snapped": int(snapped.sum()),
            }
    plane_stats: list[dict] = []
    if snap_ground and np.any(valid):
        scale = float(np.median(depth_grid[valid_grid])) if np.any(valid_grid) else 1.0
        image_points = (dirs * depth_grid[..., None])[:, :width]
        image_normals = normals_grid[:, :width]
        for normal, plane_d, mask in fit_planes(
            image_points, image_normals, valid, snapped[:, :width], scale
        ):
            seam = np.concatenate([mask, mask[:, :1]], axis=1)
            depth_grid = _snap_mask(depth_grid, dirs, seam, normal, plane_d)
            snapped |= seam
            plane_stats.append(
                {
                    "normal": [float(v) for v in normal],
                    "d": float(plane_d),
                    "snapped": int(mask.sum()),
                }
            )
    points = (dirs * depth_grid[..., None]).reshape(-1, 3)
    raw_faces = grid_faces(valid_grid)
    faces = cull_faces(points, raw_faces, np.zeros(3, dtype=np.float64))
    edges = depth_edge_mask(depth_grid[:, :width], valid, snapped[:, :width])
    edge_grid = np.concatenate([edges, edges[:, :1]], axis=1).reshape(-1)
    if faces.size:
        faces = faces[~edge_grid[faces].any(axis=1)]
    ys, xs = np.mgrid[0:height, 0 : width + 1]
    uvs = np.stack([(xs + 0.5) / float(width), (ys + 0.5) / float(height)], axis=-1).reshape(-1, 2)
    piece = Piece(
        points.astype(np.float32),
        faces.astype(np.int32),
        uvs.astype(np.float32),
        "pano",
    )
    stats = {
        "faces": int(faces.shape[0]),
        "faces_dropped": int(raw_faces.shape[0] - faces.shape[0]),
        "ground": ground,
        "planes": plane_stats,
        "snapped_idx": np.flatnonzero(snapped.reshape(-1)).astype(np.int64),
    }
    return piece, stats


def save_piece(path: Path, piece: Piece) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        verts=np.asarray(piece.verts, dtype=np.float32),
        faces=np.asarray(piece.faces, dtype=np.int32),
        uvs=np.asarray(piece.uvs, dtype=np.float32),
        tex=np.array(piece.tex),
    )


def load_ground(scene: str) -> GroundPlane | None:
    """Fitted floor plane from ``pano_mesh.json``, or None when geometry has no floor."""
    path = world_dir(scene) / "pano_mesh.json"
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    ground = data.get("ground") if isinstance(data, dict) else None
    if not isinstance(ground, dict):
        return None
    normal = np.asarray(ground.get("normal"), dtype=np.float64).reshape(-1)
    if normal.shape != (3,) or not np.isfinite(normal).all():
        return None
    length = float(np.linalg.norm(normal))
    if length < 1e-8:
        return None
    try:
        plane_d = float(ground["d"])
    except (KeyError, TypeError, ValueError):
        return None
    if not np.isfinite(plane_d):
        return None
    erp_dist = None
    depth_path = world_dir(scene) / "pano_depth.npz"
    if depth_path.is_file():
        with np.load(depth_path) as blob:
            erp_dist = np.asarray(blob["depth"], dtype=np.float32)
            if "valid" in blob.files:
                erp_dist = np.array(erp_dist, copy=True)
                erp_dist[~np.asarray(blob["valid"], dtype=bool)] = np.nan
    return GroundPlane(normal / length, plane_d, erp_dist)


def load_piece(path: Path) -> Piece:
    with np.load(path, allow_pickle=False) as data:
        tex = data["tex"]
        name = tex.item() if isinstance(tex, np.ndarray) else tex
        return Piece(
            np.asarray(data["verts"], dtype=np.float32),
            np.asarray(data["faces"], dtype=np.int32),
            np.asarray(data["uvs"], dtype=np.float32),
            str(name),
        )


def ensure_pano_mesh(scene: str, erp_rgb: np.ndarray, device: str) -> Piece:
    del erp_rgb
    from engine.depth.keys import depth_key
    from engine.depth.scene_depth import depth_at, ensure_scene_depth
    from engine.world.defaults import MESH_CANVAS

    folder = world_dir(scene)
    folder.mkdir(parents=True, exist_ok=True)
    mesh_path = folder / "pano_mesh.npz"
    json_path = folder / "pano_mesh.json"
    current = depth_key(scene)
    stored = None
    if json_path.is_file():
        try:
            stored = json.loads(json_path.read_text(encoding="utf-8")).get("depth_key")
        except (OSError, json.JSONDecodeError):
            stored = None
    if mesh_path.is_file() and current and stored == current:
        return load_piece(mesh_path)
    scene_depth = ensure_scene_depth(scene, device)
    depth, valid = depth_at(scene_depth, MESH_CANVAS, fill_sky=True)
    normals = None
    if scene_depth.normals is not None:
        normals = cv2.resize(scene_depth.normals, MESH_CANVAS, interpolation=cv2.INTER_NEAREST)
    np.savez_compressed(
        folder / "pano_depth.npz",
        depth=np.asarray(depth, dtype=np.float32),
        valid=np.asarray(valid, dtype=bool),
    )
    piece, stats = build_pano_mesh(depth, valid, normals=normals)
    save_piece(mesh_path, piece)
    public = {key: value for key, value in stats.items() if key != "snapped_idx"}
    public["depth_key"] = scene_depth.key
    json_path.write_text(json.dumps(public) + "\n", encoding="utf-8")
    _log(
        f"STATUS // world pano mesh faces={public['faces']} dropped={public['faces_dropped']} "
        f"ground={public['ground']}"
    )
    return piece


def invalidate_pano_mesh(scene: str) -> None:
    folder = world_dir(scene)
    for name in ("pano_depth.npz", "pano_mesh.npz", "pano_mesh.json"):
        path = folder / name
        if path.is_file():
            path.unlink()
