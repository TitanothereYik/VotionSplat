"""HiRes images are the 8K panorama where that mesh is visible, then this
frame's own WorldStereo pixels in the holes.

The shared layer is a guide-sized triangle per lifted pixel. It is there so
the next generated chunk sees the same wall. Drawn into HiRes it becomes a
cloud of triangles, so the final frame does not use it.

``hires/source.txt`` says which set Reconstruct uses. ``world`` is this
composite, stored as ``hires/NNN.png``. ``frame`` is the WorldStereo
keyframes upscaled with no panorama, stored as ``hires/frame/NNN.png``.
Either set can stay on disk. The stamp picks the active one.
Older folders were a pinhole reprojection and are not complete.
The reprojection helpers stay for the z-buffer check.
"""

from __future__ import annotations

from pathlib import Path

import json

import cv2
import numpy as np

from engine.worldstereo import defaults as D

SCALE = 2
HIRES_H = D.SIZE_H * SCALE
HIRES_W = D.SIZE_W * SCALE
MOGE_LEVEL = 6
DEPTH_LONG_SIDE = 4096
ULTRASHARP_FILE = "4x-UltraSharp.pth"
HIRES_COMPOSITE = "world"
HIRES_FRAME = "frame"
HIRES_SOURCES = (HIRES_COMPOSITE, HIRES_FRAME)
HIRES_SOURCE = HIRES_COMPOSITE


def scaled_k(K: np.ndarray, width: int = HIRES_W, height: int = HIRES_H) -> np.ndarray:
    out = np.array(K, dtype=np.float64, copy=True)
    out[0, 0] *= width / float(D.SIZE_W)
    out[0, 2] *= width / float(D.SIZE_W)
    out[1, 1] *= height / float(D.SIZE_H)
    out[1, 2] *= height / float(D.SIZE_H)
    return out


def depth_cloud(depth: np.ndarray, mask: np.ndarray, pano: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """OpenCV points and 8K colours. Invalid MoGe pixels are dropped."""
    depth = np.asarray(depth, dtype=np.float32)
    valid = np.asarray(mask, dtype=bool) & np.isfinite(depth) & (depth > 0)
    height, width = depth.shape
    step = 1
    while (height // step) * (width // step) > 2_500_000:
        step *= 2
    if step > 1:
        depth = depth[::step, ::step]
        valid = valid[::step, ::step]
        height, width = depth.shape
    ys, xs = np.nonzero(valid)
    if xs.size == 0:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint8)
    # Pixel centres on the depth grid, mapped back onto the full ERP.
    full_h, full_w = int(pano.shape[0]), int(pano.shape[1])
    uu = (xs.astype(np.float64) + 0.5) / width
    vv = (ys.astype(np.float64) + 0.5) / height
    lon = uu * (2.0 * np.pi) - np.pi
    lat = (np.pi / 2.0) - vv * np.pi
    cos_lat = np.cos(lat)
    direc = np.stack([cos_lat * np.sin(lon), -np.sin(lat), cos_lat * np.cos(lon)], axis=1)
    points = (direc * depth[ys, xs, None]).astype(np.float32)
    pu = np.clip(np.rint(uu * full_w - 0.5), 0, full_w - 1).astype(np.int64)
    pv = np.clip(np.rint(vv * full_h - 0.5), 0, full_h - 1).astype(np.int64)
    colors = np.asarray(pano)[pv, pu].astype(np.uint8)
    return points, colors


def reproject(
    points: np.ndarray,
    colors: np.ndarray,
    w2c: np.ndarray,
    K: np.ndarray,
    height: int = HIRES_H,
    width: int = HIRES_W,
) -> tuple[np.ndarray, np.ndarray]:
    """Nearest-point splat. Returns RGB and a bool gate (True = 8K pixel)."""
    rgb = np.zeros((height, width, 3), np.uint8)
    gate = np.zeros((height, width), dtype=bool)
    if points.shape[0] == 0:
        return rgb, gate
    pose = np.asarray(w2c, dtype=np.float64)
    cam = np.asarray(points, dtype=np.float64) @ pose[:3, :3].T + pose[:3, 3]
    z = cam[:, 2]
    keep = np.isfinite(z) & (z > 1e-3)
    if not np.any(keep):
        return rgb, gate
    cam = cam[keep]
    z = z[keep]
    cols = np.asarray(colors)[keep]
    u = K[0, 0] * cam[:, 0] / z + K[0, 2]
    v = K[1, 1] * cam[:, 1] / z + K[1, 2]
    ui = np.rint(u).astype(np.int64)
    vi = np.rint(v).astype(np.int64)
    inside = (ui >= 0) & (ui < width) & (vi >= 0) & (vi < height) & np.isfinite(u) & np.isfinite(v)
    if not np.any(inside):
        return rgb, gate
    ui, vi, z, cols = ui[inside], vi[inside], z[inside], cols[inside]
    order = np.argsort(-z, kind="stable")
    ui, vi, cols = ui[order], vi[order], cols[order]
    # Far points are written first. A nearer point overwrites the same pixel.
    flat = vi * width + ui
    rgb.reshape(-1, 3)[flat] = cols
    gate.reshape(-1)[flat] = True
    return rgb, gate


def composite(pano_rgb: np.ndarray, gate: np.ndarray, fill_rgb: np.ndarray) -> np.ndarray:
    out = np.array(fill_rgb, copy=True)
    out[gate] = pano_rgb[gate]
    return out


def composite_world(
    pano_rgb: np.ndarray,
    pano_empty: np.ndarray,
    layer_rgb: np.ndarray | None,
    layer_hit: np.ndarray | None,
    own: np.ndarray,
    feather_px: int,
    gate_pano: int,
    gate_layer: int,
    gate_own: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Panorama where its mesh hit, then the layer, then this frame.

    ``layer_*`` are already at the panorama resolution. A missing layer leaves
    the frame in every hole.
    """
    background = np.array(own, copy=True)
    hit = None
    if layer_rgb is not None and layer_hit is not None:
        hit = np.asarray(layer_hit, dtype=bool)
        background[hit] = np.asarray(layer_rgb)[hit]
    valid = (~np.asarray(pano_empty, dtype=bool)).astype(np.uint8)
    dist = cv2.distanceTransform(valid, cv2.DIST_L2, 3)
    weight = np.clip(dist / float(feather_px), 0.0, 1.0)
    out = np.round(
        weight[..., None] * np.asarray(pano_rgb, dtype=np.float32)
        + (1.0 - weight[..., None]) * background.astype(np.float32)
    ).astype(np.uint8)
    use = weight >= 0.5
    gate = np.full(out.shape[:2], int(gate_own), dtype=np.uint8)
    if hit is not None:
        gate[hit & ~use] = int(gate_layer)
    gate[use] = int(gate_pano)
    return out, gate


def choice_file(scene_root: Path) -> Path:
    return Path(scene_root) / "worldstereo" / "hires_choice.txt"


def read_choice(scene_root: Path) -> str:
    """Path selected on Generate. Empty when the scene has never chosen."""
    path = choice_file(scene_root)
    if not path.is_file():
        return ""
    text = path.read_text(encoding="utf-8").strip()
    return text if text in HIRES_SOURCES else ""


def write_choice(scene_root: Path, source: str) -> None:
    if source not in HIRES_SOURCES:
        raise ValueError(f"unknown HiRes choice {source}")
    path = choice_file(scene_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source + "\n", encoding="utf-8")


def active_choice(scene_root: Path) -> str:
    """Generate-tab path. A scene that has never chosen follows its rail stamps."""
    saved = read_choice(scene_root)
    if saved:
        return saved
    rails = Path(scene_root) / "worldstereo" / "rails"
    stamps = []
    if rails.is_dir():
        for folder in sorted(rails.glob("rail_*/front")):
            stamp = hires_stamp(folder)
            if stamp:
                stamps.append(stamp)
    choice = HIRES_FRAME if stamps and all(item == HIRES_FRAME for item in stamps) else HIRES_COMPOSITE
    write_choice(scene_root, choice)
    return choice


def choice_name(source: str) -> str:
    return "WorldStereo" if source == HIRES_FRAME else "Composite"


def dataset_images(scene_root: Path) -> str:
    """``frame`` or ``composite`` from the last Reconstruct. Empty when unknown."""
    path = Path(scene_root) / "worldstereo" / "dataset" / "sfm.json"
    if not path.is_file():
        return ""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return ""
    images = str(data.get("images") or "")
    return images if images in ("frame", "composite") else ""


def choice_matches_dataset(choice: str, images: str) -> bool:
    if choice == HIRES_FRAME:
        return images == "frame"
    if choice == HIRES_COMPOSITE:
        return images == "composite"
    return False


def hires_dir_for(folder: Path, source: str) -> Path | None:
    """Image folder for a choice, when that choice has frames."""
    if not hires_frames(folder, source):
        return None
    if source == HIRES_FRAME:
        return folder / "hires" / HIRES_FRAME
    if source == HIRES_COMPOSITE:
        return folder / "hires"
    return None


def hires_stamp(folder: Path) -> str:
    """Active HiRes set. Empty when the folder is not one of the two choices."""
    stamp = folder / "hires" / "source.txt"
    if not stamp.is_file():
        return ""
    text = stamp.read_text(encoding="utf-8").strip()
    return text if text in HIRES_SOURCES else ""


def hires_frames(folder: Path, source: str) -> list[Path]:
    """Frames for one choice. Composite stays in ``hires/``; the upscale is ``hires/frame/``."""
    if source == HIRES_FRAME:
        root = folder / "hires" / HIRES_FRAME
    elif source == HIRES_COMPOSITE:
        root = folder / "hires"
    else:
        return []
    if not root.is_dir():
        return []
    return sorted(root.glob("[0-9][0-9][0-9].png"))


def hires_ready(folder: Path, source: str, nframe: int) -> bool:
    return len(hires_frames(folder, source)) >= int(nframe)


def active_hires_dir(folder: Path) -> Path | None:
    """Directory the preview reads. None when the active choice has no frames."""
    source = hires_stamp(folder)
    if not source or not hires_frames(folder, source):
        return None
    if source == HIRES_FRAME:
        return folder / "hires" / HIRES_FRAME
    return folder / "hires"


def use_hires(folder: Path, source: str) -> bool:
    """Point Reconstruct and the preview at ``source`` when those frames exist."""
    if source not in HIRES_SOURCES or not hires_frames(folder, source):
        return False
    hires = folder / "hires"
    hires.mkdir(parents=True, exist_ok=True)
    (hires / "source.txt").write_text(source + "\n", encoding="utf-8")
    return True


def hires_complete(folder: Path, nframe: int, source: str | None = None) -> bool:
    """The active set is complete. Pass ``source`` to test that choice even if it is not active."""
    if source is None:
        source = hires_stamp(folder)
        if not source:
            return False
    return hires_ready(folder, source, nframe)


def prepare_depth_image(pano: np.ndarray, long_side: int = DEPTH_LONG_SIDE) -> np.ndarray:
    height, width = pano.shape[:2]
    long_side = int(long_side)
    if max(height, width) <= long_side:
        return pano
    scale = long_side / float(max(height, width))
    return cv2.resize(pano, (max(2, int(round(width * scale))), max(2, int(round(height * scale)))), interpolation=cv2.INTER_AREA)
