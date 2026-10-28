"""Viewport camera math (numpy only, no torch). Shared by the trainer and viewport.

Conventions: COLMAP / OpenCV world-to-camera (``w2c``), camera +Z looks forward,
+Y down. The rails **Star** (first pose of ``rails/rail_00.json``) is the default
viewport pose (lock #43). Orbit = yaw / pitch about the Star, wheel dollies along
the look direction (zoom is not product-capped), RMB pans in the image plane.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class ViewState:
    """Latest camera request from Qt (``VIEW`` line) or the persisted JSON."""

    yaw: float = 0.0
    pitch: float = 0.0
    zoom: float = 1.0
    pan_x: float = 0.0
    pan_y: float = 0.0
    # WASD/QE flythrough (UE-editor style): persistent accumulators in "reach" units
    # (see orbit_w2c), driven by app.widgets.splat_view while LMB is held. Unlike
    # pan_x/pan_y (small, clamped drag offsets) these are meant to let the user travel
    # arbitrarily far from the Star/training-camera base pose.
    fly_fwd: float = 0.0  # W/S
    fly_right: float = 0.0  # D/A
    fly_up: float = 0.0  # E/Q
    width: int = 1280
    height: int = 720
    cam: int = -1  # >= 0: base pose is training camera #cam instead of the Star
    gt: bool = False  # side-by-side render | ground truth when cam >= 0
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict) -> "ViewState":
        vs = cls()
        try:
            vs.yaw = float(data.get("yaw", vs.yaw))
            vs.pitch = float(np.clip(float(data.get("pitch", vs.pitch)), -1.45, 1.45))
            zoom = float(data.get("zoom", vs.zoom))
            if not math.isfinite(zoom) or zoom <= 0.0:
                zoom = 1.0
            vs.zoom = float(min(max(zoom, 1e-6), 1e9))
            vs.pan_x = float(np.clip(float(data.get("pan_x", vs.pan_x)), -64.0, 64.0))
            vs.pan_y = float(np.clip(float(data.get("pan_y", vs.pan_y)), -64.0, 64.0))
            for _f in ("fly_fwd", "fly_right", "fly_up"):
                v = float(data.get(_f, getattr(vs, _f)))
                setattr(vs, _f, float(np.clip(v, -1.0e6, 1.0e6)) if math.isfinite(v) else 0.0)
            vs.width = int(max(64, min(int(data.get("w", data.get("width", vs.width))), 3840)))
            vs.height = int(max(36, min(int(data.get("h", data.get("height", vs.height))), 2160)))
            vs.cam = int(data.get("cam", -1))
            vs.gt = bool(data.get("gt", False))
        except (TypeError, ValueError):
            pass
        return vs

    def to_dict(self) -> dict:
        return {
            "yaw": self.yaw,
            "pitch": self.pitch,
            "zoom": self.zoom,
            "pan_x": self.pan_x,
            "pan_y": self.pan_y,
            "fly_fwd": self.fly_fwd,
            "fly_right": self.fly_right,
            "fly_up": self.fly_up,
            "w": self.width,
            "h": self.height,
            "cam": self.cam,
            "gt": self.gt,
        }

    def pose_key(self) -> tuple:
        return (
            round(self.yaw, 6),
            round(self.pitch, 6),
            round(self.zoom, 6),
            round(self.pan_x, 6),
            round(self.pan_y, 6),
            round(self.fly_fwd, 6),
            round(self.fly_right, 6),
            round(self.fly_up, 6),
            self.width,
            self.height,
            self.cam,
            self.gt,
        )


def look_at_w2c(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
    """OpenCV w2c: camera +Z looks at target, +Y down (same as COLMAP views)."""
    eye = np.asarray(eye, dtype=np.float64).reshape(3)
    target = np.asarray(target, dtype=np.float64).reshape(3)
    forward = target - eye
    n = float(np.linalg.norm(forward))
    forward = np.array([0.0, 0.0, 1.0]) if n < 1e-8 else forward / n
    world_down = np.array([0.0, 1.0, 0.0])
    right = np.cross(world_down, forward)
    rn = float(np.linalg.norm(right))
    if rn < 1e-6:
        world_down = np.array([1.0, 0.0, 0.0])
        right = np.cross(world_down, forward)
        rn = float(np.linalg.norm(right))
    right = right / max(rn, 1e-8)
    down = np.cross(forward, right)
    down = down / (float(np.linalg.norm(down)) + 1e-8)
    rot = np.stack([right, down, forward], axis=0)
    w2c = np.eye(4, dtype=np.float64)
    w2c[:3, :3] = rot
    w2c[:3, 3] = -rot @ eye
    return w2c


def rotate(v: np.ndarray, axis: np.ndarray, ang: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64).reshape(3)
    v = np.asarray(v, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(axis))
    if n < 1e-8 or abs(ang) < 1e-12:
        return v
    axis = axis / n
    c, s = math.cos(ang), math.sin(ang)
    return v * c + np.cross(axis, v) * s + axis * float(np.dot(axis, v)) * (1.0 - c)


def w2c_from_rail_json(data) -> np.ndarray | None:
    if isinstance(data, list) and data:
        m = np.asarray(data[0], dtype=np.float64)
        if m.shape == (4, 4):
            return m
    if isinstance(data, dict):
        raw = str(data.get("look_at") or "0, 0, 1")
        parts = [p for p in raw.replace(",", " ").split() if p]
        if len(parts) >= 3:
            try:
                tgt = np.array(
                    [float(parts[0]), -float(parts[1]), float(parts[2])],
                    dtype=np.float64,
                )
            except ValueError:
                return None
            return look_at_w2c(np.zeros(3, dtype=np.float64), tgt)
    return None


def star_w2c(scene_dir: Path, cam_center: np.ndarray | None) -> np.ndarray:
    """Eye at the rails Star (origin of every rail). Look along rail 0's first pose."""
    rails = Path(scene_dir) / "rails"
    if rails.is_dir():
        for path in sorted(rails.glob("rail_*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            pose = w2c_from_rail_json(data)
            if pose is not None:
                return pose
    eye = np.zeros(3, dtype=np.float64)
    if cam_center is not None:
        eye = np.asarray(cam_center, dtype=np.float64).reshape(3)
    return look_at_w2c(eye, eye + np.array([0.0, 0.0, 1.0], dtype=np.float64))


def orbit_w2c(
    base: np.ndarray,
    yaw: float,
    pitch: float,
    zoom: float,
    pan_x: float,
    pan_y: float,
    reach: float,
    fly_fwd: float = 0.0,
    fly_right: float = 0.0,
    fly_up: float = 0.0,
) -> np.ndarray:
    """yaw / pitch look around ``base``. zoom=1 stays there; wheel dollies along look.

    ``fly_fwd``/``fly_right``/``fly_up`` are the UE-editor-style WASD/QE flythrough
    accumulators (see ``ViewState``), in the same "reach" units as the wheel dolly and
    RMB pan below -- unlike ``pan_x``/``pan_y`` they are unclamped so a flythrough can
    cover the whole scene, not just a small drag's worth.
    """
    base = np.asarray(base, dtype=np.float64)
    if (
        abs(yaw) < 1e-8
        and abs(pitch) < 1e-8
        and abs(float(zoom) - 1.0) < 1e-8
        and abs(pan_x) < 1e-8
        and abs(pan_y) < 1e-8
        and abs(fly_fwd) < 1e-8
        and abs(fly_right) < 1e-8
        and abs(fly_up) < 1e-8
    ):
        return base
    r, t = base[:3, :3], base[:3, 3]
    eye0 = -r.T @ t
    look = r[2, :3]
    right = r[0, :3]
    ln = float(np.linalg.norm(look))
    look = look / max(ln, 1e-8)
    world_up = np.array([0.0, -1.0, 0.0], dtype=np.float64)  # editor +Y
    look = rotate(look, world_up, yaw)
    right = rotate(right, world_up, yaw)
    rn = float(np.linalg.norm(right))
    if rn < 1e-6:
        right = np.cross(world_up, look)
        rn = float(np.linalg.norm(right))
    right = right / max(rn, 1e-8)
    look = rotate(look, right, -pitch)
    look = look / max(float(np.linalg.norm(look)), 1e-8)
    reach = max(float(reach), 1.0)
    dolly = math.log(max(float(zoom), 1e-12)) * reach + float(fly_fwd) * reach
    eye = eye0 + look * dolly
    w2c0 = look_at_w2c(eye, eye + look)
    pan_scale = reach * 0.08
    eye = (
        eye
        + w2c0[0, :3] * (pan_x * pan_scale + float(fly_right) * reach)
        + w2c0[1, :3] * (pan_y * pan_scale - float(fly_up) * reach)
    )
    return look_at_w2c(eye, eye + look)


def read_view_json(path: Path) -> ViewState:
    if not Path(path).is_file():
        return ViewState()
    try:
        return ViewState.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
    except Exception:
        return ViewState()


def write_view_json(path: Path, state: ViewState) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(state.to_dict()), encoding="utf-8")
    try:
        tmp.replace(path)
    except OSError:
        path.write_text(json.dumps(state.to_dict()), encoding="utf-8")


def viewport_intrinsics(width: int, height: int, fov_scale: float = 0.5) -> np.ndarray:
    """Pinhole K for the free viewport: fx = fov_scale * width (~90 deg horizontal)."""
    width = int(width)
    height = int(height)
    fx = float(fov_scale) * float(width)
    return np.array(
        [[fx, 0.0, width * 0.5], [0.0, fx, height * 0.5], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def fit_training_intrinsics(
    K: np.ndarray,
    scaled_w: int,
    scaled_h: int,
    view_w: int,
    view_h: int,
    *,
    resolution_scale: float = 1.0,
    max_w: int = 1920,
    max_h: int = 1080,
) -> tuple[int, int, np.ndarray]:
    """Map a training camera onto the viewport without anamorphic stretch.

    ``K`` is COLMAP / full-resolution. ``scaled_w`` / ``scaled_h`` are the training
    image after ``resolution_scale``. Scaling fx by ``view_w / full_w`` and fy by
    ``view_h / full_h`` independently keeps both FOVs but squashes a square cube
    face into a wide plate (circles become ellipses). Fit the camera's aspect
    inside the plate and scale fx/fy by the same factor; Qt letterboxes the rest.
    """
    s = float(resolution_scale) if float(resolution_scale) > 0.0 else 1.0
    full_w = max(1.0, float(scaled_w) / s)
    full_h = max(1.0, float(scaled_h) / s)
    view_w = max(1, int(view_w))
    view_h = max(1, int(view_h))
    max_w = max(1, int(max_w))
    max_h = max(1, int(max_h))
    fit = min(view_w / full_w, view_h / full_h, max_w / full_w, max_h / full_h)
    if full_w >= full_h:
        rh = max(1, int(round(full_h * fit)))
        rw = max(1, int(round(rh * full_w / full_h)))
    else:
        rw = max(1, int(round(full_w * fit)))
        rh = max(1, int(round(rw * full_h / full_w)))
    if rw > max_w or rh > max_h:
        fit2 = min(max_w / rw, max_h / rh)
        rw = max(1, int(round(rw * fit2)))
        rh = max(1, int(round(rh * fit2)))
    ksc = rw / full_w
    kk = np.asarray(K, dtype=np.float64)
    return (
        rw,
        rh,
        np.array(
            [
                [float(kk[0, 0]) * ksc, 0.0, float(kk[0, 2]) * ksc],
                [0.0, float(kk[1, 1]) * ksc, float(kk[1, 2]) * ksc],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        ),
    )
