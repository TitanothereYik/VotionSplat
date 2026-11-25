"""Unreal camera (cm, degrees) to an OpenCV world-to-camera matrix in the PLY frame.

The actor is at the origin with identity rotation. UE axes map to PLY with
A = [[1, 0, 0], [0, 0, -1], [0, -1, 0]], so (x, y, z) -> (x, -z, -y).
"""

from __future__ import annotations

import math

import numpy as np

A = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, -1.0, 0.0]], dtype=np.float64)


def rot_rows(pitch: float, yaw: float, roll: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """UE camera axes in world cm. Angles are degrees."""
    pr, yr, rr = math.radians(pitch), math.radians(yaw), math.radians(roll)
    cp, sp = math.cos(pr), math.sin(pr)
    cy, sy = math.cos(yr), math.sin(yr)
    cr, sr = math.cos(rr), math.sin(rr)
    forward = np.array([cp * cy, cp * sy, sp], dtype=np.float64)
    right = np.array([sr * sp * cy - cr * sy, sr * sp * sy + cr * cy, -sr * cp], dtype=np.float64)
    up = np.array([-(cr * sp * cy + sr * sy), cy * sr - cr * sp * sy, cr * cp], dtype=np.float64)
    return forward, right, up


def ue_to_opencv(
    loc_cm,
    pitch: float,
    yaw: float,
    roll: float,
    fov_h_deg: float,
    width: int,
    height: int,
    actor_loc_cm=(0.0, 0.0, 0.0),
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(w2c 4x4, K 3x3)`` for a camera looking at a splat actor at the origin."""
    loc = np.asarray(loc_cm, dtype=np.float64) - np.asarray(actor_loc_cm, dtype=np.float64)
    forward, right, up = rot_rows(pitch, yaw, roll)
    c2w_r = np.column_stack([A @ right, A @ (-up), A @ forward])
    det = float(np.linalg.det(c2w_r))
    if det < 0.99:
        raise ValueError(f"camera basis determinant {det}, expected +1")
    c2w_t = A @ (loc / 100.0)
    w2c = np.eye(4, dtype=np.float64)
    w2c[:3, :3] = c2w_r.T
    w2c[:3, 3] = -c2w_r.T @ c2w_t
    fx = 0.5 * width / math.tan(math.radians(fov_h_deg) * 0.5)
    k = np.array([[fx, 0.0, width * 0.5], [0.0, fx, height * 0.5], [0.0, 0.0, 1.0]], dtype=np.float64)
    return w2c, k


def look_at_rotation(from_cm, to_cm) -> tuple[float, float, float]:
    delta = np.asarray(to_cm, dtype=np.float64) - np.asarray(from_cm, dtype=np.float64)
    dx, dy, dz = (float(v) for v in delta)
    yaw = math.degrees(math.atan2(dy, dx))
    pitch = math.degrees(math.atan2(dz, math.hypot(dx, dy)))
    return pitch, yaw, 0.0


def _self_test() -> None:
    forward, right, up = rot_rows(0.0, -90.0, 0.0)
    c2w = np.column_stack([A @ right, A @ (-up), A @ forward])
    if not np.allclose(c2w, np.eye(3), atol=1e-8):
        raise AssertionError(c2w)
    rng = np.random.default_rng(1)
    for _ in range(100):
        pitch, yaw, roll = rng.uniform(-80, 80, size=3)
        f, r, u = rot_rows(float(pitch), float(yaw), float(roll))
        basis = np.column_stack([A @ r, A @ (-u), A @ f])
        if abs(float(np.linalg.det(basis)) - 1.0) > 1e-6:
            raise AssertionError((pitch, yaw, roll, np.linalg.det(basis)))
    w2c, k = ue_to_opencv((0, 0, 0), 0, -90, 0, 90, 1920, 1080)
    point = np.array([0.0, 0.0, 5.0, 1.0])
    cam = w2c @ point
    pix = k @ (cam[:3] / cam[2])
    if abs(float(pix[0]) - 960) > 1e-4 or abs(float(pix[1]) - 540) > 1e-4:
        raise AssertionError(pix)
    print("ue_camera self-test OK")


if __name__ == "__main__":
    _self_test()
