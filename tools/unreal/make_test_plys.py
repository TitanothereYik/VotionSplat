"""Synthetic Votion-tagged PLYs. Each file isolates one part of the raster rule."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from engine.splat import defaults as D  # noqa: E402
from engine.splat.ply_export import SH_C0, write_splat_ply  # noqa: E402

OUT = ROOT / "unreal" / "test_assets"
C0 = SH_C0


def _logit(opacity: float) -> float:
    o = float(np.clip(opacity, 1e-6, 1.0 - 1e-6))
    return float(np.log(o / (1.0 - o)))


def _pack(xyz, scale, rot, rgb, opacity, sh_rest=None) -> dict:
    xyz = np.asarray(xyz, dtype=np.float32)
    n = xyz.shape[0]
    scale = np.asarray(scale, dtype=np.float32)
    if scale.ndim == 1:
        scale = np.tile(scale, (n, 1))
    rot = np.asarray(rot, dtype=np.float32)
    if rot.ndim == 1:
        rot = np.tile(rot, (n, 1))
    rgb = np.asarray(rgb, dtype=np.float32)
    if rgb.ndim == 1:
        rgb = np.tile(rgb, (n, 1))
    if sh_rest is None:
        sh_rest = np.zeros((0, 3, n), dtype=np.float32)
    opa = np.asarray(opacity, dtype=np.float32).reshape(1, n)
    return {
        "xyz": xyz.T,
        "scale": np.log(np.clip(scale, 1e-8, None)).T.astype(np.float32),
        "rot": rot.T,
        "sh_0": ((rgb - 0.5) / C0).T[None].astype(np.float32),
        "sh_rest": np.asarray(sh_rest, dtype=np.float32),
        "opacity": opa,
    }


def _write(name: str, arrays: dict, raster: str, near: float, degree: int) -> None:
    path, count = write_splat_ply(
        OUT / f"{name}.ply", arrays, color_space=D.COLOR_SPACE_VOTION,
        votion_raster=raster, votion_near_plane=near, votion_sh_degree=degree,
    )
    print(f"{path.name} {count}")


def _quat_z(degrees: float) -> np.ndarray:
    half = np.radians(degrees) * 0.5
    return np.array([np.cos(half), 0.0, 0.0, np.sin(half)], dtype=np.float32)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    raster, near = "gsplat_antialiased", 0.01
    _write("T1_center", _pack([[0, 0, 5]], [0.05, 0.05, 0.05], [1, 0, 0, 0], [0.9, 0.2, 0.2], [_logit(0.9)]),
           raster, near, 0)
    _write("T2_offaxis", _pack([[2, -1, 5]], [0.05, 0.05, 0.05], [1, 0, 0, 0], [0.2, 0.8, 0.3], [_logit(0.9)]),
           raster, near, 0)

    angles = np.arange(24) * 7.5
    xs = (np.arange(6) - 2.5) * 0.5
    ys = (np.arange(4) - 1.5) * 0.5
    grid = np.array([[x, y, 6.0] for y in ys for x in xs], dtype=np.float32)
    _write("T3_aniso", _pack(
        grid, np.tile([0.2, 0.004, 0.004], (24, 1)), np.stack([_quat_z(a) for a in angles]),
        np.tile([0.85, 0.75, 0.55], (24, 1)), np.full(24, _logit(0.9)),
    ), raster, near, 0)

    gx, gy = np.meshgrid(np.linspace(-2, 2, 40), np.linspace(-1, 1, 20))
    scales = np.geomspace(0.0003, 0.006, gx.size)
    t4_scale = np.stack([scales, scales, scales * 0.2], axis=1)
    _write("T4_subpixel", _pack(
        np.stack([gx.ravel(), gy.ravel(), np.full(gx.size, 8.0)], axis=1),
        t4_scale, [1, 0, 0, 0], [0.7, 0.75, 0.8], np.full(gx.size, _logit(0.95)),
    ), raster, near, 0)
    _write("T4_subpixel_litegs", _pack(
        np.stack([gx.ravel(), gy.ravel(), np.full(gx.size, 8.0)], axis=1),
        t4_scale, [1, 0, 0, 0], [0.7, 0.75, 0.8], np.full(gx.size, _logit(0.95)),
    ), "litegs_classic", 0.2, 0)

    discs = np.array([[-0.2, 0, 4], [0, 0, 5], [0.2, 0, 6]], dtype=np.float32)
    _write("T5_stack", _pack(
        discs, np.tile([0.4, 0.4, 0.01], (3, 1)), [1, 0, 0, 0],
        [[0.9, 0.1, 0.1], [0.1, 0.9, 0.1], [0.1, 0.1, 0.9]], np.full(3, _logit(0.6)),
    ), raster, near, 0)

    rng = np.random.default_rng(0)
    direction = rng.normal(size=(2000, 3))
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    means = direction + np.array([0.0, 0.0, 6.0])
    rest = (rng.normal(size=(15, 3, 2000)) * 0.3).astype(np.float32)
    _write("T6_sh3", _pack(means, np.full((2000, 3), 0.03), [1, 0, 0, 0],
                           np.full((2000, 3), 0.5), np.full(2000, _logit(0.8)), rest), raster, near, 3)
    _write("T6_sh3_litegs", _pack(means, np.full((2000, 3), 0.03), [1, 0, 0, 0],
                                  np.full((2000, 3), 0.5), np.full(2000, _logit(0.8)), rest),
           "litegs_classic", 0.2, 3)

    zs = np.linspace(0.002, 0.05, 50)
    _write("T7_near", _pack(
        np.stack([np.linspace(-0.5, 0.5, 50), np.zeros(50), zs], axis=1),
        np.full((50, 3), 0.004), [1, 0, 0, 0], [0.9, 0.4, 0.2], np.full(50, _logit(0.95)),
    ), raster, near, 0)

    colours = np.array([[(i * 40 % 255) / 255, (i * 85 % 255) / 255, (i * 130 % 255) / 255] for i in range(8)])
    _write("T8_termination", _pack(
        np.stack([np.zeros(8), np.zeros(8), 3.0 + 0.1 * np.arange(8)], axis=1),
        np.tile([0.35, 0.35, 0.005], (8, 1)), [1, 0, 0, 0], colours, np.full(8, _logit(0.999)),
    ), raster, near, 0)
    _write("T9_big", _pack([[0, 0, 2]], [3, 3, 3], [1, 0, 0, 0], [0.4, 0.6, 0.9], [_logit(0.5)]),
           raster, near, 0)

    cameras = [
        {"name": "main", "loc_cm": [0, 0, 0], "pitch": 0, "yaw": -90, "roll": 0, "fov_h": 90, "W": 1920, "H": 1080},
        {"name": "side", "loc_cm": [0, 0, 0], "pitch": 0, "yaw": -60, "roll": 0, "fov_h": 90, "W": 1920, "H": 1080},
    ]
    cam_path = ROOT / "unreal" / "validation" / "jobs" / "tests_cams.json"
    cam_path.parent.mkdir(parents=True, exist_ok=True)
    cam_path.write_text(json.dumps(cameras, indent=2), encoding="utf-8")
    print(cam_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
