"""Write a camera JSON for the reference renderer and the Unreal capture script."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ue_camera import A, look_at_rotation  # noqa: E402


def _xyz(path: Path) -> np.ndarray:
    names: list[str] = []
    count = 0
    with path.open("rb") as handle:
        while True:
            line = handle.readline().decode("ascii", errors="replace").strip()
            if line.startswith("element vertex"):
                count = int(line.split()[-1])
            elif line.startswith("property float "):
                names.append(line.split()[-1])
            elif line == "end_header":
                offset = handle.tell()
                break
            elif not line:
                raise ValueError(f"{path} has no end_header")
    data = np.memmap(path, dtype="<f4", mode="r", offset=offset, shape=(count, len(names)))
    index = {name: i for i, name in enumerate(names)}
    xyz = np.stack([np.asarray(data[:, index[k]], dtype=np.float64) for k in ("x", "y", "z")], axis=1)
    return (A @ (xyz * 100.0).T).T


def origin_cameras(width: int, height: int, fov: float) -> list[dict]:
    shots = [("yaw0", 0.0, 0.0), ("yaw90", 0.0, 90.0), ("yaw180", 0.0, 180.0),
             ("yaw270", 0.0, 270.0), ("yaw45_pitch-20", -20.0, 45.0)]
    return [_cam(name, (0.0, 0.0, 0.0), pitch, yaw, 0.0, fov, width, height) for name, pitch, yaw in shots]


def orbit_cameras(xyz_ue: np.ndarray, width: int, height: int, fov: float) -> list[dict]:
    center = np.median(xyz_ue, axis=0)
    radius = float(np.quantile(np.linalg.norm(xyz_ue - center, axis=1), 0.80))
    radius = max(radius, 50.0)
    cameras = []
    for angle in (0.0, 90.0, 180.0, 270.0):
        rad = np.radians(angle)
        loc = center + 1.5 * radius * np.array([np.cos(rad), np.sin(rad), 0.3])
        pitch, yaw, roll = look_at_rotation(loc, center)
        cameras.append(_cam(f"orbit{int(angle)}", loc, pitch, yaw, roll, fov, width, height))
    return cameras


def _cam(name, loc, pitch, yaw, roll, fov, width, height) -> dict:
    return {
        "name": name,
        "loc_cm": [float(v) for v in loc],
        "pitch": float(pitch),
        "yaw": float(yaw),
        "roll": float(roll),
        "fov_h": float(fov),
        "W": int(width),
        "H": int(height),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ply", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mode", choices=("origin", "orbit"), required=True)
    parser.add_argument("--W", type=int, default=1920)
    parser.add_argument("--H", type=int, default=1080)
    parser.add_argument("--fov", type=float, default=90.0)
    args = parser.parse_args()
    cameras = origin_cameras(args.W, args.H, args.fov) if args.mode == "origin" else orbit_cameras(
        _xyz(args.ply), args.W, args.H, args.fov)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(cameras, indent=2), encoding="utf-8")
    print(f"wrote {args.out} ({len(cameras)} cameras)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
