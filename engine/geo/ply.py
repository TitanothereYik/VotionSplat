"""Write a simple colored point cloud PLY (ASCII)."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def write_xyz_rgb_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> Path:
    pts = np.asarray(xyz, dtype=np.float32)
    cols = np.asarray(rgb, dtype=np.uint8)
    n = int(pts.shape[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write("ply\n")
        fh.write("format ascii 1.0\n")
        fh.write(f"element vertex {n}\n")
        fh.write("property float x\n")
        fh.write("property float y\n")
        fh.write("property float z\n")
        fh.write("property uchar red\n")
        fh.write("property uchar green\n")
        fh.write("property uchar blue\n")
        fh.write("end_header\n")
        for i in range(n):
            x, y, z = pts[i]
            r, g, b = cols[i]
            fh.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")
    return path
