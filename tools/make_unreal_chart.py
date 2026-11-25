"""Write a splat test chart of known display colours for measuring Unreal's colour chain.

Layout seen from the front: 4 rows of 8 greys ascending left to right, top row first, starting at
black in the top-left; then one row of colour patches. Each patch is a dense grid of opaque, flat,
identical Gaussians, so its centre shows the patch colour in any blend space. Run:
    .venv\\Scripts\\python.exe tools\\make_unreal_chart.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.splat import defaults as D  # noqa: E402
from engine.splat.ply_export import SH_C0, write_splat_ply  # noqa: E402

GREYS = (0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.23, 0.26,
         0.30, 0.34, 0.38, 0.42, 0.46, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0)
COLOURS = ((0.6, 0.1, 0.1), (0.1, 0.6, 0.1), (0.1, 0.1, 0.6), (0.9, 0.5, 0.2),
           (0.25, 0.08, 0.08), (0.08, 0.25, 0.08), (0.08, 0.08, 0.25), (0.35, 0.3, 0.28))
COLS = 8
PATCH = 0.4  # metres
GAP = 0.1
STEP = 0.02
OUT = ROOT / "docs" / "unreal" / "Votion_Plugin_Chart.ply"


def patches() -> list[tuple[float, float, float]]:
    return [(g, g, g) for g in GREYS] + list(COLOURS)


def build() -> dict:
    cells = patches()
    rows = len(cells) // COLS
    g = np.arange(STEP / 2, PATCH, STEP)
    gx, gy = np.meshgrid(g, g)
    xyz, rgb = [], []
    for i, colour in enumerate(cells):
        r, c = divmod(i, COLS)
        x0 = c * (PATCH + GAP) - COLS * (PATCH + GAP) / 2
        y0 = r * (PATCH + GAP) - rows * (PATCH + GAP) / 2
        pts = np.stack([x0 + gx.ravel(), y0 + gy.ravel(), np.zeros(gx.size)], 1)
        xyz.append(pts)
        rgb.append(np.tile(colour, (pts.shape[0], 1)))
    xyz = np.concatenate(xyz).astype(np.float32)
    rgb = np.concatenate(rgb).astype(np.float32)
    n = xyz.shape[0]
    scale = np.tile(np.log([STEP * 0.75, STEP * 0.75, 0.001]), (n, 1)).astype(np.float32)
    return {
        "xyz": xyz.T,
        "scale": scale.T,
        "rot": np.tile([1.0, 0.0, 0.0, 0.0], (n, 1)).astype(np.float32).T,
        "sh_0": ((rgb - 0.5) / SH_C0).T[None].astype(np.float32),
        "sh_rest": np.zeros((0, 3, n), dtype=np.float32),
        "opacity": np.full((1, n), np.log(0.995 / 0.005), dtype=np.float32),
    }


def main() -> int:
    path, n = write_splat_ply(
        OUT, build(),
        color_space=D.COLOR_SPACE_VOTION, votion_raster="gsplat_antialiased",
        votion_near_plane=0.01, votion_sh_degree=0,
    )
    print(f"wrote {path} ({n} Gaussians, {len(patches())} patches)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
