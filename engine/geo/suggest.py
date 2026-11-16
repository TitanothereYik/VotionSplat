"""Suggest paths — SplatKit ``core.path_suggest.suggest_paths`` at pin f59de25.

Pure numpy. Safe to call from Qt (no torch). Cap is 8 in SplatKit; Votion fills
the four default slots (graph CAMERA 1–4).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from engine.job import REPO_ROOT


def suggest_paths(points: np.ndarray, count: int = 4) -> list[dict]:
    sk = REPO_ROOT / "vendor" / "splatkit"
    if str(sk) not in sys.path:
        sys.path.insert(0, str(sk))
    from core.path_suggest import suggest_paths as _suggest

    n = max(1, min(int(count), 8))
    return _suggest(np.asarray(points, dtype=np.float32), count=n)


def load_scene_cloud(scene_root: Path) -> np.ndarray:
    npz = scene_root / "moge" / "cloud.npz"
    if not npz.is_file():
        raise FileNotFoundError(str(npz))
    with np.load(npz) as data:
        pts = np.array(data["points"], copy=True)
    if pts.ndim != 2 or pts.shape[0] < 50:
        raise ValueError(
            f"scene reference cloud too sparse to plan against ({0 if pts.ndim != 2 else pts.shape[0]} points)"
        )
    return pts
