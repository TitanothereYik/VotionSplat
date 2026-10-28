"""Anchor / look-at parsers copied from SplatKit nodes/camera_plot.py at pin f59de25.

No ComfyUI imports. Keep behavior identical to ``_camplot_parse_anchors_ext``.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np


def parse_anchors_ext(text: str) -> tuple[np.ndarray, np.ndarray]:
    """Parse the anchors widget into positions and optional per-anchor look targets.

    Each line / JSON row is 3 numbers ``x, y, z`` or 6 ``x, y, z, tx, ty, tz``.
    Blank lines and ``#`` comments ignored. Raises ValueError if < 2 points.
    """
    raw = (text or "").strip()
    if not raw:
        raise ValueError(
            "anchors is empty -- give at least 2 points the camera "
            "should fly through (e.g. one 'x,y,z' per line)."
        )
    rows: list[list[float]] | None = None
    try:
        data = json.loads(raw)
        rows = [[float(v) for v in row] for row in data]
    except Exception:
        rows = None
    if rows is None:
        rows = []
        for line_raw in raw.splitlines():
            line = line_raw.split("#", 1)[0].strip()
            if not line:
                continue
            parts = [p for p in line.replace(",", " ").split() if p != ""]
            if len(parts) not in (3, 6):
                raise ValueError(
                    f"could not parse anchor line {line_raw!r}: expected 3 numbers "
                    f"'x,y,z' or 6 'x,y,z,tx,ty,tz', got {len(parts)}."
                )
            rows.append([float(p) for p in parts])

    n = len(rows)
    if n < 2:
        raise ValueError(
            f"need at least 2 anchor points to define a fly-through path; got {n}."
        )
    positions = np.empty((n, 3), dtype=np.float64)
    targets = np.full((n, 3), np.nan, dtype=np.float64)
    for i, row in enumerate(rows):
        if len(row) not in (3, 6):
            raise ValueError(f"anchor {i} must have 3 or 6 numbers; got {len(row)}.")
        positions[i] = row[:3]
        if len(row) == 6:
            targets[i] = row[3:6]
    positions[0] = 0.0
    return positions, targets


def parse_point(text: str, default: tuple[float, float, float] = (0.0, 0.0, 1.0)) -> np.ndarray:
    """Parse look_at_target: a single 'x, y, z'. Blank/unparsable -> default."""
    t = (text or "").strip()
    if t:
        parts = [p for p in t.replace(",", " ").split() if p != ""]
        if len(parts) == 3:
            try:
                return np.array([float(p) for p in parts], dtype=np.float64)
            except ValueError:
                pass
    return np.array(default, dtype=np.float64)


def format_anchors_xyz(positions: Any) -> str:
    pts = np.asarray(positions, dtype=np.float64)
    lines = []
    for row in pts:
        lines.append(f"{row[0]:.3f}, {row[1]:.3f}, {row[2]:.3f}")
    return "\n".join(lines)


def _fill_nan_targets(positions: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Same rule as SplatKit ``_camplot_fill_targets`` (next-segment fallback)."""
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


def format_anchors(positions: Any, targets: Any | None = None, *, six: bool = False) -> str:
    """xyz lines, or 6-number ``x,y,z,tx,ty,tz`` when ``six`` (per_point_look)."""
    pts = np.asarray(positions, dtype=np.float64)
    if not six:
        return format_anchors_xyz(pts)
    if targets is None:
        tg = np.full(pts.shape, np.nan, dtype=np.float64)
    else:
        tg = np.asarray(targets, dtype=np.float64)
        if tg.shape != pts.shape:
            tg = np.full(pts.shape, np.nan, dtype=np.float64)
    filled = _fill_nan_targets(pts, tg)
    lines = []
    for p, t in zip(pts, filled):
        lines.append(
            f"{p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}, {t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}"
        )
    return "\n".join(lines)


def format_point(xyz: Any) -> str:
    row = np.asarray(xyz, dtype=np.float64).reshape(3)
    return f"{row[0]:.3f}, {row[1]:.3f}, {row[2]:.3f}"


def self_test() -> int:
    xyz = "0,0,0\n0.2,0.2,0.5"
    pos, tgt = parse_anchors_ext(xyz)
    if pos.shape != (2, 3) or not np.isnan(tgt).all():
        print("FAIL 3-number parse", pos, tgt)
        return 1
    if not np.allclose(pos[0], 0.0):
        print("FAIL star locked to origin")
        return 1
    six = format_anchors(pos, None, six=True)
    pos2, tgt2 = parse_anchors_ext(six)
    if pos2.shape != (2, 3) or tgt2.shape != (2, 3) or not np.all(np.isfinite(tgt2)):
        print("FAIL 6-number roundtrip", six, tgt2)
        return 1
    filled = _fill_nan_targets(pos, np.full_like(pos, np.nan))
    if not np.allclose(filled[0], pos[0] + (pos[1] - pos[0])):
        print("FAIL fill first from next", filled)
        return 1
    print("anchors self-test: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(self_test())
