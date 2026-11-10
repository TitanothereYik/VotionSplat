"""Front view of a rail camera.

OpenCV camera: +X right, +Y down, +Z forward. The rail pose is the front view.
"""

from __future__ import annotations

VIEWS = ("front",)


def diffusion_num_frames(keyframes: int) -> int:
    """Wan temporal factor 4. ``k`` keyframes need ``4*(k-1)+1`` video frames."""
    k = int(keyframes)
    if k < 2:
        raise ValueError(f"need at least 2 keyframes, got {k}")
    return 4 * (k - 1) + 1


def rail_view_rel(rail_index: int, view: str = "front") -> str:
    name = str(view or "front").strip().lower()
    if name not in VIEWS:
        raise ValueError(f"unknown view {view}")
    return f"worldstereo/rails/rail_{int(rail_index):02d}/{name}"
