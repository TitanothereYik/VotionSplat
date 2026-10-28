"""2:1 ERP check. Tolerance locked in Phase 1 Q&A."""

from __future__ import annotations

from engine.pano.defaults import ASPECT_2TO1_TOLERANCE


def is_2to1(width: int, height: int, tolerance: float = ASPECT_2TO1_TOLERANCE) -> bool:
    if height <= 0 or width <= 0:
        return False
    return abs((width / height) - 2.0) <= tolerance


def aspect_error(width: int, height: int) -> str:
    ratio = (width / height) if height else 0.0
    return (
        f"reject non-2:1 ({width}x{height}, width/height={ratio:.4f}; "
        f"need |width/height - 2| <= {ASPECT_2TO1_TOLERANCE})"
    )
