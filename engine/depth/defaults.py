"""Scene-depth numbers."""

from __future__ import annotations

from engine.job import REPO_ROOT

# Bake-off on alpine (2026-11-11): MoGe-2 had the flattest ground and facades, MIT weights.
DEPTH_BACKEND = "moge2"
BACKENDS = ("moge1", "moge2", "pager", "dap")
DEPTH_CANVAS = (4096, 2048)

MOGE1_LEVEL = 9
MOGE1_MERGE = (1920, 960)
MOGE1_CANVAS = (2048, 1024)

# Alpine's MoGe v1 camera sat about 0.20 above the floor. Every backend is scaled to that.
SCALE_CAMERA_HEIGHT = 0.20
SCALE_MEDIAN_FALLBACK = 1.0

VENV_ROOT = REPO_ROOT / ".venvs"
RUNNER_TIMEOUT_S = 1800

# With no bake-off reference, a flipped or rolled depth must beat identity by this much
# Spearman correlation between depth edges and panorama edges before it is applied.
CONVENTION_EDGE_MARGIN = 0.05
