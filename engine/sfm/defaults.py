"""P2 SphereSfM knobs copied from SplatKit SphereSfMDataset INPUT_TYPES at pin f59de25.

Reconstruct dummy has no matcher / stride widgets. Do not invent UI for these —
the worker uses the node defaults. DualRes graph node 84 is exhaustive + HiRes;
P2 is single-trajectory SphereSfMDataset (no HiRes).
"""

from __future__ import annotations

# SphereSfMDataset INPUT_TYPES optional widgets (nodes/spheresfm.py).
FRAME_STRIDE = 1
MAX_FRAMES = 0  # 0 = no cap
MATCHER_TYPE = "sequential"
FACE_SIZE = 0  # 0 = auto (~equirect_w/4)
MAX_NUM_FEATURES = 8192
PEAK_THRESHOLD = 0.0066
EDGE_THRESHOLD = 10.0
MAX_NUM_MATCHES = 32768
FILTER_MAX_REPROJ_ERROR = 4.0
FILTER_MIN_TRI_ANGLE = 1.5
INIT_MIN_TRI_ANGLE = 4.0
INIT_MIN_NUM_INLIERS = 30
INIT_MAX_FORWARD_MOTION = 1.0
IMAGE_ORDER = "camera_major"
INITIAL_PANO_MODE = "replace"
INITIAL_PANO_HIRES = True
REUSE_SOLVE = False

HELP_RECONSTRUCT = """RECONSTRUCT
Keep or ditch rails after WAN. Ditched rails are left out of SphereSfM.
Re-gen WAN on a hallucinated rail (edit the prompt on Generate, or change the path
on Geometry and Play, then Re-gen). Reconstruct uses kept WAN clips when present.
Confirm rails on Geometry first.
Training images are gnomonic 90° SIMPLE_PINHOLE cube faces (Votion remaps
sphere_cubic_reprojecer output so plaza lines stay straight). Do not un-bend the
sparse cloud to fix a bowed splat — Retrain after Reconstruct.
"""
