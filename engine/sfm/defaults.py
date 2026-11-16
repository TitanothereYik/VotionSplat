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
Keep or ditch rails. A ditched rail is left out of the known-pose dataset. The files stay on disk.
Re-gen resamples that rail's 21-frame front view and marks later rails stale. Joint is one sample, so re-gen there is off; use Generate again or Start over.
Confirm rails on Geometry first.
Known poses writes worldstereo/dataset from the active HiRes set and the poses WorldStereo already used. Composite also adds the panorama point cloud and six 90° faces of the 8K pano. WorldStereo upscale uses only those frames.
Train writes worldstereo/train. Leftover WAN files on disk are ignored.
"""
