"""Numbers copied at the WorldStereo pin. Do not invent new ones.

Code pin: FuchengSu/WorldStereo ``5ba8dc3d37c48ecef99bbd603815ced513a4a431``
(``vendor/worldstereo.sha``).

Config pin: ``hanshanxue/WorldStereo`` snapshot
``ac2ad97ecb043fe80c2f19cd1898006becb9d66e``,
``worldstereo-memory-dmd/config.json``. The FP8 card says that file was
exported from commit ``2adb716``.
"""

from __future__ import annotations

# config.json at the snapshot above.
BASE_MODEL = "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"
NFRAME = 21
SIZE_H = 480
SIZE_W = 832
STEPS = 4
SEED = 1024
RENDER_RADIUS = 0.008
POINTS_PER_PIXEL = 8
BACKGROUND = (0, 0, 0)
SOBEL_THRESHOLD = 0.35
START_ELEVATION = 5.0
DMD_START_STEPS = 250
DMD_END_STEPS = 1000
GEN_SHIFT = 5.0
LATENT_COND_MODE = "first_frame_only"

# Votion rail length 81, every 4th frame, inclusive of frame 0 and frame 80.
RAIL_STRIDE = 4

# README default, not a config.json field.
ALIGN_NFRAME = 8

# Example camera.json at the code pin (view0/traj0 and pano_bank/cameras.json).
# fx = 240.1777, fy = 240, cx = 416, cy = 240 → frame is 832×480.
# Horizontal FoV = 2*atan((W/2)/fx) = 120°. Vertical FoV = 2*atan((H/2)/fy) = 90°.
FOV_H_DEG = 120.0
FOV_V_DEG = 90.0

# pano_bank/images/0000.png … 0026.png at the code pin.
PANO_BANK_VIEWS = 27

# pointcloud.point_rendering: mask is 1 where the z-buffer is -1 (no point).
# data_utils thresholds the mp4 at 0.5, so white in render_mask.mp4 is empty
# and black is a rendered point. Opposite of Votion's validity mask
# (white = known).
MASK_WHITE_IS_EMPTY = True

FP8_REPO = "apozz/WorldStereo-fp8"
FP8_FILE = "WorldStereo-memory-dmd-fp8_e4m3fn_scaled.safetensors"
# Content-Length of the resolve URL for that file (snapshot af612e18446099338f7a6e90e6755105ffac59e0).
FP8_BYTES = 20_356_086_900
CONFIG_REPO = "hanshanxue/WorldStereo"
CONFIG_SUBFOLDER = "worldstereo-memory-dmd"
DINO_REPO = "facebook/dinov2-base"
