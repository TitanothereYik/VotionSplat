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

# Joint mode. One sample covers every kept rail. Raise either number to grow the clip;
# joint_plan splits the sequence when the total exceeds JOINT_MAX_FRAMES_PER_GEN.
JOINT_FRAMES_PER_RAIL = 20
# Keyframes in one sample. The checkpoint was trained on 21. A 40-frame sample
# overran the 24 GB card (CUDA out of memory), so each rail is its own sample.
# The next rail starts on the previous sample's last frame and uses the shared bank.
JOINT_MAX_FRAMES_PER_GEN = 20

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

HELP_GENERATE = """GENERATE
WorldStereo 2.0 is the Quality generator. One process, one memory bank. Retrieval uses the HY-World far plane, eight times the MoGe depth median.
Joint is the default. Each kept rail gets 20 frames, and each rail is its own sample so the 24 GB card can hold it. The next rail starts on the previous sample's last frame and retrieves the shared memory bank. The camera goes out along one rail and back along the next, starting at the star.
Per rail is the other option. Each kept rail gets its own 21-frame front view. Re-gen selected rail restores memory from the previous rail, then marks later rails stale.
Generate resumes. Start over closes the open preview, then clears rails, memory, the depth cache, and the geometry cache after you confirm.
The keyframe chips show the selected rail, or the whole Joint sequence.
The guide video is a solid surface built from the 8K panorama's depth, so straight lines stay straight.
After each rail, the gaps WorldStereo filled are saved into a shared layer, and later rails see them in their guide.
HiRes has two choices. Composite paints the 8K panorama where the mesh is visible, and the WorldStereo frame fills the holes. WorldStereo upscales the generated frames and does not paint the panorama. HiRes all writes the selected choice. Reconstruct uses whichever set is active.
An empty prompt blocks Generate. Confirm rails on Geometry first.
"""

