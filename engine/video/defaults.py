"""P3 WAN / HiRes / DualRes knobs copied from the SMPL graph.

Comments cite ``docs/260825_MICKMUMPITZ_3DGS-Dataset-Creator_1-0_SMPL.json`` node ids.
Widget *names* come from SplatKit / Comfy ``INPUT_TYPES`` at the vendor pin, not from
paraphrase. Do not swap LoRA order. Do not silently replace ``prompt.txt``.
"""

from __future__ import annotations

from engine.job import REPO_ROOT

# --- WAN clip (SplatKit_WanI2VMaskedConditioning, widgets [1440, 720, 81, 'black', False]) ---
WAN_WIDTH = 1440  # from SMPL json WanI2VMaskedConditioning
WAN_HEIGHT = 720  # from SMPL json WanI2VMaskedConditioning
WAN_LENGTH = 81  # from SMPL json WanI2VMaskedConditioning; must match Plot Camera length
HOLE_FILL = "black"  # from SMPL json WanI2VMaskedConditioning
INVERT_MASK = False  # from SMPL json WanI2VMaskedConditioning; white=valid/known
CLIP_VISION_ENCODE = "center"  # from SMPL json CLIP-Vision encode
CLIP_VISION_FILE = "clip_vision_h.safetensors"
VAE_FILE = "wan_2.1_vae.safetensors"
HOLE_UPSCALER = "4x-UltraSharp.pth"

# Model chain from JSON links 1→2→4. Do not swap.
UNET_FILE = "Wan2_1-I2V-14B-720P_fp8_e4m3fn.safetensors"  # from SMPL json UNETLoader id 4
PANO_LORA_FILE = "pano_video_gen_720p_comfy.safetensors"  # from SMPL json LoraLoaderModelOnly id 5
PANO_LORA_STRENGTH = 0.98  # from SMPL json node id 5
LIGHTX2V_FILE = "lightx2v_T2V_14B_cfg_step_distill_v2_lora_rank64_bf16.safetensors"  # id 7
LIGHTX2V_STRENGTH = 1  # from SMPL json node id 7
MODEL_SAMPLING_SHIFT = 5  # from SMPL json ModelSamplingSD3 id 8
UMT5_FILE = "umt5_xxl_fp8_e4m3fn_scaled.safetensors"  # CLIP type wan

# KSampler stored [0, 'fixed', 4, 1, 'euler', 'normal', 1]; steps linked to PrimitiveInt id 55.
WAN_SEED = 0  # from SMPL json KSampler
WAN_STEPS = 8  # from SMPL json PrimitiveInt id 55 STEPS (stored widget 4 is unused)
WAN_CFG = 1  # from SMPL json KSampler
WAN_SAMPLER = "euler"  # from SMPL json KSampler
WAN_SCHEDULER = "normal"  # from SMPL json KSampler
WAN_DENOISE = 1  # from SMPL json KSampler

# CreateVideo widgets [16, 8]. Names from ComfyUI CreateVideo at portable pin v0.32.0
# (graph properties.ver is 0.24.0, which only had fps; v0.32.0 adds bit_depth).
CREATE_VIDEO_FPS = 16  # from SMPL json CreateVideo widgets_values[0]
CREATE_VIDEO_BIT_DEPTH = 8  # from SMPL json CreateVideo widgets_values[1] / v0.32.0 bit_depth

# CLIP text: style suffix is optional; do not silently replace prompt.txt.
STYLE_SUFFIX = (  # from SMPL json CLIPTextEncode id 10
    "A high quality panoramic video of the scene, photorealistic, very clear, "
    "smooth camera movement, ultra-detailed, high resolution. Panorama."
)
NEGATIVE = (  # from SMPL json CLIPTextEncode id 11
    "The video is not of a high quality, it has a low resolution. Distortion. strange artifacts."
)

# --- HiRes Composite (SplatKit_HiResComposite node 96) ---
HIRES_SET_NAME = "hires_composite"  # from SMPL json node id 96
HIRES_TRAJ_INDEX = 0  # from SMPL json node id 96; rails 0..3, add-path 4
HIRES_OUTPUT_WIDTH = 8192  # from SMPL json node id 96 (add-path HiRes JSON is also 8192)
HIRES_BASE_MODE = "geometry"  # from SMPL json node id 96
HIRES_FRAMES = "0-80/2"  # from SMPL json node id 96 (graph default, 41 frames)
HIRES_FRAMES_PRESETS = (
    ("0-80/2", "0-80/2 (graph default, 41)"),
    ("0-15,16-/8", "0-15,16-/8 (SplatKit rec, 25)"),
    ("all", "all (81)"),
)
HIRES_PROXY_WIDTH = 2048  # from SMPL json node id 96
HIRES_GEOM_SCALE = 2  # from SMPL json node id 96
HIRES_MOGE_LEVEL = 6  # from SMPL json node id 96. Plot Camera stays 9. Do not unify.
HIRES_MERGE_LONG = 1440  # from SMPL json node id 96
HIRES_DEPTH_GRID = "geometry_res"  # from SMPL json node id 96
HIRES_MOGE_CKPT = "auto (download)"  # from SMPL json node id 96
HIRES_RHO_HI = 4  # from SMPL json node id 96
HIRES_TONE_WORK = 1024  # from SMPL json node id 96
HIRES_PREFETCH = True  # from SMPL json node id 96
HIRES_SAVE_VIDEO = False  # from SMPL json node id 96
HIRES_DEBUG_SAVE_GRAPH = "all"  # from SMPL json node id 96; product UI default is off
HIRES_DEBUG_SAVE_PRODUCT = "off"
HIRES_GATE_MODE = "hard_soft_edge"  # from SMPL json node id 96
HIRES_TONE_MODE = "rgb"  # per-channel gain of WAN fill toward the 8K photo (hue+exposure)
HIRES_AUTO_NAME = False  # from SMPL json node id 96; add-path node True
HIRES_SAVE_PROXIES = True  # from SMPL json node id 96
# SMPL node 96 copied gate_edge=0 / edge_erode=0 — a binary 8K|WAN cut. The 720p WAN
# clip has no seam because the sampler paints the whole frame; HiRes pastes the
# panorama on top, so 0 px is a clean mask edge (SplatKit: "no in-between blend
# band"). 16.0 OUTPUT-px Gaussian sigma (~±32 px join at 8192) is visible when the
# plate is fitted to a 1080p monitor; 4.0 (SplatKit's old shipped feather) vanishes
# at that scale. edge_erode pulls the photo back first so the fade is not a smear ring.
HIRES_GATE_EDGE = 16.0
HIRES_EDGE_ERODE = 2
HIRES_DISC_ERODE = 8  # from SMPL json node id 96
HIRES_BG_FILL = "wan"  # from SMPL json node id 96
HIRES_ADD_AUTO_NAME = True  # from SMPL json add-path HiRes (before node 90)
HIRES_ADD_TRAJ_INDEX = 4  # from SMPL json add-path HiRes traj_index

# --- Dual-res SphereSfM (node 84 widgets_values + INPUT_TYPES names at pin f59de25) ---
# Phase3.md prose said “hires width 8192”; at this pin that slot is max_num_features.
# There is no hires_width widget. 8K width is HiRes output_width + the PNG files.
DUALRES_HIRES_DIR = ""  # from SMPL json node id 84; product uses hires_manifest wires
DUALRES_OUTPUT_NAME = "my_scene_hires"  # from SMPL json node id 84; replace with Votion scene name
DUALRES_MATCHER = "exhaustive"  # from SMPL json node id 84
DUALRES_ON_SPLIT = "stop"  # from SMPL json node id 84
DUALRES_FACE_SIZE = 0  # from SMPL json node id 84
DUALRES_MAX_NUM_FEATURES = 8192  # from SMPL json node id 84 widgets_values[5]
DUALRES_PEAK_THRESHOLD = 0.0066  # from SMPL json node id 84
DUALRES_EDGE_THRESHOLD = 10  # from SMPL json node id 84
DUALRES_MAX_NUM_MATCHES = 32768  # from SMPL json node id 84
DUALRES_FILTER_MAX_REPROJ_ERROR = 4  # from SMPL json node id 84
DUALRES_FILTER_MIN_TRI_ANGLE = 1.5  # from SMPL json node id 84
DUALRES_INIT_MIN_TRI_ANGLE = 4  # from SMPL json node id 84
DUALRES_INIT_MIN_NUM_INLIERS = 30  # from SMPL json node id 84
DUALRES_INIT_MAX_FORWARD_MOTION = 1  # from SMPL json node id 84
DUALRES_IMAGE_ORDER = "camera_major"  # from SMPL json node id 84
DUALRES_HIRES_GLOB = "*.png"  # from SMPL json node id 84
DUALRES_FRAME_STRIDE = 1  # from SMPL json node id 84
DUALRES_MAX_FRAMES = 0  # from SMPL json node id 84
DUALRES_CLEANUP_HIRES_SOURCE = False  # from SMPL json node id 84

# Add-path DualRes node 90 widgets_values[7] = 4096 = max_num_features at pin f59de25.
ADD_DUALRES_MAX_NUM_FEATURES = 4096  # from SMPL json node id 90
ADD_DUALRES_MATCHER = "exhaustive"  # from SMPL json node id 90
ADD_DUALRES_FRAME_STRIDE = 1  # from SMPL json node id 90
ADD_DUALRES_MAX_FRAMES = 0  # from SMPL json node id 90
ADD_DUALRES_ADJUST_EXISTING = False  # from SMPL json node id 90
ADD_DUALRES_RETRIANGULATE = True  # from SMPL json node id 90
ADD_DUALRES_FACE_SIZE = 0  # from SMPL json node id 90

MASK_SPLIT_REF = REPO_ROOT / "docs" / "media" / "ref-hires-mask-split.png"

# Tokenizer + JSON configs only (not a second 14B pack). Spike 2026-09-03 picked Diffusers.
WAN_DIFFUSERS_CONFIG_ID = "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers"
# Ampere 3090: cuDNN rejects Float8_e4m3fn (tools/spike_wan_lora_mask.txt). Compute is bfloat16
# with accelerate cpu_offload so 14B bf16 (~28 GB) stays on RAM, not the 24 GB card.
WAN_TRANSFORMER_COMPUTE = "bfloat16"

HELP_GENERATE = """GENERATE
Two stages: WAN 1440×720 first, then Confirm WAN → HiRes 8K.
WAN prompt must describe this pano. Empty blocks Generate. The 2D green-outpaint
instruction is stripped; WAN fills black holes in the control video, not green.
Click a rail number to preview that clip. Re-gen selected rail after editing the
prompt, seed, or the Geometry path (Play that rail first).
Style suffix is optional. Negative is editable. Seed default 0.
HiRes feathers the 8K photograph into the WAN hole (seam feather, default 16 px) so the
mask join is not a hard edge, then RGB-gains the WAN fill toward the surrounding photo.
The 720p WAN clip itself is already one generated frame — re-run Confirm WAN → HiRes
after changing feather or tone; do not re-Generate WAN for that.
WAN loader: Diffusers 0.40 (spike 2026-09-03). Matrix-3D script did not run on this PC.

WorldStereo 2.0 is an experimental Quality generator, off by default. WAN stays
the default and stays installed. The choice stays disabled until the FP8 weights
and DINOv2 rows are READY. Fast and Scout do not offer it. One process generates
every rail and carries the memory bank across them. White in the WorldStereo mask
is empty. Confirm → HiRes on that generator does not start; WAN HiRes is unchanged.
Switching back to WAN leaves WAN outputs where they are. Known-pose Reconstruct
writes worldstereo/dataset from the keyframes and does not change the WAN COLMAP
model. Pinhole HiRes is not in this build.
"""
