"""Panorama widget values ported from the Krea graph and the panorama node pack (MIT, pin 4d5ff7c). See THIRD_PARTY_NOTICES.md. Do not invent names."""

from __future__ import annotations

# Native Ostris Edit is ported (pack pin 7756566). 2D Images Generate is enabled.
OSTRIS_READY = True

GENERATE_WIDTH = 2048
GENERATE_HEIGHT = 1024
FINAL_WIDTH = 8192
FINAL_HEIGHT = 4096

# 360 Images: |width/height - 2| <= this (locked in P1 Q&A).
ASPECT_2TO1_TOLERANCE = 0.01

GREEN_ERP = "#00ff00"
# EmptyImage node 37 widgets [2048, 1024, 1, 65280] = #00FF00
GREEN_CANVAS_COLOR = 65280

# Krea2OstrisEditModelPatch widget [true]. Default in node source is False; graph is True.
OSTRIS_KV_CACHE = True
# TextEncodeKrea2OstrisEdit node 45 widget
OSTRIS_NEGATIVE = "ugly, low quality"
# FluxKontextMultiReferenceLatentMethod nodes 47 and 49
FLUX_REF_METHOD = "index_timestep_zero"

# PerspToErpWarp at panorama-nodes pin 4d5ff7c.
# JSON widgets: [2048, 1024, 1, 70, 0, 0, 0, 2, 10, 0.55, "#000000"]
WARP_LENGTH = 1
H_FOV_DEG_DEFAULT = 70.0
H_FOV_DEG_MIN = 10.0
H_FOV_DEG_MAX = 170.0
H_FOV_DEG_STEP = 0.5
WARP_YAW_DEG = 0.0
WARP_PITCH_DEG = 0.0
WARP_MASK_FEATHER = 0
WARP_SUPERSAMPLE = 2.0
WARP_GRID_SPACING_DEG = 10
WARP_GUIDE_OPACITY = 0.55
WARP_FILL_COLOR = "#000000"

# 8K paste of the source photo after the upscale tail. Feather is pixels at 8192 wide.
PHOTO_FEATHER_PX = 128
PHOTO_PYRAMID_LEVELS = 5

# HarmonizeBoundary JSON [1, 8, 800, true]
HARMONIZE_STRENGTH = 1.0
HARMONIZE_SOLVE_SCALE = 8
HARMONIZE_ITERATIONS = 800
HARMONIZE_WRAP_HORIZONTAL = True

# Krea2Reference JSON [8]
KREA2_REFERENCE_MAX_MEGAPIXELS = 8.0

# RollHorizontal JSON [0.5]
ROLL_FRACTION = 0.5

# SeamMask JSON [2048, 1024, 160, 32]
SEAM_STRIP_WIDTH = 160
SEAM_STRIP_FEATHER = 32

OUTPAINT_INSTRUCTION = (
    "Fill the green spaces according to the image. Outpaint as a seamless 360 "
    "equirectangular panorama (2:1). Keep the horizon level. Match left and right edges."
)


def wan_prompt_from_stored(text: str) -> str:
    """Scene sentence for WAN. Drops the 2D green-ERP outpaint instruction.

    WAN fills black holes in the control video (``hole_fill='black'``), not green.
    ``prompt.txt`` still stores the P1 concat; do not silently replace that file.
    """
    body = (text or "").strip()
    inst = OUTPAINT_INSTRUCTION.strip()
    if body.startswith(inst):
        body = body[len(inst) :].strip()
    return body

# Always appended to 2D surroundings (typed or Florence-2). Exact user wording.
SURROUNDINGS_SUFFIX = "no peoples, no cars"


def with_surroundings_suffix(text: str) -> str:
    """Append SURROUNDINGS_SUFFIX once. Empty stays empty (still blocks Generate)."""
    body = (text or "").strip().rstrip(" .,")
    if not body:
        return ""
    if body.lower().endswith(SURROUNDINGS_SUFFIX):
        return body
    return body + ", " + SURROUNDINGS_SUFFIX

TRIGGER_PREFIX = (
    "img-txt-2-360 A full 360 degree equirectangular panorama in 2:1 spherical projection,"
)

UPSCALER_PROMPT_DEFAULT = "High resolution photography"

# CLIPLoader widgets
CLIP_FILENAME = "qwen3vl_4b_fp8_scaled.safetensors"
CLIP_TYPE = "krea2"
CLIP_DEVICE = "default"

UNET_FILENAME = "krea2_turbo_fp8_scaled.safetensors"
UNET_WEIGHT_DTYPE = "default"
VAE_FILENAME = "wan_2.1_vae.safetensors"
LORA_T2I_FILENAME = "krea2_t2i_360_erp_lora_v1.safetensors"
LORA_T2I_STRENGTH = 1.0
LORA_OUTPAINT_FILENAME = "krea2_oedit_360_erp_outpaint_lora_v1.safetensors"
LORA_OUTPAINT_STRENGTH = 1.0
REALESRGAN_FILENAME = "RealESRGAN_x2.pth"

# KSampler TEXT→PANO JSON [322344328372862, "fixed", 10, 1, "er_sde", "simple", 1]
TEXT_PANO_SEED_DEFAULT = 322344328372862
TEXT_PANO_STEPS = 10
TEXT_PANO_CFG = 1.0
TEXT_PANO_SAMPLER = "er_sde"
TEXT_PANO_SCHEDULER = "simple"
TEXT_PANO_DENOISE = 1.0

# SEAM INPAINT node 31 TEXT JSON [8, "fixed", 10, 1, "er_sde", "simple", 0.55]
TEXT_SEAM_SEED_DEFAULT = 8
TEXT_SEAM_STEPS = 10
TEXT_SEAM_CFG = 1.0
TEXT_SEAM_SAMPLER = "er_sde"
TEXT_SEAM_SCHEDULER = "simple"
TEXT_SEAM_DENOISE = 0.55

# KSampler IMAGE→PANO JSON [12345, "fixed", 10, 1, "er_sde", "simple", 1]
IMAGE_PANO_SEED_DEFAULT = 12345
IMAGE_PANO_STEPS = 10
IMAGE_PANO_CFG = 1.0
IMAGE_PANO_SAMPLER = "er_sde"
IMAGE_PANO_SCHEDULER = "simple"
IMAGE_PANO_DENOISE = 1.0

# SEAM INPAINT node 63 IMAGE JSON [12345, "fixed", 10, 4, "euler", "simple", 0.45]
IMAGE_SEAM_SEED_DEFAULT = 12345
IMAGE_SEAM_STEPS = 10
IMAGE_SEAM_CFG = 4.0
IMAGE_SEAM_SAMPLER = "euler"
IMAGE_SEAM_SCHEDULER = "simple"
IMAGE_SEAM_DENOISE = 0.45

# ModelSamplingFlux IMAGE→PANO JSON [0.90625, 0.5, 2048, 1024]
FLUX_MAX_SHIFT = 0.90625
FLUX_BASE_SHIFT = 0.5

# ImageScale JSON ["lanczos", 8192, 4096, "disabled"]
SCALE_METHOD = "lanczos"
SCALE_CROP = "disabled"

# ImageCASharpening+ JSON [0.5]
CAS_AMOUNT = 0.5

# ImageAddNoise JSON [0, "fixed", 0.01]
NOISE_SEED = 0
NOISE_STRENGTH = 0.01

# UltimateSDUpscaleNoUpscale node 77 JSON
# [0, "fixed", 2, 1, "er_sde", "simple", 0.15, "Linear", 512, 512, 32, 64, "None", 1, 64, 8, 16, true, false, 1]
# Names from ssitu INPUT_TYPES at pin 627c871. Tile w/h linked PrimitiveInt 1024.
# Trailing 1 is unnamed in INPUT_TYPES (not passed).
USD_SEED = 0
USD_STEPS = 2
USD_CFG = 1.0
USD_SAMPLER = "er_sde"
USD_SCHEDULER = "simple"
USD_DENOISE = 0.15
USD_MODE_TYPE = "Linear"
USD_TILE = 1024
USD_MASK_BLUR = 32
USD_TILE_PADDING = 64
USD_SEAM_FIX_MODE = "None"
USD_SEAM_FIX_DENOISE = 1.0
USD_SEAM_FIX_WIDTH = 64
USD_SEAM_FIX_MASK_BLUR = 8
USD_SEAM_FIX_PADDING = 16
USD_FORCE_UNIFORM_TILES = True
USD_TILED_DECODE = False

# Florence-2. Pinned after first successful engine.workers.caption (2026-09-02).
# microsoft/Florence-2-large remote code fails on transformers 5.16; this is the
# transformers-native conversion of the same Microsoft Florence-2-large (Apache-2.0).
FLORENCE_HUB_ID = "florence-community/Florence-2-large"
FLORENCE_REVISION = "4271c66b88cdbc05735372ec13b2360108de5317"
FLORENCE_TASK = "<CAPTION>"

HELP_PANORAMA = """PANORAMA
graph how-to (MarkdownNote id 4) · Votion labels

Pick 2D Images (your photo is placed on the sphere, Krea 2 + Ostris Edit + ERP outpaint LoRA fills the rest), 360 Images (load a 2:1 ERP; skip Krea), or text (Krea 2 + 360 LoRA paints the pano from the scene prompt). Only the selected mode is computed.

Pano resolution: keep 2:1. 2048 x 1024 is the sweet spot for Krea 2. Both modes, the ERP reference and the timestep curve follow it. Final size after the shared tail is 8192 x 4096.

2D Images: load the photo, set h_fov, and write one sentence about the surroundings (or Florence-2 caption, then edit). Detect fills h_fov from the photo's lens EXIF; without EXIF (or after a crop) it uses a MoGe guess, which can be 10 degrees off on a straight-on facade. Generate runs native Ostris Edit + ERP outpaint LoRA, then pastes the original photo back onto the 8K panorama with a feathered seam.

text: scene sentence after the locked trigger prefix. After Generate, prompt.txt = prefix + sentence. Votion does not need a dummy 2D image in text mode.

360 Images: 2:1 file becomes pano.png. Empty prompt.txt is OK in P1. Optional Upscale to 8K runs the shared tail.

Give the upscaler prompt a short description. A few words are enough. Do not put scene details there.

The selected 2K pano and the FINAL 8K pano land in outputs/<scene>/ as pano.png.
"""
