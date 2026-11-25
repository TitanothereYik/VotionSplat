"""P2 in-app trainer knobs (LiteGS backend, 2026-09-07). Dummy: docs/Votion_3DGS_2.0_Phase2.html splat desk.

Torch-free: Qt imports this module.
"""

from __future__ import annotations

import os
from pathlib import Path

STRATEGY_LITEGS = "LiteGS"
STRATEGY_MCMC_COMPACT = "MCMC Compact"
STRATEGIES = (STRATEGY_LITEGS, STRATEGY_MCMC_COMPACT)
# argv keys <-> labels (engine.workers.splat --strategy)
STRATEGY_KEYS = {"litegs": STRATEGY_LITEGS, "mcmc_compact": STRATEGY_MCMC_COMPACT}
# Old gsplat labels: a checkpoint carrying one of these is params-only -> Retrain required.
LEGACY_STRATEGIES = ("Splat3", "MCMC")


def strategy_from_key(key: str) -> str:
    return STRATEGY_KEYS.get(str(key).lower(), STRATEGY_LITEGS)


def strategy_to_key(label: str) -> str:
    for k, v in STRATEGY_KEYS.items():
        if v == label:
            return k
    return "litegs"


MAX_STEPS = 10000  # default in the box; the user may type any integer >= 1
GAUSSIAN_BUDGET = 3_000_000  # editable; LiteGS target_primitives / MCMC Compact prune target
GAUSSIAN_CAP = GAUSSIAN_BUDGET  # old name, kept for the docs reader
RESOLUTION_SCALES = (1.0, 0.5, 0.25)
SH_DEGREE = 3
# SH warm-up and densify cycles are step-based (engine.splat.controllers.DensifySchedule):
# one SH band per clamp(max_steps // 10, 200, 1000) steps; LiteGS: ~24 densify cycles over
# 5 %..80 %; MCMC Compact: the 3DGS cadence below over the same window.
MCMC_DENSIFY_INTERVAL = 100  # steps per MCMC Compact densify cycle (3DGS densification_interval)
MCMC_DELTA_SPLIT = 0.05  # target fraction of Gaussians split/cloned per cycle (MH chain target)
# Neither controller's prune keeps LiteGS's screen_size_threshold (vendor dead code: the
# constant is read but nothing checks a Gaussian's rendered footprint against it -- see
# vendor/litegs/litegs/training/densify.py). Split/clone only fires on Gaussians the
# gradient already flagged as high-error (engine.splat.controllers.mcmc_compact.tau_s
# picks split-vs-clone *within* that candidate set); a Gaussian on a low-gradient region
# can grow quietly and unboundedly under plain Adam. At full source resolution (HiRes
# dual-res reconstruct, ~2048px cube faces) that runaway growth both blurs the render
# (one huge soft Gaussian standing in for detail) and, combined with SphereSfM cameras
# clustered close to scene content, can cover enough screen-space tiles to overflow the
# CUDA binning table's 32-bit size (crash).
#
# A flat world-space cap doesn't work on this scene shape: SphereSfM cameras cluster
# tightly (``norm_radius`` = 1.1x the camera-position spread, ~6.7 units here) while the
# reconstructed content spans much farther (``scene_scale``, ~47 units) -- background/
# distant structure legitimately needs Gaussians far bigger than any single world-space
# threshold that also has to stay small enough to be safe up close. Tried a flat 0.1x
# cap: it capped legitimate far/background coverage too, starving those regions of fit
# and sending the split-threshold Markov chain into a feedback loop of ever-more splits
# chasing gradients a capped Gaussian can never resolve (loss rose monotonically, PSNR
# fell from ~13 dB to ~7 dB by step 11k with N ballooning past 2.7M -- worse than no
# clamp at all). What actually blows up the binning table is *screen-space* size, which
# is world-space scale divided by camera distance -- so the cap has to scale with
# distance too. Backend-level (``litegs_backend.clamp_runaway_scale``, called after every
# optimizer step and right after gaussian init / checkpoint load so even the very first
# render is covered) so both controllers share the same guard: each Gaussian's
# world-space scale (already exp'd) is clamped to this fraction of its own distance to
# ``cam_center`` (the camera-position centroid -- SphereSfM cameras cluster tightly, so
# any one camera is a reasonable stand-in for "the rig" at rendering time). A far
# background Gaussian keeps room to grow; only genuinely close-to-camera Gaussians (the
# ones that produced the 0.5-0.8x scale/distance ratios behind both crashes) get
# clamped.
SCALE_CLAMP_CAM_DIST_FRACTION = 0.15
# Second tier: an absolute ceiling on top of the distance-proportional one. Sparse
# panoramic SfM reconstructs a handful of points at wildly overestimated depth --
# typically sky/horizon features with too little parallax for stable triangulation --
# 20x+ farther from the camera cluster than the reconstructed scene itself
# (``scene_scale``). The proportional clamp alone lets those inherit a huge *absolute*
# scale (seen: up to 16 units, 2.4x norm_radius, on an alpine_01 retrain) since it only
# looks at distance-to-centroid, not whether that distance is trustworthy. Well above the
# distance clamp's own typical output (p99.9 of a real run's max-scale sat at
# ~0.6x norm_radius) so it only culls that reconstruction-error tail, not real geometry.
SCALE_CLAMP_ABS_NORM_RADIUS = 1.0
# Root-cause companion to the clamp above: don't just cap the *scale* of a bad
# triangulation's Gaussian, drop the point before it ever seeds one. Diagnostic on
# alpine_01 dualres (2026-09-08): 16% of points3D sat >1x the camera-cluster extent
# from the *nearest camera* (some 30-100+ units out against a ~6.7 norm_radius rig) --
# reflective glass facades and near-parallel panoramic rays both produce these
# low-confidence-but-numerically-"consistent" (near-zero reprojection error) far
# triangulations. COLMAP's per-point `error` field is unpopulated by colmap_sphere.exe
# (always -1), so nearest-camera distance is the signal; generous on purpose so real
# distant background (e.g. a farther skyscraper, or a forward-facing plane in front
# of a small rig) survives -- this only culls the extreme tail. Must NOT use distance
# to the camera *centroid*: content in front of a clustered rig is supposed to be
# farther than the rig radius. 0 disables.
INIT_POINT_MAX_CAM_DIST_MULT = 4.0
# 3DGS/LiteGS default: densify through 80 % of the run. An earlier 0.4 workaround existed
# only because MCMC Compact's budget prune re-competed the *entire* population every
# cycle once N hit the cap (PSNR sat ~12 until the window closed). The controller now
# skips split/clone at the cap, so the 80 % window is safe and short 10k runs can still
# grow (0.4 on 10k stopped densify at step 4k with N≈117k — too few for a sharp alpine).
MCMC_DENSIFY_UNTIL_FRACTION = 0.8
CKPT_EVERY = 100  # steps
HOLDOUT_EVERY = 8  # every 8th COLMAP view when hold-out eval is on
VIEW_EVERY_STEPS = 10  # viewport refresh while training when the camera is still
VIEW_MAX_FPS_TRAINING = 30  # orbit cap while training (idle view / pause: uncapped, ~100 fps)
PREVIEW_WIDTH = 1280  # PNG fallback frame width
PREVIEW_REL = "_work/splat_preview.png"
VIEW_REL = "_work/splat_view.json"  # persisted camera (written at worker exit and by Qt)
VIEWPORT_MAX_W = 1920
VIEWPORT_MAX_H = 1080
VIEW_BG = (8, 16, 24)  # opaque navy plate behind the splat
# WASD/QE flythrough (UE-editor style, app.widgets.splat_view): units are "reach" per
# second (reach = distance from eye to the scene center, see Viewport._compute_reach) --
# 0.6 crosses the whole scene in ~1.7s, matching the wheel-dolly's existing pace. Shift
# sprints at 3x, same convention as Unreal's editor viewport.
VIEWPORT_FLY_SPEED = 0.6  # reach/sec
VIEWPORT_FLY_SPRINT = 3.0  # Shift multiplier
COMPACT_MIN_OPACITY = 0.005

# PLY export colour. Training stays sRGB. Coordinates stay COLMAP/OpenCV.
# The Unreal plugin converts on import. Do not sRGB-to-linear bake SH for the plugin preset.
# Older export names map to the VotionSplat preset so old commands still run.
COLOR_SPACE_SRGB = "srgb"
COLOR_SPACE_LINEAR = "linear"
COLOR_SPACE_VOTION = "votion"
COLOR_SPACES = (COLOR_SPACE_VOTION, COLOR_SPACE_SRGB, COLOR_SPACE_LINEAR)
COLOR_SPACE_DEFAULT = COLOR_SPACE_VOTION
COLOR_SPACE_ALIASES = {
    "wallgs": COLOR_SPACE_VOTION,
    "wall": COLOR_SPACE_VOTION,
    "unreal": COLOR_SPACE_VOTION,
    "ue": COLOR_SPACE_VOTION,
    "wallgs_acescg": COLOR_SPACE_VOTION,
    "acescg": COLOR_SPACE_VOTION,
    "aces": COLOR_SPACE_VOTION,
    "luma": COLOR_SPACE_VOTION,
    "lumaai": COLOR_SPACE_VOTION,
    "srgb": COLOR_SPACE_SRGB,
    "3dgs": COLOR_SPACE_SRGB,
    "gamma": COLOR_SPACE_SRGB,
    "linear": COLOR_SPACE_LINEAR,
    "lin": COLOR_SPACE_LINEAR,
    "votion": COLOR_SPACE_VOTION,
    "votion_ue": COLOR_SPACE_VOTION,
    "plugin": COLOR_SPACE_VOTION,
}
COLOR_SPACE_CHOICES = (
    (COLOR_SPACE_VOTION, "Unreal / VotionSplat Export"),
    (COLOR_SPACE_SRGB, "3DGS / sRGB (Brush, Postshot, SuperSplat)"),
    (COLOR_SPACE_LINEAR, "Linear Rec.709, tagged (other engines)"),
)
VOTION_UE_SETUP = (
    "Unreal / VotionSplat Export: drag splat.ply into the Content Browser with the Votion Splat plugin. "
    "No settings. The file keeps the trained splats, and the plugin draws them with the same rule as this viewport."
)
# VotionSplat plugin header. The file is trained colour, unsoftened; the plugin rasterises it.
VOTION_TAG_VERSION = 1
VOTION_RASTERS = ("gsplat_antialiased", "gsplat_classic", "litegs_classic")
LITEGS_NEAR_PLANE = 0.2
GSPLAT_NEAR_PLANE = 0.01
# Postshot colour tags in the PLY header for the linear preset.
COLOR_SPACE_TAGS = {
    COLOR_SPACE_LINEAR: ("comment postshot.color.primaries=Rec709", "comment postshot.color.eotf=Linear"),
}
# OCIO config the plugin inverse-view LUT is baked from (tools/unreal/bake_inverse_view_lut.py).
OCIO_CONFIG_DEFAULT = Path(__file__).resolve().parents[2] / "docs" / "unreal" / "VS_OCIO.ocio"
OCIO_SOURCE = "ACEScg"
OCIO_DISPLAY = "sRGB - Display"
OCIO_VIEW = "ACES 1.0 - SDR Video"


def ocio_config() -> Path:
    return Path(os.environ.get("VOTION_OCIO_CONFIG") or OCIO_CONFIG_DEFAULT)


# HY-World export "Soften like the viewport". The app preview is gsplat antialiased at logical
# pixels, then upscaled by the display scaling; classic splat renderers draw at physical pixels.
# Sizes are logical pixels at SOFTEN_FOCAL, seen from each splat's nearest training camera.
SOFTEN_FOCAL = 960.0  # app viewport at its 1920 px cap, 90 deg horizontal
SOFTEN_DPR = 1.5  # Windows display scaling on the Votion workstation
SOFTEN_UE_EPS_PX2 = 0.3 / SOFTEN_DPR**2  # classic renderers' 0.3 px^2 low-pass
SOFTEN_EXTRA_PX2 = 1.6  # baked footprint growth; least error vs the upscaled preview is 1.0-1.6
HELP_SOFTEN = (
    "The trainer fades splats thinner than a pixel, and the preview is upscaled by Windows display "
    "scaling. Classic splat renderers draw every splat at full opacity and full resolution, so thin "
    "splats show as needles and dots. On: each splat is widened slightly and its opacity rescaled to "
    "match the preview, for the sRGB and linear exports. Off: the trained splats as they are. "
    "The VotionSplat export skips this, because the plugin draws the viewport rule itself."
)


def color_space_from_key(key: str) -> str:
    return COLOR_SPACE_ALIASES.get(str(key).strip().lower(), COLOR_SPACE_DEFAULT)


CKPT_REL = "checkpoints/splat_latest.pt"
META_REL = "checkpoints/splat_meta.json"
PLY_REL = "splat.ply"
PLY_COMPACT_REL = "splat_compact.ply"

LICENSE_LINE = (
    "LiteGS (Moore Threads) is under the Inria Gaussian-Splatting License: research and "
    "evaluation use only, no commercial use without Inria consent. Built locally at Setup; "
    "never redistributed."
)

HELP_SPLAT = f"""SPLAT
Backend is LiteGS (vendored, built at Setup). Pick LiteGS (default densify: TamingGS scores, Gaussian budget is the target) or MCMC Compact (Jung & Hong 2025 section 3.1: Metropolis-Hastings adaptive split threshold + stochastic importance prune to the same budget), then Train.
Continue resumes the 100-step checkpoint with optimizer and scheduler state; switching controller needs Retrain. Checkpoints from the old gsplat trainer are params-only and show "Retrain required".
Max steps is yours — any integer >= 1, no product cap; epoch = steps / training views. Gaussian budget is editable (default 3M). Resolution scale 1 / 0.5 / 0.25 trains on downsampled frames. Hold-out eval keeps every 8th view out and reports PSNR.
Pause / Resume stay in-process; Stop kills the worker (job bar Stop is the same). SH warms up one band per ~10 % of the run (max 1000 steps); densify runs between 5 % and 80 % of the steps whatever the step count (LiteGS ~24 cycles; MCMC Compact one cycle per {MCMC_DENSIFY_INTERVAL} steps, splitting ~{int(MCMC_DELTA_SPLIT * 100)} % per cycle, so it grows steadily and ends well below LiteGS at the same budget).
Viewport is the live LiteGS raster over shared memory (no PNG polling): LMB look, RMB pan, wheel dolly, zoom uncapped, Reset is the rails Star. Click a training camera to jump there; GT compare shows render | ground truth side by side. A bowed ground after an old Reconstruct is gnomonic cube-face GT — Reconstruct then Retrain, do not Continue.
HUD: state, step / epoch, loss EMA, PSNR, Gaussians / budget, it/s, viewport fps, VRAM. Log drawer shows controller events (densify, prune, eps_curr).
Export splat.ply is Inria PLY. Unreal / VotionSplat Export (default) keeps the trained splats for the Votion Splat plugin. {VOTION_UE_SETUP} Switch to 3DGS / sRGB for Brush, Postshot, SuperSplat, or Linear Rec.709 for other real-time engines (tagged Rec.709 linear). Export compact prunes opacity < {COMPACT_MIN_OPACITY} and can drop the f_rest bands. Do not import moge/pointcloud.ply; do not scale the Gaussian actor.
If the LiteGS spike failed: Train disabled; Open COLMAP folder still works (Brush).
{LICENSE_LINE}
"""
