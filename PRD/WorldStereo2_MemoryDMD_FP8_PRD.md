# PRD — WorldStereo 2.0 (memory DMD, FP8) as an experimental Quality generator

**Product:** Votion 3DGS 2.0  
**Track:** experimental generator, gated. Not a numbered phase.  
**Depends on:** Phase 3 working end to end on `alpine_02` (four WAN rails, HiRes, SphereSfM, LiteGS splat), because that is the baseline it must beat.  
**Unlocks:** nothing by itself. If it wins the bake-off, a follow-up PRD decides whether it becomes the Quality default.  
**Out of scope:** replacing WAN, changing Fast / Scout, HY-Pano, WorldNav, WorldMirror, HY-World’s 3DGS trainer, any change to Phases 0–2.

**Sources (do not drift):**

- Model code: [FuchengSu/WorldStereo](https://github.com/FuchengSu/WorldStereo) (Apache-2.0, `LICENSE.md`)
- Official weights: [hanshanxue/WorldStereo](https://huggingface.co/hanshanxue/WorldStereo), subfolder `worldstereo-memory-dmd` (commit `2adb716` is the FP8 source)
- FP8 build: [apozz/WorldStereo-fp8](https://huggingface.co/apozz/WorldStereo-fp8), file `WorldStereo-memory-dmd-fp8_e4m3fn_scaled.safetensors`
- Papers: WorldStereo [arXiv:2603.02049](https://arxiv.org/abs/2603.02049) (CVPR 2026); HY-World 2.0 [arXiv:2604.14268](https://arxiv.org/abs/2604.14268) §5 (WorldStereo 2.0), §7 (composition)
- Votion background: [docs/WAN_Replacement_Review_and_Alternatives.md](../docs/WAN_Replacement_Review_and_Alternatives.md) §4.2 and §6, [docs/WorldMirror2_Report.md](../docs/WorldMirror2_Report.md)
- Certainty log: [docs/Votion_3DGS_2.0_Index.md](../docs/Votion_3DGS_2.0_Index.md)
- Phase 3 baseline: [Phase3_Dataset_PRD.md](Phase3_Dataset_PRD.md)
- Set index: [README.md](README.md)

If this PRD and the certainty log disagree, the certainty log wins. **This PRD does not reopen anything by being written.** Section 0 lists what the owner has to sign first.

---

## 0. Gates (owner, before any code)

| Gate | Why | Who |
|------|-----|-----|
| **G1 Reopen** | The Index rejects “Hunyuan / HY-World 2.0” (two huge models, four GPUs) and `PRD/README.md` says do not reopen HY-World 2.0. WorldStereo is HY-World 2.0’s World Expansion stage, published separately. Using only this one model, single-GPU, needs a dated certainty-log entry. | Yik |
| **G2 Weights licence** | Code is Apache-2.0. The official weight repo `hanshanxue/WorldStereo` carries **no** licence file (24-byte README). The FP8 repack says “WorldStereo overlay: MIT (inherited)” plus Apache-2.0 for the Wan base. HY-World 2.0’s own repo is the Tencent community licence (excludes EU, UK, Korea). Which one governs the weights is **UNKNOWN**. Ask the authors or accept the risk in writing. | Yik |
| **G3 Experimental only** | Ships behind a Generate-page toggle, off by default, Quality profile only. WAN stays the default and stays installed. | Yik |

**Signed 2026-10-29 by Yik:** G1, G2, and G3 are Index certainty-log entries #58, #59, and #60. The Generate-page choice exists only as an experimental Quality generator, off by default. WAN stays the default. The control itself is still WS-5: it stays disabled until the weights row is READY.

**Correction to the earlier review:** `WAN_Replacement_Review_and_Alternatives.md` lists WorldStereo under the Tencent territory licence. That is the HY-World 2.0 repo’s licence. The standalone WorldStereo repo is Apache-2.0; see G2 for the weights.

---

## 1. Product overview

| Field | Value | Status |
|-------|--------|--------|
| Name | Votion 3DGS | Locked |
| Owner | **Yik**. MIT product. Small private team. | Locked |
| Version | **2.0** only | Locked |
| Feature | **WorldStereo 2.0 generator**: per-rail perspective keyframes generated from a point-cloud render, occlusion mask, and camera, with a **memory bank shared across rails** | Proposed |
| Model | `worldstereo-memory-dmd`, 4-step DMD, Wan 2.1 I2V **14B 480P** base + camera ControlNet + spatial-stereo memory blocks | Proposed |
| Weights on the 3090 | Scaled FP8 e4m3fn DiT, 20.35 GB on disk, resident on 24 GB per its card | Proposed; **measure** |
| High-level goal | Fix the measured multi-rail inconsistency (`alpine_01`: frame 1 alone 30.8 dB, frames 1–3 jointly 22.3 dB, hold-out 15.6 dB) with a model trained for cross-trajectory memory, without leaving one 3090 | Proposed |

---

## 2. Problem & objectives

**Problem.** WAN fills each rail independently. Two rails that look at the same hole invent two different things. SphereSfM and LiteGS then fit a scene that disagrees with itself. The single-view overfit proves the raster and optimizer are fine; the multi-view loss is the data.

WorldStereo 2.0 is built for that failure:

- **Global-Geometric Memory (GGM):** a ControlNet branch conditioned on renders of a point cloud that is extended as trajectories are generated.
- **Spatial-Stereo Memory (SSM++):** retrieved keyframes from earlier trajectories are stitched side by side with each target frame inside the main DiT, so new frames attend to what was already generated.
- **Memory bank:** starts from perspective views cut from the panorama; each finished trajectory adds keyframes (`align_nframe` 8 per clip by default).

**Objectives:**

1. Generate all rails of one scene through WorldStereo `memory-dmd` on one RTX 3090, sequentially, with the memory bank carried across rails.
2. Keep Votion’s inputs: Krea 8K ERP, MoGe at `moge_level` 9, Plot Camera rails starting at the star. No HY-Pano, no WorldNav, no VLM.
3. Produce a COLMAP dataset with **known poses** from the generated keyframes plus gnomonic faces of the 8K pano, and train it with the existing LiteGS backend.
4. Restore 8K detail where the pano can answer, by a pinhole version of HiRes.
5. Measure, on `alpine_02`, against the Phase 3 WAN baseline: joint fit of overlapping frames from different rails, hold-out PSNR, peak VRAM, wall time.
6. Native Python only. No ComfyUI process, no WSL, no `torchrun`, no NCCL.

**Not an objective:** beating WAN per clip. WorldStereo outputs ~480p perspective frames; WAN outputs 1440×720 ERP. The win condition is consistency (§8).

---

## 3. Target users & use cases

Audience: small private team, Quality profile, RTX 3090 24 GB. Persona job titles: UNKNOWN.

| ID | Scenario | Outcome |
|----|----------|---------|
| WS-S1 | User sets **Generator = WorldStereo (experimental)** on Generate, rails already Confirmed | Generate runs WorldStereo per rail in the listed order; WAN does not load |
| WS-S2 | User reorders rails before Generate | Order is the memory order. Later rails retrieve from earlier ones |
| WS-S3 | User Ditches rail 2 after all four ran | Warning: rails 3–4 retrieved from rail 2. Offer **Re-gen rails 3–4** or keep as is |
| WS-S4 | User Re-gens rail 1 | Rails 2–N marked stale (dirty chip). Re-gen of later rails is offered, not forced |
| WS-S5 | Weights missing | Home status `MISSING_worldstereo_fp8` (and any missing base part), Generate toggle disabled |
| WS-S6 | FP8 load OOMs on the 3090 | Job fails with a VRAM line in the log; toggle stays available; no silent fallback to BF16 streaming |
| WS-S7 | User switches Generator back to WAN | WAN rails, HiRes, SphereSfM untouched; WorldStereo outputs stay in their own folder |

### User stories

- As a user I can pick WAN or WorldStereo per scene on Generate, and WAN is the default.
- As a user I can see, per rail, the point-cloud render, its mask, and the generated keyframe side by side.
- As a user I can see how many memory-bank frames each rail retrieved.
- As a user I can Keep / Ditch / Re-gen a rail, and the UI tells me which later rails depended on it.
- As a user I can train the result on the same Splat page with the same controllers.

---

## 4. Functional requirements

### 4.1 Environment spike (first, outside the app)

Scratch folder, Votion venv (Python 3.12, cu128). WorldStereo’s README targets conda Python 3.11; record what breaks.

1. Vendor `FuchengSu/WorldStereo` at a pinned SHA under `vendor/worldstereo/` with its `LICENSE.md` and a NOTICE. Do **not** vendor `Tencent-Hunyuan/HY-World-2.0` (different licence, and only needed for WorldMirror post-processing, which this PRD skips).
2. Dependencies to prove on Windows, in this order, each written to `docs/env_report.txt`:
   - `diffusers` with `WanTransformer3DModel` (already used by `engine/video/wan_runtime.py`)
   - **PyTorch3D** (README step 3; used for point-cloud rendering). No official Windows wheel for 3.12 + cu128 is known; try a local build with the same toolchain `engine/cuda_jit.py` uses for LiteGS. If it fails, see §4.4 fallback.
   - MoGe: WorldStereo pins MoGe commit `0286b495…`. Votion vendors MoGe already. Record whether the vendored copy satisfies its imports.
   - DINOv2 (memory retrieval scoring). Exact checkpoint: read from the WorldStereo config at pin. Must load from `<install_root>\models`, never `torch.hub` over the network.
3. Load path: Python API `WorldStereo.from_pretrained(..., subfolder="worldstereo-memory-dmd", sp_world_size=1, fsdp=False, device=cuda:0)`, with the DiT weights swapped for the FP8 file (§4.3). No `torch.distributed` init. If the code path insists on a process group, patch it to a no-op single rank and record the patch.
4. Run one trajectory of the repo’s `examples/panorama` scene (if shipped at the pin; else skip to §4.2 with `alpine_02`).

**Exit:** one 21-keyframe trajectory written, peak VRAM and wall time recorded on the 3090. No UI work before this passes.

### 4.2 Votion scene → WorldStereo inputs (adapter)

WorldStereo’s preprocessing for panoramic scenes is **not released** (README TODO: “Release data pre-processing pipelines for panoramic and multi-trajectory scenes”, unchecked). Votion writes its own adapter, `engine/worldstereo/adapter.py`, from the documented input layout:

```
<scene>/worldstereo/input/
  meta_info.json            {"scene_type": "panorama"}
  start_frame.png           per direction, see below
  prompt.json               {"short caption", "medium caption", "long caption"}
  render_results/
    <view_id>/              one per (rail, direction)
      <traj_id>/
        render.mp4          point-cloud render along the rail
        render_mask.mp4     binary occlusion mask
        camera.json         {"extrinsic": [4x4 w2c, OpenCV], "intrinsic": [3x3]}
```

Do **not** write `panorama.png` into the input folder: its presence triggers the VLM single-path inference.

**Mapping rules:**

| WorldStereo field | Votion source |
|-------------------|---------------|
| Global point cloud (GGM) | MoGe on the 8K pano at `moge_level` 9 (same run Plot Camera uses). Export points, not the mesh. Sky handling: copy what Plot Camera already does; do not invent a sky segmenter |
| Initial memory bank | Perspective cuts of the 8K pano at the origin. WorldStereo 1.0 used 27 views at 90°×120° FoV; HY-World 2.0 says “perspective views subdivided from the input panorama”. Read the count and FoV from the code at pin |
| Trajectory poses | Plot Camera rail, `length` 81, every 4th frame → **21 poses** (0, 4, …, 80). Matches `nframe: 21` in the `memory-dmd` config. Rails already start at the star, which is where the memory bank was cut |
| Direction per rail | The rail's own look is **front**. Up to six views: front, back, left, right, up, down. Front is always on. Up and down are outside the training range (`START_ELEVATION` 5°) and are generated last, with `ood_view` in `meta.json` |
| Frame layout | **Spaced** (default). Slot order is front, left, right, up, down, back, repeating across the 21 rail positions, so six views make the same number of kept images as one view. Each non-front run prepends an anchor at rail position 0 (pano cut at the origin) and discards it after sampling. **Separate** still runs a full 21-frame trajectory per view. Generation order stays front, back, left, right, up, down |
| Default rails | Four straight `look_forward` lines: forward, back, left, right. Shortened to 80% of the free distance to the MoGe cloud. Suggest paths stays. Previously confirmed rails are kept |
| `start_frame.png` | Pinhole cut of the 8K pano at the origin facing that direction, at the generation size |
| `render.mp4` / `render_mask.mp4` | Point-cloud render of the MoGe cloud along the 21 poses. Match the training renderer from the config: `render_radius` 0.008, `points_per_pixel` 8, `background_color` black. Mask polarity: read from the code at pin, record it here (**UNKNOWN** now) |
| Intrinsics | From the chosen size bucket and FoV. Field of view: **UNKNOWN**, read from the example `camera.json` at pin |
| Size | `scale_map` bucket **480×832** (H×W, landscape). Other buckets exist (e.g. 576×704); do not use 720p, the DMD config lists no 720p bucket |
| Prompt | `prompt.txt` copied into all three caption levels. Empty `prompt.txt` blocks Generate, same as WAN |
| Seed | Config default `1024`. Editable on Generate |

Write each (rail, direction) as its own `view_id/traj_id`. The adapter output is deterministic: same scene, same rails, same settings produce the same files.

### 4.3 FP8 loading

The FP8 file is Kijai-style **scaled** FP8 with diffusers key names (`blocks.0.attn1.to_q.weight`), matching `hanshanxue/WorldStereo`’s layout.

| Item | Rule |
|------|------|
| Which tensors are FP8 | 549 rank-2 Linear weights (attention Q/K/V/O, FFN, ControlNet linears), each with a `<name>.scale_weight` BF16 scalar. Conv, norms, biases, `time_*`, `add_*` (I2V image cross-attention) stay BF16 |
| Marker | A `scaled_fp8` tensor is present; the loader checks it and refuses a file without it |
| Ampere path | Replace each FP8 `nn.Linear` with a module that keeps the weight in `float8_e4m3fn` and computes `F.linear(x, w.to(x.dtype) * scale, bias)` per call. **No speed-up on the 3090** (no FP8 matmul); the gain is memory only |
| Do not | Dequantize the whole DiT to BF16 at load (`engine/worldstereo/encoders.py` does that only for umT5). That is ~35 GB for the DiT and defeats the point |
| Code | `engine/worldstereo/fp8.py`. Own implementation from the recipe on the model card; do not copy ComfyUI-WanVideoWrapper code |
| Text encoder, CLIP-Vision, VAE | Wan 2.1 I2V 480P and 720P share umT5-XXL, CLIP-Vision H and the Wan 2.1 VAE. Loaders live in `engine/worldstereo/encoders.py`. |
| Keyframe-VAE | WorldStereo encodes each keyframe independently (spatial-only compression). It reuses the Wan VAE weights; confirm at pin |
| Order on GPU | Encode text and CLIP first, move them off GPU, then load the FP8 DiT. Text encoder never resident with the DiT |

**VRAM budget (estimate, not measured):** DiT ~20.35 GB resident. Per trajectory: 21 frames at 480×832 → latent 60×104 → about 1,560 tokens per frame after 2×2 patching → ~33k tokens; retrieved reference frames are stitched along width, so up to ~2× tokens on retrieved frames. Headroom on 24 GB is ~3 GB for activations, PyTorch3D, and DINOv2. **Measure before any UI work.** If it does not fit:

1. Lower `align_nframe` or the retrieval cap (fewer stitched references).
2. Use a smaller `scale_map` bucket.
3. Offload a fixed number of DiT blocks to CPU (block swap), implemented natively.

Do not fall back to the BF16 34.86 GB file with partial loading in the product. That is a separate decision.

### 4.4 Point-cloud render

Preferred: WorldStereo’s own PyTorch3D path, so the conditioning matches training.

Fallback if PyTorch3D will not build on Windows: `engine/worldstereo/render.py`, a torch point rasterizer that reproduces `render_radius` 0.008 and `points_per_pixel` 8 with a black background. Accept it only if, on the same cloud and camera, its render matches PyTorch3D’s to within a threshold recorded at the spike (run the comparison once on a Linux box or from a PyTorch3D render shipped in the repo’s examples).

**Domain note:** Matrix-3D prefers mesh renders over point clouds (Moiré, wrong occlusion). WorldStereo was trained on point-cloud renders with augmentation for floaters and depth bleeding. Feed **point clouds** first. A mesh-render variant is a later A/B, not the default.

### 4.5 Generation worker

`engine.workers.worldstereo`, one process for the whole scene, because the memory bank lives across rails and views. Generate resumes. Start over clears `worldstereo/rails` and `worldstereo/memory` after a confirmation.

1. Load once: text encoder pass, CLIP pass, unload; FP8 DiT; VAE; DINOv2.
2. Initialise the memory bank from the pano perspective cuts and the MoGe cloud. On resume, restore the snapshot written by the last finished trajectory.
3. For each rail in user order, then each selected view in order front, back, left, right, up, down: sample 4 DMD steps, decode keyframes, write them, update the memory bank. Spaced passes `num_frames = 4*(k-1)+1` for a run of k frames (anchor included) and sets `align_nframe` to `min(8, k-1)` so a short run does not ask the memory bank for 8 frames.
4. Persist the bank after every view at `memory/after_<view_id>/`. When the last selected view of a rail finishes, also write `memory/after_rail_NN/`. Re-gen of rail k restores the `after_rail` snapshot of the previous rail in the user order, resamples every selected view of rail k, and writes `STALE.txt` into later rails.
5. A trajectory is finished when `meta.json` lists its keyframes, those files exist, and there is no `STALE.txt`. Separate runs still expect 21 frames. A folder whose `layout` does not match the request is unfinished. Generate skips finished ones. When the last spaced view of a rail finishes, write `rail_NN/combined/` with positions 000..020 interleaved.
6. Emit `WS view` lines and `PROGRESS` lines. HiRes does not start automatically.

Outputs per view:

```
<scene>/worldstereo/rails/rail_NN/<view>/
  keyframes/000.png …             480×832 RGB. Separate is 000..020. Spaced uses that view's positions only, e.g. front `000,006,012,018`
  render/render.mp4               the conditioning render, without the anchor
  mask/render_mask.mp4            the conditioning mask, without the anchor
  camera.json                     the kept rotated poses (no anchor)
  retrieved.json                  which memory frames were used
  meta.json                       seed, steps, bucket, weights SHA, wall time, peak VRAM, view, ood_view, layout, positions, anchor
```

Spaced also writes `rail_NN/combined/keyframes/000.png` … `020.png`, `combined.json` (position, view, w2c), and a combined render and mask for the preview. The short-run wall time versus the 7439 s / 30.9 GB alpine_02 front trajectory is not measured yet.

**2026-11-04.** Joint mode replaces Spaced, six-view Separate, and Dense. One sample covers every kept rail (10 frames each, 40 frames, out-and-back, frame 0 at the star). Per-rail generation stays, front view only, 21 frames. Forty keyframes is past the trained 21-frame clip. Quality, time, and VRAM of that sample are unmeasured.

Do not delete leftover WAN files on disk. WorldStereo writes only under `<scene>/worldstereo/`.

### 4.6 Pinhole HiRes

WorldStereo frames are ~0.4 MP. The Phase 3 rule still holds: sharpness comes from reprojecting the 8K pano through geometry, not from upscaling the generated frame.

`engine/hires/pinhole.py`, one click, **HiRes all**. One process loads MoGe and `4x-UltraSharp` once and loops every kept rail and every generated view. A view whose `hires/` is already complete is skipped.

- For each keyframe camera, render the 8K pano through MoGe geometry at `moge_level` **6** into a pinhole at **2× the keyframe size** (960×1664).
- Where the reprojection is valid, keep the 8K pixels. Where it is not, upsample the WorldStereo keyframe with `4x-UltraSharp` and use it.
- Write `hires/NNN.png` and `gate_masks/NNN.png`. White = 8K, black = generated.
- Log mean coverage. Not a ship gate.

The ERP HiRes worker is removed. Git history is the only copy.

### 4.7 Known-pose dataset

WorldStereo outputs pinhole frames with the poses Votion chose. SphereSfM is not needed for them.

`engine/sfm/known_pose.py` writes a COLMAP model:

| Part | Content |
|------|---------|
| `cameras` | `PINHOLE` per size; intrinsics from §4.2 scaled to the HiRes size |
| `images` | Every kept keyframe (HiRes version) with its pose. Plus the six 90° gnomonic faces of the 8K pano at the origin (same remap as Index #57) |
| `points3D` | MoGe pano cloud, plus per-keyframe depth aligned to the rendered cloud depth by per-frame scale and shift on valid pixels, outliers dropped. Algorithm from HY-World 2.0 §7.1.2 described in words; Votion’s own code, MoGe depth instead of WorldMirror |
| Consistency check | Optional: run COLMAP `point_triangulator` with poses fixed (from the SphereSfM build) and report mean reprojection error per rail. This is a free cross-rail consistency number |

Kept / Ditched rails follow the Reconstruct page. Ditched rails are excluded. Newest dataset is shown after the job finishes (Phase 3 rule).

### 4.8 Splat

Unchanged. LiteGS backend, LiteGS or MCMC Compact controller, full-SH `splat.ply`, hold-out PSNR. HY-World trains RGB-only Gaussians; Votion keeps SH (Index lock on reflections).

### 4.9 UI (Generate and Reconstruct only)

Generate page left rail adds:

- **Generator:** `WAN 2.1 + pano LoRA` (default) | `WorldStereo 2.0 (experimental)`. Disabled with a tooltip unless G1–G3 are recorded and weights are READY.
- **Views per rail:** 1 (default) – 4.
- **Rail order** list (drag), shown only for WorldStereo.
- Seed (default 1024), prompt as today. No Negative field (DMD is CFG-free); hide it for WorldStereo.
- **Generate** / Stop / **Re-gen selected rail** / **Confirm → HiRes** / Open frames folder.

Viewport while running: rail chips; three-up split `render | mask | keyframe`; a counter “retrieved N memory frames”. Never loads the Phase 3 dummy PNG.

Reconstruct page: Keep / Ditch / Re-gen per rail. Ditch or Re-gen on rail k shows which later rails retrieved from it (from `retrieved.json`). Dataset source switch: `SphereSfM (WAN)` | `Known poses (WorldStereo)`.

Home: model status rows `worldstereo_fp8`, `worldstereo_config`, DINOv2, plus any 480P-Diffusers parts the spike proves are needed. Per-cell download bars as today.

### 4.10 Files

```
vendor\worldstereo\                 pinned SHA, LICENSE.md, NOTICE
engine\worldstereo\__init__.py
engine\worldstereo\defaults.py      every number with "from config.json at <sha>" comments
engine\worldstereo\fp8.py
engine\worldstereo\adapter.py
engine\worldstereo\render.py        fallback only
engine\workers\worldstereo.py
engine\hires\pinhole.py
engine\sfm\known_pose.py
tools\model_manifest.json           new entries, exact filenames, no invented sizes
```

`defaults.py` starting values, copied from the `worldstereo-memory-dmd` config and README (re-copy at pin):

| Name | Value |
|------|-------|
| `BASE_MODEL` | `Wan-AI/Wan2.1-I2V-14B-480P-Diffusers` |
| `NFRAME` | 21 |
| `SIZE` | 480 × 832 |
| `STEPS` | 4 (DMD) |
| `SEED` | 1024 |
| `ALIGN_NFRAME` | 8 |
| `RENDER_RADIUS` | 0.008 |
| `POINTS_PER_PIXEL` | 8 |
| `BACKGROUND` | black |
| `SOBEL_THRESHOLD` | 0.35 |
| `START_ELEVATION` | 5.0 |
| `RAIL_STRIDE` | 4 (Votion: 81-frame rail → 21 poses) |

### 4.11 Acceptance criteria

- [x] G1, G2, G3 recorded in the certainty log with dates (Index #58–#60, 2026-10-29). #61 (2026-10-30) makes WorldStereo the Quality route.
- [x] `env_report.txt` records one WorldStereo trajectory on the 3090 (21 frames, FP8 stayed scaled, text encoder off the GPU, peak about 30.85 GB).
- [x] FP8 DiT stays resident; text encoder is never on GPU at the same time. Recorded on that spike.
- [x] Four rails of `alpine_02`, one view each, generated in one worker run with memory carried across rails. That run used the old one-view folder layout.
- [ ] Re-gen restores the previous rail's snapshot in the user order and marks later rails stale. The old index bug is fixed in code; it has not been run on the GPU.
- [ ] Pinhole HiRes writes 960×1664 frames and `gate_masks`; mean coverage logged, not 0.0.
- [ ] Known-pose COLMAP for the six-view layout. The old one-view `alpine_02` dataset did train in LiteGS (hold-out about 28.5 dB). Brush was not checked. The new per-view names are not yet built.
- [x] LiteGS trained the old one-view known-pose set on the 3090 without OOM after the generator had exited.
- [ ] The bake-off numbers in §8. Not run. #61 is an owner decision, not that bake-off.
- [x] WAN generator code is removed. Leftover WAN files on disk are not deleted by the app.
- [ ] No ComfyUI, no WSL, no `torchrun`, no network call at run time (DINOv2, MoGe, weights all local).

### 4.12 Do not

- Do not delete leftover WAN files on disk from the app. Git history is the only copy of the removed generator.
- Do not install or import `HY-World-2.0` (WorldMirror, HY-Pano, WorldNav).
- Do not write `panorama.png` into the WorldStereo input folder (VLM path).
- Do not stretch 480p keyframes to the HiRes size and call it HiRes.
- Do not dequantize the FP8 DiT to BF16 at load. Do not silently fall back to the BF16 file.
- Do not feed ERP frames to WorldStereo; it takes pinhole cameras.
- Do not drop SH from the splat to match HY-World.

---

## 5. Non-functional requirements

| Topic | Locked | UNKNOWN (measure, do not invent) |
|-------|--------|----------------------------------|
| VRAM | 24 GB Quality, FP8 DiT resident, text encoder offloaded | Peak per trajectory at 480×832 with retrieval |
| Speed | 4 DMD steps, CFG-free | Seconds per trajectory on the 3090. FP8 gives no matmul speed-up on Ampere. HY-World reports World Expansion at 286 s for a whole scene on multiple H20s with sequence parallel; not comparable |
| Disk | Exact filenames in the manifest | Total pack size until the spike confirms which 480P-Diffusers parts are needed (FP8 DiT alone is 20.35 GB) |
| Quality | FP8 card: “<1% perceptible on short clips, faint colour drift possible on long sequences” | Whether colour drift shows across 4 rails × 21 frames |
| Reliability | Bank persisted after each trajectory; Stop leaves the last finished trajectory valid | — |
| Security | Local only, no telemetry, no hub calls at run time | — |
| Licence | Code Apache-2.0 with NOTICE | Weights still UNKNOWN. Risk accepted for the default route 2026-10-30 (Index #61), and earlier for the experiment (Index #59). Not a settled MIT grant |

---

## 6. UX/UI references

| Artifact | Use |
|----------|-----|
| [Phase3.html #gen-ui](../docs/Votion_3DGS_2.0_Phase3.html#gen-ui) | Generate chrome; add the Generator selector and three-up split |
| [Phase3.html #recon-ui](../docs/Votion_3DGS_2.0_Phase3.html#recon-ui) | Keep / Ditch / Re-gen; add dependency warning and dataset source switch |
| [votion-plan.css](../docs/votion-plan.css) | Colours, type, drawers |

No HTML dummy exists for this track. Do not edit the Phase 3 dummy for it. If the toggle graduates, a dummy is made first.

**Flow:** Geometry (rails, Confirm; optionally views per rail) → Generate (Generator = WorldStereo, rail order, Generate) → review keyframes → Confirm → HiRes (pinhole) → Reconstruct (Keep / Ditch, Known poses) → Splat.

---

## 7. Technical constraints & dependencies

| Item | Constraint |
|------|------------|
| OS | Native Windows, Votion venv Python 3.12, CUDA 12.8 |
| Model code | `vendor/worldstereo`, pinned, Apache-2.0. Diffusers `WanTransformer3DModel` subclasses `WorldStereoModel` / `WorldStereoRefSModel`; pipeline `RefKFDMDGeneratorPipeline` (`pipeline_dmd_keyframe.py`) |
| Weights | `apozz/WorldStereo-fp8` (DiT + `config.json`); base parts reused from the Phase 3 manifest where the spike proves compatible |
| Rendering | PyTorch3D (preferred) or `engine/worldstereo/render.py` (fallback, accepted only by §4.4 check) |
| Retrieval | DINOv2, local file |
| Distributed | None. `sp_world_size=1`, `fsdp=False`. No NCCL on Windows |
| Attention | SDPA. SageAttention optional as in Phase 3; FlashAttention not required |
| HY-World code | Not used. Section 7 alignment is reimplemented from the paper text |

**Risks:**

1. PyTorch3D on Windows / Python 3.12 / cu128.
2. VRAM headroom with stitched retrieval at 480×832.
3. Weights licence still UNKNOWN. Yik accepted that risk on 2026-10-29 (Index #59). The FP8 card’s MIT sentence is not a settled licence.
4. No released panoramic preprocessing; the adapter is Votion’s guess at the training distribution (render settings, FoV, start elevation).
5. Output is ~0.4 MP pinhole; pinhole HiRes is new code and carries the sharpness.
6. FP8 upcast per matmul on Ampere may make it slower than BF16 would be on a bigger card.
7. Memory errors compound: a bad early rail is retrieved by every later rail.

---

## 8. Success metrics

Done = §4.11. Graduation from experimental needs all of these on `alpine_02`, against Phase 3 WAN on the same rails and prompt:

| Metric | Baseline source | Pass |
|--------|-----------------|------|
| Joint fit, three overlapping frames from different rails | WAN on `alpine_02` (record first; `alpine_01` was 22.3 dB) | Higher than WAN |
| Hold-out PSNR, every 8th view held out | WAN on `alpine_02` (`alpine_01` was 15.6 dB) | Higher than WAN |
| Mean reprojection error, poses fixed | WorldStereo only | Recorded |
| Peak VRAM | — | ≤ 24 GB, no OOM across four rails |
| Wall time, four rails | WAN four rails | Recorded; owner decides if acceptable |

If WorldStereo does not beat WAN on joint fit, the toggle stays experimental or is removed. HiRes coverage is logged, not gated (Phase 3 rule).

---

## 9. Timeline & milestones

No calendar dates. Order:

| Step | Meaning | Exit |
|------|---------|------|
| WS-0 | Gates G1–G3 | Certainty log entries |
| WS-1 | Environment spike (§4.1, §4.3) | One trajectory, VRAM and time in `env_report.txt` |
| WS-2 | Adapter + worker on `alpine_02`, four rails, memory across rails (§4.2, §4.5) | `retrieved.json` shows cross-rail retrieval |
| WS-3 | Known-pose dataset + LiteGS train with 480p keyframes only (§4.7, §4.8) | First joint-fit and hold-out numbers |
| WS-4 | Pinhole HiRes (§4.6), retrain | Numbers with 8K detail |
| WS-5 | UI (§4.9), manifest, Help drawer text | §4.11 all ticked |
| WS-6 | Owner decision, not the §8 bake-off | Closed 2026-10-30 (Index #61): WorldStereo is the Quality route. WAN code removed. |

WS-3 before WS-4 on purpose: if consistency does not improve at 480p, pinhole HiRes is not worth building.

---

## Open questions

1. ~~Weights licence for `hanshanxue/WorldStereo` (G2).~~ **Accepted as UNKNOWN, 2026-10-29 (Index #59).** Not resolved to MIT or to the Tencent community licence.
2. Mask polarity and default FoV in `camera.json`: read at pin.
3. Does WorldStereo’s pipeline accept Votion’s Comfy-format umT5 / CLIP-Vision / VAE, or does it need the 480P-Diffusers subfolders?
4. Exact DINOv2 checkpoint the retrieval code loads.
5. Views per rail: is one forward view enough for coverage, or does each rail need 3–4 yaw directions? Measure coverage at WS-3.
6. Does pinhole HiRes at 2× keyframe size give LiteGS enough detail, or does it need 4× (1920×3328)?
