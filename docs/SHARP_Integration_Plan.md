# SHARP integration plan

**Status:** plan only. This file does not change the app, the phase specs, or the locked pipeline.  
**Repo studied:** [apple-aiml-research/ml-sharp](https://github.com/apple-aiml-research/ml-sharp) (`main`, three commits at the time of this plan).  
**Paper:** [Sharp Monocular View Synthesis in Less Than a Second](https://arxiv.org/abs/2512.10685) (arXiv:2512.10685).  
**Code home this plan assumes:** `D:\Votion3DGS\`.  
**Date:** 2026-10-26.

---

## 1. Decision this plan asks for

SHARP turns **one perspective photograph** into a 3D Gaussian splat in a single forward pass. On a Sharp scene that replaces Compute geometry, Plot Camera, WAN, HiRes, SphereSfM, and the LiteGS / MCMC Compact **training loop**.

It does not turn a photo into an explorable 360 world. The network does not outpaint, does not invent the back of the room, and its own demo camera only moves a small fraction of the near depth (section 3). The slice-a-360-and-stitch experiment already recorded in [Votion_3DGS_2.0_Index.md](Votion_3DGS_2.0_Index.md) stays rejected.

Two owner choices sit in front of any code:

| Id | Choice | What it means |
|----|--------|----------------|
| **L** | Licence | The released **weights** are research-only and exclude product use. Section 4. Pick L1, L2, or L3 before Phase C. |
| **D** | Product scope | **D1** — add a Sharp path beside the current world path. **D2** — after a roam test, retire the world path. Section 5 lists what D2 deletes. D1 is the plan this document implements. D2 is a later decision, not the first commit. |

Recommended sequence: accept a licence option, build the Sharp path (D1), fly the splat in the existing viewport, and only then decide D2.

---

## 2. What the SHARP repository actually does

Facts below are taken from the published source, not from the paper’s claims alone.

### 2.1 Command

```
sharp predict -i <image or folder> -o <out> [-c sharp_2572gikvuh.pt] [--render]
sharp render  -i <ply or folder>   -o <videos>
```

Default checkpoint, downloaded by `torch.hub` on first predict if `-c` is omitted:

`https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt`

Cache location if left alone: `~/.cache/torch/hub/checkpoints/`. Votion must not use that path (section 7.4).

Prediction runs on CPU, CUDA, or MPS. `--render` / `sharp render` require CUDA and the `gsplat` package. This plan does not take that renderer (section 7.6).

### 2.2 One image in, one ply out

`src/sharp/cli/predict.py`:

1. Load RGB. Focal length comes from EXIF (`FocalLengthIn35mmFilm`, else `FocalLength`, else **30 mm**). Millimetres convert with the 35 mm film diagonal:

   `f_px = f_mm * hypot(width, height) / hypot(36, 24)`.

2. Resize the photo to **1536×1536** with bilinear sampling. The image is stretched to a square. There is no letterbox.
3. Run `RGBGaussianPredictor` in NDC, then unproject to metric Euclidean space with the resized intrinsics.
4. Write `<stem>.ply`.

A folder of images writes **one ply per file**. The repository has no fusion, no bundle adjustment, and no fine-tuning loop. `src/sharp/utils/training.py` is a tiny helper, not a trainer.

### 2.3 What the network is

`PredictorParams` defaults in `src/sharp/models/params.py`:

| Piece | Value | Consequence |
|-------|--------|-------------|
| Backbone | DINOv2-L/16, 384, built by subclassing `timm` | Two encoders (image + sliding-pyramid patch). Weights come from the single `.pt`. `create_vit` does **not** download a separate DINOv2 checkpoint. `timm` is still an import. |
| Depth layers | `num_monodepth_layers = 2` | A front surface and a second layer behind it. That second layer is the disocclusion the model can show when the camera shifts a little. |
| Gaussian init | stride **2**, `num_layers = 2` | On the 1536² canvas: `(1536 / 2)² × 2 = 1,179,648` Gaussians. Count is fixed. There is no budget, no densify, no prune. |
| Colour | predicted `linearRGB`, activation sigmoid | Export converts to sRGB degree-0 spherical harmonics so public viewers that skip gamma still look roughly right. |
| Scale | metric, absolute | Camera motion is in metres. Scene content sits roughly at `(0, 0, +z)`. |

### 2.4 Coordinate frame and ply layout

README and `unproject_gaussians`: **OpenCV**. +X right, +Y down, +Z forward. The input camera is the identity extrinsic. That matches the Votion splat viewport (`engine/splat/camera.py`: COLMAP / OpenCV, +Z forward, +Y down).

`save_ply` writes an Inria-style **vertex** element:

`x y z  f_dc_0 f_dc_1 f_dc_2  opacity  scale_0 scale_1 scale_2  rot_0 rot_1 rot_2 rot_3`

Opacity is an inverse-sigmoid logit. Scale is a log. There is **no** `nx ny nz`, **no** `f_rest_*` (no view-dependent colour). Extra elements sit beside `vertex`: `extrinsic`, `intrinsic`, `image_size`, `frame`, `disparity` (10th and 90th percentile of 1/z), `color_space`, `version` (1.5.0). A loader that reads the `vertex` element by property name can ignore them. A loader that requires `f_rest` or normals needs the adapter in section 7.5.

### 2.5 How far the camera is allowed to move

`TrajectoryParams` in `src/sharp/utils/camera.py`, used by `sharp render`:

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `type` | `rotate_forward` | Small orbit plus a slight push in. |
| `max_disparity` | `0.08` | Lateral offset ≈ `0.08 × image diagonal in focal units × near depth`. |
| `max_zoom` | `0.15` | Forward offset ≈ 15 % of the near depth. |
| `num_steps` | 60 | One demo loop. |
| `lookat_mode` | `point` | Looks at a point down the optical axis, at least 2 m away. |

That trajectory is the authors’ own statement of the working volume: nearby views of the photographed frustum. Walking through a wall, orbiting a building, or turning 180° is outside it. The second Gaussian layer covers a thin band of disocclusion inside that volume. It is not a video inpainter.

### 2.6 Dependencies declared by the repo

`pyproject.toml`: `click`, `gsplat`, `imageio[ffmpeg]`, `matplotlib`, `pillow-heif`, `plyfile`, `scipy`, `timm`, `torch`, `torchvision`. Recommended environment: **Python 3.13**. Ruff’s target is `py39`, and the syntax used is valid on **Python 3.12**. Votion stays on the locked 3.12 + CUDA 12.8 venv (section 7.3). A second conda env is not part of this plan.

Predict needs `torch`, `timm`, `numpy`, `pillow`, `plyfile`. `gsplat`, `imageio`, `matplotlib`, and `pillow-heif` exist for the demo video and HEIC. This plan does not add them.

---

## 3. Where it sits in Votion

Current locked path, from the index:

```
2D / 360 / text
  → Krea 8K ERP
  → MoGe geometry + Plot Camera rails
  → WAN 720p hole fill
  → HiRes reproject of the 8K ERP
  → SphereSfM + gnomonic cube faces
  → LiteGS or MCMC Compact training
  → splat.ply (SH degree 3)
```

That path exists so a person can **leave the photographed view**. WAN invents the holes. HiRes pastes the 8K texture back. Training reconciles hundreds of views into one splat with view-dependent colour. Measured on `alpine_01`: the raster and the optimizer are sound, and the multi-view PSNR stays low because the generated views disagree (index, measured table).

SHARP skips that reconciliation by never generating the other views.

```
2D photograph + horizontal FOV
  → one forward pass (~1 s after the checkpoint is loaded)
  → ~1.18 M Gaussians, degree-0 colour, metric OpenCV
  → existing Splat viewport + existing ply export
```

| Votion input | Sharp path |
|--------------|------------|
| **2D Images** drop | This is the input. Use the photo itself, with the page `h_fov` (default 70, range 10–170). |
| **360 Images** | No. An equirectangular image is not a pinhole. Slicing it into faces and running SHARP per face is the experiment the index already rejected (seams, no off-axis detail). |
| **Text → ERP** | No single photographed frustum. Text worlds stay on the Krea → WAN path. |

`h_fov` becomes a focal length, not a green warp for Krea:

```
f_px = (width / 2) / tan(radians(h_fov) / 2)
```

Pass that `f_px` into `predict_image`. Do not let the worker fall through to SHARP’s 30 mm default when the page already has 70°. 30 mm on 35 mm film is about 62° horizontal, which is a different picture from the slider default. Log the `h_fov` and `f_px` that were used.

Do not pre-resize to 1536. `predict_image` already stretches to 1536×1536 and scales `fx` and `fy` independently. Letterboxing would feed the network a different image from the one it was trained on.

### 3.1 What the existing viewport can show without a new renderer

The Splat page already rasters Gaussians through LiteGS over shared memory (`engine/splat/viewport.py`, `app/widgets/splat_view.py`). OpenCV matches. A Sharp ply can be loaded into that raster **if** a view-only worker supplies the LiteGS Gaussian layout (section 7.5) and the default camera is the **input camera at the origin**, looking +Z.

`star_w2c` today looks for `rails/rail_00.json`, and if that file is missing it places the eye on the Gaussian centroid. For a Sharp scene the centroid is in the middle of the photographed volume, so that fallback would start the user inside the geometry. The Sharp worker writes an identity pose and the viewport uses it (section 7.5).

WASD fly still works. Past the small volume in section 2.5 the image will smear and reveal empty space. The Help drawer on a Sharp scene says so in one sentence: the splat is the photographed view plus a small parallax, in metres.

### 3.2 What SHARP does not give the Unreal export

Locked export (`engine/splat/ply_export.py`) writes SH degree 3 so MLSLabsRenderer can shade view-dependent colour. SHARP has degree 0 only. Reflections will not change with the viewing angle. The Luma export path already drops `f_rest` and is the closer match. Both export buttons can still run: MLSLabs gets zeros in the higher bands, Luma gets DC. The Splat Help line for a Sharp scene states that colour is view-independent.

The Gaussian count (~1.18 M) is under the Luma cap (`LUMA_MAX_GAUSSIANS`) and under the 3 M training budget. The budget box does not apply.

---

## 4. Licence gate

Two different Apple licences ship in the repo. The index currently folds both into “non-commercial.” They are not the same, and the weights are the blocker.

### 4.1 Code — `LICENSE`

Personal, non-exclusive copyright licence to use, modify, and redistribute the **source**, with the notice kept when redistributing unmodified. Apple’s name and marks may not be used to promote a derived product. No patent licence is granted. This is not MIT. Votion’s MIT licence cannot be pasted on top of these files.

Practical consequence: vendor the tree under `vendor/sharp/` with `LICENSE`, `LICENSE_MODEL`, and `ACKNOWLEDGEMENTS` beside it. Votion code that only calls it stays MIT. The Help drawer names the Apple software licence the same way it already names the Inria licence on LiteGS.

### 4.2 Weights — `LICENSE_MODEL`

The checkpoint is an “Apple Machine Learning Research Model.” The grant is limited to **Research Purposes**, defined as non-commercial scientific research and academic development. The licence then says, in terms:

> “Research Purposes” does not include any commercial exploitation, **product development** or **use in any commercial product or service**.

Derivatives (fine-tunes, architectural changes) are under the same limit. Redistribution requires the agreement text plus the attribution sentence, and derivatives must be labeled as derivatives.

Votion 2.0 is a product: Inno Setup, a named app, a private team audience, an MIT licence (index certainty log, items 24 and 26; PRD “What is not in 2.0”). Wiring this checkpoint into that app is product use under the sentence above, even if the team does not sell seats.

### 4.3 Options

| Id | Scope | Allowed to start |
|----|--------|------------------|
| **L1** | Offline research spike. A script on this machine, weights downloaded by the person running it, ply opened in the existing viewer or SuperSplat. Not a stage chip, not in the installer, not in `model_manifest.json`. | Phase B only. |
| **L2** | In-app path for this private team. Weights are **not** bundled. The user downloads `sharp_2572gikvuh.pt` into `models\sharp\`. Help shows the research-model sentence and the attribution line. The MIT app does not claim the weights are MIT. | Phase C onward, only after the owner writes down that they accept the product-use conflict personally. This document does not clear it. |
| **L3** | Written permission from Apple that covers Votion 2.0, then the checkpoint may be an ordinary model row and may be considered for the installer. | Phase C onward. |

No phase past B starts on a silent default. If the owner picks neither L2 nor L3, the plan stops at the spike and the world path stays the product.

### 4.4 Third-party code inside the vendor tree

`ACKNOWLEDGEMENTS` carries the Apache-2.0 notice for `timm` (Ross Wightman). Keep that file in the vendor folder. `gsplat` is a pip dependency of their demo renderer; this plan does not vendor it and does not add it to the Votion venv.

---

## 5. What becomes unnecessary

“Unnecessary” here means **not executed for a Sharp scene**. Deleting the module from the repository is **D2**, and only after the roam test in Phase F. Until that decision the world path keeps every file below.

### 5.1 Skipped on every Sharp scene (D1)

These stages are the dataset factory. SHARP replaces their output (a set of posed images) with the ply.

| Stage | Why it is idle | Code that implements it |
|-------|----------------|-------------------------|
| Krea 2 ERP, seam inpaint, 8K upscale | SHARP consumes the perspective photo. `pano.png` is the wrong image. | `engine/pano/` (`krea_runtime.py`, `krea2_ref.py`, `warp.py`, `harmonize.py`, `roll.py`, `aspect.py`, `noise.py`, `cas.py`, `comfy_bootstrap.py`), `engine/workers/pano.py` |
| Florence-2 caption and the forced `no peoples, no cars` suffix | Those strings exist to prompt the outpaint. | `engine/workers/caption.py` |
| Panorama seeds, upscaler prompt, live green warp as a *generate* step | `h_fov` survives, as the focal length. The warp preview is optional and unused by predict. | `app/windows/panorama_page.py` generate controls; `engine/pano/warp_preview.py` |
| Compute geometry (MoGe) | Depth is inside the SHARP checkpoint (two-layer monodepth). A separate MoGe cloud is not an input. | `engine/workers/moge.py`, `vendor/moge/`, `engine/geo/scene_ref.py`, `engine/pano/scene_io.py` |
| Plot Camera, rails, look modes, FLOOR / SIDE editor, mesh-flight preview | There is no drone path. The only camera SHARP knows is the photo’s own. | `engine/geo/rails.py`, `engine/geo/rail.py`, `engine/geo/anchors.py`, `engine/geo/suggest.py`, `engine/geo/ply.py`, `engine/workers/camplot.py`, `app/windows/geometry_page.py`, `app/widgets/geo_view.py`, SplatKit plot-camera core under `vendor/splatkit/` |
| WAN 2.1 14B, pano LoRA, LightX2V, masked condition | Nothing to inpaint. The second Gaussian layer is the only hole fill, and it is not generative. | `engine/video/`, `engine/workers/wan.py`, `app/windows/generate_page.py` |
| HiRes composite (8K reproject into WAN holes) | No WAN frames and no 8K ERP to reproject. Sharpness is whatever the 1536² network kept from the source photo. | `engine/workers/hires.py` |
| SphereSfM, `colmap_sphere.exe`, gnomonic cube remap, dual-res dataset | No multi-view solve. The index #57 bow fix exists because ERP cube faces were bent; SHARP never makes those faces. | `engine/workers/sfm.py`, `engine/sfm/`, `engine/splat/colmap_io.py`, `app/windows/reconstruct_page.py`, `app/widgets/sparse_view.py` |
| Keep / Ditch / Re-gen rail | No rails. | Reconstruct page, `dataset.json` rail rows |
| LiteGS **training** and MCMC Compact | The ply is the network output. Densify, prune, SH warm-up, hold-out PSNR, the step budget, Continue, and Retrain have no signal to optimise. | `engine/splat/trainer.py`, `engine/splat/controllers/`, `engine/workers/splat.py` train / continue / retrain modes |
| Gaussian budget, max steps, resolution scale, hold-out toggle, controller chips | Count is fixed at ~1.18 M. Resolution is fixed at 1536² inside predict. | `app/windows/splat_page.py` train controls, `engine/splat/defaults.py` schedule constants |
| Training-camera list and GT compare | There is one camera: the input view. A hold-out view does not exist. | `app/windows/splat_page.py` camera list, `tools/render_gt_compare.py` |
| SfM scale clamps and far-point culling | Those guards exist because panoramic triangulation invents distant junk (`SCALE_CLAMP_*`, `INIT_POINT_MAX_CAM_DIST_MULT` in `engine/splat/defaults.py`). SHARP’s metric depth does not go through that solver. | `engine/splat/backend/litegs_backend.py` clamp / init-cull paths, `tools/diag_scale_stats.py` |
| Train watchdog | No long train. | `tools/train_watchdog.py` |

Home weight rows that a Sharp-only session never reads: Krea 2, both Krea LoRAs, Qwen3-VL, Wan VAE, RealESRGAN, Florence-2, MoGe, Wan 2.1 14B, pano video LoRA, LightX2V, umT5, CLIP-Vision, 4x-UltraSharp, and the SphereSfM binary. The Quality pack in the index is on the order of 45–60 GB. The Sharp checkpoint is one file. Its byte size is recorded on first download; it is not invented here.

### 5.2 Deleted from the product only if D2 is accepted

D2 means Votion no longer builds walkable 360 worlds. Text → world, 360-image import, four-rail datasets, and view-dependent reflections leave with it. The six stage chips collapse toward Home, a thin image page (drop + `h_fov`), and Splat.

Under D2 the files in the table above can be removed, plus:

- Phase 1 / 2 / 3 model rows in `tools/model_manifest.json` and the matching Home status cells.
- `vendor/splatkit/` camera-plot and SphereSfM shims, `vendor/moge/`, `bin/colmap_sphere.exe`.
- The Inria **training** licence surface (densify controllers). The LiteGS **raster** stays if the viewport still uses it, and the Help line for that raster stays with it.
- Locked behaviours that only make sense for a trained multi-view splat: step schedule (index #56), gnomonic remap (index #57), hold-out PSNR (#54), shared training-camera GT compare, rail Star as the default pose when the star was the pano centre. The Sharp default pose is the photo camera, which happens to sit at the origin too, so the Star *button* can remain as “reset to the input camera.”

D2 also drops the reason the world path was built: roam, invented rooms, and view-dependent reflections (index, “Hard rules from the Mickmumpitz video”). Do not take D2 because the training loop is slow. Take it only if the Phase F roam test is the product the team actually wants.

### 5.3 Still required on a Sharp scene

| Piece | Why it stays |
|-------|----------------|
| PySide6 shell, six chips, job strip, Log, Help, scene folder, tab persistence | The splat still has to be launched and looked at. Chips for Geometry / Generate / Reconstruct stay in the window under D1; they show an empty-state line (“This scene was built with Sharp”) instead of running. |
| 2D image drop and the `h_fov` slider | They are the Sharp inputs. |
| Job bus (`engine/job.py`) and subprocess isolation | Qt still must not `import torch`. |
| Splat viewport, shared-memory ring, LMB / RMB / wheel / WASD | This is the viewer. |
| LiteGS **raster** (`vendor/litegs/`, `engine/splat/backend/litegs_backend.py` render path, `tools/build_litegs.py`) | Already built on this machine and already speaks OpenCV. Reuse it for display. Setup’s “litegs raster 1-step” gate stays, because the viewport uses that raster. |
| `splat.ply` export, compact export, Open in Brush / LichtFeld / Postshot, Unreal 5.5 slot | Same delivery. Colour is degree 0 (section 3.2). |
| `models\` under the install root, per-file download bars | One new row, not the user-profile torch cache. |

---

## 6. Explicit non-goals

These are the ways this integration fails if they are “just tried”:

1. **Do not slice `pano.png` or an ERP into cube faces, run SHARP, and merge the plys.** Concatenating plys stacks disagreeing surfaces. The index rejection stands. UniSHARP is a different project and is not a fallback here.
2. **Do not fine-tune the checkpoint** on WAN frames or on alpine COLMAP. The repo has no training entry point, and a fine-tune is a Model Derivative under `LICENSE_MODEL`.
3. **Do not add `gsplat` to the Votion venv** to call `sharp render`. The product viewport already orbits. `gsplat` was the wheel question that moved the trainer to LiteGS.
4. **Do not change the 1536² internal resolution**, run half precision, or quantise the `.pt` as part of the first integration. Measure VRAM first (Phase B).
5. **Do not feed SHARP a MoGe depth map** to “improve” it. The released predictor computes its own two-layer depth.
6. **Do not run the LiteGS optimiser on the Sharp ply** to “add SH” or chase the 3 M budget. A few hundred Adam steps will not invent unseen rooms, and they will wash the feed-forward geometry. A later experiment can be a separate note. It is not this plan.
7. **Do not point `torch.hub` at `C:\Users\<user>\.cache`.** Index item 29.

---

## 7. Implementation design (after L2 or L3)

D1 only. The world path’s workers, pages, and manifests stay.

### 7.1 Vendor

```
vendor/sharp/          checkout of ml-sharp, import root = vendor/sharp/src
vendor/sharp.sha       full commit recorded at pin time
vendor/sharp/LICENSE
vendor/sharp/LICENSE_MODEL
vendor/sharp/ACKNOWLEDGEMENTS
```

Pin a commit. Do not track `main`. Do not pip-install the `sharp` distribution into the venv (that would pull `gsplat`). The worker adds `vendor/sharp/src` to `sys.path` and calls `sharp.cli.predict.predict_image` and `sharp.utils.gaussians.save_ply`.

Call the library. Do not shell out to the `sharp` console script. The worker still needs a progress line the job strip understands; the CLI does not emit `PROGRESS`.

### 7.2 New modules

```
engine/sharp/focal.py       h_fov + image size → f_px
engine/sharp/adapt.py       Gaussians3D or SHARP ply → LiteGS unclustered numpy + Inria ply
engine/sharp/view.py        view-only raster loop (no optimiser, no densify)
engine/workers/sharp.py     python -m engine.workers.sharp
```

`engine/job.py` gains `sharp_argv(scene)` next to `splat_argv`. `MainWindow` starts it the same way it starts pano and splat jobs: one subprocess, `PROGRESS` lines, Stop kills the process.

Worker arguments:

```
--scene <name>
--image <path>          the dropped 2D file, copied to outputs/<scene>/sharp/input.png
--h-fov <float>         page value, default 70
--checkpoint <path>     models/sharp/sharp_2572gikvuh.pt
--device cuda:0
```

The worker refuses to run if the checkpoint is missing, and the job strip says to download it. It refuses an image whose aspect is an ERP (width ≈ 2 × height) with a log line that this path wants the perspective photo. That guard is there so a 360 file dropped on the panorama page cannot silently be stretched through the network.

### 7.3 Environment

Stay on the existing Python 3.12 venv and the locked cu128 torch. Phase B is also the import spike: `timm` and `plyfile` on that interpreter. Add only what the import of `sharp.models` and `sharp.utils.gaussians` actually needs. Record versions in `requirements.lock` the same way Phase 0 recorded torch. Do not create the conda env from the SHARP README. Do not move the app to Python 3.13.

`timm`’s `VisionTransformer` constructor is the compatibility risk (SHARP subclasses it and reaches into `patch_embed`, `blocks`, `num_prefix_tokens`). Pin the `timm` version that Phase B proves, and do not float it.

### 7.4 Weights

New manifest row, same downloader as the other models:

| Field | Value |
|-------|--------|
| id | `sharp` |
| filename | `sharp_2572gikvuh.pt` |
| local path | `models/sharp/sharp_2572gikvuh.pt` |
| url | `https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt` |
| sets | `sharp` (not `p1`, not `all`, until the owner says the world pack and the Sharp file download together) |

Download with the existing byte-progress cell. After the first successful file, write the measured size and a sha256 into the manifest. Do not call `torch.hub.load_state_dict_from_url` (it writes the user-profile hub cache). `torch.load(path, weights_only=True)` from `models\sharp\`.

Help attribution, shown whenever a Sharp scene is open:

> Apple Machine Learning Research Model is licensed under the Apple Machine Learning Research Model License Agreement.

### 7.5 Scene files and the ply adapter

```
outputs/<scene>/sharp/input.png
outputs/<scene>/sharp/meta.json
outputs/<scene>/sharp/gaussians.ply     raw save_ply output, kept for diff against upstream
outputs/<scene>/sharp/camera.json       4×4 identity world-to-camera, OpenCV
outputs/<scene>/splat.ply               adapter output, what Export and the viewport load
outputs/<scene>/checkpoints/splat_meta.json
```

`meta.json` fields: `source=sharp`, `h_fov`, `f_px`, `width`, `height`, `gaussian_count`, `seconds_load`, `seconds_infer`, `peak_vram_mb`, `checkpoint_sha256`, `vendor_sha`, `color_space=srgb`, `sh_degree=0`.

`splat_meta.json` uses a new strategy key `sharp` so the Splat page can tell it apart from `litegs` and `mcmc_compact`. Continue / Retrain on this meta shows the same kind of blocked state the page already uses for a foreign checkpoint (“Sharp scene — nothing to train”), and does not offer to run Adam.

Adapter rules:

- Read positions, `f_dc_*`, opacity logits, log-scales, and `rot_*` from the SHARP vertex element.
- Confirm quaternion order on the Phase B still: SHARP’s renderer and Inria both store `rot_0` as the first component. One LiteGS frame of the input view must match the source photo before the adapter is trusted. If it is mirrored or rotated 180°, fix the order in `adapt.py` and record the finding in `meta.json`. Do not guess in the plan’s code.
- Write `f_rest` as zeros (45 floats) so the existing full-SH writer and MLSLabs importer keep their property layout. Values stay zero. Do not run `bake_sh_color`. SHARP already stored sRGB in `f_dc` via `(srgb - 0.5) / SH_C0`, which is the same DC evaluation LiteGS uses (`ply_export.SH_C0`).
- Write `nx ny nz` as zeros if the Inria writer requires them.
- Leave positions in metres. Do not recenter on the centroid and do not scale to the alpine `norm_radius`.
- Copy opacity and scale logits through. Do not run `clamp_runaway_scale` or the far-point cull on this load.
- Viewport base pose is `sharp/camera.json` (identity), not `star_w2c`’s centroid fallback. Reset / Star returns there.
- The view worker is the existing raster and the existing shared-memory protocol (`SPLAT\tshm\t…`, stdin `VIEW`). It does not construct an optimiser and it does not read COLMAP.

### 7.6 UI

**Panorama, 2D Images mode.** Add one button, **Sharp splat**, beside the existing Generate. It is enabled when a 2D file is loaded and `models/sharp/sharp_2572gikvuh.pt` is present. It reads `h_fov` from the slider. It does not require `pano.png`, a caption, or a seed. Clicking it starts `engine.workers.sharp` and, on success, navigates to Splat.

360 Images and text keep their current Generate. They do not show a working Sharp splat button. A short hint under the button on those modes: Sharp uses the perspective photo.

**Splat page, when `splat_meta.json` says `sharp`.**

| Control | State |
|---------|--------|
| LiteGS / MCMC Compact, steps, budget, resolution, hold-out, Train, Pause, Continue, Retrain | Disabled. Hint: feed-forward splat, 1,179,648 Gaussians, degree-0 colour. |
| Viewport, Star reset, Export, Export compact, Open in… | Active. |
| Training cameras, GT compare | Hidden. One line replaces them: input camera, `h_fov`, infer seconds. |
| Help | Research-model attribution, nearby-view limit, view-independent colour, Apple software licence on the vendor tree, Inria line kept because the raster is still LiteGS. |

**Geometry, Generate, Reconstruct, when the scene has `sharp/meta.json` and no rails.** Empty state, one line, no job started. The world-path buttons stay for scenes that have a pano. A scene may contain both a finished world splat and a Sharp splat only if the files do not share `splat.ply`. Rule: **Sharp splat writes `splat.ply` and refuses to overwrite it when `splat_meta.json` exists with strategy `litegs` or `mcmc_compact`, unless the user confirms replace.** The world trainers get the same guard in reverse. One scene, one active `splat.ply`.

**Home.** One model cell, “Sharp”, with the same per-file progress bar as the others (index item 30). It is in the Sharp download set, not in the P1 pack.

**Job strip.** `PROGRESS` at three marks: checkpoint loaded, inference finished, ply written. Stop kills the process. There is no Pause: the forward pass is one call.

### 7.7 Viewport camera on a metric scene

`orbit_w2c` dollies in units of “reach” (distance from the eye to the scene center). With the eye at the origin and the cloud in front of it, reach is on the order of the median depth in metres. Wheel and WASD already use that, so a room and a landscape both stay controllable. Phase E checks a close subject and a far landscape and only then changes `VIEWPORT_FLY_SPEED` if the metres feel wrong. Do not special-case a scale factor before that check.

The first frame must be the input photograph, as close as degree-0 Gaussians can get. That is the acceptance still in Phase E. Yaw and fly come after that still matches.

---

## 8. Phases

Each phase ends before the next one starts. No phase edits the world-path trainers.

### Phase A — Owner decision (no code)

- Record L1, L2, or L3 in a line at the top of this file (or in the certainty log when the owner is ready to amend it).
- Record D1 as the implementation scope. D2 stays open.
- If the choice is L1, only Phase B runs.

### Phase B — Spike on this PC (script, not a page)

Machine: RTX 3090 24 GB, Python 3.12.10, existing venv. Checkpoint saved under `models\sharp\`, not the user profile.

1. Import `sharp.models` on 3.12. Note the `timm` version that works.
2. Run predict on one real 2D photo used by this project (the alpine source still, or the panorama-page drop). Pass `h_fov` 70 as `f_px`, and also run once with SHARP’s 30 mm fallback, so the two focals can be compared.
3. Record wall time split into load vs inference, peak VRAM, gaussian count, ply byte size, sha256 of the checkpoint.
4. Render the identity camera with the **existing** LiteGS raster (a twenty-line script is enough). Compare to the source photo: orientation, vertical flip, FOV crop, brightness.
5. Orbit once by about the demo volume (`max_disparity` 0.08, `max_zoom` 0.15) and once by a rail-length dolly (the Phase 2 default knots go out past 1 m). Save both stills under `outputs/<scene>/sharp/_spike/`. The second still is the picture of what D2 gives up.
6. Open the raw ply in one external tool the project already targets (Brush or SuperSplat) and note whether the extra ply elements or the missing `f_rest` are rejected.

Failure here stops the plan. Likely failures worth catching before any UI: `timm` API drift, VRAM over 24 GB, a quaternion or Y-flip mismatch, an external tool refusing the ply.

### Phase C — Worker and files

After L2 or L3, and after Phase B’s identity still matches.

- Vendor pin, `vendor/sharp.sha`.
- `engine/workers/sharp.py`, `engine/sharp/focal.py`, `engine/sharp/adapt.py`.
- Manifest row and Home cell.
- Scene layout in section 7.5.
- Overwrite guard on `splat.ply`.
- Unit check: `f_px` for width 1536 and `h_fov` 70 matches the formula in section 3. Gaussian count on a tiny solid-colour image is 1,179,648 (the count does not depend on image content).

### Phase D — View-only splat

- `engine/sharp/view.py` publishes the shared-memory ring and honours `VIEW` / `STOP`.
- Splat page branch on strategy `sharp` (section 7.6).
- Star reset reads `sharp/camera.json`.
- Geometry / Generate / Reconstruct empty states when the scene is Sharp-only.

### Phase E — Button and exports

- **Sharp splat** on the 2D panorama rail.
- Export `splat.ply` and compact through the existing writer. Confirm MLSLabs-layout properties (zeros in `f_rest`) and a Luma DC file.
- Open-in paths pointed at the adapted ply.
- Acceptance still: identity view vs source photo, on the real Splat page, not only the spike script.
- Close-up photo and a deep landscape, fly speed left alone unless the metres are unusable.
- Help text: attribution, nearby-view limit, degree-0 colour.

### Phase F — Roam test, then D2 or not

Same photo through both paths, looked at in the Splat viewport:

| Move | World path | Sharp path, expected |
|------|------------|----------------------|
| Input framing | As sharp as the 8K reproject | Close to the source photo |
| Small orbit, a few percent of depth | Stable | Stable, this is the demo volume |
| Dolly of about a metre (Phase 2 default rail) | A trained compromise across WAN frames | Holes and smear; record the still |
| Turn past the photo’s frustum | WAN-invented content, with the known inconsistency | Empty |
| Specular (pond, glass, screen) | SH degree 3 can keep a reflection | Colour stays pasted on the surface |

The owner looks at that table and either keeps D1 (both paths) or accepts D2 and schedules the deletions in section 5.2 as a separate change. This plan does not perform those deletions.

---

## 9. Acceptance checks

Phase B

- [ ] Predict runs in the 3.12 venv on the 3090.
- [ ] Checkpoint file lives under `models\sharp\`, sha256 written down.
- [ ] Load time, infer time, and peak VRAM written into the spike note.
- [ ] Identity LiteGS still matches the photo (no flip, FOV in the right neighbourhood at `h_fov` 70).
- [ ] A rail-length dolly still is saved, so the roam limit is a picture rather than a sentence.
- [ ] One external viewer opens the ply.

Phase E, D1 shipped

- [ ] 2D drop → Sharp splat → Splat page, without MoGe, WAN, SphereSfM, or a training step.
- [ ] 360 and text modes do not start the Sharp worker.
- [ ] An existing `litegs` / `mcmc_compact` `splat.ply` is not overwritten without confirmation.
- [ ] Stop on the job strip kills the worker.
- [ ] Qt process does not import `torch` or `sharp`.
- [ ] Help shows the model-licence attribution.
- [ ] Export writes a ply Brush can open and a ply with the MLSLabs property set (`f_rest` present, values 0).
- [ ] World path on a second scene still trains. This change must not be the one that breaks alpine.

Phase F

- [ ] The four-move table above is filled with stills from this machine.
- [ ] D1 or D2 is written down. D2 does not start inside the Phase E change.

---

## 10. Risks

| Risk | What to do |
|------|------------|
| Weights licence vs a shipping app | Phase A. Do not bury this under the technical work. |
| `timm` and the `VisionTransformer` subclass drift | Pin the version Phase B used. |
| First click is much slower than “less than a second” | That figure is inference after the `.pt` is loaded. The one-shot worker pays load every time. Record both numbers. A warm worker is a later optimisation, not a Phase C requirement. |
| 1536² stretch of a very wide or very tall photo | Official behaviour. The identity still will show it. If it is unacceptable, the fix is a crop on the panorama page before predict, not a change to the network’s input size. |
| Degree-0 colour looks flat next to a trained alpine splat | Expected. Section 3.2. It is a reason to keep D1, not a bug in the adapter. |
| LiteGS raster licence stays in the viewer | Honest Help line. Replacing the raster with something else is outside this plan. |
| Apple marks | The button can say “Sharp splat” as a description of the technique. The About / Help text does not use Apple logos or imply Apple ships Votion. |
| Checkpoint URL moves | The manifest row is the one place the URL lives. A failed download is a normal model-cell error, same as Krea. |

---

## 11. Source map

| Question | File in ml-sharp |
|----------|------------------|
| CLI and 1536² resize | `src/sharp/cli/predict.py` |
| Demo orbit limits | `src/sharp/utils/camera.py` (`TrajectoryParams`) |
| Ply, OpenCV unproject, sRGB export | `src/sharp/utils/gaussians.py` |
| EXIF focal length, 30 mm fallback | `src/sharp/utils/io.py` |
| Layer count, stride, DINOv2-L preset | `src/sharp/models/params.py` |
| No pretrained download at init | `src/sharp/models/encoders/vit_encoder.py` (`TimmViT(config)` only) |
| CUDA demo renderer (not used) | `src/sharp/cli/render.py`, `src/sharp/utils/gsplat.py` |
| Code licence / model licence | `LICENSE`, `LICENSE_MODEL` |

Votion files a later implementation touches, and only these:

| File | Change |
|------|--------|
| `vendor/sharp/**`, `vendor/sharp.sha` | New pin. |
| `engine/sharp/**`, `engine/workers/sharp.py` | New. |
| `engine/job.py` | `sharp_argv`. |
| `engine/splat/camera.py` | Base pose may be `sharp/camera.json`. |
| `app/windows/main_window.py` | Start the worker, route `PROGRESS`, Home cell. |
| `app/windows/panorama_page.py` | One button on 2D Images. |
| `app/windows/splat_page.py` | Disabled train controls when strategy is `sharp`. |
| `app/windows/geometry_page.py`, `generate_page.py`, `reconstruct_page.py` | Empty state when the scene is Sharp-only. |
| `tools/model_manifest.json` | One row. |

Left untouched until a separate D2 change: `engine/splat/trainer.py`, `engine/splat/controllers/`, `engine/workers/wan.py`, `engine/workers/hires.py`, `engine/workers/sfm.py`, `engine/workers/moge.py`, `engine/sfm/cubic_reproject.py`, the Krea / WAN model rows.
