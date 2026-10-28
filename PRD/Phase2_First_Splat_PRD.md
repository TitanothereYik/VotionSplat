# PRD — Phase 2: First splat (pano → geometry → one rail → SfM → ply)

**Product:** Votion 3DGS 2.0  
**Phase:** 2 of 4  
**Depends on:** Phase 1 `outputs/<scene>/pano.png` + MoGe weights  
**Unlocks:** Phase 3 WAN / HiRes / multi-rail  
**Graph:** [docs/260825_MICKMUMPITZ_3DGS-Dataset-Creator_1-0_SMPL.json](../docs/260825_MICKMUMPITZ_3DGS-Dataset-Creator_1-0_SMPL.json) — P2 implements **SETUP + CAMERA PLOT 1 + SphereSfM (single trajectory, no HiRes)** only  

**Sources (do not drift):**

- Dummy UI: [docs/Votion_3DGS_2.0_Phase2.html](../docs/Votion_3DGS_2.0_Phase2.html) · [live](https://titanothereyik.github.io/votion-3dgs-2-html/Votion_3DGS_2.0_Phase2.html)
- Spec: [docs/Votion_3DGS_2.0_Phase2.md](../docs/Votion_3DGS_2.0_Phase2.md)
- Interaction: [docs/rail-dummy.js](../docs/rail-dummy.js)
- SplatKit Plot Camera chrome still: `docs/media/ref-splatkit-plot-camera.png`
- Set index: [README.md](README.md)

If Phase 1 **2D Images** is unavailable, Phase 2 still runs on **360 Images** or **text** (or a fixture 360). Ostris is READY as of 2026-09-03.

---

## 1. Product overview

| Field | Value |
|-------|--------|
| Name | Votion 3DGS |
| Owner | **Yik**. MIT. Small private team. |
| Version | **2.0** only (no `x.y.z`) |
| High-level goal (this phase) | Prove the **splat path without WAN**: Compute geometry → **one** camera rail → preview mesh flight → SphereSfM → train Gaussians on LiteGS (**LiteGS** or **MCMC Compact** controller) → `splat.ply`. This is the first Unreal-validating slice. |

P2 may use a **fixture control video** (mesh reprojection only) if WAN weights are not downloaded — black holes expected. SfM/train must still run so the plumbing is real. Optional WAN hole-fill in P2 **only if** 14B weights are READY — still one rail, no HiRes.

---

## 2. Problem & objectives

**Problem:** A pano is not a 3DGS. The user must place a drone rail on MoGe geometry, export COLMAP pinholes, and train Gaussians in-app. V1 mixed WorldFM `transforms.json`; V2 uses SphereSfM → COLMAP `images/` + `sparse/0`.

**Objectives:**

1. **Compute geometry** is a real product button (not a hidden worker).
2. One rail only in P2 (Quality four-rail default is Phase 3).
3. Origin **star 0** locked at `(0,0,0)`.
4. In-app train (LiteGS backend) writes full-SH `splat.ply` with the user-picked **LiteGS** (default) or **MCMC Compact** controller.
5. If the LiteGS spike failed: Train disabled; COLMAP folder still valid for Brush.

---

## 3. Target users & use cases

Human persona job titles: UNKNOWN. Audience: small private team.

| ID | Scenario |
|----|----------|
| P2-S1 | No `pano.png` → clear error: run Phase 1. No second file picker that bypasses P1 modes |
| P2-S2 | Compute geometry → FLOOR + SIDE **side by side** + star + **visible rail gizmos** |
| P2-S3 | User drags red cameras 1–3 and the orange look-at; star cannot move |
| P2-S4 | Play mesh flight on the **Preview flight pane under both plates** (black holes OK) then Confirm |
| P2-S5 | Reconstruct → `colmap/sparse/0`; viewport is sparse cloud (not cube-face stills) |
| P2-S6 | Splat: pick LiteGS or MCMC Compact, set budget, Train → `splat.ply`; Pause / Resume; Continue same controller (optimizer state kept); switch needs Retrain |
| P2-S7 | LiteGS spike failed → Open COLMAP folder (Brush) |
| P2-S8 | Switch chips away and back → generated views remain (Panorama ERP, Geometry preview + Confirm, Splat raster) |
| P2-S9 | Background jobs update the strip under the chips (not log-only) |
| P2-S10 | Live splat viewport is rasterized 3DGS (not a point cloud); reset = rails Star; LMB look, RMB pan, wheel dolly |
| P2-S11 | Max steps any integer ≥ 1; Stop on Splat rail and job bar; MCMC Compact same Gaussian raster as LiteGS |
| P2-S12 | Splat: click a training camera to jump the viewport there; GT compare render \| ground truth; HUD shows loss / PSNR / Gaussians / it/s / fps / VRAM |
| P2-S12 | Needle takeover / glowing whites / crushed blacks as training proceeds are bugs |

### User stories (from dummy)

- As a user I can compute MoGe geometry from the Phase 1 pano.
- As a user I cannot edit rails until geometry exists.
- As a user I preview the mesh drone flight **before** Confirm (Play is on the pane under FLOOR | SIDE, not the left rail).
- As a user I see the COLMAP sparse cloud after Reconstruct, not cube-face stills.
- As a user I can leave a tab and come back without losing generated views.
- As a user I see a job progress strip while background work runs.
- As a user I choose **LiteGS** or **MCMC Compact** and a Gaussian budget before Train; I can Pause and watch the live raster while paused.
- As a user I watch a live raster of the Gaussians while training (Postshot-style), not a point cloud.
- As a user I Stop training and Continue from the last checkpoint.
- As a user I orbit from the rails Star (LMB look, RMB pan, wheel dolly) with no zoom cap.

---

## 4. Functional requirements

### 4.1 Pages this phase fills

| Chip | Dummy | Behavior |
|------|-------|----------|
| Geometry | [Phase2.html §02–03](../docs/Votion_3DGS_2.0_Phase2.html#geo) | Two states: empty then Plot Camera |
| Reconstruct | [§04](../docs/Votion_3DGS_2.0_Phase2.html#recon) | SphereSfM |
| Splat | same HTML, second desk | Train / Stop / Continue / Retrain / Export |

Generate chip in P2 still points at Phase 3 (WAN not in P2 default).

### 4.2 Compute geometry

SplatKit display name: `Plot Camera - Compute Geometry (internal)` (`SplatKit_CameraPlotSceneReference`). Depth-only: MoGe → sparse cloud + dense **FLOOR** (top-down X/Z) and **SIDE** (Z/Y) orthos. No WAN, no mesh fly-through yet. Overlay point budget in SplatKit: **4000** (min 500, max 40000).

Disabled until `pano.png` exists.

### 4.3 Plot Camera (one rail)

`SplatKit_CameraPlotRenderControlGeo`. Splines the rail and renders `control_video` + `control_mask` + `rail_json`.

**Look modes** (copy SplatKit `INPUT_TYPES`; **no** dropdown called Height; Height is **Y** of an anchor):

| Widget value | Video name | Behavior |
|---|---|---|
| `look_forward` | Look Forward | +Z follows path tangent. SplatKit node default. |
| `look_at_target` | orange look-at | Every frame aims at **one** world point. P2 default (SMPL node **27**). |
| `per_point_look` | Per Point Look | Each anchor has its own look target. |

Legacy `fixed_forward` may still render old graphs. Do not put it in the Votion dropdown.  
`look_forward` + drag a look arrow **switches that rail to `per_point_look`**.

**P2 default (node 27):** `look_at_target`, **81** frames (min 9, max 257, step 4 — must match WAN length later), `moge_level` **9** (graph; node default is 6). Integer **9** is not fps.

Look-at default: `-0.108, 0.073, 1.953` (updates when the orange handle moves).

**Default knots (xyz only):**

```
0.000, 0.000, 0.000
0.199, 0.219, 0.488
-0.184, -0.093, 0.920
-0.065, 0.039, 1.404
```

Parser: copy SplatKit `_camplot_parse_anchors_ext` — 3 numbers, 6 numbers, or JSON; ≥2 points; `#` comments; Catmull-Rom. The multiline **anchors** widget is source of truth if the 3D widget fails.

**SETUP node 41:** Dataset Project analog. Widgets `['120_studio-garden-hires', False]`. Use Votion scene name. Do not invent a name for the bool until `INPUT_TYPES`.

**Suggest paths** is dim in P2 (Phase 3). Confirm unlocks Reconstruct.

### 4.4 Viewport gizmos (locked 2026-08-31)

FLOOR and SIDE sit **side by side**. Preview flight sits **under both**. Dummy backdrops (user-supplied street — do not swap random MoGe/Matrix stills):

| Pane | File |
|------|------|
| FLOOR X → Z ↑ | `docs/media/geo-floor-top.png` |
| SIDE Z → Y ↑ | `docs/media/geo-side-elev.png` |
| Preview flight | `docs/media/geo-preview-overall.png` |

| Gizmo | Behavior |
|-------|----------|
| Cyan **★ 0** | Pano origin `(0,0,0)` — **not** draggable or deletable |
| Red **1, 2, 3…** | Draggable; other pane tracks X/Z |
| Cyan look arrows | Follow look target |
| Orange look-at | Drag in `look_at_target`; all arrows aim at it |
| **geo** / **reset view** | Overlay / pan-zoom on each pane |

**Play lives on the Preview flight pane only** — no Preview button on the left rail. Black holes OK. Then Confirm.

Frame: **+Z** into the pano, **+X** right, **+Y** up. Every drone clip starts at the star so Phase 3 can reproject 8K.

Product: prefer dense MoGe orthos of *this scene’s* pano; the three PNGs are dummy stand-ins.

Geometry viewport implementation: **QPainter** plates + gizmos on one widget (locked 2026-09-03). Rail path, star 0, red 1–3, and look arrows **must be visible**. Do not ship QWebEngine. Do not stack vispy Line/Markers over Image. Do not embed Comfy `camera_plot_geo.js`.

Switching away from Geometry and back **keeps** the last Preview-flight frame and Confirm.

### 4.5 Reconstruct

SphereSfM → COLMAP `images/` + `sparse/0` (**gnomonic** `SIMPLE_PINHOLE` cube faces; Index #57). **Not** WorldFM `transforms.json`. Left rail: Reconstruct, `num_images` / `num_points`. Viewport: interactive **sparse cloud** (`points3D.bin` + camera centers; drag orbit, wheel zoom). Not cube-face stills. Dual-res HiRes waits for Phase 3.

### 4.6 Splat train

`engine/splat/` over the vendored **LiteGS** backend (rebuilt 2026-09-07; V1 `train_splat_live.py` retired). Dummy: [Phase2.html splat desk](../docs/Votion_3DGS_2.0_Phase2.html#recon).

| Control | Locked value |
|---------|----------------|
| Backend | **LiteGS** ([MooreThreads/LiteGS](https://github.com/MooreThreads/LiteGS), `vendor/litegs` at `vendor/litegs.sha`). Inria Gaussian-Splatting licence: vendor as-is, build the three CUDA extensions locally at Setup (`tools/build_litegs.py`, pip route then CMake fallback), never redistribute binaries, licence line in Help. Train gated on `litegs raster 1-step: OK` in `docs/env_report.txt`. |
| Controller | **LiteGS** (default) or **MCMC Compact** chips |
| LiteGS | `DensityControllerTamingGS` with `target_primitives` = the Gaussian budget, `opacity_reset_mode='decay'`, `prune_mode='weight'` |
| MCMC Compact | Jung & Hong, *MCMC-Guided Compact 3DGS* (SIGGRAPH Asia 2025 TC, [doi 10.1145/3757376.3771401](https://dl.acm.org/doi/10.1145/3757376.3771401)), **section 3.1 only**, clean room: Metropolis-Hastings walk of the split threshold toward a target split ratio (proposal noise tuned to ~0.44 acceptance) + Mini-Splatting-style stochastic importance prune to the same budget. Normal-gradient term omitted (no learned normals). BRDF / normals / MALA / deferred shading out of scope. **Same Gaussian raster as LiteGS** (opacity, anisotropic scale, rotation, SH). |
| ADC | **Not offered** |
| Max steps | Any integer **≥ 1**. Box default **10000**. **No product cap.** Epoch = steps / training views, shown too. |
| Gaussian budget | Editable, default **3,000,000** (1M hides 8K gain — SplatKit HiRes doc). Real controller target, not decorative. |
| Resolution scale | 1 / 0.5 / 0.25 (downsampled training frames) |
| SH degree | 3; warm-up one band per clamp(steps / 10, 200, 1000) steps |
| Densify schedule | Step-based: ~24 cycles between 5 % and 80 % of the steps; opacity decay at most every 2000 steps |
| Hold-out eval | Toggle; every 8th SphereSfM view held out; PSNR reported each epoch (HUD + `splat_meta.json`) |
| Buttons | Train / **Pause** / **Stop** / Continue / Retrain / Export `splat.ply` / **Export compact** (opacity < 0.005 pruned, optional drop `f_rest`) / Open COLMAP folder. Stop also on the job bar (kills). Pause / Resume in-process over stdin. Continue resumes the last 100-step checkpoint **with optimizer and scheduler state**. |
| Checkpoint | `checkpoints/splat_latest.pt` format `votion-litegs-1` (params + Adam + scheduler + controller state) every 100 steps + `splat_meta.json`. Params-only checkpoints from the gsplat trainer show **Retrain required**. |
| Viewport | Live **rasterized 3DGS** (ellipses as a surface), like Postshot, over **shared memory** (`votion_splat_<pid>`, 2-slot RGB8 ring; Qt polls at 60 Hz; PNG fallback). Not a point cloud. Not stats-only. HUD: state, step / epoch, loss EMA, PSNR, Gaussians / budget, it/s, viewport fps, VRAM; loss / PSNR sparkline. |
| Camera | Unlocked. LMB look/orbit, RMB pan, wheel dolly. Zoom **not** product-capped. Reset / default = rails **Star** (first pose of `rail_00.json`). Do **not** tighten the orbit. Camera goes to the worker over stdin (`VIEW`); renders after the next step (≤ 1 step latency, 30 fps cap while training), idle view renders only on change. |
| Training cameras | List grouped by rail (hold-out marked); click jumps the viewport there; **GT compare** shows render \| ground truth. Reset returns to the Star. |
| Look | Elongated / mixed-angle ellipses are expected. Hairline needles taking over the scene, glowing whites, and crushed blacks as training proceeds are **bugs**. |
| Continue | Same checkpoint controller; switching LiteGS ↔ MCMC Compact needs Retrain |
| Export | **Unreal / MLSLabs** (default): Inria full SH, trained sRGB, prune `opacity < 0.02`, clamp anisotropy 64, no count cap. **Unreal / Luma**: DC only (Niagara ignores `f_rest`), same prune/needles, keep ≤ 1.9M so Luma does not chunk; place `_Baked`. Picker also has **3DGS / sRGB** (Brush, Postshot, SuperSplat) and **Linear**. Compact prunes opacity < 0.005 and can drop `f_rest`. |

Brush / LichtFeld / Postshot stay Open-in (Phase 4).

If the LiteGS spike failed: Train disabled; Open COLMAP folder still works.

Unreal import of `splat.ply` is a **manual** check via MLSLabsRenderer. Write pass/fail to `outputs/<scene>/unreal_check.txt`. Do not claim Unreal success in code.

### 4.7 Workers (subprocess; Qt never holds WAN/MoGe)

1. `engine.workers.moge` — Compute geometry; `pointcloud.ply` / ortho PNGs  
2. `engine.workers.camplot` — Preview / Confirm  
3. `engine.workers.sfm` — `colmap_sphere`  
4. `engine.workers.splat` — `engine.splat.trainer.SplatTrainer`; modes `train` / `continue` / `retrain` / `view` / `export` / `export_compact` / `export_compact_dc`; args `--strategy litegs|mcmc_compact --max-steps --budget --res-scale --holdout`. Stdout `SPLAT\t…` (step, loss, psnr, gaussians, its, vram, fps, state, ctl, ckpt, shm, cameras, ply, error); stdin `VIEW` / `PAUSE` / `RESUME` / `STOP`. The idle `view` worker keeps the checkpoint on GPU and re-renders on camera change only.  

Unload GPU between workers (process exit). Qt (`app.main`) never imports torch.

### 4.8 Vendor

Clone [ComfyUI-SplatKit](https://github.com/mickmumpitz/ComfyUI-SplatKit) into `vendor/splatkit` at a **pinned git SHA**, record in `vendor/splatkit.sha`. Use `core/` + `shim/` (MIT, no ComfyUI imports). Rasterizer = torch/triton shim — **not** nvdiffrast. Do not use Matrix-3D PanoLRM.

Clone [MooreThreads/LiteGS](https://github.com/MooreThreads/LiteGS) into `vendor/litegs` at a **pinned git SHA**, record in `vendor/litegs.sha`, copy `LICENSE.md` next to it (Inria Gaussian-Splatting licence — research / evaluation only). Build `simple-knn`, `fused_ssim`, `litegs_fused` locally at Setup via `engine/cuda_jit.prepare` (pip route, CMake fallback); the installer never ships these binaries.

`colmap_sphere.exe`: copy SplatKit SHA-256 check; **do not invent a hash**. Turing+ GPU; 3090 is OK per SplatKit README.

MoGe: `Ruicheng/moge-vitl` `model.pt`.

### 4.9 Outputs

```
outputs/<scene>/
  pano.png
  prompt.txt
  moge/
  rails/rail_00.json
  control/control_video.mp4
  control/control_mask.mp4   (or PNG sequence — match SplatKit)
  colmap/images/
  colmap/sparse/0/
  checkpoints/splat_latest.pt   votion-litegs-1: params + Adam + scheduler + controller state
  checkpoints/splat_meta.json   controller, step, epoch, gaussians, budget, psnr, psnr_eval, done
  _work/splat_view.json         persisted viewport camera (worker writes at exit)
  _work/splat_preview.png       PNG fallback frame
  splat.ply                     full SH (degree 3)
  splat_compact.ply             optional compact export
```

### 4.10 Acceptance criteria

- [ ] Missing `pano.png` → clear error (run Phase 1 first).
- [ ] Compute geometry disabled until pano exists; rail editor disabled until geometry exists.
- [ ] After compute: FLOOR and SIDE **side by side** with gizmos visible; origin star at `(0,0,0)` and not draggable; 2+ knots do not crash.
- [ ] Preview flight sits **under both** plates. Mesh-only flight preview plays **before** Confirm (black holes expected).
- [ ] Switching chips restores generated views (Panorama ERP, Geometry preview + Confirm).
- [ ] `colmap/sparse/0` exists after Reconstruct. Viewport shows the sparse cloud (not cube-face stills).
- [ ] Background jobs update the strip under the chips (not log-only).
- [x] LiteGS spike line in `env_report.txt`; Train gated on it. *(2026-09-07)*
- [x] If the LiteGS spike passed: full-SH `splat.ply` with the selected controller (LiteGS default); Continue same scene **and** same controller, resuming **with optimizer state**. *(2026-09-07)*
- [x] Switching LiteGS ↔ MCMC Compact disables Continue until Retrain; old gsplat params-only checkpoints show “Retrain required”. *(2026-09-07)*
- [x] Splat viewport is a live raster over shared memory (not a point cloud). Reset is the rails Star. Max steps accepts any integer ≥ 1. Pause / Resume in-process; Stop kills the train job. *(2026-09-07)*
- [x] Viewport: camera change → new frame under 50 ms idle (33 ms median); ≥ 20 fps orbit during training (32 fps); training it/s within ~20 % while orbiting. *(bench 2026-09-07)*
- [x] `alpine_01` Retrain with LiteGS to 10k steps: hold-out PSNR reported, Gaussians ≤ budget, time recorded; MCMC Compact run ends below the LiteGS count at equal budget (511k vs 2.26M at 3M; hold-out 15.7 vs 15.6 dB). *(2026-09-07; absolute PSNR limited by WAN + SphereSfM view inconsistency — Index Measured table)*
- [x] MCMC Compact uses the same per-Gaussian raster as LiteGS.
- [x] If the LiteGS spike failed: UI says so; COLMAP still valid for Brush.
- [x] `app.main` never imports torch. *(offscreen check 2026-09-07)*
- [ ] `splat.ply` / `splat_compact.ply` load in Brush (manual).
- [ ] Unreal check documented in `unreal_check.txt`.

### 4.11 Do not

- No Krea (Phase 1). No 4-rail default. No HiRes Composite 8K. No WorldFM frames.
- Do not name a look mode `forward_look` or Height. Widget is `look_forward`.
- Do not treat **9** as fps.
- Do not “fix” HiRes `moge_level` 6 in P2 (Plot Camera stays 9).
- Follow SplatKit export for OpenCV vs OpenGL; do not reuse WorldFM conversion blindly.
- Do not ship LiteGS CUDA binaries in the installer (Inria licence); do not implement the relighting half of Jung & Hong.
- Do not clamp Gaussian scales or enforce the budget by hand in the trainer (`_clamp_gaussian_scales` / `_enforce_cap` are gone); the controller owns density.

---

## 5. Non-functional requirements

| Topic | Locked | UNKNOWN |
|-------|--------|---------|
| Performance | One rail; unload GPU between workers; ~3M cap | Train time, SfM time |
| Reliability | Black-hole mesh video acceptable for plumbing; Continue/Retrain | Checkpoint format beyond `splat_latest.pt` |
| Security | Local files only | Telemetry of scenes |
| Viewport lib | **QPainter** FLOOR/SIDE (locked 2026-09-03) | — |

---

## 6. UX/UI references

| Artifact | Use |
|----------|-----|
| [Phase2.html §02](../docs/Votion_3DGS_2.0_Phase2.html#geo) | Empty geometry |
| [Phase2.html §03](../docs/Votion_3DGS_2.0_Phase2.html#plot) | After compute + one rail |
| [Phase2.html §04](../docs/Votion_3DGS_2.0_Phase2.html#recon) | Reconstruct + LiteGS / MCMC Compact splat desk (budget, Pause, HUD, camera list, GT compare) |
| [rail-dummy.js](../docs/rail-dummy.js) | Drag 1–3 + orange look-at; LiteGS / MCMC Compact chips, Pause toggle, camera list |
| `geo-floor-top.png` / `geo-side-elev.png` / `geo-preview-overall.png` | Dummy plates |
| `ref-splatkit-plot-camera.png` | Chrome to port |

---

## 7. Technical constraints & dependencies

- SplatKit `core/` + `shim/` at pinned SHA  
- MoGe ViT-L `model.pt`  
- `colmap_sphere.exe` SHA from SplatKit  
- LiteGS vendored at `vendor/litegs.sha` (Inria licence, research only); three CUDA extensions (`simple_knn`, `fused_ssim`, `litegs_fused`) built locally by `tools/build_litegs.py` via `engine/cuda_jit.prepare()` — never shipped  
- MCMC Compact = clean-room port of Jung & Hong §3.1 (no public code)  
- V1 files to **copy then adapt** (not import across WSL): `job_control.py`, `camera_presets.py` (templates later in P3), `camera_path_viz.py`, `train_splat_live.py`, `theme.css`  
- Do not copy `worldfm/` or WSL scripts  

Native WAN i2v is **not** required for P2 exit (fixture control video allowed).

---

## 8. Success metrics

Done = §4.10. No extra KPIs. One-rail black-hole splat is **not** the Quality bar (that is Phase 3). Unreal playback: manual, recorded, not a CI claim.

---

## 9. Timeline & milestones

**No calendar dates.** Order: vendor SplatKit + MoGe → Compute geometry + QPainter FLOOR/SIDE → SphereSfM COLMAP → LiteGS build → LiteGS + MCMC Compact trainer → shared-memory viewport → Splat page → ply. Trainer rebuild locked 2026-09-07.

---

## Open questions (this phase)

1. ~~Did P0 gsplat spike pass on this PC?~~ Superseded: LiteGS spike (`litegs raster 1-step: OK` in `env_report.txt`) is the Train gate; passed 2026-09-07 on this PC.
2. SplatKit git SHA to pin (do not leave floating `main`)?
3. control_mask as mp4 vs PNG sequence — match SplatKit at vendor pin; do not invent.
