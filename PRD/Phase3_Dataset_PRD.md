# PRD — Phase 3: Dataset (WAN, 8K HiRes, multi-rail SfM)

**Product:** Votion 3DGS 2.0  
**Phase:** 3 of 4  
**Depends on:** Phase 2 plumbing + Phase 1 8K `pano.png`  
**Unlocks:** Phase 4 productize  
**Graph:** [docs/260825_MICKMUMPITZ_3DGS-Dataset-Creator_1-0_SMPL.json](../docs/260825_MICKMUMPITZ_3DGS-Dataset-Creator_1-0_SMPL.json) (`last_node` 175)  

**Sources (do not drift):**

- Dummy UI: [docs/Votion_3DGS_2.0_Phase3.html](../docs/Votion_3DGS_2.0_Phase3.html) · [live](https://titanothereyik.github.io/votion-3dgs-2-html/Votion_3DGS_2.0_Phase3.html)
- Spec: [docs/Votion_3DGS_2.0_Phase3.md](../docs/Votion_3DGS_2.0_Phase3.md) — **copy knot lists, HiRes widgets, SphereSfM widgets from there; do not paraphrase numbers**
- Video rules: [Org_Idea.md](../docs/Org_Idea.md) · [YouTube](https://www.youtube.com/watch?v=eJuYBNrD8HI) 00:05:00 / 00:14:55
- HiRes known limits: [SplatKit HIRES_COMPOSITE.md](https://github.com/mickmumpitz/ComfyUI-SplatKit/blob/main/docs/HIRES_COMPOSITE.md)
- Set index: [README.md](README.md)

This is the quality leap. WAN stays **1440×720**. Sharpness is the **8K reprojection trick**, not a bigger video model and not SeedVR.

**Superseded for the Quality route (Index #61, 2026-10-30).** Generate is WorldStereo memory-DMD. See [WorldStereo2_MemoryDMD_FP8_PRD.md](WorldStereo2_MemoryDMD_FP8_PRD.md). The graph numbers below stay as the copied source.

**Scene depth (2026-11-11).** Geometry, Plot Camera, the guide, and HiRes read one cache, `outputs/<scene>/depth/`. The default backend is MoGe-2, picked by Yik from the alpine bake-off. A rail plotted on an older depth must be re-plotted before Generate. The key is `rails/rail_XX.depth_key`, which Start over keeps.

**Geometry editor (2026-11-12).** Dragging a point and switching Active rail update the floor and side views only. They do not rebuild Generate or Reconstruct. Play writes the rail file and starts the mesh flight. Confirm rails refreshes the other pages.

---

## 1. Product overview

| Field | Value |
|-------|--------|
| Name | Votion 3DGS |
| Owner | **Yik**. MIT. Small private team. |
| Version | **2.0** only (no `x.y.z`) |
| High-level goal (this phase) | Native port of the Mickmumpitz **dataset** graph: **four** camera rails (optional 5th), WAN 2.1 I2V **14B fp8 720p** + Matrix-3D pano LoRA + LightX2V, **HiRes Composite `base_mode=geometry`** from the Phase 1 8K pano, dual-res SphereSfM. |

There is **no camera count that produces a perfect scan**. Extra rails buy explorable volume, not per-view fidelity, and do not fix glass.

---

## 2. Problem & objectives

**Problem:** One panorama cannot constrain a 3D scene. Mesh control video has **black holes**. Wan + Matrix-3D LoRA inpaints holes at 720p; that video is too soft to splat. HiRes puts the 8K photograph back wherever the mesh can answer, and leaves WAN only in holes.

**Objectives:**

1. Quality default: **4 rails**, sequential on one 3090.
2. Empty `prompt.txt` **blocks Generate** (must describe the actual pano).
3. WAN output **measured** 1440×720 ERP.
4. HiRes frames on disk at 8192×4096; mean coverage logged; not 0.0.
5. Product Generate never loads `docs/media/ref-hires-mask-split.png`. Live split: left validity mask, right WAN fill (including low-res). HTML dummy keeps that PNG.
6. Dual-res COLMAP training images are **gnomonic pinhole cube faces**, not equirect (Index #57).
7. Do not stretch WAN to 8K and skip composite. Do not use SeedVR instead of HiRes.
8. WAN and HiRes are **two stages**. Confirm WAN before 8K. Reconstruct Keep/Ditch/Re-gen; newest sparse after Reconstruct finishes.

---

## 3. Target users & use cases

Human persona job titles: UNKNOWN. Audience: small private team. Quality hardware: RTX 3090 24GB.

| ID | Scenario | Rails / notes |
|----|----------|----------------|
| P3-S1 | Quality first open | Copy four SMPL knot sets (studio-garden demo), user edits |
| P3-S2 | Suggest paths | Four SplatKit archetypes, all start at star; do not overwrite Confirm’d rails without asking |
| P3-S3 | Hero object (pond) | Keep 4; one rail `look_at_target` at the object (graph camera 1) |
| P3-S4 | Corridor / street | push-in `look_forward` |
| P3-S5 | Optional 5th | Add rail → `SphereSfMAddToDatasetDualRes`; HiRes width **4096** on that path — copy as-is |
| P3-S6 | Glass / mirrors | Do **not** add rails expecting a fix |
| P3-S7 | Wall clip | Allowed; Wan invents the far side — user must preview first |
| P3-S8 | Empty prompt | Block Generate |

Suggest-paths cap **≤ 8** (`path_suggest.py`). Video: four clips usually enough.

### User stories (from dummy)

- As a user I can switch among four (plus optional 5th) rails on Geometry.
- As a user I must Confirm rails before Generate.
- As a user I can edit WAN seed `0`, style suffix, and Negative before Generate.
- As a user I can Open frames folder.
- As a user I can pick HiRes frames preset `0-80/2` (graph default) or `0-15,16-/8` or `all` — do not silently replace the graph default.
- As a user I review 720p WAN (click a rail) before **Confirm WAN → HiRes**, and can re-gen a rail first.
- As a user I can Ditch a hallucinated rail or Re-gen WAN on that rail before Reconstruct.
- As a user, when Reconstruct finishes, I always see the newest kept-rail sparse.

---

## 4. Functional requirements

### 4.1 Geometry page (P3)

Dummy: [Phase3.html §geo-ui](../docs/Votion_3DGS_2.0_Phase3.html#geo-ui). Same Plot Camera chrome and **three street plates** as Phase 2.

Differences from P2:

- Four rail slots + optional **+ Add rail**
- Active rail dropdown (dummy: `1 look_at_target · pond`)
- **Suggest paths** on
- Active rail solid; others dimmed
- **Confirm rails** gates Generate
- Play still on Preview flight pane only
- `moge_level` **shared** (graph **9** on every Plot Camera). Mixed levels break HiRes cache reuse
- Warn if a rail’s first knot is far from origin (editor also locks knot 0)
- Changing knots after Confirm dirties the rail

**Suggest-path archetypes** (SplatKit `core/path_suggest.py`, default four, all start at origin):

| Order | Label | Orientation | Motion |
|---|---|---|---|
| 1 | `push-in` | `look_forward` | Gentle S-dolly into the most open direction |
| 2 | `arc-sweep` | `look_forward` | Lateral S-arc |
| 3 | `crane-rise` | `per_point_look` | Rise while pushing in |
| 4 | `pull-back` | `per_point_look` | Retreat, hold aim forward |

**SMPL demo rails:** copy **verbatim** from [Phase3.md SMPL demo rails](../docs/Votion_3DGS_2.0_Phase3.md#smpl-demo-rails-copy-then-let-the-user-edit). Nodes **27, 122, 133, 144**; optional **157**. All `length=81`, `moge_level=9`, first knot `0,0,0`.

How-many-cameras table: [Phase3.md](../docs/Votion_3DGS_2.0_Phase3.md#how-many-camera-sets). Rails 2–4 knot numbers also sit in markdown (nodes 133, 144, optional 157) — HTML says copy them; do not invent a fifth path schema.

### 4.2 Generate page (WAN + HiRes)

Dummy: [Phase3.html §gen-ui](../docs/Votion_3DGS_2.0_Phase3.html#gen-ui).

Left rail:

- `prompt.txt` required; empty blocks Generate
- **Seed (WAN)** visible, default `0`, editable
- Style suffix (graph CLIP id 10) **visible**, default = graph sentence, user may edit or clear
- **Negative** (graph CLIP id 11) **visible**, pre-filled, editable
- HiRes frames: graph default `0-80/2`; optional `0-15,16-/8`; `all` = 81
- Debug save: product default **off** (graph `debug_save='all'` is huge)
- **Generate WAN** / Stop / **Re-gen selected rail** / **Confirm WAN → HiRes** / **Open frames folder**
- HiRes does **not** start when Generate WAN finishes

Viewport while WAN runs: clickable **Rail N** chips; **live split** (left = validity mask, right = last WAN frame, including low-res). Product **never** loads `docs/media/ref-hires-mask-split.png` — that file is HTML dummy only.

After HiRes, `gate_masks` from the job means white = 8K, black = WAN (not feathered circles). HTML §04 still uses the dummy PNG to explain that legend.

Rails run **sequentially** on one 3090. Coverage ≈ 0 → `output_width` too small vs 8K.

**SMPL CLIP strings (defaults; do not silently replace `prompt.txt`):**

- Positive id 10: `A high quality panoramic video of the scene, photorealistic, very clear, smooth camera movement, ultra-detailed, high resolution. Panorama.`
- Negative id 11: `The video is not of a high quality, it has a low resolution. Distortion. strange artifacts.`

Product: **positive = `prompt.txt` (actual scene)** plus optional style suffix (default = id 10). Wrong prompt degrades hole fill (SplatKit README).

### 4.3 WAN model chain (order from JSON `links` 1→2→4 — do not swap)

1. `UNETLoader` id 4 `Wan2_1-I2V-14B-720P_fp8_e4m3fn.safetensors`
2. Pano LoRA id 5 `pano_video_gen_720p_comfy.safetensors` strength **0.98**
3. LightX2V id 7 `lightx2v_T2V_14B_cfg_step_distill_v2_lora_rank64_bf16.safetensors` strength **1**
4. `ModelSamplingSD3` id 8 shift **5**

`SplatKit_WanI2VMaskedConditioning` widgets **`[1440, 720, 81, 'black', False]`** = width, height, length, `hole_fill`, `invert_mask`. CLIP-Vision: `clip_vision_h.safetensors`, encode **`center`**. VAE: `wan_2.1_vae.safetensors`. Hole-fill upscaler: `4x-UltraSharp.pth`.

KSampler stored `[0, 'fixed', 4, 1, 'euler', 'normal', 1]` but **steps is linked** to PrimitiveInt id 55 **STEPS = 8**. Effective: seed 0, steps **8**, cfg 1, `euler`, `normal`, denoise 1. The stored `4` is unused.

`control_mask`: **white = valid/known, black = hole** (`invert_mask=False`). Paint holes `hole_fill='black'` **before** VAE encode.

GGUF `wan2.1-i2v-14b-720p-Q4_K_M.gguf` is the ≤16 GB path — **not** 3090 Quality.

Do not load Skywork `.ckpt` without `convert_pano_lora.py`. Use `pano_video_gen_720p_comfy.safetensors` only.

Native WAN i2v. Do **not** shell out to hidden ComfyUI. Largest P3 unknown: Diffusers WanPipeline vs Matrix-3D `code/panoramic_image_to_video.py` on **Windows** (their README is Linux-tested). Do not invent a third loader.

SageAttention / Triton-Windows: try; if import fails, continue. If Sage is on, apply the **black-frame patch** (copy from SplatKit / Kijai at vendor pin).

### 4.4 HiRes Composite

Copy SMPL node 96 widgets from [Phase3.md HiRes widgets](../docs/Votion_3DGS_2.0_Phase3.md#hires-widgets-from-smpl-node-96-copy). Locked highlights:

- `base_mode=geometry`, `output_width=8192`, frames **`0-80/2`**
- Alternate preset `0-15,16-/8` is SplatKit-recommended cheaper — expose, do not silently replace graph default
- Plot Camera `moge_level=9`, HiRes `moge_level=6` — **copy both forever** (locked 2026-09-01). Do not unify.
- Graph `tone_mode='off'` (not SplatKit default `luma`)
- `traj_index` 0…3 (and 4 on add-path). Add-path HiRes `auto_name` **True**, width widget **4096**
- `panorama_geometry` must be the **same** pano Plot Camera / WAN saw

Do not enable `base_mode=wan` as the splat default.

### 4.5 Dual-res SphereSfM

Copy node 84 widgets from [Phase3.md](../docs/Votion_3DGS_2.0_Phase3.md#dual-res-spheresfm-node-84-widgets). Replace `my_scene_hires` with Votion scene name. Hires width **8192**. Matcher `exhaustive`. Add-path node 90 hires width **4096** — copy as-is.

After SphereSfM, Votion **gnomonic-remaps** every cube-face PNG (`engine/sfm/cubic_reproject.py`, locked 2026-09-08 / Index #57). Do not run vanilla COLMAP pinhole on ERP. Do not un-bend `points3D` to flatten a bowed splat. Retrain after Reconstruct.

Train cap ~**3M** Gaussians (train itself may complete in P2/P4; P3 must leave a COLMAP dataset that supports it without OOM after WAN has exited).

### 4.6 Workers

1. `engine.workers.wan` — one rail per process; load 14B fp8 + LoRAs; generate; unload; exit. Does **not** start HiRes.  
2. `engine.workers.hires` — only after Confirm WAN → HiRes; write 8K PNGs to disk (**never** as one giant GPU tensor)  
3. `engine.workers.sfm` — kept rails; wipe previous sparse; dual-res: low-res proxies for matching, 8K cube faces for training images; **gnomonic remap** of every face PNG  

HiRes ~11 GB peak (SplatKit, 5090) **after** WAN has exited.

Copy WAN/HiRes defaults into `engine/video/defaults.py` with comments `from SMPL json node id N`.

Per rail: Plot Camera → Wan Conditioning + KSampler + VAEDecode + CreateVideo widgets `[16, 8]` (name from Comfy core `INPUT_TYPES` at implement time) → HiRes. After four composites: `SplatKit_SphereSfMDatasetDualRes` id 84. Optional add: id 90.

Each rail owns `rail_json` (shared `condition_dir` overwrites — do not let four rails clobber one file).

### 4.7 Reconstruct page (P3)

Dummy: [Phase3.html #recon-ui](../docs/Votion_3DGS_2.0_Phase3.html#recon-ui). P2 Reconstruct dummy stays single-trajectory.

- Keep / Ditch per rail. Ditched rails are left out of SphereSfM.
- Re-gen WAN on one rail (jumps to Generate).
- Reconstruct uses kept WAN clips (else control video).
- After the job **finishes loading**, always show this run’s sparse (`sfm.json` + `colmap/sparse/0`). Do not keep a previous 1-rail solve on screen.

### 4.8 Acceptance criteria

- [ ] Four rails, sequential WAN, four `rail_json` files that do not overwrite.
- [ ] Optional 5th Add-rail writes via `SphereSfMAddToDatasetDualRes` without colliding (`auto_name` True on that HiRes node).
- [ ] WAN outputs 1440×720 ERP clips (measure, record in `env_report`).
- [ ] Empty `prompt.txt` blocks Generate.
- [ ] HiRes frames on disk at 8192×4096; mean coverage logged; not 0.0.
- [ ] UI shows live validity mask | WAN fill during Generate (not the dummy PNG). After HiRes, `gate_masks` means white = 8K, black = WAN.
- [ ] Dual-res COLMAP `images/` are **gnomonic** pinhole cube faces, not equirect.
- [ ] In-app train with 3M cap completes without OOM on 3090 after WAN has exited.
- [ ] SageAttention missing → warning in log, job still completes. If Sage is on: no all-black Wan frames.
- [ ] Suggest paths fills four distinct archetypes, all starting at the star.
- [ ] Product Generate never loads `ref-hires-mask-split.png`; live split is mask | WAN; click rail switches preview.
- [ ] Generate WAN does not start HiRes; Confirm WAN → HiRes is a second click.
- [ ] Reconstruct Ditch excludes that rail from SfM; Re-gen WAN targets one rail.
- [ ] After Reconstruct finishes, the sparse viewport is the newest kept-rail solve.

### 4.9 Do not

- Do not stretch WAN to 8K and skip composite.
- Do not use SeedVR (or any video upscaler) **instead of** HiRes geometry composite.
- Do not run PanoLRM or keep WorldFM as a parallel engine.
- Do not claim extra rails fix glass.
- Do not default `frames` to `0-15,16-/8` while documenting the graph as `0-80/2`.
- Do not auto-start HiRes after WAN. Do not load `ref-hires-mask-split.png` in the product UI.

---

## 5. Non-functional requirements

| Topic | Locked | UNKNOWN |
|-------|--------|---------|
| Performance | Sequential rails; LightX2V 8 steps; unload 14B before HiRes | Wall-clock per rail; 5B Fast path (P5) |
| Reliability | Cancel mid-WAN (Phase 4 also tests this); coverage log | Auto-retry black frames beyond Sage patch |
| VRAM | 24GB Quality; GGUF is not Quality | Whether 14B fp8 + LoRAs actually fit — prove on the 3090 |
| Security | Local; no token UI; **no telemetry** | — |

---

## 6. UX/UI references

| Artifact | Use |
|----------|-----|
| [Phase3.html #geo-ui](../docs/Votion_3DGS_2.0_Phase3.html#geo-ui) | Four-rail Geometry dummy |
| [Phase3.html #gen-ui](../docs/Votion_3DGS_2.0_Phase3.html#gen-ui) | WAN then Confirm HiRes; live mask\|WAN; dummy PNG stays in HTML |
| [Phase3.html #recon-ui](../docs/Votion_3DGS_2.0_Phase3.html#recon-ui) | Keep / Ditch / Re-gen; newest sparse |
| `docs/media/ref-hires-mask-split.png` | HTML showcase of the split (not loaded by the product) |
| Same `geo-*.png` plates as P2 | Editor backdrop |
| [Phase3.html pipeline figures](../docs/Votion_3DGS_2.0_Phase3.html) | WAN / LightX2V / Matrix-3D stills |

**Flow:** Geometry (4 rails, preview each, Confirm) → Generate WAN (review 720p, optional re-gen) → Confirm WAN → HiRes → Reconstruct (keep/ditch, newest sparse) → Splat page (Phase 2/4) trains.

---

## 7. Technical constraints & dependencies

Model files: [Index video/dataset table](../docs/Votion_3DGS_2.0_Index.md) and [Phase3.md models](../docs/Votion_3DGS_2.0_Phase3.md#models-from-the-graph-notes).

Native port = SplatKit `core/` functions, not Comfy nodes. WAN sampling = Diffusers / thin loader of the **same safetensors**.

CreateVideo `[16, 8]` names: UNKNOWN until Comfy core `INPUT_TYPES`.

---

## 8. Success metrics

Done = §4.8. Mean HiRes coverage: **log it**. Not a ship gate (locked 2026-09-01). SplatKit bathroom figures are reference, not Votion pass/fail.

---

## 9. Timeline & milestones

**No calendar dates.** Order: four-rail editor + Suggest paths → native WAN i2v on Windows 3090 (largest risk) → user Confirm → HiRes geometry composite → dual-res SphereSfM (kept rails, newest sparse).

---

## Open questions (this phase)

1. WAN loader on Windows: Diffusers vs Matrix-3D script vs other **named** port — which, after a spike, not before?
2. CreateVideo widget names for `[16, 8]` after reading Comfy core `INPUT_TYPES`.
