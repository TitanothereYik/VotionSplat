# PRD — Phase 1: Panorama (2D / 360 / text → 8K ERP)

**Product:** Votion 3DGS 2.0  
**Phase:** 1 of 4  
**Depends on:** [Phase 0 PRD](Phase0_Skeleton_PRD.md) window + venv + Krea 2 weights (Florence-2 if auto-caption is used)  
**Unlocks:** Phase 2 first splat  
**Out of scope:** MoGe, WAN, SphereSfM, Gaussian train  

**Sources (do not drift):**

- Dummy UI: [docs/Votion_3DGS_2.0_Phase1.html](../docs/Votion_3DGS_2.0_Phase1.html) · [live](https://titanothereyik.github.io/votion-3dgs-2-html/Votion_3DGS_2.0_Phase1.html)
- Spec: [docs/Votion_3DGS_2.0_Phase1.md](../docs/Votion_3DGS_2.0_Phase1.md)
- Graph: `d:\AI_3DGS\260825_MICKMUMPITZ_Krea2-360Pano-Creator_1-0.json` (copy also under `docs\`)
- How-to copy: graph **MarkdownNote id 4** → Panorama Help (do not rewrite the meaning)
- Set index: [README.md](README.md)

If HTML dummy and this PRD disagree, the dummy + certainty log win. Graph widget lists in Phase1.md win over paraphrases here.

---

## 1. Product overview

| Field | Value |
|-------|--------|
| Name | Votion 3DGS |
| Owner | **Yik** (personal). License **MIT**. Audience: small private team. |
| Version | **2.0** only (no `x.y.z`) |
| High-level goal (this phase) | The 2.0 happy path is **one 2D image → a full 3DGS scene**. This phase only writes **`outputs/<scene>/pano.png`** (8K ERP). Phase 2 turns that file into the first splat. Phase 3 fills holes with WAN. |

Three first-class inputs, one output:

| UI mode | Graph MODE | What happens |
|---------|------------|----------------|
| **2D Images (default)** | `false` = B IMAGE→PANO | Warp 2D onto green ERP → Ostris Edit + ERP outpaint LoRA → seam node **63** → 8K tail |
| **360 Images** | skip Krea | User file is already 2:1 ERP; optional **Upscale to 8K** |
| **text** | `true` = A TEXT→PANO | Krea 2 Turbo + 360 T2I LoRA → seam node **31** → 8K tail |

Native Python. No ComfyUI in the product. No WSL. ComfyUI required a dummy INPUT IMAGE in text mode; **Votion must not.**

---

## 2. Problem & objectives

**Problem:** A 3DGS dataset needs a sharp 8K equirectangular pano. Users arrive with a normal photo, an existing 360, or a text sentence. V1/Comfy graphs do this inside ComfyUI. V2 must do it in-app.

**Objectives:**

1. Default toggle is **2D Images**.
2. Native **Ostris Edit** is **READY** (2026-09-03). 2D Images Generate is enabled when a photo is loaded and surroundings are non-empty. Do not shell out to ComfyUI.
3. Generate resolution **2048×1024**; final **8192×4096**.
4. After success, `prompt.txt` is written for later WAN (rules differ by mode — §4).
5. Viewport is a **user-draggable split**: left 2:1 (Fit or 100% + slide) | right spherical look (yaw + pitch).

---

## 3. Target users & use cases

Human persona job titles: UNKNOWN. Audience: small private team (locked). Hardware: Quality path RTX 3090 (from Index).

| ID | Scenario | Notes |
|----|----------|--------|
| P1-S1 | User drops one 2D photo, types surroundings, Generate | Ostris READY; suffix `no peoples, no cars` |
| P1-S2 | User chooses Florence-2, clicks **Caption**, edits the box, Generate | Caption is a separate worker; suffix still appended |
| P1-S3 | Manual surroundings box empty | **Block Generate** |
| P1-S4 | User loads a 2:1 Poly Haven / 360 file | No Krea; **Use as pano.png**; optional Upscale to 8K; keep upscaler prompt |
| P1-S5 | User types a scene sentence (text mode) | Locked trigger prefix stays; empty sentence blocks Generate |
| P1-S6 | Non-2:1 file in 360 Images | Reject; do not letterbox |
| P1-S7 | Left pane Fit vs 100%; drag the split | 360 look stays spherical |

### User stories (from dummy)

- As a user I can switch **2D Images / 360 Images / text** without leaving the Panorama page.
- As a user I can **type or drag** **h_fov** (default **70**, range **10–170**, step **0.5**) and see the green FOV window change live.
- As a user I can press **Detect** to fill h_fov from the photo's lens EXIF. A cropped photo, or one with no lens tags, uses a MoGe guess and says so. Locked 2026-11-11.
- As a user, after a 2D Images Generate, I see the original photo pasted back inside the h_fov footprint, with a feathered seam. Locked 2026-11-11.
- As a user I can **Fit** the full 2:1 in the pane or view **100%** native pixels with 2D slide, and drag the split vs the 360 look.
- As a user I can edit IMAGE→PANO seed `12345`, IMAGE seam seed `12345`, TEXT→PANO seed `322344328372862`, TEXT seam seed `8`, and the upscaler prompt (default `High resolution photography`).
- As a user I can read graph how-to in the Help drawer.

---

## 4. Functional requirements

### 4.1 Left rail — match [Phase1.html §01](../docs/Votion_3DGS_2.0_Phase1.html#ui)

**Toggle:** 2D Images (default) / 360 Images / text. Native Ostris is READY — 2D Generate is enabled when a photo is loaded and surroundings are non-empty.

**2D Images:**

- Image drop **shows a thumbnail**
- **h_fov** slider **and** typed number, default **70**, range **10–170**, step **0.5**. Graph pin: [ComfyUI-Mickmumpitz-Nodes](https://github.com/mickmumpitz/ComfyUI-Mickmumpitz-Nodes) `e3df7e1199c8170335fe8c7d3dc0ecf1839d2784`. Raw widget list to copy (do not invent labels): `[2048, 1024, 1, 70, 0, 0, 0, 2, 10, 0.55, "#000000"]`
- Surroundings radios: **Type a sentence** | **Florence-2 caption, then edit**
- Always append **`no peoples, no cars`** (typed or Florence-2)
- **Caption** button only when Florence-2 is selected → `engine.workers.caption`
- Seeds: IMAGE→PANO `12345`; seam INPAINT node 63 `12345` (visible, editable)
- Upscaler prompt default `High resolution photography` (visible, editable)
- Generate enabled when photo + non-empty surroundings (Ostris READY)

**360 Images:**

- 2:1 picker. Reject non-2:1.
- Store as-is even if width ≠ 8192. Visible **Upscale to 8K** runs the shared 8K tail.
- Keep the **upscaler prompt** box on this rail (graph ⑤).
- Quality HiRes later refuses if width ≠ 8192.
- `prompt.txt` empty is OK in P1; Phase 3 blocks WAN if still empty.
- Buttons: **Use as pano.png**, **Upscale to 8K**
- No Krea, no Ostris, no Florence-2 required

**text:**

- Locked trigger prefix (not editable as a meaning change):
  `img-txt-2-360 A full 360 degree equirectangular panorama in 2:1 spherical projection,`
- User sentence appended after that comma
- Empty sentence → block Generate
- Seeds: TEXT→PANO `322344328372862`; seam node 31 `8`
- Generate enabled (no Ostris)

### 4.2 Viewport

- Split is **user-draggable**. **Left** 2:1 ERP. **Right** spherical HDRI-style look (yaw + pitch), not a pan-only unwrapped strip
- Left **Fit** = full picture in the pane. **100%** = native pixels + 2D slide. Right 360 stays the spherical look
- After 2D Generate, the left pane shows the 8K ERP. Switching chips away and back **keeps** that ERP (not the green warp). Locked 2026-09-03.
- **2D Images before Generate:** left is Warp **green `#00ff00` ERP canvas** (2:1), photo in a **curved FOV window**, **+** crosshair, footer **2048 × 1024**. Type or drag h_fov. After successful Generate, left becomes the **8K ERP**
- **360 Images / text:** left = unwrapped 2:1 ERP, right = spherical look of that file
- Dummy FOV crop reference: `docs/media/ref-fov-green-crop.png`

### 4.3 2D Images pipeline (copy JSON; do not invent nodes)

Full node list: [Phase1.md §2D Images](../docs/Votion_3DGS_2.0_Phase1.md). Summary:

```
2D image
  → MickmumpitzPanoWarp (h_fov default 70, type or drag, 10–170 step 0.5)
  → green canvas #00ff00 + ImageCompositeMasked
  → Krea2OstrisEditModelPatch (true) + ERP OUTPAINT LoRA krea2_oedit_360_erp_outpaint_lora_v1 strength 1
  → TextEncodeKrea2OstrisEdit
       positive = outpaint instruction + " " + surroundings
  → ModelSamplingFlux IMAGE→PANO widgets 0.90625, 0.5, 2048, 1024
  → KSampler IMAGE→PANO: seed 12345, steps 10, cfg 1, er_sde, simple, denoise 1
  → Harmonize boundary (widgets 1, 8, 800, true — names from node source at pin)
  → Put original 2D image back
  → roll 50% → SEAM INPAINT node 63 → roll back
  → shared 8K tail
  → paste the original photo back (2D Images only, feathered seam)
```

**Outpaint instruction (leave as-is):**

`Fill the green spaces according to the image. Outpaint as a seamless 360 equirectangular panorama (2:1). Keep the horizon level. Match left and right edges.`

Graph demo surroundings `Professional photography of living room.` is **demo content**, not a silent default. Always append **`no peoples, no cars`**.

After successful 2D Generate: `prompt.txt` = outpaint instruction + space + surroundings (with suffix).

**Photo paste (2026-11-11):** `engine/pano/photo_paste.py` runs after the 8K tail on 2D Images only. It warps the source photo with the same `persp_to_erp` placement as generation and blends a 128 px cosine feather through a 5-level Laplacian pyramid. The pre-paste file is `outputs/<scene>/_work/pano_before_paste.png`. 360 Images and text clear it.

**Detect (2026-11-11):** `engine/workers/detect_fov.py`. EXIF focal length is used when the pixel shape matches the EXIF pixel size. Otherwise MoGe v1 estimates the field of view. That guess can be about 10° off on a straight-on facade.

**Ostris pack pin in graph:** [ostris/ComfyUI-Krea2-Ostris-Edit](https://github.com/ostris/ComfyUI-Krea2-Ostris-Edit) `7756566160c4a1b24bb1bd9f0ff3ced1a83d7547`. Native port **READY** (2026-09-03). Do not shell out to ComfyUI.

UNET / CLIP / VAE / LoRA filenames: [Index](../docs/Votion_3DGS_2.0_Index.md). CLIPLoader: `qwen3vl_4b_fp8_scaled.safetensors`, type `krea2`. UNET: `krea2_turbo_fp8_scaled` (3090). Not nvfp4.

### 4.4 text pipeline

```
trigger prefix + user scene sentence
  → CLIPTextEncode SCENE PROMPT
  → empty negative (ConditioningZeroOut, cfg 1)
  → 360 PANO LoRA krea2_t2i_360_erp_lora_v1 strength 1
  → KSampler TEXT→PANO: seed 322344328372862, steps 10, cfg 1, er_sde, simple, denoise 1
  → roll 50% → SEAM INPAINT node 31 → roll back
  → shared 8K tail
```

After successful text Generate: `prompt.txt` = locked trigger prefix + user sentence.

### 4.5 Shared 8K tail (2D Images and text only)

Copy [Phase1.md §Shared 8K tail](../docs/Votion_3DGS_2.0_Phase1.md) including UltimateSDUpscale node **77** widget list. Upscaler prompt CLIP: visible box, default `High resolution photography` — do **not** put scene details there (video rule).

If UltimateSDUpscale cannot be ported: that is a **Quality blocker**. Do **not** ship Real-ESRGAN 2× + lanczos as Quality and claim graph ⑤. Locked 2026-09-01: Quality 8K tail **must match graph ⑤**.

**Seam INPAINT is not shared.** Do **not** copy TEXT seam (node 31) onto 2D Images (node 63). Widget table is in Phase1.md.

### 4.6 Workers

- `engine.workers.pano` — one process, then exit (VRAM back before Phase 2 MoGe). 2D Images pastes the photo after the 8K tail.
- `engine.workers.caption` — Florence-2 only fills the surroundings box
- `engine.workers.detect_fov` — fills h_fov from EXIF, else MoGe v1. Qt does not import it.

Florence-2: Microsoft Florence-2 family, Apache-2.0. Hub (first successful P1 caption): `florence-community/Florence-2-large` rev `4271c66b88cdbc05735372ec13b2360108de5317`. Use a short caption task (one sentence). Always append **`no peoples, no cars`**. Dummy Caption paste (HTML only): `[Florence-2] edit this sentence before Generate, no peoples, no cars`.

### 4.7 Acceptance criteria

- [ ] **2D Images** is the default toggle.
- [ ] **360 Images:** 2:1 file becomes `pano.png` without Krea.
- [ ] **text:** 2:1 image **without** a 2D input image; trigger prefix still on the Krea prompt.
- [ ] **2D Images:** Generate is enabled when Ostris is READY, a photo is loaded, and surroundings are non-empty.
- [ ] **2D Images:** original image region recognizable; outpaint instruction unchanged; left/right seam not a visible cut in the 360 viewer.
- [ ] Florence-2 fills the surroundings box **and** appends `no peoples, no cars`; user can edit; empty manual box blocks Generate.
- [ ] Typed surroundings also end with `no peoples, no cars` in `prompt.txt`.
- [ ] Left 2:1 **Fit / 100%** and a draggable split; right 360 tilts and looks spherical.
- [ ] After Generate, switching chips away and back still shows the 8K ERP (not the green warp).
- [ ] **h_fov** accepts a typed value as well as the slider (10–170, step 0.5). **Detect** fills it from EXIF, or from MoGe when EXIF is missing or the photo was cropped.
- [ ] **2D Images:** after the 8K tail, the original photo is pasted back with a feathered seam. Pixels outside that footprint stay the generated panorama.
- [ ] **2D Images** and **text:** final size **measured** (expect 8192×4096) and recorded in `env_report` / scene log.
- [ ] Same 8K file is what Phase 3 HiRes uses as geometry pano **and** texture.

### 4.8 Do not

- Do not invent Warp widget names; read `INPUT_TYPES` at the Mickmumpitz-Nodes pin.
- Do not invent a Florence-2 Hub revision (pinned: `florence-community/Florence-2-large` @ `4271c66`).
- Do not generate at 8K natively in Krea. Do not skip seam roll.
- Do not strip trigger prefix / outpaint instruction / surroundings suffix `no peoples, no cars`.
- Do not bring HunyuanWorld / FLUX Fill back.

---

## 5. Non-functional requirements

| Topic | Locked | UNKNOWN |
|-------|--------|---------|
| Performance | Generate at 2048×1024 then upscale; worker exits to free VRAM | Seconds-per-pano, batch size |
| Security | Local weights; no token UI; **no telemetry** | — |
| Reliability | Empty required fields block Generate | Retry policy |
| Scalability | One scene at a time (not specified otherwise) | Queue of many panos |

---

## 6. UX/UI references

| Artifact | Use |
|----------|-----|
| [Phase1.html](../docs/Votion_3DGS_2.0_Phase1.html) | Full dummy: toggle, FOV crop, three-mode cards, `.pipe` graphs |
| `docs/media/ref-fov-green-crop.png` | Green ERP + curved FOV chrome to port |
| `docs/media/matrix3d-i2p.png` | Dummy 2D still |
| `docs/media/polyhaven-venice.png` | Dummy 360 / ERP result |
| [votion-plan.css](../docs/votion-plan.css) `.warp-*` / `.erp-*` | FOV crop layout |

Docs section **02** under the dummy is a **3-column equal-height** card grid (2D / 360 / text). Graphs in **03** are `.pipe` diagrams.

**Flow (2D):** drop photo (thumbnail) → set h_fov (type or drag) → type or Caption surroundings (`no peoples, no cars` appended) → Generate → left pane switches to 8K ERP → `pano.png` + `prompt.txt` on disk.

---

## 7. Technical constraints & dependencies

| Dependency | Role | Pin / note |
|------------|------|------------|
| Krea 2 Turbo fp8 | UNET | Index URL; not nvfp4 on 3090 |
| krea2 T2I 360 LoRA | text path | Index |
| krea2 oedit 360 outpaint LoRA | 2D path | Index |
| Qwen3-VL 4B fp8 CLIP | text encoder | type `krea2` |
| wan_2.1_vae | VAE | Index |
| RealESRGAN_x2 | 8K tail | Index |
| Ostris Edit | 2D path | Native port READY 2026-09-03; graph SHA above |
| Mickmumpitz PanoWarp | FOV crop | h_fov 10–170 step 0.5; other names from pin |
| UltimateSDUpscale | tiled refine | Pack pin in Phase1.md; **required for Quality** (no labeled MVP) |
| Florence-2 | optional caption | `florence-community/Florence-2-large` @ `4271c66` |

Harmonize / UltimateSDUpscale extra widget **names**: copy arrays from the Krea JSON; name from node source at pin. Do not invent names.

---

## 8. Success metrics

Done = §4.7 checklist. No extra KPIs (locked 2026-09-01).

---

## 9. Timeline & milestones

**No calendar dates.** Order: Warp + 360 + text → native Ostris READY (done 2026-09-03) enables 2D Generate → 8K tail **graph ⑤ parity** (UltimateSDUpscale required for Quality).

---

## Open questions (this phase)

Spikes only:

1. ~~Native Ostris Edit on Windows — READY when?~~ **READY 2026-09-03.**
2. ~~Florence-2 exact Hub id / revision~~ **Pinned:** `florence-community/Florence-2-large` @ `4271c66`.
3. ~~Warp `INPUT_TYPES` min/max for h_fov~~ **10–170, step 0.5.** Other Warp widget **names** still from pin.
4. Harmonize and UltimateSDUpscale remaining widget **names** (read at pin — arrays already copied)
