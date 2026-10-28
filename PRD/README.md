# Votion 3DGS 2.0 — PRD set

These files are the **product requirements** for building the native Windows app. They are compiled from the HTML dummy UI and the locked phase markdown. They do **not** invent owners, KPIs, dates, Hub ids, Warp widget names, or installer brands.

## How to read

| If this disagrees with… | Winner |
|-------------------------|--------|
| HTML dummy + [Index certainty log](../docs/Votion_3DGS_2.0_Index.md) | Those two |
| Phase markdown (graph numbers, node widgets) | Phase markdown |
| This PRD | Update the PRD; do not “fix” the dummy |

**UNKNOWN** in these files means: do not fill it in during implementation. Ask first.

## Files

| PRD | Phase | Build this first? | Dummy UI | Spec markdown |
|-----|-------|-------------------|----------|----------------|
| [Phase0_Skeleton_PRD.md](Phase0_Skeleton_PRD.md) | P0 Skeleton | Yes | [Phase0.html](../docs/Votion_3DGS_2.0_Phase0.html) | [Phase0.md](../docs/Votion_3DGS_2.0_Phase0.md) |
| [Phase1_Panorama_PRD.md](Phase1_Panorama_PRD.md) | P1 Pano | After P0 | [Phase1.html](../docs/Votion_3DGS_2.0_Phase1.html) | [Phase1.md](../docs/Votion_3DGS_2.0_Phase1.md) |
| [Phase2_First_Splat_PRD.md](Phase2_First_Splat_PRD.md) | P2 First splat | After P1 `pano.png` | [Phase2.html](../docs/Votion_3DGS_2.0_Phase2.html) | [Phase2.md](../docs/Votion_3DGS_2.0_Phase2.md) |
| [Phase3_Dataset_PRD.md](Phase3_Dataset_PRD.md) | P3 Dataset | After P2 plumbing | [Phase3.html](../docs/Votion_3DGS_2.0_Phase3.html) | [Phase3.md](../docs/Votion_3DGS_2.0_Phase3.md) |
| [Phase4_Productize_PRD.md](Phase4_Productize_PRD.md) | P4 Product | After Quality path works | [Phase4.html](../docs/Votion_3DGS_2.0_Phase4.html) | [Phase4.md](../docs/Votion_3DGS_2.0_Phase4.md) |

Live HTML pack: [titanothereyik.github.io/votion-3dgs-2-html](https://titanothereyik.github.io/votion-3dgs-2-html/)

Shared chrome CSS: [votion-plan.css](../docs/votion-plan.css)  
Geometry / Splat dummy JS: [rail-dummy.js](../docs/rail-dummy.js)

## Product (locked, all phases)

- **Name:** Votion 3DGS
- **Line:** 2.0 (native Windows; not an upgrade of V1)
- **Owner / publisher:** **Yik** (personal). Locked 2026-09-01 PRD Q&A.
- **First audience:** a **small private team** (not public download, not solo-only). Locked 2026-09-01.
- **License:** **MIT**. Locked 2026-09-01.
- **Shipping version string:** **2.0** only — no `x.y.z` (locked 2026-09-01).
- **Code home:** `D:\Votion3DGS\`
- **Install root:** folder the user selected in Inno Setup (dev default `D:\Votion3DGS\`). Hub cache: `<install_root>\.hf_cache\`. Ready weights: `<install_root>\models\`. Never `C:\Users\<user>\.cache\huggingface`. Locked 2026-09-02.
- **Download progress:** each Home model-status cell has its own bar during **Download models**. Locked 2026-09-03.
- **Job strip:** every background job drives a progress strip under the stage chips. Not log-only. Locked 2026-09-03.
- **Generate dummy PNG:** `ref-hires-mask-split.png` is HTML-only. Product live-splits mask | WAN. Click rail to preview. WAN then Confirm HiRes. Reconstruct Keep/Ditch/Re-gen; newest sparse after load. Locked 2026-09-04.
- **Splat train viewport:** live rasterized 3DGS (not a point cloud). Max steps any integer ≥ 1. Star reset; LMB look, RMB pan, wheel dolly; zoom uncapped. Stop on rail and job bar. MCMC Compact shares the LiteGS Gaussians. Needle takeover / crushed tones are bugs. Locked 2026-09-05.
- **Splat trainer rebuild:** vendored **LiteGS** backend (built locally at Setup, Inria licence, never shipped) with two density controllers, **LiteGS** (default) and **MCMC Compact** (Jung & Hong 2025 §3.1 only). Shared-memory viewport, Pause / Resume, real Gaussian budget, optimizer-inclusive checkpoint, full-SH `splat.ply` + compact export, hold-out PSNR, old gsplat checkpoints show “Retrain required”. Locked 2026-09-07.
- **Gnomonic cube faces:** SphereSfM `SPHERE` match, then Votion remaps cube-face PNGs to true 90° gnomonic pinholes. Do not COLMAP-pinhole raw ERP; do not un-bend `points3D` to fix a bowed splat. Retrain after Reconstruct. Locked 2026-09-08.
- **V1 archive (do not mix):** `d:\InSpatio World\Yik Votion WorldFM\` (Gradio + WSL)
- **Engine:** native Python. No ComfyUI in the product. No WSL.
- **Shell:** PySide6, Engineering Blueprint look from V1 (`theme.css` tokens)
- **Quality GPU:** RTX 3090 24GB, CUDA device **0** unless the user picks another

Happy path: **2D Images** (or 360 Images / text) → 8K ERP → Compute geometry → rails → Generate WAN (review 720p) → Confirm HiRes → SphereSfM (kept rails) → in-app LiteGS (**LiteGS** or **MCMC Compact** controller, Gaussian budget) → full-SH `splat.ply` → optional Open in Brush / LichtFeld / Postshot → UE 5.5 + MLSLabsRenderer.

## Shared window (every page)

Copied from [Index HTML copies](../docs/Votion_3DGS_2.0_Index.md#html-copies-dummy-ui-spec):

1. Title bar **Votion 3DGS**
2. Brand row + live status
3. Six **stage chips** that **navigate**: Home / Panorama / Geometry / Generate / Reconstruct / Splat
4. **Job strip** under the chips on every page (caption + bar). Home also keeps per-cell download bars.
5. **Left rail** = page controls. **Right viewport** = preview / editor
6. Bottom **Log** drawer (subprocess) and **Help** drawer (page-specific). Not a Home accordion. Not a seventh page.
7. Switching chips **restores** generated views (do not reset to empty / green warp). Locked 2026-09-03.

## What is not in 2.0

Do not re-open: slice+SHARP, UniSHARP, HY-World 2.0, NVIDIA Lyra, Marble, WorldFM/HunyuanWorld, PanoLRM, SeedVR as the sharpness path, ComfyUI UI, WSL.

P5 stretch only (not a 2.0 blocker): SeedVR2 experiment, Matrix-3D opt recon, macOS. Matrix-3D 5B as Fast default is a **2.0** lock.

## Locked from PRD Q&A (2026-09-01) — do not re-open

- Owner: Yik (personal)
- Audience: small private team
- License: MIT
- Version string: **2.0** only (no `x.y.z`)
- Timeline: **no calendar dates** — phase order is the schedule
- Success: **phase acceptance checklists only** (no extra KPIs)
- HiRes coverage: **log only**, not a ship gate
- Fast default WAN: **Matrix-3D 5B** (locked 2026-09-01). GGUF Q4 is not the Fast default. Exact 5B filename: copy from Matrix-3D README at implement — do not invent.
- Geometry viewport: **QPainter** plates + gizmos (locked 2026-09-03). Do not ship QWebEngine. vispy is not the Geometry editor.
- Generate / Reconstruct P3 (locked 2026-09-04): dummy PNG HTML-only; live mask|WAN; click rails; two-stage WAN then HiRes; Keep/Ditch/Re-gen; newest sparse after Reconstruct.
- Splat viewport P2 (locked 2026-09-05): live raster; uncapped max steps; Star camera LMB/RMB/wheel; Stop; MCMC Compact = LiteGS Gaussians; needle takeover / crushed tones are bugs.
- Splat trainer P2 (locked 2026-09-07): LiteGS backend vendored and built locally (Inria licence, not redistributed); controllers LiteGS | MCMC Compact (Jung & Hong §3.1 only, relighting out of scope); shared-memory viewport; Pause / Resume; real budget; optimizer-inclusive checkpoint; full-SH ply; hold-out PSNR; old checkpoints need Retrain. Index #48–56.
- Gnomonic cube faces (locked 2026-09-08): Index #57. `engine/sfm/cubic_reproject.py` after Reconstruct. No pinhole-on-ERP. No `points3D` un-bend as the undistort.
- Installer: **Inno Setup**
- Profiles at 2.0 launch: **Quality + Fast + Scout** all ship
- Telemetry: **none** — fully local, no phoning home
- Plot Camera `moge_level` **9** and HiRes `moge_level` **6**: **copy both forever** (do not unify)

## Still UNKNOWN (do not invent) — spikes only

These wait on a machine/read, not on a product decision:

1. Harmonize / UltimateSDUpscale remaining widget **names** (read at pin — arrays already copied). Florence Hub, Ostris READY, and h_fov 10–170 are locked 2026-09-03.
2. Native WAN i2v Windows loader (spike in P3)
3. ~~gsplat wheel on 3.12+cu128 (P0 spike)~~ — moot 2026-09-07: trainer is LiteGS, built locally (spike passed, `docs/env_report.txt`)
4. SplatKit git SHA to pin
5. CreateVideo `[16, 8]` names (Comfy core `INPUT_TYPES`)
6. Disk / Python / driver: **measured 2026-09-01 on this PC** — Python **3.12.10** present; RTX **3090** 24GB driver **595.79**; `D:` free **142.3 GB** (Quality model pack ~45–60 GB can fit; still check before a WAN 14B pull). Do not invent torch patch or CUDA toolkit version until `env_report.py`.
