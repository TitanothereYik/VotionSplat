# PRD — Phase 4: Productize (installer, profiles, Open in…, field guide)

**Product:** Votion 3DGS 2.0  
**Phase:** 4 of 4 (installable product)  
**Depends on:** Phases 0–3 working on the 3090 **Quality** path  
**Unlocks:** Windows users can install without cloning a research tree  

**Sources (do not drift):**

- Dummy UI: [docs/Votion_3DGS_2.0_Phase4.html](../docs/Votion_3DGS_2.0_Phase4.html) · [live](https://titanothereyik.github.io/votion-3dgs-2-html/Votion_3DGS_2.0_Phase4.html)
- Spec: [docs/Votion_3DGS_2.0_Phase4.md](../docs/Votion_3DGS_2.0_Phase4.md)
- Installer ideas (native Windows, not WSL): `Yik Votion WorldFM/docs/SmallInstaller_EasyUsers_Plan.md`
- Field guide structure to rewrite: `Yik Votion WorldFM/ui/field_guide.md`
- Set index: [README.md](README.md)

Installer is a `.pipe` diagram in the HTML (Inno/NSIS → Start Menu → install dir), **not** an in-app page.

---

## 1. Product overview

| Field | Value |
|-------|--------|
| Name | Votion 3DGS |
| Owner / publisher (installer) | **Yik** (personal). License **MIT**. |
| First audience | Small private team |
| Version string on About / installer | **2.0** only (no `x.y.z`) |
| High-level goal (this phase) | Quality / Fast / Scout profiles, **small** installer (code + scripts, **not** 60 GB of weights), in-app field guide, crash reload, Unreal export checklist, **Open in Brush / LichtFeld / Postshot**. |

**Locked:** models download **post-install** into the folder the user selected in Inno Setup. Hub cache is `<install_root>\.hf_cache\`; ready weights are `<install_root>\models\`. Do **not** write Hub cache to `C:\Users\<user>\.cache\huggingface`. Do **not** embed WAN 14B in the installer. No Hugging Face token UI for the default manifest (public URLs in the Comfy notes).

---

## 2. Problem & objectives

**Problem:** Phases 0–3 are a developer tree on `D:\Votion3DGS\`. A Windows user still cannot install, recover from a crash, open COLMAP in Brush, or play `splat.ply` in Unreal from a documented path.

**Objectives:**

1. Fresh Win11 + 3090: Setup → Download Quality models → Launch → **2D Images (or 360 / text)** → 8K ERP → Confirm rails → Generate → `splat.ply` without WSL or ComfyUI running. Ostris is READY for 2D Images.
2. Fast/Scout selectable; Scout skips HiRes. **Quality + Fast + Scout all ship** on 2.0 day-1 (locked 2026-09-01).
3. Kill mid-WAN: cancel works; scene reloadable.
4. Open-in-Explorer on COLMAP works even if Brush is not installed.
5. `splat.ply` imports and plays in **UE 5.5 + MLSLabsRenderer** (manual check → `unreal_check.txt`).

---

## 3. Target users & use cases

Human persona job titles: UNKNOWN. Audience: small private team. Implied skill from dummy: can install a Windows app, optionally install Brush / LichtFeld / Postshot / Unreal 5.5.

| ID | Scenario |
|----|----------|
| P4-S1 | First-run Setup creates venv and checks NVIDIA driver |
| P4-S2 | Download models (Quality) after install into the chosen folder: Hub cache `.hf_cache`, ready files in `models` |
| P4-S3 | Crash mid-job → Reload path / Reload 3DGS from disk (port V1) |
| P4-S4 | Splat: LiteGS or MCMC Compact, budget, Pause, Continue, Open Explorer / Brush / LichtFeld / Postshot |
| P4-S5 | Missing trainer exe → open the **docs URL**, do not invent install flags |
| P4-S6 | Import ply into UE 5.5 MLSLabsRenderer; MoGe `pointcloud.ply` is **not** the Unreal asset |
| P4-S7 | Home Help = full V2 field guide; other pages = that stage’s recipe only |

### User stories (from dummy)

- As a user I can pick Quality / Fast / Scout on Home once P4 unlocks Fast/Scout.
- As a user I can Continue train or Retrain with LiteGS or MCMC Compact, and Pause a run without losing it.
- As a user I can open the COLMAP folder without Brush installed.
- As a user I never enter an HF token for the default pack.

---

## 4. Functional requirements

### 4.1 Profiles

| Profile | WAN | HiRes | Rails | Notes |
|---------|-----|-------|--------|--------|
| **Quality** (locked default, 3090) | 14B fp8 720p + LightX2V | 8K geometry | 4 | Locked |
| Fast | **Matrix-3D 5B** (default). GGUF Q4 is not the Fast default. | 8K optional | 2–4 | Exact 5B filename from Matrix-3D README at implement — do not invent. |
| Scout | 5B or skip WAN | skip | 1 | Plumbing / path check |

P0 Home shows Fast/Scout **disabled** until this phase.

Gaussian cap ~**3M**; document why 1M is wrong for 8K datasets (SplatKit HiRes doc).

### 4.2 Installer

Inno Setup (locked 2026-09-01). Do not use NSIS unless a later lock changes this.

Start Menu:

1. Votion 3DGS (`tools\launch.bat`)
2. Download models
3. Setup (create venv, check NVIDIA driver)

Install dir default: `D:\Votion3DGS` (space-free). Also **test a path with spaces**. That chosen folder **is** `<install_root>`: Hub cache `.hf_cache\` and ready `models\` live there.

Small relative to the model pack.

### 4.3 Splat page (Open in…)

Dummy: [Phase4.html §03](../docs/Votion_3DGS_2.0_Phase4.html).

- **LiteGS | MCMC Compact** chips (same rules as [Phase 2 PRD](Phase2_First_Splat_PRD.md) §4.6 — live raster over shared memory, Star camera, uncapped max steps, editable Gaussian budget, Pause / Stop, MCMC Compact = same Gaussians). LiteGS is built locally at Setup (Inria licence; never shipped in the installer); Help shows the licence line.
- Continue train (with optimizer state) / Pause / **Stop** / Reload 3DGS from disk
- Open `outputs/<scene>/colmap` in Explorer
- **Brush** — [github.com/ArthurBrussee/brush](https://github.com/ArthurBrussee/brush) if `brush.exe` on PATH or Settings path. Field-guide click path: Brush → Directory → `outputs/<scene>` → Start. Live cloud on top, training frame below.
- **LichtFeld Studio** — [LichtFeld-Studio](https://github.com/MrNeRF/LichtFeld-Studio); primitive cap ~3M
- **Postshot** — [jawset.com](https://www.jawset.com/)

If exe missing: open docs URL, do not invent install flags.

### 4.4 Unreal (interim lock)

Engine **5.5** + **MLSLabsRenderer** for real-time `splat.ply` (3DGS / 4DGS).

- GitHub: [mlslabs/MLSLabsGaussianSplattingRenderer-UE](https://github.com/mlslabs/MLSLabsGaussianSplattingRenderer-UE)
- Fab: [listing](https://www.fab.com/listings/f91b57cc-958d-40dd-a455-2535bf00e588)

Workflow: copy `Plugins/MLSLabsRenderer` per their README, import `splat.ply`, place Gaussian actor. Field-guide recipe “first Unreal splat” points here.

V1 Luma / NanoGS: **superseded**. Do not document as current.

**Update slot — UE Renderer:** empty placeholder. If a better player appears, update Index certainty log. Until then MLSLabs stays.

### 4.5 Field guide (Help drawer)

Placement locked 2026-08-31: **Help drawer** at the bottom, same chrome as Log. Not a Home accordion, not a stage chip, not a seventh page.

Copy:

| Page | Help |
|------|------|
| Home | Full V2 guide |
| Panorama | Graph MarkdownNote id 4 with Votion labels |
| Geometry / Generate / Reconstruct / Splat | That stage’s recipe only |

Content bullets to port: [Phase4.md field guide](../docs/Votion_3DGS_2.0_Phase4.md#field-guide-in-app) (inputs, Ostris READY, trigger prefix, h_fov type-or-drag 10–170, surroundings suffix `no peoples, no cars`, spherical 360 look, WAN prompt must match pano, look modes, preview flight, 8K, glass, why 3DGS, Brush path, LiteGS/MCMC Compact + licence, Unreal recipe, corridor / pond / crane).

### 4.6 Settings (`D:\Votion3DGS\config.json`)

- GPU index
- ComfyUI root (default `D:\ComfyUI_windows_portable_360`) for copy-if-exists
- Paths to Brush / LichtFeld / Postshot
- Last scene name
- Profile
- Last splat strategy (`splat3` or `mcmc`)

Do **not** store HF tokens.

Crash reload: port V1 “Reload path from disk” / “Reload 3DGS from disk”.

### 4.7 Acceptance criteria

- [ ] Fresh Win11 + 3090: Setup → Download Quality models → Launch → 2D Images (or 360 / text) → 8K ERP → rails → Generate → `splat.ply` without WSL or ComfyUI. Ostris is READY for 2D.
- [ ] Splat page: LiteGS (default) and MCMC Compact both train to `splat.ply` (full SH); switching requires Retrain; Help shows the LiteGS licence line.
- [ ] Fast/Scout selectable; Scout skips HiRes.
- [ ] Kill mid-WAN: cancel works; scene reloadable.
- [ ] Open-in-Explorer works even if Brush is not installed.
- [ ] Installer is small relative to the model pack.
- [ ] After Download models, Hub cache is under the chosen install dir (`.hf_cache`), not `C:\Users\<user>\.cache\huggingface`.
- [ ] `splat.ply` imports and plays in UE 5.5 via MLSLabsRenderer (manual; `unreal_check.txt`).

### 4.8 Not this phase (P5)

Matrix-3D 5B is the **2.0 Fast default** (locked 2026-09-01). SeedVR2 experiment only, macOS, Matrix-3D opt reconstruction — [Index P5](../docs/Votion_3DGS_2.0_Index.md#p5-stretch-not-a-20-blocker).

---

## 5. Non-functional requirements

| Topic | Locked | UNKNOWN |
|-------|--------|---------|
| Install size | Code + bootstrap; weights later | Max MB of the installer |
| Security | No token UI; no tokens in config.json; **no telemetry** — fully local (locked 2026-09-01) | Code signing, SmartScreen (not specified) |
| Reliability | Cancel + crash reload | Auto-update channel |
| Support OS | Windows (Space-in-path tested) | Windows 10 vs 11 only; macOS is P5 |
| Localization | English dummy | Other languages |

---

## 6. UX/UI references

| Artifact | Use |
|----------|-----|
| [Phase4.html Open in dummy](../docs/Votion_3DGS_2.0_Phase4.html) | LiteGS/MCMC Compact + Brush/LichtFeld/Postshot |
| [Phase4.html Home settings](../docs/Votion_3DGS_2.0_Phase4.html) | Profile dropdown, config.json hint, Help = field guide |
| `docs/media/brush-og.png`, `lichtfeld-*.jpg/webp`, `mlslabs-*.jpg` | External trainer / Unreal stills |
| [rail-dummy.js](../docs/rail-dummy.js) | LiteGS/MCMC Compact chips on this page too |

---

## 7. Technical constraints & dependencies

- Same venv as P0 (3.12 + cu128)
- Public HF only for default manifest
- Unreal **5.5** + MLSLabsRenderer (user-installed, not bundled)
- Brush / LichtFeld / Postshot optional
- Inno Setup (locked 2026-09-01)

---

## 8. Success metrics

Done = §4.7. No extra KPIs/OKRs (locked 2026-09-01). Engineering bar: one 3090 machine completes Quality loop without WSL/Comfy.

---

## 9. Timeline & milestones

**No calendar dates.** 2.0 launch **is** this phase’s exit (Inno + field guide + Open in… + crash reload + Fast/Scout unlocked), not P0.

---

## Open questions (this phase)

None that are product decisions. Remaining work is implementation of Inno + the P0–P3 path. GitHub Pages dummy may be used as the team preview; do not invent a second product site.
