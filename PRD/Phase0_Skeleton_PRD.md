# PRD — Phase 0: Desktop skeleton

**Product:** Votion 3DGS 2.0  
**Phase:** 0 of 4 (build this first)  
**Depends on:** nothing  
**Unlocks:** Phase 1 panorama  
**Out of scope:** diffusion, MoGe, WAN, SfM, splat train  

**Sources (do not drift):**

- Dummy UI: [docs/Votion_3DGS_2.0_Phase0.html](../docs/Votion_3DGS_2.0_Phase0.html) · [live](https://titanothereyik.github.io/votion-3dgs-2-html/Votion_3DGS_2.0_Phase0.html)
- Spec: [docs/Votion_3DGS_2.0_Phase0.md](../docs/Votion_3DGS_2.0_Phase0.md)
- Certainty log: [docs/Votion_3DGS_2.0_Index.md](../docs/Votion_3DGS_2.0_Index.md)
- Set index: [README.md](README.md)

If HTML dummy and this PRD disagree, the dummy + certainty log win.

---

## 1. Product overview

| Field | Value | Status |
|-------|--------|--------|
| Name | **Votion 3DGS** | Locked |
| Owner / publisher | **Yik** (personal) | Locked 2026-09-01 |
| First audience | Small private team | Locked 2026-09-01 |
| License | **MIT** | Locked 2026-09-01 |
| Version string | **2.0** only (no `x.y.z`) | Locked 2026-09-01 |
| Code home | `D:\Votion3DGS\` | Locked |
| High-level goal (this phase) | A native Windows window that starts, shows model status, downloads Hugging Face weights into `<install_root>\models` (Hub cache in `<install_root>\.hf_cache`), and runs a **cancelable subprocess job queue**. No browser, no WSL, no ComfyUI UI. This phase does **not** generate panoramas or splats. | Locked |

**Stack (locked):** independent venv Python **3.12** + CUDA **12.8** + PyTorch **cu128** wheel. Do **not** reuse `D:\ComfyUI_windows_portable_360\python_embeded` (measured: Python 3.13 + torch `2.13.0+cu130`). Exact PyTorch *patch* is UNKNOWN until first successful install — write it to `requirements.lock`.

**Visual:** port tokens from V1 `Yik Votion WorldFM/ui/theme.css` (JetBrains Mono, dark cyanotype). Do not invent a new brand palette.

---

## 2. Problem & objectives

**Problem:** V1 (Yik Votion WorldFM) is Gradio + WSL + WorldFM. V2 must be a native desktop engine that never shells to ComfyUI. Before any pano/splat work, the app must exist as a real window with a real downloader and a real cancelable job bus.

**Objectives (this phase):**

1. Double-click launch opens a desktop window titled **Votion 3DGS**.
2. Weights lookup: ComfyUI models folder first, else Hugging Face, else `MISSING_<id>` with the Index URL (do not invent a second URL).
3. Qt process never `import torch` for inference. Workers are `python -m engine.workers.*`.
4. Record the real environment in `docs/env_report.txt` (anti-hallucination).

**Business / marketing goal:** UNKNOWN (no public launch; small private team).

---

## 3. Target users & use cases

**Hardware persona (locked):** RTX 3090 24GB, Windows, CUDA device index default **0**.  
**First audience (locked 2026-09-01):** small private team. Owner: **Yik**.  
**Job titles / company names beyond that:** UNKNOWN.

### Scenarios this phase must support

| ID | Scenario | Outcome |
|----|----------|---------|
| P0-S1 | User launches `tools\launch.bat` | Native window, not a browser, not `localhost:7860` |
| P0-S2 | User sets scene name (dummy: `alpine_01`) | Name is the future `outputs/<scene>/` stem |
| P0-S3 | User clicks **Download models** with **P1 subset** (default) | Krea 2 + LoRAs + VAE + CLIP + RealESRGAN start; WAN 14B is not forced. Hub cache lands in `<install_root>\.hf_cache`, ready files in `<install_root>\models`. Each status cell shows its own progress bar |
| P0-S4 | User clicks **All** | Full Index manifest (including Phase 3 WAN files) |
| P0-S5 | User clicks **Stop** during download or dummy job | Subprocess ends; log shows cancel |
| P0-S6 | Required file missing on HF and not in Comfy | Status `MISSING_<id>` + Index URL, no hang |

### User stories (derived from dummy, not a new lock)

- As a user I can see which model files are `MISSING_*` vs `READY` without opening Explorer.
- As a user I can see a progress bar on each status cell while a download runs.
- As a user I can download a P1 subset so I am not forced to pull ~14B WAN before Phase 3.
- As a user I can stop a running job.
- As a user I can read Home Help in a bottom drawer (not a seventh page).

Fast / Scout profile rows stay **disabled** until Phase 4. Panorama Generate stays **disabled** until Phase 1.

---

## 4. Functional requirements

### 4.1 Window chrome

Must match [Phase 0 HTML §02](../docs/Votion_3DGS_2.0_Phase0.html#home):

- Title **Votion 3DGS**, brand row, live status (`idle` until a job)
- Stage chips: Home (on) / Panorama / Geometry / Generate / Reconstruct / Splat — **navigate** even if later pages are stubs
- **Job strip** under the chips (caption + bar) on every page, including dummy jobs and later workers. Locked 2026-09-03.
- Left rail: Scene name, Profile (Quality selected; Fast/Scout disabled), CUDA index `0`, download set **P1 subset** vs **All**, **Download models**, **Stop**
- Right viewport: model status grid. During **Download models**, **each cell has its own progress bar** (queued empty; active file shows % from measured bytes).
- Log drawer + Help drawer

Six named pages exist as stubs except Home. Page names are the P0 list — do not rename chips.

### 4.2 Deliverables

From [Phase0.md deliverables](../docs/Votion_3DGS_2.0_Phase0.md):

1. Folder tree from the Index (empty `engine` packages with `__init__.py` is enough).
2. PySide6 main window as above.
3. `tools/model_manifest.json` — every file from Index tables: id, repo, filename, local relative path, `public: true`. **No invented file sizes.**
4. `tools/download_models.py` — copy-if-exists from `D:\ComfyUI_windows_portable_360\ComfyUI\models\<folder>\<filename>`, else `huggingface_hub` into `<install_root>\.hf_cache` then hardlink/copy to `<install_root>\models\<folder>\`. Each Home status cell shows a progress bar.
5. Status machine: `MISSING_*` | `READY`. Subset READY is OK in P0.
6. Job bus: start worker, stream stdout, cancel (port `Yik Votion WorldFM/ui/job_control.py`). The strip under the chips shows job progress (`PROGRESS` lines), not log-only.
7. `tools/env_report.py` → `docs/env_report.txt`: `nvidia-smi`, `python --version`, `torch.__version__`, `torch.version.cuda`.
8. Copy of phase docs under `docs\`.

### 4.3 Downloader order

Per manifest entry:

1. `D:\ComfyUI_windows_portable_360\ComfyUI\models\<folder>\<filename>`
2. If found: copy (or hardlink if same drive) to `<install_root>\models\<folder>\<filename>`
3. Else: HF download into `<install_root>\.hf_cache\` then hardlink/copy to that dest. Never `C:\Users\<user>\.cache\huggingface`.
4. Else: `MISSING_<id>` + Index URL

Inventory **2026-08-31**: that Comfy `models\` had **none** of the required `.safetensors`. Copy-from-Comfy is a no-op until those files appear.

**P1 subset (default):** Krea 2 + LoRAs + VAE + CLIP + RealESRGAN; Florence-2 if captioning will be used.  
**P2 subset** (MoGe `model.pt`) may wait until Phase 2.  
**SphereSfM binary** may wait until Phase 2.

### 4.4 CUDA / gsplat spike (document, do not fake)

After torch installs: `torch.cuda.is_available()` on the 3090.

Then try `gsplat`. UNKNOWN whether a Windows wheel exists for 3.12 + cu128. Record in `env_report.txt`:

- Wheel + 1-step dummy raster → P2 trainer unblocked
- Also record that `DefaultStrategy` (Splat3) and `MCMCStrategy` (MCMC) import
- Fail → P2 must say trainer blocked; COLMAP export still proceeds (Brush)

Do not claim gsplat in the UI until this spike passes.

> **Superseded 2026-09-07.** The trainer moved to a vendored **LiteGS** backend; the P2 Train gate is now the LiteGS spike (`tools/build_litegs.py` builds `simple-knn`, `fused_ssim`, `litegs_fused` locally via `engine/cuda_jit.prepare`, writes `litegs raster 1-step: OK` to `env_report.txt`; `status.litegs_spike_ok()`). Same fallback: fail → Train disabled, COLMAP folder still valid for Brush. See [Phase 2 PRD](Phase2_First_Splat_PRD.md) §4.6.

### 4.5 Dummy worker

P0 ships `engine.workers.dummy` that prints ~20 lines and exits, to prove cancel + log streaming.

### 4.6 Expected files

```
D:\Votion3DGS\
  app\main.py
  app\windows\main_window.py
  app\theme.qss
  engine\job.py
  engine\workers\dummy.py
  tools\download_models.py
  tools\model_manifest.json
  tools\env_report.py
  tools\launch.bat
```

### 4.7 Acceptance criteria

- [ ] Double-click `tools\launch.bat` opens a desktop window titled **Votion 3DGS** (not a browser).
- [ ] `env_report.txt` exists and shows Python 3.12, CUDA 12.8 torch, `cuda.is_available() True` on the 3090.
- [ ] Download of **Krea 2 fp8 + CLIP + VAE** reaches `models\` (or a clear missing-URL error — not a hang). MoGe may still be missing.
- [ ] A Hub download writes cache under `<install_root>\.hf_cache\` (P0: `D:\Votion3DGS\.hf_cache`), not `C:\Users\<user>\.cache\huggingface`.
- [ ] During Download models, each Home status cell shows its own progress bar.
- [ ] Background jobs (including the dummy worker) update the strip under the chips.
- [ ] Dummy job streams logs; Stop ends the subprocess.
- [ ] No WSL, no `localhost:7860`, no ComfyUI process.

### 4.8 Explicitly do not

- Do not wrap Gradio.
- Do not call WorldFM / HunyuanWorld.
- Do not vendor the full SplatKit tree until P2 (only job/theme).
- Do not pin a fictional torch version in docs after install — write the real one.

---

## 5. Non-functional requirements

| Topic | Locked | UNKNOWN (do not invent) |
|-------|--------|-------------------------|
| Performance | Qt does not hold torch for inference. Downloader checks free space before WAN 14B. | Throughput, UI fps, download concurrency |
| Security | No HF token UI. Do not store HF tokens. **No telemetry** — fully local, no phoning home (locked 2026-09-01). | Code signing / SmartScreen (not specified) |
| Scalability | Single local GPU index | Multi-GPU, multi-machine |
| Reliability | Cancelable jobs; env_report as the truth file | Auto-update, watchdog |
| Localization | English dummy copy | Other languages |
| Accessibility | — | Screen reader, high contrast |

If Python 3.12 is missing: **stop and install it** — do not silently use Comfy 3.13. **This PC already has 3.12.10** (measured 2026-09-01).  
If NVIDIA driver cannot satisfy CUDA 12.8: stop after `nvidia-smi` in env_report.

---

## 6. UX/UI references

| Artifact | Use |
|----------|-----|
| [Phase0.html §02 Home dummy](../docs/Votion_3DGS_2.0_Phase0.html#home) | Pixel-level chrome, rail widgets, status grid |
| [votion-plan.css](../docs/votion-plan.css) | Colors, type, drawers |
| V1 `ui/theme.css` | Tokens to port into `app/theme.qss` |
| [Index.html](../docs/Votion_3DGS_2.0_Index.html) | Product-level happy-path dummy |

**Flow:** Launch → Home idle → (optional) change scene / GPU → Download P1 subset → per-cell bars + log stream → READY rows appear → Stop if needed.

---

## 7. Technical constraints & dependencies

| Item | Constraint |
|------|------------|
| OS | Native Windows (tested intent: Windows 11 + 3090) |
| Python | 3.12 only |
| CUDA / torch | 12.8 / cu128 official wheel from pytorch.org **at install time** |
| GUI | PySide6 |
| HF | `huggingface_hub` for public files |
| ComfyUI | **Weights copy source only.** Root `D:\ComfyUI_windows_portable_360`. Never the product UI, never `python_embeded` |
| Optional later | `triton-windows`, SageAttention — try; do not block P0 |

**Manifest URLs:** copy from [Index model tables](../docs/Votion_3DGS_2.0_Index.md#model-filenames-from-the-two-comfyui-graphs-you-attached). Do not invent sizes or extra files. Optional nvfp4 Krea is for RTX 50xx — **not** the 3090 Quality default.

**Risks:** gsplat vs 3.12/cu128 still unknown until spike. Python **3.12.10** is installed. Driver **595.79** (nvidia-smi CUDA 13.2). `D:` free **142.3 GB**.

---

## 8. Success metrics

**Locked definition of done:** the acceptance checklist in §4.7. No extra KPIs/OKRs (locked 2026-09-01).

---

## 9. Timeline & milestones

| Milestone | Meaning | Date |
|-----------|---------|------|
| 0.1–0.6 as in Phase0.md | Build order | No calendar — do these in listed order |

Launch of an installable product is **Phase 4**, not this phase. **No calendar dates** — phase order is the schedule (locked 2026-09-01).

---

**About box:** product **Votion 3DGS**, version **2.0**, owner **Yik**, license **MIT**.

## Open questions (this phase)

Spikes only: gsplat wheel on 3.12+cu128; torch patch written to `requirements.lock` after first install.
