# Review of the H3 plan, and better options for the WAN hole fill

**Status:** plan only. No project file is changed. [H3_WAN_Replacement_Plan.md](H3_WAN_Replacement_Plan.md) is left as written; section 2 lists what in it is wrong or stale.  
**Date:** 2026-10-26.  
**Machine:** RTX 3090 24 GB (Quality profile lock).  
**Feature under review:** Phase 3 masked panoramic fill: Plot Camera mesh render + validity mask → video model → 2:1 ERP clip with only the holes invented → HiRes 8K composite → SphereSfM. See [Votion_3DGS_2.0_Phase3.md](Votion_3DGS_2.0_Phase3.md).

---

## 1. Verdict

MiniMax H3 is **not** the best replacement for WAN on this feature. It can be made to run, but it is a general perspective model that has never been trained on mesh renders, validity masks, or equirect frames. It also carries the strictest licence on the list (no US, EU, UK, or Korea), and it emits a soundtrack the pipeline throws away.

The current path is stronger than the H3 plan gave it credit for. The Matrix-3D pano LoRA on Wan 2.1 was trained on **exactly** this conditioning: mesh render plus binary validity mask, on equirect panoramas, from 116,759 UE5 trajectories (Matrix-Pano). None of the generic inpainters (H3 Fun ControlNet, LTX-2.5 IC-LoRA, Wan VACE) were.

The measured weakness in this project is not per-clip quality. It is **disagreement between rails**: `alpine_01` frame 1 alone fits to 30.8 dB, frames 1–3 together only 22.3 dB, hold-out 15.6 dB ([index measured table](Votion_3DGS_2.0_Index.md)). Swapping one per-rail inpainter for another does not address that. A replacement is only “better” if it keeps the trained render+mask behaviour **and** shares memory across rails.

Ranked recommendation:

| Rank | Option | Why |
|------|--------|-----|
| 1 | **Keep WAN + pano LoRA, add progressive rail memory** (section 5) | No new model or licence. Directly targets the measured cross-rail inconsistency. |
| 2 | **Spike WorldStereo 2.0** (HY-World 2.0’s World Expansion model) against WAN | Only candidate trained on render + mask from a panorama **with** a cross-trajectory memory bank. A community FP8 build is listed as resident on 24 GB. Needs the owner to reopen a recorded rejection and accept a territory licence. |
| 3 | **Wan360** for the Fast / Scout profiles | Apache-2.0, ERP-native, 5B, follows camera poses. It has no mask input, so it repaints whole frames. It is a Matrix-3D 5B challenger, not a Quality replacement. |
| 4 | H3 + Fun ControlNet Union 2.0 | Runnable on a 3090 per a user report, but generic, licence-restricted, and slower. Bake-off entrant only. |
| 5 | Wan VACE / LTX-2.5 in-outpaint | Generic masked inpainters. VACE is the Apache fallback if every other licence is refused. |

The same approaches, plus the current Fast model and everything that was looked at and set aside, are in [Comparison of all approaches](#comparison-of-all-approaches).

---

## Comparison of all approaches

Every approach in this plan, on the same columns. “Yes” means it holds for this feature. “Verify” means the spike has to measure it. Plot Camera can only emit lengths of `9 + 4k`. Detail for each row is in sections 4 and 5.

### Does it do the job

| Approach | Role | Trained on render + mask | 360° ERP output | Keeps known pixels | Shares memory across rails | Verdict |
|----------|------|--------------------------|-----------------|--------------------|----------------------------|---------|
| Wan 2.1 I2V 14B + Matrix-3D pano LoRA | Quality, today | Yes. Mesh render + mask, ERP, UE5 | Yes. 1440×720 | Yes. Masked latent concat | No. Each rail is sampled alone | Keep as the Quality baseline |
| Same WAN + progressive rail memory | Quality, first spike | Same weights | Yes. Same canvas | Yes. Later rails treat earlier fills as known | Yes. The hole mask shrinks rail by rail | Do this first. No new model or licence |
| WorldStereo 2.0, memory DMD, FP8 | Quality candidate | Yes. Point-cloud render + occlusion mask. A mesh render is a domain shift | Probably perspective keyframes. Verify | Yes, through `render_mask` | Yes. Panorama memory bank | Only trained model with cross-rail memory. Owner must reopen the HY-World rejection |
| MiniMax H3 + Fun ControlNet Union 2.0 | Bake-off only | No. Generic perspective inpaint and depth | 2:1 only if the input is 2:1. Poles and the seam are untrained | Yes, after the mask is inverted | No | Strictest licence. Zero-shot on this task |
| Wan360 | Fast / Scout challenger | No. One ERP plus a camera path. No mask | Yes, native. Trained at 1920×960 | No. It repaints the whole frame. HiRes pastes the photo back | No | Challenger to the 5B Fast model, not to Quality |
| Matrix-3D 5B pano LoRA | Fast today. Scout uses the low-VRAM build | Yes. Same recipe, smaller base | Yes. 720p LoRA | Yes. Same mask path as Quality | No | Stays the Fast default until Wan360 wins on time and joint fit |
| Wan 2.1 VACE 14B | Apache fallback | No. Generic control and inpaint | Whatever size it is fed. Not ERP-trained | Yes. Holes painted gray 127 | No | Use if every territory licence is refused |
| Wan 2.2 VACE-Fun A14B | Newer Apache inpaint | No | Whatever size it is fed. Not ERP-trained | Yes. Same VACE mask rule | No | Same role as Wan 2.1 VACE, with two 14B experts |
| LTX-2.5 + in-outpaint IC-LoRA | Not recommended | No. Generic perspective inpaint | Whatever size it is fed. Not ERP-trained | Yes. Laplacian blend on the mask border | No | The border blend is the useful idea. The model is not a reason to switch |
| Retrain the pano LoRA on Wan 2.2 or VACE | Not available | It would be, on Matrix-Pano | It would be | It would be | It would not, unless memory is added separately | Cannot start. Matrix-Pano is still unreleased |

### Can it run here

| Approach | Canvas × frames × fps | Length on `9 + 4k` | 3090 | Steps | Audio | Licence |
|----------|----------------------|--------------------|------|-------|-------|---------|
| Wan 2.1 + pano LoRA | 1440×720 × 81 × 16 | Yes. 81 | Running today | 8, with LightX2V | No | Apache-2.0 base. Matrix-3D code MIT. Weights gated behind a contact form |
| WAN + progressive rail memory | Same as WAN | Yes. 81 | Same budget, plus one MoGe pass per rail | 8 | No | No new licence |
| WorldStereo 2.0 memory DMD | Record it on the spike | Verify | FP8 DMD is 20.35 GB and stays resident. BF16 DMD is 34.9 GB and does not fit. The memory variant is about 22 GB. The camera-only variant is 10.9 GB and has no memory bank | 4, DMD | No | Excludes the EU, the UK, and Korea. Over 1M monthly users needs Tencent’s approval |
| H3 + Union 2.0 | 1408×704 × 141 × 24 | Yes. 141. 73 is on the grid but below H3’s trained range. 124 is not on the grid | About 20 GiB with pose control, one report. About 18 min for a 12 s clip. Union 2.0 itself is 4.53 GB INT8 | 20–40. Turbo builds claim 4 | Yes. Strip it | Excludes the US, the EU, the UK, and Korea. Credit line. Over $20M revenue needs authorisation |
| Wan360 | 1920×960 × 161 × 16 | Yes. 161 | 5B. Expected to fit. Measure it | Per the MUGEN repo | No | Apache-2.0 weights. The MUGEN dataset is research-only and is not needed |
| Matrix-3D 5B | Same 81 × 16 contract, 720p LoRA | Yes. 81 | Fast is the 16 GB-class 5B. Scout is the README’s ~12 GB low-VRAM build | Same family as Quality. Measure | No | Same as the 14B LoRA |
| Wan 2.1 VACE 14B | About 720p × 81 × 16 | Yes. 81 | Q8_0 GGUF listed at 23.5 GB. Tight | About 20 | No | Apache-2.0. The pano LoRA does not load: VACE is T2V, the LoRA is I2V |
| Wan 2.2 VACE-Fun A14B | 81 × 16 | Yes. 81 | Two 14B experts. INT8 builds exist. Measure | Lightning LoRA claims 4 | No | Apache-family |
| LTX-2.5 + IC-LoRA | Not fixed for this plan | Verify | 22B plus a Gemma 4 12B encoder. 24–32 GB with FP8 or FP4. Tight | 8 low-res, then 2 high-res | Yes. The audio stream can be frozen out | Free under $10M yearly revenue |
| Retrain the pano LoRA | — | — | Training is a multi-GPU job, not a 3090 inference job | — | — | Blocked on the dataset, not on a territory clause |

### What the pipeline has to feed it

| Approach | Inputs | White in the mask means | Change if it wins |
|----------|--------|-------------------------|-------------------|
| Wan 2.1 + pano LoRA | Mesh render, validity mask, CLIP-Vision of the panorama | Known. Leave it | None. This is the current path |
| WAN + progressive rail memory | The same, plus geometry lifted from earlier rails | Known. The mask gets smaller each rail | Keep / Ditch must re-run every rail after the one you ditch. HiRes needs a class for “filled by an earlier rail” |
| WorldStereo 2.0 | `render.mp4`, `render_mask.mp4`, `camera.json` (OpenCV w2c), memory bank | Verify | May replace SphereSfM for these frames if the output is perspective with known poses. HiRes would need a perspective reprojection |
| H3 + Union 2.0 | Source video, inverted mask, depth. Control video must match width, height, and frame count exactly | Regenerate | New worker, 24 fps, HiRes stride re-specced off 141 frames (`0-140/2` is 71 frames) |
| Wan360 | One ERP panorama and a camera trajectory (panoramic Plücker rays) | No mask input | Fast / Scout only. Hole borders may not meet MoGe’s geometry |
| Matrix-3D 5B | Same as the 14B Quality path | Known | Already the Fast lock. No new wiring |
| Wan 2.1 VACE 14B | Control video, mask, reference image. Holes painted 127 gray | Regenerate | New conditioning pack. LoRA shelf for the pano LoRA does not carry over |
| Wan 2.2 VACE-Fun A14B | Same VACE inputs, high-noise and low-noise experts | Regenerate | Same as Wan 2.1 VACE, with two checkpoints |
| LTX-2.5 + IC-LoRA | Source video and a mask video, plus the Gemma encoder | Inpaint | New runtime and a 12B text encoder. Audio VAE is in the graph |
| Retrain the pano LoRA | A Matrix-Pano-style set: mesh render, mask, ERP video, poses | Known, if the current pack is kept | No port until the dataset exists |

### Looked at and set aside

| Approach | Why it is not in the bake-off |
|----------|-------------------------------|
| Wan 2.5, 2.6, 3.0 | API only. No weights to run locally |
| Fal, MiniMax API, Comfy Router hosted H3 | No mask field, and it leaves the machine. Breaks local-only |
| H3 360 LoRAs (shamanic, rehan) | Text-to-360 at 21:9. They ignore the mesh render and the rail |
| Fun ControlNet v1 (`pre_norm`) | Superseded for this GPU. ComfyUI 0.38 loads Union 2.0 natively |
| WorldStereo camera-only (10.9 GB) | Fits easily, and drops the memory bank, which is the reason to consider WorldStereo |
| World-Shaper | Panorama image editor. It can inpaint one ERP view. It has no temporal model |
| SphereForge | Rebuilds real 360 footage, then inpaints with Stable Diffusion after the splat. Not a video fill |

---

## 2. Corrections to `H3_WAN_Replacement_Plan.md`

These change numbers someone would act on. Apply them before that plan is used for a spike.

### 2.1 The frame-length table is wrong

The H3 plan says lengths that satisfy both Plot Camera and H3 are **56, 124, 192**, and picks 124. Plot Camera lengths in this codebase are `9 + 4k` (`engine/geo/rail.py`; `engine/geo/defaults.py`: min 9, step 4, max 257). 124 is not on that grid, so the Geometry page cannot produce it.

`17n + 5` lands on `9 + 4k` only when `n` is a multiple of 4:

| Frames | H3 grid | Plot Camera grid | Seconds at 24 fps | Inside H3 trained range (~124–362)? |
|--------|---------|------------------|-------------------|-------------------------------------|
| 73 | 17·4+5 | 9+4·16 | 3.04 | No |
| **141** | 17·8+5 | 9+4·33 | 5.875 | **Yes** |
| 209 | 17·12+5 | 9+4·50 | 8.71 | Yes |

**141** is the H3 rail length that needs no grid change. WAN’s 81 frames would snap down to 73 under H3’s rule, which is below its trained range. HiRes `0-80/2` (41 frames) has no meaning on 141 frames. A same-density spec would be `0-140/2` (71 frames, about 1.7× today’s SfM matching work) or a sparser stride. That is an owner choice, not a default to slip in.

### 2.2 Union 2.0 is no longer stuck at 80 GB

The H3 plan says the 3090-runnable path is the older Fun ControlNet v1 (`pre_norm`, dark holes). That is stale:

- ComfyUI **0.38.0** (2026-09-29) loads **Union 2.0** natively. Comfy-Org repack: **4.53 GB INT8**, 8.38 GB BF16 (GenVidKit H3 ControlNet write-up). The 13.5 GB figure is the original Alibaba file.
- The `post_norm` mid-gray hole recipe (the one that “follows Wan 2.1”) is therefore available on this GPU class.

### 2.3 There is a 3090 data point now

A user report (r/StableDiffusion, 2026-09-17, “MiniMax H3 pose ControlNet on a single 3090”), pose control rather than depth inpaint, so treat it as a lower bound:

| | Value |
|--|------|
| GPU | RTX 3090 24 GB |
| Canvas, length | 1312×736, 294 frames (12.25 s) |
| Peak VRAM | **16.2 GiB** without control, **20.0 GiB** with |
| Wall time | **~14–15 min** without, **~18 min** with, per clip |
| Gotchas | Only the pruned “curve-form” checkpoint loads (`adaln_proj.linear` `[96768, 8]`). The control video must match the generation’s **exact** width, height, and frame count or the node hard-fails. |

So the H3 plan’s main VRAM gate is probably passable. The real costs are time (four rails is about an hour) and that exact-shape rule: the worker must emit 1408×704×141 control and mask videos, not rely on snapping.

[Wan2GP](https://github.com/deepbeepmeep/Wan2GP/blob/main/docs/MODELS.md) also ships an H3 FL2VA ControlNet-Union Pruned (20B) with inpaint and outpaint, plus tiling and memory profiles. It is a second local runner to cross-check VRAM against Comfy.

### 2.4 The plan underweights what the pano LoRA already does

The H3 plan frames WAN as “a model plus a mask pack.” Matrix-3D’s paper says the opposite matters most: the LoRA is trained on **mesh render + mask** guidance, and the authors chose mesh over point-cloud renders specifically because point clouds give Moiré and wrong occlusion. H3 Fun ControlNet was trained on generic perspective control and inpaint clips. A spike has to measure ERP poles, the back seam, and disocclusion edges against WAN on the **same** rail before anything else, because that is where an untrained model loses.

### 2.5 Licence context it should state

| Model family | Excluded territories | Other limits |
|--------------|---------------------|--------------|
| MiniMax H3 + Fun ControlNet + the equirect LoRAs | US, EU, UK, Korea | Credit “Powered by MiniMax H3”; >$20 M revenue needs authorisation |
| Tencent HY-World 2.0 (WorldStereo 2.0) | EU, UK, Korea | >1 M MAU needs Tencent approval |
| LTX-2.x | none | Free under $10 M yearly revenue |
| Wan 2.1 / 2.2 / VACE / Wan360 weights | none | Apache-2.0 |
| Matrix-3D | none stated | Code MIT; HF weights gated behind a contact form |

H3 is the only candidate that excludes the US.

---

## 3. Selection criteria

A replacement is judged on these, in order:

1. **Trained on the task.** Render (mesh or point cloud) + validity mask in, filled frame out.
2. **Equirectangular.** The seam joins, poles collapse, and the output stays 2:1 for SphereSfM.
3. **Cross-rail consistency.** Rails that look at the same hole invent the same thing. This is the measured gap.
4. **Fits the 3090** at a canvas and length the Plot Camera grid can produce.
5. **Licence** the owner can accept for a local, private-team product.
6. **Keeps HiRes.** Native canvases top out near 1–2 MP. Sharpness remains the 8K geometry reprojection.

---

## 4. Candidates

### 4.1 Baseline: Wan 2.1 I2V 14B + Matrix-3D pano LoRA (current)

| Criterion | Status |
|-----------|--------|
| Trained on render + mask | **Yes.** Matrix-3D: mesh render + binary mask, invisible faces black, latent concat. SplatKit’s `WanI2VMaskedConditioning` reproduces it. |
| ERP | **Yes.** 1440×720. |
| Cross-rail | **No.** Each rail is sampled independently. Rail 2 does not know what rail 1 invented. |
| 3090 | Yes, running today. |
| Licence | Apache base; Matrix-3D code MIT, weights gated. |

No newer Matrix-3D release exists. The last update is the September 2025 5B low-VRAM model. The **Matrix-Pano** dataset needed to retrain the LoRA on a newer base is still “a curated subset will be open-sourced” ([issue #3](https://github.com/SkyworkAI/Matrix-3D/issues/3) open since 2025-08). Retraining the recipe on Wan 2.2 or VACE is not possible today without building a new UE5 dataset.

### 4.2 WorldStereo 2.0: HY-World 2.0 World Expansion

Sources: [HY-World 2.0 worldgen README](https://github.com/Tencent-Hunyuan/HY-World-2.0/blob/main/hyworld2/worldgen/README.md), [WorldStereo README](https://github.com/FuchengSu/WorldStereo/blob/main/README.md), [HF](https://huggingface.co/tencent/HY-World-2.0), [FP8 build](https://huggingface.co/apozz/WorldStereo-fp8), CVPR 2026.

| Criterion | Status |
|-----------|--------|
| Trained on render + mask | **Yes.** Wan transformer + camera embedding + ControlNet on point-cloud renders. Per-trajectory input is `render.mp4`, `render_mask.mp4` (binary occlusion), `camera.json` (OpenCV w2c 4×4 + intrinsic). |
| ERP | `--task_type panorama` with `meta_info.json {"scene_type": "panorama"}` expands **from** a panorama. `camera.json` carries a pinhole intrinsic, so the trajectory output is probably **perspective** keyframes, not 2:1 ERP. **Verify.** |
| Cross-rail | **Yes, by design.** `PanoramaMemoryBank` retrieves reference panorama content across trajectories; Global-Geometric Memory feeds an incrementally updated point cloud; Spatial-Stereo Memory restricts attention to 3D correspondences. This is aimed squarely at the 22.3 dB problem. |
| 3090 | ~17B. Variants listed by the VNCCS loader: `worldstereo-camera` 10.9 GB transformer; `worldstereo-memory` ~22 GB; `worldstereo-memory-dmd` 34.9 GB BF16 (4-step DMD, the recommended mode). Community **FP8 DMD: 20.35 GB, resident on 24 GB**; no FP8 matmul speed-up on Ampere. Official examples use `torchrun --nproc_per_node=8 --fsdp`; the README documents a single-GPU path without FSDP. |
| Licence | Tencent HY-World 2.0 Community: excludes EU, UK, Korea; >1 M MAU needs approval. |

Three issues stop this from being a drop-in:

1. **Recorded rejection.** The index rejects “Hunyuan / HY-World 2.0: two huge models in memory at once; designed for four GPUs.” WorldStereo is only the expansion stage of that system, and the FP8 build answers the memory point. Reopening it is the owner’s call, written into the certainty log, not a side effect of this plan.
2. **Point cloud vs mesh.** WorldStereo was trained on point-cloud warps. Votion renders a MoGe mesh, which Matrix-3D says is cleaner. Feeding a mesh render is a domain shift: likely fine, but measure it.
3. **Perspective output.** If the panorama task emits pinhole keyframes with known poses, the downstream changes. HY-World trains 3DGS on those poses directly, without SfM. In Votion that would bypass SphereSfM and the gnomonic cube-face remap (index #57) for these frames, and HiRes would need a perspective reprojection mode. That is a larger change than swapping a sampler, and the spike must find out before anyone ports code.

### 4.3 Wan360 (MUGEN, SIGGRAPH Asia 2026)

Sources: [AlayaLab/MUGEN](https://github.com/AlayaLab/MUGEN), [paper](https://alaya-lab.github.io/MUGEN/assets/paper.pdf), [AlayaLab/Wan360](https://huggingface.co/AlayaLab/Wan360).

| Criterion | Status |
|-----------|--------|
| Trained on render + mask | **No.** Input is one ERP panorama, a prompt, and a camera trajectory (panoramic Plücker embedding on ERP rays). It does not take the mesh render or the validity mask. |
| ERP | **Yes, natively.** Periodic longitude RoPE (seam), ERP-aware circular padding (VAE edges), random roll yaw (orientation). Trained at **1920×960, 161 frames, 16 fps**, which is a larger ERP canvas than WAN’s 1440×720. |
| Cross-rail | No. |
| 3090 | 5B (Wan2.2-TI2V-5B lineage via Wan2.2-Fun-5B-Control-Camera). Expected to fit comfortably; measure. |
| Licence | Weights and model code **Apache-2.0**. The MUGEN dataset is non-commercial research only; Votion does not need the dataset. |
| Grid | 161 = 9 + 4·38, so it is already on the Plot Camera grid, and 16 fps matches today. |

Results reported in the paper: best FVD (476.6) and translation error (0.184) against OmniRoam, CubeComposer, DynamicScaler, and 4K4DGen. There is no head-to-head against Matrix-3D.

Because it ignores the mask, Wan360 repaints every pixel from its own idea of the scene, steered only by poses. HiRes still pastes the 8K photograph back where the mesh can answer, which is what HiRes already does for WAN. The risk is at hole borders: Wan360’s implied geometry is not MoGe’s geometry, so invented content may not line up with reprojected content. That is acceptable for the **Fast** profile (today: Matrix-3D 5B) and **Scout**, where speed and fit matter more. It is not the Quality path.

### 4.4 MiniMax H3 + Fun ControlNet Union 2.0

Covered in the H3 plan, with the corrections in section 2. Generic perspective inpainter, mask + depth control, ~20 GiB on a 3090 per one report, 24 fps on the `17n+5` grid (use 141 frames), audio stripped, US/EU/UK/KR excluded. Bake-off entrant at most.

### 4.5 Wan 2.1 VACE 14B / Wan 2.2 VACE-Fun A14B

| | |
|--|--|
| Inputs | `WanVaceToVideo`: control video (depth etc.), masks, reference image. Inpaint convention: mask set **and** the masked input painted 127 gray. |
| Contract | ~720p, multiples of 16, **81 frames at 16 fps**, which is identical to today’s WAN contract. |
| 3090 | 2.1 VACE 14B GGUF Q8_0 listed at 23.5 GB ([vramready](https://vramready.com/models/wan-2-1-vace-14b/)); 2.2 VACE-Fun is two 14B experts (high / low noise), INT8 convrot builds exist. |
| Licence | Apache-2.0. |
| Gap | Generic. VACE is built on Wan **T2V**, and the pano LoRA was trained on Wan **I2V**, so that LoRA does not load on VACE. Nothing ERP-specific is trained in. |

Use only if every other licence is refused: same frame contract, fully permissive, and a lot of community inpaint knowledge.

### 4.6 LTX-2.5 + In-Outpainting IC-LoRA

| | |
|--|--|
| Inputs | Source video + mask video (white = inpaint). Two-stage: 8-step low-res pass then 2-step high-res refine, Laplacian-pyramid blend at both stages ([LTX docs](https://docs.ltx.io/open-source-model/feature-guides/editing-effects/in-outpainting.md)). IC-LoRA `ltx-2.3-22b-ic-lora-in-outpainting-0.9` reused on LTX-2.5. Union depth control exists. |
| 3090 | 22B transformer + Gemma 4 12B encoder; 24–32 GB with FP8 / FP4 reported. Tight. |
| Licence | LTX-2.x Community, free under $10 M revenue. |
| Gap | Generic perspective. Audio VAE in the graph (frozen-audio stream can be removed). |

The Laplacian blend at mask borders is the most interesting idea here, because seam blending between kept and invented pixels is exactly what HiRes does later. It is not a reason to adopt the model.

### 4.7 Not candidates

- **Wan 2.5 / 2.6 / 3.0:** API only, no weights.
- **Fal / MiniMax / Comfy Router hosted H3:** no mask field; breaks local-only.
- **H3 360 LoRAs (shamanic, rehan):** text-to-360 at 21:9; ignore the mesh and the rail.
- **World-Shaper** (ICML 2026): panorama *image* editing that can inpaint rendered ERP views for a GS pipeline. Per-frame image model; no temporal model. Worth reading for its ERP-aware training; not a video replacement.
- **SphereForge:** reconstructs real 360 footage; inpaints with Stable Diffusion after GS. Not generative video.

---

## 5. The better way that needs no new model: progressive rail memory

WorldStereo’s Global-Geometric Memory is, at its core, “update the point cloud with what you already generated before rendering the next trajectory.” Votion can do a version of that around the existing WAN:

```
rail 0: mesh render + mask → WAN fill → frames F0
        lift F0's filled pixels into 3D:
          depth for F0 (MoGe on the filled ERP frames),
          scale/shift-aligned to the mesh depth on known pixels,
          keep only pixels the mask marked as holes
        append them to the scene geometry  (new triangles or points, tagged "generated, rail 0")
rail 1: re-render mesh + generated geometry along rail 1
        → mask shrinks where rail 0 already invented content
        → WAN fills only what no rail has seen yet
...
```

Why it fits:

- Targets the measured failure directly: overlapping holes stop being invented twice, differently.
- Uses only code families the project already ships: MoGe, Plot Camera rendering, the mask pipeline, HiRes.
- Model-agnostic. If WorldStereo or anything else later wins the bake-off, the same loop still helps.

Costs and risks (no published Votion measurement exists; these are the things to measure):

- **Rail order becomes meaningful.** Rail 0’s mistakes become rail 1’s “known” pixels. The Keep / Ditch UI must allow ditching a rail and re-running the ones after it.
- **Depth alignment of generated pixels.** WAN fills are 720p and soft; MoGe on them is noisier than on the 8K pano. Use a confidence threshold and leave low-confidence pixels as holes.
- **HiRes gate meaning.** HiRes `gate_masks` today mean “white = 8K photograph, black = WAN.” Pixels lifted from another rail are neither. Add a third class or keep them black so they stay WAN-coloured in the composite.
- **Re-gen.** Re-generating rail k invalidates rails after k. Generate page needs that dependency spelled out.

This is the first spike because it needs no new licence, no new weights, and no change to the Quality GPU contract.

---

## 6. Bake-off plan

Same scene (`alpine_02`, which has four rails, WAN clips, HiRes, and a splat), same rails, same `prompt.txt`, same seed where meaningful. Everything below runs outside the app in a scratch folder. No change to `engine/video/` or the model manifest.

### 6.1 Entrants

| Id | Entrant | Canvas × frames × fps | Runner |
|----|---------|----------------------|--------|
| W | WAN 2.1 + pano LoRA (baseline, existing outputs) | 1440×720 × 81 × 16 | already done |
| W+M | W with progressive rail memory (section 5) | same | scripted outside the app |
| WS | WorldStereo-memory-dmd FP8, panorama task, Votion mesh render as `render.mp4` | per WorldStereo; record | WorldStereo single-GPU path |
| H3 | H3 FL2VA pruned + Union 2.0, depth + inverted mask | 1408×704 × **141** × 24 | ComfyUI ≥ 0.38 |
| W360 | Wan360 (Fast / Scout only) | 1920×960 × 161 × 16 | MUGEN inference |

WS runs only if the owner reopens HY-World and accepts its territory. H3 runs only if the owner accepts its territory. W, W+M, and W360 have no licence gate.

### 6.2 Per-rail measurements

| Metric | Definition |
|--------|------------|
| Peak VRAM, wall time | per rail, on the 3090 |
| Known-pixel preservation | mean absolute error between output and control frame where the validity mask is white |
| Back seam | correlation of leftmost vs rightmost column (rehan’s definition; real 360 footage ~0.92) |
| Hole plausibility | three fixed hole crops per rail, side by side, reviewed by eye |
| HiRes coverage | fraction of white `gate_masks` after the existing HiRes composite |

### 6.3 The measurement that decides

Run SphereSfM and the LiteGS trainer on all four kept rails per entrant, same schedule as the `alpine_01` measurements:

- **Hold-out PSNR** (index baseline 15.6 dB on `alpine_01`; record `alpine_02`’s own WAN baseline first).
- **Joint fit of three overlapping frames** from different rails (baseline 22.3 dB vs 30.8 dB single-frame).
- Sparse point count and registered image count from SphereSfM.

A model that looks better per clip and does not raise the joint fit has not fixed the problem this project actually measured.

For WS, if its output is perspective, record whether SphereSfM still registers it, or whether a known-pose dataset was needed. That answer decides whether WS is a sampler swap or a pipeline change.

### 6.4 Exit rules

- W+M beats W on joint fit → adopt progressive memory first, independent of any model swap.
- WS beats W+M on joint fit, fits 24 GB, and the owner accepts the licence → WorldStereo becomes the Quality candidate. Write a separate port plan that covers perspective output, HiRes reprojection, and the SphereSfM bypass.
- H3 does not beat W on joint fit → drop H3 for this feature. Its strengths (audio, text rendering, 15 s clips) are not used here.
- W360 beats Matrix-3D 5B on the Fast profile at equal or lower time → propose it for Fast / Scout.

---

## 7. Contract summary for whoever ports the winner

| | WAN today | WorldStereo | H3 | Wan360 | VACE |
|--|-----------|-------------|----|--------|------|
| Input | mesh render + mask + CLIP-Vision of pano | point-cloud render + mask + camera.json (+ memory) | source + inverted mask + depth | ERP pano + trajectory | control + mask (+127 gray) |
| Output | 2:1 ERP | probably perspective; verify | 2:1 if fed 2:1 | 2:1 ERP | as fed |
| Frames / fps | 81 / 16 | verify | 141 / 24 | 161 / 16 | 81 / 16 |
| Mask white means | known | verify | regenerate | (no mask) | regenerate |
| Steps | 8 (LightX2V) | 4 (DMD) | 20–40, turbo 4 | per repo | ~20, Lightning 4 |
| Audio | no | no | yes, strip | no | no |
| Licence gate | none | EU/UK/KR excluded | US/EU/UK/KR excluded | none | none |
| Plot Camera grid | native | verify | 141 on grid | 161 on grid | native |

---

## 8. Owner decisions

1. Run the progressive-memory spike with the current WAN first? (No licence or model change.)
2. Reopen HY-World 2.0 **only** for its WorldStereo expansion model, for a measured spike? If yes, confirm the team is inside its territory.
3. Keep H3 in the bake-off at all? If yes, confirm territory. If no, archive `H3_WAN_Replacement_Plan.md` as research.
4. Accept Wan360 as a Fast / Scout challenger to Matrix-3D 5B?
5. If H3 stays, apply the section 2 corrections to its plan (141 frames, Union 2.0 in Comfy 0.38, the 3090 data point, the trained-vs-zero-shot risk).

---

## 9. Sources

| Topic | Source |
|-------|--------|
| Plot Camera length grid | `engine/geo/rail.py` (`9 + 4 * k`), `engine/geo/defaults.py` |
| WAN contract | `engine/video/defaults.py`, `engine/video/masked_cond.py` |
| Measured cross-view gap | [Votion_3DGS_2.0_Index.md](Votion_3DGS_2.0_Index.md), measured table |
| Matrix-3D mesh vs point-cloud renders, Matrix-Pano | [arXiv 2508.08086](https://arxiv.org/abs/2508.08086), [project](https://matrix-3d.github.io/), [licence](https://github.com/SkyworkAI/Matrix-3D/blob/main/LICENSE) |
| Pano LoRA conversion | [mickmumpitz/Wan2.1-Pano360-LoRA](https://huggingface.co/mickmumpitz/Wan2.1-Pano360-LoRA) |
| WorldStereo 2.0 | [HY-World 2.0 worldgen](https://github.com/Tencent-Hunyuan/HY-World-2.0/blob/main/hyworld2/worldgen/README.md), [WorldStereo](https://github.com/FuchengSu/WorldStereo), [FP8](https://huggingface.co/apozz/WorldStereo-fp8), [VNCCS loader card](https://comfy.icu/node/VNCCS_LoadWorldStereoModel); licence: Tencent HY-World 2.0 Community licence in that repo |
| Wan360 | [MUGEN](https://github.com/AlayaLab/MUGEN), [paper](https://alaya-lab.github.io/MUGEN/assets/paper.pdf), [weights](https://huggingface.co/AlayaLab/Wan360) |
| H3 on a 3090 | r/StableDiffusion user report (2026-09-17), [Wan2GP models](https://github.com/deepbeepmeep/Wan2GP/blob/main/docs/MODELS.md) |
| Union 2.0 in Comfy | GenVidKit H3 ControlNet write-up, [Union 2.0 card](https://huggingface.co/alibaba-pai/MiniMax-H3-Fun-Controlnet-Union-2.0) |
| VACE | [ali-vilab/VACE](https://github.com/ali-vilab/vace/), [Wan2.2-VACE-Fun-A14B](https://huggingface.co/alibaba-pai/Wan2.2-VACE-Fun-A14B), [vramready](https://vramready.com/models/wan-2-1-vace-14b/) |
| LTX-2.5 inpaint | [LTX in/outpainting](https://docs.ltx.io/open-source-model/feature-guides/editing-effects/in-outpainting.md) |
| Open-weights landscape (Wan 2.5+ API only) | [modeltrends](https://modeltrends.ai/ai-model-guides/best-open-source-ai-video-models) |
