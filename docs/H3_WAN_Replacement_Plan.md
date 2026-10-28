# MiniMax H3 as a replacement for WAN 2.1

**Status:** plan only. This file does not change the app, the phase specs, or the locked WAN path.  
**Date:** 2026-10-26.  
**What WAN does today:** Phase 3, [Votion_3DGS_2.0_Phase3.md](Votion_3DGS_2.0_Phase3.md) section “How WAN fills missing area.”  
**Machine this plan is written for:** RTX 3090 24 GB, the locked Quality GPU.

---

## 1. The feature that has to be replaced

WAN is not a general text-to-video button. On each rail it does one job:

```
8K ERP + MoGe mesh
  → Plot Camera flies the mesh
  → control video (known pixels) + validity mask (white = known, black = hole)
  → Wan 2.1 I2V 14B, masked, 1440×720, 81 frames, 16 fps
  → user reviews the 720p ERP clip
  → HiRes pastes the 8K photograph back onto known pixels
  → WAN pixels remain only in the holes
  → SphereSfM
```

Locked numbers (`engine/video/defaults.py`, SMPL graph):

| Knob | WAN today |
|------|-----------|
| Canvas | **1440×720**, 2:1 equirectangular |
| Length | **81** frames, **16** fps (about 5.1 s) |
| Hole colour before the VAE | **black** |
| Mask meaning | white = keep, black = generate |
| Conditioning | 36-channel pack: noise + 4-ch mask + 16-ch latent (`engine/video/masked_cond.py`) |
| Identity | CLIP-Vision of the pano, encode `center` |
| Weights | I2V 14B fp8, Matrix-3D pano LoRA at 0.98, LightX2V at 1.0 |
| Sampler | 8 steps, cfg 1, euler, shift 5 |
| Prompt | `prompt.txt` of the actual scene, plus a visible style suffix |

The pano LoRA exists so the filled clip stays a panorama while the camera moves. HiRes exists because WAN still repaints the whole frame, and a splat needs the original photograph wherever the mesh can answer. Sharpness stays the 8K reprojection, not a bigger video model.

A replacement has to do all three of these, or it is a different product:

1. **Stay on the rail.** The camera path is the Plot Camera spline, already baked into the control video. A prompt that says “dolly forward” is not that spline.
2. **Fill only the holes.** Known mesh pixels are the photograph. The model invents the unseen side of a wall, the occluded room, the sky the pano never held.
3. **Stay equirectangular.** SphereSfM matches these frames as `SPHERE`. A 16:9 or 21:9 cinematic shot cannot be dropped into that dataset.

---

## 2. What MiniMax H3 is

[MiniMax H3](https://github.com/MiniMax-AI/MiniMax-H3/) (announced 2026-07-31, [model write-up](https://www.minimax.io/blog/minimax-h3)) is an open-weights omni-modal video model. One forward pass emits picture and stereo audio. Two checkpoints:

| Checkpoint | Tasks | Inputs |
|------------|--------|--------|
| **FL2VA** | Text-to-video, first frame, last frame, both | Text, optional start and end images |
| **Ref2VA** | Reference-to-video | Text plus reference images, video, and audio |

Open weights target about **1 megapixel**, short edge **768**, **24 fps**, trained range about **5–15 s**. Frame counts snap to **`17n + 5`** (124 frames ≈ 5.17 s). 2K on the MiniMax platform is a hosted upscale (`H3-Regenerate-2K`), not a local weight. Local ComfyUI templates top out near **1344×768**.

Nothing in the base FL2VA / Ref2VA checkpoints takes a validity mask or a depth video. Those two checkpoints alone do not replace WAN.

---

## 3. Every tool that was checked

Read each row as “what this tool can actually be given,” not as a slogan.

### 3.1 Hole fill that can follow a control video

This is the WAN-shaped job.

| Tool | What it accepts | Hole fill | Follows the rail | Equirect wrap | Where it runs |
|------|-----------------|-----------|------------------|---------------|----------------|
| **[Fun ControlNet Union 2.0](https://huggingface.co/alibaba-pai/MiniMax-H3-Fun-Controlnet-Union-2.0)** in [VideoX-Fun](https://github.com/aigc-apps/VideoX-Fun) | Source video + mask + one control video (Canny, Depth, HED, MLSD, Pose, Scribble, Layout, or Gray) | **Yes.** White mask = regenerate, black = keep. `control_in_dim = 49` (latent + masked latent + mask). **`post_norm`**: holes sit at mid-gray after ImageNet norm, which the card describes as the Wan 2.1 hole recipe. Script named `examples/minimax_h3_fun/predict_v2v_control_inpaint.py`. Config `minimax_h3_control_inpaint_post_norm.yaml`. | **Yes**, if the control video is the Plot Camera depth (or gray) render. Frame count snaps to the largest `17n+5` the clip can hold, max 15 s. Canvas keeps the control video’s aspect ratio, both sides multiples of 32, inside the pixel budget. | Only if the source frames are already ERP. The checkpoint is not a panorama LoRA. | Local Python. Published memory: transformer ~62 GB plus Qwen3-VL ~62 GB, and that pair does not fit one 80 GB GPU without `model_group_offload` or `model_cpu_offload_and_qfloat8`. |
| **Fun ControlNet Union v1** ([HF](https://huggingface.co/alibaba-pai/MiniMax-H3-Fun-Controlnet-Union), [ComfyUI template](https://docs.comfy.org/tutorials/video/minimax/minimax-h3-fun-controlnet)) | Same idea, five conditions (no Scribble / Layout / Gray) | **Yes**, via `MiniMaxH3FunControlNetApply`. Mask `1` = regenerate. v1 holes are **`pre_norm`** (extreme dark in VAE space). The card says 2.0 blends more cleanly. | Same as 2.0 | Same | ComfyUI **0.35.0+**. Packaged files: `minimax_h3_fun_controlnet_union_pruned_int8_convrot.safetensors` in `model_patches/`, on either `fl2va` or `ref2va`. This is the build that can be tried on a 3090. It is the v1 recipe. |
| **ComfyUI latent noise mask** ([PR 15375](https://github.com/Comfy-Org/ComfyUI/pull/15375), [Per-Row Masking](https://github.com/ethanfel/ComfyUI-MiniMaxH3-PerRowMasking)) | A video latent plus a mask | Partial. Protected rows stay near the conditioning timestep; generated rows denoise. The mask snaps to H3’s 2×2 latent grid, about **32×32 source pixels**, so edges are blocky. | No depth video. It inpaints a latent; it does not ingest Plot Camera depth. | No | Comfy custom / core patch. The Per-Row repo says it is a temporary shim. |
| **[LanPaint 2.0](https://comfyui-wiki.com/en/news/2026-08-09-lanpaint-h3-av)** | Masked video (and audio) inpaint sampler | Yes, as a sampler, not as a control branch | No published depth-rail mode | No | Comfy. The wiki’s own table is about **38–43 GB** for 40–81 frames at 480p–880p, 20 steps, and it says a 24 GB card is for shorter clips. That figure is the wiki’s, not MiniMax’s. |

Union 2.0 is the closest mechanical match to `WanI2VMaskedConditioning`: a mask channel, a known-pixel latent, and a control video, with the hole colour chosen to match Wan. v1 is the same idea with a worse hole colour and fewer conditions, and it is the one ComfyUI currently templates.

GitHub issue [aigc-apps/VideoX-Fun#509](https://github.com/aigc-apps/VideoX-Fun/issues/509) (opened 2026-08-25) reports `predict_v2v_control_inpaint.py` missing. The model card still names it. Phase B starts by checking that the pinned VideoX-Fun commit actually contains the script. The ComfyUI Apply node is the implementation the Comfy docs show as present today.

### 3.2 Panorama video that does not fill the mesh holes

These make a 360 clip. They do not take the validity mask, and they do not follow `rail_json`.

| Tool | What it actually generates | Canvas | Local? |
|------|----------------------------|--------|--------|
| **[shamanic/minimax-h3-equi360-lora](https://huggingface.co/shamanic/minimax-h3-equi360-lora)** | Full-sphere monoscopic ERP video. Trigger `equirect360`. Documented call is fal `minimax/h3/text-to-video/lora`. Evaluated at **21:9, 768P, 5 s**, strength 1.0, prompt expansion off. The card then says: squeeze that 21:9 frame to 2:1. Sample prompts use a **stationary** camera. Rank 32, 2500 steps, reviewed v2 sha256 `a1e3ebd2b79be92d5e970c6f516519f8bd7f8e4271a71e7a7d7c8e938e157cf7`. | 21:9, then a resize. That resize is not the Plot Camera 2:1. | Weights can be downloaded. The published recipe is a fal call. |
| **[rehan-fal/minimax-h3-360-equirect-lora](https://huggingface.co/rehan-fal/minimax-h3-360-equirect-lora)** | Same family. Trigger `eqr360`. fal `minimax/h3/text-to-video/lora`. Strength **0.75**. 768P is the real generation; 2K and 4K are upscales of that pass. Author’s seam correlation is about 0.95 versus 0.55 with no LoRA. They also say H3 has **no wrap-around attention**, so the back seam can still split and their packager blends it. | 21:9, then ffmpeg scale to 2:1 plus spherical metadata. | fal. |
| **[Burgstall ComfyUI-Seamless-Equirectangular](https://github.com/Burgstall-labs/ComfyUI-Seamless-Equirectangular)** | Keeps a **2:1** first-frame panorama seamless through H3 FL2VA. Nodes: `EquirectMiniMaxH3VAEDecode` (dual-rotation decode; H3’s VAE is a tiled ViT, so circular conv padding does nothing), `EquirectPerStepRoll` (rolls the latent and the FL2VA guide together; `random_per_step`, `start_percent=0.25`), `EquirectWrappedNoise`, `EquirectMiniMaxH3RoPE` (start at blend 0.5, enable last). Example workflow `MiniMax-H3-SEAMLESS-FL2VA.json`. | True 2:1, from a seamless first frame. | Comfy custom nodes. The camera move is whatever FL2VA invents from the prompt and the end frame. There is no mask input. |
| **fal [`minimax/h3/image-to-video/lora`](https://fal.ai/models/minimax/h3/image-to-video/lora/api)** | First frame (and optional last frame) plus LoRAs. The image sets the canvas. No mask field, no depth video. | The input image’s aspect. 480P / 768P native; 2K and 4K upscale a 768P result. | Hosted. |

Using one of these instead of WAN would throw away the rail. The clip would be a new 360 video of a similar place, and HiRes would reproject the 8K pano along a path the video did not follow. SphereSfM would then solve two disagreeing motions. That is the failure mode the HiRes stage was added to stop.

### 3.3 Base runners that are not the hole-fill feature

| Tool | Role | Why it is not the replacement by itself |
|------|------|------------------------------------------|
| [ComfyUI native templates](https://docs.comfy.org/tutorials/video/minimax/minimax-h3) (0.30.0+; ControlNet 0.35.0+; `pruned_w6a8` needs 0.38.0+) | T2V, I2V, R2V, multiframe guides, FastH3 | I2V locks the first frame (and maybe the last). It does not inpaint per-frame holes. |
| [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3) | Quantized files for that UI | The 3090-sized builds. See section 5. |
| [MiniMax create-task API](https://platform.minimax.io/docs/api-reference/video-generation-v2-create) | `MiniMax-H3`, `H3-Highspeed`, `H3-Lite`, `H3-Max`. Text, first/last frame, or references. Up to 2K on the full H3 id. | The content array has no mask and no control video. Image-to-video and reference-to-video cannot be mixed in one call. |
| [Comfy Router](https://docs.comfy.org/development/comfy-router/models/minimax/minimax-h3/code) `minimax/minimax-h3` | Hosted H3, picture plus stereo audio | Same inputs. The documented body has no silent-render switch; a pipeline that does not want the soundtrack has to strip the track after download. |
| [SGLang local deploy](https://platform.minimax.io/docs/guides/local-deploy-h3) | `/v1/videos`, official FL2VA weights | Pinned reference is **8× B200**. The page says the quickstart does not include the Comfy inpaint masks. |
| [vLLM-Omni](https://platform.minimax.io/docs/guides/local-deploy-h3) | FL2VA + Ref2VA HTTP | Community recipe, multi-GPU. No published mask inpaint. |
| diffusers `ModularPipeline.from_pretrained("MiniMaxAI/MiniMax-H3")` | Load FL2VA / Ref2VA | No published inpaint pipeline. A native port should follow VideoX-Fun’s inpaint script, not this loader, until a spike shows otherwise. |
| [lightx2v/Minimax-h3-Turbo](https://huggingface.co/lightx2v/Minimax-h3-Turbo) and the Comfy turbo files (`minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16`, `…_4step_…_768p`, `minimax_h3_ref2v_turbo_4step_v0.1`) | Fewer steps | Speed, the way LightX2V is speed for WAN. They do not add a mask or an ERP prior. Turbo LoRAs have no adaln tensors, so they load on both full and pruned builds. A 4-step ref2v turbo is what the ControlNet template offers. |
| FastH3 / sparse attention (`sol-attn` on base H3, `vsa` only on FastH3 weights) | Long-clip speed in Comfy 0.35+ | Optional after a quality spike. Not a conditioning method. |
| Unofficial `Minimax_h3_latent_Upscaler` | 24-channel latent upscale, then a refine pass | Still not 8K, and the refine runs at the target size so it does not save peak VRAM. HiRes geometry mode stays the sharpness path. |

### 3.4 The combination that maps onto Votion

No single row above is the WAN feature. The mapping is:

```
Plot Camera (unchanged)
  control RGB          known pixels, already 2:1 ERP
  validity mask        white = known
  depth render         same rail, same frames
        │
        ▼
Fun ControlNet Union inpaint          ← this is the WAN replacement
  source video = control RGB
  mask         = inverted validity    (H3 white = regenerate)
  control      = depth                (Gray as a fallback spike)
  prompt       = prompt.txt
  guidance     = 1.0
  canvas       = 2:1 inside the H3 pixel budget (section 4)
  length       = 124 frames at 24 fps (section 4)
        │
        ├─ if the back seam tears: Burgstall H3 decode / per-step roll
        │    (spike only; not proven together with ControlNet)
        │
        ▼
strip the audio track
        │
        ▼
existing review UI (mask | filled frame)
        │
        ▼
existing HiRes geometry composite     ← still required
        │
        ▼
SphereSfM
```

The equirect LoRAs stay off that path until a spike shows the ControlNet fill leaving the sphere. They were trained as text-to-video on 21:9, then squeezed. Feeding a squeezed 21:9 clip into a 2:1 mask from Plot Camera misaligns every hole.

---

## 4. Contract changes if H3 takes the rail

These are not drop-in numbers. HiRes, SphereSfM, and the Generate page all assume the WAN contract.

### 4.1 Frames and fps

| | WAN | H3 |
|--|-----|----|
| fps | 16 | 24, fixed |
| Legal lengths | Plot Camera step 4, default 81 | `17n+5`, and 24 fps |
| Lengths that satisfy both step 4 and `17n+5`, inside the Plot Camera max of 257 | | **56, 124, 192** |

**124 frames** is the match for today’s ~5 s rail (`17×7+5`, about 5.17 s). 81 is not on the H3 grid; the Fun card snaps down to the largest legal length, so an 81-frame control video would be shortened, and HiRes `0-80/2` would index frames that no longer exist.

HiRes frame specs to rewrite when the length changes, not before the spike agrees on 124:

- Graph default `0-80/2` is “every other frame of an 81-frame clip.”
- The analogue on 124 frames is a new spec, written down after the spike, not invented as `0-123/2` in the product until the rail length is actually changed.
- `merge_long` is **1440** because that is the WAN width. It follows the new canvas width.

### 4.2 Canvas

H3’s trained area is about **768 × 1344 = 1,032,192** pixels, both sides multiples of 32. A 2:1 frame inside that budget:

| Width × height | Pixels | 2:1 | ×32 |
|----------------|--------|-----|-----|
| 1440×720 (WAN) | 1,036,800 | yes | 720 is not a multiple of 32 |
| **1408×704** | 991,232 | yes | yes |
| 1344×672 | 903,168 | yes | yes |

**1408×704** is the candidate Quality canvas. It is slightly under the trained area and it is the same projection as the control video. Do not generate 21:9 and scale to 2:1 on this path: the mask would be resampled onto a frame the model did not inpaint.

Fun ControlNet keeps the control video’s aspect and fits a pixel budget. The spike should hand it a 1408×704 ERP control video rather than a 1440×720 one, so the snap does not crop the seam.

### 4.3 Mask polarity and hole colour

| | Votion control mask | H3 Fun inpaint |
|--|---------------------|----------------|
| White | known, keep | regenerate |
| Black | hole, generate | keep |

The worker inverts the Plot Camera mask at the H3 boundary. The Generate page can keep drawing white as “known” for the user. The inversion is not a UI change.

WAN paints holes **black** before the VAE. Union **2.0 `post_norm`** zeros the holes **after** ImageNet normalisation, which lands on mid-gray, and the card says that is the blend that matches Wan. Union **v1 `pre_norm`** lands on extreme dark. A 3090 spike that only has the Comfy v1 patch is testing the darker recipe. Record which recipe the still came from. Do not run both a black pre-fill and `post_norm`.

### 4.4 What HiRes still has to do

The Phase 3 note stands: a video model that repaints known pixels is fine as a movie and wrong as a splat. Fun ControlNet claims the black mask region stays fixed. The spike measures that claim: mean absolute error on known pixels, identity view versus the control frame. If the known region is actually preserved, HiRes has less to repair and coverage should rise. If H3 repaints the whole frame the way WAN does, HiRes stays mandatory and the gate mask keeps its current meaning (white = 8K photograph, black = model fill).

Either way, H3’s native canvas is about 1 MP. It does not replace the 8192-wide geometry composite. SeedVR stays rejected as the sharpness path.

### 4.5 Audio

H3’s base pass emits 32 kHz stereo. Fun ControlNet sets `control_apply_audio = false`, so the control branch does not steer audio, and the model can still write a soundtrack. SphereSfM consumes frames. The worker drops the audio stream before the clip is saved as the rail video. There is no product use for the generated voice, music, or sound effects.

### 4.6 Sampler

| | WAN | H3 Fun |
|--|-----|--------|
| Steps | 8 (LightX2V) | Card samples use **40**. Comfy turbo checkbox uses a **4-step** ref2v LoRA. |
| Guidance | cfg 1 | **1.0 exactly.** Above 1 applies guidance twice and the card says the image gets worse. |
| Shift | 5 (SD3 node) | Model default shift 12, audio shift 3. Do not copy the WAN shift. |

Start the spike at 20 steps (Comfy’s non-turbo default) and 40 (the card). Try the 4-step turbo only after a 20-step still exists to compare. Turbo is the speed trade, the way LightX2V is for WAN, and the Comfy guide says it softens fine detail and weakens audio.

---

## 5. The 3090

Published full-weight inference does not target this GPU. The Union 2.0 card: the ~62 GB transformer and the ~62 GB Qwen3-VL encoder do not both sit on one 80 GB GPU. SGLang’s reference box is 8× B200.

The only published builds that are even in range of 24 GB are the Comfy-Org quantizations ([guide](https://docs.comfy.org/tutorials/video/minimax/minimax-h3)):

| Build | File size | Note |
|-------|-----------|------|
| `pruned_w6a8` | 16 GB | Smallest. ComfyUI 0.38.0+. |
| `pruned_int8_convrot` | 21 GB | Template default. On RTX 30-series the guide prefers INT8 over FP8: no FP8 path, INT8 matmul about 2× FP16. |
| `pruned_fp8_scaled` | 21 GB | Same size. Weaker choice on a 3090. |
| `pruned_bf16` | 40 GB | Does not fit. |
| `bf16` | 66 GB | Does not fit. |

File size is not peak VRAM. The load is that file, plus the video VAE, plus the audio VAE, plus the text encoder (`qwen3vl_32b_minimax_h3_nvfp4_awq` is the one the ControlNet template uses), plus the control branch (v1 full file ~6.8 GB, 2.0 full file ~13.5 GB; Comfy’s pruned patch is smaller and is the v1 set). Comfy’s own deploy page says a minimum VRAM was **not published** and must be measured. CPU offload is the documented escape when it does not fit, and it is slower than dropping a precision step.

`docs/video_models_comparison.md` already calls H3 heavy for 24 GB and much happier at 32–48 GB. That note is still the right warning. It is stale on control: Fun ControlNet now exists, so “no spatial control” is no longer true. This plan does not edit that table.

**Gate.** If a 1408×704, 124-frame, depth-guided inpaint does not finish on the 3090, H3 does not become the Quality video model. WAN stays. A larger GPU would be a new profile, decided separately, not slipped in as a silent requirement.

---

## 6. Licence

[MiniMax H3 Community License](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE) (dated 2026-08-02). The Fun ControlNet and both equirect LoRAs are derivatives and say they sit under the same licence.

- **Applicable Territory** is worldwide **except** the European Union, the United Kingdom, the Republic of Korea, and the United States. The grant is only inside that territory.
- People in an excluded territory are told to contact MiniMax. Comfy’s guide says local commercial use in those countries, or anywhere above $20 M yearly revenue, needs a paid licence sold through Comfy. Comfy Cloud generations include commercial rights; that path phones out, which this product does not do.
- Commercial use inside the free grant expects a visible **“Powered by MiniMax H3”** credit. The Help drawer is the place, same pattern as the LiteGS Inria line.
- Outputs are not model derivatives. A LoRA trained on H3 is a derivative and keeps the licence.

Votion is a local product for a small private team, with no telemetry. Before any weight is downloaded, the owner confirms the team is inside the Applicable Territory (or has a written MiniMax / Comfy licence). This plan does not guess the country from the machine.

Hosted calls (fal, MiniMax API, Comfy Router, Comfy Cloud) are legal to use as a **visual reference** during the spike. They are not a worker in the app: the index locks fully local inference and forbids a hidden ComfyUI shell-out. A fal 360 clip is also the wrong geometry for the hole-fill test (section 3.2).

---

## 7. What becomes unnecessary if H3 takes this job

“Unnecessary” means the WAN stage no longer runs. HiRes, rails, MoGe, and SphereSfM stay, because H3 does not replace them (section 4.4, section 1).

### 7.1 Weights that leave the Quality video pack

| File | Why it was loaded |
|------|-------------------|
| `Wan2_1-I2V-14B-720P_fp8_e4m3fn.safetensors` | The inpainter |
| `pano_video_gen_720p_comfy.safetensors` | ERP prior at strength 0.98. There is no H3 LoRA trained on the Matrix-3D UE5 drone flights. Do not point the shamanic or rehan LoRA at this filename and call it done. |
| `lightx2v_T2V_14B_cfg_step_distill_v2_lora_rank64_bf16.safetensors` | 8-step distill. An H3 turbo LoRA is the optional replacement, after the 20-step still. |
| `umt5_xxl_fp8_e4m3fn_scaled.safetensors` | WAN text encoder |
| `clip_vision_h.safetensors` | WAN image identity. H3 FL2VA uses the first-frame / source latent instead. |
| `wan_2.1_vae.safetensors` | WAN VAE |

`4x-UltraSharp.pth` stays. HiRes still upscales the fill before it composites into the 8K frame. The node title “Upscaler for the WAN hole-fill” would be renamed in a later product change; the weight stays.

Krea, Florence, MoGe, and the panorama stills are Phase 1 and Phase 2. H3 does not generate `pano.png`.

### 7.2 Code that is WAN-shaped

| Piece | Fate |
|-------|------|
| `engine/video/masked_cond.py` | The 36-channel Wan pack. Replaced by the 49-channel Fun inpaint pack. |
| `engine/video/wan_runtime.py`, `engine/workers/wan.py` | Become the H3 worker, or a sibling worker, after the spike. |
| WAN rows in `engine/video/defaults.py` | Width 1440, height 720, length 81, fps 16, shift 5, euler, CLIP-Vision `center`, LoRA order. Replaced by section 4. |
| SageAttention **black-frame patch** copied for WAN | WAN-specific. H3’s Sage notes are different: INT8 attention can morph the end of a clip and garble text; the Comfy guide says stay on default attention for `int8_convrot`, or move to bf16 (which does not fit this GPU). Do not copy the WAN Sage patch across. |
| GGUF `wan2.1-i2v-14b-720p-Q4_K_M.gguf` | The ≤16 GB WAN fallback. H3 has no equivalent GGUF on this plan. Fast / Scout profiles need their own decision. |

### 7.3 What stays even after a successful swap

- Plot Camera, rails, star at the origin, FLOOR / SIDE, preview flight. The control video is still the hole map.
- Generate page: `prompt.txt` required, seed, style suffix, negative, live split (known mask | filled frame), per-rail re-gen, Confirm before HiRes.
- HiRes geometry mode, `moge_level` 6 versus Plot Camera `moge_level` 9, gate masks, 8192-wide output.
- Keep / Ditch, SphereSfM, gnomonic cube faces, the trainer.
- The rule that flying through a wall is allowed and must be previewed first. H3 will invent the other side the same way WAN does, if the depth control actually drives it. The preview gate stays.

---

## 8. Phases

No phase edits `engine/video/` or the model manifest. The spike lives in a scratch folder outside the app.

### Phase A — Owner checks (no download required to decide)

1. Territory: inside the community-licence map, or a written licence. Otherwise stop.
2. Accept that HiRes and the rails stay. H3 replaces the masked WAN sample only.
3. Accept that 124 frames at 24 fps and a 1408×704 canvas replace 81 frames at 16 fps and 1440×720, if the spike works.

### Phase B — One rail, on this 3090, outside the app

Use an existing scene that already has a Plot Camera control video and mask (alpine is enough).

1. Confirm whether `examples/minimax_h3_fun/predict_v2v_control_inpaint.py` exists at the VideoX-Fun commit you pin. If it does not, the runnable inpainter is the ComfyUI 0.35+ Fun ControlNet template (v1 `pre_norm`). Say which one produced the stills.
2. Build a 1408×704, 124-frame ERP control video and a depth video from the same rail. Invert the validity mask for H3.
3. Run depth-guided inpaint, guidance 1.0, 20 steps, pruned INT8 or `w6a8`, NVFP4 text encoder. Record peak VRAM, system RAM, and wall time.
4. Measure known-pixel error against the control frames. Save the back-seam still (left edge vs right edge of the ERP).
5. If the seam splits, add Burgstall’s H3 VAE decode and per-step roll **in a second run**. If that combination is unstable (the Burgstall readme warns that RoPE, wrapped noise, and rolling can flash or morph), stop stacking and keep the first run.
6. Do not add the equirect LoRA in the first two runs. A third run may add shamanic at strength 1.0 or rehan at 0.75 only to see whether the seam improves, and only on the 2:1 canvas, not on a 21:9 squeeze.
7. Optional speed run: 4-step turbo, same seed, compared to the 20-step still.
8. One hosted fal or MiniMax clip is allowed as a picture of “H3 the movie.” It is not the acceptance test. The acceptance test is the masked rail.

**Stop the plan** if peak VRAM cannot be made to fit with offload in a time the team will actually wait, or if known pixels are destroyed and the seam is open. WAN remains the Quality path.

### Phase C — HiRes dry fit (still no app change)

Take the Phase B clip and run it through the existing HiRes composite by hand (same `rail_json`, geometry mode, UltraSharp, 8192).

- Coverage (fraction of white gate mask) versus the WAN clip of the same rail.
- A hole that WAN filled and H3 filled, side by side, after HiRes.
- A known region, to see whether the 8K photograph survived.

This is the picture that decides whether the swap is worth a native port. Higher coverage with worse hole invention is not automatically a win; the Generate page exists so a person can ditch a rail.

### Phase D — Native port, only after B and C pass

The product stays native Python. The port target is the VideoX-Fun inpaint path (49-channel control, `post_norm`, guidance 1.0), called from `engine.workers` the way WAN is called now. ComfyUI is the prototype, not a subprocess.

Sketch, not a commitment of filenames until the spike picks v1 or 2.0:

- New weights under `models/`, install-root cache, one Home progress cell. Record sha256 on first download. Do not invent sizes.
- Worker writes the same rail outputs the Generate page already plays (filled frames, mask split, mp4 without audio).
- `prompt.txt`, seed, and the review / re-gen / Confirm split stay.
- Frame length and HiRes frame spec move together, in one change, from 81 to the length Phase B actually ran.
- Help: “Powered by MiniMax H3”, the territory line, and the note that 8K sharpness is still HiRes.

Fast and Scout (16 GB and 12 GB) are out of scope. The `w6a8` file is 16 GB before the text encoder and the control branch. Those profiles keep WAN, or they wait for a measured smaller build. This plan does not invent one.

### Phase E — What not to delete even after D

Do not remove Plot Camera, HiRes, UltraSharp, SphereSfM, or the Keep / Ditch UI in the same change that swaps the sampler. Section 7 is the deletion list for the WAN weights and the 36-channel pack only.

---

## 9. Acceptance

Phase B

- [ ] Named which inpainter ran (Union 2.0 script, or Comfy v1), and the hole recipe (`post_norm` or `pre_norm`).
- [ ] 124-frame, 1408×704, depth-guided fill from a real rail.
- [ ] Peak VRAM and wall time written down for the 3090.
- [ ] Known-pixel error written down.
- [ ] Back-seam still saved.
- [ ] Audio absent from the file that would go to SphereSfM.

Phase C

- [ ] HiRes coverage number for this clip and for the WAN clip of the same rail.
- [ ] One hole crop and one known-pixel crop after HiRes.

Phase D starts only if those stills are good enough that the team would ditch the WAN rail. Otherwise this document stays a research note and the locked WAN path stays the product.

---

## 10. Risks

| Risk | Consequence |
|------|-------------|
| 24 GB cannot hold the quantised stack at 124×1408×704 | H3 is not the Quality model on the locked GPU. |
| Only v1 `pre_norm` runs on Comfy, and the holes look burned-in | The Wan-like blend is the 2.0 recipe, which today is published against full weights and an 80 GB offload story. |
| `predict_v2v_control_inpaint.py` still missing at the pin | Prototype in Comfy; do not block on a filename the issue tracker already could not find. |
| Depth control fights the ERP projection | Poles and the back meridian are where this shows up. The seam still is the test, not the centre of the frame. |
| Equirect LoRA trained at 21:9 | Squeezing it onto the mask grid moves every hole. Keep it out of the first spike. |
| H3 repaints known pixels | HiRes still has to restore the 8K photo, and the swap buys a different inpainter rather than a cleaner composite. |
| Turbo 4-step used as the default because WAN used 8 | The card’s results are 40 steps. Judge turbo against a 20-step still. |
| Hosted API used as the worker | Breaks the local-only lock, and the hosted APIs have no mask field. |
| Territory | Weights downloaded from an excluded territory are outside the community grant. |

---

## 11. Source map

| Question | Source |
|----------|--------|
| WAN contract in this repo | `engine/video/defaults.py`, `engine/video/masked_cond.py`, Phase 3 “How WAN fills missing area” |
| H3 tasks, FL2VA / Ref2VA | [MiniMax-AI/MiniMax-H3](https://github.com/MiniMax-AI/MiniMax-H3/) |
| Hosted API inputs (no mask) | [Create video task](https://platform.minimax.io/docs/api-reference/video-generation-v2-create) |
| Local Comfy sizes, 768×1344, `17n+5`, turbo, Sage | [ComfyUI H3 guide](https://docs.comfy.org/tutorials/video/minimax/minimax-h3) |
| Comfy mask inpaint + depth | [Fun ControlNet tutorial](https://docs.comfy.org/tutorials/video/minimax/minimax-h3-fun-controlnet), [Apply node](https://docs.comfy.org/built-in-nodes/MiniMaxH3FunControlNetApply) |
| Wan-like `post_norm` inpaint, 49 channels, 8 conditions | [Union 2.0 model card](https://huggingface.co/alibaba-pai/MiniMax-H3-Fun-Controlnet-Union-2.0) |
| 2:1 seam nodes for H3’s ViT VAE | [Burgstall seamless equirect](https://github.com/Burgstall-labs/ComfyUI-Seamless-Equirectangular) |
| Text-to-360 LoRAs (not hole fill) | [shamanic equi360](https://huggingface.co/shamanic/minimax-h3-equi360-lora), [rehan 360](https://huggingface.co/rehan-fal/minimax-h3-360-equirect-lora) |
| Quantized files | [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3) |
| Licence territory | [LICENSE](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE) |
| SGLang is not the 3090 inpaint path | [Local deploy](https://platform.minimax.io/docs/guides/local-deploy-h3) |
