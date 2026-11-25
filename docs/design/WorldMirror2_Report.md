# WorldMirror-2 report

**Status:** report only. No project file is changed.  
**Date:** 2026-10-26.  
**Machine this is judged against:** RTX 3090 24 GB, the locked Quality GPU.  
**Question:** does WorldMirror-2 replace the video model, MoGe, SphereSfM, or the splat trainer?

Short answer: it replaces none of them. It is the feed-forward reconstruction model inside Tencent HY-World 2.0. In that system it runs **after** the video model, on ordinary perspective frames, and its own authors say it is not built for panoramas. The index already archives the parent system.

---

## 1. What it is

WorldMirror-2 (also written WorldMirror 2.0, checkpoint folder `HY-WorldMirror-2.0`) is a ~1.2B feed-forward network. One pass over a set of photos or a video predicts, together:

- a 3D point map in world coordinates
- per-view depth and a per-pixel depth validity mask
- surface normals
- camera poses (OpenCV camera-to-world) and intrinsics
- one 3D Gaussian per pixel: position, scale, quaternion, opacity, and **degree-0** colour (a single RGB, no view-dependent lobes)

Weights and code ship inside [tencent/HY-World-2.0](https://huggingface.co/tencent/HY-World-2.0), loaded with `subfolder="HY-WorldMirror-2.0"`. The model card in the repo’s `DOCUMENTATION.md` is the API. The paper is [arXiv:2604.14268](https://arxiv.org/abs/2604.14268). WorldMirror 1.0 is the earlier model, [arXiv:2510.10726](https://arxiv.org/abs/2510.10726), accepted to ICML 2026. SplatKit’s `SphereSfMDataset` comment still names that older WorldMirror as one of the feed-forward dataset paths next to VGGT. Votion did not take that path.

It can also be conditioned. Any of camera pose, intrinsics, and depth may be passed in, or left out. During training each prior is dropped half the time, so inference accepts any subset.

### What changed from WorldMirror 1.0

| | 1.0 | 2.0 |
|--|-----|-----|
| Position encoding | Absolute RoPE. Quality drops when the test resolution leaves the training grid | Normalized RoPE. Patch coordinates are mapped into \[-1, 1\], so a new resolution is a denser sample of the same range |
| Depth | Supervised by depth only | Depth plus a depth-to-normal loss, so depth and normals have to agree |
| Bad pixels | A confidence weight inside the loss | An explicit depth-mask head at inference |
| Data | Public sets | Plus internal Unreal renders, and monocular normal pseudo-labels on real photos |
| Inference | Single GPU, FP32 | Optional BF16, sequence parallel, and FSDP |
| Training pixels | 100k–250k | 50k–500k, with the view count derived from a token budget |

The high-resolution result is the one that matters. At 756×1036, WorldMirror 1.0’s camera AUC@30 on RealEstate10K falls from 86.13 to 66.29, and its novel-view PSNR falls from 21.34 to 17.78. WorldMirror 2.0 holds AUC@30 at 86.89 and PSNR at 19.98. That is a real upgrade for perspective reconstruction. It is not a new skill at equirectangular video.

---

## 2. Where it sits in HY-World 2.0

HY-World 2.0 is four stages. WorldMirror-2 is the last one, and only part of it.

| Stage | Model | Job |
|-------|--------|-----|
| 1. Panorama | HY-Pano 2.0 (~80B, or a ~425M Qwen backend) | Text or one photo → 360° ERP. Default about 1920×960 |
| 2. Trajectory | WorldNav | MoGe2 depth on perspective cuts of that ERP, aligned into one panoramic cloud, then collision-free paths |
| 3. Expansion | WorldStereo 2.0 (~17B) | Point-cloud render + mask → new keyframes, with a memory bank across paths |
| 4. Composition | **WorldMirror-2, then a separate 3DGS trainer** | Depth and normals on the keyframes, aligned back onto the MoGe cloud, then an optimized splat |

Tencent’s own composition does **not** ship the feed-forward `gaussians.ply` as the world. Section 7 of the paper uses WorldMirror for depth and normals, then:

1. Renders the MoGe panoramic cloud into each keyframe and solves a per-frame scale and shift (in disparity) so WorldMirror’s depth lands in the panorama’s coordinates.
2. Throws the frame away if the alignment is an outlier.
3. Initialises a new 3DGS from that fused cloud and **trains** it.
4. Uses view-independent RGB, because they judge the generated scenes to have negligible reflections. Depth loss is MoGe-aligned WorldMirror depth. Normal loss is MoGe2, on every frame.
5. Optionally extracts a mesh by TSDF fusion for collision, not for the viewer.

The paper’s sentence on panoramas, section 7.1.1: WorldMirror 2.0 “is not explicitly tailored for panoramic reconstruction.” They cut the ERP into perspective views, run the model on those, and still need MoGe2 for the metric panorama.

---

## 3. The contract

| | Value |
|--|--------|
| Input | A folder of images, or one video. Optional `camera_params.json` and a folder of depth maps (`.npy`, `.exr`, 16-bit PNG) |
| Video sampling | Default 1 fps, motion-aware strategy, **1–32 frames** |
| Resolution | `target_size` default **952** on the long edge. Resize, then **center-crop** to a multiple of 14 |
| Trained pixel range | 50k–500k per image. Eval resolutions: 189×259, 378×518, 756×1036 |
| Cameras | OpenCV. Extrinsics are renormalised so the first view is the origin |
| Gaussian colour | `sh` shaped `[N, 1, 3]`. Degree 0 only |
| Gaussian count | One per pixel (`views × H × W`) before pruning. Default cap after voxel prune: 5,000,000 |
| Point cloud | Default cap 2,000,000 |
| Also written | `depth/`, `normal/`, `camera_params.json`, `points.ply`, optional COLMAP `sparse/0/`, optional fly-through mp4 |
| Heads you can unload | `camera`, `depth`, `normal`, `points`, `gs` |
| Multi-GPU | Sequence parallel + FSDP. Input count must be at least the GPU count |

Center-crop is the wrong operation for a 2:1 ERP. A 1440×720 frame cropped toward a square loses the poles or the side seam. There is no circular padding and no spherical camera model in this pipeline.

---

## 4. Published numbers

Point maps, 7-Scenes accuracy (lower is better). Baselines are at medium resolution.

| Method | 7-Scenes Acc. mean |
|--------|--------------------|
| VGGT | 0.046 |
| π³ | 0.048 |
| WorldMirror 1.0, medium | 0.043 |
| WorldMirror 1.0, high | 0.079 |
| WorldMirror 2.0, medium | 0.033 |
| WorldMirror 2.0, high | 0.037 |
| WorldMirror 2.0, high, pose + intrinsics + depth | 0.012 |

Camera and depth on RealEstate10K, and novel-view synthesis averaged on RealEstate10K and DL3DV:

| Method | Pose AUC@30 ↑ | Depth AbsRel ↓ | NVS PSNR ↑ |
|--------|---------------|----------------|------------|
| VGGT | 77.62 | 0.256 | — |
| π³ | 85.90 | 0.151 | — |
| AnySplat | — | — | 18.57 |
| WorldMirror 2.0, high | 86.89 | 0.162 | 19.98 |

Normals: best mean angular error they report on ScanNet is 12.3° at medium resolution, ahead of StableNormal (16.0°) and GeoWizard (16.7°).

Feed-forward novel-view PSNR near 20 dB is the model drawing a new view from its own Gaussians. It is a different number from Votion’s hold-out 15.6 dB, which is an **optimized** splat of WAN frames that disagree with each other. A 20 dB feed-forward splat does not clear the bar set by the single-frame overfit in this project (30.8 dB on `alpine_01` frame 1). The appearance is also degree-0, so a pond cannot change reflection with the camera. The index keeps view-dependent spherical harmonics for that reason.

### Memory, as published

Table 14 of the paper. Resolution 518×378. GPU is an **H20**, not a 3090. The table does not say which heads were enabled.

| Setup | 32 views | 64 views |
|-------|----------|----------|
| 1 GPU, FP32 | 24.95 GB, 2.45 s | 38.56 GB, 6.27 s |
| 1 GPU, BF16 | **15.10 GB**, 2.11 s | **25.06 GB**, 5.65 s |
| 4 GPU, BF16 + sequence parallel + FSDP | 14.04 GB, 0.93 s | 24.67 GB, 2.20 s |

Read for this machine: a batch of about 32 views at 518×378 in BF16 is in range of 24 GB. A batch of 64 is not, on the published card. `alpine_01` reconstructs **972** views. Those cannot enter one forward pass. Chunking them loses the joint pose the model is for, unless each chunk is given camera priors and the clouds are aligned afterwards, which is the job SphereSfM and the trainer already do.

High resolution (756×1036) is about four times the pixels of 518×378. The paper does not publish a VRAM figure there. Expect the 32-view BF16 budget to move up.

---

## 5. Against each Votion stage

| Votion stage | WorldMirror-2 |
|--------------|----------------|
| Krea 8K ERP | Untouched. HY-Pano 2.0 is the panorama model in this family, and it emits ~2K, not 8K. |
| MoGe mesh | Still required in Tencent’s pipeline. WorldMirror depth is scale-free until it is aligned to the MoGe cloud. MoGe2 also supplies the normals used on every training frame. |
| Plot Camera + validity mask | Untouched. WorldMirror does not inpaint holes and does not take a mask video. |
| Video model | Untouched. WorldMirror consumes frames. It does not generate them. WorldStereo is the sibling that does. |
| HiRes 8K composite | Untouched. Inference tops out near 1 MP per view, center-cropped. |
| SphereSfM | The only stage it overlaps. SphereSfM matches the **equirect** frames as spherical cameras, then Votion rewrites cube faces as true gnomonic pinholes (index item 57). WorldMirror wants perspective images and center-crops. Feeding it the raw ERP is the failure mode item 57 already forbids for ordinary COLMAP. |
| Splat trainer | Still required if the goal is the trained splat. Tencent trains a second 3DGS on top of WorldMirror. Their feed-forward Gaussians are degree-0 colour, capped at 5M, and are not the shipped world. |
| Reflections | Lost if the degree-0 head or Tencent’s view-independent trainer is used as the final colour. |

SplatKit already offered a WorldMirror dataset path beside VGGT, and the SphereSfM node was written as the alternative: classical matches and a spherical bundle adjustment, because a feed-forward network is a learned guess and a static pan will not triangulate. WorldMirror-2 is a stronger version of that learned guess on perspective benchmarks. It does not add a spherical camera.

---

## 6. Licence and the recorded rejection

WorldMirror-2 ships under the HY-World 2.0 community licence: no use in the EU, the UK, or Korea, and more than 1,000,000 monthly active users needs Tencent’s approval. Same gate as WorldStereo 2.0.

The index rejection stands:

> Hunyuan / HY-World 2.0 — two huge models in memory at once; designed for four GPUs.  
> V1 is not upgraded. HunyuanWorld / WorldFM / HY-World 2.0 stay archived.

WorldMirror-2 alone is the small model in that stack (~1.2B, and a 32-view BF16 pass can fit 24 GB). The rejection was about the whole system, which also loads HY-Pano (~80B) and WorldStereo (~17B). Using only WorldMirror-2 would be a narrower reopening than using WorldStereo. It is still an owner decision, and it still carries the territory clause. This report does not reopen it.

---

## 7. The only spike that would be informative

Skip it unless the owner accepts the territory and explicitly reopens HY-World for this one model.

If that happens, the test is a pose-and-depth check, not a splat replacement:

1. Take the gnomonic cube faces SphereSfM already writes for one `alpine_02` rail. Do not feed the ERP.
2. Run WorldMirror-2 in BF16 at `target_size` 518, GS head disabled, on one chunk of at most 32 faces.
3. Compare its cameras and depth to the SphereSfM `cameras.bin` and to MoGe on the same faces: rotation error, and depth scale-and-shift against MoGe on pixels the validity mask calls known.
4. Record peak VRAM on the 3090.

A win would mean a faster geometric prior for a later trainer. It would not retire SphereSfM, because 972 views do not fit, the ERP path is unsupported, and the colour model cannot carry reflections. A loss means the archived decision was right for this scene type.

---

## 8. Sources

| Topic | Source |
|-------|--------|
| API, crop, 32-frame cap, degree-0 SH, priors | [HY-World 2.0 DOCUMENTATION.md](https://github.com/Tencent-Hunyuan/HY-World-2.0/blob/main/DOCUMENTATION.md) |
| Model size, date | [HY-World 2.0 model zoo](https://github.com/Tencent-Hunyuan/HY-World-2.0) |
| Paper, “not tailored for panoramic reconstruction”, tables 11–14, composition stage | [arXiv:2604.14268](https://arxiv.org/abs/2604.14268) |
| WorldMirror 1.0 memory table (H20, 518×378) | [arXiv:2510.10726](https://arxiv.org/abs/2510.10726), appendix F |
| Weights | [tencent/HY-World-2.0](https://huggingface.co/tencent/HY-World-2.0), subfolder `HY-WorldMirror-2.0` |
| Reflection lock | The shipped splat keeps view-dependent colour. A degree-0 head does not. |
| Why SphereSfM exists next to WorldMirror | `vendor/splatkit/nodes/spheresfm.py` |
| The video model in the same family | WorldStereo, documented in `docs/design/WorldStereo2_MemoryDMD_FP8_PRD.md`. |
