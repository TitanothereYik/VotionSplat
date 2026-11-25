# Models

Home reads `tools/model_manifest.json`. Sizes are the files on disk, in decimal GB. Core is about 78 GB. Optional is about 7 GB. The downloader also wants about 20 GB free for the download itself.

| Label | Stage | GB | Group | Source |
| --- | --- | --- | --- | --- |
| Krea 2 | Panorama | 13.14 | core | Comfy-Org/Krea-2 |
| Krea 2 text encoder | Panorama | 5.24 | core | Comfy-Org/Krea-2 |
| Wan VAE | Panorama | 0.25 | core | Comfy-Org/Wan_2.1_ComfyUI_repackaged |
| Krea 2 360 ERP LoRA | Panorama | 0.23 | core | Hugging Face, see tools/model_manifest.json |
| Krea 2 360 outpaint LoRA | Panorama | 0.23 | core | Hugging Face, see tools/model_manifest.json |
| RealESRGAN x2 | Panorama | 0.07 | core | ai-forever/Real-ESRGAN |
| MoGe | Panorama | 1.26 | core | Ruicheng/moge-vitl |
| Qwen3-VL 8B | Trajectories | 17.55 | core | Qwen/Qwen3-VL-8B-Instruct |
| ZIM | Trajectories | 1.26 | core | naver-iv/zim-anything-vitl |
| Grounding DINO | Trajectories | 0.69 | core | IDEA-Research/grounding-dino-tiny |
| MoGe 2 | Trajectories | 1.32 | core | Ruicheng/moge-2-vitl-normal |
| SAM 3 | Trajectories | 3.45 | core | bundled in the installer |
| WorldStereo | Expand | 20.36 | core | apozz/WorldStereo-fp8 |
| WorldStereo config | Expand | 0.00 | core | hanshanxue/WorldStereo |
| Wan 480P config | Expand | 0.00 | core | Wan-AI/Wan2.1-I2V-14B-480P-Diffusers |
| Wan 720P configs | Expand | 0.02 | core | Wan-AI/Wan2.1-I2V-14B-720P-Diffusers |
| Wan VAE config | Expand | 0.00 | core | Wan-AI/Wan2.1-T2V-14B-Diffusers |
| UMT5 | Expand | 6.74 | core | Comfy-Org/Wan_2.1_ComfyUI_repackaged |
| CLIP vision | Expand | 1.26 | core | Comfy-Org/Wan_2.1_ComfyUI_repackaged |
| DINOv2 | Expand | 0.35 | core | facebook/dinov2-base |
| SeedVR2 3B | HiRes | 3.39 | core | numz/SeedVR2_comfyUI |
| SeedVR2 VAE | HiRes | 0.50 | core | numz/SeedVR2_comfyUI |
| LPIPS VGG16 | Splat | 0.55 | core | https://download.pytorch.org/models/vgg16-397923af.pth |
| Difix | Splat | 5.17 | optional | nvidia/difix_ref |
| Florence-2 | Panorama | 1.56 | optional | florence-community/Florence-2-large |

Weight files go in `models`. Hub snapshots go in `.hf_cache`. SAM 3 is `models\sam3\sam3.pt` and is not downloaded.
