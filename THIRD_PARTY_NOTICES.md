# Third-party notices

VotionSplat includes or was built from the following works. Each pin is the commit this tree was taken from.

## Shipped libraries

- ComfyUI, GPL-3.0. Copyright the ComfyUI authors. https://github.com/comfyanonymous/ComfyUI tag v0.32.0, commit c2bcbecd82ec5ae66594340b395c24ef0217b238. Used as a library for Panorama Generate.
- ComfyUI_UltimateSDUpscale, GPL-3.0. https://github.com/ssitu/ComfyUI_UltimateSDUpscale commit 627c871f14532b164331f08d0eebfbf7404161ee.
- ComfyUI-Krea2-Ostris-Edit, MIT. Copyright (c) 2026 Ostris, LLC. https://github.com/ostris/ComfyUI-Krea2-Ostris-Edit commit 7756566160c4a1b24bb1bd9f0ff3ced1a83d7547.
- ComfyUI-SplatKit, MIT. Copyright (c) 2026 mickmumpitz. https://github.com/mickmumpitz/ComfyUI-SplatKit tag 1.1.4, commit f59de2529772b01ffa627545552a00d812af7475. Camera rails, depth helpers, and the SphereSfM bundle.
- WorldStereo, Apache-2.0. https://github.com/FuchengSu/WorldStereo commit 5ba8dc3d37c48ecef99bbd603815ced513a4a431.
- ComfyUI-SeedVR2_VideoUpscaler, Apache-2.0. https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler commit 4490bd1f482e026674543386bb2a4d176da245b9.
- Difix3D, NVIDIA License. https://github.com/nv-tlabs/Difix3D commit c76edc595586e16732c91ddee82f3a6d83a8a9cc.
- Recast Navigation, zlib. Copyright (c) 2009 Mikko Mononen. Commit 9f4ce64458dfae86e1239c525ddc219c4e9e06f1.
- HY-World 2.0 world-generation module. https://github.com/Tencent-Hunyuan/HY-World-2.0 commit df9988efb87bfc0f4947eb3889411cf957478b06. The pinned tree has no LICENSE file.
- fused_ssim, MIT. Copyright (c) 2024 Rahul Goel. Copied from the LiteGS tree at commit 409da0b987ff84d5bcc76a1d3040715d27c16527.
- gsplat 1.5.3, used from the installed package, plus the CUDA extension built for the HY-World trainer.
- SAM 3, SAM License (Meta, last updated November 19, 2025). https://github.com/facebookresearch/sam3 package version 0.1.0; no commit was recorded for this copy. The code is in `vendor/sam3` and the weights ship as `models/sam3/sam3.pt`. The full agreement is `vendor/sam3/LICENSE`, and the installer also puts a copy at `docs/licenses/SAM_LICENSE.txt`.

### GPL note

ComfyUI and ComfyUI_UltimateSDUpscale are GPL-3.0, and the installer ships them inside the app. The 3.0 installer is for internal use. Do not post it publicly unless the matching source (this repository at the release tag plus the pinned upstream commits above) is offered with it.

## Model weights

Home downloads these on first run, except SAM 3, which the installer ships. The license is the one the model card declares at the pinned revision.

- Krea 2 (Comfy-Org/Krea-2), Krea 2 Community License. https://huggingface.co/krea/Krea-2-Turbo/blob/main/LICENSE.pdf
- 360 ERP LoRAs, Krea 2 Community License. See the link at the end of this file.
- Qwen3-VL-8B-Instruct (Qwen/Qwen3-VL-8B-Instruct), Apache-2.0.
- MoGe (Ruicheng/moge-vitl) and MoGe 2 (Ruicheng/moge-2-vitl-normal), MIT.
- ZIM Anything ViT-L (naver-iv/zim-anything-vitl), CC-BY-4.0.
- Grounding DINO tiny (IDEA-Research/grounding-dino-tiny), Apache-2.0.
- Florence-2 large (florence-community/Florence-2-large), MIT.
- DINOv2 base (facebook/dinov2-base), Apache-2.0.
- WorldStereo (hanshanxue/WorldStereo config, apozz/WorldStereo-fp8 weights), MIT.
- Wan 2.1 VAE, CLIP vision and configs (Comfy-Org/Wan_2.1_ComfyUI_repackaged, Wan-AI/Wan2.1-I2V-14B-480P-Diffusers, Wan-AI/Wan2.1-I2V-14B-720P-Diffusers, Wan-AI/Wan2.1-T2V-14B-Diffusers), Apache-2.0.
- SeedVR2 3B (numz/SeedVR2_comfyUI), Apache-2.0.
- Difix reference model (nvidia/difix_ref), NVIDIA License, see `LICENSE.txt` in that repo.
- Real-ESRGAN (ai-forever/Real-ESRGAN). The model card declares no license.
- VGG16 ImageNet weights for LPIPS, from https://download.pytorch.org/models/vgg16-397923af.pth (torchvision).

## Ported code

These files are not a checkout of the upstream repo. The behavior was ported and the copyright stays with the upstream author.

- Panorama warp, roll, seam mask, boundary harmonize, and the Krea 2 full-resolution reference. Ported from ComfyUI-Mickmumpitz-Nodes, MIT, Copyright (c) 2025 MickMumpitz. https://github.com/mickmumpitz/ComfyUI-Mickmumpitz-Nodes commit 4d5ff7c433884631599c3b2d82011a2cbbeea37e. See `engine/pano/warp.py`, `engine/pano/roll.py`, `engine/pano/harmonize.py`, `engine/pano/krea2_ref.py`, and `engine/pano/warp_preview.py`.
- Camera-plot rails and anchors. Ported from ComfyUI-SplatKit, MIT, Copyright (c) 2026 mickmumpitz, commit f59de2529772b01ffa627545552a00d812af7475. See `engine/geo/rail.py` and `engine/geo/anchors.py`.

The 360 LoRA weights downloaded by Home come from https://huggingface.co/mickmumpitz/Krea2-360-ERP-LoRAs. Those files are not in this repository.
