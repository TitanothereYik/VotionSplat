"""Native Krea 2 text + IMAGE (Ostris) paths + shared 8K tail. Graph widgets; no Comfy process."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from engine.job import job_progress
from engine.pano import defaults as D
from engine.pano.cas import contrast_adaptive_sharpen
from engine.pano.comfy_bootstrap import import_comfy_nodes, import_ostris_edit, import_usd_no_upscale
from engine.pano.harmonize import harmonize_boundary
from engine.pano.krea2_ref import krea2_full_res_reference
from engine.pano.noise import add_noise
from engine.pano.roll import roll_horizontal, seam_mask
from engine.pano.warp import persp_to_erp


def _pil_to_image(im: Image.Image) -> torch.Tensor:
    arr = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


def _green_canvas(width: int, height: int) -> torch.Tensor:
    """EmptyImage node 37 color 65280 = #00FF00."""
    color = int(D.GREEN_CANVAS_COLOR)
    r = ((color >> 16) & 0xFF) / 255.0
    g = ((color >> 8) & 0xFF) / 255.0
    b = (color & 0xFF) / 255.0
    t = torch.zeros(1, int(height), int(width), 3, dtype=torch.float32)
    t[..., 0] = r
    t[..., 1] = g
    t[..., 2] = b
    return t


def _composite_masked(destination: torch.Tensor, source: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """ImageCompositeMasked x=0, y=0, resize_source=false. Mask 1 = source."""
    dest = destination.clone()
    src = source
    m = mask.float()
    if m.ndim == 2:
        m = m.unsqueeze(0)
    if m.ndim == 4:
        m = m[..., 0]
    if m.shape[1:] != dest.shape[1:3]:
        m = torch.nn.functional.interpolate(
            m.unsqueeze(1), size=dest.shape[1:3], mode="bilinear", align_corners=False
        ).squeeze(1)
    if src.shape[1:3] != dest.shape[1:3]:
        src = torch.nn.functional.interpolate(
            src.movedim(-1, 1), size=dest.shape[1:3], mode="bilinear", align_corners=False
        ).movedim(1, -1)
    m = m.clamp(0, 1).unsqueeze(-1)
    return dest * (1.0 - m) + src * m


def _image_to_pil(t: torch.Tensor) -> Image.Image:
    arr = (t[0].detach().clamp(0, 1).cpu().numpy() * 255.0).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _unwrap(node_out):
    """First slot from a Comfy tuple or V3 NodeOutput."""
    args = getattr(node_out, "args", None)
    if args is not None and not isinstance(node_out, (tuple, list)):
        return args[0] if args else node_out
    if isinstance(node_out, (tuple, list)):
        return node_out[0]
    return node_out


class KreaSession:
    def __init__(self) -> None:
        self.N = import_comfy_nodes()
        self.unet = None
        self.clip = None
        self.vae = None
        self.model_360 = None
        self.model_edit = None
        self.ostris_encode_cls = None
        self.upscale_model = None
        self.usd_cls = None

    def load_base_stack(self) -> None:
        print("STATUS // load UNET / CLIP / VAE / RealESRGAN", flush=True)
        job_progress(1, 8, "load UNET / CLIP / VAE")
        self.unet = _unwrap(self.N["UNETLoader"]().load_unet(D.UNET_FILENAME, D.UNET_WEIGHT_DTYPE))
        self.clip = _unwrap(self.N["CLIPLoader"]().load_clip(D.CLIP_FILENAME, D.CLIP_TYPE, D.CLIP_DEVICE))
        self.vae = _unwrap(self.N["VAELoader"]().load_vae(D.VAE_FILENAME))
        self.upscale_model = _unwrap(self.N["UpscaleModelLoader"].execute(D.REALESRGAN_FILENAME))
        self.usd_cls = import_usd_no_upscale()

    def load_text_stack(self) -> None:
        self.load_base_stack()
        print("STATUS // load 360 T2I LoRA (MODEL_360); USD stays on base MODEL", flush=True)
        self.model_360 = _unwrap(
            self.N["LoraLoaderModelOnly"]().load_lora_model_only(
                self.unet, D.LORA_T2I_FILENAME, D.LORA_T2I_STRENGTH
            )
        )

    def load_image_stack(self) -> None:
        self.load_base_stack()
        print(
            "STATUS // Ostris Edit patch kv_cache=True; ERP outpaint LoRA; ModelSamplingFlux IMAGE",
            flush=True,
        )
        job_progress(2, 8, "Ostris Edit + outpaint LoRA")
        patch_cls, encode_cls = import_ostris_edit()
        self.ostris_encode_cls = encode_cls
        patched = _unwrap(patch_cls().patch(self.unet, D.OSTRIS_KV_CACHE))
        loraed = _unwrap(
            self.N["LoraLoaderModelOnly"]().load_lora_model_only(
                patched, D.LORA_OUTPAINT_FILENAME, D.LORA_OUTPAINT_STRENGTH
            )
        )
        self.model_edit = _unwrap(
            self.N["ModelSamplingFlux"]().patch(
                loraed,
                D.FLUX_MAX_SHIFT,
                D.FLUX_BASE_SHIFT,
                D.GENERATE_WIDTH,
                D.GENERATE_HEIGHT,
            )
        )

    def encode_prompt(self, text: str):
        cond = _unwrap(self.N["CLIPTextEncode"]().encode(self.clip, text))
        zero = _unwrap(self.N["ConditioningZeroOut"]().zero_out(cond))
        return cond, zero

    def sample_text_pano(self, prompt: str, seed: int) -> torch.Tensor:
        positive, negative = self.encode_prompt(prompt)
        latent = _unwrap(self.N["EmptyLatentImage"]().generate(D.GENERATE_WIDTH, D.GENERATE_HEIGHT, 1))
        print(
            f"STATUS // KSampler TEXT-PANO seed={seed} steps={D.TEXT_PANO_STEPS} "
            f"cfg={D.TEXT_PANO_CFG} {D.TEXT_PANO_SAMPLER}/{D.TEXT_PANO_SCHEDULER}",
            flush=True,
        )
        job_progress(3, 8, "KSampler TEXT-PANO")
        latent = _unwrap(
            self.N["KSampler"]().sample(
                self.model_360,
                int(seed),
                D.TEXT_PANO_STEPS,
                D.TEXT_PANO_CFG,
                D.TEXT_PANO_SAMPLER,
                D.TEXT_PANO_SCHEDULER,
                positive,
                negative,
                latent,
                D.TEXT_PANO_DENOISE,
            )
        )
        return _unwrap(self.N["VAEDecode"]().decode(self.vae, latent))

    def seam_inpaint_text(self, image: torch.Tensor, prompt: str, seam_seed: int) -> torch.Tensor:
        print("STATUS // roll 50% - SEAM INPAINT node 31 - roll back", flush=True)
        job_progress(5, 8, "seam inpaint")
        rolled = roll_horizontal(image, D.ROLL_FRACTION)
        mask = seam_mask(int(rolled.shape[2]), int(rolled.shape[1]))
        latent = _unwrap(self.N["VAEEncode"]().encode(self.vae, rolled))
        latent = _unwrap(self.N["SetLatentNoiseMask"]().set_mask(latent, mask))
        positive, negative = self.encode_prompt(prompt)
        latent = _unwrap(
            self.N["KSampler"]().sample(
                self.model_360,
                int(seam_seed),
                D.TEXT_SEAM_STEPS,
                D.TEXT_SEAM_CFG,
                D.TEXT_SEAM_SAMPLER,
                D.TEXT_SEAM_SCHEDULER,
                positive,
                negative,
                latent,
                D.TEXT_SEAM_DENOISE,
            )
        )
        out = _unwrap(self.N["VAEDecode"]().decode(self.vae, latent))
        return roll_horizontal(out, D.ROLL_FRACTION)

    def shared_8k_tail(self, image: torch.Tensor, upscaler_prompt: str) -> torch.Tensor:
        print("STATUS // 8K tail graph 5 RealESRGAN - lanczos 8192x4096 - CAS 0.5 - noise 0.01 - USD", flush=True)
        job_progress(6, 8, "RealESRGAN 8K")
        # VAE decode can leave requires_grad; USD does .cpu().numpy() (vendor/ultimatesd/utils.py).
        image = image.detach()
        upscaled = _unwrap(self.N["ImageUpscaleWithModel"].execute(self.upscale_model, image))
        scaled = _unwrap(
            self.N["ImageScale"]().upscale(
                upscaled, D.SCALE_METHOD, D.FINAL_WIDTH, D.FINAL_HEIGHT, D.SCALE_CROP
            )
        )
        sharp = contrast_adaptive_sharpen(scaled.detach(), D.CAS_AMOUNT)
        noisy = add_noise(sharp, D.NOISE_SEED, D.NOISE_STRENGTH).detach()
        positive, negative = self.encode_prompt(upscaler_prompt)
        print(
            f"STATUS // UltimateSDUpscaleNoUpscale tile={D.USD_TILE} steps={D.USD_STEPS} "
            f"denoise={D.USD_DENOISE} model=MODEL (base Krea)",
            flush=True,
        )
        job_progress(7, 8, "Ultimate SD Upscale")
        # Positional args match UltimateSDUpscaleNoUpscale.upscale(), not JSON widget order
        # (JSON seam_fix_width sits before seam_fix_mask_blur; the Python method swaps those two).
        refined = _unwrap(
            self.usd_cls().upscale(
                noisy,
                self.unet,
                positive,
                negative,
                self.vae,
                D.USD_SEED,
                D.USD_STEPS,
                D.USD_CFG,
                D.USD_SAMPLER,
                D.USD_SCHEDULER,
                D.USD_DENOISE,
                D.USD_MODE_TYPE,
                D.USD_TILE,
                D.USD_TILE,
                D.USD_MASK_BLUR,
                D.USD_TILE_PADDING,
                D.USD_SEAM_FIX_MODE,
                D.USD_SEAM_FIX_DENOISE,
                D.USD_SEAM_FIX_MASK_BLUR,
                D.USD_SEAM_FIX_WIDTH,
                D.USD_SEAM_FIX_PADDING,
                D.USD_FORCE_UNIFORM_TILES,
                D.USD_TILED_DECODE,
            )
        )
        return refined

    def text_to_8k(
        self,
        prompt: str,
        seed: int,
        seam_seed: int,
        upscaler_prompt: str,
    ) -> Image.Image:
        img = self.sample_text_pano(prompt, seed)
        img = self.seam_inpaint_text(img, prompt, seam_seed)
        img = self.shared_8k_tail(img, upscaler_prompt)
        job_progress(8, 8, "pano done")
        return _image_to_pil(img)

    def upscale_erp_to_8k(self, image: Image.Image, upscaler_prompt: str) -> Image.Image:
        t = _pil_to_image(image)
        out = self.shared_8k_tail(t, upscaler_prompt)
        job_progress(8, 8, "8K done")
        return _image_to_pil(out)

    def _ostris_encode(self, prompt: str, image: torch.Tensor):
        return _unwrap(self.ostris_encode_cls().encode(self.clip, prompt, vae=None, image1=image))

    def _flux_ref_method(self, conditioning):
        return _unwrap(
            self.N["FluxKontextMultiReferenceLatentMethod"].execute(conditioning, D.FLUX_REF_METHOD)
        )

    def _image_refs(self, surroundings: str, canvas: torch.Tensor):
        prompt = D.OUTPAINT_INSTRUCTION + " " + surroundings
        print(
            "STATUS // TextEncodeKrea2OstrisEdit vae unconnected; Krea2Reference 8MP; FluxKontext",
            flush=True,
        )
        positive = self._flux_ref_method(
            krea2_full_res_reference(self._ostris_encode(prompt, canvas), self.vae, canvas)
        )
        negative = self._flux_ref_method(
            krea2_full_res_reference(
                self._ostris_encode(D.OSTRIS_NEGATIVE, canvas), self.vae, canvas
            )
        )
        return prompt, positive, negative

    def sample_image_pano(self, positive, negative, seed: int) -> torch.Tensor:
        latent = _unwrap(self.N["EmptyLatentImage"]().generate(D.GENERATE_WIDTH, D.GENERATE_HEIGHT, 1))
        print(
            f"STATUS // KSampler IMAGE-PANO seed={seed} steps={D.IMAGE_PANO_STEPS} "
            f"cfg={D.IMAGE_PANO_CFG} {D.IMAGE_PANO_SAMPLER}/{D.IMAGE_PANO_SCHEDULER}",
            flush=True,
        )
        job_progress(3, 8, "KSampler IMAGE-PANO")
        latent = _unwrap(
            self.N["KSampler"]().sample(
                self.model_edit,
                int(seed),
                D.IMAGE_PANO_STEPS,
                D.IMAGE_PANO_CFG,
                D.IMAGE_PANO_SAMPLER,
                D.IMAGE_PANO_SCHEDULER,
                positive,
                negative,
                latent,
                D.IMAGE_PANO_DENOISE,
            )
        )
        return _unwrap(self.N["VAEDecode"]().decode(self.vae, latent))

    def seam_inpaint_image(
        self,
        image: torch.Tensor,
        ref_canvas: torch.Tensor,
        positive,
        negative,
        seam_seed: int,
    ) -> torch.Tensor:
        print(
            "STATUS // roll 50% - SEAM INPAINT node 63 - rolled green canvas refs - roll back",
            flush=True,
        )
        job_progress(5, 8, "seam inpaint")
        rolled = roll_horizontal(image, D.ROLL_FRACTION)
        rolled_ref = roll_horizontal(ref_canvas, D.ROLL_FRACTION)
        mask = seam_mask(int(rolled.shape[2]), int(rolled.shape[1]))
        latent = _unwrap(self.N["VAEEncode"]().encode(self.vae, rolled))
        latent = _unwrap(self.N["SetLatentNoiseMask"]().set_mask(latent, mask))
        pos = krea2_full_res_reference(positive, self.vae, rolled_ref)
        neg = krea2_full_res_reference(negative, self.vae, rolled_ref)
        latent = _unwrap(
            self.N["KSampler"]().sample(
                self.model_edit,
                int(seam_seed),
                D.IMAGE_SEAM_STEPS,
                D.IMAGE_SEAM_CFG,
                D.IMAGE_SEAM_SAMPLER,
                D.IMAGE_SEAM_SCHEDULER,
                pos,
                neg,
                latent,
                D.IMAGE_SEAM_DENOISE,
            )
        )
        out = _unwrap(self.N["VAEDecode"]().decode(self.vae, latent))
        return roll_horizontal(out, D.ROLL_FRACTION)

    def image_to_8k(
        self,
        photo: Image.Image,
        surroundings: str,
        h_fov_deg: float,
        seed: int,
        seam_seed: int,
        upscaler_prompt: str,
    ) -> Image.Image:
        src = _pil_to_image(photo)
        print(
            f"STATUS // PanoWarp {D.GENERATE_WIDTH}x{D.GENERATE_HEIGHT} h_fov={h_fov_deg}",
            flush=True,
        )
        job_progress(2, 8, "h_fov warp")
        warp_img, inpaint = persp_to_erp(src, h_fov_deg=float(h_fov_deg))
        photo_mask = 1.0 - inpaint
        green = _green_canvas(D.GENERATE_WIDTH, D.GENERATE_HEIGHT)
        canvas = _composite_masked(green, warp_img, photo_mask)
        _prompt, positive, negative = self._image_refs(surroundings, canvas)
        decoded = self.sample_image_pano(positive, negative, seed)
        canvas = canvas.to(device=decoded.device, dtype=decoded.dtype)
        warp_img = warp_img.to(device=decoded.device, dtype=decoded.dtype)
        inpaint = inpaint.to(device=decoded.device)
        photo_mask = photo_mask.to(device=decoded.device)
        print("STATUS // HarmonizeBoundary 1 / 8 / 800 / wrap", flush=True)
        job_progress(4, 8, "harmonize")
        harm = harmonize_boundary(decoded, canvas, inpaint)
        print("STATUS // Put original photo back (warp control_video + photo mask)", flush=True)
        put_back = _composite_masked(harm, warp_img, photo_mask)
        img = self.seam_inpaint_image(put_back, canvas, positive, negative, seam_seed)
        img = self.shared_8k_tail(img, upscaler_prompt)
        job_progress(8, 8, "pano done")
        return _image_to_pil(img)


def run_text_pano(
    prompt: str,
    seed: int,
    seam_seed: int,
    upscaler_prompt: str,
) -> Image.Image:
    with torch.no_grad():
        session = KreaSession()
        session.load_text_stack()
        return session.text_to_8k(prompt, seed, seam_seed, upscaler_prompt)


def run_8k_tail_only(image: Image.Image, upscaler_prompt: str) -> Image.Image:
    with torch.no_grad():
        session = KreaSession()
        session.load_base_stack()
        return session.upscale_erp_to_8k(image, upscaler_prompt)


def run_image_pano(
    photo: Image.Image,
    surroundings: str,
    h_fov_deg: float,
    seed: int,
    seam_seed: int,
    upscaler_prompt: str,
) -> Image.Image:
    with torch.no_grad():
        session = KreaSession()
        session.load_image_stack()
        return session.image_to_8k(photo, surroundings, h_fov_deg, seed, seam_seed, upscaler_prompt)
