"""Diffusers WAN 2.1 I2V runtime. Qt never imports this.

Spike 2026-09-03: Diffusers 0.40 loads the Kijai fp8 UNET; Matrix-3D script does not
run on this Windows 3090. Ampere cuDNN has no Float8_e4m3fn — transformer compute is
bfloat16 with accelerate cpu_offload (14B bf16 on RAM). VAE uses enable_tiling
(locked). LoRA order: pano 0.98 then LightX2V 1. No Comfy process.
Hub: tokenizer + JSON configs only from WAN_DIFFUSERS_CONFIG_ID. Weights stay in models/.
"""

from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import torch

from engine.job import REPO_ROOT, huggingface_cache_root, job_progress
from engine.video import defaults as V
from engine.video.masked_cond import hidden_36, pack_mask_diffusers, splatkit_masked_pixels

MODELS = REPO_ROOT / "models"
UNET = MODELS / "diffusion_models" / V.UNET_FILE
VAE = MODELS / "vae" / V.VAE_FILE
PANO_LORA = MODELS / "loras" / V.PANO_LORA_FILE
LIGHTX2V = MODELS / "loras" / V.LIGHTX2V_FILE
UMT5 = MODELS / "text_encoders" / V.UMT5_FILE
CLIP_VISION = MODELS / "clip_vision" / V.CLIP_VISION_FILE

CONFIG_ALLOW = (
    "tokenizer/**",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "spiece.model",
    "vocab.txt",
    "text_encoder/config.json",
    "image_encoder/config.json",
    "image_encoder/preprocessor_config.json",
    "image_processor/**",
    "scheduler/**",
    "model_index.json",
)


def log(msg: str) -> None:
    print(msg, flush=True)


def _empty() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _from_single_file(cls, path: Path, **kwargs):
    try:
        return cls.from_single_file(str(path), **kwargs)
    except TypeError:
        if "dtype" in kwargs:
            kwargs = dict(kwargs)
            kwargs["torch_dtype"] = kwargs.pop("dtype")
            return cls.from_single_file(str(path), **kwargs)
        raise


def ensure_diffusers_configs() -> Path:
    """Small JSON/tokenizer files only. Never the Wan-AI 7-shard 14B pack."""
    from huggingface_hub import snapshot_download

    dest = snapshot_download(
        V.WAN_DIFFUSERS_CONFIG_ID,
        allow_patterns=list(CONFIG_ALLOW),
        ignore_patterns=["*.safetensors", "*.bin", "*.pth", "*.msgpack"],
        cache_dir=str(huggingface_cache_root() / "hub"),
    )
    log(f"STATUS // wan configs {dest}")
    return Path(dest)


def dequant_fp8_scaled(path: Path) -> dict:
    """Comfy fp8_e4m3fn_scaled: weight * scale_weight → bf16 for transformers UMT5."""
    from safetensors.torch import load_file

    sd = load_file(str(path))
    scales = {k[: -len(".scale_weight")]: v for k, v in sd.items() if k.endswith(".scale_weight")}
    out = {}
    for key, tensor in sd.items():
        if key.endswith(".scale_weight"):
            continue
        stem = key[: -len(".weight")] if key.endswith(".weight") else None
        if stem is not None and stem in scales:
            out[key] = (tensor.float() * scales[stem].float()).to(torch.bfloat16)
        else:
            out[key] = tensor
    return out


def load_umt5(cfg_root: Path):
    from transformers import UMT5Config, UMT5EncoderModel

    cfg_dir = cfg_root / "text_encoder"
    log(f"STATUS // UMT5 dequant {UMT5.name}")
    state = dequant_fp8_scaled(UMT5)
    config = UMT5Config.from_pretrained(str(cfg_dir), local_files_only=True)
    model = UMT5EncoderModel(config)
    model.to(dtype=torch.bfloat16)
    missing, unexpected = model.load_state_dict(state, strict=False)
    log(f"STATUS // UMT5 missing={len(missing)} unexpected={len(unexpected)}")
    del state
    _empty()
    return model


def load_clip(cfg_root: Path):
    from safetensors.torch import load_file
    from transformers import CLIPVisionConfig, CLIPVisionModel

    cfg_dir = cfg_root / "image_encoder"
    config = CLIPVisionConfig.from_pretrained(str(cfg_dir), local_files_only=True)
    model = CLIPVisionModel(config)
    state = load_file(str(CLIP_VISION))
    missing, unexpected = model.load_state_dict(state, strict=False)
    log(f"STATUS // CLIP-Vision missing={len(missing)} unexpected={len(unexpected)}")
    del state
    return model


def vae_stats(vae, device, dtype):
    mean = torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1).to(device, dtype)
    std = (1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1)).to(device, dtype)
    return mean, std


def encode_prompt(cfg_root: Path, prompt: str, max_length: int = 512) -> torch.Tensor:
    from transformers import AutoTokenizer

    tok_dir = cfg_root / "tokenizer" if (cfg_root / "tokenizer").is_dir() else cfg_root
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(tok_dir), local_files_only=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(str(cfg_root), local_files_only=True)
    encoder = load_umt5(cfg_root).to("cuda")
    encoder.eval()
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    ids = text_inputs.input_ids.to("cuda")
    mask = text_inputs.attention_mask.to("cuda")
    with torch.no_grad():
        hidden = encoder(ids, mask).last_hidden_state
    seq = int(mask.gt(0).sum())
    hidden = hidden[0, :seq]
    pad = hidden.new_zeros(max_length - seq, hidden.shape[-1])
    out = torch.cat([hidden, pad], dim=0).unsqueeze(0).cpu()
    del encoder, hidden, ids, mask, tokenizer
    _empty()
    return out


def encode_clip(cfg_root: Path, pano_rgb: np.ndarray) -> torch.Tensor:
    from PIL import Image
    from transformers import CLIPImageProcessor

    proc_dir = cfg_root / "image_processor" if (cfg_root / "image_processor").is_dir() else cfg_root
    try:
        processor = CLIPImageProcessor.from_pretrained(str(proc_dir), local_files_only=True)
    except Exception:
        processor = CLIPImageProcessor()
    image = Image.fromarray(pano_rgb, mode="RGB")
    model = load_clip(cfg_root).to("cuda")
    model.eval()
    batch = processor(images=image, return_tensors="pt")
    batch = {k: v.to("cuda") for k, v in batch.items()}
    with torch.no_grad():
        embeds = model(**batch, output_hidden_states=True).hidden_states[-2]
    out = embeds.cpu()
    del model, embeds, batch
    _empty()
    return out


def encode_masked_video(
    rgb: torch.Tensor,
    mask: torch.Tensor,
    width: int,
    height: int,
    length: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """VAE-encode hole-filled control. Returns noise-shaped zeros, packed mask, concat latent (CPU)."""
    from diffusers import AutoencoderKLWan
    from diffusers.pipelines.wan.pipeline_wan_i2v import retrieve_latents

    vae = _from_single_file(AutoencoderKLWan, VAE, dtype=torch.float32)
    vae.enable_tiling()
    vae.to("cuda")
    vae.eval()
    image, mask_valid = splatkit_masked_pixels(
        rgb, mask, width, height, length, V.HOLE_FILL, V.INVERT_MASK
    )
    video = image.permute(3, 0, 1, 2).unsqueeze(0) * 2.0 - 1.0
    video = video.to(device="cuda", dtype=torch.float32)
    with torch.no_grad():
        latent = retrieve_latents(vae.encode(video), sample_mode="argmax")
    mean, std = vae_stats(vae, latent.device, latent.dtype)
    latent = (latent - mean) * std
    _, z_dim, t_lat, lh, lw = latent.shape
    packed = pack_mask_diffusers(mask_valid, lh, lw, temporal=int(vae.config.scale_factor_temporal or 4))
    packed = packed.to(device=latent.device, dtype=latent.dtype)
    noise = torch.zeros((1, z_dim, t_lat, lh, lw), device="cpu", dtype=torch.float32)
    packed_cpu = packed.cpu()
    latent_cpu = latent.cpu()
    del vae, image, mask_valid, video, latent, packed, mean, std
    _empty()
    return noise, packed_cpu, latent_cpu


def load_transformer_with_loras():
    from accelerate import cpu_offload
    from diffusers import WanTransformer3DModel
    from diffusers.loaders.lora_pipeline import WanLoraLoaderMixin

    log("STATUS // transformer from_single_file bfloat16 (Ampere cannot compute fp8)")
    model = _from_single_file(WanTransformer3DModel, UNET, dtype=torch.bfloat16)
    pano = WanLoraLoaderMixin.lora_state_dict(str(PANO_LORA), local_files_only=True)
    pano = WanLoraLoaderMixin._maybe_expand_t2v_lora_for_i2v(model, pano)
    WanLoraLoaderMixin.load_lora_into_transformer(pano, transformer=model, adapter_name="pano")
    del pano
    _empty()
    lx = WanLoraLoaderMixin.lora_state_dict(str(LIGHTX2V), local_files_only=True)
    lx = WanLoraLoaderMixin._maybe_expand_t2v_lora_for_i2v(model, lx)
    WanLoraLoaderMixin.load_lora_into_transformer(lx, transformer=model, adapter_name="lightx2v")
    del lx
    _empty()
    model.set_adapters(["pano", "lightx2v"], weights=[V.PANO_LORA_STRENGTH, V.LIGHTX2V_STRENGTH])
    model.eval()
    cpu_offload(model, execution_device=torch.device("cuda"), offload_buffers=True)
    log("STATUS // transformer cpu_offload + LoRAs")
    return model


def denoise(
    transformer,
    noise: torch.Tensor,
    packed_mask: torch.Tensor,
    concat_latent: torch.Tensor,
    prompt_embeds: torch.Tensor,
    image_embeds: torch.Tensor,
    seed: int,
    steps: int,
    rail_index: int = 0,
    preview_dir: Path | None = None,
    mask_png: Path | None = None,
) -> torch.Tensor:
    from diffusers import FlowMatchEulerDiscreteScheduler
    from diffusers.utils.torch_utils import randn_tensor

    scheduler = FlowMatchEulerDiscreteScheduler(shift=V.MODEL_SAMPLING_SHIFT)
    scheduler.set_timesteps(steps, device="cpu")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    latents = randn_tensor(noise.shape, generator=generator, device="cpu", dtype=torch.float32)
    packed_mask = packed_mask.to(dtype=torch.float32)
    concat_latent = concat_latent.to(dtype=torch.float32)
    prompt_embeds = prompt_embeds.to(device="cuda", dtype=torch.bfloat16)
    image_embeds = image_embeds.to(device="cuda", dtype=torch.bfloat16)
    timesteps = scheduler.timesteps
    for i, t in enumerate(timesteps):
        job_progress(i + 1, steps, f"wan step {i + 1}/{steps}")
        latent_in = hidden_36(latents, packed_mask, concat_latent).to(device="cuda", dtype=torch.bfloat16)
        timestep = t.expand(latents.shape[0]).to("cuda")
        with torch.no_grad():
            pred = transformer(
                hidden_states=latent_in,
                timestep=timestep,
                encoder_hidden_states=prompt_embeds,
                encoder_hidden_states_image=image_embeds,
                return_dict=False,
            )[0]
        pred = pred.float().cpu()
        latents = scheduler.step(pred, t, latents, return_dict=False)[0]
        del latent_in, pred
        _empty()
        log(f"STATUS // wan step {i + 1}/{steps}")
        if preview_dir is not None:
            emit_latent_preview(rail_index, i + 1, steps, latents, preview_dir, mask_png)
    del prompt_embeds, image_embeds
    _empty()
    return latents


def decode_latents(latents: torch.Tensor) -> np.ndarray:
    from diffusers import AutoencoderKLWan
    from diffusers.video_processor import VideoProcessor

    vae = _from_single_file(AutoencoderKLWan, VAE, dtype=torch.float32)
    vae.enable_tiling()
    vae.to("cuda")
    vae.eval()
    latents = latents.to(device="cuda", dtype=torch.float32)
    mean, std = vae_stats(vae, latents.device, latents.dtype)
    latents = latents / std + mean
    with torch.no_grad():
        video = vae.decode(latents, return_dict=False)[0]
    processor = VideoProcessor(vae_scale_factor=int(vae.config.scale_factor_spatial or 8))
    frames = processor.postprocess_video(video, output_type="np")[0]
    del vae, video, latents, mean, std
    _empty()
    return frames


def emit_mask_still(mask: torch.Tensor, path: Path) -> Path:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    frame = mask[min(mask.shape[0] // 2, mask.shape[0] - 1)]
    if frame.ndim == 3:
        frame = frame[..., 0]
    arr = frame.detach().float().cpu().numpy()
    if arr.max() <= 1.0:
        arr = arr * 255.0
    u8 = np.clip(arr, 0, 255).astype(np.uint8)
    Image.fromarray(u8, mode="L").convert("RGB").save(path)
    return path


def emit_latent_preview(
    rail_index: int,
    step: int,
    steps: int,
    latents: torch.Tensor,
    preview_dir: Path,
    mask_png: Path | None,
) -> None:
    """Cheap RGB from latent channels so the viewport updates during sampling."""
    from PIL import Image

    preview_dir.mkdir(parents=True, exist_ok=True)
    body = latents[0, :3]
    t = int(body.shape[1] // 2)
    rgb = body[:, t].detach().float().cpu().numpy().transpose(1, 2, 0)
    rgb = rgb - float(rgb.min())
    peak = float(rgb.max())
    rgb = rgb / (peak + 1e-6)
    im = Image.fromarray(np.clip(rgb * 255.0, 0, 255).astype(np.uint8), mode="RGB")
    im = im.resize((V.WAN_WIDTH, V.WAN_HEIGHT), Image.Resampling.BILINEAR)
    path = preview_dir / "live_rgb.png"
    im.save(path)
    mask = mask_png if mask_png is not None else path
    log(f"WAN\tpreview\t{int(rail_index)}\t{int(step)}\t{mask}\t{path}\t{int(steps)}")


def emit_preview_frames(
    rail_index: int,
    frames: np.ndarray,
    control_rgb: torch.Tensor,
    control_mask: torch.Tensor,
    width: int,
    height: int,
    length: int,
    preview_dir: Path,
) -> None:
    """Write mask + WAN RGB stills and log WAN\\tframe lines for the Qt viewport."""
    from PIL import Image

    from engine.video.masked_cond import splatkit_masked_pixels

    _, mask_valid = splatkit_masked_pixels(
        control_rgb,
        control_mask,
        width,
        height,
        length,
        V.HOLE_FILL,
        V.INVERT_MASK,
    )
    preview_dir.mkdir(parents=True, exist_ok=True)
    count = min(len(frames), int(mask_valid.shape[0]))
    for i in range(count):
        mv = mask_valid[i].detach().cpu().numpy()
        mask_u8 = (np.clip(mv, 0.0, 1.0) * 255.0).astype(np.uint8)
        mask_rgb = np.stack([mask_u8, mask_u8, mask_u8], axis=-1)
        mask_png = preview_dir / f"mask_{i:04d}.png"
        Image.fromarray(mask_rgb, mode="RGB").save(mask_png)

        frame = frames[i]
        if frame.dtype != np.uint8:
            frame = np.clip(frame * 255.0, 0, 255).astype(np.uint8)
        rgb_png = preview_dir / f"frame_{i:04d}.png"
        Image.fromarray(frame, mode="RGB").save(rgb_png)
        log(f"WAN\tframe\t{int(rail_index)}\t{i}\t{mask_png}\t{rgb_png}")


def write_mp4(frames: np.ndarray, path: Path, fps: int) -> tuple[int, int]:
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    if frames.dtype != np.uint8:
        pix = np.clip(frames * 255.0, 0, 255).astype(np.uint8)
    else:
        pix = frames
    height, width = pix[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(path), fourcc, float(fps), (width, height))
    if not out.isOpened():
        raise RuntimeError(f"VideoWriter failed to open {path}")
    for frame in pix:
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    out.release()
    return width, height


def run_rail(
    prompt: str,
    negative: str,
    pano_rgb: np.ndarray,
    control_rgb: torch.Tensor,
    control_mask: torch.Tensor,
    out_mp4: Path,
    seed: int,
    width: int = V.WAN_WIDTH,
    height: int = V.WAN_HEIGHT,
    length: int = V.WAN_LENGTH,
    steps: int = V.WAN_STEPS,
    rail_index: int = 0,
    preview_dir: Path | None = None,
) -> tuple[int, int]:
    try:
        import sageattention  # noqa: F401

        log("STATUS // SageAttention present — SplatKit/Kijai black-frame patch not applied (import only)")
    except Exception:
        log("STATUS // SageAttention missing — continuing (PRD: warn, do not fail)")

    job_progress(1, 10, "wan configs")
    cfg = ensure_diffusers_configs()
    # cfg=1: skip negative encoder (graph WAN_CFG=1). Keep the string in the request only.
    _ = negative
    mask_png = None
    if preview_dir is not None:
        mask_png = emit_mask_still(control_mask, preview_dir / "live_mask.png")
        log(f"WAN\tmask\t{int(rail_index)}\t{mask_png}")
    job_progress(2, 10, "wan text")
    prompt_embeds = encode_prompt(cfg, prompt)
    job_progress(3, 10, "wan clip")
    image_embeds = encode_clip(cfg, pano_rgb)
    job_progress(4, 10, "wan vae encode")
    noise, packed, concat = encode_masked_video(control_rgb, control_mask, width, height, length)
    if hidden_36(noise, packed, concat).shape[1] != 36:
        raise RuntimeError(f"WAN concat channels {hidden_36(noise, packed, concat).shape} != 36")
    job_progress(5, 10, "wan transformer")
    transformer = load_transformer_with_loras()
    job_progress(6, 10, "wan sample")
    latents = denoise(
        transformer,
        noise,
        packed,
        concat,
        prompt_embeds,
        image_embeds,
        seed,
        steps,
        rail_index=rail_index,
        preview_dir=preview_dir,
        mask_png=mask_png,
    )
    del transformer, noise, packed, concat, prompt_embeds, image_embeds
    _empty()
    job_progress(8, 10, "wan decode")
    frames = decode_latents(latents)
    del latents
    _empty()
    job_progress(9, 10, "wan write")
    if preview_dir is not None:
        from PIL import Image

        preview_dir.mkdir(parents=True, exist_ok=True)
        live = preview_dir / "live_rgb.png"
        for i, frame in enumerate(frames):
            pix = frame
            if pix.dtype != np.uint8:
                pix = np.clip(pix * 255.0, 0, 255).astype(np.uint8)
            Image.fromarray(pix, mode="RGB").save(live)
            mask_out = mask_png if mask_png is not None else live
            log(f"WAN\tframe\t{int(rail_index)}\t{i}\t{mask_out}\t{live}")
    w, h = write_mp4(frames, out_mp4, V.CREATE_VIDEO_FPS)
    log(f"WAN\tdone\t{int(rail_index)}\t{out_mp4}")
    log(f"WAN\tmeasured\t{w}x{h}\t{out_mp4}")
    job_progress(10, 10, "wan done")
    return w, h
