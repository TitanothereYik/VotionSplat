"""Wan 2.1 pieces WorldStereo still loads: umT5, CLIP-Vision H, and the VAE.

The WAN 14B generator is gone. These files stay because the WorldStereo DiT
was trained on this VAE and these encoders.
"""

from __future__ import annotations

import gc
from pathlib import Path

import torch

from engine.job import REPO_ROOT

UMT5_FILE = "umt5_xxl_fp8_e4m3fn_scaled.safetensors"
VAE_FILE = "wan_2.1_vae.safetensors"
CLIP_VISION_FILE = "clip_vision_h.safetensors"
# Home downloads vae/config.json from this snapshot (manifest id wan_vae_config).
VAE_CONFIG_REPO = "Wan-AI/Wan2.1-T2V-14B-Diffusers"
VAE_CONFIG_REVISION = "38ec498cb3208fb688890f8cc7e94ede2cbd7f68"

MODELS = REPO_ROOT / "models"
UMT5 = MODELS / "text_encoders" / UMT5_FILE
VAE = MODELS / "vae" / VAE_FILE
CLIP_VISION = MODELS / "clip_vision" / CLIP_VISION_FILE


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


def dequant_fp8_scaled(path: Path) -> dict:
    """fp8_e4m3fn_scaled checkpoint: weight * scale_weight, then bfloat16 for UMT5."""
    from safetensors.torch import load_file

    state = load_file(str(path))
    scales = {key[: -len(".scale_weight")]: value for key, value in state.items() if key.endswith(".scale_weight")}
    out = {}
    for key, tensor in state.items():
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

    print(f"STATUS // UMT5 dequant {UMT5.name}", flush=True)
    state = dequant_fp8_scaled(UMT5)
    config = UMT5Config.from_pretrained(str(cfg_root / "text_encoder"), local_files_only=True)
    model = UMT5EncoderModel(config)
    model.to(dtype=torch.bfloat16)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"STATUS // UMT5 missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    del state
    _empty()
    return model


def load_vae():
    from diffusers import AutoencoderKLWan
    from huggingface_hub import try_to_load_from_cache

    kwargs = {"dtype": torch.bfloat16}
    cached = try_to_load_from_cache(VAE_CONFIG_REPO, "vae/config.json", revision=VAE_CONFIG_REVISION)
    if isinstance(cached, str):
        kwargs["config"] = str(Path(cached).parents[1])
        kwargs["subfolder"] = "vae"
    vae = _from_single_file(AutoencoderKLWan, VAE, **kwargs)
    vae.enable_tiling()
    vae.eval()
    return vae
