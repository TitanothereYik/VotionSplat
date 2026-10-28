"""P3 spike: apply pano LoRA 0.98 then LightX2V 1, plus masked concat.

Locked 2026-09-03: spike_lora_mask, install_peft. Do not write engine.workers.wan
until tools/spike_wan_lora_mask.txt records both stages.

No 81-frame sample. No Comfy shell-out. Hub cache = install-root .hf_cache.

Run:  python tools/spike_wan_lora_mask.py
"""

from __future__ import annotations

import gc
import json
import os
import sys
import time
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from engine.job import apply_huggingface_cache_env, worker_env
from engine.video import defaults as V

REPORT = REPO_ROOT / "tools" / "spike_wan_lora_mask.txt"
MODELS = REPO_ROOT / "models"
UNET = MODELS / "diffusion_models" / V.UNET_FILE
VAE = MODELS / "vae" / V.VAE_FILE
PANO_LORA = MODELS / "loras" / V.PANO_LORA_FILE
LIGHTX2V = MODELS / "loras" / V.LIGHTX2V_FILE

# Concat probe uses 5 frames (not WAN_LENGTH=81). Spatial is the graph 1440x720.
PROBE_LENGTH = 5
WIDTH = V.WAN_WIDTH
HEIGHT = V.WAN_HEIGHT


def log(msg: str) -> None:
    print(msg, flush=True)


def vram() -> str:
    try:
        import torch

        if not torch.cuda.is_available():
            return "cuda=False"
        free, total = torch.cuda.mem_get_info()
        alloc = torch.cuda.memory_allocated()
        return f"alloc={alloc/1e9:.2f}GB free={free/1e9:.2f}/{total/1e9:.2f}GB"
    except Exception as exc:
        return f"vram_err={exc}"


def _empty() -> None:
    gc.collect()
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def convert_lora(path: Path) -> dict:
    """Diffusers WanLoraLoaderMixin.lora_state_dict — Comfy diffusion_model.* → PEFT."""
    from diffusers.loaders.lora_pipeline import WanLoraLoaderMixin

    t0 = time.time()
    out: dict = {"file": path.name, "path": str(path)}
    log(f"STATUS // convert {path.name}")
    try:
        state = WanLoraLoaderMixin.lora_state_dict(str(path), local_files_only=True)
        keys = list(state.keys())
        out["ok"] = True
        out["n_keys"] = len(keys)
        out["sample"] = keys[:6]
        out["has_transformer_prefix"] = all(k.startswith("transformer.") for k in keys)
        out["has_lora_A"] = any("lora_A" in k for k in keys)
        out["has_lora_B_bias"] = any("lora_B.bias" in k for k in keys)
        out["seconds"] = round(time.time() - t0, 2)
    except Exception as exc:
        out["ok"] = False
        out["error"] = f"{type(exc).__name__}: {exc}"
        out["trace"] = traceback.format_exc()[-3000:]
        out["seconds"] = round(time.time() - t0, 2)
        log(f"STATUS // convert FAIL {path.name} {out['error']}")
    return out


def load_transformer():
    import torch
    from diffusers import WanTransformer3DModel

    log(f"STATUS // load transformer {vram()}")
    t0 = time.time()
    try:
        model = WanTransformer3DModel.from_single_file(
            str(UNET),
            dtype=torch.float8_e4m3fn,
        )
    except TypeError:
        model = WanTransformer3DModel.from_single_file(
            str(UNET),
            torch_dtype=torch.float8_e4m3fn,
        )
    model.to(device="cuda")
    cfg = {k: model.config[k] for k in ("in_channels", "out_channels", "image_dim", "num_layers", "text_dim")}
    log(f"STATUS // transformer CUDA {time.time()-t0:.1f}s {cfg} {vram()}")
    return model, {"seconds": round(time.time() - t0, 1), "config": cfg, "vram": vram()}


def apply_loras(model) -> dict:
    from diffusers.loaders.lora_pipeline import WanLoraLoaderMixin

    t0 = time.time()
    out: dict = {"order": ["pano", "lightx2v"], "strengths": [V.PANO_LORA_STRENGTH, V.LIGHTX2V_STRENGTH]}
    log(f"STATUS // load_lora pano strength={V.PANO_LORA_STRENGTH} {vram()}")
    try:
        pano = WanLoraLoaderMixin.lora_state_dict(str(PANO_LORA), local_files_only=True)
        pano = WanLoraLoaderMixin._maybe_expand_t2v_lora_for_i2v(model, pano)
        WanLoraLoaderMixin.load_lora_into_transformer(pano, transformer=model, adapter_name="pano")
        del pano
        _empty()
        out["pano"] = {"ok": True, "vram": vram()}
        log(f"STATUS // pano on transformer {vram()}")

        log(f"STATUS // load_lora lightx2v strength={V.LIGHTX2V_STRENGTH} {vram()}")
        lx = WanLoraLoaderMixin.lora_state_dict(str(LIGHTX2V), local_files_only=True)
        lx = WanLoraLoaderMixin._maybe_expand_t2v_lora_for_i2v(model, lx)
        WanLoraLoaderMixin.load_lora_into_transformer(lx, transformer=model, adapter_name="lightx2v")
        del lx
        _empty()
        model.set_adapters(["pano", "lightx2v"], weights=[V.PANO_LORA_STRENGTH, V.LIGHTX2V_STRENGTH])
        names = list(getattr(model, "peft_config", {}).keys())
        out["lightx2v"] = {"ok": True, "vram": vram()}
        out["adapters"] = names
        out["ok"] = True
        out["seconds"] = round(time.time() - t0, 1)
        log(f"STATUS // adapters {names} {vram()}")
    except Exception as exc:
        out["ok"] = False
        out["error"] = f"{type(exc).__name__}: {exc}"
        out["trace"] = traceback.format_exc()[-4000:]
        out["seconds"] = round(time.time() - t0, 1)
        log(f"STATUS // lora FAIL {out['error']}")
    return out


def splatkit_masked_pixels(control_rgb, control_mask, width, height, length, hole_fill, invert_mask):
    """SplatKit WanI2VMaskedConditioning pixel stage (vendor/splatkit/nodes/wan.py). Native torch."""
    import torch
    import torch.nn.functional as F

    fill_val = 0.0 if hole_fill == "black" else 0.5
    vid = control_rgb[:length]
    if vid.ndim != 4 or vid.shape[-1] != 3:
        raise ValueError(f"control_rgb expected [T,H,W,3], got {tuple(vid.shape)}")
    vid = F.interpolate(vid.movedim(-1, 1), size=(height, width), mode="bilinear", align_corners=False)
    vid = vid.movedim(1, -1)
    image = torch.ones((length, height, width, 3), device=vid.device, dtype=vid.dtype) * fill_val
    image[: vid.shape[0]] = vid[..., :3]

    m = control_mask[:length]
    if m.ndim == 4:
        m = m[..., 0]
    m = F.interpolate(m.unsqueeze(1).float(), size=(height, width), mode="bilinear", align_corners=False).squeeze(1)
    m = (m > 0.5).float()
    if invert_mask:
        m = 1.0 - m
    mask_valid = torch.zeros((length, height, width), device=m.device, dtype=torch.float32)
    mask_valid[: m.shape[0]] = m
    image[mask_valid < 0.5] = fill_val
    return image, mask_valid


def diffusers_pack_mask(mask_valid, latent_h, latent_w, temporal=4):
    """Diffusers WanImageToVideoPipeline.prepare_latents mask pack, but per-frame validity (not first-frame-only)."""
    import torch
    import torch.nn.functional as F

    length, _, _ = mask_valid.shape
    mv = F.interpolate(mask_valid.unsqueeze(1), size=(latent_h, latent_w), mode="area").squeeze(1)
    # Comfy concat_mask: 0 = known, 1 = generate. Diffusers I2V ones() then zeros on later frames
    # means 1 = keep/known on frame 0. Match Diffusers: 1 = known, 0 = generate.
    known = (mv > 0.5).float()
    b = 1
    mask_lat_size = known.unsqueeze(0).unsqueeze(0)  # [1,1,T,h,w]
    first = mask_lat_size[:, :, 0:1]
    first = torch.repeat_interleave(first, dim=2, repeats=temporal)
    packed = torch.concat([first, mask_lat_size[:, :, 1:]], dim=2)
    packed = packed.view(b, -1, temporal, latent_h, latent_w)
    packed = packed.transpose(1, 2)
    return packed, known


def masked_concat_stage(vae) -> dict:
    import torch
    from diffusers.pipelines.wan.pipeline_wan_i2v import retrieve_latents

    t0 = time.time()
    length, h, w = PROBE_LENGTH, HEIGHT, WIDTH
    log(f"STATUS // masked concat {length}x{w}x{h} hole_fill={V.HOLE_FILL} {vram()}")
    out: dict = {
        "length": length,
        "width": w,
        "height": h,
        "hole_fill": V.HOLE_FILL,
        "invert_mask": V.INVERT_MASK,
        "note": "synthetic control: left half known, right half hole. alpine_01 has no control mp4.",
    }
    try:
        # [T,H,W,3] in [0,1]. White=valid on left half.
        rgb = torch.full((length, h, w, 3), 0.4, device="cuda")
        rgb[:, :, : w // 2, 0] = 0.8
        mask = torch.zeros((length, h, w), device="cuda")
        mask[:, :, : w // 2] = 1.0

        image, mask_valid = splatkit_masked_pixels(
            rgb, mask, w, h, length, V.HOLE_FILL, V.INVERT_MASK
        )
        hole_frac = float((mask_valid < 0.5).float().mean())
        out["hole_frac"] = round(hole_frac, 4)
        # Diffusers VAE: [-1, 1], layout [B, C, T, H, W] (not [B, T, C, H, W]).
        video = image.permute(3, 0, 1, 2).unsqueeze(0) * 2.0 - 1.0
        video = video.to(device="cuda", dtype=torch.float32)
        latent = retrieve_latents(vae.encode(video), sample_mode="argmax")
        out["concat_latent_shape"] = list(latent.shape)
        _, z_dim, t_lat, lh, lw = latent.shape
        packed, known = diffusers_pack_mask(mask_valid, lh, lw, temporal=4)
        packed = packed.to(device=latent.device, dtype=latent.dtype)
        condition = torch.concat([packed, latent], dim=1)
        noise = torch.zeros((1, z_dim, t_lat, lh, lw), device=latent.device, dtype=latent.dtype)
        hidden = torch.concat([noise, condition], dim=1)
        out["mask_packed_shape"] = list(packed.shape)
        out["condition_shape"] = list(condition.shape)
        out["hidden_states_shape"] = list(hidden.shape)
        out["in_channels_36"] = hidden.shape[1] == 36
        out["ok"] = True
        out["seconds"] = round(time.time() - t0, 1)
        out["vram"] = vram()
        log(f"STATUS // concat hidden={tuple(hidden.shape)} hole_frac={hole_frac:.3f} {vram()}")
        del rgb, mask, image, mask_valid, video, latent, packed, known, condition, noise, hidden
    except Exception as exc:
        out["ok"] = False
        out["error"] = f"{type(exc).__name__}: {exc}"
        out["trace"] = traceback.format_exc()[-4000:]
        out["seconds"] = round(time.time() - t0, 1)
        log(f"STATUS // concat FAIL {out['error']}")
    _empty()
    return out


def tiny_forward(model) -> dict:
    """One transformer step on a tiny 36-ch latent. Not a product sample."""
    import torch

    t0 = time.time()
    out: dict = {}
    log(f"STATUS // tiny forward in_channels={model.config.in_channels} {vram()}")
    try:
        in_ch = int(model.config.in_channels)
        compute = next(model.parameters()).dtype
        hidden = torch.zeros(1, in_ch, 1, 8, 16, device="cuda", dtype=compute)
        timestep = torch.tensor([500], device="cuda")
        text = torch.zeros(1, 32, int(model.config.text_dim), device="cuda", dtype=compute)
        kwargs = {
            "hidden_states": hidden,
            "timestep": timestep,
            "encoder_hidden_states": text,
            "return_dict": False,
        }
        image_dim = model.config.image_dim
        if image_dim:
            kwargs["encoder_hidden_states_image"] = torch.zeros(
                1, 8, int(image_dim), device="cuda", dtype=compute
            )
        out["compute_dtype"] = str(compute)
        with torch.no_grad():
            pred = model(**kwargs)[0]
        out["ok"] = True
        out["pred_shape"] = list(pred.shape)
        out["seconds"] = round(time.time() - t0, 1)
        out["vram"] = vram()
        log(f"STATUS // tiny forward pred={tuple(pred.shape)} {vram()}")
        del hidden, text, pred
    except Exception as exc:
        out["ok"] = False
        out["error"] = f"{type(exc).__name__}: {exc}"
        out["trace"] = traceback.format_exc()[-4000:]
        out["seconds"] = round(time.time() - t0, 1)
        log(f"STATUS // tiny forward FAIL {out['error']}")
    _empty()
    return out


def pick(report: dict) -> dict:
    lora = report.get("apply_lora") or {}
    concat = report.get("masked_concat") or {}
    if lora.get("ok") and concat.get("ok") and concat.get("in_channels_36"):
        return {
            "ready_for_worker": True,
            "note": (
                "Diffusers loaded pano 0.98 then LightX2V 1; SplatKit hole-fill packed to "
                "36-ch hidden_states. Still no 81-frame sample."
            ),
        }
    return {
        "ready_for_worker": False,
        "note": "LoRA apply and/or masked concat failed. Do not write engine.workers.wan.",
    }


def main() -> int:
    env = worker_env(0)
    os.environ.update(env)
    apply_huggingface_cache_env(os.environ)
    log("STATUS // spike_wan_lora_mask start")
    report: dict = {
        "locked": {"next_after_spike": "spike_lora_mask", "peft_install": "install_peft"},
        "vram_start": vram(),
    }
    try:
        import peft

        report["peft"] = {"ok": True, "version": peft.__version__}
    except Exception as exc:
        report["peft"] = {"ok": False, "error": str(exc)}
        report["pick"] = {"ready_for_worker": False, "note": "peft missing"}
        REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        return 3

    report["convert_pano"] = convert_lora(PANO_LORA)
    report["convert_lightx2v"] = convert_lora(LIGHTX2V)
    if not (report["convert_pano"].get("ok") and report["convert_lightx2v"].get("ok")):
        report["pick"] = pick(report)
        REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        log(f"STATUS // pick {report['pick']}")
        return 4

    model = None
    vae = None
    try:
        import torch
        from diffusers import AutoencoderKLWan

        model, report["transformer"] = load_transformer()
        report["apply_lora"] = apply_loras(model)
        # Ampere 3090: cuDNN has no Float8_e4m3fn (measured). Do not keep 14B on GPU
        # during VAE encode — 1440×720×5 with 17.4 GB adapters OOMs.
        report["tiny_forward"] = {
            "ok": False,
            "skipped": True,
            "measured": "RuntimeError: getCudnnDataTypeFromScalarType() not supported for Float8_e4m3fn",
        }
        del model
        model = None
        _empty()
        log(f"STATUS // transformer unloaded {vram()}")

        log(f"STATUS // load VAE {vram()}")
        try:
            vae = AutoencoderKLWan.from_single_file(str(VAE), dtype=torch.float32)
        except TypeError:
            vae = AutoencoderKLWan.from_single_file(str(VAE), torch_dtype=torch.float32)
        vae.to("cuda")
        report["masked_concat"] = masked_concat_stage(vae)
    except Exception as exc:
        report["fatal"] = f"{type(exc).__name__}: {exc}"
        report["trace"] = traceback.format_exc()[-4000:]
        log(f"STATUS // fatal {report['fatal']}")
    finally:
        del model, vae
        _empty()

    report["vram_end"] = vram()
    report["pick"] = pick(report)
    REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    log(f"STATUS // pick {report['pick']}")
    log(f"STATUS // wrote {REPORT}")
    return 0 if report["pick"].get("ready_for_worker") else 5


if __name__ == "__main__":
    raise SystemExit(main())
