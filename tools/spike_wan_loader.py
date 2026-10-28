"""P3 WAN loader spike — Diffusers WanPipeline vs Matrix-3D panoramic_image_to_video.py.

Locked 2026-09-03 PRD Q&A: spike both named ports on this 3090, then pick.
Do not invent a third loader. Do not write engine.workers.wan until this file
records a winner.

Run:  python -m tools.spike_wan_loader
Hub cache: install-root .hf_cache (same as workers). Never C:\\Users\\…\\.cache.
"""

from __future__ import annotations

import importlib
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

REPORT = REPO_ROOT / "tools" / "spike_wan_loader.txt"
MODELS = REPO_ROOT / "models"
UNET = MODELS / "diffusion_models" / V.UNET_FILE
VAE = MODELS / "vae" / V.VAE_FILE
PANO_LORA = MODELS / "loras" / V.PANO_LORA_FILE
LIGHTX2V = MODELS / "loras" / V.LIGHTX2V_FILE
UMT5 = MODELS / "text_encoders" / V.UMT5_FILE
CLIP_VISION = MODELS / "clip_vision" / V.CLIP_VISION_FILE


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


def _pkg(name: str) -> dict:
    try:
        mod = importlib.import_module(name)
        return {"ok": True, "version": getattr(mod, "__version__", "?")}
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def stage_files() -> dict:
    rows = {
        "unet": UNET,
        "vae": VAE,
        "pano_lora": PANO_LORA,
        "lightx2v": LIGHTX2V,
        "umt5": UMT5,
        "clip_vision": CLIP_VISION,
    }
    out = {}
    for key, path in rows.items():
        out[key] = {
            "path": str(path),
            "exists": path.is_file(),
            "bytes": path.stat().st_size if path.is_file() else 0,
        }
    return out


def stage_imports() -> dict:
    names = [
        "torch",
        "diffusers",
        "accelerate",
        "transformers",
        "safetensors",
        "peft",
        "sageattention",
        "diffsynth",
        "xfuser",
        "modelscope",
    ]
    return {name: _pkg(name) for name in names}


def stage_matrix3d() -> dict:
    """Probe Matrix-3D panoramic_image_to_video.py on Windows. No second 14B download."""
    import torch

    nccl = torch.distributed.is_nccl_available()
    gloo = torch.distributed.is_gloo_available()
    diffsynth = _pkg("diffsynth")
    xfuser = _pkg("xfuser")
    modelscope = _pkg("modelscope")
    return {
        "script": "SkyworkAI/Matrix-3D code/panoramic_image_to_video.py",
        "uses": "DiffSynth-Studio WanVideoPipeline + torchrun + xfuser + nccl",
        "wants_weights": "Wan-AI/Wan2.1-I2V-14B-720P 7-shard + pano_video_gen_720p.bin (not Comfy keys)",
        "our_weights": "Kijai fp8 single file + pano_video_gen_720p_comfy.safetensors + LightX2V",
        "same_safetensors": False,
        "nccl_available": bool(nccl),
        "gloo_available": bool(gloo),
        "diffsynth": diffsynth,
        "xfuser": xfuser,
        "modelscope": modelscope,
        "import_ok": bool(diffsynth.get("ok") and xfuser.get("ok")),
        "windows_run_ok": False,
        "reason": (
            "script requires nccl torchrun, DiffSynth WanVideoPipeline, and Wan-AI 7-shard "
            "checkpoints + DiffSynth .bin LoRA. Product files are Comfy/Kijai safetensors + LightX2V. "
            "Not a thin loader of the same files."
        ),
    }


def stage_diffusers_symbols() -> dict:
    import diffusers

    names = [n for n in dir(diffusers) if "Wan" in n]
    out = {"version": getattr(diffusers, "__version__", "?"), "wan_names": names}
    for attr in (
        "WanImageToVideoPipeline",
        "WanPipeline",
        "WanTransformer3DModel",
        "AutoencoderKLWan",
        "FlowMatchEulerDiscreteScheduler",
    ):
        out[attr] = hasattr(diffusers, attr)
    return out


def _from_single_file(cls, path: Path, **kwargs):
    """Diffusers Wan docs use dtype=; older signatures used torch_dtype=."""
    try:
        return cls.from_single_file(str(path), **kwargs)
    except TypeError:
        if "dtype" in kwargs:
            kwargs = dict(kwargs)
            kwargs["torch_dtype"] = kwargs.pop("dtype")
            return cls.from_single_file(str(path), **kwargs)
        raise


def stage_diffusers_load_transformer() -> dict:
    """Load the Quality UNET (Kijai fp8) through Diffusers from_single_file.

    3090 is Ampere: store fp8, compute in bfloat16. Do not leave the 14B on GPU
    after the probe. Docs: WanTransformer3DModel.from_single_file (local path).
    """
    import gc

    import torch
    from diffusers import WanTransformer3DModel

    t0 = time.time()
    result = {"file": str(UNET), "vram_before": vram()}
    log(f"STATUS // diffusers from_single_file {UNET.name} {vram()}")
    model = None
    try:
        model = _from_single_file(
            WanTransformer3DModel,
            UNET,
            dtype=torch.float8_e4m3fn,
        )
        result["load_dtype"] = "float8_e4m3fn"
        result["param_dtype"] = str(next(model.parameters()).dtype)
        cfg_keys = list(model.config.keys())
        result["config"] = {k: model.config[k] for k in cfg_keys[:12]}
        log(f"STATUS // transformer on CPU {time.time()-t0:.1f}s dtype={result['param_dtype']} {vram()}")
        # 14B bf16 ≈ 28 GB; 3090 is 24 GB. Keep fp8 storage (Comfy/Kijai Quality path).
        model.to(device="cuda")
        result["cuda"] = True
        result["cuda_dtype"] = str(next(model.parameters()).dtype)
        result["vram_after_cuda"] = vram()
        result["ok"] = True
        result["seconds"] = round(time.time() - t0, 1)
        log(f"STATUS // transformer on CUDA {result['cuda_dtype']} {result['seconds']}s {vram()}")
    except Exception as exc:
        result["ok"] = False
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["trace"] = traceback.format_exc()[-4000:]
        result["seconds"] = round(time.time() - t0, 1)
        log(f"STATUS // transformer FAIL {result['error']}")
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    result["vram_after"] = vram()
    return result


def stage_diffusers_load_vae() -> dict:
    import gc

    import torch
    from diffusers import AutoencoderKLWan

    t0 = time.time()
    result = {"file": str(VAE), "vram_before": vram()}
    log(f"STATUS // AutoencoderKLWan.from_single_file {VAE.name}")
    try:
        vae = _from_single_file(AutoencoderKLWan, VAE, dtype=torch.float32)
        vae.to("cuda")
        result["ok"] = True
        result["seconds"] = round(time.time() - t0, 1)
        result["vram_after_cuda"] = vram()
        del vae
    except Exception as exc:
        result["ok"] = False
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["trace"] = traceback.format_exc()[-2500:]
        result["seconds"] = round(time.time() - t0, 1)
        log(f"STATUS // vae FAIL {result['error']}")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def stage_diffusers_lora_keys() -> dict:
    """Do not apply LoRAs yet. Count keys so we know Comfy vs Diffusers naming."""
    from safetensors import safe_open

    def sample(path: Path, n: int = 6) -> dict:
        keys = []
        with safe_open(str(path), framework="pt", device="cpu") as fh:
            keys = list(fh.keys())
        return {
            "file": path.name,
            "n_keys": len(keys),
            "sample": keys[:n],
            "has_diffusion_model": any(k.startswith("diffusion_model.") for k in keys),
            "has_lora_down": any("lora_down" in k for k in keys),
            "has_lora_A": any("lora_A" in k for k in keys),
        }

    return {"pano": sample(PANO_LORA), "lightx2v": sample(LIGHTX2V)}


def pick(report: dict) -> dict:
    diff_tf = report.get("diffusers_transformer") or {}
    matrix = report.get("matrix3d") or {}
    if diff_tf.get("ok"):
        winner = "diffusers_WanTransformer3DModel.from_single_file"
        note = (
            "Diffusers loaded the Kijai fp8 UNET on this 3090. Matrix-3D script is not a "
            "thin loader of the same safetensors (DiffSynth + NCCL + Wan-AI 7-shard)."
        )
    elif matrix.get("import_ok") and matrix.get("windows_run_ok"):
        winner = "matrix3d_panoramic_image_to_video"
        note = "Matrix-3D import and Windows run succeeded."
    else:
        winner = None
        note = (
            "Neither named port loaded the product safetensors on this machine yet. "
            "Do not invent a third loader. Do not write engine.workers.wan."
        )
    return {"winner": winner, "note": note}


def main() -> int:
    env = apply_huggingface_cache_env(worker_env(0))
    os.environ.update(env)
    log("STATUS // spike_wan_loader start")
    report: dict = {
        "locked": {
            "first_slice": "WAN 14B already in models/",
            "wan_loader": "spike_then_pick",
            "dualres_8192": "json_input_types (max_num_features)",
            "sfm_when": "generate_then_recon_shows",
            "confirm_play": "play_each",
            "max_rails": 5,
            "rail_files": "per_rail_dir",
            "first_open_defaults": "smpl_four",
        },
        "files": stage_files(),
        "imports": stage_imports(),
        "vram_start": vram(),
    }
    missing = [k for k, v in report["files"].items() if not v["exists"]]
    if missing:
        report["pick"] = {"winner": None, "note": f"missing files {missing}"}
        REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        log(f"STATUS // missing {missing}")
        return 2

    report["matrix3d"] = stage_matrix3d()
    log(f"STATUS // matrix3d nccl={report['matrix3d']['nccl_available']} diffsynth={report['matrix3d']['diffsynth']}")

    if not report["imports"]["diffusers"]["ok"]:
        report["pick"] = {
            "winner": None,
            "note": "diffusers not installed. pip install diffusers accelerate then re-run.",
        }
        REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        log("STATUS // install diffusers then re-run")
        print(json.dumps(report["pick"], indent=2), flush=True)
        return 3

    report["diffusers_symbols"] = stage_diffusers_symbols()
    log(f"STATUS // diffusers {report['diffusers_symbols'].get('version')} names={report['diffusers_symbols'].get('wan_names')}")
    report["lora_keys"] = stage_diffusers_lora_keys()
    report["diffusers_vae"] = stage_diffusers_load_vae()
    report["diffusers_transformer"] = stage_diffusers_load_transformer()
    report["pick"] = pick(report)
    REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    log(f"STATUS // pick {report['pick']}")
    log(f"STATUS // wrote {REPORT}")
    return 0 if report["pick"].get("winner") else 4


if __name__ == "__main__":
    raise SystemExit(main())
