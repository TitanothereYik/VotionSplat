"""WorldStereo memory-DMD FP8 spike. Outside the app. One example trajectory.

Does not touch WAN outputs. Does not call WorldMirror / HY-World.

Patches recorded in docs/env_report.txt:

* Process group is gloo, rank 0 of 1. No NCCL, no device mesh.
  ``ParallelDims.build_mesh`` is a no-op so sequence-parallel stays off.
* DINOv2 loads from ``models/dinov2-base`` with ``local_files_only``.
* The DiT is the scaled FP8 file, not a BF16 dequant of the whole checkpoint.
"""

from __future__ import annotations

import gc
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor" / "worldstereo"))

os.environ["HF_HOME"] = str(ROOT / ".hf_cache")
os.environ["HF_HUB_CACHE"] = str(ROOT / ".hf_cache" / "hub")
os.environ["HF_XET_CACHE"] = str(ROOT / ".hf_cache" / "xet")
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
# Force loopback. A machine-level MASTER_ADDR (this PC's DNS name) is not a
# bindable address, and gloo then fails with WSAEADDRNOTAVAIL.
os.environ["MASTER_ADDR"] = "127.0.0.1"
os.environ["MASTER_PORT"] = "29531"
os.environ["RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"

import torch
import torch.distributed as dist

from engine.worldstereo import defaults as D
from engine.worldstereo.fp8 import load_scaled_fp8_dit

REPORT = ROOT / "docs" / "env_report.txt"
SCENE = ROOT / "vendor" / "worldstereo" / "examples" / "panorama" / "9c2dd62c7bb946a49e8e51278277eb6a"
OUT = ROOT / "outputs" / "_worldstereo_spike"
VIEW = "view0"
TRAJ = "traj0"
WAN_CFG = (
    ROOT
    / ".hf_cache"
    / "hub"
    / "models--Wan-AI--Wan2.1-I2V-14B-720P-Diffusers"
    / "snapshots"
    / "eb849f76dfa246545b65774a9e25943ee69b3fa3"
)


def log(msg: str) -> None:
    print(msg, flush=True)


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(key): _jsonable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(value) for value in obj]
    if hasattr(obj, "item") and callable(obj.item):
        try:
            return obj.item()
        except (ValueError, RuntimeError):
            return obj.tolist()
    return obj


def _gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024**3)


def append_report(text: str) -> None:
    with REPORT.open("a", encoding="utf-8") as handle:
        handle.write(text)
        if not text.endswith("\n"):
            handle.write("\n")


def patch_single_rank() -> None:
    """One process, gloo. The stock scripts init NCCL and a CUDA device mesh."""
    log(f"STATUS // MASTER_ADDR={os.environ.get('MASTER_ADDR')} MASTER_PORT={os.environ.get('MASTER_PORT')}")
    if not dist.is_initialized():
        dist.init_process_group(backend="gloo", rank=0, world_size=1, init_method="tcp://127.0.0.1:29531")
    from src.sp_utils.parallel_states import ParallelDims

    def build_mesh(self, device_type):  # type: ignore[no-untyped-def]
        self.world_mesh = None
        return None

    ParallelDims.build_mesh = build_mesh  # type: ignore[method-assign]


def pin_dinov2(local_dir: Path) -> None:
    from src.retrieval_wm import CameraSelector

    def _load_model(self, extractor: str) -> None:
        if extractor != "dinov2":
            raise ValueError(extractor)
        from transformers import AutoImageProcessor, AutoModel

        self.processor = AutoImageProcessor.from_pretrained(str(local_dir), local_files_only=True)
        self.model = AutoModel.from_pretrained(str(local_dir), local_files_only=True).to(self.device)

    CameraSelector._load_model = _load_model  # type: ignore[method-assign]


def ensure_files() -> tuple[Path, Path, Path]:
    from huggingface_hub import hf_hub_download, snapshot_download

    config_path = Path(
        hf_hub_download(D.CONFIG_REPO, "config.json", subfolder=D.CONFIG_SUBFOLDER)
    )
    local_weights = ROOT / "models" / "diffusion_models" / D.FP8_FILE
    if local_weights.is_file():
        size = local_weights.stat().st_size
        if size != D.FP8_BYTES:
            raise RuntimeError(
                f"{local_weights} is {size} bytes, expected {D.FP8_BYTES}. "
                "Leave the curl download running; do not start a second copy."
            )
        weights = local_weights
    else:
        weights = Path(hf_hub_download(D.FP8_REPO, D.FP8_FILE))
    transformer_cfg = Path(
        hf_hub_download(D.BASE_MODEL, "transformer/config.json")
    )
    dino = ROOT / "models" / "dinov2-base"
    if not (dino / "config.json").is_file():
        snapshot_download(D.DINO_REPO, local_dir=str(dino))
    return config_path, weights, transformer_cfg


def _materialize_rope(rope) -> None:
    from diffusers.models.transformers.transformer_wan import WanRotaryPosEmbed

    real = WanRotaryPosEmbed(rope.attention_head_dim, tuple(rope.patch_size), rope.max_seq_len)
    rope.freqs_cos = real.freqs_cos
    rope.freqs_sin = real.freqs_sin


def build_dit(transformer_cfg: Path, ws_cfg_path: Path, weights: Path, device: torch.device):
    from omegaconf import OmegaConf

    from models.worldstereo import WorldStereoRefSModel

    raw = json.loads(transformer_cfg.read_text(encoding="utf-8"))
    skip = {"_class_name", "_diffusers_version", "_name_or_path", "_use_default_values"}
    kwargs = {key: value for key, value in raw.items() if key not in skip}
    ws_cfg = OmegaConf.create(json.loads(ws_cfg_path.read_text(encoding="utf-8")))
    log("STATUS // building WorldStereoRefSModel on the meta device")
    with torch.device("meta"):
        model = WorldStereoRefSModel(
            controlnet_cfg=ws_cfg.controlnet_cfg,
            base_model=str(ws_cfg.base_model),
            **kwargs,
        )
    # build_controlnet copies patch_embedding. Give it a real tensor first.
    for name in ("weight", "bias"):
        current = getattr(model.patch_embedding, name)
        empty = torch.empty(current.shape, dtype=torch.bfloat16, device="cpu")
        setattr(model.patch_embedding, name, torch.nn.Parameter(empty, requires_grad=False))
    model.build_controlnet(load_uni3c=False, freeze_backbone=False)
    # RoPE tables are computed in __init__. Under the meta device they have no data.
    _materialize_rope(model.rope)
    if getattr(model, "controlnet_rope", None) is not None and model.controlnet_rope.freqs_cos.is_meta:
        _materialize_rope(model.controlnet_rope)
    log(f"STATUS // loading FP8 DiT {weights.name}")
    torch.cuda.reset_peak_memory_stats(device)
    stats = load_scaled_fp8_dit(model, weights, device)
    model.to(device)
    model.eval()
    log(
        f"STATUS // fp8_linear={stats['fp8_linear']} bf16_linear={stats['bf16_linear']} "
        f"other={stats['other']} unused={stats['unused']} dtype={model.dtype} "
        f"resident_gb={torch.cuda.memory_allocated(device) / (1024**3):.2f} "
        f"peak_gb={_gb():.2f}"
    )
    if model.dtype != torch.bfloat16:
        raise RuntimeError(f"DiT compute dtype is {model.dtype}, expected bfloat16")
    return model, ws_cfg, stats


def encode_then_drop(prompt: str, image_path: Path) -> tuple[torch.Tensor, torch.Tensor, object, object, object]:
    """Text encoder and CLIP run, then leave the GPU before the DiT is resident."""
    from PIL import Image
    from transformers import AutoTokenizer, CLIPImageProcessor

    from engine.worldstereo.encoders import load_umt5

    cfg = WAN_CFG
    tokenizer = AutoTokenizer.from_pretrained(str(cfg / "tokenizer"), local_files_only=True)
    text = load_umt5(cfg).to("cuda").eval()
    tokens = tokenizer(
        prompt,
        padding="max_length",
        max_length=512,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        hidden = text(tokens.input_ids.to("cuda"), tokens.attention_mask.to("cuda")).last_hidden_state
    length = int(tokens.attention_mask.gt(0).sum())
    hidden = hidden[0, :length]
    pad = hidden.new_zeros(512 - length, hidden.shape[-1])
    prompt_embeds = torch.cat([hidden, pad], dim=0).unsqueeze(0).cpu()
    text.to("cpu")
    del hidden, tokens
    gc.collect()
    torch.cuda.empty_cache()
    log(f"STATUS // text encoder off GPU; prompt embeds {tuple(prompt_embeds.shape)}")

    processor = CLIPImageProcessor.from_pretrained(str(cfg / "image_processor"), local_files_only=True)
    clip = load_clip_vision(cfg).to("cuda").eval()
    batch = processor(images=Image.open(image_path).convert("RGB"), return_tensors="pt")
    batch = {key: value.to("cuda") for key, value in batch.items()}
    with torch.no_grad():
        image_embeds = clip(**batch, output_hidden_states=True).hidden_states[-2].cpu()
    clip.to("cpu")
    del batch
    gc.collect()
    torch.cuda.empty_cache()
    log(f"STATUS // CLIP off GPU; image embeds {tuple(image_embeds.shape)}")
    return prompt_embeds, image_embeds, text, clip, tokenizer, processor


def load_clip_vision(cfg_root: Path):
    """Comfy clip_vision_h keys are ``vision_model.*``. CLIPVisionModel is not.

    ``visual_projection`` and ``position_ids`` are not parameters of this module.
    """
    from safetensors.torch import load_file
    from transformers import CLIPVisionConfig, CLIPVisionModel

    from engine.worldstereo.encoders import CLIP_VISION

    config = CLIPVisionConfig.from_pretrained(str(cfg_root / "image_encoder"), local_files_only=True)
    model = CLIPVisionModel(config)
    state = load_file(str(CLIP_VISION))
    remapped = {}
    for key, value in state.items():
        name = key[len("vision_model.") :] if key.startswith("vision_model.") else key
        remapped[name] = value
    missing, unexpected = model.load_state_dict(remapped, strict=False)
    log(f"STATUS // CLIP-Vision missing={len(missing)} unexpected={sorted(unexpected)}")
    if missing:
        raise RuntimeError(f"CLIP vision is missing weights: {missing[:8]}")
    del state, remapped
    return model


def load_vae():
    from diffusers import AutoencoderKLWan

    from engine.worldstereo.encoders import VAE_FILE, _from_single_file

    path = ROOT / "models" / "vae" / VAE_FILE
    vae = _from_single_file(AutoencoderKLWan, path, dtype=torch.bfloat16)
    vae.enable_tiling()
    vae.eval()
    return vae


def prepare_memory(ws_cfg) -> torch.Tensor | None:
    """Write the memory mp4, then drop DINOv2 before the DiT is resident."""
    import numpy as np
    import imagesize
    from diffusers.utils import export_to_video

    from src.retrieval_wm import SimpleMemoryBank

    device = torch.device("cuda:0")
    width, height = imagesize.get(str(SCENE / "pano_bank" / "images" / "0000.png"))
    bank = SimpleMemoryBank(
        cfg=ws_cfg,
        root_path=str(SCENE),
        image_width=width,
        image_height=height,
        device=device,
        max_reference=8,
        align_nframe=D.ALIGN_NFRAME,
        rank=0,
        world_size=1,
    )
    cam = json.loads((SCENE / VIEW / TRAJ / "camera.json").read_text(encoding="utf-8"))
    tar_w2cs = torch.tensor(np.array(cam["extrinsic"]), dtype=torch.float32, device=device)
    tar_ks = torch.tensor(np.array(cam["intrinsic"]), dtype=torch.float32, device=device)
    retrieved, ref_index, ref_index_dict, ref_w2cs, _ = bank.retrieval(tar_w2cs, tar_ks, view_id=VIEW, traj_id=TRAJ)
    out_dir = OUT / VIEW / TRAJ
    mem_dir = out_dir / "memory_inputs"
    mem_dir.mkdir(parents=True, exist_ok=True)
    export_to_video(retrieved / 255, str(mem_dir / "worldstereo-memory-dmd.mp4"), fps=16)
    if ref_index_dict is not None:
        (mem_dir / "worldstereo-memory-dmd_ref_index.json").write_text(
            json.dumps(_jsonable(ref_index_dict), indent=2), encoding="utf-8"
        )
    if ref_w2cs is not None:
        (mem_dir / "worldstereo-memory-dmd_ref_w2cs.json").write_text(
            json.dumps(ref_w2cs.detach().cpu().numpy().tolist(), indent=2), encoding="utf-8"
        )
    kept = None if ref_index is None else ref_index.detach().cpu()
    n_ref = None if kept is None else int(kept.shape[0])
    del bank, retrieved, tar_w2cs, tar_ks, ref_w2cs, ref_index
    gc.collect()
    torch.cuda.empty_cache()
    log(f"STATUS // retrieval wrote memory mp4; refs={n_ref}; DINOv2 off GPU")
    return kept


def run_one(model, ws_cfg, prompt_embeds, image_embeds, text, clip, tokenizer, processor, vae, ref_index) -> dict:
    from diffusers.utils import export_to_video

    from models.dmd_scheduler import FlowGeneratorScheduler
    from models.pipelines.pipeline_dmd_keyframe import RefKFDMDGeneratorPipeline
    from src.data_utils import load_mutli_traj_dataset

    device = torch.device("cuda:0")
    vae = vae.to(device)
    scheduler = FlowGeneratorScheduler(
        start_timesteps=int(ws_cfg.dmd_start_steps),
        num_train_timesteps=int(ws_cfg.dmd_end_steps),
        shift=float(ws_cfg.gen_shift),
        use_timestep_transform=True,
        dmd_steps=int(ws_cfg.dmd_steps),
        rank=dist.get_rank(),
    )
    # check_inputs treats image and image_embeds as alternatives. This pipeline
    # uses image as the VAE frame and image_embeds as the already-computed CLIP
    # embedding, so CLIP can stay off the GPU.
    original_check = RefKFDMDGeneratorPipeline.check_inputs

    def check_inputs(self, prompt, negative_prompt, image, height, width, prompt_embeds=None, negative_prompt_embeds=None, image_embeds=None, callback_on_step_end_tensor_inputs=None):
        return original_check(
            self,
            prompt,
            negative_prompt,
            image,
            height,
            width,
            prompt_embeds,
            negative_prompt_embeds,
            None,
            callback_on_step_end_tensor_inputs,
        )

    RefKFDMDGeneratorPipeline.check_inputs = check_inputs
    pipeline = RefKFDMDGeneratorPipeline(
        tokenizer=tokenizer,
        text_encoder=text,
        image_encoder=clip,
        image_processor=processor,
        transformer=model,
        vae=vae,
        scheduler=scheduler,
        device=device,
        vae_compile=False,
    )
    # Confirm the text encoder was not moved back onto the GPU with the DiT.
    text_device = next(text.parameters()).device
    log(f"STATUS // text encoder device with DiT resident: {text_device}")
    if text_device.type == "cuda":
        raise RuntimeError("text encoder is on GPU at the same time as the DiT")

    out_dir = OUT / VIEW / TRAJ
    meta = load_mutli_traj_dataset(
        cfg=ws_cfg,
        input_path=str(SCENE),
        output_path=str(OUT),
        view_id=VIEW,
        traj_id=TRAJ,
        device=device,
        ref_index=ref_index,
        model_type="worldstereo-memory-dmd",
        task_type="panorama",
    )
    kwargs = {key: value for key, value in meta.items() if value is not None}
    kwargs.pop("prompt", None)
    kwargs.update(
        prompt_embeds=prompt_embeds.to(device=device, dtype=torch.bfloat16),
        image_embeds=image_embeds.to(device=device, dtype=torch.bfloat16),
        generator=torch.Generator(device=device).manual_seed(D.SEED),
        output_type="pt",
        latent_cond_mode=str(ws_cfg.latent_cond_mode),
        mode="test",
    )
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        frames = pipeline(**kwargs).frames[0].float()
    elapsed = time.perf_counter() - started
    frames_np = frames.cpu().permute(0, 2, 3, 1).numpy()
    export_to_video(frames_np, str(out_dir / "worldstereo-memory-dmd_result.mp4"), fps=16)
    peak = _gb()
    log(f"STATUS // trajectory frames={frames_np.shape[0]} size={frames_np.shape[1]}x{frames_np.shape[2]} seconds={elapsed:.1f} peak_gb={peak:.2f}")
    return {
        "frames": int(frames_np.shape[0]),
        "height": int(frames_np.shape[1]),
        "width": int(frames_np.shape[2]),
        "seconds": elapsed,
        "peak_gb": peak,
        "retrieved": None if ref_index is None else len(ref_index),
    }


def main() -> int:
    load_only = "--load-only" in sys.argv
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    patch_single_rank()
    log("STATUS // gloo rank 0/1, CUDA device mesh skipped")

    try:
        import pytorch3d  # noqa: F401

        p3d = f"import OK {getattr(pytorch3d, '__version__', 'present')}"
    except Exception as exc:
        p3d = f"MISSING {type(exc).__name__}: {exc}"
    log(f"STATUS // pytorch3d {p3d}")

    config_path, weights, transformer_cfg = ensure_files()
    dino = ROOT / "models" / "dinov2-base"
    pin_dinov2(dino)

    prompt = json.loads((SCENE / VIEW / TRAJ / "traj_caption.json").read_text(encoding="utf-8"))["prompt"]
    image_path = SCENE / VIEW / "start_frame.png"
    encoded = encode_then_drop(prompt, image_path)
    prompt_embeds, image_embeds, text, clip, tokenizer, processor = encoded
    if torch.cuda.memory_allocated(device) > 512 * 1024**2:
        raise RuntimeError("GPU still holds more than 512 MB after the text encoder and CLIP left")

    ref_index = None
    if not load_only and not p3d.startswith("MISSING"):
        from omegaconf import OmegaConf

        preview = OmegaConf.create(json.loads(config_path.read_text(encoding="utf-8")))
        ref_index = prepare_memory(preview)
        if torch.cuda.memory_allocated(device) > 512 * 1024**2:
            raise RuntimeError("GPU still holds more than 512 MB after retrieval")

    model, ws_cfg, stats = build_dit(transformer_cfg, config_path, weights, device)
    vae = load_vae()
    result = {
        "pytorch3d": p3d,
        "fp8": stats,
        "dit_gb": torch.cuda.memory_allocated(device) / (1024**3),
        "dit_dtype": str(model.dtype),
        "weights": str(weights),
    }
    if load_only or p3d.startswith("MISSING"):
        result["trajectory"] = "skipped"
    else:
        result["trajectory"] = run_one(
            model, ws_cfg, prompt_embeds, image_embeds, text, clip, tokenizer, processor, vae, ref_index
        )

    stamp = datetime.now(timezone.utc).isoformat()
    lines = [
        "",
        "## worldstereo spike",
        f"written={stamp}",
        f"code_sha=5ba8dc3d37c48ecef99bbd603815ced513a4a431",
        f"pytorch3d: {p3d}; source 75ebeeaea0908c5527e7b1e305fbc7681382db47; pulsar global.h uint/ushort macros replaced with using-aliases so CUDA 12.8 cuda/std headers compile",
        "process_group: gloo world_size=1; ParallelDims.build_mesh no-op (no NCCL, no device mesh)",
        f"cuda_alloc={os.environ.get('PYTORCH_CUDA_ALLOC_CONF', '')}",
        f"dinov2: {dino} local_files_only",
        f"fp8_file: {weights}",
        f"fp8_linear={stats['fp8_linear']} bf16_linear={stats['bf16_linear']} other={stats['other']} unused={stats['unused']}",
        f"dit_dtype={model.dtype} dit_allocated_gb={result['dit_gb']:.2f}",
        "text_encoder_with_dit: cpu",
        "clip: comfy keys remapped by stripping vision_model.; image and image_embeds both passed (check_inputs mutual exclusion skipped)",
        f"mask: white is empty (zbuf == -1). FoV {D.FOV_H_DEG:.0f}x{D.FOV_V_DEG:.0f} at {D.SIZE_H}x{D.SIZE_W}. pano bank views={D.PANO_BANK_VIEWS}",
        f"result: {json.dumps(result['trajectory'])}",
    ]
    append_report("\n".join(lines) + "\n")
    log("STATUS // env_report updated")
    if isinstance(result["trajectory"], str):
        return 2
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:
        append_report(f"\n## worldstereo spike\nfailed={type(exc).__name__}: {exc}\n")
        raise
