"""WorldStereo loaders used by the product worker.

Moved out of ``tools/spike_worldstereo.py`` so the app does not import a spike.
The spike re-exports these names.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch

from engine.job import REPO_ROOT
from engine.worldstereo import defaults as D

_WS = REPO_ROOT / "vendor" / "worldstereo"
if str(_WS) not in sys.path:
    sys.path.insert(0, str(_WS))

WAN_CFG = (
    REPO_ROOT
    / ".hf_cache"
    / "hub"
    / "models--Wan-AI--Wan2.1-I2V-14B-720P-Diffusers"
    / "snapshots"
    / "eb849f76dfa246545b65774a9e25943ee69b3fa3"
)


def _log(msg: str) -> None:
    print(msg, flush=True)


def jsonable(obj):
    if isinstance(obj, dict):
        return {str(key): jsonable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(value) for value in obj]
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


def patch_single_rank() -> None:
    """One process, gloo. The stock scripts init NCCL and a CUDA device mesh."""
    import torch.distributed as dist

    _log(f"STATUS // MASTER_ADDR={os.environ.get('MASTER_ADDR')} MASTER_PORT={os.environ.get('MASTER_PORT')}")
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

    config_path = Path(hf_hub_download(D.CONFIG_REPO, "config.json", subfolder=D.CONFIG_SUBFOLDER))
    local_weights = REPO_ROOT / "models" / "diffusion_models" / D.FP8_FILE
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
    transformer_cfg = Path(hf_hub_download(D.BASE_MODEL, "transformer/config.json"))
    dino = REPO_ROOT / "models" / "dinov2-base"
    if not (dino / "config.json").is_file():
        snapshot_download(D.DINO_REPO, local_dir=str(dino))
    return config_path, weights, transformer_cfg


def _materialize_rope(rope) -> None:
    from diffusers.models.transformers.transformer_wan import WanRotaryPosEmbed

    real = WanRotaryPosEmbed(rope.attention_head_dim, tuple(rope.patch_size), rope.max_seq_len)
    rope.freqs_cos = real.freqs_cos
    rope.freqs_sin = real.freqs_sin


def build_dit(
    transformer_cfg: Path, ws_cfg_path: Path, weights: Path, device: torch.device, load_device=None
):
    """``load_device`` is where the weights are put. ``None`` puts them on ``device``."""
    from omegaconf import OmegaConf

    from engine.worldstereo.fp8 import load_scaled_fp8_dit
    from models.worldstereo import WorldStereoRefSModel

    raw = json.loads(transformer_cfg.read_text(encoding="utf-8"))
    skip = {"_class_name", "_diffusers_version", "_name_or_path", "_use_default_values"}
    kwargs = {key: value for key, value in raw.items() if key not in skip}
    ws_cfg = OmegaConf.create(json.loads(ws_cfg_path.read_text(encoding="utf-8")))
    _log("STATUS // building WorldStereoRefSModel on the meta device")
    with torch.device("meta"):
        model = WorldStereoRefSModel(
            controlnet_cfg=ws_cfg.controlnet_cfg,
            base_model=str(ws_cfg.base_model),
            **kwargs,
        )
    for name in ("weight", "bias"):
        current = getattr(model.patch_embedding, name)
        empty = torch.empty(current.shape, dtype=torch.bfloat16, device="cpu")
        setattr(model.patch_embedding, name, torch.nn.Parameter(empty, requires_grad=False))
    model.build_controlnet(load_uni3c=False, freeze_backbone=False)
    _materialize_rope(model.rope)
    if getattr(model, "controlnet_rope", None) is not None and model.controlnet_rope.freqs_cos.is_meta:
        _materialize_rope(model.controlnet_rope)
    _log(f"STATUS // loading FP8 DiT {weights.name}")
    torch.cuda.reset_peak_memory_stats(device)
    target = device if load_device is None else torch.device(load_device)
    stats = load_scaled_fp8_dit(model, weights, target)
    model.to(target)
    model.eval()
    _log(
        f"STATUS // fp8_linear={stats['fp8_linear']} bf16_linear={stats['bf16_linear']} "
        f"other={stats['other']} unused={stats['unused']} dtype={model.dtype} "
        f"resident_gb={torch.cuda.memory_allocated(device) / (1024**3):.2f} "
        f"peak_gb={_gb():.2f}"
    )
    if model.dtype != torch.bfloat16:
        raise RuntimeError(f"DiT compute dtype is {model.dtype}, expected bfloat16")
    return model, ws_cfg, stats


def load_clip_vision(cfg_root: Path):
    """clip_vision_h checkpoint keys are ``vision_model.*``. CLIPVisionModel is not."""
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
    _log(f"STATUS // CLIP-Vision missing={len(missing)} unexpected={sorted(unexpected)}")
    if missing:
        raise RuntimeError(f"CLIP vision is missing weights: {missing[:8]}")
    del state, remapped
    return model


def load_vae():
    from engine.worldstereo.encoders import load_vae as _load

    return _load()
