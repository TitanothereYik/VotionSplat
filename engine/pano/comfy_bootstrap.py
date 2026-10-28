"""Load ComfyUI v0.32.0 as a library. Call apply_comfy_argv BEFORE importing comfy/nodes.

No Comfy server. No python_embeded. Qt must never import this module.
"""

from __future__ import annotations

import sys
from pathlib import Path

from engine.job import REPO_ROOT

COMFY_ROOT = REPO_ROOT / "vendor" / "comfyui"
USD_ROOT = REPO_ROOT / "vendor" / "ultimatesd"
OSTRIS_ROOT = REPO_ROOT / "vendor" / "ostris-edit"


def apply_comfy_argv(models_dir: Path, output_dir: Path, cuda_index: int) -> None:
    """Replace sys.argv so comfy.cli_args does not eat the worker flags."""
    models_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    sys.argv = [
        sys.argv[0],
        "--base-directory",
        str(COMFY_ROOT.resolve()),
        "--models-directory",
        str(models_dir.resolve()),
        "--output-directory",
        str(output_dir.resolve()),
        "--disable-all-custom-nodes",
        "--cuda-device",
        str(int(cuda_index)),
    ]


def ensure_comfy_on_path() -> None:
    root = str(COMFY_ROOT.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)


def ensure_usd_on_path() -> None:
    """Append (do not prepend) so ``import nodes`` stays ComfyUI, not USD."""
    usd = str(USD_ROOT.resolve())
    if usd not in sys.path:
        sys.path.append(usd)


def import_comfy_nodes():
    """Import Comfy node classes after apply_comfy_argv + ensure_comfy_on_path."""
    ensure_comfy_on_path()
    import comfy.options

    comfy.options.enable_args_parsing(True)
    import nodes  # noqa: F401  — registers built-ins; extras imported below
    from comfy_extras.nodes_flux import FluxKontextMultiReferenceLatentMethod
    from comfy_extras.nodes_model_advanced import ModelSamplingFlux
    from comfy_extras.nodes_upscale_model import ImageUpscaleWithModel, UpscaleModelLoader
    from nodes import (
        CLIPLoader,
        CLIPTextEncode,
        ConditioningZeroOut,
        EmptyLatentImage,
        ImageScale,
        KSampler,
        LoraLoaderModelOnly,
        SetLatentNoiseMask,
        UNETLoader,
        VAEDecode,
        VAEEncode,
        VAELoader,
    )

    return {
        "CLIPLoader": CLIPLoader,
        "FluxKontextMultiReferenceLatentMethod": FluxKontextMultiReferenceLatentMethod,
        "CLIPTextEncode": CLIPTextEncode,
        "ConditioningZeroOut": ConditioningZeroOut,
        "EmptyLatentImage": EmptyLatentImage,
        "ImageScale": ImageScale,
        "ImageUpscaleWithModel": ImageUpscaleWithModel,
        "UpscaleModelLoader": UpscaleModelLoader,
        "KSampler": KSampler,
        "LoraLoaderModelOnly": LoraLoaderModelOnly,
        "ModelSamplingFlux": ModelSamplingFlux,
        "SetLatentNoiseMask": SetLatentNoiseMask,
        "UNETLoader": UNETLoader,
        "VAEDecode": VAEDecode,
        "VAEEncode": VAEEncode,
        "VAELoader": VAELoader,
    }


def import_usd_no_upscale():
    """UltimateSDUpscaleNoUpscale at pin 627c871. Needs repositories/ultimate_sd_upscale.

    During import, USD must be first on sys.path so ``from utils import`` hits
    vendor/ultimatesd/utils.py. ``nodes`` is already cached as ComfyUI from
    import_comfy_nodes. After import, restore path order so Comfy's utils package
    is not shadowed.
    """
    ensure_comfy_on_path()
    script = USD_ROOT / "repositories" / "ultimate_sd_upscale" / "scripts" / "ultimate-upscale.py"
    if not script.is_file():
        raise FileNotFoundError(
            f"Ultimate SD Upscale script missing: {script}. "
            "Run: git -C vendor/ultimatesd submodule update --init"
        )
    import importlib.util

    usd = str(USD_ROOT.resolve())
    sys.path.insert(0, usd)
    try:
        spec = importlib.util.spec_from_file_location("votion_usdu_nodes", USD_ROOT / "nodes.py")
        if spec is None or spec.loader is None:
            raise ImportError("Could not load vendor/ultimatesd/nodes.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules["votion_usdu_nodes"] = mod
        spec.loader.exec_module(mod)
    finally:
        if sys.path and sys.path[0] == usd:
            sys.path.pop(0)
        util = sys.modules.get("utils")
        util_file = getattr(util, "__file__", "") or ""
        if util is not None and "ultimatesd" in util_file.replace("\\", "/"):
            sys.modules["usdu_utils"] = sys.modules.pop("utils")
    return mod.UltimateSDUpscaleNoUpscale


def import_ostris_edit():
    """Krea2OstrisEditModelPatch + TextEncodeKrea2OstrisEdit at graph pin 7756566.

    Load nodes.py via importlib (same pattern as USD). Do not import the pack as
    ``nodes`` and do not import ``__init__.py`` (relative ``from .nodes``).
    """
    ensure_comfy_on_path()
    script = OSTRIS_ROOT / "nodes.py"
    if not script.is_file():
        raise FileNotFoundError(
            f"Ostris Edit pack missing: {script}. "
            "Run tools\\bootstrap_p1_vendor.bat"
        )
    import importlib.util

    spec = importlib.util.spec_from_file_location("votion_ostris_edit", script)
    if spec is None or spec.loader is None:
        raise ImportError("Could not load vendor/ostris-edit/nodes.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["votion_ostris_edit"] = mod
    spec.loader.exec_module(mod)
    return mod.Krea2OstrisEditModelPatch, mod.TextEncodeKrea2OstrisEdit
