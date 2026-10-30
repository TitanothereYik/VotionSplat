"""Load the scaled-FP8 WorldStereo DiT without expanding it to BF16.

Ampere has no FP8 matmul. Each Linear keeps its weight in float8_e4m3fn and
upcasts that one matrix inside the call:

    F.linear(x, weight.to(x.dtype) * scale, bias)

The file must contain the ``scaled_fp8`` marker. A file without it is refused.
"""

from __future__ import annotations

import gc
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open


class ScaledFP8Linear(nn.Module):
    """nn.Linear replacement. Weight stays float8_e4m3fn.

    The bits are stored as uint8. ``float8_e4m3fn`` counts as a floating dtype,
    and ``ModelMixin.dtype`` returns the first floating parameter. A float8
    weight would make the pipeline cast activations to float8. The forward
    views the bits back and upcasts that one matrix.
    """

    def __init__(self, weight: torch.Tensor, scale: torch.Tensor, bias: torch.Tensor | None) -> None:
        super().__init__()
        if weight.dtype != torch.float8_e4m3fn:
            raise TypeError(f"FP8 linear weight is {weight.dtype}, expected float8_e4m3fn")
        if weight.ndim != 2:
            raise ValueError(f"FP8 linear weight rank is {weight.ndim}, expected 2")
        bits = weight.detach().contiguous().view(torch.uint8)
        self.weight = nn.Parameter(bits, requires_grad=False)
        self.scale = nn.Parameter(scale.reshape(()).to(dtype=torch.bfloat16), requires_grad=False)
        if bias is None:
            self.register_parameter("bias", None)
        else:
            self.bias = nn.Parameter(bias, requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.scale.to(dtype=x.dtype)
        weight = self.weight.view(torch.float8_e4m3fn).to(dtype=x.dtype) * scale
        bias = None if self.bias is None else self.bias.to(dtype=x.dtype)
        return F.linear(x, weight, bias)


def _parent(root: nn.Module, qualname: str) -> tuple[nn.Module, str]:
    parts = qualname.split(".")
    module = root
    for part in parts[:-1]:
        module = getattr(module, part)
    return module, parts[-1]


def _assign_param(module: nn.Module, name: str, tensor: torch.Tensor) -> None:
    current = getattr(module, name)
    if isinstance(current, nn.Parameter):
        setattr(module, name, nn.Parameter(tensor, requires_grad=False))
    else:
        setattr(module, name, tensor)


def load_scaled_fp8_dit(model: nn.Module, path: Path, device: torch.device) -> dict[str, int]:
    """Fill ``model`` from a scaled FP8 safetensors file.

    Linears whose weight is float8 become ``ScaledFP8Linear``. Every other
    tensor is copied in its stored dtype (BF16 for norms, convs, biases).
    Nothing is multiplied by its scale until that Linear runs.
    """
    path = Path(path)
    stats = {"fp8_linear": 0, "bf16_linear": 0, "other": 0}
    consumed: set[str] = set()

    with safe_open(str(path), framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        if "scaled_fp8" not in keys:
            raise RuntimeError(f"{path.name} has no scaled_fp8 marker; refusing to load")
        consumed.add("scaled_fp8")

        linears = [(name, mod) for name, mod in model.named_modules() if isinstance(mod, nn.Linear)]
        for name, linear in linears:
            weight_key = f"{name}.weight"
            if weight_key not in keys:
                continue
            weight = handle.get_tensor(weight_key)
            bias_key = f"{name}.bias"
            bias = handle.get_tensor(bias_key) if bias_key in keys else None
            parent, attr = _parent(model, name)
            if weight.dtype == torch.float8_e4m3fn:
                scale_key = f"{name}.scale_weight"
                if scale_key not in keys:
                    raise RuntimeError(f"{weight_key} is FP8 but {scale_key} is missing")
                scale = handle.get_tensor(scale_key)
                replacement = ScaledFP8Linear(
                    weight.to(device=device),
                    scale.to(device=device),
                    None if bias is None else bias.to(device=device),
                )
                setattr(parent, attr, replacement)
                consumed.update({weight_key, scale_key})
                if bias is not None:
                    consumed.add(bias_key)
                stats["fp8_linear"] += 1
            else:
                fresh = nn.Linear(weight.shape[1], weight.shape[0], bias=bias is not None, device=device, dtype=weight.dtype)
                with torch.no_grad():
                    fresh.weight.copy_(weight.to(device=device))
                    if bias is not None:
                        fresh.bias.copy_(bias.to(device=device))
                setattr(parent, attr, fresh)
                consumed.add(weight_key)
                if bias is not None:
                    consumed.add(bias_key)
                stats["bf16_linear"] += 1
            del weight, bias

        for qualname, param in list(model.named_parameters()):
            if qualname in consumed or qualname not in keys:
                continue
            tensor = handle.get_tensor(qualname).to(device=device)
            parent, attr = _parent(model, qualname)
            _assign_param(parent, attr, tensor)
            consumed.add(qualname)
            stats["other"] += 1

        for qualname, buf in list(model.named_buffers()):
            if qualname in consumed or qualname not in keys:
                continue
            tensor = handle.get_tensor(qualname).to(device=device)
            parent, attr = _parent(model, qualname)
            _assign_param(parent, attr, tensor)
            consumed.add(qualname)
            stats["other"] += 1

        missing = []
        for qualname, param in model.named_parameters():
            if param.is_meta or param.device.type == "meta":
                missing.append(qualname)
        if missing:
            sample = ", ".join(missing[:8])
            raise RuntimeError(f"{len(missing)} parameters still on the meta device ({sample})")

        unused = sorted(keys - consumed)
        stats["unused"] = len(unused)

    gc.collect()
    return stats
