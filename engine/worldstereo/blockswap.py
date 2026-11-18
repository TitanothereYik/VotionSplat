"""Keep parts of the WorldStereo DiT in system RAM between uses.

The DiT runs its 40 blocks one after another, so only the running block has
to be on the card. ``park_modules`` keeps one copy of each module's tensors in
system RAM, puts a module on the card just before it runs, and points it back
at the RAM copy right after. Weights never change, so nothing is copied back
and FP8 stays FP8.

``offload_states`` does the same for the controlnet's 20 per-layer outputs.
Each one is copied to system RAM as it is made and put back on the card when
the DiT adds it.

Nothing under ``vendor/`` is edited. Everything here is a hook.
"""

from __future__ import annotations

import gc

import torch
import torch.nn as nn


class ParkedBlocks:
    def __init__(self) -> None:
        self.parked = 0
        self.resident = 0
        self.bytes = 0
        self.pinned = False
        self.states = False
        self.loads = 0
        self.unloads = 0
        self.handles: list = []
        self.slots: set = set()

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []


def _slots(block: nn.Module) -> list[tuple[nn.Module, str, bool]]:
    """Every tensor a module owns as (owner module, name, is_parameter)."""
    found = []
    for module in block.modules():
        for name, param in module._parameters.items():
            if param is not None:
                found.append((module, name, True))
        for name, buf in module._buffers.items():
            if buf is not None:
                found.append((module, name, False))
    return found


def _point(module: nn.Module, name: str, is_param: bool, tensor: torch.Tensor) -> None:
    if is_param:
        module._parameters[name].data = tensor
    else:
        module._buffers[name] = tensor


def tensor_bytes(module: nn.Module) -> int:
    total = 0
    for owner, name, is_param in _slots(module):
        tensor = owner._parameters[name] if is_param else owner._buffers[name]
        total += tensor.numel() * tensor.element_size()
    return total


def park_modules(modules, device, state: ParkedBlocks | None = None, pin: bool | None = None) -> ParkedBlocks:
    """Park every module in ``modules``. Pass ``state`` to add to an earlier call."""
    device = torch.device(device)
    if state is None:
        state = ParkedBlocks()
        state.pinned = device.type == "cuda" and (pin is None or bool(pin))
    for block in modules:
        entries = []
        for module, name, is_param in _slots(block):
            if (id(module), name) in state.slots:
                continue
            current = module._parameters[name] if is_param else module._buffers[name]
            master = current.detach().to("cpu")
            if state.pinned:
                master = master.pin_memory()
            _point(module, name, is_param, master)
            entries.append((module, name, is_param, master))
            state.slots.add((id(module), name))
            state.bytes += master.numel() * master.element_size()
        if not entries:
            continue

        def load(_module, _args, entries=entries):
            for module, name, is_param, master in entries:
                _point(module, name, is_param, master.to(device, non_blocking=True))
            state.loads += 1

        def unload(_module, _args, _output, entries=entries):
            for module, name, is_param, master in entries:
                _point(module, name, is_param, master)
            state.unloads += 1

        state.handles.append(block.register_forward_pre_hook(load))
        state.handles.append(block.register_forward_hook(unload, always_call=True))
        state.parked += 1
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return state


def park_blocks(model: nn.Module, device: torch.device, resident: int = 0) -> ParkedBlocks:
    """Park ``model.blocks[resident:]`` in system RAM. ``resident=0`` parks every block."""
    blocks = getattr(model, "blocks", None)
    if not isinstance(blocks, nn.ModuleList) or len(blocks) == 0:
        raise RuntimeError("model has no blocks list to park")
    resident = max(0, min(int(resident), len(blocks)))
    state = park_modules(list(blocks)[resident:], device)
    state.resident = resident
    return state


def place_rest(model: nn.Module, device: torch.device, state: ParkedBlocks | None = None) -> int:
    """Put every tensor that is not parked on ``device``. Returns the bytes placed."""
    device = torch.device(device)
    parked = set() if state is None else state.slots
    placed = 0
    for module in model.modules():
        for name, param in module._parameters.items():
            if param is None or (id(module), name) in parked:
                continue
            param.data = param.data.to(device)
            placed += param.numel() * param.element_size()
        for name, buf in list(module._buffers.items()):
            if buf is None or (id(module), name) in parked:
                continue
            module._buffers[name] = buf.to(device)
            placed += buf.numel() * buf.element_size()
    return placed


class _RamStates(list):
    """The controlnet outputs, held in system RAM. Reading one puts it on the card."""

    def __init__(self, items, device: torch.device) -> None:
        super().__init__(items)
        self._device = device

    def __getitem__(self, index):
        item = super().__getitem__(index)
        if isinstance(item, torch.Tensor):
            return item.to(self._device, non_blocking=True)
        return item


def offload_states(model: nn.Module, device: torch.device, state: ParkedBlocks) -> None:
    """Keep the controlnet's per-layer outputs in system RAM until the DiT adds them."""
    if state.states:
        return
    controlnet = getattr(model, "controlnet", None)
    outs = getattr(controlnet, "proj_out", None)
    if controlnet is None or not isinstance(outs, nn.ModuleList) or len(outs) == 0:
        raise RuntimeError("model has no controlnet outputs to offload")
    device = torch.device(device)
    buffers: dict = {}

    def keep(index: int):
        def hook(_module, _args, output):
            buf = buffers.get(index)
            if buf is None or buf.shape != output.shape or buf.dtype != output.dtype:
                buf = torch.empty(output.shape, dtype=output.dtype, pin_memory=state.pinned)
                buffers[index] = buf
            buf.copy_(output)
            return buf

        return hook

    def wrap(_module, _args, output):
        return _RamStates(output, device)

    for index, layer in enumerate(outs):
        state.handles.append(layer.register_forward_hook(keep(index)))
    state.handles.append(controlnet.register_forward_hook(wrap))
    state.states = True
