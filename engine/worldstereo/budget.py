"""Fit the WorldStereo DiT to the card and the RAM this machine has.

Four levels, lightest first. Each one keeps less on the card:

    card        everything stays on the card
    blocks      the 40 DiT blocks wait in system RAM
    states      blocks, and the controlnet's 20 outputs wait in system RAM
    controlnet  states, and the 20 controlnet blocks wait in system RAM

``Fit`` reads the card and the RAM, caps what PyTorch may take so a misfit
fails at once instead of spilling into system memory, picks the lightest level
that should fit, and steps down one level when a sample still runs out. What
it measures is saved in ``models/expand_fit.json`` and used the next time.
"""

from __future__ import annotations

import gc
import json
from pathlib import Path

import torch

from engine.job import REPO_ROOT
from engine.worldstereo.blockswap import offload_states, park_modules, place_rest, tensor_bytes

GB = 1024**3
LEVELS = ("card", "blocks", "states", "controlnet")
# Card memory one sample needs on top of the weights that stay on the card, in GB,
# measured on the RTX 3090 at 832x480, 21 frames. The cap limits reserved memory, and
# parking fragments the cache, so the parked levels use reserved, not allocated, peaks:
# "blocks" 18.29 reserved with 1.97 on the card, "states" 11.79 with 1.97,
# "controlnet" 10.38 with 0.56. "card" is the allocated peak, 31.88 with 19.24.
WORK_GB = {"card": 12.7, "blocks": 16.4, "states": 9.9, "controlnet": 9.9}
# Lowest budget that passed a capped run (11 GB passed, 9 and 7 GB did not).
MIN_BUDGET_GB = 11.0
HEADROOM_GB = 1.0  # kept free above the predicted need
RESERVE_GB = 0.75  # card memory left for other programs
RAM_SPARE_GB = 4.0  # system RAM that must stay free
PIN_SPARE_GB = 8.0  # page-lock the RAM copy only when this much RAM stays free
FIT_FILE = REPO_ROOT / "models" / "expand_fit.json"


def is_oom(exc: BaseException) -> bool:
    if isinstance(exc, torch.OutOfMemoryError):
        return True
    accel = getattr(torch, "AcceleratorError", None)
    return accel is not None and isinstance(exc, accel) and "out of memory" in str(exc).lower()


def read_rows(key: str) -> dict:
    try:
        data = json.loads(FIT_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    rows = data.get(key) if isinstance(data, dict) else None
    return rows if isinstance(rows, dict) else {}


def write_rows(key: str, rows: dict) -> None:
    try:
        data = json.loads(FIT_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        data = {}
    if not isinstance(data, dict):
        data = {}
    data[key] = rows
    FIT_FILE.parent.mkdir(parents=True, exist_ok=True)
    FIT_FILE.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def fits(level: str, on_card_gb: float, budget_gb: float, rows: dict) -> bool:
    """True when ``level`` should fit in ``budget_gb``. Saved results beat the estimate."""
    row = rows.get(level) or {}
    failed = row.get("fail_gb")
    if failed is not None and budget_gb <= float(failed) + 0.5:
        return False
    peak = row.get("peak_gb")
    if peak is not None:
        need = max(float(peak) + HEADROOM_GB, float(row.get("reserved_gb") or 0.0))
        return need <= budget_gb
    return on_card_gb + WORK_GB[level] + HEADROOM_GB <= budget_gb


def pick(levels, on_card: dict, budget_gb: float, rows: dict) -> str:
    """The first level that should fit. The last one when none does; the cap then decides."""
    for level in levels:
        if fits(level, on_card[level], budget_gb, rows):
            return level
    return levels[-1]


def ram_free_gb() -> float:
    import psutil

    return psutil.virtual_memory().available / GB


def ram_used_gb() -> float:
    import psutil

    return psutil.Process().memory_info().rss / GB


def machine(device: torch.device) -> dict:
    import psutil

    torch.cuda.empty_cache()
    free, total = torch.cuda.mem_get_info(device)
    ram = psutil.virtual_memory()
    return {
        "gpu": torch.cuda.get_device_name(device),
        "vram_total_gb": total / GB,
        "vram_usable_gb": (free + torch.cuda.memory_reserved(device)) / GB,
        "ram_total_gb": ram.total / GB,
        "ram_free_gb": ram.available / GB,
    }


class Stages:
    """Card memory at three points of one sample. It only reads; nothing is reset."""

    def __init__(self, model, device: torch.device) -> None:
        self.device = device
        self.pre = 0.0
        self.dit = 0.0
        self.states = 0.0
        self._seen = False
        self._before = 0
        model.register_forward_pre_hook(self._dit_in)
        model.register_forward_hook(self._dit_out)
        controlnet = getattr(model, "controlnet", None)
        if controlnet is not None:
            controlnet.register_forward_pre_hook(self._cn_in)
            controlnet.register_forward_hook(self._cn_out)

    def start(self) -> None:
        self.pre = self.dit = self.states = 0.0
        self._seen = False

    def _dit_in(self, _module, _args) -> None:
        if not self._seen:
            self._seen = True
            self.pre = torch.cuda.max_memory_allocated(self.device) / GB

    def _dit_out(self, _module, _args, _output) -> None:
        self.dit = torch.cuda.max_memory_allocated(self.device) / GB

    def _cn_in(self, _module, _args) -> None:
        self._before = torch.cuda.memory_allocated(self.device)

    def _cn_out(self, _module, _args, _output) -> None:
        grown = (torch.cuda.memory_allocated(self.device) - self._before) / GB
        self.states = max(self.states, grown)


class Fit:
    def __init__(self, device, level: str = "auto", cap_gb: float = 0.0, log=print) -> None:
        if level != "auto" and level not in LEVELS:
            raise ValueError(f"unknown level {level}")
        self.device = torch.device(device)
        self.want = level
        self.cap_gb = float(cap_gb)
        self.log = log
        self.level = ""
        self.state = None
        self.stages = None
        self.budget_gb = 0.0
        self.on_card_gb = 0.0
        self.key = ""
        self.rows: dict = {}

    def _on_card(self) -> float:
        if self.device.type != "cuda":
            return 0.0
        return torch.cuda.memory_allocated(self.device) / GB

    def _pin(self, extra_gb: float) -> bool:
        return self.device.type == "cuda" and ram_free_gb() >= extra_gb + PIN_SPARE_GB

    def _sizes(self, model) -> dict:
        total = tensor_bytes(model)
        blocks = sum(tensor_bytes(block) for block in model.blocks)
        control = sum(tensor_bytes(block) for block in model.controlnet.controlnet_blocks)
        rest = (total - blocks) / GB
        return {"card": total / GB, "blocks": rest, "states": rest, "controlnet": rest - control / GB}

    def _fail(self, level: str) -> None:
        row = dict(self.rows.get(level) or {})
        row["fail_gb"] = round(max(float(row.get("fail_gb") or 0.0), self.budget_gb), 2)
        self.rows[level] = row
        if self.key:
            write_rows(self.key, self.rows)

    def _from_ram(self, model, level: str) -> None:
        """The model is in system RAM. Park what the level parks and put the rest on the card."""
        self.state = park_modules(list(model.blocks), self.device, pin=self._pin(0.0))
        if level == "controlnet":
            park_modules(list(model.controlnet.controlnet_blocks), self.device, self.state)
        failed = False
        try:
            place_rest(model, self.device, self.state)
        except Exception as exc:
            if not is_oom(exc):
                raise
            failed = True
        if failed:
            raise RuntimeError("VRAM // the parts of the DiT that must stay on the card do not fit")
        if level in ("states", "controlnet"):
            offload_states(model, self.device, self.state)

    def build(self, build_dit, transformer_cfg, config_path, weights):
        """Read the machine, load the DiT at the chosen level. Returns (model, ws_cfg)."""
        info = machine(self.device)
        total = info["vram_total_gb"]
        usable = info["vram_usable_gb"] - RESERVE_GB
        self.budget_gb = min(self.cap_gb, usable) if self.cap_gb > 0 else usable
        torch.cuda.set_per_process_memory_fraction(max(0.01, min(1.0, self.budget_gb / total)), self.device)
        self.key = f"{info['gpu']}|{total:.0f}"
        self.rows = read_rows(self.key)
        file_gb = Path(weights).stat().st_size / GB
        early = torch.cuda.max_memory_allocated(self.device) / GB
        self.log(
            f"STATUS // fit gpu={info['gpu']} vram_gb={total:.1f} budget_gb={self.budget_gb:.2f} "
            f"ram_free_gb={info['ram_free_gb']:.1f} ram_total_gb={info['ram_total_gb']:.1f} "
            f"early_peak_gb={early:.2f} asked={self.want}"
        )
        if self.budget_gb < MIN_BUDGET_GB:
            self.log(
                f"STATUS // fit budget_gb={self.budget_gb:.2f} is below the tested floor of "
                f"{MIN_BUDGET_GB:g} GB; the sample will likely run out of card memory"
            )
        start = self.want
        if start == "auto":
            start = "card" if fits("card", file_gb, self.budget_gb, self.rows) else "blocks"
        model = ws_cfg = None
        if start == "card":
            failed = False
            try:
                model, ws_cfg, _stats = build_dit(transformer_cfg, config_path, weights, self.device)
            except Exception as exc:
                if not is_oom(exc):
                    raise
                failed = True
            if failed:
                model = ws_cfg = None
                gc.collect()
                torch.cuda.empty_cache()
                self._fail("card")
                if self.want == "card":
                    raise RuntimeError("VRAM // the whole DiT does not fit on this card (level=card)")
                start = "blocks"
        if model is None:
            free = ram_free_gb()
            if free < file_gb + RAM_SPARE_GB:
                raise RuntimeError(
                    f"RAM // {free:.1f} GB of system RAM is free; keeping the DiT in RAM needs about "
                    f"{file_gb + RAM_SPARE_GB:.0f} GB"
                )
            model, ws_cfg, _stats = build_dit(
                transformer_cfg, config_path, weights, self.device, load_device="cpu"
            )
            if self.want == "auto":
                start = pick(LEVELS[1:], self._sizes(model), self.budget_gb, self.rows)
            self._from_ram(model, start)
        self.level = start
        self.on_card_gb = self._on_card()
        self.stages = Stages(model, self.device)
        parked = 0.0 if self.state is None else self.state.bytes / GB
        pinned = False if self.state is None else self.state.pinned
        self.log(
            f"STATUS // fit level={self.level} on_card_gb={self.on_card_gb:.2f} "
            f"parked_gb={parked:.2f} pinned={pinned}"
        )
        return model, ws_cfg

    def step_down(self, model) -> bool:
        """The sample ran out of card memory. Go one level down. False when there is none."""
        self._fail(self.level)
        index = LEVELS.index(self.level)
        if self.want != "auto" or index + 1 >= len(LEVELS):
            return False
        nxt = LEVELS[index + 1]
        if nxt == "blocks":
            need = sum(tensor_bytes(block) for block in model.blocks) / GB
            if ram_free_gb() < need + RAM_SPARE_GB:
                return False
            self.state = park_modules(list(model.blocks), self.device, pin=self._pin(need))
        elif nxt == "states":
            offload_states(model, self.device, self.state)
        else:
            park_modules(list(model.controlnet.controlnet_blocks), self.device, self.state)
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        self.level = nxt
        self.on_card_gb = self._on_card()
        self.log(f"STATUS // fit stepped down to level={self.level} on_card_gb={self.on_card_gb:.2f}")
        return True

    def start(self) -> None:
        if self.stages is not None:
            self.stages.start()

    def record(self, seconds: float, peak_gb: float, reserved_gb: float = 0.0) -> None:
        row = dict(self.rows.get(self.level) or {})
        row["peak_gb"] = round(max(float(row.get("peak_gb") or 0.0), float(peak_gb)), 2)
        if reserved_gb > 0:
            row["reserved_gb"] = round(max(float(row.get("reserved_gb") or 0.0), float(reserved_gb)), 2)
        row["work_gb"] = round(max(float(row.get("work_gb") or 0.0), float(peak_gb) - self.on_card_gb), 2)
        row["seconds"] = round(float(seconds), 1)
        self.rows[self.level] = row
        if self.key:
            write_rows(self.key, self.rows)

    def summary(self) -> dict:
        parked = 0.0 if self.state is None else self.state.bytes / GB
        return {
            "level": self.level,
            "asked": self.want,
            "budget_gb": round(self.budget_gb, 2),
            "on_card_gb": round(self.on_card_gb, 2),
            "parked_gb": round(parked, 2),
            "pinned": bool(self.state is not None and self.state.pinned),
            "key": self.key,
            "floor_gb": MIN_BUDGET_GB,
        }

    def line(self) -> str:
        pre = dit = states = 0.0
        if self.stages is not None:
            pre, dit, states = self.stages.pre, self.stages.dit, self.stages.states
        return (
            f"level={self.level} on_card_gb={self.on_card_gb:.2f} pre_gb={pre:.2f} dit_gb={dit:.2f} "
            f"states_gb={states:.2f} ram_gb={ram_used_gb():.1f}"
        )
