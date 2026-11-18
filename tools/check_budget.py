"""CPU check for the memory fit. ``--gpu`` adds short checks on the card."""

from __future__ import annotations

import copy
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn as nn

from engine.worldstereo import budget
from engine.worldstereo.blockswap import park_modules, place_rest, tensor_bytes
from engine.worldstereo.budget import GB, Fit, fits, is_oom, pick
from engine.worldstereo.fp8 import ScaledFP8Linear


def check(cond: bool, msg: str) -> None:
    if not cond:
        raise SystemExit(msg)


class Block(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        weight = torch.randn(dim, dim).to(torch.float8_e4m3fn)
        self.fp8 = ScaledFP8Linear(weight, torch.tensor(0.5), torch.randn(dim).to(torch.bfloat16))
        self.plain = nn.Linear(dim, dim)
        self.table = nn.Parameter(torch.randn(1, dim))
        self.register_buffer("offset", torch.randn(dim))

    def forward(self, x, ref):
        return self.plain(self.fp8(x)) + self.table + self.offset, ref + 1


class Control(nn.Module):
    def __init__(self, dim: int, count: int) -> None:
        super().__init__()
        self.controlnet_patch_embedding = nn.Linear(dim, dim)
        self.controlnet_blocks = nn.ModuleList([nn.Linear(dim, dim) for _ in range(count)])
        self.proj_out = nn.ModuleList([nn.Linear(dim, dim) for _ in range(count)])

    def forward(self, hidden):
        states = []
        for index, block in enumerate(self.controlnet_blocks):
            hidden = block(hidden)
            states.append(self.proj_out[index](hidden))
        return states


class Model(nn.Module):
    def __init__(self, dim: int = 8, count: int = 4) -> None:
        super().__init__()
        self.head = nn.Linear(dim, dim)
        self.blocks = nn.ModuleList([Block(dim) for _ in range(count)])
        self.controlnet = Control(dim, 2)
        self.table = nn.Parameter(torch.randn(1, dim))
        self.register_buffer("freq", torch.randn(dim))

    def forward(self, x):
        states = self.controlnet(hidden=self.controlnet.controlnet_patch_embedding(x))
        x = self.head(x) + self.table + self.freq
        ref = torch.zeros(1, device=x.device)
        for index, block in enumerate(self.blocks):
            x, ref = block(x, ref)
            if index < len(states):
                x += states[index]
        return x, ref


def run(model, x):
    with torch.no_grad():
        return model(x)[0]


def same(a, b) -> bool:
    return torch.allclose(a.float(), b.float(), atol=1e-6)


def ladder(device: torch.device) -> None:
    torch.manual_seed(0)
    base = Model().eval().to(device)
    x = torch.randn(2, 8, device=device)
    want = run(base, x)
    model = copy.deepcopy(base)
    fit = Fit(device, "auto", log=lambda _line: None)
    fit.level = "card"
    fit.budget_gb = 10.0
    fit.key = "check|1"
    for level, parked in (("blocks", 4), ("states", 4), ("controlnet", 6)):
        check(fit.step_down(model), f"no step down to {level}")
        check(fit.level == level, f"level is {fit.level}, expected {level}")
        check(fit.state.parked == parked, f"{level} parked {fit.state.parked}")
        check(same(want, run(model, x)) and same(want, run(model, x)), f"output changed at {level}")
    check(fit.state.states, "the controlnet outputs are not offloaded")
    check(fit.state.loads == fit.state.unloads and fit.state.loads > 0, "load and unload do not match")
    check(model.blocks[0].plain.weight.device.type == "cpu", "a parked block is not in RAM")
    check(not fit.step_down(model), "stepped past the last level")
    rows = budget.read_rows("check|1")
    check(all(rows[level]["fail_gb"] == 10.0 for level in budget.LEVELS), f"failures not saved {rows}")


def from_ram(device: torch.device) -> None:
    torch.manual_seed(1)
    base = Model().eval()
    x = torch.randn(2, 8)
    want = run(copy.deepcopy(base).to(device), x.to(device))
    for level, parked in (("blocks", 4), ("states", 4), ("controlnet", 6)):
        model = copy.deepcopy(base)
        fit = Fit(device, level, log=lambda _line: None)
        fit._from_ram(model, level)
        check(fit.state.parked == parked, f"{level} parked {fit.state.parked} from RAM")
        check(same(want, run(model, x.to(device))), f"output changed at {level} from RAM")
        check(model.head.weight.device.type == device.type, "the rest was not placed")
        check(model.blocks[0].plain.weight.device.type == "cpu", "a parked block is not in RAM")


def sizes() -> None:
    model = Model().eval()
    cpu = torch.device("cpu")
    state = park_modules(list(model.blocks), cpu)
    check(state.bytes == sum(tensor_bytes(block) for block in model.blocks), "parked bytes are wrong")
    check(place_rest(model, cpu, state) == tensor_bytes(model) - state.bytes, "placed bytes are wrong")


def choose() -> None:
    budget.WORK_GB = {"card": 12.7, "blocks": 13.1, "states": 6.7, "controlnet": 6.7}
    budget.HEADROOM_GB = 1.0
    on_card = {"card": 19.0, "blocks": 3.2, "states": 3.2, "controlnet": 2.5}
    parked = budget.LEVELS[1:]
    check(pick(parked, on_card, 22.7, {}) == "blocks", "22.7 GB should pick blocks")
    check(pick(parked, on_card, 15.0, {}) == "states", "15 GB should pick states")
    check(pick(parked, on_card, 9.0, {}) == "controlnet", "9 GB should fall to the last level")
    check(not fits("card", 19.0, 22.7, {}), "22.7 GB should not fit level card")
    check(fits("card", 19.0, 40.0, {}), "40 GB should fit level card")
    rows = {"blocks": {"fail_gb": 22.7}}
    check(pick(parked, on_card, 22.7, rows) == "states", "a saved failure was ignored")
    check(pick(parked, on_card, 30.0, rows) == "blocks", "a failure at 22.7 GB blocked 30 GB")
    rows = {"blocks": {"peak_gb": 16.0}}
    check(pick(parked, on_card, 16.5, rows) == "states", "a saved peak was ignored")
    check(pick(parked, on_card, 17.0, rows) == "blocks", "a saved peak of 16 GB blocked 17 GB")
    rows = {"blocks": {"peak_gb": 16.0, "reserved_gb": 18.3}}
    check(pick(parked, on_card, 17.0, rows) == "states", "a saved reserved peak was ignored")
    check(pick(parked, on_card, 18.5, rows) == "blocks", "a reserved peak of 18.3 GB blocked 18.5 GB")


def records() -> None:
    fit = Fit(torch.device("cpu"), "blocks", log=lambda _line: None)
    fit.level = "blocks"
    fit.key = "check|2"
    fit.budget_gb = 20.0
    fit.on_card_gb = 3.0
    fit.record(100.0, 16.0, 18.0)
    fit.record(90.0, 15.0, 17.0)
    row = budget.read_rows("check|2")["blocks"]
    check(row["peak_gb"] == 16.0 and row["work_gb"] == 13.0 and row["seconds"] == 90.0, f"saved row {row}")
    check(row["reserved_gb"] == 18.0, f"saved reserved {row}")
    check(not fit.step_down(nn.Linear(2, 2)), "a fixed level stepped down")
    check(budget.read_rows("check|2")["blocks"]["fail_gb"] == 20.0, "the failure was not saved")
    check("level=blocks" in fit.line(), "the log text has no level")
    try:
        Fit(torch.device("cpu"), "bad")
    except ValueError:
        pass
    else:
        raise SystemExit("an unknown level was accepted")


def cap(device: torch.device) -> None:
    total = torch.cuda.get_device_properties(device).total_memory
    torch.cuda.empty_cache()
    torch.cuda.set_per_process_memory_fraction(2 * GB / total, device)
    small = torch.empty(GB, dtype=torch.uint8, device=device)
    refused = False
    try:
        torch.empty(2 * GB, dtype=torch.uint8, device=device)
    except Exception as exc:
        check(is_oom(exc), f"the cap raised {type(exc).__name__}, not out of memory")
        refused = True
    check(refused, "a 2 GB cap did not stop 3 GB")
    del small
    torch.cuda.empty_cache()
    torch.cuda.set_per_process_memory_fraction(1.0, device)


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    with tempfile.TemporaryDirectory() as folder:
        budget.FIT_FILE = Path(folder) / "expand_fit.json"
        cpu = torch.device("cpu")
        ladder(cpu)
        from_ram(cpu)
        sizes()
        choose()
        records()
        print("budget ok")
        if "--gpu" in args:
            device = torch.device("cuda:0")
            ladder(device)
            from_ram(device)
            cap(device)
            print("budget gpu ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
