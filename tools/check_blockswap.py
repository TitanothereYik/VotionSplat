"""CPU check for block parking. No GPU, no weights."""

from __future__ import annotations

import torch
import torch.nn as nn

from engine.worldstereo.blockswap import park_blocks
from engine.worldstereo.fp8 import ScaledFP8Linear


def check(cond: bool, msg: str) -> None:
    if not cond:
        raise SystemExit(msg)


class Block(nn.Module):
    def __init__(self, dim: int, fail: bool = False) -> None:
        super().__init__()
        weight = torch.randn(dim, dim).to(torch.float8_e4m3fn)
        self.fp8 = ScaledFP8Linear(weight, torch.tensor(0.5), torch.randn(dim).to(torch.bfloat16))
        self.plain = nn.Linear(dim, dim)
        self.table = nn.Parameter(torch.randn(1, dim))
        self.register_buffer("offset", torch.randn(dim))
        self.fail = fail

    def forward(self, x, ref):
        if self.fail:
            raise ValueError("planned failure")
        return self.plain(self.fp8(x)) + self.table + self.offset, ref + 1


class Model(nn.Module):
    def __init__(self, dim: int = 8, count: int = 4, fail_at: int = -1) -> None:
        super().__init__()
        self.head = nn.Linear(dim, dim)
        self.blocks = nn.ModuleList([Block(dim, fail=index == fail_at) for index in range(count)])

    def forward(self, x):
        x = self.head(x)
        ref = torch.zeros(1)
        for block in self.blocks:
            x, ref = block(x, ref)
        return x, ref


def main() -> int:
    torch.manual_seed(0)
    model = Model().eval()
    x = torch.randn(2, 8)
    with torch.no_grad():
        before, ref_before = model(x)
    names = [name for name, _ in model.named_parameters()]
    ids = [id(param) for param in model.parameters()]

    state = park_blocks(model, torch.device("cpu"), resident=1)
    check(state.parked == 3 and state.resident == 1, f"parked {state.parked} resident {state.resident}")
    check(state.bytes > 0, "no bytes counted")
    check(not state.pinned, "a cpu run must not pin")
    with torch.no_grad():
        after, ref_after = model(x)
        again, _ = model(x)
    check(torch.equal(before, after) and torch.equal(before, again), "output changed after parking")
    check(torch.equal(ref_before, ref_after), "second output changed")
    check(state.loads == 6 and state.unloads == 6, f"loads {state.loads} unloads {state.unloads}")
    check(names == [name for name, _ in model.named_parameters()], "parameter names changed")
    check(ids == [id(param) for param in model.parameters()], "parameter objects changed")
    check(model.blocks[1].fp8.weight.dtype == torch.uint8, "fp8 bits are no longer uint8")

    state.remove()
    with torch.no_grad():
        model(x)
    check(state.loads == 6, "hooks still fire after remove")

    broken = Model(fail_at=2).eval()
    state = park_blocks(broken, torch.device("cpu"))
    try:
        with torch.no_grad():
            broken(x)
    except ValueError:
        pass
    else:
        raise SystemExit("planned failure did not raise")
    check(state.loads == 3 and state.unloads == 3, f"after failure loads {state.loads} unloads {state.unloads}")

    try:
        park_blocks(nn.Linear(2, 2), torch.device("cpu"))
    except RuntimeError:
        pass
    else:
        raise SystemExit("a model without blocks was accepted")
    print("blockswap ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
