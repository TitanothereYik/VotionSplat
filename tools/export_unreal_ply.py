"""CPU export of splat.ply from a LiteGS checkpoint (no CUDA)."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from engine.splat import defaults as D
from engine.splat.backend import litegs_env as LE
from engine.splat.ply_export import SH_C0, _sigmoid, load_splat_ply_arrays, write_splat_ply


def arrays_from_ckpt(ckpt: Path) -> dict:
    LE.add_litegs_paths()
    ck = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    params = {g["name"]: g["params"][0].detach().float().contiguous() for g in ck["optimizer"].param_groups}

    def flat(t: torch.Tensor) -> np.ndarray:
        return t.reshape(*t.shape[:-2], t.shape[-1] * t.shape[-2]).numpy()

    return {
        "xyz": flat(params["xyz"]),
        "scale": flat(params["scale"]),
        "rot": flat(params["rot"]),
        "sh_0": flat(params["sh_0"]),
        "sh_rest": flat(params["sh_rest"]),
        "opacity": flat(params["opacity"]),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--scene", default=r"D:\Votion3DGS\outputs\alpine_02")
    p.add_argument("--color-space", default=D.COLOR_SPACE_LUMA, choices=list(D.COLOR_SPACES))
    args = p.parse_args()
    scene = Path(args.scene)
    ckpt = scene / D.CKPT_REL
    out = scene / D.PLY_REL
    arrays = arrays_from_ckpt(ckpt)
    rgb = np.clip(arrays["sh_0"][0].T * SH_C0 + 0.5, 0, 1)
    print("N", arrays["xyz"].shape[-1], "ckpt rgb median", np.median(rgb, 0))
    path, n = write_splat_ply(out, arrays, color_space=args.color_space)
    a = load_splat_ply_arrays(path)
    rgb2 = np.clip(a["sh_0"][0].T * SH_C0 + 0.5, 0, 1)
    sig = _sigmoid(a["opacity"].reshape(-1))
    print("wrote", path, "n", n, "bytes", path.stat().st_size)
    print("ply rgb median", np.median(rgb2, 0), "f_rest", a["sh_rest"].shape)
    print("sigmoid p01 p50 p90", np.percentile(sig, [1, 50, 90]), "min", float(sig.min()))


if __name__ == "__main__":
    main()
