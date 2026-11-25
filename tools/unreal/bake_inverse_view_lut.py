"""Bake a 3D LUT from display sRGB back to linear Rec.709 through the inverse ACES SDR view.

The shader samples this LUT, then applies the Rec.709-to-working matrix. Output:
    unreal/VotionSplatHost/Plugins/VotionSplat/Resources/VotionSplat_InverseView.bin
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "unreal" / "VotionSplatHost" / "Plugins" / "VotionSplat" / "Resources" / "VotionSplat_InverseView.bin"


def _transform(ocio, config, display: str, view: str, direction):
    transform = ocio.DisplayViewTransform()
    transform.setSrc("Linear Rec.709 (sRGB)")
    transform.setDisplay(display)
    transform.setView(view)
    transform.setDirection(direction)
    return config.getProcessor(transform).getDefaultCPUProcessor()


def _sample(lut: np.ndarray, display: np.ndarray) -> np.ndarray:
    n = lut.shape[0]
    pos = np.clip(display, 0.0, 1.0) * (n - 1)
    base = np.floor(pos).astype(np.int32)
    frac = (pos - base).astype(np.float32)
    nxt = np.minimum(base + 1, n - 1)
    out = np.zeros_like(display, dtype=np.float32)
    for b0 in (0, 1):
        for g0 in (0, 1):
            for r0 in (0, 1):
                weight = ((frac[:, 2] if b0 else 1 - frac[:, 2])
                          * (frac[:, 1] if g0 else 1 - frac[:, 1])
                          * (frac[:, 0] if r0 else 1 - frac[:, 0]))
                bi = nxt[:, 2] if b0 else base[:, 2]
                gi = nxt[:, 1] if g0 else base[:, 1]
                ri = nxt[:, 0] if r0 else base[:, 0]
                out += lut[bi, gi, ri] * weight[:, None]
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "docs" / "unreal" / "VS_OCIO.ocio")
    parser.add_argument("--display", default="sRGB - Display")
    parser.add_argument("--view", default="ACES 1.0 - SDR Video")
    parser.add_argument("--size", type=int, default=65)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    import PyOpenColorIO as ocio

    config = ocio.Config.CreateFromFile(str(args.config))
    inverse = _transform(ocio, config, args.display, args.view, ocio.TRANSFORM_DIR_INVERSE)
    forward = _transform(ocio, config, args.display, args.view, ocio.TRANSFORM_DIR_FORWARD)
    n = args.size
    grid = np.linspace(0.0, 1.0, n, dtype=np.float32)
    blue, green, red = np.meshgrid(grid, grid, grid, indexing="ij")
    rgb = np.ascontiguousarray(np.stack([red, green, blue], axis=-1).reshape(-1, 3))
    before = rgb.copy()
    inverse.applyRGB(rgb)
    if np.allclose(rgb, before):
        raise RuntimeError("inverse DisplayView left the samples unchanged")
    lut = rgb.reshape(n, n, n, 3)
    rgba = np.empty((n, n, n, 4), dtype=np.float32)
    rgba[..., :3] = lut
    rgba[..., 3] = 1.0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(b"VSLUT001" + struct.pack("<I", n) + np.ascontiguousarray(rgba).tobytes())

    rng = np.random.default_rng(0)
    samples = np.concatenate([
        rng.random((200_000, 3), dtype=np.float32),
        np.linspace(0, 1, 256, dtype=np.float32)[:, None].repeat(3, axis=1),
    ])
    exact = samples.copy()
    inverse.applyRGB(exact)
    forward.applyRGB(exact)
    recovered = _sample(lut, samples)
    forward.applyRGB(recovered)
    usable = np.max(np.abs(exact - samples), axis=1) <= 1e-3
    err = np.max(np.abs(recovered[usable] - samples[usable]), axis=1) * 255.0
    within = float((err <= 0.5).mean()) if err.size else 1.0
    print(f"wrote {args.out} N={n} usable={int(usable.sum())} within_0.5={within:.6f} "
          f"mean_levels={float(err.mean()) if err.size else 0:.4f} max_levels={float(err.max()) if err.size else 0:.4f}")
    if within < 0.9999 and n < 129:
        print("round trip is outside 0.5 level on more than 0.01% of samples; rerun with --size 129")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
