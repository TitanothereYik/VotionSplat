"""Compare a reference frame with an Unreal capture. Exit 0 on pass, 1 on fail."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

THRESHOLDS = {
    "synthetic": {"mean": 0.35, "p999": 1.5, "max": 3.0},
    "gsplat": {"mean": 0.6, "p999": 3.0, "frac8": 1e-4},
    "classic": {"mean": 1.5, "p99": 4.0},
    "ocio": {"mean": 0.8, "p999": 4.0},
}


def _load(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        image = np.load(path)
        if image.dtype != np.uint8:
            image = np.clip(np.round(image * 255.0), 0, 255)
        return image[..., :3].astype(np.float32)
    image = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
    return image


def _shift_error(error: np.ndarray) -> tuple[int, int, float]:
    best = (0, 0, float(error.mean()))
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            shifted = error
            if dx or dy:
                shifted = np.roll(np.roll(error, dy, axis=0), dx, axis=1)
            value = float(shifted.mean())
            if value < best[2]:
                best = (dx, dy, value)
    return best


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--profile", required=True, choices=tuple(THRESHOLDS))
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--heatmap", type=Path, required=True)
    parser.add_argument("--crop", default="")
    args = parser.parse_args()
    ref, test = _load(args.ref), _load(args.test)
    report: dict = {"ref": str(args.ref), "test": str(args.test), "profile": args.profile}
    if ref.shape != test.shape:
        report["pass"] = False
        report["error"] = f"shape {tuple(test.shape)} != {tuple(ref.shape)}"
        _finish(args, report, None)
        return 1
    if args.crop:
        x, y, w, h = (int(v) for v in args.crop.split(","))
        ref, test = ref[y:y + h, x:x + w], test[y:y + h, x:x + w]
    error = np.max(np.abs(ref - test), axis=2)
    dx, dy, _shifted = _shift_error(error)
    flat = np.sort(error.reshape(-1))
    report.update({
        "mean": float(error.mean()),
        "p99": float(flat[int(0.99 * (flat.size - 1))]),
        "p99.9": float(flat[int(0.999 * (flat.size - 1))]),
        "max": float(error.max()),
        "frac_gt2": float((error > 2).mean()),
        "frac_gt8": float((error > 8).mean()),
        "best_shift": [dx, dy],
    })
    limits = THRESHOLDS[args.profile]
    failed = []
    if (dx, dy) != (0, 0):
        failed.append(f"misaligned by ({dx},{dy})")
    if "mean" in limits and report["mean"] > limits["mean"]:
        failed.append(f"mean {report['mean']:.3f} > {limits['mean']}")
    if "p99" in limits and report["p99"] > limits["p99"]:
        failed.append(f"p99 {report['p99']:.3f} > {limits['p99']}")
    if "p999" in limits and report["p99.9"] > limits["p999"]:
        failed.append(f"p99.9 {report['p99.9']:.3f} > {limits['p999']}")
    if "max" in limits and report["max"] > limits["max"]:
        failed.append(f"max {report['max']:.3f} > {limits['max']}")
    if "frac8" in limits and report["frac_gt8"] > limits["frac8"]:
        failed.append(f"frac>8 {report['frac_gt8']:.3e} > {limits['frac8']}")
    report["pass"] = not failed
    report["failures"] = failed
    _finish(args, report, error)
    print(json.dumps({k: report[k] for k in ("pass", "mean", "p99", "p99.9", "max", "failures")}))
    return 0 if report["pass"] else 1


def _finish(args, report: dict, error: np.ndarray | None) -> None:
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if error is not None:
        args.heatmap.parent.mkdir(parents=True, exist_ok=True)
        heat = np.clip(error * 16.0, 0, 255).astype(np.uint8)
        Image.fromarray(heat, mode="L").save(args.heatmap)


if __name__ == "__main__":
    raise SystemExit(main())
