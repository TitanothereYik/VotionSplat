"""Find a photo's horizontal FOV: lens EXIF when present, else a MoGe v1 estimate. Qt never imports this."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

from engine.job import REPO_ROOT
from engine.pano.defaults import H_FOV_DEG_MAX, H_FOV_DEG_MIN, H_FOV_DEG_STEP


FULL_FRAME_DIAGONAL_MM = math.hypot(36.0, 24.0)
EXIF_IFD = 0x8769
TAG_FOCAL_MM = 37386
TAG_FOCAL_35MM = 41989
TAG_PLANE_X_RES = 41486
TAG_PLANE_UNIT = 41488
TAG_EXIF_WIDTH = 40962
TAG_EXIF_HEIGHT = 40963
PLANE_UNIT_MM = {2: 25.4, 3: 10.0, 4: 1.0}


def _snap(degrees: float) -> float:
    step = float(H_FOV_DEG_STEP)
    degrees = round(degrees / step) * step
    return min(float(H_FOV_DEG_MAX), max(float(H_FOV_DEG_MIN), degrees))


def exif_hfov(path: Path) -> tuple[float, str] | None:
    """Horizontal FOV across the raw pixel width, from the lens EXIF. None if absent or cropped."""
    from PIL import Image

    with Image.open(path) as image:
        width, height = image.size
        exif = image.getexif()
    if not exif or width <= 0 or height <= 0:
        return None
    sub = exif.get_ifd(EXIF_IFD)
    exif_w = float(sub.get(TAG_EXIF_WIDTH) or 0)
    exif_h = float(sub.get(TAG_EXIF_HEIGHT) or 0)
    if exif_w > 0 and exif_h > 0:
        same = abs(exif_w / exif_h - width / height) < 0.01
        swapped = abs(exif_h / exif_w - width / height) < 0.01
        if not (same or swapped):
            return None
    f35 = float(sub.get(TAG_FOCAL_35MM) or 0)
    if f35 > 0:
        sensor_w = FULL_FRAME_DIAGONAL_MM * width / math.hypot(width, height)
        return math.degrees(2.0 * math.atan(sensor_w / (2.0 * f35))), f"EXIF 35mm {f35:g}mm"
    focal = float(sub.get(TAG_FOCAL_MM) or 0)
    x_res = float(sub.get(TAG_PLANE_X_RES) or 0)
    unit = PLANE_UNIT_MM.get(int(sub.get(TAG_PLANE_UNIT) or 2))
    if focal > 0 and x_res > 0 and unit and exif_w > 0:
        sensor_long = max(exif_w, exif_h) / x_res * unit
        sensor_w = sensor_long * width / max(width, height)
        return math.degrees(2.0 * math.atan(sensor_w / (2.0 * focal))), f"EXIF {focal:g}mm"
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Detect horizontal FOV")
    parser.add_argument("--image", required=True)
    args = parser.parse_args(argv)
    path = Path(args.image)
    if not path.is_file():
        print(f"STATUS // missing photo {path}", flush=True)
        return 2
    try:
        from_exif = exif_hfov(path)
    except (OSError, ValueError, TypeError, ZeroDivisionError):
        from_exif = None
    if from_exif is not None:
        degrees, source = from_exif
        print(f"STATUS // h_fov from {source}", flush=True)
        print(f"HFOV\t{_snap(degrees):.1f}", flush=True)
        return 0
    print("STATUS // no usable lens EXIF; MoGe estimate, often 10 deg off on facades", flush=True)
    splatkit = REPO_ROOT / "vendor" / "splatkit"
    if str(splatkit) not in sys.path:
        sys.path.insert(0, str(splatkit))
    import numpy as np
    import torch
    from PIL import Image
    from core.matrix3d_pipeline import get_moge_model

    with Image.open(path) as image:
        image.load()
        photo = image.convert("RGB")
    width, height = photo.size
    long_side = max(width, height, 1)
    if long_side > 1024:
        scale = 1024.0 / float(long_side)
        photo = photo.resize((max(1, int(round(width * scale))), max(1, int(round(height * scale)))), Image.Resampling.LANCZOS)
    array = np.asarray(photo, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    ckpt = REPO_ROOT / "models" / "MoGe" / "model.pt"
    if not ckpt.is_file():
        print(f"STATUS // missing MoGe weights {ckpt}", flush=True)
        return 2
    model = get_moge_model(ckpt=str(ckpt), device=device)
    try:
        output = model.infer(tensor.to(device), fov_x=None)
        intrinsics = output["intrinsics"].detach().float().cpu().numpy()
        fx = float(intrinsics[0, 0] if intrinsics.ndim == 2 else intrinsics[0, 0, 0])
        if fx <= 1e-6:
            print("STATUS // MoGe returned no focal length", flush=True)
            return 1
        degrees = math.degrees(2.0 * math.atan(0.5 / fx))
        print(f"HFOV\t{_snap(degrees):.1f}", flush=True)
        return 0
    finally:
        model.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    sys.exit(main())
