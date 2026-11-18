"""Load nvidia/difix_ref and clean one alpine_02 frame. Runs in the difix venv."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
os.environ["HF_HOME"] = str(REPO / ".hf_cache")
os.environ["HF_HUB_CACHE"] = str(REPO / ".hf_cache" / "hub")
os.environ["HF_XET_CACHE"] = str(REPO / ".hf_cache" / "xet")
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
sys.path.insert(0, str(REPO / "vendor" / "difix3d" / "src"))


def main() -> int:
    import torch
    from PIL import Image

    sys.path.insert(0, str(REPO))
    from engine.workers.hy_difix import load_pipeline

    images = REPO / "outputs" / "alpine_02" / "hyworld" / "gs_data" / "images"
    src = images / "view1-traj0_000010.png"
    if not src.is_file():
        found = sorted(images.glob("view*.png"))
        if not found:
            print("STATUS // no view image in gs_data")
            return 1
        src = found[len(found) // 2]
    stem, index = src.stem.rsplit("_", 1)
    ref = images / f"{stem}_{int(index) - 1:06d}.png"
    if not ref.is_file():
        ref = src
    out = REPO / "outputs" / "alpine_02" / "hyworld" / "_votion" / "bench" / "difix"
    out.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    pipe = load_pipeline("nvidia/difix_ref", str(REPO / "vendor" / "difix3d" / "src"))
    pipe.to("cuda")
    load_s = time.perf_counter() - started
    image = Image.open(src).convert("RGB")
    reference = Image.open(ref).convert("RGB")
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    times = []
    result = None
    for _ in range(3):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        result = pipe(
            "remove degradation",
            image=image,
            ref_image=reference,
            num_inference_steps=1,
            timesteps=[199],
            guidance_scale=0.0,
        ).images[0]
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    assert result is not None
    result.save(out / "check_out.png")
    peak = torch.cuda.max_memory_allocated() / 2**30
    print(f"src {src.name} ref {ref.name}")
    print(f"load_s {load_s:.1f}")
    print(f"image_s {sum(times) / len(times):.2f}")
    print(f"peak_gb {peak:.2f}")
    print("check_difix ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
