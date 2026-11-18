"""Clean one round of novel views with nvidia/difix_ref. Runs in the difix venv."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


def local_snapshot() -> Path | None:
    """A finished nvidia/difix_ref snapshot in the local Hugging Face cache."""
    root = Path(os.environ.get("HF_HUB_CACHE", "")) / "models--nvidia--difix_ref" / "snapshots"
    if not root.is_dir():
        return None
    for snap in root.iterdir():
        unet = snap / "unet" / "diffusion_pytorch_model.safetensors"
        vae = snap / "vae" / "diffusion_pytorch_model.safetensors"
        if unet.is_file() and vae.is_file() and unet.stat().st_size > 3_000_000_000:
            return snap
    return None


def load_pipeline(model: str, src: str):
    """Load Difix. A finished local snapshot wins over another download."""
    sys.path.insert(0, str(src))
    from pipeline_difix import DifixPipeline

    source = str(local_snapshot() or model)
    last = None
    for attempt in range(1, 9):
        try:
            return DifixPipeline.from_pretrained(source, trust_remote_code=True)
        except Exception as exc:
            last = exc
            print(f"STATUS // difix load failed ({type(exc).__name__}), retry {attempt}", flush=True)
            time.sleep(3)
    raise last

REPO = Path(__file__).resolve().parents[2]
os.environ.setdefault("HF_HOME", str(REPO / ".hf_cache"))
os.environ.setdefault("HF_HUB_CACHE", str(REPO / ".hf_cache" / "hub"))
os.environ.setdefault("HF_XET_CACHE", str(REPO / ".hf_cache" / "xet"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


def main() -> int:
    from engine.hyworld.train_run import DIFIX_MODEL, DIFIX_SRC

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round-dir", required=True)
    parser.add_argument("--model", default=DIFIX_MODEL)
    parser.add_argument("--src", default=str(DIFIX_SRC))
    args = parser.parse_args()
    folder = Path(args.round_dir)
    done = folder / "done.json"
    if done.is_file():
        print(f"STATUS // difix round already done {folder.name}", flush=True)
        return 0
    from PIL import Image

    poses = json.loads((folder / "poses.json").read_text(encoding="utf-8"))
    fixed = folder / "fixed"
    fixed.mkdir(parents=True, exist_ok=True)
    pipe = load_pipeline(args.model, args.src)
    if hasattr(pipe, "set_progress_bar_config"):
        pipe.set_progress_bar_config(disable=True)
    pipe.to("cuda")
    started = time.perf_counter()
    total = len(poses)
    for index, pose in enumerate(poses):
        name = str(pose["name"])
        image = Image.open(folder / "novel" / f"{name}.png").convert("RGB")
        reference = Image.open(pose["ref"]).convert("RGB")
        output = pipe(
            "remove degradation",
            image=image,
            ref_image=reference,
            num_inference_steps=1,
            timesteps=[199],
            guidance_scale=0.0,
        ).images[0]
        output.resize((int(pose["width"]), int(pose["height"])), Image.LANCZOS).save(fixed / f"{name}.png")
        pct = int(100 * (index + 1) / max(total, 1))
        print(f"PROGRESS\tjob\t{index + 1}\t{total}\t{pct}\tdifix {name}", flush=True)
    seconds = time.perf_counter() - started
    payload = {"count": total, "seconds": round(seconds, 1)}
    tmp = folder / "done.json.tmp"
    tmp.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    os.replace(tmp, done)
    print(f"STATUS // difix round done {total} views {seconds:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
