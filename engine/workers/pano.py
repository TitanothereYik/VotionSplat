"""engine.workers.pano — one process, then exit (VRAM back before Phase 2 MoGe)."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from engine.job import REPO_ROOT, job_progress
from engine.pano import defaults as D
from engine.pano.aspect import aspect_error, is_2to1
from engine.pano.comfy_bootstrap import apply_comfy_argv
from engine.pano.scene_io import copy_as_pano, pano_path, scene_dir, write_prompt


def _cuda_index() -> int:
    try:
        return int(os.environ.get("CUDA_DEVICE_INDEX", "0"))
    except ValueError:
        return 0


def _parse(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Votion Phase 1 panorama worker")
    p.add_argument("--mode", required=True, choices=("2d", "360", "text"))
    p.add_argument("--action", default="generate", choices=("generate", "use", "upscale"))
    p.add_argument("--scene", default="alpine_01")
    p.add_argument("--image", default="")
    p.add_argument("--sentence", default="")
    p.add_argument("--prompt", nargs="?", const="", default="")
    p.add_argument("--upscaler-prompt", default=D.UPSCALER_PROMPT_DEFAULT)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--seam-seed", type=int, default=None)
    p.add_argument("--hfov", type=float, default=D.H_FOV_DEG_DEFAULT)
    args = p.parse_args(argv)
    if args.seed is None:
        args.seed = D.IMAGE_PANO_SEED_DEFAULT if args.mode == "2d" else D.TEXT_PANO_SEED_DEFAULT
    if args.seam_seed is None:
        args.seam_seed = D.IMAGE_SEAM_SEED_DEFAULT if args.mode == "2d" else D.TEXT_SEAM_SEED_DEFAULT
    return args


def _clamp_hfov(value: float) -> float:
    return max(D.H_FOV_DEG_MIN, min(D.H_FOV_DEG_MAX, float(value)))


def _clear_pano_source(scene: str) -> None:
    folder = scene_dir(scene) / "pano_source"
    if folder.exists():
        shutil.rmtree(folder)


def _save_pano_source(scene: str, photo: Image.Image, h_fov: float) -> None:
    folder = scene_dir(scene) / "pano_source"
    folder.mkdir(parents=True, exist_ok=True)
    photo.save(folder / "photo.png", format="PNG")
    payload = {
        "h_fov": float(h_fov),
        "yaw": 0.0,
        "pitch": 0.0,
        "width": int(photo.size[0]),
        "height": int(photo.size[1]),
    }
    (folder / "pano_source.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _run_2d(
    scene: str,
    image: Path,
    surroundings: str,
    h_fov_deg: float,
    seed: int,
    seam_seed: int,
    upscaler_prompt: str,
) -> int:
    surroundings = D.with_surroundings_suffix(surroundings)
    if not surroundings:
        print("STATUS // empty surroundings - block Generate", flush=True)
        return 2
    if not image.is_file():
        print(f"STATUS // missing 2D file: {image}", flush=True)
        return 2
    hfov = _clamp_hfov(h_fov_deg)
    prompt = D.OUTPAINT_INSTRUCTION + " " + surroundings
    print("STATUS // 2D Images - Ostris Edit + ERP outpaint LoRA", flush=True)
    print("STATUS // outpaint instruction locked", flush=True)
    with Image.open(image) as im:
        im.load()
        photo = im.convert("RGB")
    _save_pano_source(scene, photo, hfov)
    apply_comfy_argv(REPO_ROOT / "models", pano_path(scene).parent, _cuda_index())
    from engine.pano.krea_runtime import run_image_pano
    from engine.pano.photo_paste import paste_photo
    from engine.pano.scene_io import save_pano_image

    out = run_image_pano(photo, surroundings, hfov, seed, seam_seed, upscaler_prompt)
    before = scene_dir(scene) / "_work"
    before.mkdir(parents=True, exist_ok=True)
    out.convert("RGB").save(before / "pano_before_paste.png", format="PNG")
    pasted = paste_photo(np.asarray(out.convert("RGB")), np.asarray(photo), hfov)
    print(f"STATUS // photo pasted at 8K feather={D.PHOTO_FEATHER_PX}", flush=True)
    save_pano_image(scene, Image.fromarray(pasted))
    write_prompt(scene, prompt)
    print(f"PROMPT\t{prompt}", flush=True)
    return 0


def _load_erp(path: Path) -> Image.Image:
    with Image.open(path) as im:
        im.load()
        rgb = im.convert("RGB")
    w, h = rgb.size
    if not is_2to1(w, h):
        raise ValueError(aspect_error(w, h))
    return rgb


def _run_360_use(scene: str, image: Path, prompt: str) -> int:
    _clear_pano_source(scene)
    print("STATUS // 360 Images - Use as pano.png (no Krea)", flush=True)
    rgb = _load_erp(image)
    dest = copy_as_pano(scene, image)
    write_prompt(scene, prompt)
    print(f"PROMPT\t{prompt}", flush=True)
    print(f"STATUS // stored as-is {rgb.size[0]}x{rgb.size[1]} -> {dest}", flush=True)
    return 0


def _run_360_upscale(scene: str, image: Path, prompt: str, upscaler_prompt: str) -> int:
    _clear_pano_source(scene)
    print("STATUS // 360 Images - Upscale to 8K (shared tail)", flush=True)
    rgb = _load_erp(image)
    write_prompt(scene, prompt)
    apply_comfy_argv(REPO_ROOT / "models", pano_path(scene).parent, _cuda_index())
    from engine.pano.krea_runtime import run_8k_tail_only

    out = run_8k_tail_only(rgb, upscaler_prompt)
    from engine.pano.scene_io import save_pano_image

    save_pano_image(scene, out)
    print(f"PROMPT\t{prompt}", flush=True)
    return 0


def _run_text(scene: str, sentence: str, seed: int, seam_seed: int, upscaler_prompt: str) -> int:
    sentence = (sentence or "").strip()
    if not sentence:
        print("STATUS // empty scene sentence - block Generate", flush=True)
        return 2
    # Prefix is locked. User sentence is appended after the comma (dummy + space).
    _clear_pano_source(scene)
    prompt = D.TRIGGER_PREFIX + " " + sentence
    print("STATUS // text - no 2D input image", flush=True)
    print(f"STATUS // Krea prompt prefix locked", flush=True)
    apply_comfy_argv(REPO_ROOT / "models", pano_path(scene).parent, _cuda_index())
    from engine.pano.krea_runtime import run_text_pano
    from engine.pano.scene_io import save_pano_image

    out = run_text_pano(prompt, seed, seam_seed, upscaler_prompt)
    save_pano_image(scene, out)
    write_prompt(scene, prompt)
    print(f"PROMPT\t{prompt}", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parse(argv)
    print(f"STATUS // panorama mode={args.mode} action={args.action} scene={args.scene}", flush=True)
    job_progress(0, 1, f"pano {args.mode} {args.action}")
    try:
        if args.mode == "2d":
            return _run_2d(
                args.scene,
                Path(args.image),
                args.sentence,
                args.hfov,
                args.seed,
                args.seam_seed,
                args.upscaler_prompt,
            )
        if args.mode == "360":
            image = Path(args.image)
            if not image.is_file():
                print(f"STATUS // missing 360 file: {image}", flush=True)
                return 2
            if args.action == "upscale":
                return _run_360_upscale(args.scene, image, args.prompt, args.upscaler_prompt)
            return _run_360_use(args.scene, image, args.prompt)
        if args.mode == "text":
            return _run_text(
                args.scene,
                args.sentence,
                args.seed,
                args.seam_seed,
                args.upscaler_prompt,
            )
        print(f"STATUS // unknown mode {args.mode}", flush=True)
        return 2
    except Exception as exc:
        print(f"STATUS // pano failed: {type(exc).__name__}: {exc}", flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
