"""WorldStereo worker: one process, one memory bank.

Qt never imports this. Run as ``python -m engine.workers.worldstereo --scene NAME``.

Joint mode writes ``<scene>/worldstereo/joint`` and splits each rail into
``rails/rail_NN/front``. Per-rail mode writes one 21-frame front view per rail.
An empty prompt refuses the run. Generate resumes by default.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ["MASTER_ADDR"] = "127.0.0.1"
os.environ["MASTER_PORT"] = "29531"
os.environ["RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from engine.job import apply_huggingface_cache_env, job_progress
from engine.pano.scene_io import prompt_path
from engine.worldstereo import defaults as D
from engine.worldstereo.adapter import build_joint_scene, build_scene, list_trajs


def _log(msg: str) -> None:
    print(msg, flush=True)


def _request(path: str) -> dict:
    if not path:
        return {}
    file = Path(path)
    if not file.is_file():
        return {}
    payload = json.loads(file.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def main(argv: list[str] | None = None) -> int:
    apply_huggingface_cache_env(os.environ)
    parser = argparse.ArgumentParser(description="Votion WorldStereo (one process)")
    parser.add_argument("--scene", required=True)
    parser.add_argument("--mode", default="", help="joint (default) or rail")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--restart", action="store_true")
    parser.add_argument("--regen-rail", type=int, default=-1)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--request", type=str, default="")
    parser.add_argument("--rails", default="")
    parser.add_argument("--seed", type=int, default=D.SEED)
    args = parser.parse_args(argv)

    payload = _request(args.request)
    prompt = str(payload.get("prompt") or "").strip()
    if not prompt:
        path = prompt_path(args.scene)
        if path.is_file():
            prompt = path.read_text(encoding="utf-8").strip()
    if not prompt:
        _log("STATUS // empty prompt blocks Generate")
        return 2

    mode = str(args.mode or payload.get("mode") or "joint").strip().lower()
    if mode not in {"joint", "rail"}:
        _log(f"STATUS // unknown mode {mode}")
        return 2
    order: list[int] | None = None
    if args.rails.strip():
        order = [int(part) for part in args.rails.split(",") if part.strip()]

    if mode == "joint":
        if int(args.regen_rail) >= 0:
            _log("STATUS // joint is one sample; --regen-rail ignored")
        frames = payload.get("frames_per_rail")
        frames_per_rail = D.JOINT_FRAMES_PER_RAIL if frames is None else int(frames)
        rails = order or []
        if not rails:
            _log("STATUS // joint needs the kept rail list")
            return 2
        _log(f"STATUS // worldstereo prepare mode=joint rails={','.join(str(i) for i in rails)} frames_per_rail={frames_per_rail}")
        job_progress(0, 1, "prepare joint")
        try:
            out = build_joint_scene(args.scene, rails, prompt, frames_per_rail=frames_per_rail)
        except (FileNotFoundError, ValueError, RuntimeError) as exc:
            _log(f"STATUS // {exc}")
            return 2
        _log(f"STATUS // worldstereo input {out}")
        if args.prepare_only:
            job_progress(1, 1, "inputs ready")
            return 0
        from engine.worldstereo.runtime import generate_joint

        return generate_joint(
            args.scene,
            rails,
            prompt,
            frames_per_rail=frames_per_rail,
            resume=not bool(args.restart),
            seed=int(args.seed),
        )

    try:
        jobs = list_trajs(args.scene, order=order)
    except FileNotFoundError as exc:
        _log(f"STATUS // {exc}")
        return 2
    _log(
        "STATUS // worldstereo prepare mode=rail jobs="
        + str(len(jobs))
        + " "
        + " ".join(f"{job.rail_index}:{job.view}" for job in jobs[:12])
    )
    job_progress(0, len(jobs), "prepare inputs")
    try:
        out = build_scene(args.scene, jobs, prompt)
    except FileNotFoundError as exc:
        _log(f"STATUS // {exc}")
        return 2
    _log(f"STATUS // worldstereo input {out}")
    if args.prepare_only:
        job_progress(len(jobs), len(jobs), "inputs ready")
        return 0
    from engine.worldstereo.runtime import generate

    regen = None if int(args.regen_rail) < 0 else int(args.regen_rail)
    return generate(
        args.scene,
        jobs,
        prompt,
        resume=not bool(args.restart),
        regen_rail=regen,
        seed=int(args.seed),
    )


if __name__ == "__main__":
    sys.exit(main())
