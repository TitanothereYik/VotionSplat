"""WorldStereo worker: one process, every rail, one memory bank.

Qt never imports this. Run as ``python -m engine.workers.worldstereo --scene NAME``.

Outputs land under ``<scene>/worldstereo/`` only. WAN files are not written.
HiRes does not start. An empty ``prompt.txt`` refuses the run.
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
from engine.worldstereo.adapter import build_scene, list_trajs


def _log(msg: str) -> None:
    print(msg, flush=True)


def main(argv: list[str] | None = None) -> int:
    apply_huggingface_cache_env(os.environ)
    parser = argparse.ArgumentParser(description="Votion WorldStereo (one process, all rails)")
    parser.add_argument("--scene", required=True)
    parser.add_argument("--start-rail", type=int, default=0)
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--only", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--request", type=str, default="")
    parser.add_argument("--rails", default="")
    parser.add_argument("--seed", type=int, default=D.SEED)
    args = parser.parse_args(argv)

    prompt = ""
    views = 1
    if args.request:
        req_path = Path(args.request)
        if req_path.is_file():
            payload = json.loads(req_path.read_text(encoding="utf-8"))
            prompt = str(payload.get("prompt") or "").strip()
            views = int(payload.get("views") or 1)
    if not prompt:
        path = prompt_path(args.scene)
        if path.is_file():
            prompt = path.read_text(encoding="utf-8").strip()
    if not prompt:
        _log("STATUS // empty prompt blocks Generate")
        return 2
    if views != 1:
        _log("STATUS // views per rail is 1 in this build (the rail direction)")

    try:
        jobs = list_trajs(args.scene, limit=max(1, int(args.limit)))
    except FileNotFoundError as exc:
        _log(f"STATUS // {exc}")
        return 2
    if args.rails.strip():
        wanted = []
        for part in args.rails.split(","):
            part = part.strip()
            if part:
                wanted.append(int(part))
        by_index = {job.rail_index: job for job in jobs}
        ordered = [by_index[index] for index in wanted if index in by_index]
        if not ordered:
            _log("STATUS // rail order matched no Plot Camera rails")
            return 2
        jobs = ordered

    _log(
        "STATUS // worldstereo prepare "
        + " ".join(f"{job.rail_index}:{job.direction}" for job in jobs)
    )
    job_progress(0, len(jobs), "prepare inputs")
    try:
        out = build_scene(args.scene, jobs, prompt)
    except FileNotFoundError as exc:
        _log(f"STATUS // {exc}")
        return 2
    _log(f"STATUS // worldstereo input {out}")
    _log("WS\tinputs\tready")
    if args.prepare_only:
        job_progress(len(jobs), len(jobs), "inputs ready")
        return 0

    from engine.worldstereo.runtime import generate

    return generate(args.scene, jobs, prompt, int(args.start_rail), bool(args.only), seed=int(args.seed))


if __name__ == "__main__":
    sys.exit(main())
