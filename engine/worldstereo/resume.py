"""Which trajectories still need sampling, and which memory snapshot to restore.

A per-rail trajectory is done when it has ``NFRAME`` front keyframes, ``meta.json``,
and no ``STALE.txt``. Layouts from older runs (spaced, dense, joint) are not a
finished per-rail clip. A joint sample is done when ``joint.json`` lists every
keyframe and ``STALE.txt`` is absent. Snapshots are named after the trajectory
that had just been added (``after_<view_id>``). ``after_joint`` is the joint sample.
``after_rail_XX`` is what a later rail's re-gen restores.
"""

from __future__ import annotations

import json
from pathlib import Path

from engine.pano.scene_io import scene_dir
from engine.worldstereo import defaults as D
from engine.worldstereo.adapter import TrajJob


def rail_dir(scene: str, job: TrajJob) -> Path:
    return scene_dir(scene) / job.output_rel


def _meta(path: Path) -> dict | None:
    file = path / "meta.json"
    if not file.is_file():
        return None
    try:
        data = json.loads(file.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def trajectory_complete(path: Path, nframe: int = D.NFRAME, layout: str | None = None) -> bool:
    """Done for a per-rail front clip: ``nframe`` keyframes, meta, and no ``STALE.txt``.

    A legacy folder with no layout is treated as separate. Spaced, dense, and
    joint folders are unfinished in per-rail mode, so Generate runs them again.
    """
    frames = path / "keyframes"
    meta = _meta(path)
    if not frames.is_dir() or meta is None:
        return False
    if (path / "STALE.txt").is_file():
        return False
    saved = str(meta.get("layout") or "separate")
    if saved in {"spaced", "dense", "joint"}:
        return False
    if layout is not None and saved != str(layout):
        return False
    count = len(list(frames.glob("[0-9][0-9][0-9].png")))
    return count >= int(nframe)


def joint_complete(scene: str, frames_per_rail: int | None = None) -> bool:
    """The joint sample is done when every listed keyframe exists and nothing is stale.

    A saved run with a different ``frames_per_rail`` is not this job. The 10-frame
    alpine result must not block a 20-frame run.
    """
    dest = scene_dir(scene) / "worldstereo" / "joint"
    path = dest / "joint.json"
    if not path.is_file() or (dest / "STALE.txt").is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    if frames_per_rail is not None and int(payload.get("frames_per_rail") or 0) != int(frames_per_rail):
        return False
    rows = payload.get("frames")
    if not isinstance(rows, list) or not rows:
        return False
    keyframes = dest / "keyframes"
    return all((keyframes / f"{int(row.get('index', -1)):03d}.png").is_file() for row in rows)


def memory_root(scene: str) -> Path:
    return scene_dir(scene) / "worldstereo" / "memory"


def view_snapshot(scene: str, job: TrajJob) -> Path:
    return memory_root(scene) / f"after_{job.view_id}"


def rail_snapshot(scene: str, rail_index: int) -> Path:
    return memory_root(scene) / f"after_rail_{int(rail_index):02d}"


def rails_in_order(jobs: list[TrajJob]) -> list[int]:
    seen: list[int] = []
    for job in jobs:
        if job.rail_index not in seen:
            seen.append(job.rail_index)
    return seen


def job_complete(scene: str, job: TrajJob) -> bool:
    """One per-rail front trajectory."""
    return trajectory_complete(rail_dir(scene, job), layout=job.layout)


def pending_jobs(scene: str, jobs: list[TrajJob]) -> list[TrajJob]:
    """Jobs from the first unfinished trajectory through the end of the order."""
    for index, job in enumerate(jobs):
        if not job_complete(scene, job):
            return jobs[index:]
    return []


def resume_snapshot(scene: str, jobs: list[TrajJob], pending: list[TrajJob]) -> Path | None:
    """Snapshot written by the trajectory immediately before ``pending``."""
    if not pending or not jobs or pending[0] is jobs[0]:
        return None
    previous = jobs[jobs.index(pending[0]) - 1]
    return view_snapshot(scene, previous)


def regen_plan(jobs: list[TrajJob], rail_index: int) -> tuple[list[TrajJob], int | None, list[TrajJob]]:
    """Views to resample, the previous rail in user order, and later views to mark stale.

    ``previous`` is None when ``rail_index`` is the first rail in the order.
    """
    order = rails_in_order(jobs)
    if rail_index not in order:
        raise ValueError(f"rail {rail_index} is not in this run")
    pos = order.index(int(rail_index))
    previous = order[pos - 1] if pos else None
    later = set(order[pos + 1 :])
    rerun = [job for job in jobs if job.rail_index == int(rail_index)]
    stale = [job for job in jobs if job.rail_index in later]
    return rerun, previous, stale
