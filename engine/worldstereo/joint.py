"""One WorldStereo sample covering every kept rail.

Frames alternate direction: rail A outward, rail B back to the star, and so on.
``JOINT_FRAMES_PER_RAIL`` and ``JOINT_MAX_FRAMES_PER_GEN`` are the only sizes.
Raising them grows the clip; a sequence longer than one sample is split into
chunks that chain through the memory bank.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from engine.pano.scene_io import scene_dir
from engine.worldstereo import defaults as D
from engine.worldstereo.views import rail_view_rel


@dataclass(frozen=True)
class JointStop:
    rail: int
    pose: int


@dataclass(frozen=True)
class JointChunk:
    index: int
    frames: tuple[JointStop, ...]


def sample_poses(count: int, n: int) -> list[int]:
    """``n`` pose indices from 0 through ``count - 1``, including both ends."""
    count = int(count)
    n = int(n)
    if count < 2 or n < 2:
        raise ValueError(f"need at least 2 poses, got count={count} n={n}")
    if n > count:
        raise ValueError(f"cannot take {n} poses from a rail of {count}")
    raw = np.linspace(0, count - 1, n).round().astype(int).tolist()
    out: list[int] = []
    for value in raw:
        pose = int(value)
        if out and pose <= out[-1]:
            pose = out[-1] + 1
        if pose >= count:
            break
        out.append(pose)
    if len(out) != n or out[0] != 0 or out[-1] != count - 1:
        raise ValueError(f"pose sample {out} does not cover 0..{count - 1} in {n} steps")
    return out


def joint_order(rail_counts: list[tuple[int, int]], frames_per_rail: int) -> list[JointStop]:
    """Out-and-back. Even rails run from the star; odd rails run back to it.

    ``rail_counts`` is ``(rail index, pose count)`` in the user's rail order.
    """
    if not rail_counts:
        raise ValueError("joint needs at least one rail")
    frames: list[JointStop] = []
    for step, (rail, count) in enumerate(rail_counts):
        poses = sample_poses(int(count), int(frames_per_rail))
        if step % 2 == 1:
            poses = list(reversed(poses))
        frames.extend(JointStop(int(rail), int(pose)) for pose in poses)
    return frames


def _split_chunks(frames: list[JointStop], max_per_gen: int) -> list[JointChunk]:
    limit = int(max_per_gen)
    if limit < 2:
        raise ValueError(f"max frames per generation must be at least 2, got {limit}")
    if len(frames) <= limit:
        return [JointChunk(0, tuple(frames))]
    chunks: list[list[JointStop]] = []
    rest = list(frames)
    while len(rest) > limit:
        take = limit if len(rest) - limit >= 2 else len(rest) - 2
        chunks.append(rest[:take])
        rest = rest[take:]
    chunks.append(rest)
    return [JointChunk(index, tuple(chunk)) for index, chunk in enumerate(chunks)]


def joint_plan(
    rail_counts: list[tuple[int, int]],
    frames_per_rail: int | None = None,
    max_per_gen: int | None = None,
) -> list[JointChunk]:
    per = D.JOINT_FRAMES_PER_RAIL if frames_per_rail is None else int(frames_per_rail)
    cap = D.JOINT_MAX_FRAMES_PER_GEN if max_per_gen is None else int(max_per_gen)
    return _split_chunks(joint_order(rail_counts, per), cap)


def joint_dir(scene: str) -> Path:
    return scene_dir(scene) / "worldstereo" / "joint"


def _write_mp4(frames_bgr: list[np.ndarray], path: Path, fps: int = 16) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames_bgr[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"VideoWriter failed to open {path}")
    try:
        for frame in frames_bgr:
            writer.write(frame)
    finally:
        writer.release()


def write_joint(
    scene: str,
    order: list[int],
    frames: list[JointStop],
    images_rgb: np.ndarray,
    color_bgr: list[np.ndarray],
    mask_bgr: list[np.ndarray],
    w2cs: np.ndarray,
    K: np.ndarray,
    *,
    frames_per_rail: int,
    seed: int,
    seconds: float,
    peak_gb: float,
) -> Path:
    """Write the sample-order preview under ``worldstereo/joint``."""
    if len(frames) != int(images_rgb.shape[0]) or len(frames) != len(color_bgr) or len(frames) != int(w2cs.shape[0]):
        raise RuntimeError(
            f"joint has {len(frames)} stops, {images_rgb.shape[0]} images, "
            f"{len(color_bgr)} renders, {w2cs.shape[0]} poses"
        )
    dest = joint_dir(scene)
    keyframes = dest / "keyframes"
    if keyframes.exists():
        shutil.rmtree(keyframes)
    keyframes.mkdir(parents=True)
    rows = []
    for index, stop in enumerate(frames):
        name = f"{index:03d}.png"
        cv2.imwrite(str(keyframes / name), cv2.cvtColor(images_rgb[index], cv2.COLOR_RGB2BGR))
        rows.append(
            {
                "index": index,
                "rail": int(stop.rail),
                "pose": int(stop.pose),
                "w2c": np.asarray(w2cs[index], dtype=np.float64).tolist(),
            }
        )
    _write_mp4(color_bgr, dest / "render" / "render.mp4")
    _write_mp4(mask_bgr, dest / "mask" / "render_mask.mp4")
    camera = {
        "extrinsic": np.asarray(w2cs, dtype=np.float64).tolist(),
        "intrinsic": [np.asarray(K, dtype=np.float64).tolist() for _ in frames],
    }
    (dest / "camera.json").write_text(json.dumps(camera), encoding="utf-8")
    payload = {
        "order": [int(rail) for rail in order],
        "frames_per_rail": int(frames_per_rail),
        "frames": rows,
        "seed": int(seed),
        "seconds": float(seconds),
        "peak_gb": float(peak_gb),
    }
    (dest / "joint.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    stale = dest / "STALE.txt"
    if stale.is_file():
        stale.unlink()
    return dest


def split_joint(scene: str, K: np.ndarray | None = None) -> list[Path]:
    """Copy each rail's frames into ``rails/rail_NN/front`` in ascending pose order."""
    dest = joint_dir(scene)
    payload = json.loads((dest / "joint.json").read_text(encoding="utf-8"))
    rows = payload.get("frames") or []
    if not isinstance(rows, list) or not rows:
        raise RuntimeError(f"{dest / 'joint.json'} has no frames")
    by_rail: dict[int, list[dict]] = {}
    for row in rows:
        by_rail.setdefault(int(row["rail"]), []).append(row)
    written: list[Path] = []
    for rail, items in by_rail.items():
        items.sort(key=lambda row: int(row["pose"]))
        folder = scene_dir(scene) / rail_view_rel(rail, "front")
        keyframes = folder / "keyframes"
        keyframes.mkdir(parents=True, exist_ok=True)
        expected = {f"{int(row['pose']):03d}.png" for row in items}
        for old in keyframes.glob("[0-9][0-9][0-9].png"):
            if old.name not in expected:
                old.unlink()
        w2cs = []
        colors = []
        masks = []
        cap_c = cv2.VideoCapture(str(dest / "render" / "render.mp4"))
        cap_m = cv2.VideoCapture(str(dest / "mask" / "render_mask.mp4"))
        try:
            for row in items:
                src = dest / "keyframes" / f"{int(row['index']):03d}.png"
                shot = keyframes / f"{int(row['pose']):03d}.png"
                shutil.copyfile(src, shot)
                w2cs.append(row["w2c"])
                cap_c.set(cv2.CAP_PROP_POS_FRAMES, int(row["index"]))
                cap_m.set(cv2.CAP_PROP_POS_FRAMES, int(row["index"]))
                ok_c, frame_c = cap_c.read()
                ok_m, frame_m = cap_m.read()
                if not ok_c or not ok_m:
                    raise RuntimeError(f"joint video missing frame {row['index']}")
                colors.append(frame_c)
                masks.append(frame_m)
        finally:
            cap_c.release()
            cap_m.release()
        intrinsic = np.asarray(K, dtype=np.float64) if K is not None else None
        if intrinsic is None:
            joint_cam = json.loads((dest / "camera.json").read_text(encoding="utf-8"))
            intrinsic = np.asarray(joint_cam["intrinsic"][0], dtype=np.float64)
        camera = {
            "extrinsic": w2cs,
            "intrinsic": [intrinsic.tolist() for _ in items],
        }
        (folder / "camera.json").write_text(json.dumps(camera), encoding="utf-8")
        _write_mp4(colors, folder / "render" / "render.mp4")
        _write_mp4(masks, folder / "mask" / "render_mask.mp4")
        positions = [int(row["pose"]) for row in items]
        meta = {
            "layout": "joint",
            "positions": positions,
            "frames": len(positions),
            "view": "front",
            "rail_index": int(rail),
            "anchor": False,
            "seed": payload.get("seed"),
            "seconds": payload.get("seconds"),
            "peak_gb": payload.get("peak_gb"),
        }
        (folder / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        written.append(folder)
    return written
