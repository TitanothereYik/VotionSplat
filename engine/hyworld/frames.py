"""Which frames the splat trains on: the Expand frames or a HiRes scale. No torch: the Qt process imports this."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from engine.hyworld import hires as HR
from engine.hyworld.paths import gs_data, hy_root
from engine.hyworld.train_run import _link, _png_size

FRAMES = ("expand", "x2", "x4")
SCALES = {"expand": 1, "x2": 2, "x4": 4}
CROP = {"x4": 1280}  # full 4x frames need over 24 GB of VRAM, so each step trains on a random square crop
STAMP = "votion_frames.json"
_STILLS = ("panorama_", "polar_")


def frames_dir(scene: str, key: str) -> Path:
    """``gs_data`` for the Expand frames; HiRes sets sit beside it as ``gs_data_x2`` / ``gs_data_x4``."""
    return gs_data(scene) if key == "expand" else hy_root(scene) / f"gs_data_{key}"


def _cameras(scene: str) -> dict | None:
    try:
        data = json.loads((gs_data(scene) / "cameras.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def video_frames(cameras: dict) -> dict[str, list[tuple[str, int]]]:
    """``view0/traj0`` -> [(image name, video frame index)] for every Expand frame in the dataset."""
    items: dict[str, list[tuple[str, int]]] = {}
    for name, cam in cameras.items():
        if not isinstance(cam, dict) or name.startswith(_STILLS):
            continue
        head, _, index = name.rpartition("_")
        view, _, traj = head.rpartition("-")
        if not view or not traj.startswith("traj") or not index.isdigit():
            continue
        items.setdefault(f"{view}/{traj}", []).append((name, int(index)))
    return items


def _base_size(scene: str, cameras: dict) -> tuple[int, int]:
    width, height = int(cameras.get("width") or 0), int(cameras.get("height") or 0)
    if width and height:
        return width, height
    first = next(iter(sorted((gs_data(scene) / "images").glob("*.png"))), None)
    return _png_size(first) if first is not None else (0, 0)


def frame_options(scene: str) -> list[dict]:
    """One row per frame source: ``key``, ``label``, ``ok``, ``note``."""
    cameras = _cameras(scene)
    if cameras is None:
        return [
            {"key": key, "label": "Expand frames" if key == "expand" else f"HiRes {SCALES[key]}x",
             "ok": False, "note": "Build the dataset first."}
            for key in FRAMES
        ]
    width, height = _base_size(scene, cameras)
    rows = [{"key": "expand", "label": f"Expand frames ({width}x{height})", "ok": True, "note": ""}]
    items = video_frames(cameras)
    for key in ("x2", "x4"):
        scale = SCALES[key]
        label = f"HiRes {scale}x ({width * scale}x{height * scale})"
        missing = [item for item in items if HR.hires_state(scene, item, scale) != "done"]
        if not items:
            note = "The dataset has no Expand frames."
        elif len(missing) == len(items):
            note = f"Run HiRes {scale}x to train on {scale}x frames."
        elif missing:
            note = f"HiRes {scale}x is missing or out of date for {len(missing)} of {len(items)} paths."
        else:
            note = ""
        rows.append({"key": key, "label": label, "ok": not note, "note": note})
    return rows


def _stamp(scene: str, key: str, items: dict) -> dict:
    stat = (gs_data(scene) / "cameras.json").stat()
    stamp = {"key": key, "base": [stat.st_size, stat.st_mtime_ns], "items": {}}
    for item in sorted(items):
        meta = (HR.hires_dir(scene, item, SCALES[key]) / "meta.json").stat()
        stamp["items"][item] = [meta.st_size, meta.st_mtime_ns]
    return stamp


def build_frames(scene: str, key: str, log=print) -> Path:
    """The training folder for ``key``. HiRes sets hardlink the dataset and swap in the upscaled frames."""
    base = gs_data(scene)
    if key == "expand":
        return base
    if key not in SCALES:
        raise ValueError(f"unknown frames {key}")
    row = next(row for row in frame_options(scene) if row["key"] == key)
    if not row["ok"]:
        raise RuntimeError(row["note"])
    cameras = _cameras(scene) or {}
    items = video_frames(cameras)
    scale = SCALES[key]
    out = frames_dir(scene, key)
    stamp = _stamp(scene, key, items)
    try:
        old = json.loads((out / STAMP).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        old = None
    if old == stamp:
        log(f"STATUS // {out.name} is up to date")
        return out
    swap: dict[str, Path] = {}
    for item, rows in items.items():
        folder = HR.hires_dir(scene, item, scale) / "frames"
        for name, index in rows:
            src = folder / f"{index:03d}.png"
            if not src.is_file():
                raise RuntimeError(f"missing HiRes frame {src}")
            swap[name] = src
    staging = out.with_name(out.name + ".new")
    if staging.exists():
        shutil.rmtree(staging)
    for src in sorted(base.rglob("*")):
        if not src.is_file():
            continue
        rel = src.relative_to(base)
        if rel.as_posix() in ("cameras.json", STAMP):
            continue
        if rel.parts[0] == "images" and src.stem in swap:
            continue
        _link(src, staging / rel)
    merged = dict(cameras)
    for name, src in swap.items():
        _link(src, staging / "images" / f"{name}.png")
        w0, h0 = _png_size(base / "images" / f"{name}.png")
        w1, h1 = _png_size(src)
        K = [[float(v) for v in row] for row in cameras[name]["intrinsic"]]
        K[0] = [v * w1 / w0 for v in K[0]]
        K[1] = [v * h1 / h0 for v in K[1]]
        merged[name] = {**cameras[name], "intrinsic": K}
    (staging / "cameras.json").write_text(json.dumps(merged) + "\n", encoding="utf-8")
    (staging / STAMP).write_text(json.dumps(stamp) + "\n", encoding="utf-8")
    if out.exists():
        shutil.rmtree(out)
    os.replace(staging, out)
    kept = sum(1 for name, cam in merged.items() if isinstance(cam, dict) and name not in swap)
    log(f"STATUS // {out.name}: {len(swap)} frames at {scale}x, {kept} views kept at the Expand size")
    return out
