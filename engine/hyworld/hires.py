"""HiRes for expanded HY-World clips with SeedVR2 3B. No torch: the Qt process imports this."""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

from engine.hyworld.paths import RESULT_NAME, render_root, votion_dir
from engine.job import REPO_ROOT

PYTHON = REPO_ROOT / ".venvs" / "seedvr2" / "Scripts" / "python.exe"
VENDOR = REPO_ROOT / "vendor" / "seedvr2"
MODEL_DIR = REPO_ROOT / "models" / "SEEDVR2"
DIT = "seedvr2_ema_3b_fp8_e4m3fn.safetensors"
VAE = "ema_vae_fp16.safetensors"
FIT_FILE = REPO_ROOT / "models" / "hires_fit.json"
SCALE = 2
SCALES = (2, 4)
SCALE_NAMES = {2: "2x (1664x960)", 4: "4x (3328x1920)"}
SEED = 42
COLOR = "lab"
FPS = 16
PREVIEW_LONG_SIDE = 1664  # the 4x preview plays in the same small pane as 2x
# tools/probe_seedvr2_cap.py: cudaMallocAsync enforced the cap; expandable_segments is unsupported here.
ALLOC_CONF = "backend:cudaMallocAsync"
CAP_WORKS = True

LEVELS = ("whole", "t1024", "t768", "t512")
CHOICES = ("auto", *LEVELS)
# 4x: the DiT cannot hold 21 frames at 3328x1920 on 24 GB (out of memory even at t512), so
# 4x levels shrink the DiT batch instead. Overlaps split 21 frames into equal batches.
LEVELS_X4 = ("b17", "b13", "b9", "b5")
CHOICES_X4 = ("auto", *LEVELS_X4)
ALL_CHOICES = ("auto", *LEVELS, *LEVELS_X4)
NAMES = {
    "auto": "Auto (fit this machine)",
    "whole": "Whole frame",
    "t1024": "Decode tiles 1024",
    "t768": "Decode 768, encode 1024",
    "t512": "Decode 512, encode 768",
    "b17": "17-frame batches",
    "b13": "13-frame batches",
    "b9": "9-frame batches",
    "b5": "5-frame batches",
}
# (decode tile, decode overlap, encode tile, encode overlap) in output pixels. 0 is untiled.
TILES = {
    "whole": (0, 0, 0, 0),
    "t1024": (1024, 128, 0, 0),
    "t768": (768, 96, 1024, 128),
    "t512": (512, 64, 768, 96),
}
TILES_X4 = (768, 96, 768, 96)
# (batch_size, temporal_overlap)
BATCH_X4 = {"b17": (17, 13), "b13": (13, 5), "b9": (9, 3), "b5": (5, 1)}
# Reserved card GB for one 21-frame 1664x960 clip before anything is measured. RTX 3090
# untiled: encode 20.1 reserved, decode 26.9 allocated and spilled. Tiled rows scale by tile area.
WORK_GB = {"whole": 27.5, "t1024": 20.5, "t768": 13.0, "t512": 8.0}
WORK_GB_X4 = {"b17": 23.0, "b13": 19.0, "b9": 15.0, "b5": 11.0}
HEADROOM_GB = 1.0
RESERVE_GB = 0.75


def read_rows(key: str) -> dict:
    try:
        data = json.loads(FIT_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    rows = data.get(key) if isinstance(data, dict) else None
    return rows if isinstance(rows, dict) else {}


def write_rows(key: str, rows: dict) -> None:
    try:
        data = json.loads(FIT_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        data = {}
    if not isinstance(data, dict):
        data = {}
    data[key] = rows
    FIT_FILE.parent.mkdir(parents=True, exist_ok=True)
    FIT_FILE.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def levels(scale: int = SCALE) -> tuple[str, ...]:
    return LEVELS_X4 if scale == 4 else LEVELS


def choices(scale: int = SCALE) -> tuple[str, ...]:
    return CHOICES_X4 if scale == 4 else CHOICES


def tiles(level: str, scale: int = SCALE) -> tuple[int, int, int, int]:
    return TILES_X4 if scale == 4 else TILES[level]


def batch(level: str, scale: int = SCALE) -> tuple[int, int]:
    return BATCH_X4[level] if scale == 4 else (21, 0)


def fit_key(gpu: str, total_gb: float, scale: int = SCALE) -> str:
    """2x keeps the original key so its saved rows stay valid."""
    key = f"{gpu}|{total_gb:.0f}"
    return key if scale == SCALE else f"{key}|x{scale}"


def need_gb(level: str, rows: dict, scale: int = SCALE) -> float:
    """cudaMallocAsync can fill its pool up to the cap, so a budget that already passed bounds the need."""
    row = rows.get(level) or {}
    if "reserved_gb" in row:
        need = float(row["reserved_gb"]) + 0.25
        if "ok_gb" in row:
            need = min(need, float(row["ok_gb"]))
        return need
    return (WORK_GB_X4 if scale == 4 else WORK_GB)[level] + HEADROOM_GB


def fits(level: str, budget_gb: float, rows: dict, scale: int = SCALE) -> bool:
    """False when this budget already failed, else when the measured or estimated need fits."""
    row = rows.get(level) or {}
    failed = row.get("fail_gb")
    if failed is not None and budget_gb <= float(failed) + 0.5:
        return False
    return need_gb(level, rows, scale) <= budget_gb


def pick(budget_gb: float, rows: dict, asked: str, scale: int = SCALE) -> str:
    """The asked level, or the first one that should fit. The last one when none does."""
    ladder = levels(scale)
    if asked != "auto":
        if asked not in ladder:
            raise ValueError(f"level {asked} is not a {scale}x level")
        return asked
    for level in ladder:
        if fits(level, budget_gb, rows, scale):
            return level
    return ladder[-1]


def next_level(level: str, scale: int = SCALE) -> str | None:
    ladder = levels(scale)
    index = ladder.index(level)
    if index + 1 >= len(ladder):
        return None
    return ladder[index + 1]


def record(
    rows: dict, level: str, seconds: float, peak_gb: float, reserved_gb: float, budget_gb: float = 0.0
) -> dict:
    row = dict(rows.get(level) or {})
    row["peak_gb"] = round(max(float(row.get("peak_gb") or 0.0), float(peak_gb)), 2)
    row["reserved_gb"] = round(max(float(row.get("reserved_gb") or 0.0), float(reserved_gb)), 2)
    row["seconds"] = round(float(seconds), 1)
    if budget_gb > 0:
        row["ok_gb"] = round(min(float(row.get("ok_gb") or budget_gb), float(budget_gb)), 2)
    failed = row.get("fail_gb")
    if failed is not None and float(reserved_gb) + 0.25 < float(failed):
        del row["fail_gb"]
    rows[level] = row
    return rows


def fail(rows: dict, level: str, budget_gb: float) -> dict:
    row = dict(rows.get(level) or {})
    row["fail_gb"] = round(max(float(row.get("fail_gb") or 0.0), float(budget_gb)), 2)
    rows[level] = row
    return rows


def _folder(scale: int) -> str:
    """2x keeps the original ``hires`` folder; other scales get their own."""
    return "hires" if scale == SCALE else f"hires_x{scale}"


def hires_root(scene: str, scale: int = SCALE) -> Path:
    return votion_dir(scene) / _folder(scale)


def hires_dir(scene: str, item: str, scale: int = SCALE) -> Path:
    return hires_root(scene, scale) / item


def bench_dir(scene: str, level: str, item: str, scale: int = SCALE) -> Path:
    return votion_dir(scene) / "bench" / _folder(scale) / level / item


def source_clip(scene: str, item: str) -> Path:
    return render_root(scene) / item / f"{RESULT_NAME}_result.mp4"


def preview_path(scene: str, item: str, scale: int = SCALE) -> Path:
    return hires_dir(scene, item, scale) / "preview.mp4"


def stamp(path: Path) -> dict:
    stat = Path(path).stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def folder_state(folder: Path, source: Path) -> str:
    """``none`` without meta, ``done`` when 21 frames match the source, else ``stale``."""
    try:
        meta = json.loads((Path(folder) / "meta.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "none"
    frames = sorted((Path(folder) / "frames").glob("*.png"))
    source = Path(source)
    if not source.is_file():
        return "stale"
    try:
        stat = source.stat()
    except OSError:
        return "none"
    matched = (
        len(frames) == 21
        and int(meta.get("size", -1)) == stat.st_size
        and int(meta.get("mtime_ns", -1)) == stat.st_mtime_ns
    )
    return "done" if matched else "stale"


def hires_state(scene: str, item: str, scale: int = SCALE) -> str:
    return folder_state(hires_dir(scene, item, scale), source_clip(scene, item))


def _rename(src: Path, dest: Path) -> None:
    """Windows refuses a rename while a viewer holds a file inside the folder."""
    for attempt in range(10):
        try:
            src.rename(dest)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(0.5)


def publish(folder: Path, pngs: list[Path], meta: dict) -> None:
    """Move frames into ``folder`` via a sibling ``.new``, then swap it into place."""
    folder = Path(folder)
    staging = folder.parent / (folder.name + ".new")
    previous = folder.parent / (folder.name + ".old")
    if staging.exists():
        shutil.rmtree(staging)
    frames = staging / "frames"
    frames.mkdir(parents=True)
    for index, src in enumerate(sorted(Path(path) for path in pngs)):
        shutil.move(str(src), str(frames / f"{index:03d}.png"))
    import cv2

    written = sorted(frames.glob("*.png"))
    first = cv2.imread(str(written[0]), cv2.IMREAD_COLOR)
    if first is None:
        raise RuntimeError(f"HIRES // failed to read {written[0]}")
    height, width = first.shape[:2]
    shrink = min(1.0, PREVIEW_LONG_SIDE / float(max(width, height)))
    size = (int(round(width * shrink)), int(round(height * shrink)))
    preview = staging / "preview.mp4"
    writer = cv2.VideoWriter(str(preview), cv2.VideoWriter_fourcc(*"mp4v"), FPS, size)
    if not writer.isOpened():
        raise RuntimeError(f"HIRES // failed to open {preview}")
    try:
        for path in written:
            frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if frame is None or frame.shape[1] != width or frame.shape[0] != height:
                raise RuntimeError(f"HIRES // bad preview frame {path}")
            if shrink < 1.0:
                frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
            writer.write(frame)
    finally:
        writer.release()
    (staging / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    if previous.exists():
        shutil.rmtree(previous)
    if folder.exists():
        _rename(folder, previous)
    _rename(staging, folder)
    if previous.exists():
        shutil.rmtree(previous)


def clean_scratch(scene: str, scale: int = SCALE) -> None:
    """Drop leftover scratch under the hires root only."""
    root = hires_root(scene, scale)
    if not root.is_dir():
        return
    root = root.resolve()
    tmp = root / "_tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    parents = [root]
    parents.extend(
        path for path in root.iterdir()
        if path.is_dir() and path.name != "_tmp" and not path.name.endswith((".new", ".old"))
    )
    for parent in parents:
        for child in list(parent.iterdir()):
            if not child.is_dir() or not child.name.endswith((".new", ".old")):
                continue
            child.resolve().relative_to(root)
            shutil.rmtree(child)
