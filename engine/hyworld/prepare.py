"""Copy the 8K panorama into the HY-World scene folder. No torch."""

from __future__ import annotations

import json
import shutil
import time

import cv2

from engine.hyworld.paths import copy_prompt, hy_root
from engine.pano.scene_io import pano_path


_ARCHIVED = ("render_results", "navmesh", "objects.json", "paths.json", "vis_mesh.ply")


def _archive(root) -> None:
    """Move the old plan aside. Nothing is deleted."""
    dest = root / "_votion" / "archive" / time.strftime("%Y%m%d_%H%M%S")
    moved = 0
    for name in (*_ARCHIVED, "_votion/selection.json", "_votion/preview"):
        source = root / name
        if source.exists():
            target = dest / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(target))
            moved += 1
    if moved:
        print(f"STATUS // old plan archived to {dest}", flush=True)


def prepare_scene(scene: str, scene_type: str = "", fresh: bool = False) -> None:
    """Write panorama.png at 3840x1920, the prompt, and the scene type. Runs in the worker."""
    source = pano_path(scene)
    if not source.is_file():
        raise FileNotFoundError(f"missing {source}")
    root = hy_root(scene)
    kind = (scene_type or "").strip().lower()
    meta = root / "meta_info.json"
    old = ""
    if meta.is_file():
        try:
            old = str(json.loads(meta.read_text(encoding="utf-8")).get("scene_type") or "")
        except json.JSONDecodeError:
            old = ""
    if kind in {"indoor", "outdoor"} and old and old != kind:
        fresh = True
    if fresh:
        _archive(root)
    image = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"failed to read {source}")
    image = cv2.resize(image, (3840, 1920), interpolation=cv2.INTER_AREA)
    if not cv2.imwrite(str(root / "panorama.png"), image):
        raise RuntimeError("failed to write the HY panorama")
    copy_prompt(scene)
    if kind in {"indoor", "outdoor"}:
        meta.write_text(json.dumps({"scene_type": kind}, indent=2) + "\n", encoding="utf-8")
    elif meta.is_file():
        meta.unlink()
    print(f"STATUS // hyworld scene ready {root}", flush=True)
