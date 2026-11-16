"""Depth cache identity. Json and pathlib only, so the Qt app can import this."""

from __future__ import annotations

import json
from pathlib import Path


def scene_dir(scene: str) -> Path:
    name = (scene or "alpine_01").strip() or "alpine_01"
    for ch in '<>:"/\\|?*':
        name = name.replace(ch, "_")
    return Path(__file__).resolve().parents[2] / "outputs" / name


def depth_dir(scene: str) -> Path:
    return scene_dir(scene) / "depth"


def depth_key(scene: str) -> str | None:
    path = depth_dir(scene) / "depth.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    backend = str(data.get("backend") or "").strip()
    sha = str(data.get("pano_sha1") or "").strip()
    if not backend or not sha:
        return None
    return f"{backend}:{sha}"


def rail_key_path(scene: str, index: int) -> Path:
    """Not ``.json``: ``star_w2c`` globs ``rails/rail_*.json`` for poses."""
    return scene_dir(scene) / "rails" / f"rail_{int(index):02d}.depth_key"


def write_rail_key(scene: str, index: int, key: str) -> None:
    rail_key_path(scene, index).write_text(str(key).strip() + "\n", encoding="utf-8")


def _stored_rail_key(scene: str, index: int) -> str:
    path = rail_key_path(scene, index)
    if path.is_file():
        try:
            return path.read_text(encoding="utf-8").strip()
        except OSError:
            return ""
    meta = scene_dir(scene) / "control" / f"rail_{int(index):02d}" / "camplot.json"
    if not meta.is_file():
        return ""
    try:
        data = json.loads(meta.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    return str(data.get("depth_key") or "").strip() if isinstance(data, dict) else ""


def rails_stale(scene: str) -> list[int]:
    """Rail indexes whose ``rails/rail_XX.json`` was not plotted on the current depth.

    No cached depth means nothing is stale yet. Start over keeps ``rails/`` but
    deletes ``control/``, so the key lives beside the rail json.
    """
    current = depth_key(scene)
    if current is None:
        return []
    root = scene_dir(scene) / "rails"
    if not root.is_dir():
        return []
    stale: list[int] = []
    for path in sorted(root.glob("rail_*.json")):
        suffix = path.stem.removeprefix("rail_")
        if not suffix.isdigit():
            continue
        if _stored_rail_key(scene, int(suffix)) != current:
            stale.append(int(suffix))
    return stale
