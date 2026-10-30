"""Which COLMAP model Reconstruct and the splat trainer use.

Torch-free so the Qt process can import it. Missing file means SphereSfM (WAN).
Known poses live under ``worldstereo/dataset`` and train under ``worldstereo/train``.
The WAN ``colmap/`` folder and ``splat.ply`` are not those paths.
"""

from __future__ import annotations

import json
from pathlib import Path

from engine.pano.scene_io import scene_dir

SOURCE_WAN = "wan"
SOURCE_KNOWN = "worldstereo"


def _scene_path(scene: str | Path) -> Path:
    path = Path(scene)
    if path.is_dir() and (
        (path / "worldstereo").is_dir() or (path / "colmap").is_dir() or (path / "_work").is_dir()
    ):
        return path
    return scene_dir(str(scene))


def source_file(scene: str | Path) -> Path:
    return _scene_path(scene) / "_work" / "dataset_source.json"


def read_source(scene: str | Path) -> str:
    path = source_file(scene)
    if not path.is_file():
        return SOURCE_WAN
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return SOURCE_WAN
    source = str(data.get("source") or SOURCE_WAN)
    return source if source == SOURCE_KNOWN else SOURCE_WAN


def write_source(scene: str | Path, source: str) -> None:
    chosen = SOURCE_KNOWN if source == SOURCE_KNOWN else SOURCE_WAN
    path = source_file(scene)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"source": chosen}) + "\n", encoding="utf-8")


def known_selected(scene: str | Path) -> bool:
    return read_source(scene) == SOURCE_KNOWN


def colmap_root(scene: str | Path) -> Path:
    root = _scene_path(scene)
    if known_selected(root):
        return root / "worldstereo" / "dataset"
    return root / "colmap"


def output_root(scene: str | Path) -> Path:
    """Where splat checkpoints and splat.ply are written."""
    root = _scene_path(scene)
    if known_selected(root):
        return root / "worldstereo" / "train"
    return root


def rail_folder(scene: str | Path, index: int) -> Path | None:
    rails = _scene_path(scene) / "worldstereo" / "rails"
    if not rails.is_dir():
        return None
    prefix = f"{int(index):02d}_"
    hits = sorted(path for path in rails.iterdir() if path.is_dir() and path.name.startswith(prefix))
    return hits[0] if hits else None


def keyframe_paths(rail: Path) -> list[Path]:
    folder = rail / "keyframes"
    if not folder.is_dir():
        return []
    return sorted(folder.glob("[0-9][0-9][0-9].png"))


def keyframe_count(scene: str | Path, index: int) -> int:
    rail = rail_folder(scene, index)
    if rail is None:
        return 0
    return len(keyframe_paths(rail))
