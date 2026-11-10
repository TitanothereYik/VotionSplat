"""COLMAP model for Reconstruct and the splat trainer.

Always ``worldstereo/dataset`` and ``worldstereo/train``. Old WAN ``colmap/``
folders stay on disk and are not read. A legacy ``rails/NN_<mode>/`` folder
counts as the front view.
"""

from __future__ import annotations

import json
from pathlib import Path

from engine.pano.scene_io import scene_dir
from engine.worldstereo.views import VIEWS

SOURCE_KNOWN = "worldstereo"


def _scene_path(scene: str | Path) -> Path:
    path = Path(scene)
    if path.is_dir() and (
        (path / "worldstereo").is_dir() or (path / "_work").is_dir() or (path / "pano.png").is_file()
    ):
        return path
    return scene_dir(str(scene))


def source_file(scene: str | Path) -> Path:
    return _scene_path(scene) / "_work" / "dataset_source.json"


def read_source(scene: str | Path) -> str:
    """Kept so older files still parse. The product no longer switches sources."""
    path = source_file(scene)
    if not path.is_file():
        return SOURCE_KNOWN
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return SOURCE_KNOWN
    return str(data.get("source") or SOURCE_KNOWN)


def write_source(scene: str | Path, source: str = SOURCE_KNOWN) -> None:
    path = source_file(scene)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"source": SOURCE_KNOWN}) + "\n", encoding="utf-8")


def known_selected(scene: str | Path) -> bool:
    return True


def colmap_root(scene: str | Path) -> Path:
    return _scene_path(scene) / "worldstereo" / "dataset"


def output_root(scene: str | Path) -> Path:
    """Where splat checkpoints and splat.ply are written."""
    return _scene_path(scene) / "worldstereo" / "train"


def view_folders(scene: str | Path, index: int) -> dict[str, Path]:
    """View name to folder. New layout is ``rail_NN/<view>``. Legacy ``NN_<mode>`` is front."""
    rails = _scene_path(scene) / "worldstereo" / "rails"
    if not rails.is_dir():
        return {}
    found: dict[str, Path] = {}
    nested = rails / f"rail_{int(index):02d}"
    if nested.is_dir():
        for view in VIEWS:
            folder = nested / view
            if (folder / "keyframes").is_dir() or (folder / "camera.json").is_file():
                found[view] = folder
    if found:
        return found
    prefix = f"{int(index):02d}_"
    hits = sorted(path for path in rails.iterdir() if path.is_dir() and path.name.startswith(prefix))
    if hits:
        return {"front": hits[0]}
    return {}


def rail_folder(scene: str | Path, index: int) -> Path | None:
    folders = view_folders(scene, index)
    if "front" in folders:
        return folders["front"]
    return next(iter(folders.values()), None)


def keyframe_paths(rail: Path) -> list[Path]:
    folder = rail / "keyframes"
    if not folder.is_dir():
        return []
    return sorted(folder.glob("[0-9][0-9][0-9].png"))


def keyframe_count(scene: str | Path, index: int) -> int:
    """Keyframes on disk for one rail."""
    return sum(len(keyframe_paths(folder)) for folder in view_folders(scene, index).values())
