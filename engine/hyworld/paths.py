"""Paths under outputs/<scene>/hyworld. Safe for the Qt process."""

from __future__ import annotations

import json
from pathlib import Path

from engine.pano.scene_io import prompt_path, scene_dir


def hy_root(scene: str) -> Path:
    path = scene_dir(scene) / "hyworld"
    path.mkdir(parents=True, exist_ok=True)
    return path


def votion_dir(scene: str) -> Path:
    path = hy_root(scene) / "_votion"
    path.mkdir(parents=True, exist_ok=True)
    return path


def selection_path(scene: str) -> Path:
    return votion_dir(scene) / "selection.json"


def render_root(scene: str) -> Path:
    return hy_root(scene) / "render_results"


def gs_data(scene: str) -> Path:
    return hy_root(scene) / "gs_data"


def gs_result(scene: str) -> Path:
    return hy_root(scene) / "gs_result"


def read_selection(scene: str) -> list[str] | None:
    """Kept trajectory ids such as ``view0/traj0``. None means keep every trajectory."""
    path = selection_path(scene)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    kept = payload.get("keep") if isinstance(payload, dict) else None
    if not isinstance(kept, list):
        return None
    return [str(item) for item in kept]


def write_selection(scene: str, keep: list[str]) -> Path:
    path = selection_path(scene)
    path.write_text(json.dumps({"keep": list(keep)}, indent=2) + "\n", encoding="utf-8")
    return path


def list_trajectories(scene: str) -> list[str]:
    """``view0/traj0`` ids that already have a camera.json."""
    root = render_root(scene)
    found: list[str] = []
    if not root.is_dir():
        return found
    for view in sorted(path for path in root.iterdir() if path.is_dir()):
        name = view.name
        if not (
            name.startswith("view")
            or name.startswith("target")
            or name.startswith("wonder")
            or name.startswith("reconstruct")
        ):
            continue
        for traj in sorted(view.glob("traj*")):
            if (traj / "camera.json").is_file():
                found.append(f"{name}/{traj.name}")
    return found


def apply_budget(ids: list[str], budget: int) -> list[str]:
    """Keep every origin view, and the first ``budget`` views of each other kind with all their paths."""
    kept: list[str] = []
    seen: dict[str, list[str]] = {}
    for item in ids:
        view = item.split("/", 1)[0]
        if view.startswith("view"):
            kept.append(item)
            continue
        views = seen.setdefault(view.split("_", 1)[0], [])
        if view not in views:
            if len(views) >= int(budget):
                continue
            views.append(view)
        kept.append(item)
    return kept


_VIEW_RANK = {"traj2": 0, "traj0": 1, "traj1": 2}


def ordered_ids(ids: list[str]) -> list[str]:
    """HY-World generation order (worldgen/src/data_utils.py sort_trajs)."""
    views, targets, recon0, wonders, recon1 = [], [], [], [], []
    for item in ids:
        view, traj = item.split("/", 1)
        if view.startswith("view"):
            if traj in _VIEW_RANK:
                views.append(((view, _VIEW_RANK[traj]), item))
        elif view.startswith("target"):
            targets.append(((view, traj), item))
        elif view.startswith("reconstruct"):
            if traj == "traj0":
                recon0.append(((view, traj), item))
            elif traj == "traj1":
                recon1.append(((view, traj), item))
        elif view.startswith("wonder"):
            wonders.append(((view, traj), item))
    out: list[str] = []
    for group in (views, targets, recon0, wonders, recon1):
        out.extend(item for _key, item in sorted(group))
    return out


def close_selection(keep: list[str], every: list[str]) -> list[str]:
    """A later path of a wonder / target / reconstruct view needs that view's traj0."""
    chosen = set(keep)
    for item in list(chosen):
        view, traj = item.split("/", 1)
        parent = f"{view}/traj0"
        if traj != "traj0" and not view.startswith("view") and parent in every:
            chosen.add(parent)
    return [item for item in ordered_ids(every) if item in chosen]


RESULT_NAME = "worldstereo-memory-dmd"


def traj_state(scene: str, item: str) -> str:
    """planned, guide (render.mp4 exists) or expanded (result mp4 exists)."""
    folder = render_root(scene) / item
    if (folder / f"{RESULT_NAME}_result.mp4").is_file():
        return "expanded"
    if (folder / "render.mp4").is_file():
        return "guide"
    return "planned"


def latest_checkpoint(scene: str) -> Path | None:
    folder = gs_result(scene) / "ckpts"
    if not folder.is_dir():
        return None
    found = sorted(folder.glob("ckpt_*_rank0.pt"), key=lambda path: path.stat().st_mtime)
    return found[-1] if found else None


def camera_centers(scene: str, item: str):
    """Camera centres of one path in the HY world frame, shape [N, 3]."""
    import numpy as np

    data = json.loads((render_root(scene) / item / "camera.json").read_text(encoding="utf-8"))
    w2c = np.asarray(data["extrinsic"], dtype=np.float64)
    return -np.einsum("nji,nj->ni", w2c[:, :3, :3], w2c[:, :3, 3])


def copy_prompt(scene: str) -> Path:
    """Stage 2's caption fallback reads prompt.txt beside the panorama."""
    source = prompt_path(scene)
    dest = hy_root(scene) / "prompt.txt"
    text = source.read_text(encoding="utf-8") if source.is_file() else ""
    dest.write_text(text, encoding="utf-8")
    return dest
