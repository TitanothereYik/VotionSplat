"""Multi-rail editor state and on-disk paths for Phase 3.

P2 wrote a single ``rails/rail_00.json`` plus ``control/control_video.mp4``.
P3 gives each rail its own json and control folder so four rails cannot clobber
a shared ``condition_dir`` (SplatKit HiRes tooltip).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from engine.geo import defaults as D
from engine.geo.anchors import format_anchors, parse_anchors_ext, parse_point
from engine.geo.rail import catmull_rom, snap_length
from engine.pano.scene_io import scene_dir


def rail_json_rel(index: int) -> str:
    return f"rails/rail_{int(index):02d}.json"


def control_dir_rel(index: int) -> str:
    return f"control/rail_{int(index):02d}"


def control_video_rel(index: int) -> str:
    return f"{control_dir_rel(index)}/control_video.mp4"


def control_mask_rel(index: int) -> str:
    return f"{control_dir_rel(index)}/control_mask.mp4"


def camplot_meta_rel(index: int) -> str:
    return f"{control_dir_rel(index)}/camplot.json"


def condition_dir_rel(index: int) -> str:
    return f"condition/rail_{int(index):02d}"


def wan_video_rel(index: int) -> str:
    """WAN CreateVideo output. One file per rail so four rails cannot clobber."""
    return f"wan/rail_{int(index):02d}.mp4"


def wan_preview_dir_rel(index: int) -> str:
    return f"_work/wan_preview/rail_{int(index):02d}"


def hires_frames_rel() -> str:
    """Full-resolution HiRes composite PNGs (``core.hires_composite.run_composite``'s
    ``frames/``). Lives under the SfM scratch dir, not the dataset root — see
    ``hires_proxies_rel``."""
    return "hires/_spheresfm_work/frames"


def hires_proxies_rel() -> str:
    """Old ERP HiRes proxy path. The app no longer writes it."""
    return "hires/_spheresfm_work/proxies"


def dataset_path(scene: str) -> Path:
    return scene_dir(scene) / "_work" / "dataset.json"


def dropdown_label(index: int, orientation: str, extra: str = "") -> str:
    """Dummy: ``1 look_at_target · pond``. Extra is pond / suggest archetype."""
    n = int(index) + 1
    tag = (extra or "").strip()
    if tag:
        return f"{n} {orientation} · {tag}"
    return f"{n} {orientation}"


def first_knot_off_origin(anchors_text: str) -> bool:
    """True when the typed first line is not origin. Parser still locks knot 0 to (0,0,0)."""
    for line_raw in (anchors_text or "").splitlines():
        line = line_raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = [p for p in line.replace(",", " ").split() if p]
        if len(parts) < 3:
            return False
        try:
            x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
        except ValueError:
            return False
        return abs(x) > 1e-4 or abs(y) > 1e-4 or abs(z) > 1e-4
    return False


@dataclass
class RailSpec:
    index: int
    orientation: str = D.LOOK_DEFAULT
    anchors: str = D.ANCHORS_DEFAULT
    look_at: str = D.LOOK_AT_DEFAULT
    length: int = D.LENGTH_DEFAULT
    label: str = ""
    played: bool = False
    dirty: bool = True

    def payload(self, moge_level: int = D.MOGE_LEVEL) -> dict:
        return {
            "rail_index": int(self.index),
            "orientation": self.orientation,
            "length": snap_length(int(self.length)),
            "moge_level": int(moge_level),
            "anchors": self.anchors,
            "look_at": self.look_at,
            "label": self.label,
        }

    def combo_text(self) -> str:
        return dropdown_label(self.index, self.orientation, self.label)

    def knots_and_targets(self):
        return parse_anchors_ext(self.anchors)

    def look_point(self):
        return parse_point(self.look_at, (-0.108, 0.073, 1.953))


def straight_anchors(direction: tuple[float, float, float], length: float) -> str:
    vec = np.asarray(direction, dtype=np.float64)
    norm = float(np.linalg.norm(vec))
    if norm < 1e-8:
        raise ValueError("rail direction is zero")
    end = vec / norm * float(length)
    return f"0.000, 0.000, 0.000\n{end[0]:.3f}, {end[1]:.3f}, {end[2]:.3f}"


def straight_rail(index: int) -> RailSpec:
    src = D.STRAIGHT_RAILS[int(index) % len(D.STRAIGHT_RAILS)]
    return RailSpec(
        index=int(index),
        orientation="look_forward",
        anchors=straight_anchors(src["direction"], D.STRAIGHT_RAIL_LENGTH),
        look_at=D.LOOK_AT_DEFAULT,
        length=D.LENGTH_DEFAULT,
        label=str(src["label"]),
        played=False,
        dirty=True,
    )


def depth_along(points: np.ndarray, direction: np.ndarray, cos_min: float = 0.98) -> float | None:
    """Nearest cloud range within a narrow cone around ``direction``. None if the cone is empty."""
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] == 0:
        return None
    vec = np.asarray(direction, dtype=np.float64)
    norm = float(np.linalg.norm(vec))
    if norm < 1e-8:
        return None
    unit = vec / norm
    ranges = np.linalg.norm(pts, axis=1)
    keep = ranges > 0.15
    if not np.any(keep):
        return None
    dirs = pts[keep] / ranges[keep, None]
    hit = (dirs @ unit) >= float(cos_min)
    if not np.any(hit):
        return None
    return float(ranges[keep][hit].min())


def fit_straight_rails(rails: list[RailSpec], points: np.ndarray) -> None:
    """Shorten default straight rails to 80% of the free distance along their direction."""
    for spec in rails:
        if spec.orientation != "look_forward" or spec.label not in {row["label"] for row in D.STRAIGHT_RAILS}:
            continue
        direction = next(row["direction"] for row in D.STRAIGHT_RAILS if row["label"] == spec.label)
        free = depth_along(points, np.asarray(direction, dtype=np.float64))
        if free is None:
            continue
        length = min(D.STRAIGHT_RAIL_LENGTH, D.STRAIGHT_WALL_FRACTION * free)
        if length < 0.2:
            length = 0.2
        spec.anchors = straight_anchors(direction, length)


def rail_clips_wall(spec: RailSpec, points: np.ndarray) -> bool:
    """True when a sampled knot path goes past the cloud in that direction. Warn only."""
    try:
        anchors, _targets = parse_anchors_ext(spec.anchors)
    except ValueError:
        return False
    anchors = np.asarray(anchors, dtype=np.float64).copy()
    anchors[0] = 0.0
    samples = catmull_rom(anchors, 17)
    for point in samples:
        dist = float(np.linalg.norm(point))
        if dist < 0.05:
            continue
        free = depth_along(points, point)
        if free is not None and dist > free + 0.02:
            return True
    return False


def smpl_rail(index: int) -> RailSpec:
    src = D.SMPL_RAILS[index]
    return RailSpec(
        index=index,
        orientation=src["orientation"],
        anchors=src["anchors"],
        look_at=src.get("look_at") or D.LOOK_AT_DEFAULT,
        length=int(src.get("length") or D.LENGTH_DEFAULT),
        label=src.get("label") or "",
        played=False,
        dirty=True,
    )


def default_rails() -> list[RailSpec]:
    return [straight_rail(i) for i in range(D.DEFAULT_RAIL_COUNT)]


def next_free_rail_index(rails: list[RailSpec]) -> int | None:
    used = {int(r.index) for r in rails}
    for i in range(D.MAX_RAILS):
        if i not in used:
            return i
    return None


def editor_path(scene: str) -> Path:
    return scene_dir(scene) / "_work" / "rails_editor.json"


def confirmed_path(scene: str) -> Path:
    return scene_dir(scene) / "_work" / "confirmed.json"


def save_editor(scene: str, rails: list[RailSpec], moge_level: int, active: int) -> Path:
    root = scene_dir(scene)
    work = root / "_work"
    work.mkdir(parents=True, exist_ok=True)
    payload = {
        "moge_level": int(moge_level),
        "active": int(active),
        "rails": [
            {
                "index": r.index,
                "orientation": r.orientation,
                "anchors": r.anchors,
                "look_at": r.look_at,
                "length": r.length,
                "label": r.label,
                "played": r.played,
                "dirty": r.dirty,
            }
            for r in rails
        ],
    }
    path = work / "rails_editor.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def load_editor(scene: str) -> tuple[list[RailSpec], int, int] | None:
    path = editor_path(scene)
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data.get("rails")
    if not isinstance(rows, list) or len(rows) < D.MIN_RAILS:
        return None
    rails: list[RailSpec] = []
    seen: set[int] = set()
    for i, row in enumerate(rows[: D.MAX_RAILS]):
        idx = int(row.get("index", i))
        if idx < 0 or idx >= D.MAX_RAILS or idx in seen:
            continue
        seen.add(idx)
        rails.append(
            RailSpec(
                index=idx,
                orientation=str(row.get("orientation") or D.LOOK_DEFAULT),
                anchors=str(row.get("anchors") or D.ANCHORS_DEFAULT),
                look_at=str(row.get("look_at") or D.LOOK_AT_DEFAULT),
                length=snap_length(int(row.get("length") or D.LENGTH_DEFAULT)),
                label=str(row.get("label") or ""),
                played=bool(row.get("played")),
                dirty=bool(row.get("dirty", True)),
            )
        )
    if len(rails) < D.MIN_RAILS:
        return None
    rails.sort(key=lambda r: r.index)
    moge_level = int(data.get("moge_level") or D.MOGE_LEVEL)
    active = int(data.get("active") or 0)
    active = max(0, min(active, len(rails) - 1))
    return rails, moge_level, active


def seed_rails_for_scene(scene: str) -> list[RailSpec]:
    """Four straight rails, shortened to the cloud. A saved confirmed set is kept."""
    root = scene_dir(scene)
    confirmed = confirmed_path(scene)
    if confirmed.is_file():
        data = json.loads(confirmed.read_text(encoding="utf-8"))
        rows = data.get("rails")
        if isinstance(rows, list) and rows:
            rails: list[RailSpec] = []
            seen: set[int] = set()
            for i, row in enumerate(rows[: D.MAX_RAILS]):
                idx = int(row.get("rail_index", row.get("index", i)))
                if idx < 0 or idx >= D.MAX_RAILS or idx in seen:
                    continue
                seen.add(idx)
                rails.append(
                    RailSpec(
                        index=idx,
                        orientation=str(row.get("orientation") or D.LOOK_DEFAULT),
                        anchors=str(row.get("anchors") or D.ANCHORS_DEFAULT),
                        look_at=str(row.get("look_at") or D.LOOK_AT_DEFAULT),
                        length=snap_length(int(row.get("length") or D.LENGTH_DEFAULT)),
                        label=str(row.get("label") or ""),
                        played=True,
                        dirty=False,
                    )
                )
            if rails:
                rails.sort(key=lambda spec: spec.index)
                return rails
    rails = default_rails()
    cloud = root / "moge" / "cloud.npz"
    if cloud.is_file():
        points = np.load(cloud)["points"]
        if points.ndim == 2 and points.shape[0] >= 50:
            fit_straight_rails(rails, points)
    return rails


def write_confirmed(scene: str, rails: list[RailSpec], moge_level: int) -> Path:
    rows = []
    for r in rails:
        rows.append(
            {
                **r.payload(moge_level),
                "rail": rail_json_rel(r.index),
                "video": control_video_rel(r.index),
                "mask": control_mask_rel(r.index),
            }
        )
    payload = {
        "version": 3,
        "moge_level": int(moge_level),
        "rail": rows[0]["rail"] if rows else rail_json_rel(0),
        "video": rows[0]["video"] if rows else control_video_rel(0),
        "mask": rows[0]["mask"] if rows else control_mask_rel(0),
        "rails": rows,
    }
    path = confirmed_path(scene)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def confirmed_matches(scene: str, rails: list[RailSpec], moge_level: int) -> bool:
    path = confirmed_path(scene)
    if not path.is_file():
        return False
    data = json.loads(path.read_text(encoding="utf-8"))
    saved = data.get("rails")
    if not isinstance(saved, list) or len(saved) != len(rails):
        return False
    if int(data.get("moge_level") or D.MOGE_LEVEL) != int(moge_level):
        return False
    keys = ("orientation", "length", "anchors", "look_at")
    for spec, row in zip(rails, saved):
        if spec.dirty or not spec.played:
            return False
        for key in keys:
            if str(spec.payload(moge_level).get(key, "")).strip() != str(row.get(key, "")).strip():
                return False
    return True


def all_played(rails: list[RailSpec]) -> bool:
    return bool(rails) and all(r.played and not r.dirty for r in rails)


def empty_dataset_row(index: int) -> dict:
    return {
        "index": int(index),
        "keep": True,
        "mask": control_mask_rel(index),
        "video": control_video_rel(index),
        "rail": rail_json_rel(index),
        "status": "queued",
    }


def load_dataset(scene: str) -> dict:
    path = dataset_path(scene)
    if path.is_file():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict) and isinstance(data.get("rails"), list):
                return data
        except json.JSONDecodeError:
            pass
    return {"version": 1, "rails": []}


def save_dataset(scene: str, data: dict) -> Path:
    path = dataset_path(scene)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return path


def sync_dataset_from_confirmed(scene: str) -> dict:
    """Keep ditch/WAN flags; add/drop rows to match confirmed Geometry rails."""
    data = load_dataset(scene)
    prev = {int(row.get("index", i)): row for i, row in enumerate(data.get("rails") or [])}
    gate = confirmed_path(scene)
    rows = []
    if gate.is_file():
        payload = json.loads(gate.read_text(encoding="utf-8"))
        saved = payload.get("rails")
        if isinstance(saved, list) and saved:
            for i, src in enumerate(saved):
                idx = int(src.get("index", i))
                row = empty_dataset_row(idx)
                old = prev.get(idx) or {}
                row["keep"] = bool(old.get("keep", True))
                row["status"] = str(old.get("status") or "queued")
                if src.get("video"):
                    row["video"] = str(src["video"])
                if src.get("mask"):
                    row["mask"] = str(src["mask"])
                if src.get("rail"):
                    row["rail"] = str(src["rail"])
                rows.append(row)
    if not rows:
        rows = [empty_dataset_row(i) for i in range(D.DEFAULT_RAIL_COUNT)]
    data["version"] = 1
    data["rails"] = rows
    save_dataset(scene, data)
    return data


def set_dataset_rail(scene: str, index: int, **fields) -> dict:
    data = sync_dataset_from_confirmed(scene)
    found = False
    for row in data["rails"]:
        if int(row.get("index", -1)) == int(index):
            row.update(fields)
            found = True
            break
    if not found:
        row = empty_dataset_row(index)
        row.update(fields)
        data["rails"].append(row)
        data["rails"].sort(key=lambda r: int(r.get("index", 0)))
    save_dataset(scene, data)
    return data


def kept_rail_indices(scene: str) -> list[int]:
    data = sync_dataset_from_confirmed(scene)
    return [int(row["index"]) for row in data.get("rails") or [] if row.get("keep", True)]


def suggestion_to_rail(index: int, item: dict) -> RailSpec:
    import numpy as np

    from engine.geo.anchors import format_point

    orientation = str(item.get("orientation") or "look_forward")
    if orientation not in D.LOOK_MODES:
        orientation = "look_forward"
    anchors = np.asarray(item["anchors"], dtype=np.float64)
    anchors[0] = 0.0
    raw_tgts = item.get("targets") or [None] * len(anchors)
    tgt = np.full_like(anchors, np.nan)
    for i, t in enumerate(raw_tgts):
        if t is not None:
            tgt[i] = np.asarray(t, dtype=np.float64)
    six = orientation == "per_point_look"
    text = format_anchors(anchors, tgt, six=six)
    look_at = D.LOOK_AT_DEFAULT
    if six and raw_tgts and raw_tgts[0] is not None:
        look_at = format_point(raw_tgts[0])
    label = str(item.get("label") or "")
    return RailSpec(
        index=index,
        orientation=orientation,
        anchors=text,
        look_at=look_at,
        length=D.LENGTH_DEFAULT,
        label=label,
        played=False,
        dirty=True,
    )
