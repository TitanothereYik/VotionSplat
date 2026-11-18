"""Train / Continue / Difix3D+ bookkeeping. No torch: the Qt process imports this."""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

import numpy as np

from engine.job import REPO_ROOT

DIFIX_PY = REPO_ROOT / ".venvs" / "difix" / "Scripts" / "python.exe"
DIFIX_SRC = REPO_ROOT / "vendor" / "difix3d" / "src"
DIFIX_MODEL = "nvidia/difix_ref"
DIFIX_FRACTIONS = (0.2, 0.3, 0.4, 0.55, 0.7)  # densification stops at 0.5
DIFIX_VIEWS = 48
DIFIX_LAMBDA = 0.3  # loss weight of fixed views (paper value)
AUTOSAVE_SEC = 600
NOVEL_PREFIX = "zdifix_"  # sorts after every real view, so holdout indices stay put


def trainer_flags(data_dir: str, result_dir: str, steps: int, mesh: bool) -> list[str]:
    """The world_gs_trainer argv that Train used to build inline."""
    count = int(steps)
    flags = [
        "default",
        "--data-dir", str(data_dir).replace("\\", "/"),
        "--result-dir", str(result_dir).replace("\\", "/"),
        "--max-steps", str(count),
        "--save-steps", str(count),
        "--eval-steps", str(count),
        "--ply-steps", str(count),
        "--save-ply",
        "--disable-video",
        "--disable-viewer",
        "--use-scale-regularization",
        "--antialiased",
        "--depth-loss",
        "--normal-loss",
        "--sky-depth-from-pcd",
        "--use-mask-gaussian",
        "--mask-export-stochastic",
        "--no-mask-export-anchor-protection",
        "--use-anchor-protection",
        "--strategy.refine-start-iter", str(max(1, count // 10)),
        "--strategy.refine-stop-iter", str(max(1, count // 2)),
        "--strategy.refine-every", str(max(1, count // 15)),
        "--strategy.refine-scale2d-stop-iter", str(max(1, count // 2)),
        "--strategy.reset-every", "99990",
        "--strategy.grow-grad2d", "0.0001",
        "--strategy.prune-scale3d", "0.1",
    ]
    if mesh:
        flags.append("--export_mesh")
    return flags


def fix_steps(max_steps: int) -> list[int]:
    """Steps where a Difix round runs. Each value is strictly before the last step."""
    steps = {max(1, int(max_steps * fraction)) for fraction in DIFIX_FRACTIONS}
    return sorted(step for step in steps if step < max_steps - 1)


def data_stamp(gs_data_dir: str | Path) -> dict:
    stat = (Path(gs_data_dir) / "cameras.json").stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _run_path(result_dir: Path) -> Path:
    return Path(result_dir) / "votion_run.json"


def new_run(
    result_dir: str | Path,
    *,
    steps: int,
    mesh: bool,
    difix: bool,
    views: int,
    data_dir: str | Path,
) -> dict:
    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    run = {
        "run_id": time.strftime("%Y%m%d_%H%M%S"),
        "max_steps": int(steps),
        "mesh": bool(mesh),
        "difix": bool(difix),
        "views": int(views),
        "fix_steps": fix_steps(int(steps)) if difix else [],
        "base_data_dir": str(Path(data_dir)),
        "data_stamp": data_stamp(data_dir),
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "rounds_done": 0,
    }
    write_run(result_dir, run)
    return run


def read_run(result_dir: str | Path) -> dict | None:
    path = _run_path(result_dir)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) and data.get("run_id") else None


def write_run(result_dir: str | Path, run: dict) -> None:
    path = _run_path(result_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(run, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def resume_pt(result_dir: str | Path, run: dict) -> Path:
    return Path(result_dir) / f"votion_resume_{run['run_id']}.pt"


def resume_meta(result_dir: str | Path, run: dict) -> Path:
    return Path(result_dir) / f"votion_resume_{run['run_id']}.json"


def frame_json(result_dir: str | Path, run: dict) -> Path:
    return Path(result_dir) / f"votion_frame_{run['run_id']}.json"


def state_json(result_dir: str | Path) -> Path:
    return Path(result_dir) / "votion_state.json"


def round_dir(result_dir: str | Path, run: dict, k: int) -> Path:
    return Path(result_dir) / "difix" / str(run["run_id"]) / f"round_{k}"


def merged_dir(result_dir: str | Path, run: dict, k: int) -> Path:
    return Path(result_dir) / "difix_data" / f"{run['run_id']}_r{k}"


def _read_json(path: Path) -> dict | None:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def continue_info(result_dir: str | Path) -> dict:
    """Whether Continue can pick up this result dir, and from which step."""
    empty = {"ok": False, "reason": "No run to continue.", "next_step": 0, "max_steps": 0, "difix": False}
    run = read_run(result_dir)
    if run is None:
        return empty
    info = {
        "ok": False,
        "reason": "",
        "next_step": 0,
        "max_steps": int(run.get("max_steps") or 0),
        "difix": bool(run.get("difix")),
    }
    state = _read_json(state_json(result_dir)) or {}
    if state.get("reason") == "done":
        info["reason"] = "The last run finished. Press Train."
        return info
    meta = _read_json(resume_meta(result_dir, run))
    if not meta or "next_step" not in meta:
        info["reason"] = "Nothing saved yet."
        return info
    try:
        stamp = data_stamp(run["base_data_dir"])
    except OSError:
        stamp = None
    if stamp != run.get("data_stamp"):
        info["reason"] = "Dataset changed since this run started. Press Train."
        return info
    info["ok"] = True
    info["next_step"] = int(meta["next_step"])
    return info


def _png_size(path: Path) -> tuple[int, int]:
    data = Path(path).read_bytes()[:24]
    if data[:8] != b"\x89PNG\r\n\x1a\n" or data[12:16] != b"IHDR":
        raise ValueError(f"not a png: {path}")
    return int.from_bytes(data[16:20], "big"), int.from_bytes(data[20:24], "big")


def _skew(k: np.ndarray) -> np.ndarray:
    x, y, z = (float(v) for v in k)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)


def rot_log(R: np.ndarray) -> np.ndarray:
    """Axis-angle vector of a rotation matrix."""
    R = np.asarray(R, dtype=np.float64)
    cos = float(np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0))
    theta = float(np.arccos(cos))
    if theta < 1e-8:
        return np.zeros(3, dtype=np.float64)
    if theta > np.pi - 1e-4:
        axis = np.sqrt(np.maximum((np.diag(R) + 1.0) * 0.5, 0.0))
        if R[0, 1] < 0.0:
            axis[1] = -abs(axis[1])
        if R[0, 2] < 0.0:
            axis[2] = -abs(axis[2])
        norm = float(np.linalg.norm(axis))
        if norm < 1e-8:
            axis = np.array([1.0, 0.0, 0.0])
        else:
            axis = axis / norm
        return axis * theta
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]], dtype=np.float64)
    w = w / (2.0 * np.sin(theta))
    return w * theta


def rot_exp(v: np.ndarray) -> np.ndarray:
    """Rotation matrix of an axis-angle vector."""
    v = np.asarray(v, dtype=np.float64)
    theta = float(np.linalg.norm(v))
    if theta < 1e-8:
        return np.eye(3, dtype=np.float64)
    k = v / theta
    K = _skew(k)
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def _ortho(R: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(np.asarray(R, dtype=np.float64))
    out = U @ Vt
    if np.linalg.det(out) < 0.0:
        U = U.copy()
        U[:, -1] *= -1.0
        out = U @ Vt
    return out


def _group(name: str) -> str:
    return name.rsplit("_", 1)[0]


def novel_poses(
    cameras: dict, image_dir: str | Path, *, round_k: int, count: int, seed: int
) -> list[dict]:
    """Cameras halfway between paths, in the original (unnormalised) frame."""
    image_dir = Path(image_dir)
    rows = []
    for name, cam in cameras.items():
        if not isinstance(cam, dict) or "extrinsic" not in cam:
            continue
        if name.startswith(("panorama_", "polar_", NOVEL_PREFIX)):
            continue
        extrinsic = np.asarray(cam["extrinsic"], dtype=np.float64)
        c2w = np.linalg.inv(extrinsic)
        rows.append((str(name), c2w, cam["intrinsic"]))
    rng = np.random.default_rng(int(seed))
    order = rng.permutation(len(rows))
    t = min(0.25 * int(round_k), 0.5)
    made: list[dict] = []
    for index in order:
        if len(made) >= int(count):
            break
        name_a, c2w_a, K = rows[int(index)]
        centre_a = c2w_a[:3, 3]
        fwd_a = c2w_a[:3, 2]
        best = None
        best_d = 1e30
        for name_b, c2w_b, _k in rows:
            if _group(name_b) == _group(name_a):
                continue
            fwd_b = c2w_b[:3, 2]
            if float(fwd_a @ fwd_b) <= 0.5:
                continue
            dist = float(np.linalg.norm(c2w_b[:3, 3] - centre_a))
            if 1e-6 < dist < best_d:
                best, best_d = (name_b, c2w_b), dist
        if best is None:
            continue
        _name_b, c2w_b = best
        centre = (1.0 - t) * centre_a + t * c2w_b[:3, 3]
        R = _ortho(c2w_a[:3, :3] @ rot_exp(t * rot_log(c2w_a[:3, :3].T @ c2w_b[:3, :3])))
        c2w = np.eye(4, dtype=np.float64)
        c2w[:3, :3] = R
        c2w[:3, 3] = centre
        width, height = _png_size(image_dir / f"{name_a}.png")
        made.append(
            {
                "name": f"{NOVEL_PREFIX}r{int(round_k):02d}_{len(made):04d}",
                "c2w": c2w.tolist(),
                "K": K,
                "width": width,
                "height": height,
                "ref": str(image_dir / f"{name_a}.png"),
                "src": name_a,
                "dst": _name_b,
                "t": t,
            }
        )
    return made


def _link(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return
    try:
        os.link(src, dest)
    except OSError:
        shutil.copy2(src, dest)


def build_merged(
    base_data_dir: str | Path, out_dir: str | Path, rounds: list[Path], hy_meta: str | Path
) -> None:
    """Hardlink the dataset and add each round's fixed views. cameras.json is a new file."""
    base = Path(base_data_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    # Snapshot first. The merged folder may sit inside the source tree, and linking
    # while walking would nest the same files forever.
    files: list[Path] = []
    out_resolved = out.resolve()
    for src in base.rglob("*"):
        if not src.is_file():
            continue
        try:
            src.resolve().relative_to(out_resolved)
            continue
        except ValueError:
            pass
        rel = src.relative_to(base)
        if rel.as_posix() == "cameras.json":
            continue
        files.append(src)
    for src in files:
        _link(src, out / src.relative_to(base))
    cams = json.loads((base / "cameras.json").read_text(encoding="utf-8"))
    for rnd in rounds:
        poses = json.loads((Path(rnd) / "poses.json").read_text(encoding="utf-8"))
        for pose in poses:
            name = str(pose["name"])
            _link(Path(rnd) / "fixed" / f"{name}.png", out / "images" / f"{name}.png")
            c2w = np.asarray(pose["c2w"], dtype=np.float64)
            cams[name] = {"extrinsic": np.linalg.inv(c2w).tolist(), "intrinsic": pose["K"]}
    tmp = out / "cameras.json.tmp"
    tmp.write_text(json.dumps(cams) + "\n", encoding="utf-8")
    os.replace(tmp, out / "cameras.json")
    meta = Path(hy_meta)
    parent_meta = out.parent / "meta_info.json"
    if meta.is_file() and not parent_meta.exists():
        _link(meta, parent_meta)
