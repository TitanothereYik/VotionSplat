"""Turn a Votion scene into the WorldStereo memory-DMD input tree.

The vendored loader (``load_mutli_traj_dataset``, ``sort_trajs``,
``HyMemoryBank``) reads ``pano_bank/`` and ``view*/traj*/`` at the scene
root. The README draws those same files under ``render_results/``. This writer
follows the loader, because that is what retrieval and sampling open.

``panorama.png`` is not written. That filename is what sends the stock scripts
down the unreleased VLM path.

One world frame for every rail: MoGe level-9 depth on the Plot Camera
2048×1024 canvas, unprojected with image centre +Z and +Y down. Rail cameras are
the Plot Camera ``rails/rail_XX.json`` matrices, every 4th frame. Perspective
cuts of the 8K pano use the pin's 27 origin cameras.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from engine.geo.defaults import MAX_RAILS
from engine.geo.rails import camplot_meta_rel, rail_json_rel
from engine.job import REPO_ROOT
from engine.pano.scene_io import pano_path, scene_dir
from engine.worldstereo import defaults as D
from engine.worldstereo.joint import JointChunk, JointStop, joint_plan
from engine.worldstereo.views import rail_view_rel

PIN_BANK = (
    REPO_ROOT
    / "vendor"
    / "worldstereo"
    / "examples"
    / "panorama"
    / "9c2dd62c7bb946a49e8e51278277eb6a"
    / "pano_bank"
    / "cameras.json"
)
# Plot Camera's MoGe canvas. One sample per pixel; the DiT is not loaded yet.
POINT_STRIDE = 1


@dataclass
class TrajJob:
    rail_index: int
    direction: str
    view: str
    view_id: str
    traj_id: str
    layout: str = "separate"
    frames: int | None = None

    @property
    def folder(self) -> str:
        return f"rail_{self.rail_index:02d}/{self.view}"

    @property
    def output_rel(self) -> str:
        return rail_view_rel(self.rail_index, self.view)

    @property
    def sample_count(self) -> int:
        """Frames the model sees. Per-rail runs use the 21-frame clip."""
        if self.frames is not None:
            return int(self.frames)
        return D.NFRAME


def input_root(scene: str) -> Path:
    return scene_dir(scene) / "worldstereo" / "input"


def pin_bank_cameras() -> dict:
    if not PIN_BANK.is_file():
        raise FileNotFoundError(f"pin pano bank is missing: {PIN_BANK}")
    data = json.loads(PIN_BANK.read_text(encoding="utf-8"))
    if len(data) != D.PANO_BANK_VIEWS:
        raise RuntimeError(f"pin pano bank has {len(data)} views, expected {D.PANO_BANK_VIEWS}")
    return data


def pin_K() -> np.ndarray:
    first = pin_bank_cameras()["0000"]["intrinsic"]
    return np.asarray(first, dtype=np.float64)


def subsample_indices(count: int, stride: int = D.RAIL_STRIDE, nframe: int = D.NFRAME) -> np.ndarray:
    """Frame 0, then every ``stride``, including the last frame when it lands on the stride."""
    count = int(count)
    if count < 2:
        raise RuntimeError(f"rail has {count} poses")
    idx = list(range(0, count, int(stride)))
    if idx[-1] != count - 1 and (count - 1) % int(stride) == 0:
        idx.append(count - 1)
    if len(idx) != nframe:
        idx = np.linspace(0, count - 1, nframe).round().astype(int).tolist()
    out = np.asarray(idx, dtype=np.int64)
    if out.shape[0] != nframe:
        raise RuntimeError(f"expected {nframe} poses, got {out.shape[0]}")
    return out


def cut_pinhole(erp: np.ndarray, w2c: np.ndarray, K: np.ndarray, width: int, height: int) -> np.ndarray:
    """Gnomonic cut. ``w2c`` is OpenCV. Image centre of the ERP is world +Z."""
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    xs = (np.arange(width, dtype=np.float64) + 0.5 - cx) / fx
    ys = (np.arange(height, dtype=np.float64) + 0.5 - cy) / fy
    grid_x, grid_y = np.meshgrid(xs, ys)
    dirs = np.stack([grid_x, grid_y, np.ones_like(grid_x)], axis=-1)
    world = dirs @ np.asarray(w2c[:3, :3], dtype=np.float64)
    norm = np.linalg.norm(world, axis=-1, keepdims=True).clip(1e-8)
    world = world / norm
    lon = np.arctan2(world[..., 0], world[..., 2])
    lat = np.arcsin(np.clip(-world[..., 1], -1.0, 1.0))
    eh, ew = erp.shape[:2]
    mapx = np.mod((lon + np.pi) / (2.0 * np.pi) * ew, ew).astype(np.float32)
    mapy = np.clip((np.pi / 2.0 - lat) / np.pi * (eh - 1), 0.0, float(eh - 1)).astype(np.float32)
    return cv2.remap(erp, mapx, mapy, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)


def erp_points(depth: np.ndarray, rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Radial depth on an ERP → OpenCV points. Centre pixel looks down +Z."""
    height, width = depth.shape
    xs = (np.arange(width, dtype=np.float64) + 0.5) / width
    ys = (np.arange(height, dtype=np.float64) + 0.5) / height
    uu, vv = np.meshgrid(xs, ys)
    lon = uu * (2.0 * np.pi) - np.pi
    lat = (np.pi / 2.0) - vv * np.pi
    cos_lat = np.cos(lat)
    direc = np.stack([cos_lat * np.sin(lon), -np.sin(lat), cos_lat * np.cos(lon)], axis=-1)
    points = (direc * depth[..., None]).reshape(-1, 3).astype(np.float32)
    colors = (rgb.astype(np.float32) / 255.0).reshape(-1, 3)
    return points, colors


def _read_mp4(path: Path) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open {path}")
    frames: list[np.ndarray] = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        cap.release()
    if not frames:
        raise RuntimeError(f"{path} has no frames")
    return frames


def _write_mp4(frames_bgr: list[np.ndarray], path: Path, fps: int = 16) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames_bgr[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"VideoWriter failed to open {path}")
    for frame in frames_bgr:
        writer.write(frame)
    writer.release()


def _replace_mp4(frames_bgr: list[np.ndarray], path: Path) -> None:
    """Write aside, then replace. A pane that still has the mp4 open keeps the previous file."""
    if not frames_bgr:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + "_next.mp4")
    _write_mp4(frames_bgr, tmp)
    try:
        os.replace(tmp, path)
    except OSError as exc:
        print(f"STATUS // preview busy {path.name}: {exc}", flush=True)


def publish_guide_preview(scene: str) -> None:
    """Copy the joint guide into the render and mask panes.

    The guide is written under ``input/joint_cNN/traj0``. The Generate page
    plays ``worldstereo/joint`` and ``worldstereo/rails/rail_NN/front``.
    """
    plan_path = input_root(scene) / "joint_plan.json"
    if not plan_path.is_file():
        return
    manifest = json.loads(plan_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, list) or not manifest:
        return
    root = scene_dir(scene)
    joint_colors: list[np.ndarray] = []
    joint_masks: list[np.ndarray] = []
    by_rail: dict[int, list[tuple[int, np.ndarray, np.ndarray]]] = {}
    for item in manifest:
        if not isinstance(item, dict):
            continue
        frames = item.get("frames")
        if not isinstance(frames, list) or not frames:
            continue
        anchor = int(item.get("anchor") or 0)
        view_id = str(item.get("view_id") or "")
        traj = input_root(scene) / view_id / "traj0"
        colors = _read_mp4(traj / "render.mp4")
        masks = _read_mp4(traj / "render_mask.mp4")
        need = anchor + len(frames)
        if len(colors) < need or len(masks) < need:
            raise RuntimeError(f"{view_id} guide has {len(colors)} frames, need {need}")
        colors = colors[anchor:need]
        masks = masks[anchor:need]
        joint_colors.extend(colors)
        joint_masks.extend(masks)
        for offset, row in enumerate(frames):
            if not isinstance(row, dict):
                continue
            rail = int(row["rail"])
            pose = int(row["pose"])
            by_rail.setdefault(rail, []).append((pose, colors[offset], masks[offset]))
    dest = root / "worldstereo" / "joint"
    _replace_mp4(joint_colors, dest / "render" / "render.mp4")
    _replace_mp4(joint_masks, dest / "mask" / "render_mask.mp4")
    for rail, items in by_rail.items():
        items.sort(key=lambda row: row[0])
        folder = root / rail_view_rel(rail, "front")
        _replace_mp4([frame for _, frame, _ in items], folder / "render" / "render.mp4")
        _replace_mp4([frame for _, _, frame in items], folder / "mask" / "render_mask.mp4")
    print("WS\tinputs", flush=True)


def _load_rail(root: Path, index: int) -> np.ndarray:
    path = root / rail_json_rel(index)
    if not path.is_file():
        raise FileNotFoundError(f"missing {path}")
    arr = np.asarray(json.loads(path.read_text(encoding="utf-8")), dtype=np.float64)
    if arr.ndim != 3 or arr.shape[1:] != (4, 4):
        raise RuntimeError(f"{path} is {arr.shape}, expected (T, 4, 4) w2c")
    return arr


def _direction(root: Path, index: int) -> str:
    path = root / camplot_meta_rel(index)
    if path.is_file():
        meta = json.loads(path.read_text(encoding="utf-8"))
        name = str(meta.get("orientation") or "").strip()
        if name:
            return name
    return "look_forward"


def select_poses(w2c_full: np.ndarray, job: TrajJob) -> tuple[np.ndarray, np.ndarray]:
    """Front-view poses for one rail. The model sees the same poses that are kept."""
    chosen = subsample_indices(w2c_full.shape[0])
    mats = np.asarray(w2c_full, dtype=np.float64)[chosen]
    return mats, mats


def list_trajs(
    scene: str,
    order: list[int] | None = None,
) -> list[TrajJob]:
    """One front-view job per rail. Rails follow ``order``; missing rails are skipped."""
    root = scene_dir(scene)
    if order is None:
        indices = [index for index in range(MAX_RAILS) if (root / rail_json_rel(index)).is_file()]
    else:
        indices = []
        for index in order:
            index = int(index)
            if index not in indices and (root / rail_json_rel(index)).is_file():
                indices.append(index)
    jobs: list[TrajJob] = []
    for index in indices:
        jobs.append(
            TrajJob(
                rail_index=index,
                direction=_direction(root, index),
                view="front",
                view_id=f"view{index:02d}_front",
                traj_id="traj0",
                layout="separate",
            )
        )
    if not jobs:
        raise FileNotFoundError(f"no rails/rail_XX.json under {root}")
    return jobs


def joint_counts(scene: str, order: list[int]) -> list[tuple[int, int]]:
    """``(rail, pose count)`` for the rails that exist, in ``order``."""
    root = scene_dir(scene)
    counts: list[tuple[int, int]] = []
    for index in order:
        path = root / rail_json_rel(int(index))
        if not path.is_file():
            continue
        counts.append((int(index), int(_load_rail(root, int(index)).shape[0])))
    if not counts:
        raise FileNotFoundError(f"no rails/rail_XX.json under {root}")
    return counts


def joint_chunks(scene: str, order: list[int], frames_per_rail: int | None = None) -> list[JointChunk]:
    return joint_plan(joint_counts(scene, order), frames_per_rail)


def moge_erp_depth(scene: str, pano_rgb: np.ndarray, device: str) -> tuple[np.ndarray, np.ndarray]:
    """Cached scene depth at the mesh canvas. ``valid`` is the mask before the sky fill."""
    del pano_rgb
    from engine.depth.scene_depth import depth_at, ensure_scene_depth
    from engine.world.defaults import MESH_CANVAS

    depth, valid = depth_at(ensure_scene_depth(scene, device), MESH_CANVAS, fill_sky=True)
    if not valid.any():
        raise RuntimeError("scene depth mask is empty")
    return depth, valid


def _moge_cloud(scene: str, pano_rgb: np.ndarray, device: str) -> tuple[np.ndarray, np.ndarray]:
    depth, _valid = moge_erp_depth(scene, pano_rgb, device)
    small = cv2.resize(pano_rgb, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_AREA)
    step = POINT_STRIDE
    points, colors = erp_points(depth[::step, ::step], small[::step, ::step])
    finite = np.isfinite(points).all(axis=1)
    return points[finite], colors[finite]


def _render_traj(points: np.ndarray, colors: np.ndarray, w2cs: np.ndarray, K: np.ndarray, device: str):
    ws = REPO_ROOT / "vendor" / "worldstereo"
    if str(ws) not in sys.path:
        sys.path.insert(0, str(ws))
    import torch
    from src.pointcloud import point_rendering

    nframe = w2cs.shape[0]
    Ks = np.repeat(K[None].astype(np.float32), nframe, axis=0)
    rgb, mask = point_rendering(
        torch.tensor(Ks),
        torch.tensor(w2cs.astype(np.float32)),
        torch.tensor(points),
        torch.tensor(colors),
        device,
        D.SIZE_H,
        D.SIZE_W,
        background_color=[0, 0, 0],
        render_radius=D.RENDER_RADIUS,
        points_per_pixel=D.POINTS_PER_PIXEL,
    )
    rgb_np = rgb.detach().float().clamp(0, 1).mul(255).byte().cpu().permute(0, 2, 3, 1).numpy()
    # point_rendering: 1 where the z-buffer missed. That is white in the mask video.
    empty = mask.detach().float().cpu()
    empty_np = (empty[:, 0] >= 0.5).numpy()
    del rgb, mask
    torch.cuda.empty_cache()
    return rgb_np, empty_np


def _write_bank(root: Path, erp: np.ndarray, bank: dict, K: np.ndarray) -> None:
    images = root / "pano_bank" / "images"
    images.mkdir(parents=True, exist_ok=True)
    cameras = {}
    for key in sorted(bank):
        extrinsic = np.asarray(bank[key]["extrinsic"], dtype=np.float64)
        extrinsic = extrinsic.copy()
        extrinsic[:3, 3] = 0.0
        cut = cut_pinhole(erp, extrinsic, K, D.SIZE_W, D.SIZE_H)
        cv2.imwrite(str(images / f"{key}.png"), cv2.cvtColor(cut, cv2.COLOR_RGB2BGR))
        cameras[key] = {"intrinsic": K.tolist(), "extrinsic": extrinsic.tolist()}
    (root / "pano_bank" / "cameras.json").write_text(json.dumps(cameras), encoding="utf-8")


class _Guide:
    """Renders a trajectory. ``close`` frees the mesh renderer before the DiT loads."""

    def __init__(self, kind: str, renderer=None, points=None, colors=None, device: str = "cpu", K=None):
        self.kind = kind
        self.renderer = renderer
        self.points = points
        self.colors = colors
        self.device = device
        self.K = K

    def __call__(self, w2cs: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        poses = np.asarray(w2cs)
        if self.kind == "mesh":
            from engine.world.defaults import GUIDE_SUPERSAMPLE

            rgb: list[np.ndarray] = []
            empty: list[np.ndarray] = []
            from engine.world.render import clean_guide

            for pose in poses:
                view = self.renderer.render(pose, self.K, D.SIZE_H, D.SIZE_W, supersample=GUIDE_SUPERSAMPLE)
                frame, hole = clean_guide(view.rgb, view.empty)
                rgb.append(frame)
                empty.append(hole)
            return rgb, empty
        rgb_np, empty_np = _render_traj(self.points, self.colors, poses, self.K, self.device)
        return [frame for frame in rgb_np], [frame for frame in empty_np]

    def close(self) -> None:
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None


def _prepare_input(scene: str, prompt: str, device: str):
    """Pano, pin intrinsics, and a guide renderer for the WorldStereo input root."""
    from engine.world.defaults import GUIDE_RENDERER

    root = scene_dir(scene)
    pano_file = pano_path(scene)
    if not pano_file.is_file():
        raise FileNotFoundError(f"missing {pano_file}")
    bgr = cv2.imread(str(pano_file), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"failed to read {pano_file}")
    erp = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    K = pin_K()
    out = input_root(scene)
    out.mkdir(parents=True, exist_ok=True)
    forbidden = out / "panorama.png"
    if forbidden.exists():
        forbidden.unlink()
    (out / "meta_info.json").write_text(
        json.dumps({"scene_type": "panorama"}, indent=2) + "\n", encoding="utf-8"
    )
    caption = {"short caption": prompt, "medium caption": prompt, "long caption": prompt}
    (out / "prompt.json").write_text(json.dumps(caption, indent=2) + "\n", encoding="utf-8")
    _write_bank(out, erp, pin_bank_cameras(), K)
    if GUIDE_RENDERER == "mesh":
        from engine.world.pano_mesh import ensure_pano_mesh, load_ground
        from engine.world.render import WorldRenderer

        piece = ensure_pano_mesh(scene, erp, device)
        renderer = WorldRenderer([piece], {"pano": erp}, device, ground=load_ground(scene))
        guide = _Guide("mesh", renderer=renderer, device=device, K=K)
    elif GUIDE_RENDERER == "points":
        points, colors = _moge_cloud(scene, erp, device)
        norms = np.linalg.norm(points, axis=1)
        print(
            f"STATUS // points={points.shape[0]} radius_p50={float(np.median(norms)):.3f} "
            f"radius_p95={float(np.quantile(norms, 0.95)):.3f}",
            flush=True,
        )
        guide = _Guide("points", points=points, colors=colors, device=device, K=K)
    else:
        raise RuntimeError(f"unknown GUIDE_RENDERER {GUIDE_RENDERER}")
    return root, erp, K, out, guide


def _write_traj(
    out: Path,
    view_id: str,
    erp: np.ndarray,
    w2cs: np.ndarray,
    K: np.ndarray,
    guide: _Guide,
    prompt: str,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    view = out / view_id
    traj = view / "traj0"
    traj.mkdir(parents=True, exist_ok=True)
    start = cut_pinhole(erp, w2cs[0], K, D.SIZE_W, D.SIZE_H)
    cv2.imwrite(str(view / "start_frame.png"), cv2.cvtColor(start, cv2.COLOR_RGB2BGR))
    rgb, empty = guide(w2cs)
    color_bgr = [cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) for frame in rgb]
    mask_bgr = [cv2.cvtColor(np.where(frame, 255, 0).astype(np.uint8), cv2.COLOR_GRAY2BGR) for frame in empty]
    _write_mp4(color_bgr, traj / "render.mp4")
    _write_mp4(mask_bgr, traj / "render_mask.mp4")
    camera = {"extrinsic": w2cs.tolist(), "intrinsic": [K.tolist() for _ in range(w2cs.shape[0])]}
    (traj / "camera.json").write_text(json.dumps(camera), encoding="utf-8")
    (traj / "traj_caption.json").write_text(json.dumps({"prompt": prompt}, indent=2) + "\n", encoding="utf-8")
    return color_bgr, mask_bgr


def build_scene(scene: str, jobs: list[TrajJob], prompt: str, device: str = "cuda:0") -> Path:
    """Write ``<scene>/worldstereo/input`` for one front view per rail."""
    root, erp, K, out, guide = _prepare_input(scene, prompt, device)
    try:
        for job in jobs:
            w2c_full = _load_rail(root, job.rail_index)
            w2cs, kept = select_poses(w2c_full, job)
            colors_bgr, masks_bgr = _write_traj(out, job.view_id, erp, w2cs, K, guide, prompt)
            rail_out = root / job.output_rel
            (rail_out / "render").mkdir(parents=True, exist_ok=True)
            (rail_out / "mask").mkdir(parents=True, exist_ok=True)
            _write_mp4(colors_bgr, rail_out / "render" / "render.mp4")
            _write_mp4(masks_bgr, rail_out / "mask" / "render_mask.mp4")
            camera = {
                "extrinsic": kept.tolist(),
                "intrinsic": [K.tolist() for _ in range(kept.shape[0])],
            }
            (rail_out / "camera.json").write_text(json.dumps(camera), encoding="utf-8")
    finally:
        guide.close()
    return out


def _chunk_model_stops(chunks: list[JointChunk], chunk: JointChunk) -> tuple[list[JointStop], int]:
    """Stops the model sees. A later chunk prepends the previous chunk's last frame."""
    if chunk.index == 0:
        return list(chunk.frames), 0
    previous = chunks[chunk.index - 1].frames[-1]
    return [previous, *chunk.frames], 1


def build_joint_scene(
    scene: str,
    order: list[int],
    prompt: str,
    frames_per_rail: int | None = None,
    device: str = "cuda:0",
) -> Path:
    """One input trajectory per joint chunk. The first frame of chunk 0 is the star."""
    root, erp, K, out, guide = _prepare_input(scene, prompt, device)
    chunks = joint_chunks(scene, order, frames_per_rail)
    rails = {index: _load_rail(root, index) for index, _count in joint_counts(scene, order)}
    manifest = []
    try:
        for chunk in chunks:
            stops, anchor = _chunk_model_stops(chunks, chunk)
            w2cs = np.stack([rails[stop.rail][stop.pose] for stop in stops], axis=0)
            view_id = f"joint_c{chunk.index:02d}"
            _write_traj(out, view_id, erp, w2cs, K, guide, prompt)
            manifest.append(
                {
                    "index": chunk.index,
                    "view_id": view_id,
                    "anchor": anchor,
                    "frames": [{"rail": stop.rail, "pose": stop.pose} for stop in chunk.frames],
                }
            )
    finally:
        guide.close()
    (out / "joint_plan.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    publish_guide_preview(scene)
    return out

