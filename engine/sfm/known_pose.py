"""Known-pose COLMAP dataset for WorldStereo pinholes.

WorldStereo already chose the cameras. This writer does not run SphereSfM and
does not touch ``<scene>/colmap`` or any WAN file. Images and the sparse model
go to ``<scene>/worldstereo/dataset``. ``panorama.png`` is not written.

Points are the MoGe panorama cloud, plus per-keyframe MoGe depth aligned to
that cloud's rendered depth. Scale and shift are a least-squares fit on pixels
the cloud actually hit; pixels whose residual is past three robust standard
deviations are dropped and the fit is repeated. That is the §7.1.2 alignment
from the HY-World 2.0 paper, in Votion's own code, with MoGe instead of
WorldMirror.

When ``hires/source.txt`` names a complete set, those frames are the images
and the intrinsics are scaled to that size. ``world`` is the panorama
composite. ``frame`` is the WorldStereo keyframes upscaled with no composite.
A pinhole reprojection of the panorama is not used.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

from engine.geo.rails import kept_rail_indices
from engine.hires.pinhole import (
    HIRES_FRAME,
    active_choice,
    choice_name,
    hires_frames,
    hires_ready,
    use_hires,
)
from engine.job import REPO_ROOT, job_progress
from engine.pano.scene_io import pano_path, scene_dir
from engine.sfm.dataset_source import keyframe_paths, view_folders, write_source
from engine.world.depth_align import MIN_INLIERS, align_depth, moge_frame_depth, render_z
from engine.worldstereo import defaults as D
from engine.worldstereo.adapter import _moge_cloud, cut_pinhole
from engine.worldstereo.views import VIEWS

SPLATKIT = REPO_ROOT / "vendor" / "splatkit"
if str(SPLATKIT) not in sys.path:
    sys.path.insert(0, str(SPLATKIT))

from tools.colmap_read_model import Camera, Image, qvec2rotmat  # noqa: E402
from tools.colmap_write_model import write_cameras_binary, write_images_binary  # noqa: E402

# One sample per 8 pixels of an aligned depth map. A 480×832 frame then adds
# at most a few thousand points instead of a dense depth image.
KEYFRAME_STRIDE = 16

# OpenCV cube at the origin. Columns of R_c2w are camera +X, +Y, +Z in the
# ERP world (image centre is +Z, +Y down). w2c rotation is the transpose.
_CUBE_C2W = {
    "front": np.eye(3),
    "right": np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]),
    "back": np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]]),
    "left": np.array([[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]),
    "down": np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]]),
    "up": np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]),
}
_CUBE_ORDER = ("front", "right", "back", "left", "down", "up")


def dataset_dir(scene: str) -> Path:
    return scene_dir(scene) / "worldstereo" / "dataset"


def rotmat2qvec(rot: np.ndarray) -> np.ndarray:
    """COLMAP ``rotmat2qvec``. ``qvec2rotmat`` of the result matches ``rot``."""
    matrix = np.asarray(rot, dtype=np.float64)
    rxx, ryx, rzx, rxy, ryy, rzy, rxz, ryz, rzz = matrix.flat
    k = np.array(
        [
            [rxx - ryy - rzz, 0.0, 0.0, 0.0],
            [ryx + rxy, ryy - rxx - rzz, 0.0, 0.0],
            [rzx + rxz, rzy + ryz, rzz - rxx - ryy, 0.0],
            [ryz - rzy, rzx - rxz, rxy - ryx, rxx + ryy + rzz],
        ]
    ) / 3.0
    _vals, vecs = np.linalg.eigh(k)
    qvec = vecs[[3, 0, 1, 2], np.argmax(_vals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec


def frame_name(index: int, view: str, frame_i: int) -> str:
    return f"rail_{int(index):02d}_{view}_f{int(frame_i):03d}.png"


def frame_stride(height: int, width: int) -> int:
    """Scale the 480p stride so a 2× HiRes frame does not blow the point budget."""
    scale = max(int(height) / float(D.SIZE_H), int(width) / float(D.SIZE_W), 1.0)
    return max(1, int(round(KEYFRAME_STRIDE * scale)))


def _scaled_k(K: np.ndarray, width: int, height: int) -> np.ndarray:
    out = np.array(K, dtype=np.float64, copy=True)
    out[0, 0] *= width / float(D.SIZE_W)
    out[0, 2] *= width / float(D.SIZE_W)
    out[1, 1] *= height / float(D.SIZE_H)
    out[1, 2] *= height / float(D.SIZE_H)
    return out


def _camera_key(K: np.ndarray, width: int, height: int) -> tuple:
    return (
        int(width),
        int(height),
        round(float(K[0, 0]), 4),
        round(float(K[1, 1]), 4),
        round(float(K[0, 2]), 4),
        round(float(K[1, 2]), 4),
    )


def _frame_choice(rail: Path, source: str) -> tuple[list[Path], str]:
    """Images for the Generate-tab path. No silent fall back to the other set or to 480p."""
    keyframes = keyframe_paths(rail)
    if not keyframes:
        return [], ""
    label = choice_name(source)
    if not hires_ready(rail, source, len(keyframes)):
        raise RuntimeError(
            f"{rail.parent.name}/{rail.name} has no {label} HiRes. "
            "On Generate, choose that path and press HiRes all."
        )
    use_hires(rail, source)
    return hires_frames(rail, source), "hires"


def _cube_w2c() -> list[tuple[str, np.ndarray]]:
    poses = []
    for name in _CUBE_ORDER:
        w2c = np.eye(4, dtype=np.float64)
        w2c[:3, :3] = _CUBE_C2W[name].T
        poses.append((name, w2c))
    return poses


def _write_points(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    """Packed COLMAP points3D.bin. Tracks are empty; a splat init only needs xyz and rgb."""
    count = int(xyz.shape[0])
    rec = 8 + 24 + 3 + 8 + 8
    blob = np.zeros((count, rec), dtype=np.uint8)
    ids = np.arange(1, count + 1, dtype="<u8")
    blob[:, 0:8] = ids.view(np.uint8).reshape(count, 8)
    coords = np.ascontiguousarray(xyz, dtype="<f8")
    blob[:, 8:32] = coords.view(np.uint8).reshape(count, 24)
    blob[:, 32:35] = np.clip(np.rint(rgb), 0, 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(np.uint64(count).astype("<u8").tobytes())
        handle.write(blob.tobytes())


def build(scene: str, device: str = "cuda:0") -> dict:
    root = scene_dir(scene)
    pano_file = pano_path(scene)
    if not pano_file.is_file():
        raise FileNotFoundError(f"missing {pano_file}")
    bgr = cv2.imread(str(pano_file), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"failed to read {pano_file}")
    erp = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    kept = kept_rail_indices(scene)
    choice = active_choice(root)
    rails: list[tuple[int, str, Path, list[Path], str]] = []
    for index in kept:
        folders = view_folders(root, index)
        if not folders:
            print(f"STATUS // rail {index} has no worldstereo folder — skipped", flush=True)
            continue
        for view in VIEWS:
            folder = folders.get(view)
            if folder is None:
                continue
            frames, kind = _frame_choice(folder, choice)
            if not frames:
                print(f"STATUS // rail {index} {view} has no keyframes — skipped", flush=True)
                continue
            cam_path = folder / "camera.json"
            if not cam_path.is_file():
                raise FileNotFoundError(f"missing {cam_path}")
            rails.append((index, view, folder, frames, kind))
    if not rails:
        raise RuntimeError(f"no kept WorldStereo keyframes under {root / 'worldstereo' / 'rails'}")
    # WorldStereo upscale: the dataset is those frames. The panorama cloud is
    # only the scale reference, and the six pano cube faces stay out.
    frame_only = choice == HIRES_FRAME

    out = dataset_dir(scene)
    if out.exists():
        shutil.rmtree(out)
    image_dir = out / "images"
    sparse = out / "sparse" / "0"
    image_dir.mkdir(parents=True)
    sparse.mkdir(parents=True)

    print(
        f"STATUS // known-pose {len(rails)} rail(s), images={'frame' if frame_only else 'composite'}",
        flush=True,
    )
    job_progress(0, len(rails) + 2, "known-pose MoGe cloud")
    cloud_xyz, cloud_rgb = _moge_cloud(scene, erp, device)
    cloud_rgb_u8 = np.clip(np.rint(np.asarray(cloud_rgb) * 255.0), 0, 255).astype(np.uint8)

    cameras: dict[int, Camera] = {}
    images: dict[int, Image] = {}
    cam_ids: dict[tuple, int] = {}
    extra_xyz: list[np.ndarray] = []
    extra_rgb: list[np.ndarray] = []
    per_rail: dict[str, dict] = {}
    image_id = 1

    def camera_id(K: np.ndarray, width: int, height: int) -> int:
        key = _camera_key(K, width, height)
        if key in cam_ids:
            return cam_ids[key]
        cid = len(cameras) + 1
        cameras[cid] = Camera(
            id=cid,
            model="PINHOLE",
            width=int(width),
            height=int(height),
            params=np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2]], dtype=np.float64),
        )
        cam_ids[key] = cid
        return cid

    import torch
    from core.matrix3d_pipeline import get_moge_model

    moge = get_moge_model(ckpt=str(REPO_ROOT / "models" / "MoGe" / "model.pt"), device=device)
    moge.to(device)
    step = 1
    for index, view, folder, frames, kind in rails:
        payload = json.loads((folder / "camera.json").read_text(encoding="utf-8"))
        w2cs = np.asarray(payload["extrinsic"], dtype=np.float64)
        ks = np.asarray(payload["intrinsic"], dtype=np.float64)
        if w2cs.shape[0] != len(frames):
            raise RuntimeError(
                f"rail {index} {view} has {len(frames)} frames and {w2cs.shape[0]} poses"
            )
        rels: list[float] = []
        added = 0
        for enum_i, src in enumerate(frames):
            frame_i = int(src.stem)
            pose_i = enum_i
            bgr_frame = cv2.imread(str(src), cv2.IMREAD_COLOR)
            if bgr_frame is None:
                raise RuntimeError(f"failed to read {src}")
            rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
            height, width = rgb.shape[:2]
            K = _scaled_k(ks[pose_i], width, height)
            w2c = w2cs[pose_i]
            name = frame_name(index, view, frame_i)
            cv2.imwrite(str(image_dir / name), bgr_frame)
            qvec = rotmat2qvec(w2c[:3, :3])
            if not np.allclose(qvec2rotmat(qvec), w2c[:3, :3], atol=1e-6):
                raise RuntimeError(f"qvec round-trip failed for {name}")
            images[image_id] = Image(
                id=image_id,
                qvec=qvec,
                tvec=np.asarray(w2c[:3, 3], dtype=np.float64),
                camera_id=camera_id(K, width, height),
                name=name,
                xys=np.zeros((0, 2), dtype=np.float64),
                point3D_ids=np.zeros((0,), dtype=np.int64),
            )
            image_id += 1

            fov_x = float(np.degrees(2.0 * np.arctan((width / 2.0) / float(K[0, 0]))))
            z_pred = moge_frame_depth(moge, rgb, fov_x)
            if z_pred.shape != (height, width):
                raise RuntimeError(f"MoGe depth {z_pred.shape} does not match {name} {(height, width)}")
            z_hat = render_z(cloud_xyz, w2c, K, height, width)
            fitted = align_depth(z_hat, z_pred)
            if fitted is None:
                print(f"STATUS // rail {index} frame {frame_i:03d} alignment skipped", flush=True)
            else:
                scale, shift, inlier, rel = fitted
                stride = frame_stride(height, width)
                grid = np.zeros(inlier.shape, dtype=bool)
                grid[::stride, ::stride] = True
                ys, xs = np.nonzero(inlier & grid)
                z = scale * z_pred[ys, xs].astype(np.float64) + shift
                cam = np.stack(
                    [(xs - K[0, 2]) / K[0, 0] * z, (ys - K[1, 2]) / K[1, 1] * z, z],
                    axis=1,
                )
                world = (cam - w2c[:3, 3]) @ w2c[:3, :3]
                extra_xyz.append(world.astype(np.float32))
                extra_rgb.append(rgb[ys, xs])
                rels.append(rel)
                added += int(world.shape[0])
            job_progress(step, len(rails) + 2, f"known-pose rail {index} {view} frame {frame_i + 1}/{len(frames)}")
        step += 1
        per_rail[f"{index:02d}_{view}"] = {
            "rail": index,
            "view": view,
            "frames": len(frames),
            "source": kind,
            "aligned_points": added,
            "median_rel_residual": None if not rels else float(np.median(rels)),
        }
        print(
            f"STATUS // rail {index} {view} {kind} frames={len(frames)} aligned_points={added} "
            f"median_rel={per_rail[f'{index:02d}_{view}']['median_rel_residual']}",
            flush=True,
        )

    moge.to("cpu")
    torch.cuda.empty_cache()

    face = 0
    if not frame_only:
        face = max(2, int(erp.shape[1] // 4))
        cube_k = np.array(
            [[face / 2.0, 0.0, face / 2.0], [0.0, face / 2.0, face / 2.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        for name, w2c in _cube_w2c():
            cut = cut_pinhole(erp, w2c, cube_k, face, face)
            fname = f"cube_{name}.png"
            cv2.imwrite(str(image_dir / fname), cv2.cvtColor(cut, cv2.COLOR_RGB2BGR))
            images[image_id] = Image(
                id=image_id,
                qvec=rotmat2qvec(w2c[:3, :3]),
                tvec=np.zeros(3, dtype=np.float64),
                camera_id=camera_id(cube_k, face, face),
                name=fname,
                xys=np.zeros((0, 2), dtype=np.float64),
                point3D_ids=np.zeros((0,), dtype=np.int64),
            )
            image_id += 1

    parts_xyz: list[np.ndarray] = []
    parts_rgb: list[np.ndarray] = []
    if not frame_only:
        parts_xyz.append(cloud_xyz.astype(np.float32))
        parts_rgb.append(cloud_rgb_u8)
    if extra_xyz:
        parts_xyz.extend(extra_xyz)
        parts_rgb.extend(extra_rgb)
    if not parts_xyz:
        raise RuntimeError("known-pose produced no points")
    xyz = np.concatenate(parts_xyz, axis=0)
    rgb = np.concatenate(parts_rgb, axis=0)
    finite = np.isfinite(xyz).all(axis=1)
    xyz = xyz[finite]
    rgb = rgb[finite]

    write_cameras_binary(cameras, str(sparse / "cameras.bin"))
    write_images_binary(images, str(sparse / "images.bin"))
    _write_points(sparse / "points3D.bin", xyz, rgb)
    forbidden = out / "panorama.png"
    if forbidden.exists():
        forbidden.unlink()

    meta = {
        "source": "worldstereo-known-pose",
        "num_images": len(images),
        "num_points": int(xyz.shape[0]),
        "num_rails": len(rails),
        "images": "frame" if frame_only else "composite",
        "cube_faces": 0 if frame_only else len(_CUBE_ORDER),
        "cube_size": face,
        "keyframe_stride": KEYFRAME_STRIDE,
        "point_triangulator": "not run",
        "rails": per_rail,
    }
    (out / "sfm.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    write_source(scene, "worldstereo")
    job_progress(len(rails) + 2, len(rails) + 2, "known-pose dataset written")
    print(
        f"STATUS // known-pose images={len(images)} points={xyz.shape[0]} "
        f"rails={len(rails)} dir={out}",
        flush=True,
    )
    return meta
