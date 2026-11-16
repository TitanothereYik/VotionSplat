"""Shared world layer. Each joint chunk lifts the pixels its guide left empty.

A later chunk renders those surfaces in its guide, so the next sample sees
the same wall instead of inventing it again. A pixel is lifted once: the
renderer grows as frames are lifted inside the chunk.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

from engine.job import REPO_ROOT
from engine.pano.scene_io import pano_path, scene_dir
from engine.world.defaults import GUIDE_SUPERSAMPLE, LIFT_MIN_PIXELS
from engine.world.depth_align import align_depth, moge_frame_depth
from engine.world.pano_mesh import Piece, cull_faces, ensure_pano_mesh, load_ground, world_dir
from engine.world.render import WorldRenderer, clean_guide


def _log(msg: str) -> None:
    print(msg, flush=True)


def _read_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"missing {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _read_tex(scene: str, tex: str) -> np.ndarray:
    return _read_rgb(scene_dir(scene) / tex)


def lift_frame(
    rgb: np.ndarray,
    z_pred: np.ndarray,
    view,
    w2c: np.ndarray,
    K: np.ndarray,
    tex: str,
) -> tuple[Piece | None, dict]:
    """Build a mesh over guide holes. Pure numpy; the depth fit is ``align_depth``."""
    empty = np.asarray(view.empty, dtype=bool)
    pixels = int(empty.sum())
    if pixels < LIFT_MIN_PIXELS:
        return None, {"skipped": "few"}
    fitted = align_depth(view.depth, z_pred)
    if fitted is None:
        return None, {"skipped": "align"}
    scale, shift, _inlier, rel = fitted
    z = scale * np.asarray(z_pred, dtype=np.float64) + shift
    ok = np.isfinite(z) & (z > 1e-3)
    height, width = empty.shape
    pose = np.asarray(w2c, dtype=np.float64)
    rot = pose[:3, :3]
    trans = pose[:3, 3]
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    # One triangle per pixel, with the pixel centre strictly inside it. A quad
    # split puts that centre on the shared edge, and the rasterizer drops it.
    if height < 1 or width < 1:
        return None, {"skipped": "small", "pixels": pixels}
    near = empty.copy()
    near[1:] |= empty[:-1]
    near[:-1] |= empty[1:]
    near[:, 1:] |= empty[:, :-1]
    near[:, :-1] |= empty[:, 1:]
    yy, xx = np.nonzero(near & ok)
    if yy.size == 0:
        return None, {"skipped": "cull", "pixels": pixels, "scale": float(scale), "shift": float(shift), "rel": float(rel)}
    z_pix = z[yy, xx]
    corners = np.stack(
        [
            np.stack([xx, yy], axis=1),
            np.stack([xx + 1.0, yy], axis=1),
            np.stack([xx + 0.5, yy + 1.0], axis=1),
        ],
        axis=1,
    ).astype(np.float64)
    x_cam = (corners[..., 0] - cx) / fx * z_pix[:, None]
    y_cam = (corners[..., 1] - cy) / fy * z_pix[:, None]
    cam = np.stack([x_cam, y_cam, np.broadcast_to(z_pix[:, None], x_cam.shape)], axis=-1)
    world = (cam - trans) @ rot
    verts = world.reshape(-1, 3)
    faces = np.arange(yy.size * 3, dtype=np.int32).reshape(-1, 3)
    center = -rot.T @ trans
    faces = cull_faces(verts, faces, center)
    if faces.size == 0:
        return None, {"skipped": "cull", "pixels": pixels, "scale": float(scale), "shift": float(shift), "rel": float(rel)}
    used = np.unique(faces.reshape(-1))
    remap = np.full(verts.shape[0], -1, dtype=np.int32)
    remap[used] = np.arange(used.shape[0], dtype=np.int32)
    faces = remap[faces]
    kept_verts = verts[used].astype(np.float32)
    # Constant UV per pixel so the centre samples that pixel, not a neighbour.
    pix_id = used // 3
    uvs = np.stack(
        [(xx[pix_id] + 0.5) / float(width), (yy[pix_id] + 0.5) / float(height)],
        axis=1,
    ).astype(np.float32)
    info = {
        "pixels": pixels,
        "faces": int(faces.shape[0]),
        "scale": float(scale),
        "shift": float(shift),
        "rel": float(rel),
    }
    return Piece(kept_verts, faces.astype(np.int32), uvs, tex), info


def save_layer(scene: str, chunk: int, pieces: list[Piece], infos: list[dict]) -> None:
    folder = world_dir(scene)
    folder.mkdir(parents=True, exist_ok=True)
    vert_starts: list[int] = []
    face_starts: list[int] = []
    verts: list[np.ndarray] = []
    faces: list[np.ndarray] = []
    uvs: list[np.ndarray] = []
    v0 = 0
    f0 = 0
    for piece in pieces:
        vert_starts.append(v0)
        face_starts.append(f0)
        verts.append(np.asarray(piece.verts, dtype=np.float32))
        uvs.append(np.asarray(piece.uvs, dtype=np.float32))
        faces.append(np.asarray(piece.faces, dtype=np.int32) + v0)
        v0 += int(piece.verts.shape[0])
        f0 += int(piece.faces.shape[0])
    if verts:
        verts_a = np.concatenate(verts, axis=0)
        faces_a = np.concatenate(faces, axis=0)
        uvs_a = np.concatenate(uvs, axis=0)
    else:
        verts_a = np.zeros((0, 3), np.float32)
        faces_a = np.zeros((0, 3), np.int32)
        uvs_a = np.zeros((0, 2), np.float32)
    np.savez_compressed(
        folder / f"layer_c{int(chunk):02d}.npz",
        verts=verts_a,
        faces=faces_a,
        uvs=uvs_a,
        piece_vert_start=np.asarray(vert_starts, dtype=np.int64),
        piece_face_start=np.asarray(face_starts, dtype=np.int64),
    )
    payload = {"chunk": int(chunk), "pieces": infos}
    (folder / f"layer_c{int(chunk):02d}.json").write_text(json.dumps(payload) + "\n", encoding="utf-8")


def load_layers(scene: str, upto: int) -> list[Piece]:
    folder = world_dir(scene)
    pieces: list[Piece] = []
    for chunk in range(int(upto)):
        path = folder / f"layer_c{chunk:02d}.npz"
        meta_path = folder / f"layer_c{chunk:02d}.json"
        if not path.is_file():
            continue
        records: list[dict] = []
        if meta_path.is_file():
            records = list(json.loads(meta_path.read_text(encoding="utf-8")).get("pieces") or [])
        with np.load(path, allow_pickle=False) as data:
            verts = np.asarray(data["verts"], dtype=np.float32)
            faces = np.asarray(data["faces"], dtype=np.int32)
            uvs = np.asarray(data["uvs"], dtype=np.float32)
            vstart = np.asarray(data["piece_vert_start"], dtype=np.int64)
            fstart = np.asarray(data["piece_face_start"], dtype=np.int64)
        count = int(vstart.shape[0])
        for index in range(count):
            v0 = int(vstart[index])
            f0 = int(fstart[index])
            v1 = int(vstart[index + 1]) if index + 1 < count else int(verts.shape[0])
            f1 = int(fstart[index + 1]) if index + 1 < count else int(faces.shape[0])
            tex = str(records[index]["tex"]) if index < len(records) else ""
            pieces.append(Piece(verts[v0:v1], faces[f0:f1] - v0, uvs[v0:v1], tex))
    return pieces


def clear_layers_from(scene: str, start: int) -> None:
    folder = world_dir(scene)
    if folder.is_dir():
        for path in folder.glob("layer_c*"):
            stem = path.name
            try:
                index = int(stem[7:9])
            except ValueError:
                continue
            if index >= int(start):
                path.unlink()
    if int(start) == 0:
        tex = folder / "tex_hi"
        if tex.is_dir():
            shutil.rmtree(tex)


def _pano_rgb(scene: str) -> np.ndarray:
    return _read_rgb(pano_path(scene))


def _layer_textures(scene: str, pieces: list[Piece]) -> dict[str, np.ndarray]:
    textures: dict[str, np.ndarray] = {}
    for piece in pieces:
        if piece.tex not in textures:
            textures[piece.tex] = _read_tex(scene, piece.tex)
    return textures


def ensure_layers(scene: str, manifest: list, ranges: list, upto: int, device: str) -> None:
    """Lift chunks ``c0 .. upto-1`` that do not yet have a layer file."""
    folder = world_dir(scene)
    c0 = None
    for chunk in range(int(upto)):
        if not (folder / f"layer_c{chunk:02d}.npz").is_file():
            c0 = chunk
            break
    if c0 is None:
        return
    clear_layers_from(scene, c0)
    from engine.worldstereo.adapter import input_root, pin_K

    erp = _pano_rgb(scene)
    pano = ensure_pano_mesh(scene, erp, device)
    existing = load_layers(scene, c0)
    textures = {"pano": erp, **_layer_textures(scene, existing)}
    renderer = WorldRenderer([pano, *existing], textures, device, ground=load_ground(scene))
    moge = None
    try:
        splatkit = REPO_ROOT / "vendor" / "splatkit"
        if str(splatkit) not in sys.path:
            sys.path.insert(0, str(splatkit))
        from core.matrix3d_pipeline import get_moge_model

        ckpt = REPO_ROOT / "models" / "MoGe" / "model.pt"
        moge = get_moge_model(ckpt=str(ckpt), device=device)
        moge.to(device)
        k_mat = pin_K()
        fov_x = float(np.degrees(2.0 * np.arctan(416.0 / float(k_mat[0, 0]))))
        for chunk in range(c0, int(upto)):
            item = manifest[chunk]
            anchor = int(item["anchor"])
            view_id = str(item["view_id"])
            origin, count = ranges[chunk]
            cam_path = input_root(scene) / view_id / "traj0" / "camera.json"
            camera = json.loads(cam_path.read_text(encoding="utf-8"))
            extrinsics = camera["extrinsic"]
            built: list[Piece] = []
            infos: list[dict] = []
            for offset in range(int(count)):
                global_i = int(origin) + offset
                tex = f"worldstereo/joint/keyframes/{global_i:03d}.png"
                rgb = _read_tex(scene, tex)
                w2c = np.asarray(extrinsics[anchor + offset], dtype=np.float64)
                view = renderer.render(w2c, k_mat, 480, 832, supersample=1)
                z_pred = moge_frame_depth(moge, rgb, fov_x)
                piece, info = lift_frame(rgb, z_pred, view, w2c, k_mat, tex)
                if piece is None:
                    _log(
                        f"STATUS // world lift c{chunk:02d} frame {global_i:03d} "
                        f"skipped={info.get('skipped', 'skip')}"
                    )
                    continue
                renderer.add_piece(piece, rgb)
                built.append(piece)
                infos.append({"tex": tex, **info})
                _log(
                    f"STATUS // world lift c{chunk:02d} frame {global_i:03d} "
                    f"pixels={info['pixels']} faces={info['faces']}"
                )
            save_layer(scene, chunk, built, infos)
    finally:
        if moge is not None:
            moge.to("cpu")
        renderer.close()
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def refresh_guide(scene: str, manifest: list, index: int, device: str) -> None:
    """Rewrite one chunk's guide videos with the pano mesh plus layers before it."""
    from engine.worldstereo import defaults as D
    from engine.worldstereo.adapter import _write_mp4, input_root, pin_K

    item = manifest[int(index)]
    view_id = str(item["view_id"])
    cam_path = input_root(scene) / view_id / "traj0" / "camera.json"
    camera = json.loads(cam_path.read_text(encoding="utf-8"))
    w2cs = np.asarray(camera["extrinsic"], dtype=np.float64)
    k_mat = pin_K()
    erp = _pano_rgb(scene)
    pano = ensure_pano_mesh(scene, erp, device)
    layers = load_layers(scene, int(index))
    textures = {"pano": erp, **_layer_textures(scene, layers)}
    renderer = WorldRenderer([pano, *layers], textures, device, ground=load_ground(scene))
    try:
        colors = []
        masks = []
        empty_frac = []
        for w2c in w2cs:
            view = renderer.render(w2c, k_mat, D.SIZE_H, D.SIZE_W, supersample=GUIDE_SUPERSAMPLE)
            frame, hole = clean_guide(view.rgb, view.empty)
            colors.append(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            mask = np.where(hole, 255, 0).astype(np.uint8)
            masks.append(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))
            empty_frac.append(float(hole.mean()))
        traj = cam_path.parent
        _write_mp4(colors, traj / "render.mp4")
        _write_mp4(masks, traj / "render_mask.mp4")
        mean_empty = float(np.mean(empty_frac)) if empty_frac else 0.0
        _log(f"STATUS // world guide c{int(index):02d} empty={mean_empty:.3f}")
        from engine.worldstereo.adapter import publish_guide_preview

        publish_guide_preview(scene)
    finally:
        renderer.close()
