"""Gnomonic (true pinhole) cube faces from SphereSfM equirects.

``sphere_cubic_reprojecer`` writes SIMPLE_PINHOLE 90° cameras, but its PNGs on
alpine_01 were not gnomonic: a correct remap of the same ERP is nearly straight
(~0.2 % line sag) while the binary's faces smile (~31 %). 3DGS then fits a bowl.

This pass keeps SPHERE poses and the cubic cameras.bin / images.bin, and
rewrites each ``frame_*_perspective_*.png`` by shooting pinhole rays through the
parent SPHERE camera into the ERP (OpenCV w2c, +Y down, longitude wrap).

Called from ``engine.workers.sfm`` after every Reconstruct. Can also be run on an
existing dataset::

    python -m engine.sfm.cubic_reproject --scene alpine_01
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np

from engine.job import REPO_ROOT
from engine.pano.scene_io import scene_dir

SPLATKIT = REPO_ROOT / "vendor" / "splatkit"
if str(SPLATKIT) not in sys.path:
    sys.path.insert(0, str(SPLATKIT))

from tools.colmap_read_model import qvec2rotmat, read_cameras_binary, read_images_binary  # noqa: E402

_FACE_RE = re.compile(r"frame_(\d+)_perspective_(\d+)", re.IGNORECASE)
_SPHERE_RE = re.compile(r"frame_(\d+)\.(png|jpg|jpeg)$", re.IGNORECASE)

LogFn = Callable[[str], None]


def _log(log: LogFn | None, msg: str) -> None:
    if log is not None:
        log(msg)
    else:
        print(msg, flush=True)


def _k_from_camera(cam) -> tuple[float, float, float]:
    p = np.asarray(cam.params, dtype=np.float64)
    if cam.model == "SIMPLE_PINHOLE":
        return float(p[0]), float(p[1]), float(p[2])
    if cam.model == "PINHOLE":
        fx, fy, cx, cy = (float(p[0]), float(p[1]), float(p[2]), float(p[3]))
        if abs(fx - fy) > 1e-3 * max(abs(fx), abs(fy), 1.0):
            raise RuntimeError(f"non-square PINHOLE fx={fx} fy={fy}")
        return fx, cx, cy
    raise RuntimeError(f"unsupported COLMAP camera model {cam.model}")


def _frame_index(name: str) -> int | None:
    m = _FACE_RE.search(name) or _SPHERE_RE.search(name)
    return int(m.group(1)) if m else None


def _rot(im) -> np.ndarray:
    return qvec2rotmat(np.asarray(im.qvec, dtype=np.float64))


def find_equirect_dir(colmap_dir: Path) -> Path:
    work = colmap_dir / "_spheresfm_work"
    for name in ("equirect_hires", "equirect", "equirect_lowres"):
        d = work / name
        if d.is_dir() and any(d.glob("frame_*.png")):
            return d
    raise FileNotFoundError(f"no SphereSfM equirect frames under {work}")


def find_sphere_images(colmap_dir: Path) -> dict[int, object]:
    """frame index -> SPHERE Image (OpenCV w2c). Empty if the work sparse is gone."""
    sparse = colmap_dir / "_spheresfm_work" / "sparse" / "0"
    path = sparse / "images.bin"
    if not path.is_file():
        return {}
    out: dict[int, object] = {}
    for im in read_images_binary(str(path)).values():
        idx = _frame_index(im.name)
        if idx is not None:
            out[idx] = im
    return out


def gnomonic_maps(
    width: int,
    height: int,
    f: float,
    cx: float,
    cy: float,
    r_face: np.ndarray,
    r_sphere: np.ndarray,
    erp_w: int,
    erp_h: int,
    dir_cam: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Pixel maps: pinhole (width×height) → ERP (erp_w×erp_h)."""
    if dir_cam is None:
        dir_cam = _dir_cam(width, height, f, cx, cy)
    rot = r_face.astype(np.float32) @ r_sphere.astype(np.float32).T
    d = dir_cam @ rot
    n = np.linalg.norm(d, axis=-1, keepdims=True).clip(1e-12)
    d = d / n
    lon = np.arctan2(d[..., 0], d[..., 2])
    lat = np.arcsin(np.clip(-d[..., 1], -1.0, 1.0))
    mapx = np.mod((lon + np.pi) / (2.0 * np.pi) * erp_w, erp_w).astype(np.float32)
    mapy = np.clip((np.pi / 2.0 - lat) / np.pi * (erp_h - 1), 0.0, float(erp_h - 1)).astype(
        np.float32
    )
    return mapx, mapy


def _dir_cam(width: int, height: int, f: float, cx: float, cy: float) -> np.ndarray:
    xs = (np.arange(width, dtype=np.float32) + 0.5 - np.float32(cx)) / np.float32(f)
    ys = (np.arange(height, dtype=np.float32) + 0.5 - np.float32(cy)) / np.float32(f)
    u, v = np.meshgrid(xs, ys)
    return np.stack([u, v, np.ones_like(u, dtype=np.float32)], axis=-1)


def remap_face(
    erp: np.ndarray,
    f: float,
    cx: float,
    cy: float,
    r_face: np.ndarray,
    r_sphere: np.ndarray,
    width: int,
    height: int,
    dir_cam: np.ndarray | None = None,
) -> np.ndarray:
    eh, ew = erp.shape[:2]
    mapx, mapy = gnomonic_maps(
        width, height, f, cx, cy, r_face, r_sphere, ew, eh, dir_cam=dir_cam
    )
    return cv2.remap(erp, mapx, mapy, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)


def _sphere_rot_for_frame(
    frame: int,
    sphere_by_frame: dict[int, object],
    face0_rot: dict[int, np.ndarray],
) -> np.ndarray:
    im = sphere_by_frame.get(frame)
    if im is not None:
        return _rot(im)
    if frame in face0_rot:
        return face0_rot[frame]
    raise RuntimeError(f"no SPHERE pose or face-0 fallback for frame {frame:05d}")


def rewrite_pinhole_faces(colmap_dir: Path, log: LogFn | None = None) -> int:
    """Overwrite ``images/*.png`` with gnomonic samples. Poses/K unchanged. Returns face count."""
    colmap_dir = Path(colmap_dir)
    image_dir = colmap_dir / "images"
    sparse = colmap_dir / "sparse" / "0"
    cams = read_cameras_binary(str(sparse / "cameras.bin"))
    images = read_images_binary(str(sparse / "images.bin"))
    equ_dir = find_equirect_dir(colmap_dir)
    sphere_by_frame = find_sphere_images(colmap_dir)
    face0_rot: dict[int, np.ndarray] = {}
    for im in images.values():
        m = _FACE_RE.search(im.name)
        if m and int(m.group(2)) == 0:
            face0_rot[int(m.group(1))] = _rot(im)

    erp_cache: dict[int, np.ndarray] = {}

    def erp_for(frame: int) -> np.ndarray:
        if frame not in erp_cache:
            path = equ_dir / f"frame_{frame:05d}.png"
            if not path.is_file():
                raise FileNotFoundError(path)
            bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if bgr is None:
                raise RuntimeError(f"failed to read {path}")
            erp_cache[frame] = bgr
            if len(erp_cache) > 4:
                oldest = next(iter(erp_cache))
                if oldest != frame:
                    del erp_cache[oldest]
        return erp_cache[frame]

    n = 0
    total = len(images)
    png_flags = [int(cv2.IMWRITE_PNG_COMPRESSION), 1]
    dir_cache: dict[tuple, np.ndarray] = {}
    for i, im in enumerate(images.values(), start=1):
        m = _FACE_RE.search(im.name)
        if not m:
            continue
        frame = int(m.group(1))
        cam = cams[im.camera_id]
        f, cx, cy = _k_from_camera(cam)
        key = (cam.width, cam.height, round(f, 6), round(cx, 6), round(cy, 6))
        if key not in dir_cache:
            dir_cache[key] = _dir_cam(cam.width, cam.height, f, cx, cy)
        r_face = _rot(im)
        r_sphere = _sphere_rot_for_frame(frame, sphere_by_frame, face0_rot)
        face = remap_face(
            erp_for(frame),
            f,
            cx,
            cy,
            r_face,
            r_sphere,
            cam.width,
            cam.height,
            dir_cam=dir_cache[key],
        )
        dest = image_dir / im.name
        if not dest.is_file():
            dest = image_dir / Path(im.name).name
        if not cv2.imwrite(str(dest), face, png_flags):
            raise RuntimeError(f"failed to write {dest}")
        n += 1
        if i == 1 or i % 50 == 0 or i == total:
            _log(log, f"STATUS // gnomonic cube remap {i}/{total}")
    if n == 0:
        raise RuntimeError(f"no perspective faces under {image_dir}")
    _log(log, f"STATUS // gnomonic cube remap wrote {n} faces from {equ_dir.name}")
    return n


def _self_test() -> None:
    """A world-space floor line is straight in a 90° gnomonic face, not in a lon/lat crop."""
    w, h = 1024, 512
    erp = np.zeros((h, w, 3), dtype=np.uint8)
    # Floor y_up=-1 → OpenCV Y=+1. Line (x, 1, 2), x in [-1.2, 1.2], from origin.
    for x in np.linspace(-1.2, 1.2, 2000):
        X, Y, Z = float(x), 1.0, 2.0
        n = (X * X + Y * Y + Z * Z) ** 0.5
        X, Y, Z = X / n, Y / n, Z / n
        lon = np.arctan2(X, Z)
        lat = np.arcsin(np.clip(-Y, -1.0, 1.0))
        u = int(np.mod((lon + np.pi) / (2 * np.pi) * w, w))
        v = int(np.clip((np.pi / 2.0 - lat) / np.pi * (h - 1), 0, h - 1))
        erp[v, u] = (0, 255, 0)

    size = 128
    f = size / 2.0
    r_id = np.eye(3)
    gnomonic = remap_face(erp, f, size / 2.0, size / 2.0, r_id, r_id, size, size)
    green = gnomonic[:, :, 1] > 80
    ys, xs = np.where(green)
    assert xs.size > 20, "gnomonic face missed the floor line"
    # Fit y = a + b x; a pinhole of a planar line is straight (tiny residual).
    A = np.c_[np.ones(xs.size), xs.astype(np.float64)]
    coef, *_ = np.linalg.lstsq(A, ys.astype(np.float64), rcond=None)
    resid = ys.astype(np.float64) - A @ coef
    rms = float(np.sqrt(np.mean(resid * resid)))
    assert rms < 2.5, f"gnomonic floor line rms {rms:.2f}px (want < 2.5)"

    # Lon/lat 90° crop of the same ERP: the planar line must bow.
    lon0, lat0 = 0.0, 0.0
    us = np.linspace(-np.pi / 4, np.pi / 4, size)
    vs = np.linspace(np.pi / 4, -np.pi / 4, size)
    uu, vv = np.meshgrid(us, vs)
    mx = np.mod((lon0 + uu + np.pi) / (2 * np.pi) * w, w).astype(np.float32)
    my = np.clip((np.pi / 2.0 - (lat0 + vv)) / np.pi * (h - 1), 0, h - 1).astype(np.float32)
    crop = cv2.remap(erp, mx, my, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
    gy, gx = np.where(crop[:, :, 1] > 80)
    assert gx.size > 20
    B = np.c_[np.ones(gx.size), gx.astype(np.float64), gx.astype(np.float64) ** 2]
    qc, *_ = np.linalg.lstsq(B, gy.astype(np.float64), rcond=None)
    sag = abs(float(qc[2])) * ((float(gx.max()) - float(gx.min())) / 2.0) ** 2
    assert sag > 3.0, f"lon/lat crop should bow, sag={sag:.2f}px"
    print(f"cubic_reproject self-test OK  gnomonic_rms={rms:.3f}px  lonlat_sag={sag:.1f}px")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Rewrite SphereSfM cube faces as gnomonic pinholes")
    parser.add_argument("--scene", help="Votion scene name (uses outputs/<scene>/colmap)")
    parser.add_argument("--colmap", type=Path, help="Path to a COLMAP dataset root")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        _self_test()
        return 0
    if args.colmap is not None:
        colmap = args.colmap
    elif args.scene:
        colmap = scene_dir(args.scene) / "colmap"
    else:
        parser.error("pass --scene, --colmap, or --self-test")
        return 2
    rewrite_pinhole_faces(colmap)
    sfm_json = colmap / "sfm.json"
    if sfm_json.is_file():
        meta = json.loads(sfm_json.read_text(encoding="utf-8"))
        meta["gnomonic_faces"] = True
        sfm_json.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
