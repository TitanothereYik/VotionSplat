"""Generate a synthetic COLMAP dataset (textured plane, exact known geometry) to sanity-check
the LiteGS/MCMC Compact adapter end-to-end, independent of real SfM data quality.

Why: on alpine_01 (real WAN/SphereSfM panoramic data) both controllers plateau at low PSNR
and a visibly blurry render. That could mean (a) the adapter (render/loss/backend) has a
correctness bug, or (b) the *data* is unusually hard (sparse points, reflective glass,
wide-baseline panoramic capture). This script builds a trivial, well-posed control case: a
single flat textured plane, viewed by a small pinhole camera rig, with EXACT ground-truth
camera poses/intrinsics and an EXACT (non-triangulated) points3D seeded directly from the
known plane geometry. Feeding this through the *same* trainer code path used for alpine_01
should converge to a near-perfect, sharp reconstruction in a few thousand steps if the
adapter itself is sound. If it *also* stays blurry/low-PSNR here, that is a smoking gun for
an adapter bug, not a data problem.

Run: python -m tools.gen_synth_plane --scene synth_plane_test [--n-views 60] [--res 512]
Then: python -m engine.workers.splat --scene synth_plane_test --strategy litegs --mode retrain
      --max-steps 3000 --headless --holdout
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ------------------------------------------------------------------ COLMAP binary writers
# Mirror of vendor/splatkit/tools/colmap_read_model.py's byte layout, in reverse.

def _write_cameras_binary(path: Path, cams: dict) -> None:
    """cams: {id: (model_id, width, height, params[])}. model_id=0 is SIMPLE_PINHOLE (f,cx,cy)."""
    with open(path, "wb") as fid:
        fid.write(struct.pack("<Q", len(cams)))
        for cid, (model_id, w, h, params) in cams.items():
            fid.write(struct.pack("<iiQQ", cid, model_id, w, h))
            fid.write(struct.pack("<" + "d" * len(params), *params))


def _write_images_binary(path: Path, images: list) -> None:
    """images: list of dict(id, qvec[4], tvec[3], camera_id, name, xys[N,2], point3D_ids[N])."""
    with open(path, "wb") as fid:
        fid.write(struct.pack("<Q", len(images)))
        for im in images:
            fid.write(struct.pack(
                "<idddddddi", im["id"], *im["qvec"], *im["tvec"], im["camera_id"],
            ))
            fid.write(im["name"].encode("utf-8") + b"\x00")
            xys = im["xys"]
            pids = im["point3D_ids"]
            fid.write(struct.pack("<Q", len(pids)))
            for (x, y), pid in zip(xys, pids):
                fid.write(struct.pack("<ddq", float(x), float(y), int(pid)))


def _write_points3d_binary(path: Path, points: list) -> None:
    """points: list of dict(id, xyz[3], rgb[3], error, track[(image_id,point2d_idx),...])."""
    with open(path, "wb") as fid:
        fid.write(struct.pack("<Q", len(points)))
        for p in points:
            fid.write(struct.pack(
                "<QdddBBBd", p["id"], *p["xyz"], *p["rgb"], p["error"],
            ))
            track = p["track"]
            fid.write(struct.pack("<Q", len(track)))
            for image_id, p2d_idx in track:
                fid.write(struct.pack("<ii", image_id, p2d_idx))


# ------------------------------------------------------------------ scene generation

def _rotmat2qvec(R: np.ndarray) -> np.ndarray:
    """3x3 rotation -> COLMAP qvec (qw,qx,qy,qz)."""
    m = R
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        qw = 0.25 * s
        qx = (m[2, 1] - m[1, 2]) / s
        qy = (m[0, 2] - m[2, 0]) / s
        qz = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        qw = (m[2, 1] - m[1, 2]) / s
        qx = 0.25 * s
        qy = (m[0, 1] + m[1, 0]) / s
        qz = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        qw = (m[0, 2] - m[2, 0]) / s
        qx = (m[0, 1] + m[1, 0]) / s
        qy = 0.25 * s
        qz = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        qw = (m[1, 0] - m[0, 1]) / s
        qx = (m[0, 2] + m[2, 0]) / s
        qy = (m[1, 2] + m[2, 1]) / s
        qz = 0.25 * s
    return np.array([qw, qx, qy, qz], dtype=np.float64)


def _make_texture(size: int) -> np.ndarray:
    """High-frequency, unambiguous test texture: checkerboard + colored rings + text-like
    thin lines, so a blurry reconstruction is visually obvious."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    cell = max(8, size // 24)
    for yy in range(0, size, cell):
        for xx in range(0, size, cell):
            if ((xx // cell) + (yy // cell)) % 2 == 0:
                img[yy:yy + cell, xx:xx + cell] = (235, 235, 235)
            else:
                img[yy:yy + cell, xx:xx + cell] = (20, 20, 20)
    center = (size // 2, size // 2)
    for i, r in enumerate(range(size // 16, size // 2, size // 16)):
        color = (
            int(40 + 200 * ((i * 37) % 255) / 255),
            int(40 + 200 * ((i * 91) % 255) / 255),
            int(40 + 200 * ((i * 53) % 255) / 255),
        )
        cv2.circle(img, center, r, color, thickness=max(2, size // 256))
    rng = np.random.default_rng(0)
    for _ in range(60):
        p1 = tuple(rng.integers(0, size, size=2).tolist())
        p2 = tuple(rng.integers(0, size, size=2).tolist())
        color = tuple(int(c) for c in rng.integers(0, 255, size=3))
        cv2.line(img, p1, p2, color, 1)
    cv2.putText(img, "VOTION SHARPNESS TEST", (size // 12, size // 2), cv2.FONT_HERSHEY_SIMPLEX,
                size / 900.0, (255, 0, 0), max(1, size // 400), cv2.LINE_AA)
    return img


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="synth_plane_test")
    ap.add_argument("--n-views", type=int, default=60)
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--texture-size", type=int, default=1024)
    ap.add_argument("--n-points", type=int, default=8000)
    ap.add_argument("--plane-z", type=float, default=6.0)  # keep dist/norm_radius < D.INIT_POINT_MAX_CAM_DIST_MULT
    ap.add_argument("--plane-half-extent", type=float, default=5.0)
    ap.add_argument("--rig-half-extent", type=float, default=1.3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    from engine.pano.scene_io import scene_dir

    root = Path(scene_dir(args.scene))
    colmap = root / "colmap"
    image_dir = colmap / "images"
    sparse = colmap / "sparse" / "0"
    image_dir.mkdir(parents=True, exist_ok=True)
    sparse.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    W = H = int(args.res)
    fov_deg = 60.0
    f = W / (2.0 * np.tan(np.radians(fov_deg / 2.0)))
    cx, cy = W / 2.0, H / 2.0

    texture = _make_texture(args.texture_size)
    Z0 = float(args.plane_z)
    half = float(args.plane_half_extent)
    # texture pixel (tu,tv) in [0,Tw]x[0,Th] -> world (x,y) on the z=Z0 plane, x in
    # [-half,half], y in [-half,half] (world y is "down" here, matching OpenCV +Y-down so
    # R=I cameras need no extra flip).
    Tw = Th = float(args.texture_size)
    A_tex_to_world = np.array([
        [2 * half / Tw, 0.0, -half],
        [0.0, 2 * half / Th, -half],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    n = int(args.n_views)
    side = int(np.ceil(np.sqrt(n)))
    lin = np.linspace(-args.rig_half_extent, args.rig_half_extent, side)
    centers = np.array([[x, y, 0.0] for y in lin for x in lin], dtype=np.float64)[:n]
    if centers.shape[0] < n:
        extra = centers[rng.integers(0, centers.shape[0], size=n - centers.shape[0])]
        centers = np.concatenate([centers, extra], axis=0)

    K = np.array([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]], dtype=np.float64)

    cam_id = 1
    # PINHOLE (model_id=1, [fx,fy,cx,cy]) rather than SIMPLE_PINHOLE: also readable by
    # vendor litegs/io_manager/colmap.py's own loader (which asserts model=="PINHOLE"),
    # so the same dataset can be fed through vendor's unmodified example_train.py as a
    # cross-check against our adapter.
    cameras_out = {cam_id: (1, W, H, [f, f, cx, cy])}
    images_out = []
    print(f"rendering {n} synthetic views at {W}x{H} ...")
    for i, center in enumerate(centers):
        # Small look-angle jitter for real multi-view parallax diversity (not required for
        # correctness since points3D is exact, but avoids a degenerate pure-translation rig).
        jitter = rng.normal(scale=0.03, size=3)
        # Small rotation about a random axis, applied to identity (camera nominally looks
        # straight down +Z, i.e. toward the plane).
        angle = float(np.linalg.norm(jitter))
        if angle > 1e-9:
            axis = jitter / angle
            Rj, _ = cv2.Rodrigues(axis * angle)
        else:
            Rj = np.eye(3)
        R = Rj  # world->camera rotation
        t = -R @ center
        M = np.stack([R[:, 0], R[:, 1], R[:, 2] * Z0 + t], axis=1)  # world (x,y,1) -> cam
        H_world_to_pixel = K @ M
        H_tex_to_pixel = H_world_to_pixel @ A_tex_to_world
        rendered = cv2.warpPerspective(
            texture, H_tex_to_pixel, (W, H), flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=(30, 30, 30),
        )
        name = f"view_{i:04d}.png"
        cv2.imwrite(str(image_dir / name), rendered)

        qvec = _rotmat2qvec(R)
        images_out.append({
            "id": i + 1,
            "qvec": qvec,
            "tvec": t,
            "camera_id": cam_id,
            "name": name,
            "xys": np.zeros((0, 2), dtype=np.float64),
            "point3D_ids": np.zeros((0,), dtype=np.int64),
        })
        if (i + 1) % 10 == 0 or i + 1 == n:
            print(f"  {i + 1}/{n}")

    # points3D: a grid on the plane with exact color sampled from the texture.
    n_pts = int(args.n_points)
    side_p = int(np.ceil(np.sqrt(n_pts)))
    gx = np.linspace(-half * 0.96, half * 0.96, side_p)
    gy = np.linspace(-half * 0.96, half * 0.96, side_p)
    xs, ys = np.meshgrid(gx, gy)
    xs = xs.reshape(-1)[:n_pts]
    ys = ys.reshape(-1)[:n_pts]
    tu = ((xs + half) / (2 * half) * Tw).clip(0, Tw - 1).astype(np.int64)
    tv = ((ys + half) / (2 * half) * Th).clip(0, Th - 1).astype(np.int64)
    colors = texture[tv, tu][:, ::-1]  # BGR -> RGB

    points_out = []
    for k in range(xs.shape[0]):
        points_out.append({
            "id": k + 1,
            "xyz": (float(xs[k]), float(ys[k]), Z0),
            "rgb": (int(colors[k, 0]), int(colors[k, 1]), int(colors[k, 2])),
            "error": 0.0,
            "track": [],
        })

    _write_cameras_binary(sparse / "cameras.bin", cameras_out)
    _write_images_binary(sparse / "images.bin", images_out)
    _write_points3d_binary(sparse / "points3D.bin", points_out)

    print(f"wrote {n} views to {image_dir}")
    print(f"wrote {len(points_out)} points3D + cameras/images.bin to {sparse}")
    print(f"scene root: {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
