"""Render a Votion-tagged PLY with gsplat, using the same cameras Unreal will use.

Run with the HY-World interpreter:
    .venvs\\hyworld\\Scripts\\python.exe tools\\unreal\\render_reference.py --ply ... --cams ... --out-dir ...
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ue_camera import ue_to_opencv  # noqa: E402


def _read(path: Path) -> tuple[dict[str, str], np.ndarray, list[str]]:
    tags: dict[str, str] = {}
    names: list[str] = []
    count = 0
    with path.open("rb") as handle:
        while True:
            line = handle.readline().decode("ascii", errors="replace").strip()
            if line.startswith("comment votion."):
                key, value = line.split(".", 1)[1].split("=", 1)
                tags[key] = value
            elif line.startswith("element vertex"):
                count = int(line.split()[-1])
            elif line.startswith("property float "):
                names.append(line.split()[-1])
            elif line == "end_header":
                offset = handle.tell()
                break
            elif not line:
                raise ValueError("truncated header")
    data = np.memmap(path, dtype="<f4", mode="r", offset=offset, shape=(count, len(names)))
    return tags, np.array(data, dtype=np.float32), names


def _column(data: np.ndarray, names: list[str], *keys: str) -> np.ndarray:
    index = {name: i for i, name in enumerate(names)}
    return np.stack([data[:, index[key]] for key in keys], axis=1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ply", type=Path, required=True)
    parser.add_argument("--cams", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--force-raster", default="")
    parser.add_argument("--force-near", type=float, default=0.0)
    parser.add_argument("--crop-check", default="", help="name:x,y,size")
    args = parser.parse_args()
    import torch
    import gsplat

    tags, data, names = _read(args.ply)
    if "version" not in tags and not (args.force_raster and args.force_near > 0):
        print("PLY has no votion.version tag", file=sys.stderr)
        return 2
    raster = args.force_raster or tags.get("raster", "")
    near = args.force_near if args.force_near > 0 else float(tags["near_plane"])
    degree = int(tags.get("sh_degree", "0"))
    xyz = _column(data, names, "x", "y", "z")
    scale = np.exp(_column(data, names, "scale_0", "scale_1", "scale_2"))
    rot = _column(data, names, "rot_0", "rot_1", "rot_2", "rot_3")
    dc = _column(data, names, "f_dc_0", "f_dc_1", "f_dc_2")
    rest_names = sorted((n for n in names if n.startswith("f_rest_")), key=lambda s: int(s.split("_")[-1]))
    k = (degree + 1) ** 2
    colors = np.zeros((xyz.shape[0], k, 3), dtype=np.float32)
    colors[:, 0, :] = dc
    if degree > 0 and rest_names:
        rest = _column(data, names, *rest_names[: 3 * (k - 1)])
        bands = rest.shape[1] // 3
        for coeff in range(min(bands, k - 1)):
            for channel in range(3):
                colors[:, 1 + coeff, channel] = rest[:, channel * bands + coeff]
    opacity = 1.0 / (1.0 + np.exp(-data[:, names.index("opacity")]))
    device = "cuda"
    tensors = {
        "means": torch.from_numpy(xyz).to(device),
        "quats": torch.from_numpy(rot).to(device),
        "scales": torch.from_numpy(scale).to(device),
        "opacities": torch.from_numpy(opacity).to(device),
        "colors": torch.from_numpy(colors).to(device),
    }
    mode = "antialiased" if raster == "gsplat_antialiased" else "classic"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cameras = json.loads(args.cams.read_text(encoding="utf-8"))
    crop = None
    if args.crop_check:
        name, box = args.crop_check.split(":")
        x, y, size = (int(v) for v in box.split(","))
        crop = (name, x, y, size)
    for camera in cameras:
        w2c, k = ue_to_opencv(camera["loc_cm"], camera["pitch"], camera["yaw"], camera["roll"],
                              camera["fov_h"], camera["W"], camera["H"])
        view = tensors
        if raster == "litegs_classic":
            # gsplat classic still draws a splat whose centre is far outside the frame.
            # The classic raster drops those centres (|ndc| > 1.3). The real alpine scene is covered by them.
            view = _classic_visible(tensors, w2c, k, int(camera["W"]), int(camera["H"]), near, device)
        with torch.no_grad():
            renders, _alphas, meta = gsplat.rasterization(
                **view,
                viewmats=torch.from_numpy(w2c).float().to(device)[None],
                Ks=torch.from_numpy(k).float().to(device)[None],
                width=int(camera["W"]), height=int(camera["H"]), sh_degree=degree,
                near_plane=near, far_plane=1e10, eps2d=0.3, radius_clip=0.0, packed=False,
                rasterize_mode=mode,
            )
        image = renders[0, ..., :3].clamp(0, 1).float().cpu().numpy()
        np.save(args.out_dir / f"{camera['name']}.npy", image.astype(np.float32))
        _png(args.out_dir / f"{camera['name']}.png", image)
        if crop and crop[0] == camera["name"]:
            _crop(args.out_dir, camera["name"], image, meta, tensors, degree, x=crop[1], y=crop[2], size=crop[3])
        print(camera["name"], image.shape)
    return 0


def _classic_visible(tensors: dict, w2c: np.ndarray, k: np.ndarray, width: int, height: int,
                    near: float, device: str) -> dict:
    import torch

    means = tensors["means"]
    hom = torch.cat([means, torch.ones((means.shape[0], 1), device=device)], dim=1)
    cam = hom @ torch.from_numpy(np.asarray(w2c, dtype=np.float32).T).to(device)
    z = cam[:, 2]
    u = float(k[0, 0]) * cam[:, 0] / z.clamp(min=1e-8) + float(k[0, 2])
    v = float(k[1, 1]) * cam[:, 1] / z.clamp(min=1e-8) + float(k[1, 2])
    ndc_x = u / width * 2.0 - 1.0
    ndc_y = v / height * 2.0 - 1.0
    keep = (z > near) & (ndc_x.abs() <= 1.3) & (ndc_y.abs() <= 1.3)
    return {key: value[keep] for key, value in tensors.items()}


def _png(path: Path, image: np.ndarray) -> None:
    import imageio.v2 as imageio
    imageio.imwrite(path, np.clip(np.round(image * 255.0), 0, 255).astype(np.uint8))


def _crop(out: Path, name: str, image: np.ndarray, meta: dict, tensors: dict, degree: int, x: int, y: int, size: int) -> None:
    import torch
    means = meta["means2d"][0, 0].detach().float().cpu().numpy()
    conic = meta["conics"][0, 0].detach().float().cpu().numpy()
    opac = meta["opacities"][0, 0].detach().float().cpu().numpy()
    depth = meta["depths"][0, 0].detach().float().cpu().numpy()
    # gsplat already folds antialias compensation into meta opacities when it is present.
    # Degree 0 only. Higher bands are not reconstructed here; the crop is a termination check.
    colours = np.clip(tensors["colors"].detach().float().cpu().numpy()[:, 0, :] * 0.28209479177387814 + 0.5, 0.0, None)
    x1, y1 = x + size, y + size
    keep = (means[:, 0] > x - 64) & (means[:, 0] < x1 + 64) & (means[:, 1] > y - 64) & (means[:, 1] < y1 + 64)
    keep &= opac >= 1.0 / 255.0
    order = np.argsort(depth[keep], kind="stable")[::-1]  # far to near, back-to-front blend
    acc = np.zeros((size, size, 3), dtype=np.float64)
    sel_m, sel_c, sel_o, sel_col = means[keep][order], conic[keep][order], opac[keep][order], colours[keep][order]
    ys, xs = np.mgrid[y:y1, x:x1]
    px, py = xs + 0.5, ys + 0.5
    for mean, con, opa, col in zip(sel_m, sel_c, sel_o, sel_col):
        dx, dy = mean[0] - px, mean[1] - py
        sigma = 0.5 * (con[0] * dx * dx + con[2] * dy * dy) + con[1] * dx * dy
        alpha = np.minimum(0.999, opa * np.exp(-np.maximum(sigma, 0.0)))
        alpha = np.where((sigma >= 0) & (alpha >= 1.0 / 255.0), alpha, 0.0)[..., None]
        acc = col * alpha + acc * (1.0 - alpha)
    np.save(out / f"{name}_crop_notermination.npy", np.clip(acc, 0, 1).astype(np.float32))
    np.save(out / f"{name}_crop_gsplat.npy", image[y:y1, x:x1].astype(np.float32))


if __name__ == "__main__":
    raise SystemExit(main())
