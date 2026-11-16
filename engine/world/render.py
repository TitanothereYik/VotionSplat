"""Rasterize world pieces with pytorch3d. The Qt app must not import this module."""

from __future__ import annotations

import math
import os
import sys
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pytorch3d.ops import interpolate_face_attributes
from pytorch3d.renderer import MeshRasterizer, PerspectiveCameras, RasterizationSettings
from pytorch3d.structures import Meshes

from engine.world.defaults import (
    BIN_SIZE,
    GROUND_FILL_MAX,
    GROUND_FILL_MIN,
    GROUND_FILL_REL,
    GUIDE_FEATHER_PX,
    HOLE_CLOSE_PX,
    HOLE_GROW_PX,
    MAX_FACES_PER_BIN,
    MAX_FACES_PER_BIN_CAP,
    RASTER_BIN_BUDGET,
    SLIVER_PX,
)
from engine.world.pano_mesh import GroundPlane, Piece


@dataclass
class WorldView:
    rgb: np.ndarray  # (H, W, 3) uint8; black where empty
    empty: np.ndarray  # (H, W) bool; True = no surface
    depth: np.ndarray  # (H, W) float32 camera z; NaN where empty
    source: np.ndarray  # (H, W) int8; 2 = pano, 1 = layer, 0 = empty


class _BinsGrew(Exception):
    """The bin limit doubled past the memory budget for this band; tile again."""


class WorldRenderer:
    def __init__(
        self,
        pieces: list[Piece],
        textures: dict[str, np.ndarray],
        device: str = "cuda:0",
        bin_size: int | None = BIN_SIZE,
        max_faces_per_bin: int = MAX_FACES_PER_BIN,
        ground: GroundPlane | tuple[np.ndarray, float] | None = None,
    ) -> None:
        self.device = torch.device(device)
        self.bin_size = bin_size
        self.max_faces_per_bin = int(max_faces_per_bin)
        self.ground_normal: torch.Tensor | None = None
        self.ground_d: float | None = None
        self.ground_erp: torch.Tensor | None = None
        normal_np = None
        plane_d = None
        erp_dist = None
        if isinstance(ground, GroundPlane):
            normal_np = np.asarray(ground.normal, dtype=np.float32).reshape(3)
            plane_d = float(ground.d)
            erp_dist = ground.erp_dist
        elif ground is not None:
            normal_np = np.asarray(ground[0], dtype=np.float32).reshape(3)
            plane_d = float(ground[1])
        if normal_np is not None and plane_d is not None:
            length = float(np.linalg.norm(normal_np))
            if length > 1e-8 and np.isfinite(normal_np).all() and np.isfinite(plane_d):
                self.ground_normal = torch.tensor(normal_np / length, dtype=torch.float32, device=self.device)
                self.ground_d = float(plane_d)
        if erp_dist is not None:
            grid = np.asarray(erp_dist, dtype=np.float32)[None, None]
            self.ground_erp = torch.from_numpy(np.ascontiguousarray(grid)).to(self.device)
        self._tex: dict[str, torch.Tensor] = {}
        self._tex_keys: list[str] = []
        self.verts = torch.zeros((0, 3), dtype=torch.float32, device=self.device)
        self.faces = torch.zeros((0, 3), dtype=torch.int64, device=self.device)
        self.face_uvs = torch.zeros((0, 3, 2), dtype=torch.float32, device=self.device)
        self.face_tex = torch.zeros((0,), dtype=torch.int64, device=self.device)
        for piece in pieces:
            image = textures.get(piece.tex)
            if image is None:
                raise KeyError(piece.tex)
            self.add_piece(piece, image)

    def set_texture(self, key: str, texture: np.ndarray) -> None:
        self._tex[key] = _texture_tensor(texture, self.device)
        if key not in self._tex_keys:
            self._tex_keys.append(key)

    def add_piece(self, piece: Piece, texture: np.ndarray) -> None:
        self.set_texture(piece.tex, texture)
        key_index = self._tex_keys.index(piece.tex)
        faces_np = np.asarray(piece.faces, dtype=np.int64)
        if faces_np.size == 0:
            return
        verts = torch.as_tensor(np.asarray(piece.verts, dtype=np.float32), device=self.device)
        offset = int(self.verts.shape[0])
        faces = torch.as_tensor(faces_np, device=self.device) + offset
        uvs = np.asarray(piece.uvs, dtype=np.float32)[faces_np]
        face_uvs = torch.as_tensor(uvs, device=self.device)
        face_tex = torch.full((faces_np.shape[0],), key_index, dtype=torch.int64, device=self.device)
        self.verts = torch.cat([self.verts, verts], dim=0)
        self.faces = torch.cat([self.faces, faces], dim=0)
        self.face_uvs = torch.cat([self.face_uvs, face_uvs], dim=0)
        self.face_tex = torch.cat([self.face_tex, face_tex], dim=0)

    def render(
        self,
        w2c: np.ndarray,
        K: np.ndarray,
        height: int,
        width: int,
        supersample: int = 1,
    ) -> WorldView:
        ss = max(1, int(supersample))
        if int(self.faces.shape[0]) == 0:
            return _empty_view(height, width)
        while True:
            tiles = self._tile_count(int(height) * ss, int(width) * ss)
            try:
                view = self._render_tiles(w2c, K, height, width, ss, tiles)
            except _BinsGrew:
                continue
            return self._fill_ground(view, w2c, K)

    def _render_tiles(
        self,
        w2c: np.ndarray,
        K: np.ndarray,
        height: int,
        width: int,
        ss: int,
        tiles: int,
    ) -> WorldView:
        if tiles == 1:
            return self._render_camera(w2c, K, height, width, ss)
        parts: list[WorldView] = []
        tile_h = int(math.ceil(int(height) / tiles))
        y = 0
        while y < int(height):
            band = min(tile_h, int(height) - y)
            k_tile = np.array(K, dtype=np.float64, copy=True)
            k_tile[1, 2] = float(k_tile[1, 2]) - float(y)
            parts.append(self._render_camera(w2c, k_tile, band, width, ss))
            y += band
        return WorldView(
            rgb=np.concatenate([part.rgb for part in parts], axis=0),
            empty=np.concatenate([part.empty for part in parts], axis=0),
            depth=np.concatenate([part.depth for part in parts], axis=0),
            source=np.concatenate([part.source for part in parts], axis=0),
        )

    def _bin_size_for(self, height: int, width: int) -> int:
        """Bin edge pytorch3d accepts for this image.

        Its CUDA kernel refuses a bin grid longer than 21 on a side. HiRes at
        2× is 1920×3328, and a 128 px bin makes that grid 26 long.
        """
        base = int(self.bin_size or 0)
        if base <= 0:
            return 0
        side = max(int(height), int(width), 1)
        return max(base, (side - 1 + 19) // 20)

    def _tile_count(self, height: int, width: int) -> int:
        bin_size = self._bin_size_for(height, width)
        if bin_size <= 0:
            return 1
        bins = ((int(height) + bin_size - 1) // bin_size) * ((int(width) + bin_size - 1) // bin_size)
        estimate = bins * int(self.max_faces_per_bin) * 4
        if estimate <= int(RASTER_BIN_BUDGET):
            return 1
        return max(1, int(math.ceil(estimate / float(RASTER_BIN_BUDGET))))

    def _render_camera(
        self,
        w2c: np.ndarray,
        K: np.ndarray,
        height: int,
        width: int,
        ss: int,
    ) -> WorldView:
        high_h = int(height) * ss
        high_w = int(width) * ss
        cameras = _cameras(w2c, K, high_h, high_w, ss, self.device)
        meshes = Meshes(verts=[self.verts], faces=[self.faces])
        fragments = None
        while fragments is None:
            # cull_to_frustum is left off. This pytorch3d compares one vertex's xyz with a
            # single plane (face_verts[:, axis] instead of face_verts[:, :, axis]), so a
            # triangle that covers the view can be dropped.
            settings = RasterizationSettings(
                image_size=(high_h, high_w),
                blur_radius=0.0,
                faces_per_pixel=1,
                bin_size=self._bin_size_for(high_h, high_w),
                max_faces_per_bin=self.max_faces_per_bin,
                cull_backfaces=False,
                perspective_correct=True,
                z_clip_value=1e-3,
                cull_to_frustum=False,
            )
            rasterizer = MeshRasterizer(cameras=cameras, raster_settings=settings)
            try:
                fragments = _rasterize(rasterizer, meshes)
            except RuntimeError as exc:
                if "MAX_FACES_PER_BIN" not in str(exc) or self.max_faces_per_bin >= MAX_FACES_PER_BIN_CAP:
                    raise
                self.max_faces_per_bin *= 2
                print(f"STATUS // raster bins {self.max_faces_per_bin}", flush=True)
                if self._tile_count(high_h, high_w) > 1:
                    raise _BinsGrew
        pix = fragments.pix_to_face[0, ..., 0]
        zbuf = fragments.zbuf[0, ..., 0]
        uv_map = interpolate_face_attributes(
            fragments.pix_to_face, fragments.bary_coords, self.face_uvs
        )[0, :, :, 0, :]
        rgb_t, source_t = self._sample(pix, uv_map)
        rgb = rgb_t.detach().clamp(0, 1).mul(255).byte().cpu().permute(1, 2, 0).numpy()
        source = source_t.detach().cpu().numpy().astype(np.int8)
        depth = zbuf.detach().float().cpu().numpy()
        covered = pix.detach().cpu().numpy() >= 0
        if ss == 1:
            empty = ~covered
            depth = depth.astype(np.float32)
            depth[empty] = np.nan
            rgb[empty] = 0
            source[empty] = 0
            view = WorldView(rgb, empty, depth, source)
        else:
            rgb, empty, depth, source_img = _resize_supersample(rgb, covered, depth, source, width, height)
            view = WorldView(rgb, empty, depth, source_img)
        return view

    def _sample(self, pix: torch.Tensor, uv_map: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        height, width = pix.shape
        rgb = torch.zeros((3, height, width), dtype=torch.float32, device=self.device)
        source = torch.zeros((height, width), dtype=torch.int8, device=self.device)
        valid = pix >= 0
        if not bool(valid.any()):
            return rgb, source
        flat_pix = pix[valid]
        uv = uv_map[valid].clone()
        uv[:, 0] = uv[:, 0].clamp(0.0, 1.0)
        grid = uv.mul(2.0).sub(1.0).view(1, 1, -1, 2)
        tex_ids = self.face_tex[flat_pix]
        pano_index = self._tex_keys.index("pano") if "pano" in self._tex_keys else -1
        present = torch.unique(tex_ids)
        for key_index in present.tolist():
            key = self._tex_keys[int(key_index)]
            choose = tex_ids == int(key_index)
            if not bool(choose.any()):
                continue
            tex = self._tex[key]
            sample_grid = grid[:, :, choose, :].to(dtype=tex.dtype)
            sampled = F.grid_sample(
                tex, sample_grid, mode="bilinear", padding_mode="border", align_corners=False
            )
            color = sampled[0, :, 0, :].float()
            ys, xs = torch.where(valid)
            rgb[:, ys[choose], xs[choose]] = color
            source[ys[choose], xs[choose]] = 2 if int(key_index) == pano_index else 1
        return rgb, source

    def _fill_ground(self, view: WorldView, w2c: np.ndarray, K: np.ndarray) -> WorldView:
        """Paint empty pixels that land on the ground plane with the panorama."""
        if self.ground_normal is None or self.ground_d is None or "pano" not in self._tex_keys:
            return view
        ys, xs = np.nonzero(view.empty)
        if ys.size == 0:
            return view
        device = self.device
        w2c_t = torch.as_tensor(np.asarray(w2c, dtype=np.float32), device=device)
        c2w = torch.linalg.inv(w2c_t)
        eye = c2w[:3, 3]
        rot = c2w[:3, :3]
        k_inv = torch.linalg.inv(torch.as_tensor(np.asarray(K, dtype=np.float32), device=device))
        pix = torch.stack(
            [
                torch.as_tensor(xs + 0.5, dtype=torch.float32, device=device),
                torch.as_tensor(ys + 0.5, dtype=torch.float32, device=device),
                torch.ones(int(ys.size), dtype=torch.float32, device=device),
            ],
            dim=1,
        )
        dirs = (pix @ k_inv.T) @ rot.T
        dirs = dirs / dirs.norm(dim=1, keepdim=True).clamp_min(1e-8)
        denom = dirs @ self.ground_normal
        plane = torch.dot(self.ground_normal, eye) + float(self.ground_d)
        t = torch.full((dirs.shape[0],), float("nan"), device=device)
        aimed = denom.abs() > 1e-4
        t[aimed] = -plane / denom[aimed]
        take = aimed & (t > float(GROUND_FILL_MIN)) & (t < float(GROUND_FILL_MAX))
        if not bool(take.any()):
            return view
        hits = eye.unsqueeze(0) + dirs[take] * t[take].unsqueeze(1)
        cam = (w2c_t[:3, :3] @ hits.T).T + w2c_t[:3, 3]
        front = cam[:, 2] > 1e-3
        if self.ground_erp is not None and bool(front.any()):
            seen = _sample_erp_nearest(self.ground_erp, hits[front])
            radius = hits[front].norm(dim=1).clamp_min(1e-6)
            agree = torch.isfinite(seen) & ((seen - radius).abs() / radius < float(GROUND_FILL_REL))
            front_idx = torch.nonzero(front, as_tuple=False).flatten()
            front = torch.zeros_like(front)
            front[front_idx[agree]] = True
        if not bool(front.any()):
            return view
        colors = _sample_erp(self._tex["pano"], hits[front])
        chosen = np.flatnonzero(take.detach().cpu().numpy())[front.detach().cpu().numpy()]
        view.rgb[ys[chosen], xs[chosen]] = colors.detach().clamp(0, 1).mul(255).byte().cpu().numpy()
        view.empty[ys[chosen], xs[chosen]] = False
        view.source[ys[chosen], xs[chosen]] = 2
        view.depth[ys[chosen], xs[chosen]] = cam[front, 2].detach().float().cpu().numpy()
        return view

    def close(self) -> None:
        self.verts = torch.zeros((0, 3), device="cpu")
        self.faces = torch.zeros((0, 3), dtype=torch.int64, device="cpu")
        self.face_uvs = torch.zeros((0, 3, 2), device="cpu")
        self.face_tex = torch.zeros((0,), dtype=torch.int64, device="cpu")
        self._tex.clear()
        self.ground_normal = None
        self.ground_erp = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def clean_guide(rgb: np.ndarray, empty: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Turn thin hole streaks into regions the generator can fill, and soften the edge."""
    hole = np.asarray(empty, dtype=bool)
    close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (HOLE_CLOSE_PX * 2 + 1, HOLE_CLOSE_PX * 2 + 1))
    hole = cv2.morphologyEx(hole.astype(np.uint8) * 255, cv2.MORPH_CLOSE, close) > 0
    covered = ~hole
    opened = cv2.morphologyEx(
        covered.astype(np.uint8) * 255,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (SLIVER_PX, SLIVER_PX)),
    ) > 0
    removed = covered & ~opened
    hole_frac = cv2.boxFilter(hole.astype(np.float32), -1, (7, 7), normalize=True)
    hole = hole | (removed & (hole_frac > 0.35))
    grow = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (HOLE_GROW_PX * 2 + 1, HOLE_GROW_PX * 2 + 1))
    hole = cv2.dilate(hole.astype(np.uint8) * 255, grow) > 0
    source = np.ascontiguousarray(rgb)
    filled = cv2.inpaint(source, (hole.astype(np.uint8) * 255), 5, cv2.INPAINT_TELEA)
    dist = cv2.distanceTransform((~hole).astype(np.uint8), cv2.DIST_L2, 3)
    ring = np.clip(dist / float(GUIDE_FEATHER_PX), 0.0, 1.0).astype(np.float32)[..., None]
    blurred = cv2.GaussianBlur(filled, (0, 0), 3).astype(np.float32)
    mixed = ring * source.astype(np.float32) + (1.0 - ring) * blurred
    mixed[hole] = filled[hole]
    return np.clip(np.rint(mixed), 0, 255).astype(np.uint8), hole


def _resize_supersample(
    rgb: np.ndarray,
    covered: np.ndarray,
    depth: np.ndarray,
    source: np.ndarray,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Average the supersampled frame without mixing empty black into a hit.

    A layer triangle covers the guide pixel centre and misses the other
    samples. Averaging those zeros turned the hit into a black speck.
    """
    mask = np.asarray(covered, dtype=bool)
    weight = mask.astype(np.float32)
    coverage = cv2.resize(weight, (int(width), int(height)), interpolation=cv2.INTER_AREA)
    empty = coverage < 0.5
    safe = np.maximum(coverage, 1e-6)
    summed = cv2.resize(rgb.astype(np.float32) * weight[..., None], (int(width), int(height)), interpolation=cv2.INTER_AREA)
    rgb_out = np.clip(np.rint(summed / safe[..., None]), 0, 255).astype(np.uint8)
    rgb_out[empty] = 0
    src = np.where(mask, np.asarray(source, dtype=np.float32), 0.0)
    source_out = np.rint(
        cv2.resize(src, (int(width), int(height)), interpolation=cv2.INTER_AREA) / safe
    ).astype(np.int8)
    source_out[empty] = 0
    z = np.where(mask, np.nan_to_num(np.asarray(depth, dtype=np.float32), nan=0.0), 0.0)
    depth_out = (
        cv2.resize(z, (int(width), int(height)), interpolation=cv2.INTER_AREA) / safe
    ).astype(np.float32)
    depth_out[empty] = np.nan
    return rgb_out, empty, depth_out, source_out


def _sample_erp_nearest(texture: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    grid = _erp_grid(points)
    sampled = F.grid_sample(
        texture.float(),
        grid,
        mode="nearest",
        padding_mode="border",
        align_corners=False,
    )
    return sampled[0, 0, 0]


def _erp_grid(points: torch.Tensor) -> torch.Tensor:
    direction = points / points.norm(dim=1, keepdim=True).clamp_min(1e-8)
    x, y, z = direction.unbind(dim=1)
    lat = torch.asin((-y).clamp(-1.0, 1.0))
    lon = torch.atan2(x, z)
    u = (lon + torch.pi) / (2.0 * torch.pi)
    v = (torch.pi / 2.0 - lat) / torch.pi
    return torch.stack((u.mul(2.0).sub(1.0), v.mul(2.0).sub(1.0)), dim=-1).view(1, 1, -1, 2)


def _sample_erp(texture: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """Panorama colour at each world point, looked up from the origin."""
    grid = _erp_grid(points)
    sampled = F.grid_sample(
        texture.float(),
        grid.to(dtype=torch.float32),
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )
    return sampled[0, :, 0, :].T


def _rasterize(rasterizer: MeshRasterizer, meshes: Meshes):
    """Rasterize, and fail if the binned rasterizer dropped faces.

    The overflow notice is a C print, so the Python warning filter never sees it.
    """
    sys.stdout.flush()
    sys.stderr.flush()
    saved_out = os.dup(1)
    saved_err = os.dup(2)
    fd, name = tempfile.mkstemp(prefix="votion-raster-")
    caught: list = []
    fragments = None
    error: Exception | None = None
    try:
        os.dup2(fd, 1)
        os.dup2(fd, 2)
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                fragments = rasterizer(meshes)
            sys.stdout.flush()
            sys.stderr.flush()
            _flush_c_stdio()
        except Exception as exc:
            error = exc
        finally:
            os.dup2(saved_out, 1)
            os.dup2(saved_err, 2)
    finally:
        os.close(saved_out)
        os.close(saved_err)
        os.close(fd)
        try:
            with open(name, "rb") as handle:
                blob = handle.read(262144)
        finally:
            Path(name).unlink(missing_ok=True)
    warned = b"Bin size was too small" in blob or any(
        "Bin size was too small" in str(item.message) for item in caught
    )
    if warned:
        raise RuntimeError("raise MAX_FACES_PER_BIN")
    if error is not None:
        raise error
    return fragments


def _flush_c_stdio() -> None:
    try:
        import ctypes

        ctypes.CDLL("ucrtbase").fflush(None)
    except (OSError, AttributeError):
        return


def _texture_tensor(image: np.ndarray, device: torch.device) -> torch.Tensor:
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)
    if arr.dtype == np.uint8:
        arr = arr.astype(np.float32) / 255.0
    else:
        arr = np.asarray(arr, dtype=np.float32)
        if arr.size and float(np.nanmax(arr)) > 1.5:
            arr = arr / 255.0
    tensor = torch.from_numpy(np.ascontiguousarray(arr[..., :3])).permute(2, 0, 1)[None]
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    return tensor.to(device=device, dtype=dtype)


def _cameras(
    w2c: np.ndarray,
    K: np.ndarray,
    height: int,
    width: int,
    supersample: int,
    device: torch.device,
) -> PerspectiveCameras:
    w2c_t = torch.tensor(np.asarray(w2c, dtype=np.float32), dtype=torch.float32, device=device)[None]
    c2w = torch.linalg.inv(w2c_t)
    c2w[:, :, 0] = -c2w[:, :, 0]
    c2w[:, :, 1] = -c2w[:, :, 1]
    w2c_gl = torch.linalg.inv(c2w)
    ks = np.array(K, dtype=np.float32, copy=True)
    ks[0, :] *= supersample
    ks[1, :] *= supersample
    return PerspectiveCameras(
        focal_length=torch.tensor([[ks[0, 0], ks[1, 1]]], dtype=torch.float32, device=device),
        principal_point=torch.tensor([[ks[0, 2], ks[1, 2]]], dtype=torch.float32, device=device),
        R=c2w[:, :3, :3],
        T=w2c_gl[:, :3, 3],
        in_ndc=False,
        image_size=torch.tensor([[height, width]], dtype=torch.float32, device=device),
        device=device,
    )


def _empty_view(height: int, width: int) -> WorldView:
    return WorldView(
        rgb=np.zeros((height, width, 3), np.uint8),
        empty=np.ones((height, width), dtype=bool),
        depth=np.full((height, width), np.nan, np.float32),
        source=np.zeros((height, width), np.int8),
    )
