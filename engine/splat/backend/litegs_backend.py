"""LiteGS adapter: COLMAP load, Gaussians, optimizer, render, loss, checkpoint, ply.

Imports torch and the vendored LiteGS (``vendor/litegs`` at ``vendor/litegs.sha``).
Qt never imports this module. ``spike_one_step()`` is the WP0 gate used by
``tools/build_litegs.py``.

What is LiteGS and what is Votion here
---------------------------------------
* LiteGS (as-is): ``scene.create_gaussians`` (simple_knn scale init), ``scene.cluster``
  (128-point chunks, Morton ``spatial_refine``), ``training.optimizer`` (SparseGaussianAdam,
  xyz LR scaled by ``norm_radius``), ``render.render_preprocess`` (frustum cull + compact +
  activate + SH), the ``utils.wrapper`` CUDA ops, ``fused_ssim.fused_l1_ssim_loss``,
  ``training.densify`` controllers (TamingGS is the LiteGS default).
* Votion: COLMAP load through ``engine.splat.colmap_io`` (SphereSfM cube faces are
  SIMPLE_PINHOLE, which LiteGS's own loader drops), image preload at a resolution scale,
  ``_render_core`` (a line-for-line copy of ``litegs.render.render`` that additionally
  registers a mean2d-gradient statistic for the MCMC Compact controller), the free
  viewport camera, background composite, checkpoint **with optimizer and scheduler**,
  and a fast full-SH PLY writer (``engine.splat.ply_export``).

Layouts (LiteGS): every per-Gaussian tensor is ``[..., chunks, chunk_size]``; unclustered
it is ``[..., N]``. xyz ``[3,N]``, scale (log) ``[3,N]``, rot (wxyz, unnormalized) ``[4,N]``,
sh_0 ``[1,3,N]``, sh_rest ``[(deg+1)^2-1,3,N]``, opacity (logit) ``[1,N]``. Matrices are
DX row-vector style: ``view = w2c.T``; ``proj`` from ``_proj_matrix``.
"""

from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from engine.splat import defaults as D
from engine.splat.backend import litegs_env as LE

CKPT_FORMAT = "votion-litegs-1"
CLUSTER_SIZE = 128
Z_NEAR = 0.01
Z_FAR = 5000.0
_READY = False


def ensure_litegs() -> list[str]:
    """MSVC / CUDA env + sys.path so ``import litegs`` and its CUDA modules resolve."""
    global _READY
    notes: list[str] = []
    if _READY:
        return notes
    from engine.cuda_jit import prepare

    notes.extend(prepare())
    notes.extend(LE.add_litegs_paths())
    _READY = True
    return notes


def _proj_matrix(
    fx: float, fy: float, width: int, height: int, cx: float | None = None, cy: float | None = None
) -> np.ndarray:
    """LiteGS ``PinHoleCameraInfo.proj_matrix`` (row-vector, transposed), fov from focal,
    plus the principal-point shift LiteGS omits (zero for centred SphereSfM cube faces)."""
    width = float(width)
    height = float(height)
    rx = float(fx) / (width * 0.5)
    ry = float(fy) / (height * 0.5)
    sx = 0.0 if cx is None else 2.0 * float(cx) / width - 1.0
    sy = 0.0 if cy is None else 2.0 * float(cy) / height - 1.0
    proj = np.array(
        [
            [rx, 0.0, sx, 0.0],
            [0.0, ry, sy, 0.0],
            [0.0, 0.0, Z_FAR / (Z_FAR - Z_NEAR), -Z_FAR * Z_NEAR / (Z_FAR - Z_NEAR)],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    return proj.T.copy()


def _cuda_packed(t):
    """LiteGS CUDA ``packed_accessor32`` ignores PyTorch strides and reads C-order.

    ``w2c.T`` is F-order numpy; ``torch.tensor(...).unsqueeze(0)`` keeps stride
    ``(4, 1, 4)`` instead of ``(16, 4, 1)``. Logical values still match a
    DataLoader camera (``tensor.equal`` is true) but MVPTransform / SH dirs
    see a transposed view matrix. Training then cannot lock to GT: synth PSNR
    stuck ~8 with a blob field, while a vendor ply looks sharp but misaligned.
    """
    return t.contiguous()


@dataclass
class TrainView:
    index: int
    name: str
    width: int  # after resolution scale
    height: int
    w2c: np.ndarray  # 4x4 OpenCV
    K: np.ndarray  # 3x3 at full resolution
    image_path: Path
    view: "object"  # torch [1,4,4] cuda (= w2c.T)
    proj: "object"  # torch [1,4,4] cuda
    frustum: "object"  # torch [1,6,4] cuda
    holdout: bool = False
    image: "object | None" = None  # torch uint8 [1,3,H,W] cuda

    @property
    def center(self) -> np.ndarray:
        r, t = self.w2c[:3, :3], self.w2c[:3, 3]
        return (-r.T @ t).astype(np.float64)

    @property
    def rail(self) -> str:
        """Group key: ``rail_00_front`` from ``rail_00_front_f000.png``, else ``rail_03``."""
        stem = Path(self.name).stem
        parts = stem.split("_")
        if len(parts) >= 4 and parts[0] == "rail" and parts[3].startswith("f"):
            return "_".join(parts[:3])
        if len(parts) >= 2 and parts[0].startswith("rail"):
            return "_".join(parts[:2])
        return parts[0] if parts else stem


class LiteGSBackend:
    def __init__(
        self,
        scene_dir: Path,
        device,
        *,
        sh_degree: int = 3,
        resolution_scale: float = 1.0,
        holdout: bool = False,
        cluster_size: int = CLUSTER_SIZE,
        colmap_dir: Path | None = None,
        log: Callable[[str], None] = print,
    ) -> None:
        ensure_litegs()
        import torch
        import fused_ssim
        import litegs  # noqa: F401
        from litegs import arguments, render, scene, utils
        from litegs.training import densify, optimizer
        from litegs.utils.statistic_helper import StatisticsHelperInst

        self.torch = torch
        self.fused_ssim = fused_ssim
        self.lg_arguments = arguments
        self.lg_render = render
        self.lg_scene = scene
        self.lg_utils = utils
        self.lg_densify = densify
        self.lg_optimizer = optimizer
        self.stats = StatisticsHelperInst
        self.log = log

        self.scene_dir = Path(scene_dir)
        self.colmap_dir = Path(colmap_dir) if colmap_dir is not None else self.scene_dir / "colmap"
        self.device = device
        self.sh_degree = int(sh_degree)
        self.resolution_scale = float(resolution_scale)
        self.holdout_enabled = bool(holdout)

        self.pp = arguments.PipelineParams.get_class_default_obj()
        self.pp.cluster_size = int(cluster_size)
        self.pp.sparse_grad = True
        self.pp.device_preload = True
        # Vendor example_train.py leaves this at the dataclass default (False). The
        # trainer.py path that adds `(1-trans).abs().mean()` is opt-in. Turning it on
        # here previously looked like a fix for a dark synthetic render (PSNR~8, ~45% as
        # bright as GT) -- that darkness was TamingGS deleting Gaussians not seen in the
        # current densify cycle (see LiteGSController._guard_unseen_prune). The regularizer
        # then filled the holes with large opaque Gaussians (haze). Match vendor training.
        self.pp.enable_transmitance = False
        self.pp_view = arguments.PipelineParams.get_class_default_obj()
        self.pp_view.cluster_size = int(cluster_size)
        self.pp_view.sparse_grad = True
        # Viewport only: leftover transmittance composites against VIEW_BG (navy), so
        # holes read as background instead of black. Not used as a training loss.
        self.pp_view.enable_transmitance = True
        self.op = arguments.OptimizationParams.get_class_default_obj()
        self.dp = arguments.DensifyParams.get_class_default_obj()

        self.views: list[TrainView] = []
        self.train_views: list[TrainView] = []
        self.holdout_views: list[TrainView] = []
        self.norm_radius = 1.0
        self.cam_center = np.zeros(3, dtype=np.float64)
        self.scene_scale = 1.0
        self.init_points = 0
        self.opt = None
        self.sched = None
        self._feedback_chunks = None
        self._feedback_binning = None
        self._aabb = None
        self._mean2d_wanted = False
        self.preload_device = "cuda"

    # ------------------------------------------------------------------ loading
    def load_colmap(self) -> None:
        from engine.splat.colmap_io import (
            camera_centroid_from_views,
            load_colmap_views,
            load_points3d,
            scene_scale_from_points,
            scene_scale_from_views,
        )

        torch = self.torch
        colmap = self.colmap_dir
        raw_views = load_colmap_views(colmap)
        xyz, rgb = load_points3d(colmap)
        self.cam_center = camera_centroid_from_views(raw_views)
        self.scene_scale = max(scene_scale_from_views(raw_views), scene_scale_from_points(xyz))

        centers = np.stack([-(v.w2c[:3, :3].T @ v.w2c[:3, 3]) for v in raw_views], axis=0)
        # LiteGS CameraFrameDataset.get_norm: radius = 1.1 * max distance to the camera centroid.
        diag = float(np.max(np.linalg.norm(centers - centers.mean(axis=0, keepdims=True), axis=1)))
        self.norm_radius = max(diag * 1.1, 1e-3)

        n_before = int(xyz.shape[0])
        mult = float(D.INIT_POINT_MAX_CAM_DIST_MULT)
        if mult > 0 and n_before > 0:
            # Distance to the *nearest* camera, not the camera-cluster centroid.
            # SphereSfM / forward-facing rigs sit in a tight cluster looking *out*
            # at content (a building facade, this project's synthetic plane at z=6)
            # whose centroid-distance exceeds the rig radius even when every point
            # is a legitimate, well-observed reconstruction. Using cam_center here
            # dropped 542/8000 corners of the known-geometry plane (norm_radius 2.05,
            # plane at z=6) and would similarly strip real facades on alpine_01.
            cam = centers.astype(np.float64, copy=False)
            pts = xyz.astype(np.float64, copy=False)
            nearest = np.full((n_before,), np.inf, dtype=np.float64)
            step = 4096
            for i0 in range(0, n_before, step):
                sl = pts[i0 : i0 + step, None, :]  # [B,1,3]
                d = np.sqrt(((sl - cam[None, :, :]) ** 2).sum(axis=2))  # [B,C]
                nearest[i0 : i0 + step] = d.min(axis=1)
            keep = nearest <= (mult * self.norm_radius)
            n_kept = int(keep.sum())
            if 0 < n_kept < n_before:
                xyz, rgb = xyz[keep], rgb[keep]
                self.log(
                    f"STATUS // dropped {n_before - n_kept} points3D beyond "
                    f"{mult:.1f}x norm_radius of the nearest camera ({n_before} -> {n_kept}; "
                    "low-parallax / reflection-confused SfM outliers, see defaults.INIT_POINT_MAX_CAM_DIST_MULT)"
                )
        self._init_xyz = xyz
        self._init_rgb = rgb
        self.init_points = int(xyz.shape[0])

        s = self.resolution_scale
        self.views = []
        for i, v in enumerate(raw_views):
            w = max(8, int(round(v.width * s)))
            h = max(8, int(round(v.height * s)))
            fx, fy = float(v.K[0, 0]), float(v.K[1, 1])
            cx, cy = float(v.K[0, 2]), float(v.K[1, 2])
            view = _cuda_packed(
                torch.tensor(np.ascontiguousarray(v.w2c.T, dtype=np.float32), device=self.device).unsqueeze(0)
            )
            proj = _cuda_packed(
                torch.tensor(
                    _proj_matrix(fx, fy, v.width, v.height, cx, cy), device=self.device
                ).unsqueeze(0)
            )
            frustum = _cuda_packed(self.lg_utils.viewproj_to_frustumplane(view @ proj))
            hold = self.holdout_enabled and (i % 8 == 0) and len(raw_views) >= 16
            self.views.append(
                TrainView(
                    index=i,
                    name=v.name,
                    width=w,
                    height=h,
                    w2c=v.w2c,
                    K=v.K,
                    image_path=v.image_path,
                    view=view,
                    proj=proj,
                    frustum=frustum,
                    holdout=hold,
                )
            )
        self.train_views = [v for v in self.views if not v.holdout]
        self.holdout_views = [v for v in self.views if v.holdout]
        n = len(self.views)
        self._feedback_chunks = torch.zeros((n,), dtype=torch.int32).pin_memory()
        self._feedback_binning = torch.zeros((n,), dtype=torch.int32).pin_memory()

    def gt_bytes(self) -> int:
        return int(sum(v.width * v.height * 3 for v in self.views))

    def choose_preload_device(self) -> str:
        """GPU when all frames fit in ~40 % of free VRAM, else pinned host memory."""
        torch = self.torch
        need = self.gt_bytes()
        try:
            free, _total = torch.cuda.mem_get_info(self.device)
        except Exception:
            free = 0
        return "cuda" if need <= 0.4 * free else "cpu"

    def load_image(self, v: TrainView, where: str = "cuda") -> None:
        """One GT frame as uint8 [1,3,H,W] at the resolution scale (GPU or pinned CPU)."""
        from PIL import Image

        if v.image is not None:
            return
        with Image.open(v.image_path) as im:
            rgb = im.convert("RGB")
            if rgb.size != (v.width, v.height):
                rgb = rgb.resize((v.width, v.height), Image.Resampling.LANCZOS)
            arr = np.asarray(rgb, dtype=np.uint8)
        t = self.torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 0, 1))).unsqueeze(0)
        v.image = t.to(self.device) if where == "cuda" else t.pin_memory()

    def gt(self, v: TrainView):
        """GT frame on the training device (H2D copy when preloaded on the host)."""
        if v.image is None:
            self.load_image(v, self.preload_device)
        img = v.image
        if img.device.type != "cuda":
            img = img.to(self.device, non_blocking=True)
        return img

    def preload_images(self, progress: Callable[[int, int], None] | None = None) -> str:
        """All GT frames (LiteGS ``device_preload``) to GPU, or pinned host RAM when too large."""
        where = self.choose_preload_device()
        self.preload_device = where
        for k, v in enumerate(self.views):
            self.load_image(v, where)
            if progress is not None:
                progress(k + 1, len(self.views))
        return where

    def camera_manifest(self) -> list[dict]:
        return [
            {"i": v.index, "name": v.name, "w": v.width, "h": v.height, "rail": v.rail, "holdout": v.holdout}
            for v in self.views
        ]

    # ---------------------------------------------------------------- gaussians
    def init_gaussians(self) -> None:
        torch = self.torch
        xyz = torch.tensor(self._init_xyz, dtype=torch.float32, device=self.device)
        rgb = torch.tensor(self._init_rgb, dtype=torch.float32, device=self.device)
        tensors = self.lg_scene.create_gaussians(xyz, rgb, self.sh_degree)
        tensors = self.lg_scene.cluster.cluster_points(self.pp.cluster_size, *tensors)
        params = [torch.nn.Parameter(t.contiguous()) for t in tensors]
        xyz_p, scale_p, rot_p, sh0_p, shr_p, opa_p = params
        self.opt, self.sched = self.lg_optimizer.get_optimizer(
            xyz_p, scale_p, rot_p, sh0_p, shr_p, opa_p, self.norm_radius, self.op, self.pp
        )
        self._aabb = None
        self.reset_stats(lambda _e: False)
        # simple_knn's NN-distance scale init is unclamped and, on a sparse SfM point
        # cloud (this project's panoramic scenes: few thousand COLMAP points over a huge
        # scene_scale), can start well past the crash/blur guard -- clamp before the very
        # first render, not just after each optimizer step.
        self.clamp_runaway_scale()

    def configure_schedule(self, max_steps: int) -> None:
        """xyz LR log-lerps 1.6e-4 -> 1.6e-6 (x norm_radius) over ``position_lr_max_steps``.

        Vendor LiteGS leaves that at 30000 even when ``iterations`` is shorter (see
        ``optimizer.get_optimizer`` / ``OptimizationParams.position_lr_max_steps``).
        Mapping it onto ``max_steps`` decayed xyz LR to the floor on a 3k-10k run and
        left split-displaced Gaussians stuck off the surface (synth plane z-std 0.42
        vs vendor 0.06). Never decay faster than the vendor 30k schedule; on a longer
        run, span the whole run.
        """
        self.op.iterations = int(max_steps)
        self.op.position_lr_max_steps = max(30000, int(max_steps))
        if self.sched is not None:
            self.sched.max_epochs = int(self.op.position_lr_max_steps)
            self.sched.lr_init = self.op.position_lr_init * self.norm_radius
            self.sched.lr_final = self.op.position_lr_final * self.norm_radius

    @property
    def params(self) -> dict:
        out = {}
        for group in self.opt.param_groups:
            out[group["name"]] = group["params"][0]
        return out

    def param_tuple(self):
        p = self.params
        return p["xyz"], p["scale"], p["rot"], p["sh_0"], p["sh_rest"], p["opacity"]

    @property
    def n_gaussians(self) -> int:
        xyz = self.params["xyz"]
        return int(xyz.shape[-1] * xyz.shape[-2])

    def reset_feedback(self) -> None:
        """Zero the GPU-driven-pipeline predictors (visible chunks / binning size per view).

        LiteGS predicts each view's allocation from its previous visit x1.2 / x1.5 and
        truncates the tail when the prediction is short. LiteGS densifies every 5 epochs so
        every view is re-visited between changes; Votion's step-based cycles visit a view
        less than once per cycle, so after a densify the stale prediction would clip the
        visible set for most steps. Zero = exact (synchronous) count on the next visit."""
        self.torch.cuda.synchronize()  # in-flight cudaMemcpyAsync into the pinned buffers
        if self._feedback_chunks is not None:
            self._feedback_chunks.zero_()
        if self._feedback_binning is not None:
            self._feedback_binning.zero_()

    def ensure_optimizer_state(self) -> None:
        """LiteGS ``_cat_tensors_to_optimizer`` needs Adam state for every group; a group
        whose grad was still None (sh_rest before SH warm-up) has none yet."""
        torch = self.torch
        for group in self.opt.param_groups:
            p = group["params"][0]
            state = self.opt.state[p]
            if len(state) == 0:
                state["step"] = torch.tensor(0.0, dtype=torch.float32)
                state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)

    def reset_stats(self, active_fn) -> None:
        xyz = self.params["xyz"]
        self.stats.reset(xyz.shape[-2], xyz.shape[-1], active_fn)
        self.stats.cur_sample = None

    def want_mean2d_grad(self, flag: bool) -> None:
        self._mean2d_wanted = bool(flag)

    def refresh_aabb(self) -> None:
        torch = self.torch
        xyz, scale, rot, *_ = self.param_tuple()
        with torch.no_grad():
            self._aabb = self.lg_scene.cluster.get_cluster_AABB(
                xyz, scale.exp(), torch.nn.functional.normalize(rot, dim=0)
            )

    def spatial_refine(self) -> None:
        """Morton re-sort of every parameter, grad and Adam state (LiteGS)."""
        with self.torch.no_grad():
            self.lg_scene.spatial_refine(True, self.opt, self.params["xyz"])
        self.refresh_aabb()

    # ---------------------------------------------------------------- rendering
    def _render_core(self, view, proj, xyz, scale, rot, color, opacity, valid_length, feedback, idx, sh_degree, shape, pp, collect_mean2d):
        """Copy of ``litegs.render.render`` + optional mean2d gradient statistic."""
        view = _cuda_packed(view)
        proj = _cuda_packed(proj)
        W = self.lg_utils.wrapper
        view_pos, ndc_pos = W.MVPTransform.apply(xyz, view, proj, valid_length)
        if collect_mean2d and self.stats.bStart and ndc_pos.requires_grad:
            helper = self.stats

            def _cb(h, grad, _w=shape[1], _hh=shape[0]):
                g = grad[:, :2, :]
                g = g * self.torch.tensor([[0.5 * _w], [0.5 * _hh]], device=g.device, dtype=g.dtype).unsqueeze(0)
                norm = g.norm(dim=1).sum(0)
                count = (norm > 0).to(self.torch.int32)
                h.update_mean_std("mean2d_grad", norm.unsqueeze(0), (norm * norm).unsqueeze(0), count, None)

            helper.register_tensor_grad_callback(ndc_pos, _cb)
        transform = W.CreateTransformMatrix.call_fused(scale, rot, valid_length)
        J = W.CreateRaySpaceTransformMatrix.call_fused(view_pos, proj, shape, valid_length)
        cov2d = W.CreateCov2dDirectly.call_fused(J, view, transform, valid_length)
        _ev, _evec, inv_cov2d = W.EighAndInverse2x2Matrix.call_fused(cov2d, valid_length)
        view_depth = view_pos[:, 2, :]
        tile_start, sorted_ids, primitive_visible = W.Binning.call_fused(
            ndc_pos, view_depth, inv_cov2d, opacity, valid_length, feedback, idx, shape, pp.tile_size
        )
        tiles = None
        try:
            tiles = self.stats.cached_sorted_tile_list[self.stats.cur_sample].unsqueeze(0)
        except Exception:
            tiles = None
        img, trans, depth, _normal, lst = W.GaussiansRasterFunc.apply(
            sorted_ids, tile_start, ndc_pos, inv_cov2d, color, opacity, tiles,
            shape[0], shape[1], pp.tile_size[0], pp.tile_size[1], pp.enable_transmitance, pp.enable_depth,
        )
        if self.stats.bStart:
            self.stats.update_tile_blend_count(lst, pp.tile_size[0], pp.tile_size[1])
        img = img[..., : shape[0], : shape[1]].clamp(0, 1).contiguous()
        if trans is not None:
            trans = trans[..., : shape[0], : shape[1]].contiguous()
        return img, trans, primitive_visible

    def render_train(self, view: TrainView, sh_degree: int):
        """Training render with the GPU-driven feedback buffers. Returns (img, aux)."""
        torch = self.torch
        xyz, scale, rot, sh0, shr, opa = self.param_tuple()
        idx = torch.tensor([view.index], dtype=torch.long)
        self.stats.cur_sample = view.name
        if self._aabb is None:
            self.refresh_aabb()
        origin, extend = self._aabb
        view_m = _cuda_packed(view.view)
        proj_m = _cuda_packed(view.proj)
        frustum_m = _cuda_packed(view.frustum)
        (vis_chunkid, vis_chunks_num, c_xyz, c_scale, c_rot, c_color, c_opa) = self.lg_render.render_preprocess(
            origin, extend, frustum_m, view_m, xyz, scale, rot, sh0, shr, opa,
            self._feedback_chunks, idx, self.pp, int(sh_degree),
        )
        valid_length = vis_chunks_num * self.pp.cluster_size
        img, trans, primitive_visible = self._render_core(
            view_m, proj_m, c_xyz, c_scale, c_rot, c_color, c_opa, valid_length,
            self._feedback_binning, idx, int(sh_degree), (view.height, view.width), self.pp, self._mean2d_wanted,
        )
        return img, trans, (vis_chunkid, vis_chunks_num, primitive_visible, c_scale)

    def loss(self, img, gt_u8, trans=None):
        """Photometric L1+SSIM. ``trans`` is only added when training
        ``pp.enable_transmitance`` is on (vendor default is off)."""
        gt = gt_u8.to(self.torch.float32) / 255.0
        loss = self.fused_ssim.fused_l1_ssim_loss(img, gt)
        if trans is not None:
            loss = loss + (1.0 - trans).abs().mean()
        return loss, gt

    def optimizer_step(self, aux) -> None:
        vis_chunkid, vis_chunks_num, primitive_visible, _ = aux
        self.opt.step(vis_chunkid, vis_chunks_num, primitive_visible)
        self.opt.zero_grad(set_to_none=True)
        self.sched.step()
        self.clamp_runaway_scale()

    def clamp_runaway_scale(self) -> None:
        """Cap world-space scale so a Gaussian can never grow close enough to the camera
        and big enough at once to overflow the CUDA binning table (see
        defaults.SCALE_CLAMP_CAM_DIST_FRACTION -- a flat cap either starves legitimate
        far/background coverage or is too loose up close; this scales with distance to
        the camera-cluster centroid instead). A second, absolute ceiling
        (defaults.SCALE_CLAMP_ABS_NORM_RADIUS) catches the reconstruction-error tail the
        distance-proportional term can't: sparse panoramic SfM points at wildly
        overestimated depth (unstable low-parallax triangulation, e.g. sky/horizon) would
        otherwise inherit a huge absolute scale just because they claim to be far away.
        Cheap (elementwise, no render) and shared by both controllers since it runs every
        step regardless of which one is active."""
        k = float(D.SCALE_CLAMP_CAM_DIST_FRACTION)
        abs_cap = float(D.SCALE_CLAMP_ABS_NORM_RADIUS) * float(self.norm_radius)
        if k <= 0 and abs_cap <= 0:
            return
        torch = self.torch
        with torch.no_grad():
            xyz = self.params["xyz"]  # [3, ...] same per-gaussian layout/order as "scale"
            scale = self.params["scale"].data  # [3, ...] log-space
            center = torch.as_tensor(self.cam_center, dtype=xyz.dtype, device=xyz.device)
            center = center.reshape((3,) + (1,) * (xyz.dim() - 1))
            dist = (xyz - center).pow(2).sum(dim=0).sqrt()  # [...] one axis short of xyz/scale
            min_dist = 0.05 * float(self.norm_radius)
            cap = k * dist.clamp_min(min_dist) if k > 0 else torch.full_like(dist, float("inf"))
            if abs_cap > 0:
                cap = cap.clamp_max(abs_cap)
            log_cap = cap.clamp_min(1e-6).log()
            scale.clamp_(max=log_cap.unsqueeze(0))

    def _quiet(self):
        """Context that hides viewport renders from the densify statistics."""
        backend = self

        class _Q:
            def __enter__(self_inner):
                self_inner.b = backend.stats.bStart
                self_inner.s = backend.stats.cur_sample
                backend.stats.bStart = False
                backend.stats.cur_sample = None
                return self_inner

            def __exit__(self_inner, *_a):
                backend.stats.bStart = self_inner.b
                backend.stats.cur_sample = self_inner.s
                return False

        return _Q()

    def render_pose(self, w2c: np.ndarray, K: np.ndarray, width: int, height: int, sh_degree: int, bg=(8, 16, 24)):
        """Free camera, no grad, full SH, opaque background. Returns uint8 [H,W,3] on GPU
        (or None when nothing is in view)."""
        torch = self.torch
        width, height = int(width), int(height)
        with torch.no_grad(), self._quiet():
            xyz, scale, rot, sh0, shr, opa = self.param_tuple()
            view = _cuda_packed(
                torch.tensor(
                    np.ascontiguousarray(np.asarray(w2c, dtype=np.float64).T, dtype=np.float32),
                    device=self.device,
                ).unsqueeze(0)
            )
            proj = _cuda_packed(
                torch.tensor(_proj_matrix(float(K[0, 0]), float(K[1, 1]), width, height), device=self.device).unsqueeze(0)
            )
            frustum = _cuda_packed(self.lg_utils.viewproj_to_frustumplane(view @ proj))
            origin, extend = self.lg_scene.cluster.get_cluster_AABB(
                xyz, scale.exp(), torch.nn.functional.normalize(rot, dim=0)
            )
            bgt = torch.tensor([[c / 255.0] for c in bg], device=self.device, dtype=torch.float32).reshape(1, 3, 1, 1)
            try:
                (vis_chunkid, vis_chunks_num, c_xyz, c_scale, c_rot, c_color, c_opa) = self.lg_render.render_preprocess(
                    origin, extend, frustum, view, xyz, scale, rot, sh0, shr, opa, None, None, self.pp_view, int(sh_degree)
                )
                if int(vis_chunks_num.item()) <= 0:
                    return (bgt.expand(1, 3, height, width)[0].permute(1, 2, 0) * 255.0).to(torch.uint8)
                valid_length = vis_chunks_num * self.pp_view.cluster_size
                img, trans, _pv = self._render_core(
                    view, proj, c_xyz, c_scale, c_rot, c_color, c_opa, valid_length, None, None,
                    int(sh_degree), (height, width), self.pp_view, False,
                )
            except Exception as exc:  # empty binning table etc.
                self.log(f"STATUS // viewport render skipped: {type(exc).__name__}: {exc}")
                return None
            if trans is not None:
                img = img + trans[:, :1] * bgt
            out = (img[0].clamp(0, 1).permute(1, 2, 0) * 255.0 + 0.5).to(torch.uint8)
            return out

    def render_camera(self, view: TrainView, sh_degree: int, with_gt: bool = False):
        """Render a training camera at its (scaled) size; optional GT side-by-side. uint8 [H,W(,x2),3]."""
        torch = self.torch
        out = self.render_pose(view.w2c, view.K, view.width, view.height, sh_degree)
        if out is None:
            return None
        if with_gt:
            try:
                gt = self.gt(view)[0].permute(1, 2, 0)
                out = torch.cat([out, gt], dim=1)
            except Exception as exc:
                self.log(f"STATUS // GT load skipped {view.name}: {exc}")
        return out

    @staticmethod
    def psnr(img, gt) -> float:
        mse = float(((img - gt) ** 2).mean())
        if mse <= 1e-12:
            return 99.0
        return float(-10.0 * math.log10(mse))

    def eval_holdout(self, sh_degree: int, limit: int = 16) -> float | None:
        """Mean PSNR over hold-out views (every 8th), at most ``limit`` views per call."""
        views = self.holdout_views
        if not views:
            return None
        if len(views) > limit:
            idx = np.linspace(0, len(views) - 1, limit).round().astype(int)
            views = [views[i] for i in idx]
        torch = self.torch
        total, n = 0.0, 0
        with torch.no_grad(), self._quiet():
            for v in views:
                xyz, scale, rot, sh0, shr, opa = self.param_tuple()
                origin, extend = self._aabb if self._aabb is not None else self.lg_scene.cluster.get_cluster_AABB(
                    xyz, scale.exp(), torch.nn.functional.normalize(rot, dim=0)
                )
                try:
                    view_m, proj_m, frustum_m = _cuda_packed(v.view), _cuda_packed(v.proj), _cuda_packed(v.frustum)
                    (vis_chunkid, vis_chunks_num, c_xyz, c_scale, c_rot, c_color, c_opa) = self.lg_render.render_preprocess(
                        origin, extend, frustum_m, view_m, xyz, scale, rot, sh0, shr, opa, None, None, self.pp, int(sh_degree)
                    )
                    if int(vis_chunks_num.item()) <= 0:
                        continue
                    valid_length = vis_chunks_num * self.pp.cluster_size
                    img, _t, _pv = self._render_core(
                        view_m, proj_m, c_xyz, c_scale, c_rot, c_color, c_opa, valid_length, None, None,
                        int(sh_degree), (v.height, v.width), self.pp, False,
                    )
                except Exception as exc:
                    self.log(f"STATUS // holdout render skipped {v.name}: {type(exc).__name__}: {exc}")
                    continue
                total += self.psnr(img, self.gt(v).to(torch.float32) / 255.0)
                n += 1
        return (total / n) if n else None

    # --------------------------------------------------------------- checkpoint
    def save_checkpoint(self, path: Path, meta: dict, controller_state: dict | None = None) -> None:
        torch = self.torch
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": CKPT_FORMAT,
            "optimizer": self.opt,
            "scheduler": self.sched,
            "meta": dict(meta),
            "controller_state": dict(controller_state or {}),
            "norm_radius": float(self.norm_radius),
            "sh_degree": int(self.sh_degree),
            "cluster_size": int(self.pp.cluster_size),
            "litegs_sha": LE.read_pinned_sha(),
        }
        tmp = path.with_name(path.name + ".tmp")
        torch.save(payload, str(tmp))
        # Windows: antivirus / search-indexer briefly opens a just-written multi-GB file
        # for scanning, which turns the rename into a transient PermissionError (WinError
        # 5) unrelated to training itself. A short retry clears almost all of these.
        for attempt in range(5):
            try:
                tmp.replace(path)
                return
            except PermissionError:
                if attempt == 4:
                    raise
                time.sleep(0.5 * (attempt + 1))

    @staticmethod
    def peek_checkpoint(path: Path) -> dict:
        """Meta only (no CUDA import of params). Old params-only files return format=legacy."""
        import torch

        LE.add_litegs_paths()  # the pickled optimizer / scheduler reference litegs.* classes
        try:
            ck = torch.load(str(path), map_location="cpu", weights_only=False, mmap=False)
        except Exception as exc:
            return {"format": "unreadable", "error": f"{type(exc).__name__}: {exc}"}
        if not isinstance(ck, dict) or ck.get("format") != CKPT_FORMAT:
            legacy = {"format": "legacy"}
            if isinstance(ck, dict):
                legacy["strategy"] = str(ck.get("strategy") or "")
                legacy["step"] = int(ck.get("step") or 0)
            return legacy
        meta = dict(ck.get("meta") or {})
        meta["format"] = CKPT_FORMAT
        return meta

    def load_checkpoint(self, path: Path) -> tuple[dict, dict]:
        torch = self.torch
        ck = torch.load(str(path), map_location=self.device, weights_only=False)
        if not isinstance(ck, dict) or ck.get("format") != CKPT_FORMAT:
            raise RuntimeError("checkpoint is from the old trainer (params only) — Retrain")
        self.opt = ck["optimizer"]
        self.sched = ck["scheduler"]
        # torch.optim.Optimizer.__getstate__ keeps only defaults/state/param_groups, so the
        # SparseGaussianAdam.bCluster attribute is lost through pickle; restore it.
        if not hasattr(self.opt, "bCluster"):
            self.opt.bCluster = int(ck.get("cluster_size", self.pp.cluster_size)) > 0
        self.norm_radius = float(ck.get("norm_radius", self.norm_radius))
        self.sh_degree = int(ck.get("sh_degree", self.sh_degree))
        for group in self.opt.param_groups:
            p = group["params"][0]
            if p.device != torch.device(self.device):
                group["params"][0] = torch.nn.Parameter(p.to(self.device))
        self._aabb = None
        self.reset_stats(lambda _e: False)
        self.clamp_runaway_scale()  # guard against resuming a pre-fix checkpoint
        return dict(ck.get("meta") or {}), dict(ck.get("controller_state") or {})

    # --------------------------------------------------------------------- ply
    def unclustered_numpy(self) -> dict:
        torch = self.torch
        with torch.no_grad():
            xyz, scale, rot, sh0, shr, opa = self.lg_scene.cluster.uncluster(*self.param_tuple())
            rot = torch.nn.functional.normalize(rot, dim=0)
            return {
                "xyz": xyz.detach().float().cpu().numpy(),
                "scale": scale.detach().float().cpu().numpy(),
                "rot": rot.detach().float().cpu().numpy(),
                "sh_0": sh0.detach().float().cpu().numpy(),
                "sh_rest": shr.detach().float().cpu().numpy(),
                "opacity": opa.detach().float().cpu().numpy(),
            }

    def export_ply(
        self,
        path: Path,
        *,
        compact: bool = False,
        drop_sh_rest: bool = False,
        min_opacity: float = 0.005,
        color_space: str = D.COLOR_SPACE_DEFAULT,
    ) -> tuple[Path, int]:
        from engine.splat.ply_export import write_splat_ply

        arrays = self.unclustered_numpy()
        return write_splat_ply(
            path,
            arrays,
            compact=compact,
            drop_sh_rest=drop_sh_rest,
            min_opacity=min_opacity,
            color_space=color_space,
        )

    def vram_mb(self) -> int:
        try:
            return int(self.torch.cuda.memory_allocated(self.device) / (1024 * 1024))
        except Exception:
            return 0


def spike_one_step() -> str:
    """Dummy scene: create_gaussians (simple_knn) -> cluster -> cull/compact/activate ->
    render (litegs_fused) -> fused L1+SSIM (fused_ssim_cuda) -> backward -> sparse Adam.
    Returns the env_report line. Raises on any failure so the caller records FAIL."""
    ensure_litegs()
    import numpy as np
    import torch

    if not torch.cuda.is_available():
        return "litegs raster 1-step: SKIP (torch.cuda.is_available() is False)"

    import fused_ssim
    import litegs  # noqa: F401  (pulls litegs.training -> torchmetrics, tqdm)
    from litegs import arguments, render, scene, utils
    from litegs.training import optimizer as lgs_opt
    from litegs.utils.statistic_helper import StatisticsHelperInst

    device = torch.device("cuda:0")
    torch.manual_seed(0)
    n = 4096
    xyz = torch.rand(n, 3, device=device) * 2.0 - 1.0
    xyz[:, 2] += 4.0
    color = torch.rand(n, 3, device=device)
    pp = arguments.PipelineParams.get_class_default_obj()
    op = arguments.OptimizationParams.get_class_default_obj()
    tensors = scene.create_gaussians(xyz, color, 3)
    tensors = scene.cluster.cluster_points(pp.cluster_size, *tensors)
    params = [torch.nn.Parameter(t.contiguous()) for t in tensors]
    xyz_p, scale_p, rot_p, sh0_p, shr_p, opa_p = params
    opt, sched = lgs_opt.get_optimizer(xyz_p, scale_p, rot_p, sh0_p, shr_p, opa_p, 1.0, op, pp)

    h = w = 64
    view = torch.eye(4, device=device, dtype=torch.float32).unsqueeze(0)
    proj = torch.tensor(_proj_matrix(50.0, 50.0, w, h), device=device).unsqueeze(0)
    frustum = utils.viewproj_to_frustumplane(view @ proj)
    StatisticsHelperInst.reset(xyz_p.shape[-2], xyz_p.shape[-1], lambda _e: False)

    (
        visible_chunkid,
        visible_chunks_num,
        c_xyz,
        c_scale,
        c_rot,
        c_color,
        c_opacity,
    ) = render.render_preprocess(
        None, None, frustum, view, xyz_p, scale_p, rot_p, sh0_p, shr_p, opa_p, None, None, pp, 0
    )
    valid_length = visible_chunks_num * pp.cluster_size
    img, _t, _d, _n, primitive_visible = render.render(
        view, proj, c_xyz, c_scale, c_rot, c_color, c_opacity, valid_length, None, None, 0, (h, w), pp
    )
    if tuple(img.shape) != (1, 3, h, w):
        raise RuntimeError(f"render shape {tuple(img.shape)} != (1,3,{h},{w})")
    if not torch.isfinite(img).all():
        raise RuntimeError("render produced non-finite pixels")
    gt = torch.rand(1, 3, h, w, device=device)
    loss = fused_ssim.fused_l1_ssim_loss(img, gt)
    if not torch.isfinite(loss):
        raise RuntimeError(f"loss not finite: {loss}")
    loss.backward()
    grad = xyz_p.grad
    if grad is None:
        raise RuntimeError("xyz grad is None after backward")
    vals = getattr(grad, "compacted_values", grad)
    if not torch.isfinite(vals).all():
        raise RuntimeError("xyz grad has non-finite values")
    opt.step(visible_chunkid, visible_chunks_num, primitive_visible)
    opt.zero_grad(set_to_none=True)
    sched.step()
    torch.cuda.synchronize()
    visible = int(visible_chunks_num.item()) * int(pp.cluster_size)
    return (
        f"litegs raster 1-step: OK loss={float(loss):.4f} visible={visible}/{n} "
        f"cluster={pp.cluster_size} sha={LE.read_pinned_sha()[:12]}"
    )


if __name__ == "__main__":
    t0 = time.time()
    print(spike_one_step(), f"({time.time() - t0:.1f}s)")
    sys.exit(0)
