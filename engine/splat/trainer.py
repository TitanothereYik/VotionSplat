"""LiteGS step loop for the Splat page. Qt never imports this module.

Modes (``engine.workers.splat --mode``):
  train / continue / retrain  step loop with Pause / Resume over stdin, live viewport
  view                        keep the checkpoint on GPU, re-render on camera change
  export / export_compact     write splat.ply (full SH) / splat_compact.ply
                              ``--color-space votion|srgb|linear`` (default votion)

Stdout protocol (tab separated, one line each; Qt parses ``SPLAT\\t...``)::

  SPLAT step <step> <max>            SPLAT loss <ema>            SPLAT gaussians <n>
  SPLAT epoch <epoch> <total>        SPLAT psnr <train> <eval|->  SPLAT its <steps/s>
  SPLAT vram <MB>                    SPLAT fps <viewport fps>    SPLAT budget <n>
  SPLAT state TRAINING|PAUSED|IDLE|VIEW|LOADING|DONE
  SPLAT strategy <label>             SPLAT ply <path>            SPLAT preview <png>
  SPLAT shm <name> <bytes> <max_w> <max_h>    (once, before the first frame)
  SPLAT cameras <json list>          (once)    SPLAT ctl <controller event text>
  SPLAT ckpt <step>                  SPLAT error <text>
  STATUS // <free text>              (log drawer)

Stdin protocol (Qt -> worker): ``VIEW\\t{json}``, ``PAUSE``, ``RESUME``, ``STOP``.
Stop from the job bar still kills the process; the 100-step checkpoint covers it.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from engine.job import job_progress
from engine.splat import defaults as D
from engine.splat.camera import (
    ViewState,
    fit_training_intrinsics,
    orbit_w2c,
    read_view_json,
    star_w2c,
    viewport_intrinsics,
    write_view_json,
)
from engine.splat.viewport import ControlReader, SharedFrame

# See the epoch-seed comment at the gen.manual_seed() call site: 0 keeps runs
# fully reproducible; a watchdog sets this env var on restart-after-stall only.
_epoch_seed_salt = int(os.environ.get("VOTION_EPOCH_SEED_SALT", "0") or 0)


def log(msg: str) -> None:
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:
        enc = sys.stdout.encoding or "ascii"
        print(msg.encode(enc, errors="replace").decode(enc), flush=True)


def emit(kind: str, *fields) -> None:
    log("SPLAT\t" + kind + "".join("\t" + str(f) for f in fields))


def status(msg: str) -> None:
    log(f"STATUS // {msg}")


def _write_png(rgb_u8: np.ndarray, path: Path) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    Image.fromarray(rgb_u8).save(tmp, format="PNG", compress_level=1)
    try:
        os.replace(tmp, path)
    except OSError:
        try:
            Image.fromarray(rgb_u8).save(path, format="PNG", compress_level=1)
        except OSError:
            tmp.unlink(missing_ok=True)


class Viewport:
    """Free camera over the backend: Star orbit, training-camera jump, GT compare."""

    def __init__(self, backend, scene_dir: Path, shared: SharedFrame | None, png_path: Path) -> None:
        self.backend = backend
        self.scene_dir = scene_dir
        self.shared = shared
        self.png_path = png_path
        self.state = read_view_json(scene_dir / D.VIEW_REL)
        self.base = star_w2c(scene_dir, backend.cam_center)
        self._reach: float | None = None
        self._last_key = None
        self._last_png = 0.0
        self.frames = 0
        self._fps_t0 = time.perf_counter()
        self._fps_n = 0
        self.fps = 0.0
        self.last_ms = 0.0

    def set_state(self, state: ViewState) -> None:
        self.state = state

    def invalidate(self) -> None:
        self._last_key = None
        self._reach = None

    def _compute_reach(self) -> float:
        if self._reach is not None:
            return self._reach
        torch = self.backend.torch
        with torch.no_grad():
            xyz = self.backend.params["xyz"]
            flat = xyz.detach().reshape(3, -1)
            n = flat.shape[1]
            if n > 8192:
                idx = torch.linspace(0, n - 1, 8192, device=flat.device).long()
                flat = flat[:, idx]
            center = flat.median(dim=1).values.float().cpu().numpy()
        eye0 = -self.base[:3, :3].T @ self.base[:3, 3]
        self._reach = max(float(np.linalg.norm(center - eye0)), float(self.backend.scene_scale), 1.0)
        return self._reach

    def render(self, sh_degree: int, force: bool = False, png: bool = False) -> bool:
        """Render the current camera into shared memory (and PNG when asked). Returns True on a new frame."""
        st = self.state
        key = st.pose_key()
        if not force and key == self._last_key:
            return False
        t0 = time.perf_counter()
        w = int(min(max(st.width, 64), D.VIEWPORT_MAX_W))
        h = int(min(max(st.height, 36), D.VIEWPORT_MAX_H))
        frame = None
        views = self.backend.views
        if 0 <= st.cam < len(views):
            cam = views[st.cam]
            if st.gt:
                frame = self.backend.render_camera(cam, sh_degree, with_gt=True)
            else:
                base = cam.w2c
                reach = self._compute_reach()
                w2c = orbit_w2c(
                    base, st.yaw, st.pitch, st.zoom, st.pan_x, st.pan_y, reach,
                    st.fly_fwd, st.fly_right, st.fly_up,
                )
                rw, rh, k = fit_training_intrinsics(
                    cam.K,
                    cam.width,
                    cam.height,
                    w,
                    h,
                    resolution_scale=self.backend.resolution_scale,
                    max_w=D.VIEWPORT_MAX_W,
                    max_h=D.VIEWPORT_MAX_H,
                )
                frame = self.backend.render_pose(w2c, k, rw, rh, sh_degree, bg=D.VIEW_BG)
        else:
            reach = self._compute_reach()
            w2c = orbit_w2c(
                self.base, st.yaw, st.pitch, st.zoom, st.pan_x, st.pan_y, reach,
                st.fly_fwd, st.fly_right, st.fly_up,
            )
            k = viewport_intrinsics(w, h)
            frame = self.backend.render_pose(w2c, k, w, h, sh_degree, bg=D.VIEW_BG)
        if frame is None:
            return False
        if frame.shape[1] > D.VIEWPORT_MAX_W or frame.shape[0] > D.VIEWPORT_MAX_H:
            # GT compare can exceed the slot: downscale on GPU by integer stride
            torch = self.backend.torch
            f = frame.permute(2, 0, 1).unsqueeze(0).float()
            scale = max(frame.shape[1] / D.VIEWPORT_MAX_W, frame.shape[0] / D.VIEWPORT_MAX_H)
            nh, nw = int(frame.shape[0] / scale), int(frame.shape[1] / scale)
            f = torch.nn.functional.interpolate(f, size=(nh, nw), mode="area")
            frame = f[0].permute(1, 2, 0).round().clamp(0, 255).to(torch.uint8)
        arr = frame.contiguous().cpu().numpy()
        self._last_key = key
        if self.shared is not None:
            try:
                self.shared.write(arr)
            except Exception as exc:
                status(f"viewport shm write failed: {type(exc).__name__}: {exc}")
                self.shared = None
        now = time.perf_counter()
        if png or self.shared is None:
            if now - self._last_png >= (0.2 if self.shared is None else 1.0):
                try:
                    _write_png(arr, self.png_path)
                    emit("preview", self.png_path)
                    self._last_png = now
                except Exception as exc:
                    status(f"preview png skipped: {exc}")
        self.last_ms = (now - t0) * 1000.0
        self.frames += 1
        self._fps_n += 1
        if now - self._fps_t0 >= 1.0:
            self.fps = self._fps_n / (now - self._fps_t0)
            self._fps_n = 0
            self._fps_t0 = now
        return True


class SplatTrainer:
    def __init__(
        self,
        *,
        scene_dir: Path,
        strategy: str,
        mode: str,
        max_steps: int,
        device,
        budget: int = D.GAUSSIAN_BUDGET,
        resolution_scale: float = 1.0,
        holdout: bool = False,
        headless: bool = False,
        color_space: str = D.COLOR_SPACE_DEFAULT,
    ) -> None:
        self.scene_dir = Path(scene_dir)
        self.strategy = strategy
        self.mode = mode
        self.max_steps = max(1, int(max_steps))
        self.device = device
        self.budget = max(1, int(budget))
        self.resolution_scale = float(resolution_scale)
        self.holdout = bool(holdout)
        self.headless = bool(headless)
        self.color_space = D.color_space_from_key(color_space)
        from engine.sfm.dataset_source import colmap_root, output_root

        self.colmap_dir = colmap_root(self.scene_dir)
        self.output_dir = output_root(self.scene_dir)
        self.ckpt_path = self.output_dir / D.CKPT_REL
        self.meta_path = self.output_dir / D.META_REL
        self.ply_path = self.output_dir / D.PLY_REL
        self.png_path = self.output_dir / D.PREVIEW_REL
        self.backend = None
        self.controller = None
        self.control: ControlReader | None = None
        self.shared: SharedFrame | None = None
        self.viewport: Viewport | None = None
        self._view_seq = 0
        self._pending: list[str] = []
        self._eof_sent = False
        self.state = "IDLE"

    # ------------------------------------------------------------------ helpers
    def _set_state(self, state: str) -> None:
        if state != self.state:
            self.state = state
            emit("state", state)

    def _make_backend(self):
        from engine.splat.backend.litegs_backend import LiteGSBackend, ensure_litegs

        for line in ensure_litegs():
            status(line)
        backend = LiteGSBackend(
            self.scene_dir,
            self.device,
            sh_degree=D.SH_DEGREE,
            resolution_scale=self.resolution_scale,
            holdout=self.holdout,
            colmap_dir=self.colmap_dir,
            log=log,
        )
        return backend

    def _open_control(self) -> None:
        if self.headless:
            return
        try:
            self.control = ControlReader(sys.stdin)
        except Exception as exc:
            status(f"stdin control unavailable: {exc}")
            self.control = None

    def _open_shared(self) -> None:
        try:
            self.shared = SharedFrame(max_w=D.VIEWPORT_MAX_W, max_h=D.VIEWPORT_MAX_H)
            log(self.shared.announce_line())
        except Exception as exc:
            status(f"shared memory unavailable ({exc}); PNG fallback")
            self.shared = None

    def _poll_control(self) -> list[str]:
        """Apply the latest VIEW; return control words (each word is delivered once)."""
        if self.control is None:
            return []
        seq, view = self.control.latest_view()
        if view is not None and seq != self._view_seq and self.viewport is not None:
            self._view_seq = seq
            self.viewport.set_state(view)
        cmds = self.control.pop_commands()
        if self.control.closed and not self.headless and not self._eof_sent:
            self._eof_sent = True
            cmds.append("EOF")
        return cmds

    def _take_cmds(self) -> list[str]:
        """Control words seen since the last call (including ones queued mid-step)."""
        cmds = self._pending + self._poll_control()
        self._pending = []
        return cmds

    def _write_meta(self, step: int, epoch: int, psnr_train: float | None, psnr_eval: float | None, done: bool) -> None:
        meta = {
            "format": "votion-litegs-1",
            "controller": self.strategy,
            "strategy": self.strategy,
            "step": int(step),
            "max_steps": int(self.max_steps),
            "epoch": int(epoch),
            "gaussians": int(self.backend.n_gaussians),
            "budget": int(self.budget),
            "psnr": None if psnr_train is None else round(float(psnr_train), 3),
            "psnr_eval": None if psnr_eval is None else round(float(psnr_eval), 3),
            "resolution_scale": self.resolution_scale,
            "holdout": self.holdout,
            "views": len(self.backend.views),
            "images": getattr(self, "dataset_images", ""),
            "done": bool(done),
        }
        self.meta_path.parent.mkdir(parents=True, exist_ok=True)
        self.meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    def _save(self, step: int, epoch: int, psnr_train, psnr_eval, done: bool = False) -> None:
        meta = {
            "controller": self.strategy,
            "step": int(step),
            "epoch": int(epoch),
            "max_steps": int(self.max_steps),
            "budget": int(self.budget),
            "gaussians": int(self.backend.n_gaussians),
            "psnr": psnr_train,
            "psnr_eval": psnr_eval,
        }
        ctl = self.controller.state_dict() if self.controller is not None else {}
        self.backend.save_checkpoint(self.ckpt_path, meta, ctl)
        self._write_meta(step, epoch, psnr_train, psnr_eval, done)
        emit("ckpt", step)

    def _export(self, compact: bool = False, drop_sh_rest: bool = False) -> Path:
        path = self.output_dir / (D.PLY_COMPACT_REL if compact else D.PLY_REL)
        out, n = self.backend.export_ply(
            path,
            compact=compact,
            drop_sh_rest=drop_sh_rest,
            min_opacity=D.COMPACT_MIN_OPACITY,
            color_space=self.color_space,
        )
        emit("ply", out)
        extra = ", compact" if compact else ", full SH"
        if drop_sh_rest:
            extra += ", DC only"
        status(f"exported {out} ({n} gaussians{extra}, color={self.color_space})")
        return out

    # ------------------------------------------------------------------ modes
    def run(self) -> int:
        colmap = self.colmap_dir
        if not (colmap / "sparse" / "0" / "cameras.bin").is_file():
            status(f"missing {colmap / 'sparse' / '0'} — Reconstruct first")
            return 2
        status(f"dataset {colmap}")
        if self.mode in ("train", "continue", "retrain"):
            from engine.hires.pinhole import active_choice, choice_matches_dataset, choice_name, dataset_images

            images = dataset_images(self.scene_dir)
            choice = active_choice(self.scene_dir)
            self.dataset_images = images
            if choice and not choice_matches_dataset(choice, images):
                status(
                    f"dataset is {images or 'unmarked'} but Generate is {choice_name(choice)}. Reconstruct again."
                )
                return 2
        if self.mode in ("export", "export_compact", "export_compact_dc"):
            return self.run_export()
        if self.mode == "view":
            return self.run_view()
        return self.run_train()

    def _load_for_render(self) -> bool:
        """Backend + checkpoint for view / export. False when the checkpoint is unusable."""
        from engine.splat.backend.litegs_backend import LiteGSBackend

        if not self.ckpt_path.is_file():
            status("needs checkpoints/splat_latest.pt — Train first")
            return False
        peek = LiteGSBackend.peek_checkpoint(self.ckpt_path)
        if peek.get("format") != "votion-litegs-1":
            status(f"checkpoint is from the old trainer ({peek.get('format')}) — Retrain required")
            emit("error", "Retrain required: checkpoint predates the LiteGS trainer")
            return False
        self.backend = self._make_backend()
        self.backend.load_colmap()
        meta, _ctl = self.backend.load_checkpoint(self.ckpt_path)
        self.strategy = str(meta.get("controller") or self.strategy)
        status(f"loaded {self.ckpt_path.name}: {self.strategy} step {meta.get('step')} N={self.backend.n_gaussians}")
        return True

    def run_export(self) -> int:
        self._set_state("LOADING")
        if not self._load_for_render():
            if self.mode == "export" and self.ply_path.is_file():
                emit("ply", self.ply_path)
                status("splat.ply already on disk")
                return 0
            return 2
        self._export(compact=self.mode != "export", drop_sh_rest=self.mode == "export_compact_dc")
        self._set_state("DONE")
        return 0

    def run_view(self) -> int:
        self._set_state("LOADING")
        if not self._load_for_render():
            return 2
        self._open_control()
        self._open_shared()
        emit("cameras", json.dumps(self.backend.camera_manifest()))
        emit("gaussians", self.backend.n_gaussians)
        emit("strategy", self.strategy)
        self.viewport = Viewport(self.backend, self.scene_dir, self.shared, self.png_path)
        self._set_state("VIEW")
        stamp = self._ckpt_stamp()
        self.viewport.render(D.SH_DEGREE, force=True, png=True)
        last_fps_emit = time.perf_counter()
        idle_since = time.perf_counter()
        try:
            while True:
                cmds = self._take_cmds()
                if "STOP" in cmds or "EOF" in cmds:
                    status("view worker: stdin closed" if "EOF" in cmds else "view worker: STOP")
                    break
                fresh = self._ckpt_stamp()
                if fresh != stamp:
                    stamp = fresh
                    try:
                        meta, _ = self.backend.load_checkpoint(self.ckpt_path)
                        self.viewport.invalidate()
                        emit("gaussians", self.backend.n_gaussians)
                        status(f"view reload step {meta.get('step')} N={self.backend.n_gaussians}")
                    except Exception as exc:
                        status(f"view reload skipped: {exc}")
                rendered = self.viewport.render(D.SH_DEGREE, png=True)
                now = time.perf_counter()
                if rendered:
                    idle_since = now
                    if now - last_fps_emit >= 0.5:
                        emit("fps", f"{self.viewport.fps:.1f}")
                        last_fps_emit = now
                else:
                    # no busy loop: 2 ms while the user drags, 30 ms when idle
                    time.sleep(0.002 if now - idle_since < 0.5 else 0.03)
        finally:
            self._shutdown_viewport()
        return 0

    def _ckpt_stamp(self) -> str:
        try:
            st = self.ckpt_path.stat()
            return f"{st.st_mtime_ns}:{st.st_size}"
        except OSError:
            return ""

    def _shutdown_viewport(self) -> None:
        if self.viewport is not None:
            try:
                write_view_json(self.output_dir / D.VIEW_REL, self.viewport.state)
            except Exception:
                pass
        if self.shared is not None:
            self.shared.close()
            self.shared = None

    # ------------------------------------------------------------------ train
    def run_train(self) -> int:
        from engine.splat.backend.litegs_backend import LiteGSBackend
        from engine.splat.controllers import DensifySchedule, make_controller, sh_interval

        self._set_state("LOADING")
        if self.mode == "retrain":
            for p in (self.ckpt_path, self.meta_path):
                if p.is_file():
                    p.unlink()
            status("Retrain — checkpoint cleared")
        elif self.ckpt_path.is_file() and self.meta_path.is_file():
            try:
                saved = json.loads(self.meta_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                saved = {}
            trained = str(saved.get("images") or "")
            current = str(getattr(self, "dataset_images", "") or "")
            if current and trained != current:
                status("checkpoint is not from this HiRes path — Retrain")
                return 2

        resume_meta = None
        controller_state: dict = {}
        if self.ckpt_path.is_file():
            peek = LiteGSBackend.peek_checkpoint(self.ckpt_path)
            if peek.get("format") != "votion-litegs-1":
                status(f"checkpoint is from the old trainer ({peek.get('format')}) — Retrain required")
                emit("error", "Retrain required: checkpoint predates the LiteGS trainer")
                return 2
            ck_ctl = str(peek.get("controller") or "")
            if ck_ctl and ck_ctl != self.strategy:
                status(f"checkpoint controller is {ck_ctl}; switch to {self.strategy} needs Retrain")
                emit("error", f"Retrain required: checkpoint is {ck_ctl}")
                return 2
            resume_meta = peek
        elif self.mode == "continue":
            status("Continue needs checkpoints/splat_latest.pt")
            return 2
        backend = self._make_backend()
        self.backend = backend
        t_load = time.perf_counter()
        backend.load_colmap()
        n_train = len(backend.train_views)
        status(
            f"colmap views={len(backend.views)} train={n_train} holdout={len(backend.holdout_views)} "
            f"points={backend.init_points} norm_radius={backend.norm_radius:.3f} scene_scale={backend.scene_scale:.3f} "
            f"res_scale={self.resolution_scale}"
        )

        def _progress(k: int, n: int) -> None:
            if k == n or k % 25 == 0:
                status(f"preload {k}/{n} frames")

        where = backend.preload_images(_progress)
        status(
            f"{len(backend.views)} frames ({backend.gt_bytes() / 1e6:.0f} MB) preloaded to "
            f"{'GPU' if where == 'cuda' else 'pinned host RAM'} in {time.perf_counter() - t_load:.1f}s, "
            f"VRAM {backend.vram_mb()} MB"
        )

        start_step = 0
        if resume_meta is not None:
            meta, controller_state = backend.load_checkpoint(self.ckpt_path)
            start_step = int(meta.get("step") or 0)
            status(f"Continue {self.strategy} from step {start_step} N={backend.n_gaussians}")
        else:
            backend.init_gaussians()
            status(f"init {backend.n_gaussians} Gaussians from points3D.bin ({self.strategy}, SH {D.SH_DEGREE})")
        backend.configure_schedule(self.max_steps)

        total_epochs = max(1, int(math.ceil(self.max_steps / max(n_train, 1))))
        schedule = DensifySchedule.for_run(self.max_steps, n_train)
        sh_every = sh_interval(self.max_steps, n_train)
        self.controller = make_controller(self.strategy, backend, budget=self.budget, schedule=schedule, log=log)
        schedule = self.controller.schedule  # a controller may pick its own cadence (MCMC Compact: 100 steps)
        if controller_state:
            self.controller.load_state_dict(controller_state)
        backend.want_mean2d_grad(self.controller.needs_mean2d_grad)
        backend.reset_stats(self.controller.is_densify_active)
        self.controller.events.append(f"SH warm-up: one band every {sh_every} steps up to degree {D.SH_DEGREE}")
        for ev in self.controller.events:
            emit("ctl", ev)
        self.controller.events.clear()

        self._open_control()
        self._open_shared()
        emit("cameras", json.dumps(backend.camera_manifest()))
        emit("strategy", self.strategy)
        emit("budget", self.budget)
        emit("epoch", start_step // max(n_train, 1), total_epochs)
        emit("step", start_step, self.max_steps)
        emit("gaussians", backend.n_gaussians)
        self.viewport = Viewport(backend, self.scene_dir, self.shared, self.png_path)
        if start_step >= self.max_steps:
            status(f"already at max_steps {self.max_steps}")
            self._export()
            self.viewport.render(D.SH_DEGREE, force=True, png=True)
            self._set_state("DONE")
            self._shutdown_viewport()
            return 0

        torch = backend.torch
        gen = torch.Generator(device="cpu")
        loss_ema: float | None = None
        psnr_train: float | None = None
        psnr_eval: float | None = None
        loss_acc = torch.zeros((), device=self.device)
        mse_acc = torch.zeros((), device=self.device)
        acc_n = 0
        perm: list[int] = []
        cur_epoch = -1
        cur_cycle = -1
        active_sh = min(start_step // sh_every, D.SH_DEGREE)
        t_report = time.perf_counter()
        step_report = start_step
        last_view_t = 0.0
        paused_total = 0.0
        skipped_views = 0
        stop_requested = False
        exit_code = 0
        step = start_step
        self._set_state("TRAINING")
        self.viewport.render(active_sh, force=True, png=True)
        try:
            while step < self.max_steps:
                # ---------------------------------------------------- control
                cmds = self._take_cmds()
                if "STOP" in cmds:
                    status("STOP received — saving checkpoint")
                    stop_requested = True
                    break
                if "EOF" in cmds:
                    status("stdin closed — saving checkpoint and exiting")
                    stop_requested = True
                    break
                if "PAUSE" in cmds:
                    paused_total += self._pause_loop(active_sh)
                    if self.state == "STOPPING":
                        stop_requested = True
                        break
                    t_report = time.perf_counter()
                    step_report = step

                # ------------------------------------------------ epoch begin
                epoch = step // n_train
                if epoch != cur_epoch:
                    cur_epoch = epoch
                    # Salt is 0 on a normal run (fully reproducible). A watchdog that
                    # restarts a stalled process after a resume-from-checkpoint sets
                    # VOTION_EPOCH_SEED_SALT so the *current* epoch's permutation
                    # differs from the run that just got stuck -- resuming a checkpoint
                    # otherwise recomputes the identical epoch/view order deterministically,
                    # which would walk straight back into the same pathological view.
                    gen.manual_seed(1234 + epoch + _epoch_seed_salt)
                    perm = torch.randperm(n_train, generator=gen).tolist()
                    emit("epoch", epoch, total_epochs)

                # ------------------------------------------------ cycle begin
                cycle = schedule.cycle(step)
                if cycle != cur_cycle:
                    cur_cycle = cycle
                    with torch.no_grad():
                        if cycle > 0 and schedule.active:
                            backend.spatial_refine()  # Morton re-sort after each densify
                        else:
                            backend.refresh_aabb()
                    torch.cuda.synchronize()
                    backend.stats.bStart = bool(self.controller.is_densify_active(cycle))
                active_sh = min(step // sh_every, D.SH_DEGREE)

                # ------------------------------------------------------ step
                view = backend.train_views[perm[step % n_train]]
                try:
                    img, trans, aux = backend.render_train(view, active_sh)
                    loss, gt = backend.loss(img, backend.gt(view), trans)
                    if backend.op.reg_weight > 0.0:
                        loss = loss + aux[3].square().mean() * backend.op.reg_weight
                    loss.backward()
                    if backend.stats.bStart:
                        backend.stats.backward_callback()
                    backend.optimizer_step(aux)
                    with torch.no_grad():
                        loss_acc += loss.detach()
                        mse_acc += ((img.detach() - gt) ** 2).mean()
                        acc_n += 1
                except RuntimeError as exc:
                    # Vendor LiteGS CUDA binning table indexes tile-primitive pairs with a
                    # 32-bit int (litegs/submodules/gaussian_raster/binning.cu); an
                    # unlucky view occasionally still overflows it even with the backend's
                    # per-step scale clamp (rare -- seen roughly once per 15-20k steps on
                    # a dense, full-resolution HiRes retrain). Not survivable by retrying
                    # the same view, and not worth losing an otherwise-healthy multi-hour
                    # run over: skip this one view/step and keep training. Anything else
                    # (OOM, a real bug) still propagates.
                    msg = str(exc)
                    if "negative dimension" not in msg and "pred_allocate_size" not in msg:
                        raise
                    skipped_views += 1
                    backend.opt.zero_grad(set_to_none=True)
                    if backend.stats.handle_list:
                        # _render_core() registers a mean2d-grad callback (for the
                        # densify stats) *before* the binning kernel that just threw --
                        # backward_callback() never runs for this aborted step, so that
                        # stale (tensor, callback) entry would otherwise survive into the
                        # next step's handle_list and hit `assert(tensor.grad is not
                        # None)` there once a fresh entry's grad *is* populated (observed:
                        # AssertionError in litegs/utils/statistic_helper.py backward_callback).
                        backend.stats.handle_list = []
                    if skipped_views <= 20 or skipped_views % 50 == 0:
                        status(
                            f"WARN step {step}: skipped view {view.name} -- CUDA binning "
                            f"table overflow ({skipped_views} skipped so far): {msg}"
                        )
                step += 1

                # -------------------------------------------------- cycle end
                if schedule.cycle_ends(step):
                    backend.stats.bStart = False
                    before = backend.n_gaussians
                    self.controller.step(backend.opt, cycle)
                    for ev in self.controller.events:
                        emit("ctl", ev)
                    self.controller.events.clear()
                    if backend.n_gaussians != before:
                        emit("gaussians", backend.n_gaussians)
                        self.viewport.invalidate()
                    backend.reset_feedback()  # predictors are stale after any prune / append / decay
                    backend._aabb = None

                # -------------------------------------------------- epoch end
                if (step % n_train == 0 or step >= self.max_steps) and self.holdout and backend.holdout_views:
                    got = backend.eval_holdout(active_sh)
                    if got is not None:
                        psnr_eval = got

                # ----------------------------------------------------- report
                report = step % 10 == 0 or step == self.max_steps or step == start_step + 1
                if report:
                    n = max(acc_n, 1)
                    l = float(loss_acc.item()) / n
                    mse = float(mse_acc.item()) / n
                    loss_acc.zero_()
                    mse_acc.zero_()
                    acc_n = 0
                    loss_ema = l if loss_ema is None else 0.9 * loss_ema + 0.1 * l
                    psnr_train = 99.0 if mse <= 1e-12 else -10.0 * math.log10(mse)
                    now = time.perf_counter()
                    its = (step - step_report) / max(now - t_report, 1e-6)
                    t_report = now
                    step_report = step
                    emit("step", step, self.max_steps)
                    emit("loss", f"{loss_ema:.6f}")
                    emit("gaussians", backend.n_gaussians)
                    emit("psnr", f"{psnr_train:.2f}", "-" if psnr_eval is None else f"{psnr_eval:.2f}")
                    emit("its", f"{its:.1f}")
                    emit("vram", backend.vram_mb())
                    if self.viewport.frames:
                        emit("fps", f"{self.viewport.fps:.1f}")
                    job_progress(step, self.max_steps, f"{self.strategy} {step}/{self.max_steps}")
                    if step % 100 == 0 or step == self.max_steps:
                        status(
                            f"{self.strategy} {step}/{self.max_steps} epoch {epoch}/{total_epochs} "
                            f"loss={loss_ema:.5f} psnr={psnr_train:.2f}"
                            + ("" if psnr_eval is None else f" eval={psnr_eval:.2f}")
                            + f" N={backend.n_gaussians} sh={active_sh} {its:.1f} it/s {backend.vram_mb()} MB"
                        )

                # --------------------------------------------------- viewport
                self._pending.extend(self._poll_control())  # camera moved during the step?
                moved = self.viewport.state.pose_key() != self.viewport._last_key
                now_v = time.perf_counter()
                # <= 1 step latency on camera change, but capped at VIEW_MAX_FPS_TRAINING so a
                # continuous orbit costs the step loop well under 20 % (a 720p frame ~= one step)
                if (moved and now_v - last_view_t >= 1.0 / D.VIEW_MAX_FPS_TRAINING) or step % D.VIEW_EVERY_STEPS == 0:
                    if self.viewport.render(active_sh, force=not moved, png=(step % 100 == 0)):
                        last_view_t = now_v

                # ------------------------------------------------- checkpoint
                if step % D.CKPT_EVERY == 0 and step < self.max_steps:
                    self._save(step, epoch, psnr_train, psnr_eval)
        except KeyboardInterrupt:
            status("interrupted — saving checkpoint")
            stop_requested = True
        except Exception as exc:
            status(f"train failed at step {step}: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            emit("error", f"{type(exc).__name__}: {exc}")
            exit_code = 4
            try:
                self._save(step, max(cur_epoch, 0), psnr_train, psnr_eval)
            except Exception as exc2:
                status(f"emergency checkpoint failed: {exc2}")
            self._shutdown_viewport()
            return exit_code

        done = step >= self.max_steps
        epoch = step // max(n_train, 1)
        self._save(step, epoch, psnr_train, psnr_eval, done=done)
        if done or stop_requested:
            self._export()
        try:
            self.viewport.render(active_sh, force=True, png=True)
        except Exception as exc:
            status(f"final viewport skipped: {exc}")
        emit("step", step, self.max_steps)
        emit("gaussians", backend.n_gaussians)
        if skipped_views:
            status(f"{skipped_views} view(s) skipped over the run (CUDA binning table overflow)")
        if done:
            status(f"training complete → {self.ply_path}")
        else:
            status(f"stopped at step {step}; Continue resumes with optimizer state")
        self._set_state("DONE" if done else "IDLE")
        self._shutdown_viewport()
        return exit_code

    def _pause_loop(self, active_sh: int) -> float:
        """Sit in PAUSED, keep the viewport live, until RESUME / STOP / EOF. Returns paused seconds."""
        t0 = time.perf_counter()
        self._set_state("PAUSED")
        status("paused")
        idle_since = t0
        try:
            while True:
                cmds = self._take_cmds()
                if "RESUME" in cmds:
                    break
                if "STOP" in cmds or "EOF" in cmds:
                    self.state = "STOPPING"
                    return time.perf_counter() - t0
                rendered = self.viewport.render(active_sh, png=True)
                now = time.perf_counter()
                if rendered:
                    idle_since = now
                else:
                    time.sleep(0.002 if now - idle_since < 0.5 else 0.03)
        finally:
            if self.state != "STOPPING":
                self._set_state("TRAINING")
                status("resumed")
        return time.perf_counter() - t0
