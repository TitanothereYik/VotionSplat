"""Live viewport for the HY-World trainer and the checkpoint viewer. Runs in the hyworld venv."""

from __future__ import annotations

import collections
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.append(str(REPO))

from engine.splat.camera import ViewState, orbit_w2c, viewport_intrinsics  # noqa: E402
from engine.splat.viewport import ControlReader, SharedFrame  # noqa: E402

BG = (8, 16, 24)
MAX_W, MAX_H = 1920, 1080
CAMERAS_FILE = "votion_cameras.json"


def emit(kind: str, *fields) -> None:
    print("SPLAT\t" + kind + "".join("\t" + str(f) for f in fields), flush=True)


def frame_matrix(up, facing) -> np.ndarray:
    """3x3 rotation, trainer world -> view world: up -> -Y, facing -> +Z."""
    up = np.asarray(up, dtype=np.float64).reshape(3)
    up = up / max(float(np.linalg.norm(up)), 1e-9)
    fwd = np.asarray(facing, dtype=np.float64).reshape(3)
    fwd = fwd - up * float(fwd @ up)
    fwd = fwd / max(float(np.linalg.norm(fwd)), 1e-9)
    down = -up
    return np.stack([np.cross(down, fwd), down, fwd], axis=0)


def to_view(a: np.ndarray, unit: float, c2w: np.ndarray) -> np.ndarray:
    """Trainer camera-to-world -> world-to-camera in the view world."""
    r = a @ c2w[:3, :3]
    eye = (a @ c2w[:3, 3]) / unit
    w2c = np.eye(4)
    w2c[:3, :3] = r.T
    w2c[:3, 3] = -r.T @ eye
    return w2c


def to_trainer(a: np.ndarray, unit: float, w2c: np.ndarray) -> np.ndarray:
    """Inverse of ``to_view``."""
    r = w2c[:3, :3].T
    eye = -r @ w2c[:3, 3]
    c2w = np.eye(4)
    c2w[:3, :3] = a.T @ r
    c2w[:3, 3] = a.T @ (eye * unit)
    return c2w


def scene_unit(means: np.ndarray, center) -> float:
    center = np.asarray(center, dtype=np.float64).reshape(3)
    dist = np.linalg.norm(np.asarray(means, dtype=np.float64) - center, axis=1)
    dist = dist[np.isfinite(dist)]
    unit = float(np.median(dist)) if dist.size else 1.0
    return unit if unit > 1e-6 else 1.0


def camera_rows(names: list[str]) -> list[dict]:
    rows = []
    for index, name in enumerate(names):
        if name.startswith("panorama"):
            group = "panorama"
        elif name.startswith("polar"):
            group = "polar"
        else:
            group = name.rsplit("_", 1)[0]
        rows.append({"i": index, "name": name, "rail": group, "holdout": False})
    return rows


class LiveView:
    """``render_fn(c2w 4x4 float64, K 3x3 float64, width, height) -> uint8 [height, width, 3]``."""

    def __init__(self, render_fn, up, facing, center, unit: float, cam_c2w=None) -> None:
        self.render_fn = render_fn
        self.a = frame_matrix(up, facing)
        self.unit = float(unit)
        self.home = np.eye(4)
        self.home[:3, :3] = self.a.T
        self.home[:3, 3] = np.asarray(center, dtype=np.float64).reshape(3)
        self.cam_c2w = list(cam_c2w or [])
        self.state = ViewState()
        self._seq = -1
        self._key = None
        self._frames = 0
        self._t0 = time.perf_counter()
        self.shared = SharedFrame(max_w=MAX_W, max_h=MAX_H)
        print(self.shared.announce_line(), flush=True)
        self.control = ControlReader()

    def pose(self) -> np.ndarray:
        st = self.state
        base = self.cam_c2w[st.cam] if 0 <= st.cam < len(self.cam_c2w) else self.home
        w2c = orbit_w2c(to_view(self.a, self.unit, np.asarray(base, dtype=np.float64)),
                        st.yaw, st.pitch, st.zoom, st.pan_x, st.pan_y, 1.0,
                        st.fly_fwd, st.fly_right, st.fly_up)
        return to_trainer(self.a, self.unit, w2c)

    def poll(self) -> list[str]:
        """Take the newest camera from stdin. Returns the control words since the last call."""
        seq, view = self.control.latest_view()
        if view is not None and seq != self._seq:
            self._seq = seq
            self.state = view
        return self.control.pop_commands()

    def render(self, force: bool = False) -> bool:
        key = self.state.pose_key()
        if not force and key == self._key:
            return False
        w = int(min(max(self.state.width, 64), MAX_W))
        h = int(min(max(self.state.height, 36), MAX_H))
        self.shared.write(self.render_fn(self.pose(), viewport_intrinsics(w, h), w, h))
        self._key = key
        self._frames += 1
        now = time.perf_counter()
        if now - self._t0 >= 1.0:
            emit("fps", f"{self._frames / (now - self._t0):.1f}")
            self._frames = 0
            self._t0 = now
        return True

    def close(self) -> None:
        self.shared.close()


class TrainerHook:
    """Called from Runner.train() through the edits in tools/hyworld/apply_patches.py::patch_live."""

    def __init__(self, runner) -> None:
        import torch

        self.torch = torch
        self.runner = runner
        self.every = 20
        self.paused = False
        self.ema = None
        self.view = None
        self._t = time.perf_counter()
        self._step = 0
        self._saved_at = time.perf_counter()
        self.schedulers = []
        self.run_id = os.environ.get("VOTION_RUN_ID", "")
        self.resume_in = os.environ.get("VOTION_RESUME_IN", "")
        self.resume_out = os.environ.get("VOTION_RESUME_OUT", "")
        self.state_path = os.environ.get("VOTION_STATE", "")
        self.frame_out = os.environ.get("VOTION_FRAME_OUT", "")
        try:
            self.halt_at = int(os.environ.get("VOTION_HALT_AT", "-1"))
        except ValueError:
            self.halt_at = -1
        self.halt_reason = os.environ.get("VOTION_HALT_REASON", "stop")
        self.novel_path = os.environ.get("VOTION_NOVEL", "")
        try:
            self.novel_lambda = float(os.environ.get("VOTION_NOVEL_LAMBDA", "1"))
        except ValueError:
            self.novel_lambda = 1.0
        try:
            self.autosave_sec = float(os.environ.get("VOTION_AUTOSAVE_SEC", "600"))
        except ValueError:
            self.autosave_sec = 600.0
        try:
            parser = runner.parser
            means = runner.splats["means"].detach()
            count = int(means.shape[0])
            pick = torch.linspace(0, max(count - 1, 0), min(count, 8192), device=means.device).long()
            unit = scene_unit(means[pick].float().cpu().numpy(), parser.center_point)
            index = list(runner.trainset.indices)
            index = index[:: max(1, math.ceil(len(index) / 800))]
            names = [str(parser.image_names[i]) for i in index]
            c2ws = [np.asarray(parser.camtoworlds[i], dtype=np.float64) for i in index]
            self.view = LiveView(self._render, parser.up_direction, parser.facing_direction,
                                 parser.center_point, unit, c2ws)
            Path(runner.cfg.result_dir, CAMERAS_FILE).write_text(
                json.dumps({"names": names, "c2w": [m.tolist() for m in c2ws]}), encoding="utf-8")
            emit("cameras", json.dumps(camera_rows(names)))
            emit("state", "TRAINING")
        except Exception as exc:
            print(f"STATUS // live view off: {type(exc).__name__}: {exc}", flush=True)
            self.view = None

    def _render(self, c2w, k, w, h):
        torch, r = self.torch, self.runner
        was = r.is_training
        r.is_training = False
        try:
            with torch.no_grad():
                colors, _alphas, _info = r.rasterize_splats(
                    camtoworlds=torch.from_numpy(c2w).float().to(r.device)[None],
                    Ks=torch.from_numpy(k).float().to(r.device)[None],
                    width=w,
                    height=h,
                    sh_degree=r.cfg.sh_degree,
                    near_plane=r.cfg.near_plane,
                    far_plane=r.cfg.far_plane,
                    backgrounds=torch.tensor([BG], device=r.device, dtype=torch.float32) / 255.0,
                    render_mode="RGB",
                )
            return (colors[0, ..., :3].clamp(0, 1) * 255 + 0.5).to(torch.uint8).cpu().numpy()
        finally:
            r.is_training = was

    def _words(self, step: int) -> None:
        for word in self.view.poll():
            if word == "PAUSE" and not self.paused:
                self.paused = True
                emit("state", "PAUSED")
            elif word == "RESUME" and self.paused:
                self.paused = False
                emit("state", "TRAINING")
            elif word == "STOP":
                self._halt(step)

    def _to_dev(self, value):
        torch = self.torch
        if torch.is_tensor(value):
            return value.to(self.runner.device)
        if isinstance(value, dict):
            return {key: self._to_dev(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._to_dev(item) for item in value]
        return value

    def resume(self, schedulers) -> int:
        """Restore a saved step, or 0. Writes the pinned frame the first time."""
        self.schedulers = list(schedulers)
        r = self.runner
        if self.frame_out and not os.path.isfile(self.frame_out):
            frame = {
                "transform": np.asarray(r.parser.transform).tolist(),
                "rescale": float(r.parser.rescale if r.parser.rescale is not None else 1.0),
                "up_direction": np.asarray(r.parser.up_direction).tolist(),
                "facing_direction": np.asarray(r.parser.facing_direction).tolist(),
                "center_point": np.asarray(r.parser.center_point).reshape(3).tolist(),
                "scene_scale": float(r.parser.scene_scale),
            }
            path = Path(self.frame_out)
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_text(json.dumps(frame) + "\n", encoding="utf-8")
            os.replace(tmp, path)
        if not self.resume_in:
            return 0
        torch = self.torch
        data = torch.load(self.resume_in, map_location=r.device, weights_only=False)
        if int(data["max_steps"]) != int(r.cfg.max_steps):
            print("STATUS // resume max_steps mismatch", flush=True)
            raise SystemExit(2)
        for key, value in data["splats"].items():
            r.splats[key] = torch.nn.Parameter(value.to(r.device), requires_grad=True)
        for key, opt in r.optimizers.items():
            opt.param_groups[0]["params"] = [r.splats[key]]
            opt.state = collections.defaultdict(dict)
            opt.load_state_dict(data["optimizers"][key])
        for key, value in data["strategy_state"].items():
            r.strategy_state[key] = self._to_dev(value)
        for scheduler, saved in zip(self.schedulers, data["schedulers"]):
            scheduler.load_state_dict(saved)
        rng = data["rng"]
        torch.set_rng_state(rng["torch"].cpu() if torch.is_tensor(rng["torch"]) else rng["torch"])
        torch.cuda.set_rng_state(rng["cuda"].cpu() if torch.is_tensor(rng["cuda"]) else rng["cuda"])
        np.random.set_state(rng["numpy"])
        random.setstate(rng["python"])
        step = int(data["next_step"])
        print(f"STATUS // resumed at step {step} of {r.cfg.max_steps}", flush=True)
        return step

    def save_resume(self, next_step: int, reason: str) -> None:
        if not self.run_id or not self.resume_out:
            return
        torch = self.torch
        r = self.runner
        data = {
            "version": 1,
            "run_id": self.run_id,
            "next_step": int(next_step),
            "max_steps": int(r.cfg.max_steps),
            "splats": r.splats.state_dict(),
            "optimizers": {key: opt.state_dict() for key, opt in r.optimizers.items()},
            "strategy_state": dict(r.strategy_state),
            "schedulers": [scheduler.state_dict() for scheduler in self.schedulers],
            "rng": {
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state(),
                "numpy": np.random.get_state(),
                "python": random.getstate(),
            },
        }
        path = Path(self.resume_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(str(path) + ".tmp")
        torch.save(data, tmp)
        os.replace(tmp, path)
        side = {
            "next_step": int(next_step),
            "reason": reason,
            "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        meta = Path(str(path).removesuffix(".pt") + ".json")
        meta_tmp = Path(str(meta) + ".tmp")
        meta_tmp.write_text(json.dumps(side) + "\n", encoding="utf-8")
        os.replace(meta_tmp, meta)
        self._saved_at = time.perf_counter()
        print(f"STATUS // saved resume state at step {next_step} ({reason})", flush=True)

    def loss_weight(self, data) -> float:
        names = data.get("image_name", [])
        if isinstance(names, str):
            names = [names]
        for name in names:
            if str(name).startswith("zdifix_"):
                return self.novel_lambda
        return 1.0

    def write_state(self, reason: str, next_step: int) -> None:
        if not self.state_path:
            return
        payload = {
            "run_id": self.run_id,
            "reason": reason,
            "next_step": int(next_step),
            "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        path = Path(self.state_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload) + "\n", encoding="utf-8")
        os.replace(tmp, path)

    def render_novel(self) -> None:
        """Render the poses.json cameras into the round's novel folder."""
        if not self.novel_path or not os.path.isfile(self.novel_path):
            print("STATUS // novel poses missing", flush=True)
            raise SystemExit(3)
        import imageio.v2 as imageio
        from gs.normalize import transform_cameras

        torch, r = self.torch, self.runner
        poses = json.loads(Path(self.novel_path).read_text(encoding="utf-8"))
        cam_info = json.loads(Path(r.cfg.data_dir, "cameras.json").read_text(encoding="utf-8"))
        name0 = str(r.parser.image_names[0])
        orig = np.linalg.inv(np.asarray(cam_info[name0]["extrinsic"], dtype=np.float64))
        got = transform_cameras(np.asarray(r.parser.transform, dtype=np.float64), orig[None])[0]
        if float(np.max(np.abs(got - np.asarray(r.parser.camtoworlds[0])))) > 1e-3:
            print("STATUS // novel pose frame mismatch", flush=True)
            raise SystemExit(3)
        emit("state", "RENDER")
        folder = Path(self.novel_path).parent / "novel"
        folder.mkdir(parents=True, exist_ok=True)
        was = r.is_training
        r.is_training = False
        try:
            for pose in poses:
                c2w = transform_cameras(
                    np.asarray(r.parser.transform, dtype=np.float64),
                    np.asarray(pose["c2w"], dtype=np.float64)[None],
                )[0]
                k = np.asarray(pose["K"], dtype=np.float64)
                w, h = int(pose["width"]), int(pose["height"])
                with torch.no_grad():
                    colors, _alphas, _info = r.rasterize_splats(
                        camtoworlds=torch.from_numpy(c2w).float().to(r.device)[None],
                        Ks=torch.from_numpy(k).float().to(r.device)[None],
                        width=w,
                        height=h,
                        sh_degree=r.cfg.sh_degree,
                        near_plane=r.cfg.near_plane,
                        far_plane=r.cfg.far_plane,
                        backgrounds=torch.zeros(1, 3, device=r.device, dtype=torch.float32),
                        render_mode="RGB",
                    )
                image = (colors[0, ..., :3].clamp(0, 1) * 255 + 0.5).to(torch.uint8).cpu().numpy()
                imageio.imwrite(folder / f"{pose['name']}.png", image)
        finally:
            r.is_training = was

    def _halt(self, step: int) -> None:
        r = self.runner
        emit("state", "SAVING")
        data = {
            "step": step,
            "splats": r.splats.state_dict(),
            "transform": r.parser.transform,
            "up_direction": r.parser.up_direction,
            "facing_direction": r.parser.facing_direction,
            "center_point": r.parser.center_point,
        }
        path = f"{r.ckpt_dir}/ckpt_{step}_rank{r.world_rank}.pt"
        self.torch.save(data, path)
        print(f"STATUS // training halted at step {step}; checkpoint {path}", flush=True)
        if self.run_id:
            self.save_resume(step, "stop")
            self.write_state("stop", step)
        self.finish("stop")
        raise SystemExit(0)

    def tick(self, step: int) -> None:
        """First statement of every training step."""
        if self.run_id:
            if self.halt_at >= 0 and step == self.halt_at:
                if self.halt_reason == "fix":
                    self.render_novel()
                self.save_resume(step, self.halt_reason)
                self.write_state(self.halt_reason, step)
                self.finish(self.halt_reason)
                raise SystemExit(0)
            if time.perf_counter() - self._saved_at >= self.autosave_sec:
                try:
                    self.save_resume(step, "autosave")
                except Exception as exc:
                    print(f"STATUS // autosave failed: {type(exc).__name__}: {exc}", flush=True)
        if self.view is None:
            return
        try:
            self._words(step)
            while self.paused:
                self.view.render()
                time.sleep(0.02)
                self._words(step)
            self.view.render(force=step % self.every == 0)
        except SystemExit:
            raise
        except Exception as exc:
            print(f"STATUS // live view off: {type(exc).__name__}: {exc}", flush=True)
            self.paused = False
            self.view = None

    def report(self, step: int, max_steps: int, loss: float, colors, pixels, count: int) -> None:
        self.ema = loss if self.ema is None else 0.9 * self.ema + 0.1 * loss
        last = step == max_steps - 1
        if step % 10 and not last:
            return
        try:
            torch = self.torch
            with torch.no_grad():
                mse = float(((colors.detach().clamp(0, 1) - pixels) ** 2).mean())
            now = time.perf_counter()
            its = (step - self._step) / max(now - self._t, 1e-6) if step > self._step else 0.0
            self._t, self._step = now, step
            emit("step", step + 1, max_steps)
            emit("loss", f"{self.ema:.4f}")
            emit("psnr", f"{-10.0 * math.log10(max(mse, 1e-10)):.2f}", "-")
            emit("gaussians", count)
            emit("its", f"{its:.1f}")
            emit("vram", int(torch.cuda.max_memory_allocated() / 2**20))
            if last:
                emit("state", "SAVING")
        except Exception:
            pass

    def finish(self, reason: str = "done") -> None:
        if self.run_id and reason == "done":
            self.write_state("done", int(self.runner.cfg.max_steps))
        emit("state", "DONE")
        if self.view is not None:
            self.view.close()
            self.view = None
