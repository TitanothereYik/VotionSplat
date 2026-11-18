"""WorldStereo expansion over HY-World trajectories. Scaled FP8, one GPU.

Qt never imports this. Run ``python -m engine.workers.hy_expand``.
"""

from __future__ import annotations

import gc
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from engine.hyworld.paths import hy_root, list_trajectories, ordered_ids, read_selection, render_root
from engine.worldstereo.adapter import TrajJob
from engine.worldstereo.hy_bank import HyMemoryBank
from engine.worldstereo.runtime import (
    MODEL_TYPE,
    _encode_clip,
    _encode_text,
    _pipeline,
    _retrieve,
    _sample,
    _write_memory_inputs,
    patch_single_rank,
    pin_dinov2,
)
from engine.job import REPO_ROOT
from engine.worldstereo.pipeline import ensure_files


def _log(msg: str) -> None:
    print(msg, flush=True)


def _depth_median(root: Path) -> float:
    blob = torch.load(root / "full_depth_prediction.pt", map_location="cpu", weights_only=False)
    distance = blob["distance"].detach().float().cpu()
    mask = np.array(Image.open(root / "sky_mask.png").convert("L")) > 127
    if mask.shape != tuple(distance.shape):
        import cv2

        mask = cv2.resize(mask.astype(np.uint8), (distance.shape[1], distance.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    values = distance.numpy()[mask]
    values = values[np.isfinite(values) & (values > 0)]
    if values.size == 0:
        raise RuntimeError("HY depth has no valid pixels for the memory far plane")
    return float(np.median(values))


def _hy(*fields) -> None:
    print("HY\t" + "\t".join(str(field) for field in fields), flush=True)


def _jobs(scene: str) -> list[TrajJob]:
    chosen = ordered_ids(read_selection(scene) or list_trajectories(scene))
    jobs = []
    for item in chosen:
        view_id, traj_id = item.split("/", 1)
        jobs.append(
            TrajJob(
                rail_index=len(jobs),
                direction="hy",
                view="front",
                view_id=view_id,
                traj_id=traj_id,
                layout="hy",
                frames=21,
            )
        )
    return jobs


def _result(root: Path, job: TrajJob) -> Path:
    return root / job.view_id / job.traj_id / f"{MODEL_TYPE}_result.mp4"


def _remember_video(bank, path: Path, camera: dict, view_id: str, traj_id: str) -> None:
    import cv2

    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    cap.release()
    w2cs = torch.tensor(np.array(camera["extrinsic"]), dtype=torch.float32)
    ks = torch.tensor(np.array(camera["intrinsic"]), dtype=torch.float32)
    if len(frames) != int(w2cs.shape[0]):
        raise RuntimeError(f"{path.name} has {len(frames)} frames, camera has {w2cs.shape[0]}")
    bank.update_memory(frames, w2cs, ks, view_id, traj_id)


def generate(
    scene: str,
    *,
    only: str = "",
    resume: bool = True,
    seed: int = 1024,
    level: str = "auto",
    cap_gb: float = 0.0,
    bench: bool = False,
) -> int:
    from transformers import CLIPImageProcessor

    root = render_root(scene)
    if not (root / "pano_bank" / "cameras.json").is_file():
        _log("STATUS // plan and render the trajectories first")
        return 2
    jobs = _jobs(scene)
    keys = [f"{job.view_id}/{job.traj_id}" for job in jobs]
    if not jobs:
        _log("STATUS // no trajectories selected")
        return 2
    if only and only not in keys:
        _log(f"STATUS // {only} is not in the kept list")
        return 2
    if bench and not only:
        _log("STATUS // --bench needs --only")
        return 2
    _hy("queue", json.dumps(keys))

    def wanted(key: str, job) -> bool:
        if only and key != only:
            return False
        if bench:
            return True
        return (not resume) or not _result(root, job).is_file()

    for key, job in zip(keys, jobs):
        if not wanted(key, job):
            continue
        folder = root / job.view_id / job.traj_id
        if not (folder / "render.mp4").is_file() or not (folder / "traj_caption.json").is_file():
            _log(f"STATUS // missing guide or caption for {key}. Run Render guides.")
            return 2
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    patch_single_rank()
    pin_dinov2(REPO_ROOT / "models" / "dinov2-base")
    config_path, weights, transformer_cfg = ensure_files()
    ws_cfg_json = json.loads(config_path.read_text(encoding="utf-8"))
    from omegaconf import OmegaConf

    ws_cfg = OmegaConf.create(ws_cfg_json)
    sample = next((root / "pano_bank" / "images").glob("*.png"))
    import imagesize

    width, height = imagesize.get(str(sample))
    median = _depth_median(root)
    bank = HyMemoryBank(
        cfg=ws_cfg,
        root_path=str(root).replace("\\", "/"),
        image_width=width,
        image_height=height,
        device=device,
        max_reference=8,
        align_nframe=8,
        rank=0,
        world_size=1,
        depth_median=median,
    )
    _log(f"STATUS // HY panorama memory depth_median={median:.4f} far={bank.far:.4f} bank={bank.mem_size}")
    bank.camera_selector.model.to("cpu")
    torch.cuda.empty_cache()

    prompts = {}
    for key, job in zip(keys, jobs):
        if not wanted(key, job):
            continue
        caption = root / job.view_id / job.traj_id / "traj_caption.json"
        prompts[key] = json.loads(caption.read_text(encoding="utf-8"))["prompt"]
    embeds = {}
    text = tokenizer = None
    for key, prompt in prompts.items():
        embeds[key], text, tokenizer = _encode_text(prompt)
    text_holder = {"tokenizer": tokenizer, "encoder": text}
    clip = None
    processor = None
    from engine.worldstereo.pipeline import load_clip_vision
    from engine.worldstereo.pipeline import WAN_CFG

    clip = load_clip_vision(WAN_CFG).eval()
    processor = CLIPImageProcessor.from_pretrained(str(WAN_CFG / "image_processor"), local_files_only=True)
    images = {}
    for key, job in zip(keys, jobs):
        if not wanted(key, job):
            continue
        if job.view_id not in images:
            images[job.view_id] = _encode_clip(root / job.view_id / "start_frame.png", clip, processor)
    clip.to("cpu")

    model = pipe = vae = fit = None
    work = hy_root(scene) / "_votion" / "bench" if bench else root
    total = len(jobs)
    for step, (key, job) in enumerate(zip(keys, jobs), start=1):
        result = _result(root, job)
        if not wanted(key, job):
            if result.is_file():
                camera = json.loads((root / job.view_id / job.traj_id / "camera.json").read_text(encoding="utf-8"))
                _remember_video(bank, result, camera, job.view_id, job.traj_id)
                _log(f"STATUS // skip {key}, memory={bank.mem_size}")
                _hy("state", key, "skip", "0", "0")
                print(f"PROGRESS\tjob\t{step}\t{total}\t{int(100 * step / total)}\tskip {key}", flush=True)
            continue
        try:
            ws_cfg.nframe = 21
            bank.nframe = 21
            _hy("state", key, "retrieve", "0", "0")
            _log(f"STATUS // {key} retrieve")
            retrieved, ref_index, ref_index_dict, ref_w2cs, _global, w2cs, ks = _retrieve(bank, job, root)
            _write_memory_inputs(work, job, retrieved, ref_index_dict, ref_w2cs)
            del retrieved
            if model is None:
                from engine.worldstereo.budget import Fit
                from engine.worldstereo.pipeline import build_dit, load_vae

                fit = Fit(device, level=level, cap_gb=cap_gb, log=_log)
                model, ws_cfg = fit.build(build_dit, transformer_cfg, config_path, weights)
                _hy("fit", json.dumps(fit.summary()))
                vae = load_vae()
                pipe = _pipeline(model, ws_cfg, text_holder, clip, processor, vae)
            ws_cfg.nframe = 21
            _hy("state", key, "sample", "0", "0")
            _log(f"STATUS // {key} sample")
            while True:
                note = ""
                fit.start()
                try:
                    frames, elapsed, peak = _sample(
                        pipe, ws_cfg, job, root, work, embeds[key], images[job.view_id], ref_index, seed
                    )
                except RuntimeError as exc:
                    if not str(exc).startswith("VRAM //"):
                        raise
                    note = str(exc)
                if not note:
                    break
                _log(f"STATUS // {key} did not fit at level={fit.level}: {note}")
                gc.collect()
                torch.cuda.empty_cache()
                if not fit.step_down(model):
                    raise RuntimeError(f"{note}; level={fit.level} was the last level to try")
                _hy("fit", json.dumps(fit.summary()))
            reserved = torch.cuda.max_memory_reserved(device) / (1024**3)
            fit.record(elapsed, peak, reserved)
            card = torch.cuda.get_device_properties(device).total_memory / (1024**3)
            if reserved > card - 1.0:
                _log(
                    f"STATUS // {key} reserved {reserved:.2f} GB of {card:.2f} GB; "
                    "the driver may be using system memory"
                )
            array = frames.detach().float().clamp(0, 1).mul(255).byte().cpu().permute(0, 2, 3, 1).numpy()
            if bench:
                tag = f"{job.view_id}_{job.traj_id}_{fit.level}_cap{cap_gb:g}"
                np.save(work / f"{tag}.npy", array)
                for index in (0, len(array) // 2, len(array) - 1):
                    Image.fromarray(array[index]).save(work / f"{tag}_{index:03d}.png")
                _log(
                    f"STATUS // bench {key} frames={len(array)} seconds={elapsed:.1f} peak_gb={peak:.2f} "
                    f"reserved_gb={reserved:.2f} {fit.line()}"
                )
                del frames, array
                break
            from diffusers.utils import export_to_video

            export_to_video(array.astype(np.float32) / 255.0, str(result), fps=16)
            preview = hy_root(scene) / "_votion" / "preview" / f"{job.view_id}_{job.traj_id}"
            preview.mkdir(parents=True, exist_ok=True)
            pil_frames = []
            for index, frame in enumerate(array):
                image = Image.fromarray(frame)
                image.save(preview / f"{index:03d}.png")
                pil_frames.append(image)
            bank.update_memory(pil_frames, w2cs.detach().cpu(), ks.detach().cpu(), job.view_id, job.traj_id)
            _log(
                f"STATUS // {key} frames={len(pil_frames)} "
                f"seconds={elapsed:.1f} peak_gb={peak:.2f} reserved_gb={reserved:.2f} "
                f"{fit.line()} bank={bank.mem_size}"
            )
            _hy("state", key, "done", f"{elapsed:.1f}", f"{peak:.2f}")
            print(f"PROGRESS\tjob\t{step}\t{total}\t{int(100 * step / total)}\tdone {key}", flush=True)
            del frames, array, pil_frames
        except Exception:
            _hy("state", key, "failed", "0", "0")
            raise
        if only:
            break
    if bench:
        _log("STATUS // bench done")
        return 0
    _log("STATUS // hyworld expand done; align was not started")
    return 0
