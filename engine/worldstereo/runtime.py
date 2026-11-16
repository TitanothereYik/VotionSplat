"""One-process WorldStereo memory-DMD run. The bank stays across rails.

Retrieval uses HY-World's depth-scaled far plane. The bank still keeps every
generated frame, so the next joint chunk can see the pose beside it.

Load order matches the spike that fit on this machine: text and CLIP, then
DINOv2 retrieval, then the scaled FP8 DiT. DINOv2 is moved off the GPU before
each forward. The DiT is never dequantized to BF16. WorldMirror is not called.
"""

from __future__ import annotations

import gc
import hashlib
import json
import shutil
import time
from pathlib import Path

import torch

from engine.job import REPO_ROOT
from engine.pano.scene_io import scene_dir
from engine.worldstereo import defaults as D
from engine.world.layers import clear_layers_from, ensure_layers, refresh_guide
from engine.worldstereo.adapter import TrajJob, input_root
from engine.worldstereo.views import diffusion_num_frames
from engine.worldstereo.pipeline import (
    WAN_CFG,
    build_dit,
    ensure_files,
    jsonable,
    load_clip_vision,
    load_vae,
    patch_single_rank,
    pin_dinov2,
)
from engine.worldstereo.resume import (
    pending_jobs,
    rail_dir,
    rail_snapshot,
    regen_plan,
    resume_snapshot,
    view_snapshot,
)

MODEL_TYPE = "worldstereo-memory-dmd"


def _log(msg: str) -> None:
    print(msg, flush=True)


def weights_sha256(path: Path) -> str:
    cache = path.with_suffix(path.suffix + ".sha256")
    if cache.is_file():
        return cache.read_text(encoding="utf-8").strip().split()[0]
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    hexdigest = digest.hexdigest()
    cache.write_text(hexdigest + "\n", encoding="utf-8")
    return hexdigest


def _park_dino(bank) -> None:
    bank.camera_selector.model.to("cpu")
    gc.collect()
    torch.cuda.empty_cache()


def _wake_dino(bank) -> None:
    bank.camera_selector.model.to(bank.device)


def save_bank(bank, path: Path) -> None:
    frames = path / "frames"
    if frames.exists():
        shutil.rmtree(frames)
    frames.mkdir(parents=True, exist_ok=True)
    for index, image in enumerate(bank.ref_frames):
        image.save(frames / f"{index:04d}.png")
    payload = {
        "fnames": list(bank.fnames),
        "w2cs": bank.ref_w2cs.detach().float().cpu().numpy().tolist(),
        "Ks": bank.ref_Ks.detach().float().cpu().numpy().tolist(),
        "depth_median": float(getattr(bank, "depth_median", 0.0)),
    }
    (path / "bank.json").write_text(json.dumps(payload), encoding="utf-8")


def load_bank_into(bank, path: Path) -> None:
    payload = json.loads((path / "bank.json").read_text(encoding="utf-8"))
    from PIL import Image

    frames = []
    for index in range(len(payload["fnames"])):
        frames.append(Image.open(path / "frames" / f"{index:04d}.png").convert("RGB"))
    if len(frames) != len(payload["fnames"]):
        raise RuntimeError(f"bank snapshot {path} is missing frames")
    bank.ref_frames = frames
    bank.fnames = list(payload["fnames"])
    bank.ref_w2cs = torch.tensor(payload["w2cs"], dtype=torch.float32, device=bank.device)
    bank.ref_Ks = torch.tensor(payload["Ks"], dtype=torch.float32, device=bank.device)
    if payload.get("depth_median"):
        bank.depth_median = float(payload["depth_median"])
    if hasattr(bank, "ref_depths"):
        bank.ref_depths = [None] * len(frames)
    bank.mem_size = len(frames)


def _encode_text(prompt: str):
    from transformers import AutoTokenizer

    from engine.worldstereo.encoders import load_umt5

    tokenizer = AutoTokenizer.from_pretrained(str(WAN_CFG / "tokenizer"), local_files_only=True)
    text = load_umt5(WAN_CFG).to("cuda").eval()
    tokens = tokenizer(
        prompt,
        padding="max_length",
        max_length=512,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        hidden = text(tokens.input_ids.to("cuda"), tokens.attention_mask.to("cuda")).last_hidden_state
    length = int(tokens.attention_mask.gt(0).sum())
    hidden = hidden[0, :length]
    pad = hidden.new_zeros(512 - length, hidden.shape[-1])
    embeds = torch.cat([hidden, pad], dim=0).unsqueeze(0).cpu()
    text.to("cpu")
    del hidden, tokens
    gc.collect()
    torch.cuda.empty_cache()
    _log(f"STATUS // text encoder off GPU; prompt embeds {tuple(embeds.shape)}")
    return embeds, text, tokenizer


def _remember_frames(bank, frames, w2cs, ks, view_id: str, traj_id: str, start: int) -> int:
    """Append every generated frame to the one bank shared by all rails and views.

    ``start`` skips a discarded anchor. The stock update kept eight frames and
    always dropped frame 0, so the next view could not see the pose beside it.
    """
    kept = list(frames)[start:]
    poses = w2cs[start:]
    eyes = ks[start:]
    if len(kept) != int(poses.shape[0]) or int(poses.shape[0]) != int(eyes.shape[0]):
        raise RuntimeError(
            f"memory frames {len(kept)} do not match poses {tuple(poses.shape)} / {tuple(eyes.shape)}"
        )
    bank.ref_w2cs = torch.cat([bank.ref_w2cs, poses.to(device=bank.device, dtype=torch.float32)], dim=0)
    bank.ref_Ks = torch.cat([bank.ref_Ks, eyes.to(device=bank.device, dtype=torch.float32)], dim=0)
    bank.ref_frames.extend(kept)
    if hasattr(bank, "ref_depths"):
        bank.ref_depths.extend([None] * len(kept))
    for index in range(start, start + len(kept)):
        bank.fnames.append(f"{view_id}/{traj_id}/{index:04d}")
    bank.mem_size = len(bank.ref_frames)
    return len(kept)


def _encode_clip(image_path: Path, clip, processor) -> torch.Tensor:
    from PIL import Image

    clip.to("cuda")
    batch = processor(images=Image.open(image_path).convert("RGB"), return_tensors="pt")
    batch = {key: value.to("cuda") for key, value in batch.items()}
    with torch.no_grad():
        embeds = clip(**batch, output_hidden_states=True).hidden_states[-2].cpu()
    clip.to("cpu")
    del batch
    gc.collect()
    torch.cuda.empty_cache()
    return embeds


def _open_bank(ws_cfg, scene: str, scene_input: Path):
    import imagesize

    from engine.worldstereo.hy_bank import HyMemoryBank, scene_depth_median

    sample = next((scene_input / "pano_bank" / "images").glob("*.png"))
    width, height = imagesize.get(str(sample))
    median = scene_depth_median(scene)
    bank = HyMemoryBank(
        cfg=ws_cfg,
        root_path=str(scene_input),
        image_width=width,
        image_height=height,
        device=torch.device("cuda:0"),
        max_reference=8,
        align_nframe=D.ALIGN_NFRAME,
        rank=0,
        world_size=1,
        depth_median=median,
    )
    _log(
        f"STATUS // HY panorama memory depth_median={bank.depth_median:.4f} "
        f"far={bank.far:.4f} bank={bank.mem_size}"
    )
    return bank


def _retrieve(bank, job: TrajJob, scene_input: Path):
    import numpy as np

    cam = json.loads((scene_input / job.view_id / job.traj_id / "camera.json").read_text(encoding="utf-8"))
    device = bank.device
    w2cs = torch.tensor(np.array(cam["extrinsic"]), dtype=torch.float32, device=device)
    ks = torch.tensor(np.array(cam["intrinsic"]), dtype=torch.float32, device=device)
    _wake_dino(bank)
    retrieved, ref_index, ref_index_dict, ref_w2cs, global_indices = bank.retrieval(
        w2cs, ks, view_id=job.view_id, traj_id=job.traj_id
    )
    _park_dino(bank)
    return retrieved, ref_index, ref_index_dict, ref_w2cs, global_indices, w2cs, ks


def _write_memory_inputs(work: Path, job: TrajJob, retrieved, ref_index_dict, ref_w2cs) -> None:
    from diffusers.utils import export_to_video

    mem = work / job.view_id / job.traj_id / "memory_inputs"
    mem.mkdir(parents=True, exist_ok=True)
    export_to_video(retrieved / 255.0, str(mem / f"{MODEL_TYPE}.mp4"), fps=16)
    if ref_index_dict is not None:
        (mem / f"{MODEL_TYPE}_ref_index.json").write_text(
            json.dumps(jsonable(ref_index_dict), indent=2), encoding="utf-8"
        )
    if ref_w2cs is not None:
        (mem / f"{MODEL_TYPE}_ref_w2cs.json").write_text(
            json.dumps(ref_w2cs.detach().cpu().numpy().tolist(), indent=2), encoding="utf-8"
        )


def _pipeline(model, ws_cfg, text_holder, clip, processor, vae):
    import torch.distributed as dist
    from models.dmd_scheduler import FlowGeneratorScheduler
    from models.pipelines.pipeline_dmd_keyframe import RefKFDMDGeneratorPipeline

    device = torch.device("cuda:0")
    vae = vae.to(device)
    scheduler = FlowGeneratorScheduler(
        start_timesteps=int(ws_cfg.dmd_start_steps),
        num_train_timesteps=int(ws_cfg.dmd_end_steps),
        shift=float(ws_cfg.gen_shift),
        use_timestep_transform=True,
        dmd_steps=int(ws_cfg.dmd_steps),
        rank=dist.get_rank(),
    )
    original_check = RefKFDMDGeneratorPipeline.check_inputs

    def check_inputs(
        self,
        prompt,
        negative_prompt,
        image,
        height,
        width,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        image_embeds=None,
        callback_on_step_end_tensor_inputs=None,
    ):
        return original_check(
            self,
            prompt,
            negative_prompt,
            image,
            height,
            width,
            prompt_embeds,
            negative_prompt_embeds,
            None,
            callback_on_step_end_tensor_inputs,
        )

    RefKFDMDGeneratorPipeline.check_inputs = check_inputs
    pipe = RefKFDMDGeneratorPipeline(
        tokenizer=text_holder["tokenizer"],
        text_encoder=text_holder["encoder"],
        image_encoder=clip,
        image_processor=processor,
        transformer=model,
        vae=vae,
        scheduler=scheduler,
        device=device,
        vae_compile=False,
    )
    if next(text_holder["encoder"].parameters()).device.type == "cuda":
        raise RuntimeError("text encoder is on GPU at the same time as the DiT")
    return pipe


def _sample(pipe, ws_cfg, job: TrajJob, scene_input: Path, work: Path, prompt_embeds, image_embeds, ref_index, seed: int):
    from src.data_utils import load_mutli_traj_dataset

    device = torch.device("cuda:0")
    meta = load_mutli_traj_dataset(
        cfg=ws_cfg,
        input_path=str(scene_input),
        output_path=str(work),
        view_id=job.view_id,
        traj_id=job.traj_id,
        device=device,
        ref_index=None if ref_index is None else ref_index.detach().cpu(),
        model_type=MODEL_TYPE,
        task_type="panorama",
    )
    kwargs = {key: value for key, value in meta.items() if value is not None}
    kwargs.pop("prompt", None)
    kwargs.update(
        prompt_embeds=prompt_embeds.to(device=device, dtype=torch.bfloat16),
        image_embeds=image_embeds.to(device=device, dtype=torch.bfloat16),
        generator=torch.Generator(device=device).manual_seed(int(seed)),
        output_type="pt",
        latent_cond_mode=str(ws_cfg.latent_cond_mode),
        mode="test",
        num_frames=diffusion_num_frames(job.sample_count),
    )
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    try:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            frames = pipe(**kwargs).frames[0].float()
    except (torch.cuda.OutOfMemoryError, torch.AcceleratorError) as exc:
        if isinstance(exc, torch.AcceleratorError) and "out of memory" not in str(exc).lower():
            raise
        peak = torch.cuda.max_memory_allocated(device) / (1024**3)
        alloc = torch.cuda.memory_allocated(device) / (1024**3)
        raise RuntimeError(
            f"VRAM // peak_gb={peak:.2f} allocated_gb={alloc:.2f} "
            f"WorldStereo did not fit; no BF16 fallback"
        ) from exc
    elapsed = time.perf_counter() - started
    peak = torch.cuda.max_memory_allocated(device) / (1024**3)
    return frames, elapsed, peak


def _save_frames(frames: torch.Tensor, rail_dir: Path):
    from PIL import Image

    keyframes = rail_dir / "keyframes"
    keyframes.mkdir(parents=True, exist_ok=True)
    array = frames.detach().float().clamp(0, 1).mul(255).byte().cpu().permute(0, 2, 3, 1).numpy()
    expected = {f"{index:03d}.png" for index in range(int(array.shape[0]))}
    for folder in (keyframes, rail_dir / "hires", rail_dir / "gate_masks"):
        if not folder.is_dir():
            continue
        for old in folder.glob("[0-9][0-9][0-9].png"):
            if old.name not in expected:
                old.unlink()
    for index, frame in enumerate(array):
        Image.fromarray(frame).save(keyframes / f"{index:03d}.png")
    return array


def _sources(bank, global_indices) -> list[str]:
    names = []
    seen = set()
    for index in global_indices:
        name = bank.fnames[int(index)]
        if name not in seen:
            seen.add(name)
            names.append(name)
    return names


def _select(scene: str, jobs: list[TrajJob], *, resume: bool, regen_rail: int | None):
    """Return (jobs to sample, snapshot to restore or None)."""
    if regen_rail is not None:
        selected, previous, stale_jobs = regen_plan(jobs, int(regen_rail))
        for job in stale_jobs:
            folder = rail_dir(scene, job)
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "STALE.txt").write_text(
                f"stale after re-gen of rail {int(regen_rail)}\n", encoding="utf-8"
            )
        snap = None if previous is None else rail_snapshot(scene, previous)
        return selected, snap
    if resume:
        selected = pending_jobs(scene, jobs)
        return selected, resume_snapshot(scene, jobs, selected)
    return list(jobs), None


def generate(
    scene: str,
    jobs: list[TrajJob],
    prompt: str,
    *,
    resume: bool = True,
    regen_rail: int | None = None,
    seed: int = D.SEED,
) -> int:
    """Sample ``jobs`` in order. Returns a process exit code."""
    from PIL import Image
    from transformers import CLIPImageProcessor

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    patch_single_rank()
    pin_dinov2(REPO_ROOT / "models" / "dinov2-base")
    config_path, _weights, transformer_cfg = ensure_files()
    sha = weights_sha256(_weights)
    scene_input = input_root(scene)
    work = scene_dir(scene) / "worldstereo" / "work"
    work.mkdir(parents=True, exist_ok=True)

    try:
        selected, snap = _select(scene, jobs, resume=resume, regen_rail=regen_rail)
    except ValueError as exc:
        _log(f"STATUS // {exc}")
        return 2
    if not selected:
        _log("STATUS // worldstereo already complete; nothing to resume")
        return 0

    prompt_embeds, text, tokenizer = _encode_text(prompt)
    text_holder = {"tokenizer": tokenizer, "encoder": text}
    clip = load_clip_vision(WAN_CFG).eval()
    processor = CLIPImageProcessor.from_pretrained(str(WAN_CFG / "image_processor"), local_files_only=True)
    image_embeds = {}
    for job in selected:
        image_embeds[job.view_id] = _encode_clip(scene_input / job.view_id / "start_frame.png", clip, processor)
        _log(f"STATUS // CLIP off GPU; {job.view_id} embeds {tuple(image_embeds[job.view_id].shape)}")
    clip.to("cpu")

    # Retrieval for the first selected trajectory happens before the DiT is resident.
    from omegaconf import OmegaConf

    ws_cfg = OmegaConf.create(json.loads(config_path.read_text(encoding="utf-8")))
    bank = _open_bank(ws_cfg, scene, scene_input)
    if snap is not None:
        if not (snap / "bank.json").is_file():
            _log(f"STATUS // missing memory snapshot {snap}")
            return 2
        load_bank_into(bank, snap)
        _log(f"STATUS // restored memory {snap} size={bank.mem_size}")

    model = None
    pipe = None
    vae = None
    total = len(selected)
    for step, job in enumerate(selected, start=1):
        _log(f"WS\tview\t{job.rail_index}\t{job.view}")
        ws_cfg.nframe = int(job.sample_count)
        bank.nframe = int(job.sample_count)
        _log(f"WS\trail\t{job.rail_index}\tretrieve")
        _log(f"STATUS // rail {job.rail_index} {job.view} {job.direction} retrieve")
        print(
            f"PROGRESS\tjob\t{step - 1}\t{total}\t{int(100 * (step - 1) / total)}\tretrieve {job.folder}",
            flush=True,
        )
        retrieved, ref_index, ref_index_dict, ref_w2cs, global_indices, w2cs, ks = _retrieve(
            bank, job, scene_input
        )
        _write_memory_inputs(work, job, retrieved, ref_index_dict, ref_w2cs)
        names = _sources(bank, global_indices)
        _log(f"WS\tretrieved\t{job.rail_index}\t{len(names)}")
        del retrieved
        gc.collect()
        torch.cuda.empty_cache()
        if torch.cuda.memory_allocated(device) > 512 * 1024 * 1024 and model is None:
            _log(
                f"STATUS // GPU still holds {torch.cuda.memory_allocated(device) / (1024**3):.2f} GB "
                "after retrieval"
            )

        if model is None:
            model, ws_cfg, _stats = build_dit(transformer_cfg, config_path, _weights, device)
            vae = load_vae()
            pipe = _pipeline(model, ws_cfg, text_holder, clip, processor, vae)

        ws_cfg.nframe = int(job.sample_count)
        _log(f"WS\trail\t{job.rail_index}\tsample")
        _log(f"STATUS // rail {job.rail_index} {job.view} sample")
        try:
            frames, elapsed, peak = _sample(
                pipe, ws_cfg, job, scene_input, work, prompt_embeds, image_embeds[job.view_id], ref_index, seed
            )
        except RuntimeError as exc:
            if str(exc).startswith("VRAM //"):
                _log(f"STATUS // {exc}")
                return 4
            raise
        out_dir = rail_dir(scene, job)
        array = _save_frames(frames, out_dir)
        stale = out_dir / "STALE.txt"
        if stale.is_file():
            stale.unlink()
        (out_dir / "retrieved.json").write_text(
            json.dumps(
                {
                    "view_id": job.view_id,
                    "traj_id": job.traj_id,
                    "global_indices": [int(i) for i in global_indices],
                    "fnames": names,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        kept_count = int(array.shape[0])
        (out_dir / "meta.json").write_text(
            json.dumps(
                {
                    "seed": int(seed),
                    "steps": D.STEPS,
                    "bucket": [D.SIZE_H, D.SIZE_W],
                    "weights_sha256": sha,
                    "seconds": elapsed,
                    "peak_gb": peak,
                    "frames": kept_count,
                    "rail_index": job.rail_index,
                    "direction": job.direction,
                    "view": job.view,
                    "layout": job.layout,
                    "positions": list(range(kept_count)),
                    "anchor": False,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        _log(
            f"STATUS // rail {job.rail_index} {job.view} frames={kept_count} "
            f"seconds={elapsed:.1f} peak_gb={peak:.2f} retrieved={len(names)}"
        )
        pil_frames = [Image.fromarray(frame) for frame in array]
        stored = _remember_frames(bank, pil_frames, w2cs, ks, job.view_id, job.traj_id, 0)
        _log(
            f"STATUS // memory stored {stored} frames, bank={len(bank.ref_frames)} "
            "shared across rails and views"
        )
        view_snap = view_snapshot(scene, job)
        save_bank(bank, view_snap)
        following = selected[step] if step < total else None
        if following is None or following.rail_index != job.rail_index:
            save_bank(bank, rail_snapshot(scene, job.rail_index))
        _log(f"WS\trail\t{job.rail_index}\tdone")
        _log(f"STATUS // memory saved {view_snap} size={len(bank.ref_frames)}")
        del frames, array, pil_frames, w2cs, ks, ref_index, ref_w2cs
        gc.collect()
        torch.cuda.empty_cache()
        print(
            f"PROGRESS\tjob\t{step}\t{total}\t{int(100 * step / total)}\tdone {job.folder}",
            flush=True,
        )

    del model, pipe, vae
    gc.collect()
    torch.cuda.empty_cache()
    _log("STATUS // worldstereo done; HiRes was not started")
    return 0


def _read_video(path: Path) -> list:
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open {path}")
    frames = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        cap.release()
    if not frames:
        raise RuntimeError(f"{path} has no frames")
    return frames


def generate_joint(
    scene: str,
    order: list[int],
    prompt: str,
    *,
    frames_per_rail: int = D.JOINT_FRAMES_PER_RAIL,
    resume: bool = True,
    seed: int = D.SEED,
) -> int:
    """Chained samples covering every kept rail. Later samples retrieve the shared bank."""
    import cv2
    import numpy as np
    from PIL import Image
    from transformers import CLIPImageProcessor

    from engine.worldstereo.adapter import pin_K
    from engine.worldstereo.joint import JointStop, joint_dir, split_joint, write_joint
    from engine.worldstereo.resume import joint_complete, memory_root

    if resume and joint_complete(scene, frames_per_rail):
        _log("STATUS // joint already complete; nothing to resume")
        return 0

    from omegaconf import OmegaConf

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    patch_single_rank()
    pin_dinov2(REPO_ROOT / "models" / "dinov2-base")
    config_path, _weights, transformer_cfg = ensure_files()
    scene_input = input_root(scene)
    plan_path = scene_input / "joint_plan.json"
    if not plan_path.is_file():
        _log(f"STATUS // missing {plan_path}")
        return 2
    manifest = json.loads(plan_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, list) or not manifest:
        _log("STATUS // joint plan is empty")
        return 2

    work = scene_dir(scene) / "worldstereo" / "work"
    work.mkdir(parents=True, exist_ok=True)
    dest = joint_dir(scene)
    prompt_embeds, text, tokenizer = _encode_text(prompt)
    text_holder = {"tokenizer": tokenizer, "encoder": text}
    clip = load_clip_vision(WAN_CFG).eval()
    processor = CLIPImageProcessor.from_pretrained(str(WAN_CFG / "image_processor"), local_files_only=True)

    ws_cfg = OmegaConf.create(json.loads(config_path.read_text(encoding="utf-8")))
    bank = _open_bank(ws_cfg, scene, scene_input)
    ranges: list[tuple[int, int]] = []
    cursor = 0
    for item in manifest:
        count = len(item["frames"])
        ranges.append((cursor, count))
        cursor += count

    saved_rows: list = []
    same_plan = False
    saved_path = dest / "joint.json"
    if saved_path.is_file():
        try:
            saved = json.loads(saved_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            saved = {}
        if isinstance(saved, dict) and int(saved.get("frames_per_rail") or 0) == int(frames_per_rail):
            rows = saved.get("frames")
            if isinstance(rows, list) and len(rows) == cursor:
                saved_rows = rows
                same_plan = True
    start_chunk = 0
    if resume and same_plan:
        for index, (origin, count) in enumerate(ranges):
            item_frames = manifest[index]["frames"]
            ready = True
            for offset, frame in enumerate(item_frames):
                row = saved_rows[origin + offset]
                png = dest / "keyframes" / f"{origin + offset:03d}.png"
                if (
                    int(row.get("rail", -1)) != int(frame["rail"])
                    or int(row.get("pose", -1)) != int(frame["pose"])
                    or not png.is_file()
                ):
                    ready = False
                    break
            if not ready:
                start_chunk = index
                break
        else:
            start_chunk = len(manifest)
    if (dest / "STALE.txt").is_file():
        start_chunk = 0
    if start_chunk > 0:
        snap = memory_root(scene) / "after_joint"
        if not (snap / "bank.json").is_file():
            _log("STATUS // joint memory snapshot missing; sampling the sequence again")
            start_chunk = 0
        else:
            load_bank_into(bank, snap)
            _log(f"STATUS // restored joint memory size={bank.mem_size}")

    clear_layers_from(scene, start_chunk)
    model = None
    pipe = None
    vae = None
    seconds = 0.0
    peak_gb = 0.0
    total = len(manifest)
    for index, item in enumerate(manifest):
        if index < start_chunk:
            continue
        anchor = int(item["anchor"])
        count = len(item["frames"])
        k = count + anchor
        view_id = str(item["view_id"])
        job = TrajJob(
            rail_index=int(item["frames"][0]["rail"]),
            direction="joint",
            view="front",
            view_id=view_id,
            traj_id="traj0",
            layout="joint",
            frames=k,
        )
        ws_cfg.nframe = k
        bank.nframe = k
        if anchor:
            src = dest / "keyframes" / f"{ranges[index][0] - 1:03d}.png"
            target = scene_input / view_id / "start_frame.png"
            if not src.is_file():
                _log(f"STATUS // joint chunk {index} is missing its start frame")
                return 2
            shutil.copyfile(src, target)
            if model is not None:
                model.to("cpu")
            if vae is not None:
                vae.to("cpu")
            if model is not None or vae is not None:
                torch.cuda.empty_cache()
            ensure_layers(scene, manifest, ranges, upto=index, device="cuda:0")
            refresh_guide(scene, manifest, index, device="cuda:0")
            _log(f"STATUS // joint chunk {index} continues the previous frame")
        image_embeds = _encode_clip(scene_input / view_id / "start_frame.png", clip, processor)
        if model is not None:
            model.to(device)
        if vae is not None:
            vae.to(device)
        _log(f"STATUS // joint chunk {index + 1}/{total} retrieve, {count} frames, bank={bank.mem_size}")
        print(
            f"PROGRESS\tjob\t{index}\t{total}\t{int(100 * index / total)}\tretrieve joint",
            flush=True,
        )
        retrieved, ref_index, ref_index_dict, ref_w2cs, _global, w2cs, ks = _retrieve(bank, job, scene_input)
        _write_memory_inputs(work, job, retrieved, ref_index_dict, ref_w2cs)
        del retrieved
        gc.collect()
        torch.cuda.empty_cache()
        if model is None:
            model, ws_cfg, _stats = build_dit(transformer_cfg, config_path, _weights, device)
            vae = load_vae()
            pipe = _pipeline(model, ws_cfg, text_holder, clip, processor, vae)
        ws_cfg.nframe = k
        bank.nframe = k
        _log(f"STATUS // joint chunk {index + 1}/{total} sample")
        try:
            frames, elapsed, peak = _sample(
                pipe, ws_cfg, job, scene_input, work, prompt_embeds, image_embeds, ref_index, seed
            )
        except RuntimeError as exc:
            if str(exc).startswith("VRAM //"):
                _log(f"STATUS // {exc}")
                return 4
            raise
        seconds += float(elapsed)
        peak_gb = max(peak_gb, float(peak))
        array = frames.detach().float().clamp(0, 1).mul(255).byte().cpu().permute(0, 2, 3, 1).numpy()
        kept = array[anchor:]
        if int(kept.shape[0]) != count:
            raise RuntimeError(f"joint chunk {index} kept {kept.shape[0]} frames, expected {count}")
        keyframes = dest / "keyframes"
        keyframes.mkdir(parents=True, exist_ok=True)
        origin = ranges[index][0]
        for offset, frame in enumerate(kept):
            Image.fromarray(frame).save(keyframes / f"{origin + offset:03d}.png")
        pil_frames = [Image.fromarray(frame) for frame in array]
        stored = _remember_frames(bank, pil_frames, w2cs, ks, view_id, job.traj_id, anchor)
        _log(f"STATUS // joint memory stored {stored} frames, bank={len(bank.ref_frames)}")
        save_bank(bank, memory_root(scene) / "after_joint")
        del frames, array, pil_frames, w2cs, ks, ref_index, ref_w2cs
        gc.collect()
        torch.cuda.empty_cache()

    if model is not None:
        model.to("cpu")
    if vae is not None:
        vae.to("cpu")
    torch.cuda.empty_cache()
    ensure_layers(scene, manifest, ranges, upto=len(manifest), device="cuda:0")

    images = []
    colors = []
    masks = []
    poses = []
    stops: list[JointStop] = []
    for item, (origin, count) in zip(manifest, ranges):
        anchor = int(item["anchor"])
        view_id = str(item["view_id"])
        for offset in range(count):
            bgr = cv2.imread(str(dest / "keyframes" / f"{origin + offset:03d}.png"), cv2.IMREAD_COLOR)
            if bgr is None:
                _log(f"STATUS // missing joint keyframe {origin + offset}")
                return 2
            images.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            row = item["frames"][offset]
            stops.append(JointStop(int(row["rail"]), int(row["pose"])))
        camera = json.loads((scene_input / view_id / "traj0" / "camera.json").read_text(encoding="utf-8"))
        extrinsic = np.asarray(camera["extrinsic"], dtype=np.float64)[anchor:]
        poses.append(extrinsic)
        colors.extend(_read_video(scene_input / view_id / "traj0" / "render.mp4")[anchor:])
        masks.extend(_read_video(scene_input / view_id / "traj0" / "render_mask.mp4")[anchor:])
    if (dest / "joint.json").is_file() and start_chunk > 0:
        try:
            previous = json.loads((dest / "joint.json").read_text(encoding="utf-8"))
            seconds += float(previous.get("seconds") or 0.0)
            peak_gb = max(peak_gb, float(previous.get("peak_gb") or 0.0))
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    write_joint(
        scene,
        order,
        stops,
        np.stack(images, axis=0),
        colors,
        masks,
        np.concatenate(poses, axis=0),
        pin_K(),
        frames_per_rail=int(frames_per_rail),
        seed=int(seed),
        seconds=seconds,
        peak_gb=peak_gb,
    )
    split_joint(scene, pin_K())
    _log(f"STATUS // joint frames={len(stops)} seconds={seconds:.1f} peak_gb={peak_gb:.2f}")
    _log("STATUS // worldstereo done; HiRes was not started")
    del model, pipe, vae
    gc.collect()
    torch.cuda.empty_cache()
    return 0
