"""One-process WorldStereo memory-DMD run. The bank stays across rails.

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
from engine.worldstereo.adapter import TrajJob, input_root
from tools.spike_worldstereo import (
    WAN_CFG,
    _jsonable,
    build_dit,
    ensure_files,
    load_clip_vision,
    load_vae,
    patch_single_rank,
    pin_dinov2,
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
    bank.mem_size = len(frames)


def _encode_text(prompt: str):
    from transformers import AutoTokenizer

    from engine.video.wan_runtime import load_umt5

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


def _open_bank(ws_cfg, scene_input: Path):
    import imagesize
    from src.retrieval_wm import SimpleMemoryBank

    sample = next((scene_input / "pano_bank" / "images").glob("*.png"))
    width, height = imagesize.get(str(sample))
    return SimpleMemoryBank(
        cfg=ws_cfg,
        root_path=str(scene_input),
        image_width=width,
        image_height=height,
        device=torch.device("cuda:0"),
        max_reference=8,
        align_nframe=D.ALIGN_NFRAME,
        rank=0,
        world_size=1,
    )


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
            json.dumps(_jsonable(ref_index_dict), indent=2), encoding="utf-8"
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
    )
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    try:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            frames = pipe(**kwargs).frames[0].float()
    except torch.cuda.OutOfMemoryError as exc:
        peak = torch.cuda.max_memory_allocated(device) / (1024**3)
        alloc = torch.cuda.memory_allocated(device) / (1024**3)
        raise RuntimeError(
            f"VRAM // peak_gb={peak:.2f} allocated_gb={alloc:.2f} "
            f"WorldStereo did not fit; no BF16 fallback"
        ) from exc
    elapsed = time.perf_counter() - started
    peak = torch.cuda.max_memory_allocated(device) / (1024**3)
    return frames, elapsed, peak


def _save_frames(frames: torch.Tensor, rail_dir: Path) -> None:
    from PIL import Image

    keyframes = rail_dir / "keyframes"
    keyframes.mkdir(parents=True, exist_ok=True)
    array = frames.detach().float().clamp(0, 1).mul(255).byte().cpu().permute(0, 2, 3, 1).numpy()
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


def generate(
    scene: str,
    jobs: list[TrajJob],
    prompt: str,
    start_rail: int,
    only: bool,
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

    selected = [job for job in jobs if job.rail_index >= start_rail]
    if only:
        selected = [job for job in selected if job.rail_index == start_rail]
    if not selected:
        _log(f"STATUS // no rails at or after {start_rail}")
        return 2
    if only and start_rail > 0:
        for job in jobs:
            if job.rail_index > start_rail:
                rail_dir = scene_dir(scene) / job.output_rel
                rail_dir.mkdir(parents=True, exist_ok=True)
                (rail_dir / "STALE.txt").write_text(
                    f"stale after re-gen of rail {start_rail}\n", encoding="utf-8"
                )

    prompt_embeds, text, tokenizer = _encode_text(prompt)
    text_holder = {"tokenizer": tokenizer, "encoder": text}
    clip = load_clip_vision(WAN_CFG).eval()
    processor = CLIPImageProcessor.from_pretrained(str(WAN_CFG / "image_processor"), local_files_only=True)
    image_embeds = {}
    for job in selected:
        image_embeds[job.view_id] = _encode_clip(scene_input / job.view_id / "start_frame.png", clip, processor)
        _log(f"STATUS // CLIP off GPU; {job.view_id} embeds {tuple(image_embeds[job.view_id].shape)}")
    clip.to("cpu")

    # Retrieval for the first selected rail happens before the DiT is resident.
    from omegaconf import OmegaConf

    ws_cfg = OmegaConf.create(json.loads(config_path.read_text(encoding="utf-8")))
    bank = _open_bank(ws_cfg, scene_input)
    if start_rail > 0:
        previous = jobs[start_rail - 1]
        snap = scene_dir(scene) / "worldstereo" / "memory" / f"after_{previous.folder}"
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
        _log(f"WS\trail\t{job.rail_index}\tretrieve")
        _log(f"STATUS // rail {job.rail_index} {job.direction} retrieve")
        print(f"PROGRESS\tjob\t{step - 1}\t{total}\t{int(100 * (step - 1) / total)}\tretrieve {job.folder}", flush=True)
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

        _log(f"WS\trail\t{job.rail_index}\tsample")
        _log(f"STATUS // rail {job.rail_index} sample")
        try:
            frames, elapsed, peak = _sample(
                pipe, ws_cfg, job, scene_input, work, prompt_embeds, image_embeds[job.view_id], ref_index, seed
            )
        except RuntimeError as exc:
            if str(exc).startswith("VRAM //"):
                _log(f"STATUS // {exc}")
                return 4
            raise
        rail_dir = scene_dir(scene) / job.output_rel
        array = _save_frames(frames, rail_dir)
        stale = rail_dir / "STALE.txt"
        if stale.is_file():
            stale.unlink()
        (rail_dir / "retrieved.json").write_text(
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
        (rail_dir / "meta.json").write_text(
            json.dumps(
                {
                    "seed": int(seed),
                    "steps": D.STEPS,
                    "bucket": [D.SIZE_H, D.SIZE_W],
                    "weights_sha256": sha,
                    "seconds": elapsed,
                    "peak_gb": peak,
                    "frames": int(array.shape[0]),
                    "rail_index": job.rail_index,
                    "direction": job.direction,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        _log(
            f"STATUS // rail {job.rail_index} frames={array.shape[0]} "
            f"seconds={elapsed:.1f} peak_gb={peak:.2f} retrieved={len(names)}"
        )
        pil_frames = [Image.fromarray(frame) for frame in array]
        bank.update_memory(pil_frames, w2cs, ks, view_id=job.view_id, traj_id=job.traj_id)
        snap = scene_dir(scene) / "worldstereo" / "memory" / f"after_{job.folder}"
        save_bank(bank, snap)
        _log(f"WS\trail\t{job.rail_index}\tdone")
        _log(f"STATUS // memory saved {snap} size={len(bank.ref_frames)}")
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
