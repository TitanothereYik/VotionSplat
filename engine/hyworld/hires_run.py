"""SeedVR2 3B over expanded HY-World clips. Worker only: runs in .venvs/seedvr2."""

from __future__ import annotations

import gc
import json
import os
import shutil
import sys
import time

import cv2

from engine.hyworld import hires as HR
from engine.hyworld.paths import list_trajectories, ordered_ids, read_selection, traj_state
from engine.job import job_progress

GB = 1024**3


def _hy(*fields) -> None:
    print("HY\t" + "\t".join(str(field) for field in fields), flush=True)


def _log(msg: str) -> None:
    print(msg, flush=True)


def _load_cli():
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = HR.ALLOC_CONF
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_DEVICE_INDEX", "0"))
    vendor = str(HR.VENDOR)
    if vendor not in sys.path:
        sys.path.insert(0, vendor)
    saved = sys.argv
    sys.argv = [str(HR.VENDOR / "inference_cli.py")]  # its import-time pre-parser reads sys.argv
    try:
        import inference_cli as cli
    finally:
        sys.argv = saved
    return cli


def _args(cli, level: str, short_side: int, scale: int):
    dec, dec_o, enc, enc_o = HR.tiles(level, scale)
    size, overlap = HR.batch(level, scale)
    argv = [
        str(HR.VENDOR / "inference_cli.py"), "clip.mp4",
        "--output_format", "png", "--model_dir", str(HR.MODEL_DIR), "--dit_model", HR.DIT,
        "--resolution", str(short_side * scale), "--batch_size", str(size), "--seed", str(HR.SEED),
        "--temporal_overlap", str(overlap), "--color_correction", HR.COLOR,
        "--dit_offload_device", "cpu", "--vae_offload_device", "cpu", "--tensor_offload_device", "cpu",
        "--cache_dit", "--cache_vae",
    ]
    if dec:
        argv += ["--vae_decode_tiled", "--vae_decode_tile_size", str(dec), "--vae_decode_tile_overlap", str(dec_o)]
    if enc:
        argv += ["--vae_encode_tiled", "--vae_encode_tile_size", str(enc), "--vae_encode_tile_overlap", str(enc_o)]
    saved = sys.argv
    sys.argv = argv
    try:
        return cli.parse_arguments()
    finally:
        sys.argv = saved


class Peak:
    """Card peaks for one clip. SeedVR2 resets the peak between phases when its debug log is on."""

    def __init__(self, torch) -> None:
        self.torch = torch
        self.alloc = self.rsv = 0.0
        self._real = torch.cuda.reset_peak_memory_stats

        def wrapped(device=None):
            self.take()
            self._real(device)

        torch.cuda.reset_peak_memory_stats = wrapped

    def take(self) -> tuple[float, float]:
        cuda = self.torch.cuda
        self.alloc = max(self.alloc, cuda.max_memory_allocated() / GB)
        self.rsv = max(self.rsv, cuda.max_memory_reserved() / GB)
        return self.alloc, self.rsv

    def start(self) -> None:
        self.alloc = self.rsv = 0.0
        self._real()


def _emit_fit(current: str, asked: str, budget: float, rows: dict, key: str, scale: int) -> None:
    payload = {
        "level": current,
        "asked": asked,
        "scale": scale,
        "budget_gb": round(budget, 2),
        "need_gb": round(HR.need_gb(current, rows, scale), 2),
        "measured": "reserved_gb" in (rows.get(current) or {}),
        "key": key,
        "cap": HR.CAP_WORKS,
    }
    _hy("hfit", json.dumps(payload))
    _log(
        f"STATUS // hires fit scale={scale}x level={current} asked={asked} budget_gb={budget:.2f} "
        f"need_gb={payload['need_gb']:.2f} measured={payload['measured']} cap={HR.CAP_WORKS}"
    )


def _is_oom(torch, exc: BaseException) -> bool:
    return isinstance(exc, torch.OutOfMemoryError) or "out of memory" in str(exc).lower()


def run(
    scene: str,
    *,
    only: str = "",
    resume: bool = True,
    level: str = "auto",
    cap_gb: float = 0.0,
    bench: bool = False,
    verbose: bool = False,
    scale: int = HR.SCALE,
) -> int:
    if scale not in HR.SCALES:
        _log(f"STATUS // hires: scale must be one of {HR.SCALES}")
        return 2
    for name in (HR.DIT, HR.VAE):
        path = HR.MODEL_DIR / name
        if not path.is_file():
            _log(f"STATUS // HiRes needs {path}")
            return 1
    ids = [only] if only else ordered_ids(read_selection(scene) or list_trajectories(scene))
    ids = [item for item in ids if traj_state(scene, item) == "expanded"]
    if not ids:
        _log("STATUS // hires: no expanded paths")
        return 1
    if bench and (not only or level == "auto"):
        _log("STATUS // --bench needs --only and a fixed --level")
        return 2
    if level not in HR.choices(scale):
        _log(f"STATUS // hires: level {level} is not a {scale}x level, use one of {HR.choices(scale)}")
        return 2

    HR.clean_scratch(scene, scale)
    _hy("hqueue", json.dumps(ids))
    cli = _load_cli()
    import torch

    cli.debug.enabled = verbose
    free, total = torch.cuda.mem_get_info()
    total_gb = total / GB
    budget = free / GB - HR.RESERVE_GB
    if cap_gb > 0:
        budget = min(budget, cap_gb)
    if HR.CAP_WORKS:
        torch.cuda.set_per_process_memory_fraction(max(0.01, min(1.0, budget / total_gb)))
    key = HR.fit_key(torch.cuda.get_device_name(0), total_gb, scale)
    rows = HR.read_rows(key)
    current = HR.pick(budget, rows, level, scale)
    bench_level = current
    _emit_fit(current, level, budget, rows, key, scale)

    cache: dict = {}
    peak = Peak(torch)
    for index, item in enumerate(ids):
        job_progress(index, len(ids), f"hires {scale}x {item}")
        out = HR.bench_dir(scene, bench_level, item, scale) if bench else HR.hires_dir(scene, item, scale)
        if resume and not bench and HR.hires_state(scene, item, scale) == "done":
            _hy("hires", item, "skip", "0", "0")
            continue
        src = HR.source_clip(scene, item)
        cap = cv2.VideoCapture(str(src))
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        if count != 21:
            _hy("hires", item, "failed", "0", "0")
            raise RuntimeError(f"HIRES // {item} has {count} frames, expected 21")
        taken = HR.stamp(src)
        while True:
            _hy("hires", item, "upscale", "0", "0")
            tmp = HR.hires_root(scene, scale) / "_tmp"
            if tmp.exists():
                shutil.rmtree(tmp)
            tmp.mkdir(parents=True)
            peak.start()
            started = time.perf_counter()
            caught = None
            try:
                cli.process_single_file(
                    str(src),
                    _args(cli, current, min(width, height), scale),
                    ["0"],
                    output_path=str(tmp),
                    format_auto_detected=False,
                    runner_cache=cache,
                )
            except Exception as exc:
                caught = exc
            seconds = time.perf_counter() - started
            alloc, reserved = peak.take()
            spilled = reserved > total_gb + 0.25
            if caught is not None and not _is_oom(torch, caught):
                _hy("hires", item, "failed", "0", "0")
                raise caught
            if caught is not None:
                rows = HR.fail(rows, current, budget)
                HR.write_rows(key, rows)
                cache.clear()
                gc.collect()
                torch.cuda.empty_cache()
                nxt = HR.next_level(current, scale) if level == "auto" else None
                if nxt:
                    old = current
                    current = nxt
                    _log(f"STATUS // hires {item} out of memory at {old}, stepping down to {current}")
                    _emit_fit(current, level, budget, rows, key, scale)
                    continue
                _hy("hires", item, "failed", "0", "0")
                return 1
            pngs = sorted(tmp.rglob("*.png"))
            if len(pngs) != 21:
                _hy("hires", item, "failed", "0", "0")
                raise RuntimeError(f"HIRES // {item} wrote {len(pngs)} frames, expected 21")
            expect = (height * scale, width * scale, 3)
            for path in pngs:
                image = cv2.imread(str(path), cv2.IMREAD_COLOR)
                if image is None or tuple(image.shape) != expect:
                    got = None if image is None else tuple(image.shape)
                    _hy("hires", item, "failed", "0", "0")
                    raise RuntimeError(f"HIRES // {item} frame {path.name} is {got}, expected {expect}")
            meta = {
                "source": str(src),
                "size": taken["size"],
                "mtime_ns": taken["mtime_ns"],
                "frames": 21,
                "width": expect[1],
                "height": expect[0],
                "scale": scale,
                "level": current,
                "seconds": round(seconds, 1),
                "peak_gb": round(alloc, 2),
                "reserved_gb": round(reserved, 2),
                "model": HR.DIT,
                "seed": HR.SEED,
                "color_correction": HR.COLOR,
            }
            HR.publish(out, pngs, meta)
            if spilled:
                rows = HR.fail(rows, current, budget)
                _log(
                    f"STATUS // hires {item} spilled {reserved - total_gb:.1f} GB "
                    f"into system RAM at {current}"
                )
                nxt = HR.next_level(current, scale) if level == "auto" else None
                if nxt:
                    current = nxt
                    _emit_fit(current, level, budget, rows, key, scale)
            else:
                rows = HR.record(rows, current, seconds, alloc, reserved, budget if HR.CAP_WORKS else 0.0)
            HR.write_rows(key, rows)
            _hy("hires", item, "done", f"{seconds:.1f}", f"{reserved:.2f}")
            break
    job_progress(len(ids), len(ids), "hires done")
    return 0
