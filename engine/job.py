"""Job bus helpers for subprocess workers.

Ports the Stop / cancel idea from Yik Votion WorldFM ``ui/job_control.py``
(in-process Event) onto **subprocess isolation**: the Qt process never
``import torch`` for inference. Workers are ``python -m engine.workers.*``.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def job_progress(current: int, total: int, message: str = "") -> None:
    """Stdout line for the always-on job strip. Workers only — Qt never calls this."""
    total = max(int(total), 0)
    current = max(int(current), 0)
    pct = int(round(100.0 * current / total)) if total else -1
    bits = ["PROGRESS", "job", str(current), str(total), str(pct)]
    if message:
        bits.append(str(message).replace("\t", " ").replace("\n", " "))
    print("\t".join(bits), flush=True)


class JobCancelled(Exception):
    """Raised when the user hits Stop / kill switch."""

    def __init__(self, message: str = "Job cancelled by user"):
        super().__init__(message)


def python_executable() -> str:
    """Interpreter for workers — the venv that launched the app, not Comfy python_embeded.

    If the UI was started with pythonw.exe, workers still use python.exe so stdout pipes.
    """
    exe = Path(sys.executable)
    if exe.name.lower() == "pythonw.exe":
        sibling = exe.with_name("python.exe")
        if sibling.exists():
            return str(sibling)
    return str(exe)


def dummy_argv() -> list[str]:
    return [python_executable(), "-m", "engine.workers.dummy"]


def pano_argv(args: list[str]) -> list[str]:
    return [python_executable(), "-m", "engine.workers.pano", *args]


def caption_argv(image: str) -> list[str]:
    return [python_executable(), "-m", "engine.workers.caption", "--image", image]


def detect_fov_argv(image: str) -> list[str]:
    return [python_executable(), "-m", "engine.workers.detect_fov", "--image", image]


def moge_argv(scene: str) -> list[str]:
    return [python_executable(), "-m", "engine.workers.moge", "--scene", scene]


def camplot_argv(scene: str, rail: int = 0) -> list[str]:
    return [
        python_executable(),
        "-m",
        "engine.workers.camplot",
        "--scene",
        scene,
        "--rail",
        str(int(rail)),
    ]


def known_pose_argv(scene: str) -> list[str]:
    return [python_executable(), "-m", "engine.workers.known_pose", "--scene", scene]


def splat_argv(
    scene: str,
    strategy: str,
    mode: str,
    max_steps: int | None = None,
    budget: int | None = None,
    res_scale: float | None = None,
    holdout: bool = False,
    color_space: str | None = None,
) -> list[str]:
    """LiteGS trainer worker. ``strategy`` is a defaults.STRATEGY_KEYS key (litegs | mcmc_compact)."""
    args = [
        python_executable(),
        "-m",
        "engine.workers.splat",
        "--scene",
        scene,
        "--strategy",
        strategy,
        "--mode",
        mode,
    ]
    if max_steps is not None:
        args.extend(["--max-steps", str(int(max_steps))])
    if budget is not None:
        args.extend(["--budget", str(int(budget))])
    if res_scale is not None and abs(float(res_scale) - 1.0) > 1e-9:
        args.extend(["--res-scale", str(float(res_scale))])
    if holdout:
        args.append("--holdout")
    if color_space:
        args.extend(["--color-space", str(color_space)])
    return args


def worldstereo_argv(
    scene: str,
    *,
    mode: str = "joint",
    resume: bool = True,
    regen_rail: int | None = None,
    prepare_only: bool = False,
    request: str | None = None,
    rails: list[int] | None = None,
    seed: int | None = None,
) -> list[str]:
    """One process for the whole scene. ``mode`` is joint or rail."""
    args = [
        python_executable(),
        "-m",
        "engine.workers.worldstereo",
        "--scene",
        scene,
        "--mode",
        str(mode or "joint"),
    ]
    if not resume:
        args.append("--restart")
    if regen_rail is not None:
        args.extend(["--regen-rail", str(int(regen_rail))])
    if prepare_only:
        args.append("--prepare-only")
    if request:
        args.extend(["--request", str(request)])
    if rails:
        args.extend(["--rails", ",".join(str(int(i)) for i in rails)])
    if seed is not None:
        args.extend(["--seed", str(int(seed))])
    return args


def pinhole_hires_argv(scene: str, source: str = "world") -> list[str]:
    return [
        python_executable(),
        "-m",
        "engine.workers.pinhole_hires",
        "--scene",
        scene,
        "--source",
        source,
    ]


def litegs_build_argv(rebuild: bool = False) -> list[str]:
    """Setup (Home): build the vendored LiteGS CUDA extensions + run the 1-step spike."""
    script = REPO_ROOT / "tools" / "build_litegs.py"
    args = [python_executable(), str(script)]
    if rebuild:
        args.append("--rebuild")
    return args


def download_argv(download_set: str) -> list[str]:
    script = REPO_ROOT / "tools" / "download_models.py"
    return [
        python_executable(),
        str(script),
        "--set",
        download_set,
        "--root",
        str(REPO_ROOT),
    ]


def huggingface_cache_root() -> Path:
    """Hub + Xet cache on the repo drive (D:), never the default C:\\Users\\…\\.cache."""
    return REPO_ROOT / ".hf_cache"


def apply_huggingface_cache_env(env: dict[str, str]) -> dict[str, str]:
    """Pin Hugging Face cache env vars so workers do not fill C:."""
    cache = huggingface_cache_root()
    env["HF_HOME"] = str(cache)
    env["HF_HUB_CACHE"] = str(cache / "hub")
    env["HF_XET_CACHE"] = str(cache / "xet")
    # Windows without Developer Mode cannot create Hub cache symlinks (WinError 1314).
    env["HF_HUB_DISABLE_SYMLINKS"] = "1"
    env["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    return env


def worker_env(cuda_index: int) -> dict[str, str]:
    """Environment for a worker subprocess. CUDA index is the user-picked device."""
    import os
    from pathlib import Path

    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["CUDA_DEVICE_INDEX"] = str(cuda_index)
    cuda = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8")
    if (cuda / "bin" / "nvcc.exe").is_file():
        env["CUDA_PATH"] = str(cuda)
        env["CUDA_HOME"] = str(cuda)
        env["PATH"] = str(cuda / "bin") + os.pathsep + env.get("PATH", "")
    scripts = Path(sys.executable).resolve().parent
    env["PATH"] = str(scripts) + os.pathsep + env.get("PATH", "")
    from engine.cuda_jit import msvc_cl_dir

    cl_dir = msvc_cl_dir()
    if cl_dir is not None:
        env["PATH"] = str(cl_dir) + os.pathsep + env.get("PATH", "")
    return apply_huggingface_cache_env(env)
