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
    """Interpreter for workers — the venv that launched the app.

    If the UI was started with pythonw.exe, workers still use python.exe so stdout pipes.
    """
    exe = Path(sys.executable)
    if exe.name.lower() == "pythonw.exe":
        sibling = exe.with_name("python.exe")
        if sibling.exists():
            return str(sibling)
    return str(exe)


def pano_argv(args: list[str]) -> list[str]:
    return [python_executable(), "-m", "engine.workers.pano", *args]


def caption_argv(image: str) -> list[str]:
    return [python_executable(), "-m", "engine.workers.caption", "--image", image]


def detect_fov_argv(image: str) -> list[str]:
    return [python_executable(), "-m", "engine.workers.detect_fov", "--image", image]


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
