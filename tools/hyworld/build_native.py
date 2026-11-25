"""Compile navmesh, the gsplat fork, and fused-ssim inside the hyworld venv.

Run with ``.venvs/hyworld/Scripts/python.exe``. ``prepare`` puts MSVC and CUDA
on the environment that the pip builds inherit.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from engine.cuda_jit import apply_file_patches, prepare  # noqa: E402


def _run(args: list[str], cwd: Path) -> None:
    print(">>", " ".join(args), flush=True)
    subprocess.check_call(args, cwd=str(cwd), env=os.environ.copy())


def main() -> int:
    site = Path(sys.prefix) / "Lib" / "site-packages"
    world = Path(sys.prefix) / "worldgen"
    os.environ["RECAST_PATH"] = str(REPO / "vendor" / "recastnavigation")
    os.environ["TORCH_CUDA_ARCH_LIST"] = "8.6"
    os.environ.setdefault("MAX_JOBS", "6")
    os.environ["TORCH_EXTENSIONS_DIR"] = str(Path(sys.prefix) / "torch_extensions")
    os.environ["DISTUTILS_USE_SDK"] = "1"
    os.environ["MSSdk"] = "1"
    os.environ["PIP_CONFIG_FILE"] = "nul"
    os.environ["PYTHONPATH"] = str(REPO) + os.pathsep + os.environ.get("PYTHONPATH", "")
    print("\n".join(prepare(site)), flush=True)

    _run([sys.executable, "-m", "pip", "install", "--no-build-isolation", "."], world / "third_party" / "navmesh")

    glm_src = REPO / ".venv" / "Lib" / "site-packages" / "gsplat" / "cuda" / "csrc" / "third_party" / "glm"
    glm_dst = world / "third_party" / "gsplat_maskgaussian" / "gsplat" / "cuda" / "csrc" / "third_party" / "glm"
    if glm_dst.exists():
        shutil.rmtree(glm_dst)
    shutil.copytree(glm_src, glm_dst)
    _run(
        [sys.executable, "-m", "pip", "install", "--no-build-isolation", "."],
        world / "third_party" / "gsplat_maskgaussian",
    )
    print("\n".join(apply_file_patches(site)), flush=True)

    fused = REPO / "vendor" / "fused_ssim"
    if fused.is_dir():
        _run([sys.executable, "-m", "pip", "install", "--no-build-isolation", str(fused)], REPO)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
