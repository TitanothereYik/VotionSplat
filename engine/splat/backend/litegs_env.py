"""Where the vendored LiteGS lives and how to make it importable. No torch import.

``vendor/litegs`` is a plain git clone pinned by ``vendor/litegs.sha`` (see
``tools/vendor_pins.json``). Its three CUDA extensions are built into this venv
by ``tools/build_litegs.py``: ``simple_knn._C``, ``fused_ssim_cuda`` and
``litegs_fused``. When the pip route for ``litegs_fused`` fails, the CMake route
leaves ``litegs_fused.pyd`` under ``gaussian_raster/build`` and
``add_litegs_paths()`` puts that directory on ``sys.path``.

Licence: LiteGS is under the Inria "Gaussian-Splatting License" (research and
evaluation only). It is built locally at Setup and never redistributed; the
installer ships no LiteGS binaries. ``HELP_SPLAT`` shows the licence line.
"""

from __future__ import annotations

import sys
from pathlib import Path

from engine.job import REPO_ROOT

LITEGS_ROOT = REPO_ROOT / "vendor" / "litegs"
LITEGS_SHA_FILE = REPO_ROOT / "vendor" / "litegs.sha"
LITEGS_LICENSE = LITEGS_ROOT / "LICENSE.md"
SUBMODULES = LITEGS_ROOT / "litegs" / "submodules"
GAUSSIAN_RASTER = SUBMODULES / "gaussian_raster"
SIMPLE_KNN = SUBMODULES / "simple-knn"
FUSED_SSIM = SUBMODULES / "fused_ssim"
BUILD_LOG_DIR = REPO_ROOT / "vendor" / "litegs_build"

LICENSE_LINE = (
    "LiteGS (Moore Threads) is under the Inria Gaussian-Splatting License: research and "
    "evaluation use only, no commercial use without Inria consent. Built locally at Setup; "
    "never redistributed."
)


def read_pinned_sha() -> str:
    try:
        return LITEGS_SHA_FILE.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def cmake_output_dirs() -> list[Path]:
    """Candidate dirs holding ``litegs_fused*.pyd`` from the CMake fallback."""
    build = GAUSSIAN_RASTER / "build"
    return [build / "Release", build / "RelWithDebInfo", build]


def cmake_pyd() -> Path | None:
    for folder in cmake_output_dirs():
        if not folder.is_dir():
            continue
        hits = sorted(folder.glob("litegs_fused*.pyd")) + sorted(folder.glob("litegs_fused*.so"))
        if hits:
            return hits[0]
    return None


def add_litegs_paths() -> list[str]:
    """Put ``vendor/litegs`` (the ``litegs`` package) and any CMake output on ``sys.path``."""
    notes: list[str] = []
    if not (LITEGS_ROOT / "litegs" / "__init__.py").is_file():
        notes.append(f"litegs: missing {LITEGS_ROOT}")
        return notes
    root = str(LITEGS_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    notes.append(f"litegs: sys.path += {root}")
    pyd = cmake_pyd()
    if pyd is not None:
        folder = str(pyd.parent)
        if folder not in sys.path:
            sys.path.insert(0, folder)
        notes.append(f"litegs_fused: cmake build {pyd}")
    return notes


def vendored() -> bool:
    return (LITEGS_ROOT / "litegs" / "render" / "__init__.py").is_file() and (
        GAUSSIAN_RASTER / "setup.py"
    ).is_file()
