"""Build the vendored LiteGS CUDA extensions into this venv, then run the 1-step spike.

Run from the venv (``tools/run_litegs_spike.bat`` or Setup on Home)::

    python tools/build_litegs.py            # build what is missing, then spike
    python tools/build_litegs.py --rebuild  # force pip reinstall of all three
    python tools/build_litegs.py --spike    # skip building, just spike + report

Extensions (all three live in ``vendor/litegs/litegs/submodules``):

* ``simple-knn``       -> ``simple_knn._C``     (initial scale from kNN distance)
* ``fused_ssim``       -> ``fused_ssim_cuda``   (fused L1 + SSIM loss)
* ``gaussian_raster``  -> ``litegs_fused``      (culling, projection, binning, raster)

Route: ``pip install --no-build-isolation --no-deps <dir>`` with the MSVC / CUDA env
from ``engine.cuda_jit.prepare()``. If ``litegs_fused`` fails under pip (LiteGS
issue #7: torch-injected ``-D__CUDA_NO_HALF_CONVERSIONS__``), the CMake route from the
LiteGS README is used and the resulting ``.pyd`` is picked up from the build dir by
``engine.splat.backend.litegs_env.add_litegs_paths()``.

The spike result is written to ``docs/env_report.txt`` (section ``## litegs spike``).
``engine.splat.status.litegs_spike_ok()`` reads that line; it is the Train gate.
"""

from __future__ import annotations

import argparse
import importlib
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from engine.splat.backend import litegs_env as LE  # noqa: E402
from engine.splat.status import LITEGS_SPIKE_OK_LINE, env_report_path  # noqa: E402

EXTENSIONS: list[tuple[str, Path, str]] = [
    ("simple_knn", LE.SIMPLE_KNN, "simple_knn._C"),
    ("fused_ssim", LE.FUSED_SSIM, "fused_ssim_cuda"),
    ("litegs_fused", LE.GAUSSIAN_RASTER, "litegs_fused"),
]


def log(msg: str) -> None:
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:
        # Console is cp1252 (launch.bat / Setup); the report file itself is UTF-8.
        enc = sys.stdout.encoding or "ascii"
        print(msg.encode(enc, errors="replace").decode(enc), flush=True)


def _importable(module: str) -> bool:
    """Fresh interpreter so a just-built .pyd is seen and a broken one does not poison us."""
    # torch first: the .pyd links c10 / torch_cuda DLLs that only torch puts on the DLL path.
    code = (
        "import sys\n"
        f"sys.path.insert(0, {str(REPO_ROOT)!r})\n"
        "import torch\n"
        "from engine.splat.backend.litegs_env import add_litegs_paths\n"
        "add_litegs_paths()\n"
        f"import {module}\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=_child_env(),
        check=False,
    )
    return proc.returncode == 0


def _child_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def _arch_env(env: dict[str, str]) -> str:
    """Pin TORCH_CUDA_ARCH_LIST to the present GPU so we build one arch, not all."""
    try:
        import torch

        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability(0)
            arch = f"{major}.{minor}"
            env.setdefault("TORCH_CUDA_ARCH_LIST", arch)
            return f"TORCH_CUDA_ARCH_LIST={env['TORCH_CUDA_ARCH_LIST']} (sm_{major}{minor})"
    except Exception as exc:  # pragma: no cover - torch missing is reported by env_report
        return f"TORCH_CUDA_ARCH_LIST: not set ({type(exc).__name__}: {exc})"
    return "TORCH_CUDA_ARCH_LIST: not set (no CUDA device)"


def _run_logged(cmd: list[str], cwd: Path, log_path: Path, env: dict[str, str]) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with log_path.open("w", encoding="utf-8", errors="replace") as fh:
        fh.write(f"# {datetime.now(timezone.utc).isoformat()}\n# cwd={cwd}\n# {' '.join(cmd)}\n\n")
        fh.flush()
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            stdout=fh,
            stderr=subprocess.STDOUT,
            env=env,
            check=False,
        )
        fh.write(f"\n# exit={proc.returncode} elapsed={time.time() - started:.1f}s\n")
    return proc.returncode


def _tail(path: Path, n: int = 25) -> str:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    return "\n".join(lines[-n:])


def pip_build(name: str, src: Path, env: dict[str, str]) -> tuple[bool, Path]:
    log_path = LE.BUILD_LOG_DIR / f"pip_{name}.log"
    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--no-build-isolation",
        "--no-deps",
        "--no-cache-dir",
        "--force-reinstall",
        str(src),
    ]
    log(f"[{name}] pip install {src.name} -> {log_path}")
    code = _run_logged(cmd, src, log_path, env)
    return code == 0, log_path


def cmake_build(env: dict[str, str]) -> tuple[bool, Path]:
    """LiteGS README route for gaussian_raster. Ninja generator; output next to build/."""
    log_path = LE.BUILD_LOG_DIR / "cmake_litegs_fused.log"
    cmake = shutil.which("cmake", path=env.get("PATH"))
    if cmake is None:
        log("[litegs_fused] cmake missing — pip install cmake")
        code = _run_logged(
            [sys.executable, "-m", "pip", "install", "cmake"],
            REPO_ROOT,
            LE.BUILD_LOG_DIR / "pip_cmake.log",
            env,
        )
        if code != 0:
            return False, LE.BUILD_LOG_DIR / "pip_cmake.log"
        cmake = shutil.which("cmake", path=env.get("PATH"))
        if cmake is None:
            return False, LE.BUILD_LOG_DIR / "pip_cmake.log"
    try:
        import torch

        prefix = torch.utils.cmake_prefix_path
        arch = "{}{}".format(*torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else "86"
    except Exception as exc:
        log(f"[litegs_fused] cmake route needs torch: {exc}")
        return False, log_path
    build = LE.GAUSSIAN_RASTER / "build"
    build.mkdir(parents=True, exist_ok=True)
    configure = [
        cmake,
        "-S",
        str(LE.GAUSSIAN_RASTER),
        "-B",
        str(build),
        "-G",
        "Ninja",
        "-DCMAKE_BUILD_TYPE=Release",
        f"-DCMAKE_PREFIX_PATH={prefix}",
        f"-DCMAKE_CUDA_ARCHITECTURES={arch}",
        f"-DPython3_EXECUTABLE={sys.executable}",
    ]
    log(f"[litegs_fused] cmake configure -> {log_path}")
    if _run_logged(configure, LE.GAUSSIAN_RASTER, log_path, env) != 0:
        return False, log_path
    build_cmd = [cmake, "--build", str(build), "--config", "Release"]
    log_path2 = LE.BUILD_LOG_DIR / "cmake_build_litegs_fused.log"
    log(f"[litegs_fused] cmake --build -> {log_path2}")
    if _run_logged(build_cmd, LE.GAUSSIAN_RASTER, log_path2, env) != 0:
        return False, log_path2
    return LE.cmake_pyd() is not None, log_path2


def build_all(rebuild: bool) -> list[str]:
    """Build missing extensions. Returns report lines."""
    lines: list[str] = []
    if not LE.vendored():
        lines.append(f"litegs vendor: MISSING {LE.LITEGS_ROOT} (git clone --recurse-submodules)")
        return lines
    sha = LE.read_pinned_sha()
    lines.append(f"litegs vendor: {LE.LITEGS_ROOT} sha={sha or 'unpinned'}")
    try:
        from engine.cuda_jit import prepare

        lines.extend(prepare())
    except Exception as exc:
        lines.append(f"cuda_jit.prepare: FAIL {type(exc).__name__}: {exc}")
    env = _child_env()
    if os.environ.get("VOTION_VCVARS") == "1":
        # setuptools: trust the vcvars env we just loaded instead of probing vswhere.
        env["DISTUTILS_USE_SDK"] = "1"
        env["MSSdk"] = "1"
    lines.append(_arch_env(env))
    for name, src, module in EXTENSIONS:
        if not rebuild and _importable(module):
            lines.append(f"{name} ({module}): already built")
            continue
        ok, log_path = pip_build(name, src, env)
        if ok and _importable(module):
            lines.append(f"{name} ({module}): pip build OK ({log_path.name})")
            continue
        lines.append(f"{name} ({module}): pip build FAIL ({log_path})")
        lines.append(_tail(log_path, 12))
        if name != "litegs_fused":
            continue
        ok, log_path = cmake_build(env)
        if ok:
            importlib.invalidate_caches()
            lines.append(f"{name} ({module}): cmake build OK ({LE.cmake_pyd()})")
        else:
            lines.append(f"{name} ({module}): cmake build FAIL ({log_path})")
            lines.append(_tail(log_path, 12))
    return lines


def spike() -> list[str]:
    """1-step LiteGS raster + backward on a dummy scene. Runs in a fresh interpreter."""
    code = (
        "import sys\n"
        f"sys.path.insert(0, {str(REPO_ROOT)!r})\n"
        "from engine.splat.backend.litegs_backend import spike_one_step\n"
        "print(spike_one_step())\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=_child_env(),
        check=False,
    )
    out = (proc.stdout or "").strip().splitlines()
    err = (proc.stderr or "").strip().splitlines()
    lines: list[str] = []
    if proc.returncode == 0 and out and out[-1].startswith("litegs raster 1-step: OK"):
        lines.extend(out[-6:])
    else:
        lines.append("litegs raster 1-step: FAIL")
        lines.extend(out[-10:])
        lines.extend(err[-25:])
        lines.append("P2 Train disabled until LiteGS builds; COLMAP export still proceeds (Brush).")
    return lines


def write_report(section_lines: list[str]) -> Path:
    """Replace (or append) the ``## litegs spike`` section of docs/env_report.txt."""
    path = env_report_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "## litegs spike\n" + "\n".join(section_lines) + "\n"
    if path.is_file():
        text = path.read_text(encoding="utf-8", errors="replace")
        pattern = re.compile(r"## litegs spike\n.*?(?=\n## |\Z)", re.S)
        if pattern.search(text):
            text = pattern.sub(lambda _m: body.rstrip("\n"), text)
        else:
            text = text.rstrip("\n") + "\n\n" + body
    else:
        text = (
            "## meta\n"
            f"written={datetime.now(timezone.utc).isoformat()}\n"
            f"executable={sys.executable}\n"
            "version_string=2.0\n\n" + body
        )
    path.write_text(text, encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build LiteGS CUDA extensions + spike")
    parser.add_argument("--rebuild", action="store_true", help="force pip reinstall of all three")
    parser.add_argument("--spike", action="store_true", help="skip building, just spike")
    args = parser.parse_args(argv)
    lines: list[str] = [f"built={datetime.now(timezone.utc).isoformat()}"]
    try:
        if not args.spike:
            lines.extend(build_all(rebuild=args.rebuild))
        else:
            from engine.cuda_jit import prepare

            lines.extend(prepare())
        lines.extend(spike())
    except Exception as exc:
        lines.append(f"build_litegs: FAIL {type(exc).__name__}: {exc}")
        lines.append(traceback.format_exc())
    path = write_report(lines)
    for line in lines:
        log(line)
    log(f"wrote {path}")
    return 0 if any(line.startswith(LITEGS_SPIKE_OK_LINE) for line in lines) else 1


if __name__ == "__main__":
    sys.exit(main())
