"""Windows CUDA JIT workarounds for gsplat against this venv's torch.

Measured on this PC (torch 2.11.0+cu128, gsplat 1.5.3, CUDA toolkit 12.8, MSVC 14.43):

1. ``CUDACachingAllocator.h`` names a ctor param ``small``. Windows ``rpcndr.h``
   does ``#define small char``, so nvcc sees ``bool char``.
   Upstream: pytorch/pytorch#173112, PR #179100 (rename to ``is_small``).
   The 2.11 cu128 wheel does not include that PR.

2. gsplat ``cuda/_backend.py`` puts GCC ``-Wno-attributes`` in ``extra_cflags``.
   MSVC ``cl`` treats it as ``/Wno-attributes`` and fails with D8021.
   Upstream: nerfstudio-project/gsplat#809. Still present in 1.5.3.

3. ``torch/csrc/dynamo/compiled_autograd.h`` ``IValuePacker<T>::packed_type()`` uses
   ``if constexpr (::std::is_same_v<T, ::std::string>)``. Any ``.cu`` translation unit
   that includes ``torch/extension.h`` (LiteGS ``simple-knn``, ``fused_ssim``,
   ``gaussian_raster``) fails under nvcc 12.8 + MSVC 14.43 with
   ``error C2872: 'std': ambiguous symbol`` (measured 2026-09-07, torch 2.11.0+cu128).
   torch already skips that body when ``_WIN32 && USE_CUDA`` (its own Windows build,
   see pytorch/pytorch#144707); extensions do not define ``USE_CUDA``. The patch adds
   ``|| defined(__CUDACC__)`` to that guard so nvcc takes the same not-implemented
   branch. compiled autograd is unsupported on Windows anyway.

4. This PC's CUDA 12.8 toolkit is a partial install: ``include/`` has no cuSPARSE,
   cuBLAS or cuSOLVER headers, but LiteGS ``compact.cu`` includes
   ``<ATen/cuda/CUDAContext.h>`` which needs ``cusparse.h``, ``cublas_v2.h``,
   ``cublasLt.h`` and ``cusolverDn.h`` for declarations only (nothing links against
   them). ``apply_cuda_math_headers()`` appends the include dir of a sibling 12.x
   toolkit that has them (v12.6 here) to the ``INCLUDE`` env var. ``-I v12.8/include``
   stays first, so only the missing headers resolve from the sibling.

``apply_file_patches()`` is disk-only (no torch import) so launch.bat can run it.
``prepare()`` also installs a runtime MSVC flag filter — call it **before**
``import gsplat``. The Qt process must not import torch; do not call ``prepare()``
from ``app.main``.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import sysconfig
from pathlib import Path

# pytorch/pytorch#179100 — two-line rename. Do not match ``is_small_pool``.
_STOCK_CTOR = "StreamSegmentSize(cudaStream_t s, bool small, size_t sz)"
_FIXED_CTOR = "StreamSegmentSize(cudaStream_t s, bool is_small, size_t sz)"
_STOCK_INIT = ": stream(s), is_small_pool(small), total_size(sz)"
_FIXED_INIT = ": stream(s), is_small_pool(is_small), total_size(sz)"

_STOCK_GSPLAT_CFLAGS = 'extra_cflags = [opt_level, "-Wno-attributes"]'
_FIXED_GSPLAT_CFLAGS = (
    'extra_cflags = [opt_level] if os.name == "nt" else [opt_level, "-Wno-attributes"]'
)

# Workaround 3 — nvcc .cu TUs take torch's own Windows "not implemented" branch.
_STOCK_CA_GUARD = "#if defined(_WIN32) && (defined(USE_CUDA) || defined(USE_ROCM))"
_FIXED_CA_GUARD = (
    "#if defined(_WIN32) && (defined(USE_CUDA) || defined(USE_ROCM) || defined(__CUDACC__))"
)

_GCC_ONLY_CFLAGS = frozenset({"-Wno-attributes", "/Wno-attributes"})
_FILTER_MARK = "_votion_msvc_cflag_filter"

_VCVARS64 = Path(
    r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
)
_MSVC_ROOT = Path(
    r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC"
)
_CUDA_ROOT = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8")
_VCVARS_SKIP = ("PYTHONHOME", "PYTHONPATH", "VIRTUAL_ENV")
# Workaround 4 — declaration-only headers ATen/cuda/CUDAContextLight.h pulls in.
_CUDA_MATH_HEADERS = ("cusparse.h", "cublas_v2.h", "cublasLt.h", "cusolverDn.h")


def msvc_cl_dir() -> Path | None:
    """Hostx64 cl.exe directory, or None if Build Tools are missing."""
    if os.name != "nt" or not _MSVC_ROOT.is_dir():
        return None
    hits = sorted(_MSVC_ROOT.glob("*/bin/Hostx64/x64/cl.exe"))
    return hits[-1].parent if hits else None


def apply_cuda_path() -> str:
    """Put CUDA 12.8 on PATH before vcvars / gsplat JIT. No torch import."""
    if not (_CUDA_ROOT / "bin" / "nvcc.exe").is_file():
        return f"CUDA_PATH: missing {_CUDA_ROOT}"
    os.environ["CUDA_PATH"] = str(_CUDA_ROOT)
    os.environ["CUDA_HOME"] = str(_CUDA_ROOT)
    os.environ["PATH"] = str(_CUDA_ROOT / "bin") + os.pathsep + os.environ.get("PATH", "")
    return f"CUDA_PATH={_CUDA_ROOT}"


def _has_math_headers(include_dir: Path) -> bool:
    return all((include_dir / name).is_file() for name in _CUDA_MATH_HEADERS)


def apply_cuda_math_headers() -> str:
    """Append a sibling 12.x toolkit include dir to INCLUDE when 12.8 lacks the math headers."""
    primary = _CUDA_ROOT / "include"
    if not primary.is_dir():
        return f"cuda math headers: missing {primary}"
    if _has_math_headers(primary):
        return "cuda math headers: present in v12.8"
    siblings = sorted(
        (p for p in _CUDA_ROOT.parent.glob("v12.*") if p != _CUDA_ROOT and _has_math_headers(p / "include")),
        key=lambda p: p.name,
        reverse=True,
    )
    if not siblings:
        return (
            "cuda math headers: MISSING in v12.8 and no sibling 12.x toolkit has them "
            "(install cuSPARSE/cuBLAS/cuSOLVER dev headers)"
        )
    extra = str(siblings[0] / "include")
    current = os.environ.get("INCLUDE", "")
    parts = [p for p in current.split(os.pathsep) if p]
    if extra not in parts:
        parts.append(extra)
        os.environ["INCLUDE"] = os.pathsep.join(parts)
    return f"cuda math headers: v12.8 lacks {_CUDA_MATH_HEADERS[0]}; INCLUDE += {extra}"


def apply_venv_scripts_path() -> str:
    """ninja.exe lives next to the venv interpreter."""
    scripts = Path(sys.executable).resolve().parent
    os.environ["PATH"] = str(scripts) + os.pathsep + os.environ.get("PATH", "")
    return f"venv-scripts={scripts}"


def apply_vcvars64() -> str:
    """Load MSVC env so gsplat JIT can find cl.exe. No-op if vcvars64 is missing."""
    if os.name != "nt":
        return "vcvars64: skip (not Windows)"
    if os.environ.get("VOTION_VCVARS") == "1":
        return "vcvars64: already applied"
    notes: list[str] = []
    if _VCVARS64.is_file():
        proc = subprocess.run(
            f'call "{_VCVARS64}" && set',
            shell=True,
            capture_output=True,
            check=False,
        )
        raw = proc.stdout or b""
        text = raw.decode("utf-8", errors="replace")
        if text.count("=") < 5:
            text = raw.decode("oem", errors="replace")
        n = 0
        for line in text.splitlines():
            if "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            if not key or key in _VCVARS_SKIP or key.startswith("PYTHON"):
                continue
            os.environ[key] = val
            n += 1
        notes.append(f"set-vars={n}")
    cl_dir = msvc_cl_dir()
    if cl_dir is not None:
        os.environ["PATH"] = str(cl_dir) + os.pathsep + os.environ.get("PATH", "")
        notes.append(f"cl={cl_dir / 'cl.exe'}")
    os.environ["VOTION_VCVARS"] = "1"
    if not notes:
        return f"vcvars64: missing {_VCVARS64}"
    return "vcvars64: " + "; ".join(notes)


def site_packages(root: Path | None = None) -> Path:
    if root is not None:
        return root
    return Path(sysconfig.get_path("purelib"))


def allocator_header(site: Path) -> Path:
    return site / "torch" / "include" / "c10" / "cuda" / "CUDACachingAllocator.h"


def gsplat_backend(site: Path) -> Path:
    return site / "gsplat" / "cuda" / "_backend.py"


def compiled_autograd_header(site: Path) -> Path:
    return site / "torch" / "include" / "torch" / "csrc" / "dynamo" / "compiled_autograd.h"


def patch_compiled_autograd(text: str) -> tuple[str, str]:
    """Return (new_text, status) for workaround 3."""
    if _FIXED_CA_GUARD in text:
        return text, "already"
    if _STOCK_CA_GUARD not in text:
        return text, "unrecognized"
    return text.replace(_STOCK_CA_GUARD, _FIXED_CA_GUARD, 1), "applied"


def patch_allocator_header(text: str) -> tuple[str, str]:
    """Return (new_text, status) for workaround 1."""
    if _FIXED_CTOR in text and _FIXED_INIT in text:
        return text, "already"
    if _STOCK_CTOR not in text or _STOCK_INIT not in text:
        return text, "unrecognized"
    out = text.replace(_STOCK_CTOR, _FIXED_CTOR, 1).replace(_STOCK_INIT, _FIXED_INIT, 1)
    return out, "applied"


def patch_gsplat_backend(text: str) -> tuple[str, str]:
    """Return (new_text, status) for workaround 2."""
    if _FIXED_GSPLAT_CFLAGS in text:
        return text, "already"
    if _STOCK_GSPLAT_CFLAGS not in text:
        return text, "unrecognized"
    out = text.replace(_STOCK_GSPLAT_CFLAGS, _FIXED_GSPLAT_CFLAGS, 1)
    return out, "applied"


def _write_if_changed(path: Path, original: str, updated: str) -> None:
    if updated == original:
        return
    path.write_bytes(updated.encode("utf-8"))


def apply_file_patches(site: Path | None = None) -> list[str]:
    """Patch site-packages files. Idempotent. Does not import torch."""
    site = site_packages(site)
    lines: list[str] = []

    header = allocator_header(site)
    if not header.is_file():
        lines.append(f"torch small→is_small: missing {header}")
    else:
        raw = header.read_text(encoding="utf-8")
        updated, status = patch_allocator_header(raw)
        if status == "applied":
            _write_if_changed(header, raw, updated)
        lines.append(f"torch small→is_small: {status} ({header})")

    backend = gsplat_backend(site)
    if not backend.is_file():
        lines.append(f"gsplat -Wno-attributes: missing {backend}")
    else:
        raw = backend.read_text(encoding="utf-8")
        updated, status = patch_gsplat_backend(raw)
        if status == "applied":
            _write_if_changed(backend, raw, updated)
        lines.append(f"gsplat -Wno-attributes: {status} ({backend})")

    ca_header = compiled_autograd_header(site)
    if not ca_header.is_file():
        lines.append(f"torch compiled_autograd nvcc guard: missing {ca_header}")
    else:
        raw = ca_header.read_text(encoding="utf-8")
        updated, status = patch_compiled_autograd(raw)
        if status == "applied":
            _write_if_changed(ca_header, raw, updated)
        lines.append(f"torch compiled_autograd nvcc guard: {status} ({ca_header})")

    return lines


def _strip_gcc_only(flags: list[str] | None) -> list[str] | None:
    if flags is None:
        return None
    return [flag for flag in flags if flag.strip() not in _GCC_ONLY_CFLAGS]


def install_msvc_flag_filter() -> str:
    """Strip GCC-only cflags before torch writes the ninja file. Imports torch."""
    if os.name != "nt":
        return "msvc_flag_filter: skip (not Windows)"
    import torch.utils.cpp_extension as ext

    current = ext._write_ninja_file_to_build_library
    if getattr(current, _FILTER_MARK, False):
        return "msvc_flag_filter: already installed"

    def wrapped(*args, **kwargs):  # type: ignore[no-untyped-def]
        args_list = list(args)
        if len(args_list) >= 5:
            args_list[3] = _strip_gcc_only(args_list[3])
            args_list[4] = _strip_gcc_only(args_list[4])
        if "extra_cflags" in kwargs:
            kwargs["extra_cflags"] = _strip_gcc_only(kwargs["extra_cflags"])
        if "extra_cuda_cflags" in kwargs:
            kwargs["extra_cuda_cflags"] = _strip_gcc_only(kwargs["extra_cuda_cflags"])
        return current(*args_list, **kwargs)

    setattr(wrapped, _FILTER_MARK, True)
    ext._write_ninja_file_to_build_library = wrapped
    return "msvc_flag_filter: installed"


def prepare(site: Path | None = None) -> list[str]:
    """Re-apply disk patches, MSVC env, and the flag filter. Call before gsplat."""
    lines = apply_file_patches(site)
    lines.append(apply_venv_scripts_path())
    lines.append(apply_cuda_path())
    try:
        lines.append(apply_vcvars64())
    except Exception as exc:
        lines.append(f"vcvars64: FAIL {type(exc).__name__}: {exc}")
    # After vcvars: it rewrites INCLUDE, so the math-header dir must go on afterwards.
    try:
        lines.append(apply_cuda_math_headers())
    except Exception as exc:
        lines.append(f"cuda math headers: FAIL {type(exc).__name__}: {exc}")
    try:
        lines.append(install_msvc_flag_filter())
    except Exception as exc:
        lines.append(f"msvc_flag_filter: FAIL {type(exc).__name__}: {exc}")
    return lines


def self_test() -> int:
    stock_header = (
        "struct StreamSegmentSize {\n"
        "  StreamSegmentSize(cudaStream_t s, bool small, size_t sz)\n"
        "      : stream(s), is_small_pool(small), total_size(sz) {}\n"
        "};\n"
    )
    once, st1 = patch_allocator_header(stock_header)
    twice, st2 = patch_allocator_header(once)
    if st1 != "applied" or "bool small," in once or _FIXED_CTOR not in once:
        print("FAIL torch first apply", st1, once, sep="\n")
        return 1
    if st2 != "already" or twice != once:
        print("FAIL torch idempotent", st2)
        return 1
    junk, stj = patch_allocator_header("no ctor here\n")
    if stj != "unrecognized" or junk != "no ctor here\n":
        print("FAIL torch unrecognized")
        return 1

    stock_gsplat = f"        {_STOCK_GSPLAT_CFLAGS}\n"
    g1, gs1 = patch_gsplat_backend(stock_gsplat)
    g2, gs2 = patch_gsplat_backend(g1)
    if gs1 != "applied" or _STOCK_GSPLAT_CFLAGS in g1 or _FIXED_GSPLAT_CFLAGS not in g1:
        print("FAIL gsplat first apply", gs1, g1, sep="\n")
        return 1
    if gs2 != "already" or g2 != g1:
        print("FAIL gsplat idempotent", gs2)
        return 1

    stock_ca = f"  {_STOCK_CA_GUARD}\n    TORCH_CHECK_NOT_IMPLEMENTED(false, \"x\");\n#else\n"
    c1, cs1 = patch_compiled_autograd(stock_ca)
    c2, cs2 = patch_compiled_autograd(c1)
    if cs1 != "applied" or _STOCK_CA_GUARD in c1.replace(_FIXED_CA_GUARD, "") or _FIXED_CA_GUARD not in c1:
        print("FAIL compiled_autograd first apply", cs1, c1, sep="\n")
        return 1
    if cs2 != "already" or c2 != c1:
        print("FAIL compiled_autograd idempotent", cs2)
        return 1
    if patch_compiled_autograd("nothing\n")[1] != "unrecognized":
        print("FAIL compiled_autograd unrecognized")
        return 1

    stripped = _strip_gcc_only(["-O3", "-Wno-attributes", "/MD"])
    if stripped != ["-O3", "/MD"]:
        print("FAIL strip", stripped)
        return 1
    print("cuda_jit self-test: OK")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Apply Windows CUDA JIT workarounds")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument(
        "--prepare",
        action="store_true",
        help="Also install the runtime MSVC flag filter (imports torch)",
    )
    args = parser.parse_args(argv)
    if args.self_test:
        return self_test()
    lines = prepare() if args.prepare else apply_file_patches()
    print("\n".join(lines), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
