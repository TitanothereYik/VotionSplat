"""Write docs/env_report.txt — anti-hallucination record for this machine.

Records nvidia-smi, python, torch, CUDA, and the gsplat package version. Never invents a
torch patch: whatever import torch reports is what gets written.
"""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
OUT = REPO_ROOT / "docs" / "env_report.txt"


def run(cmd: list[str]) -> str:
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        body = (proc.stdout or "") + (proc.stderr or "")
        return body.strip() or f"(exit {proc.returncode}, no output)"
    except FileNotFoundError:
        return f"NOT FOUND: {' '.join(cmd)}"


def section(title: str, body: str) -> str:
    return f"## {title}\n{body}\n"


def gsplat_spike() -> str:
    """Version record only. gsplat is not imported
    here any more because ``import gsplat`` triggers its JIT build (and relinking fails
    while a splat view worker holds ``gsplat_cuda.pyd``). Version from metadata only."""
    lines: list[str] = []
    scripts = Path(sys.executable).resolve().parent
    os.environ["PATH"] = str(scripts) + os.pathsep + os.environ.get("PATH", "")
    lines.append(f"PATH prepended with venv Scripts for ninja.exe: {scripts}")
    try:
        from engine.cuda_jit import prepare

        lines.extend(prepare())
    except Exception as exc:
        lines.append(f"cuda_jit.prepare: FAIL {type(exc).__name__}: {exc}")
    try:
        from importlib.metadata import version

        lines.append(f"gsplat installed: version={version('gsplat')} (P0 record; not the Train gate)")
    except Exception as exc:
        lines.append(f"gsplat installed: no ({type(exc).__name__}: {exc})")
    return "\n".join(lines)


def main() -> int:
    chunks: list[str] = []
    chunks.append(
        section(
            "meta",
            f"written={datetime.now(timezone.utc).isoformat()}\n"
            f"executable={sys.executable}\n"
            f"version_string=2.0",
        )
    )
    chunks.append(section("python --version", run([sys.executable, "--version"])))
    chunks.append(section("nvidia-smi", run(["nvidia-smi"])))

    torch_body = []
    try:
        import torch

        torch_body.append(f"torch.__version__={torch.__version__}")
        torch_body.append(f"torch.version.cuda={torch.version.cuda}")
        torch_body.append(f"torch.cuda.is_available()={torch.cuda.is_available()}")
        if torch.cuda.is_available():
            torch_body.append(f"torch.cuda.device_count()={torch.cuda.device_count()}")
            torch_body.append(f"torch.cuda.get_device_name(0)={torch.cuda.get_device_name(0)}")
        else:
            torch_body.append("CUDA not available to this torch build.")
    except Exception as exc:
        torch_body.append(f"torch import FAIL {type(exc).__name__}: {exc}")
    chunks.append(section("torch", "\n".join(torch_body)))
    chunks.append(section("gsplat spike", gsplat_spike()))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(chunks), encoding="utf-8")
    print(f"wrote {OUT}", flush=True)
    print(OUT.read_text(encoding="utf-8"), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
