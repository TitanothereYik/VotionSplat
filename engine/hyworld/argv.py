"""Argv for HY-World workers. The Qt process may import this."""

from __future__ import annotations

from engine.job import REPO_ROOT

_PY = REPO_ROOT / ".venvs" / "hyworld" / "Scripts" / "python.exe"
_RUN = REPO_ROOT / "tools" / "hyworld" / "run_stage.py"
_EXPAND = [str(REPO_ROOT / ".venv" / "Scripts" / "python.exe"), "-m", "engine.workers.hy_expand"]


def stage_argv(stage: str, args: list[str], head: list[str] | None = None) -> list[str]:
    return [str(_PY), str(_RUN), "--stage", stage, *(head or []), "--", *args]


def export_argv(scene: str, color_space: str) -> list[str]:
    return [str(REPO_ROOT / ".venv" / "Scripts" / "python.exe"), "-m", "engine.workers.hy_export",
            "--scene", scene, "--color-space", color_space]


EXPAND_LEVELS = ("auto", "card", "blocks", "states", "controlnet")


def hires_argv(
    scene: str, *, only: str = "", resume: bool = True, level: str = "auto", scale: int = 2
) -> list[str]:
    from engine.hyworld.hires import PYTHON, SCALE, SCALES, choices

    scale = int(scale)
    if scale not in SCALES:
        raise ValueError(f"unknown hires scale {scale}")
    argv = [str(PYTHON), "-m", "engine.workers.hy_hires", "--scene", scene]
    if only:
        argv.extend(["--only", only])
    if not resume:
        argv.append("--restart")
    if level and level != "auto":
        if level not in choices(scale):
            raise ValueError(f"unknown {scale}x hires level {level}")
        argv.extend(["--level", level])
    if scale != SCALE:
        argv.extend(["--scale", str(scale)])
    return argv


def train_loop_argv(
    scene: str, *, cont: bool = False, steps: int = 8000, mesh: bool = False, difix: bool = False
) -> list[str]:
    argv = [str(REPO_ROOT / ".venv" / "Scripts" / "python.exe"), "-m", "engine.workers.hy_train_loop", "--scene", scene]
    if cont:
        argv.append("--continue")
    else:
        argv.extend(["--steps", str(int(steps))])
        if mesh:
            argv.append("--mesh")
        if difix:
            argv.append("--difix")
    return argv


def expand_argv(scene: str, *, only: str = "", resume: bool = True, level: str = "auto") -> list[str]:
    argv = [*_EXPAND, "--scene", scene]
    if only:
        argv.extend(["--only", only])
    if not resume:
        argv.append("--restart")
    if level and level != "auto":
        if level not in EXPAND_LEVELS:
            raise ValueError(f"unknown memory level {level}")
        argv.extend(["--level", level])
    return argv
