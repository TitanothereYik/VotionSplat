"""Qt-safe LiteGS spike status. Reads ``docs/env_report.txt``. No torch."""

from __future__ import annotations

from pathlib import Path

from engine.job import REPO_ROOT

LITEGS_SPIKE_OK_LINE = "litegs raster 1-step: OK"
# Kept for the P0 report reader; the Train gate no longer uses it.
GSPLAT_SPIKE_OK_LINE = "gsplat 1-step dummy raster: OK"


def env_report_path() -> Path:
    return REPO_ROOT / "docs" / "env_report.txt"


def _report_text(path: Path | None) -> str:
    report = path or env_report_path()
    if not report.is_file():
        return ""
    return report.read_text(encoding="utf-8", errors="replace")


def litegs_spike_ok(path: Path | None = None) -> bool:
    """True only when ``tools/build_litegs.py`` recorded a real 1-step LiteGS raster.

    Train stays off if the report is missing or the line is not OK.
    COLMAP export / Open folder do not use this gate.
    """
    return LITEGS_SPIKE_OK_LINE in _report_text(path)


def gsplat_spike_ok(path: Path | None = None) -> bool:
    """P0 record only. Not a gate any more (trainer moved to LiteGS)."""
    return GSPLAT_SPIKE_OK_LINE in _report_text(path)


def litegs_spike_detail(path: Path | None = None) -> str:
    """Last non-empty line of the ``## litegs spike`` section, for the Splat hint."""
    text = _report_text(path)
    if "## litegs spike" not in text:
        return "LiteGS spike not run — Setup builds it (tools/run_litegs_spike.bat)."
    section = text.split("## litegs spike", 1)[1]
    section = section.split("\n## ", 1)[0]
    lines = [ln.strip() for ln in section.splitlines() if ln.strip()]
    for ln in lines:
        if ln.startswith("litegs raster 1-step:"):
            return ln
    return lines[-1] if lines else "LiteGS spike section empty."
