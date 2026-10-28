"""CLI for engine.cuda_jit file patches. Safe to run from launch.bat (no torch)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.cuda_jit import main

if __name__ == "__main__":
    raise SystemExit(main())
