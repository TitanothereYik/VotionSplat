"""Keep one planned trajectory for the no-toolkit scene."""

from __future__ import annotations

import sys
from pathlib import Path

from engine.hyworld.paths import list_trajectories, write_selection

scene = sys.argv[1]
ids = list_trajectories(scene)
print("planned", len(ids), flush=True)
keep = ["view0/traj0"] if "view0/traj0" in ids else (ids[:1])
if not keep:
    print("no trajectories", flush=True)
    raise SystemExit(1)
write_selection(scene, keep)
out = Path(__file__).resolve().parents[2] / ".venvs" / "notk_only.txt"
out.write_text(keep[0], encoding="utf-8")
print("keep", keep[0], flush=True)
