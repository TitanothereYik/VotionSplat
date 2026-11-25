"""Insert Votion header tags into a PLY whose vertex data was not colour-baked.

Refuses files that already carry postshot or votion comments. The header cannot prove
which export preset produced the numbers, so only use this on srgb exports
that were written without soften. HY scenes should be re-exported with the votion preset.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="src", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--raster", required=True, choices=("gsplat_antialiased", "gsplat_classic", "litegs_classic"))
    parser.add_argument("--near", type=float, required=True)
    parser.add_argument("--sh-degree", type=int, required=True)
    args = parser.parse_args()
    if not (0.0 < args.near < 1e6) or not (0 <= args.sh_degree <= 3):
        print("near plane must be positive and sh degree must be 0..3", file=sys.stderr)
        return 2
    blob = args.src.read_bytes()
    end = blob.find(b"end_header\n")
    if end < 0:
        print("end_header not found", file=sys.stderr)
        return 2
    header = blob[:end].decode("ascii", errors="replace")
    if "comment postshot." in header or "comment votion." in header:
        print("file already has postshot or votion tags; re-export it instead of tagging", file=sys.stderr)
        return 2
    if not header.startswith("ply\nformat binary_little_endian 1.0\n"):
        print("only binary little-endian PLY is supported", file=sys.stderr)
        return 2
    tags = (
        f"comment votion.version=1\n"
        f"comment votion.raster={args.raster}\n"
        f"comment votion.near_plane={float(args.near)!r}\n"
        f"comment votion.sh_degree={int(args.sh_degree)}\n"
    ).encode("ascii")
    insert = len(b"ply\nformat binary_little_endian 1.0\n")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(blob[:insert] + tags + blob[insert:])
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
