"""P0 dummy worker: ~20 stdout lines then exit. Proves log streaming and Stop."""

from __future__ import annotations

import sys
import time

from engine.job import job_progress


def main() -> int:
    print("STATUS // dummy worker start", flush=True)
    job_progress(0, 20, "dummy")
    for i in range(1, 21):
        print(f"dummy {i}/20", flush=True)
        job_progress(i, 20, f"dummy {i}/20")
        time.sleep(0.35)
    print("STATUS // dummy worker done", flush=True)
    job_progress(20, 20, "dummy done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
