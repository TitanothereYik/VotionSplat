"""Supervisor for ``engine.workers.splat`` headless CLI runs.

Why this exists
----------------
The vendored LiteGS CUDA binning kernel (litegs/submodules/gaussian_raster/binning.cu)
has one known pathological case that is *not* a crash: for a specific
view/Gaussian pairing (observed so far only in the ~20k-22k step / near-budget-cap
range on a dense HiRes retrain) the tile-binning kernel becomes catastrophically
slow -- GPU pinned at 100%, one CPU thread fully busy, but zero forward progress
-- rather than raising a ``RuntimeError`` the trainer's own try/except can catch
(see the comment at the ``except RuntimeError`` block in ``engine/splat/trainer.py``,
which *does* catch the sibling case where the kernel actually errors out).

This script wraps a training/retrain run, watches stdout for silence, and if the
child goes quiet for longer than ``--stall-timeout`` seconds while still alive
(a hang, not a crash), kills the whole process tree and relaunches with
``--mode continue`` from the last checkpoint (``CKPT_EVERY = 100`` steps, so at
most ~100 steps of rework). Each restart-after-stall also bumps
``VOTION_EPOCH_SEED_SALT`` so the resumed run's epoch permutation differs from
the one that just got stuck -- resuming a checkpoint otherwise reproduces the
*identical* deterministic view order and would walk straight back into the same
pathological view/Gaussian pairing at the same relative step.

Usage (mirrors ``engine.workers.splat``, plus watchdog-only flags)::

    python -m tools.train_watchdog --scene alpine_01 --strategy mcmc_compact \\
        --mode retrain --max-steps 30000 --holdout --stall-timeout 180 \\
        --max-restarts 20 --log-file C:/path/to/run.log

Exit codes: 0 done (step >= max-steps, meta.json done=true), 1 restart budget
exhausted, 2 immediate/repeated failure before any progress, 130 interrupted.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _scene_dir(scene: str) -> Path:
    name = (scene or "alpine_01").strip() or "alpine_01"
    for ch in '<>:"/\\|?*':
        name = name.replace(ch, "_")
    path = REPO_ROOT / "outputs" / name
    return path


def _meta_path(scene: str) -> Path:
    return _scene_dir(scene) / "checkpoints" / "splat_meta.json"


def _read_meta(scene: str) -> dict:
    p = _meta_path(scene)
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _now() -> str:
    return time.strftime("%H:%M:%S")


def _kill_tree(pid: int) -> None:
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                capture_output=True,
                timeout=20,
            )
        else:
            import signal

            os.killpg(os.getpgid(pid), signal.SIGKILL)
    except Exception as exc:
        print(f"[watchdog {_now()}] kill_tree({pid}) failed: {exc}", flush=True)


def run_once(cmd: list[str], env: dict, log_file, stall_timeout: float) -> tuple[int | None, str]:
    """Launch one attempt. Returns (returncode_or_None, reason).

    reason is one of: "exit" (process exited on its own), "stall" (we killed it).
    """
    proc = subprocess.Popen(
        cmd,
        cwd=str(REPO_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    last_output = [time.monotonic()]
    stopped = threading.Event()

    def _reader() -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            last_output[0] = time.monotonic()
            sys.stdout.write(line)
            sys.stdout.flush()
            log_file.write(line)
            log_file.flush()
            if stopped.is_set():
                break

    t = threading.Thread(target=_reader, daemon=True)
    t.start()

    reason = "exit"
    try:
        while True:
            rc = proc.poll()
            if rc is not None:
                break
            idle = time.monotonic() - last_output[0]
            if idle > stall_timeout:
                print(
                    f"[watchdog {_now()}] STALL: no output for {idle:.0f}s "
                    f"(> {stall_timeout:.0f}s threshold) -- killing pid {proc.pid} and its children",
                    flush=True,
                )
                log_file.write(f"[watchdog {_now()}] STALL detected, killing pid {proc.pid}\n")
                log_file.flush()
                _kill_tree(proc.pid)
                reason = "stall"
                break
            time.sleep(2.0)
    except KeyboardInterrupt:
        print(f"[watchdog {_now()}] interrupted -- killing pid {proc.pid}", flush=True)
        _kill_tree(proc.pid)
        stopped.set()
        raise
    stopped.set()
    try:
        rc = proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        _kill_tree(proc.pid)
        rc = proc.wait(timeout=15)
    return rc, reason


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Watchdog wrapper for engine.workers.splat")
    ap.add_argument("--scene", required=True)
    ap.add_argument("--strategy", default="litegs")
    ap.add_argument(
        "--mode", default="retrain", choices=["train", "continue", "retrain"], help="mode for the FIRST attempt only"
    )
    ap.add_argument("--max-steps", type=int, required=True)
    ap.add_argument("--budget", type=int, default=None)
    ap.add_argument("--res-scale", type=float, default=None)
    ap.add_argument("--holdout", action="store_true")
    ap.add_argument("--stall-timeout", type=float, default=180.0, help="seconds of silent-but-alive before restart")
    ap.add_argument("--max-restarts", type=int, default=20)
    ap.add_argument("--log-file", required=True)
    args = ap.parse_args(argv)

    log_path = Path(args.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(log_path, "a", encoding="utf-8", errors="replace")

    def base_cmd(mode: str) -> list[str]:
        cmd = [
            sys.executable,
            "-m",
            "engine.workers.splat",
            "--scene",
            args.scene,
            "--strategy",
            args.strategy,
            "--mode",
            mode,
            "--max-steps",
            str(args.max_steps),
            "--headless",
        ]
        if args.budget is not None:
            cmd += ["--budget", str(args.budget)]
        if args.res_scale is not None:
            cmd += ["--res-scale", str(args.res_scale)]
        if args.holdout:
            cmd.append("--holdout")
        return cmd

    restarts = 0
    stall_restarts = 0
    mode = args.mode
    t_start = time.monotonic()
    print(
        f"[watchdog {_now()}] starting: scene={args.scene} strategy={args.strategy} mode={mode} "
        f"max_steps={args.max_steps} stall_timeout={args.stall_timeout:.0f}s max_restarts={args.max_restarts} "
        f"log={log_path}",
        flush=True,
    )
    log_file.write(f"\n===== watchdog session start {_now()} mode={mode} =====\n")
    log_file.flush()

    last_seen_step = -1
    no_progress_streak = 0

    while True:
        env = dict(os.environ)
        env["VOTION_EPOCH_SEED_SALT"] = str(stall_restarts)
        cmd = base_cmd(mode)
        print(f"[watchdog {_now()}] attempt {restarts + 1}: {' '.join(cmd)} (salt={stall_restarts})", flush=True)
        log_file.write(f"[watchdog {_now()}] launch attempt {restarts + 1} mode={mode} salt={stall_restarts}\n")
        log_file.flush()

        rc, reason = run_once(cmd, env, log_file, args.stall_timeout)
        elapsed = time.monotonic() - t_start
        meta = _read_meta(args.scene)
        step = int(meta.get("step") or -1)
        done = bool(meta.get("done"))
        print(
            f"[watchdog {_now()}] attempt {restarts + 1} ended: reason={reason} rc={rc} "
            f"checkpoint_step={step} done={done} elapsed={elapsed:.0f}s",
            flush=True,
        )
        log_file.write(
            f"[watchdog {_now()}] attempt ended reason={reason} rc={rc} step={step} done={done}\n"
        )
        log_file.flush()

        if done and step >= args.max_steps:
            print(f"[watchdog {_now()}] DONE at step {step} -- exiting 0", flush=True)
            log_file.write(f"[watchdog {_now()}] DONE at step {step}\n")
            log_file.close()
            return 0

        if rc == 0 and not done:
            # Clean exit (STOP/EOF) before reaching max_steps -- headless runs
            # with no controller shouldn't hit this, but don't loop forever if
            # something external asked it to stop.
            print(f"[watchdog {_now()}] process exited 0 but not done (step={step}) -- treating as stop, exiting", flush=True)
            log_file.close()
            return 0

        # Track whether we're actually making progress across restarts.
        if step > last_seen_step:
            last_seen_step = step
            no_progress_streak = 0
        else:
            no_progress_streak += 1

        if reason == "stall":
            stall_restarts += 1
        restarts += 1

        if restarts > args.max_restarts:
            print(
                f"[watchdog {_now()}] restart budget exhausted ({restarts - 1}/{args.max_restarts}) "
                f"at step {step} -- giving up",
                flush=True,
            )
            log_file.write(f"[watchdog {_now()}] restart budget exhausted at step {step}\n")
            log_file.close()
            return 1

        if no_progress_streak >= 3:
            print(
                f"[watchdog {_now()}] no checkpoint progress across {no_progress_streak} consecutive attempts "
                f"(stuck at step {step}) -- giving up rather than spin forever",
                flush=True,
            )
            log_file.write(f"[watchdog {_now()}] no progress across {no_progress_streak} attempts, stuck at {step}\n")
            log_file.close()
            return 2

        mode = "continue"
        backoff = 3.0 if reason == "stall" else 5.0
        print(f"[watchdog {_now()}] restarting in {backoff:.0f}s as --mode continue (attempt {restarts + 1})", flush=True)
        time.sleep(backoff)


if __name__ == "__main__":
    sys.exit(main())
