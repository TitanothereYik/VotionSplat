"""Train, stop, and optionally Difix3D+, one segment at a time. No torch."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
from pathlib import Path

from engine.hyworld.argv import stage_argv
from engine.hyworld.paths import gs_data, gs_result, hy_root
from engine.hyworld.train_run import (
    AUTOSAVE_SEC,
    DIFIX_LAMBDA,
    DIFIX_PY,
    DIFIX_VIEWS,
    build_merged,
    continue_info,
    frame_json,
    merged_dir,
    new_run,
    novel_poses,
    read_run,
    resume_meta,
    resume_pt,
    round_dir,
    state_json,
    trainer_flags,
    write_run,
)

_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


class Loop:
    def __init__(self) -> None:
        self._stop = False
        self._child: subprocess.Popen | None = None
        self._forward = False
        self._lock = threading.Lock()

    def _stdin(self) -> None:
        for line in sys.stdin:
            word = line.strip().split("\t", 1)[0] if line.strip() else ""
            if word == "STOP":
                self._stop = True
            with self._lock:
                proc = self._child
                forward = self._forward
            if word == "STOP":
                if forward and proc is not None and proc.stdin is not None:
                    try:
                        proc.stdin.write("STOP\n")
                        proc.stdin.flush()
                    except OSError:
                        pass
                elif proc is not None:
                    proc.terminate()
                    print("STATUS // stopped during Difix; Continue redoes this round", flush=True)
                return
            if forward and proc is not None and proc.stdin is not None and line.strip():
                try:
                    proc.stdin.write(line if line.endswith("\n") else line + "\n")
                    proc.stdin.flush()
                except OSError:
                    pass
        self._stop = True
        with self._lock:
            proc = self._child
            forward = self._forward
        if forward and proc is not None and proc.stdin is not None:
            try:
                proc.stdin.write("STOP\n")
                proc.stdin.flush()
            except OSError:
                pass

    def _run(self, argv: list[str], env: dict, forward: bool) -> int:
        proc = subprocess.Popen(
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=env,
            creationflags=_NO_WINDOW,
        )
        with self._lock:
            self._child = proc
            self._forward = forward
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="" if line.endswith("\n") else "\n", flush=True)
        code = proc.wait()
        with self._lock:
            self._child = None
        return int(code)

    def main(self, argv: list[str] | None = None) -> int:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--scene", required=True)
        parser.add_argument("--continue", dest="cont", action="store_true")
        parser.add_argument("--steps", type=int, default=8000)
        parser.add_argument("--mesh", action="store_true")
        parser.add_argument("--difix", action="store_true")
        parser.add_argument("--difix-views", type=int, default=DIFIX_VIEWS)
        parser.add_argument("--result-dir", default="")
        parser.add_argument("--stop-at", type=int, default=-1)
        args = parser.parse_args(argv)
        result = Path(args.result_dir) if args.result_dir else gs_result(args.scene)
        result.mkdir(parents=True, exist_ok=True)
        if args.cont:
            info = continue_info(result)
            if not info["ok"]:
                print(f"STATUS // {info['reason']}", flush=True)
                return 1
            run = read_run(result)
            if run is None:
                print("STATUS // No run to continue.", flush=True)
                return 1
        else:
            run = new_run(
                result,
                steps=max(100, int(args.steps)),
                mesh=bool(args.mesh),
                difix=bool(args.difix),
                views=max(1, int(args.difix_views)),
                data_dir=gs_data(args.scene),
            )
        threading.Thread(target=self._stdin, daemon=True).start()
        stop_once = int(args.stop_at) if int(args.stop_at) > 0 else None
        while True:
            if self._stop:
                return 0
            meta = {}
            meta_path = resume_meta(result, run)
            if meta_path.is_file():
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            start = int(meta.get("next_step", 0))
            if start >= int(run["max_steps"]):
                print("STATUS // training done", flush=True)
                return 0
            done = int(run.get("rounds_done") or 0)
            pending = [step for step in list(run.get("fix_steps") or [])[done:] if step >= start]
            natural = pending[0] if pending else -1
            halt_at = natural
            reason = "fix" if natural >= 0 else "stop"
            if stop_once is not None and stop_once > start and (natural < 0 or natural > stop_once):
                halt_at = stop_once
                reason = "stop"
                stop_once = None
            data_dir = merged_dir(result, run, done) if done > 0 else Path(run["base_data_dir"])
            novel = ""
            if reason == "fix":
                k = done + 1
                folder = round_dir(result, run, k)
                folder.mkdir(parents=True, exist_ok=True)
                cameras = json.loads((Path(data_dir) / "cameras.json").read_text(encoding="utf-8"))
                poses = novel_poses(
                    cameras,
                    Path(data_dir) / "images",
                    round_k=k,
                    count=int(run["views"]),
                    seed=k,
                )
                if not poses:
                    print("STATUS // no novel poses", flush=True)
                    return 1
                pose_path = folder / "poses.json"
                pose_path.write_text(json.dumps(poses) + "\n", encoding="utf-8")
                novel = str(pose_path)
            env = os.environ.copy()
            env.update(
                {
                    "PYTHONUNBUFFERED": "1",
                    "PYTHONIOENCODING": "utf-8",
                    "VOTION_RUN_ID": str(run["run_id"]),
                    "VOTION_RESUME_IN": str(resume_pt(result, run)) if start > 0 else "",
                    "VOTION_RESUME_OUT": str(resume_pt(result, run)),
                    "VOTION_STATE": str(state_json(result)),
                    "VOTION_FRAME_OUT": str(frame_json(result, run)),
                    "VOTION_HALT_AT": str(halt_at),
                    "VOTION_HALT_REASON": reason if halt_at >= 0 else "stop",
                    "VOTION_NOVEL": novel,
                    "VOTION_NOVEL_LAMBDA": str(DIFIX_LAMBDA if run.get("difix") else 1.0),
                    "VOTION_AUTOSAVE_SEC": str(AUTOSAVE_SEC),
                }
            )
            frame = frame_json(result, run)
            if frame.is_file():
                env["VOTION_FRAME"] = str(frame)
            else:
                env.pop("VOTION_FRAME", None)
            if start > 0 and not resume_pt(result, run).is_file():
                print("STATUS // resume file missing", flush=True)
                return 1
            flags = trainer_flags(str(data_dir), str(result), int(run["max_steps"]), bool(run["mesh"]))
            code = self._run(stage_argv("world_gs_trainer", flags), env, forward=True)
            if self._stop:
                return 0
            state = {}
            state_path = state_json(result)
            if state_path.is_file():
                try:
                    state = json.loads(state_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    state = {}
            why = str(state.get("reason", ""))
            if why == "done":
                print("STATUS // training done", flush=True)
                return 0
            if why == "stop":
                return 0
            if why == "fix":
                print("SPLAT\tstate\tDIFIX", flush=True)
                difix_env = os.environ.copy()
                difix_env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])
                difix_env["PYTHONIOENCODING"] = "utf-8"
                difix_env["PYTHONUNBUFFERED"] = "1"
                code = self._run(
                    [str(DIFIX_PY), "-m", "engine.workers.hy_difix", "--round-dir", str(round_dir(result, run, done + 1))],
                    difix_env,
                    forward=False,
                )
                if self._stop:
                    return 0
                if code != 0:
                    print(f"STATUS // difix exited {code}", flush=True)
                    return 1
                k = done + 1
                build_merged(
                    run["base_data_dir"],
                    merged_dir(result, run, k),
                    [round_dir(result, run, i) for i in range(1, k + 1)],
                    hy_root(args.scene) / "meta_info.json",
                )
                run["rounds_done"] = k
                write_run(result, run)
                continue
            print(f"STATUS // trainer exited {code} without a state", flush=True)
            return 1


def main(argv: list[str] | None = None) -> int:
    return Loop().main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
