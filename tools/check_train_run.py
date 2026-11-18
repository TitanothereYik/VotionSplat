"""CPU checks for the train loop bookkeeping. No GPU."""

from __future__ import annotations

import json
import os
import struct
import tempfile
import zlib
from pathlib import Path

from engine.hyworld.train_run import (
    build_merged,
    continue_info,
    fix_steps,
    new_run,
    novel_poses,
    resume_meta,
    state_json,
    trainer_flags,
    write_run,
)
import numpy as np


def check(cond: bool, msg: str) -> None:
    if not cond:
        raise SystemExit(msg)


def tiny_png(path: Path, width: int = 8, height: int = 4) -> None:
    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    raw = b"".join(b"\x00" + b"\x00\x00\x00" * width for _ in range(height))
    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(raw))
    png += chunk(b"IEND", b"")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(png)


def _c2w(centre, yaw: float = 0.0) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    # OpenCV camera: +Z forward, +X right, +Y down. Yaw about Y.
    forward = np.array([s, 0.0, c])
    right = np.array([c, 0.0, -s])
    down = np.array([0.0, 1.0, 0.0])
    mat = np.eye(4)
    mat[:3, 0] = right
    mat[:3, 1] = down
    mat[:3, 2] = forward
    mat[:3, 3] = centre
    return mat


def check_flags() -> None:
    got = trainer_flags("D:/data/gs_data", "D:/data/gs_result", 8000, False)
    count = 8000
    want = [
        "default",
        "--data-dir", "D:/data/gs_data",
        "--result-dir", "D:/data/gs_result",
        "--max-steps", "8000", "--save-steps", "8000", "--eval-steps", "8000", "--ply-steps", "8000",
        "--save-ply", "--disable-video", "--disable-viewer", "--use-scale-regularization", "--antialiased",
        "--depth-loss", "--normal-loss", "--sky-depth-from-pcd", "--use-mask-gaussian",
        "--mask-export-stochastic", "--no-mask-export-anchor-protection", "--use-anchor-protection",
        "--strategy.refine-start-iter", str(max(1, count // 10)),
        "--strategy.refine-stop-iter", str(max(1, count // 2)),
        "--strategy.refine-every", str(max(1, count // 15)),
        "--strategy.refine-scale2d-stop-iter", str(max(1, count // 2)),
        "--strategy.reset-every", "99990",
        "--strategy.grow-grad2d", "0.0001",
        "--strategy.prune-scale3d", "0.1",
    ]
    check(got == want, f"flags {got}")
    check(trainer_flags("a", "b", 100, True)[-1] == "--export_mesh", "mesh flag")
    check(fix_steps(8000) == [1600, 2400, 3200, 4400, 5600], str(fix_steps(8000)))
    check(fix_steps(300) == [60, 90, 120, 165, 210], str(fix_steps(300)))


def check_poses_and_merge() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        images = root / "images"
        cameras = {}
        specs = {
            "view0-traj0_000000": (np.array([0.0, 0.0, 0.0]), 0.0),
            "view0-traj0_000001": (np.array([0.2, 0.0, 0.0]), 0.0),
            "view1-traj0_000000": (np.array([0.0, 0.0, 4.0]), 0.0),
            "view1-traj0_000001": (np.array([0.2, 0.0, 4.0]), 0.0),
            "view2-traj0_000000": (np.array([0.0, 0.0, 8.0]), 0.0),
            "panorama_L00_A0000": (np.array([9.0, 0.0, 0.0]), 1.0),
        }
        for name, (centre, yaw) in specs.items():
            c2w = _c2w(centre, yaw)
            cameras[name] = {"extrinsic": np.linalg.inv(c2w).tolist(), "intrinsic": [[1, 0, 4], [0, 1, 2], [0, 0, 1]]}
            tiny_png(images / f"{name}.png")
        cameras["width"] = 8
        cameras["height"] = 4
        poses = novel_poses(cameras, images, round_k=1, count=2, seed=1)
        check(len(poses) == 2, f"pose count {len(poses)}")
        check(all(p["name"].startswith("zdifix_r01_") for p in poses), "names")
        check(all(not p["ref"].endswith("panorama_L00_A0000.png") for p in poses), "pano used")
        for pose in poses:
            R = np.asarray(pose["c2w"])[:3, :3]
            check(np.allclose(R.T @ R, np.eye(3), atol=1e-6), "not orthonormal")
            check(abs(float(np.linalg.det(R)) - 1.0) < 1e-6, "det")
            a = np.linalg.inv(np.asarray(cameras[pose["src"]]["extrinsic"]))[:3, 3]
            b = np.linalg.inv(np.asarray(cameras[pose["dst"]]["extrinsic"]))[:3, 3]
            got = np.asarray(pose["c2w"])[:3, 3]
            check(np.allclose(got, (1 - 0.25) * a + 0.25 * b, atol=1e-6), f"lerp {got}")
            check(pose["ref"].endswith(pose["src"] + ".png"), "ref")

        (root / "cameras.json").write_text(json.dumps(cameras), encoding="utf-8")
        (root / "depths" / "view0-traj0_000000.png").parent.mkdir(parents=True)
        tiny_png(root / "depths" / "view0-traj0_000000.png")
        original = (root / "cameras.json").read_bytes()
        rnd = root / "round1"
        (rnd / "fixed").mkdir(parents=True)
        for pose in poses:
            tiny_png(rnd / "fixed" / f"{pose['name']}.png", 8, 4)
        (rnd / "poses.json").write_text(json.dumps(poses), encoding="utf-8")
        meta = root / "scene_meta.json"
        meta.write_text('{"scene_type": "outdoor"}\n', encoding="utf-8")
        out = root / "difix_data" / "run_r1"
        build_merged(root, out, [rnd], meta)
        check((root / "cameras.json").read_bytes() == original, "base cameras changed")
        merged = json.loads((out / "cameras.json").read_text(encoding="utf-8"))
        check(all(p["name"] in merged for p in poses), "novel keys missing")
        check("view0-traj0_000000" in merged, "base key missing")
        a = root / "images" / "view0-traj0_000000.png"
        b = out / "images" / "view0-traj0_000000.png"
        same = os.stat(a).st_ino == os.stat(b).st_ino or a.stat().st_size == b.stat().st_size
        check(same, "image was not linked or copied")
        check((out.parent / "meta_info.json").is_file(), "meta missing")


def check_continue() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        data = root / "gs_data"
        (data / "images").mkdir(parents=True)
        (data / "cameras.json").write_text("{}", encoding="utf-8")
        result = root / "result"
        check(continue_info(result)["reason"] == "No run to continue.", "empty")
        run = new_run(result, steps=300, mesh=False, difix=True, views=6, data_dir=data)
        check(continue_info(result)["reason"] == "Nothing saved yet.", continue_info(result)["reason"])
        resume_meta(result, run).write_text(json.dumps({"next_step": 40, "reason": "autosave"}) + "\n", encoding="utf-8")
        info = continue_info(result)
        check(info["ok"] and info["next_step"] == 40 and info["difix"], str(info))
        state_json(result).write_text(json.dumps({"reason": "done", "next_step": 300}) + "\n", encoding="utf-8")
        check("finished" in continue_info(result)["reason"], continue_info(result)["reason"])
        state_json(result).unlink()
        run["data_stamp"] = {"size": 1, "mtime_ns": 1}
        write_run(result, run)
        check("Dataset changed" in continue_info(result)["reason"], continue_info(result)["reason"])


def main() -> int:
    check_flags()
    check_poses_and_merge()
    check_continue()
    print("check_train_run ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
