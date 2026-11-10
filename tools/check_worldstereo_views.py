"""CPU checks for joint WorldStereo. No GPU, no weights."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np

from engine.pano.scene_io import scene_dir
from engine.sfm.dataset_source import view_folders
from engine.sfm.known_pose import frame_name
from engine.splat.backend.litegs_backend import TrainView
from engine.worldstereo.adapter import TrajJob, list_trajs, select_poses
from engine.worldstereo.joint import joint_order, joint_plan, sample_poses, split_joint, write_joint
from engine.worldstereo.resume import joint_complete, pending_jobs, regen_plan, trajectory_complete
from engine.worldstereo.views import VIEWS, diffusion_num_frames
from engine.hires.pinhole import HIRES_SOURCE, hires_complete, reproject


def check(cond: bool, msg: str) -> None:
    if not cond:
        raise SystemExit(msg)


def check_list(scene: str) -> None:
    jobs = list_trajs(scene, order=[2, 0])
    names = [(job.rail_index, job.view, job.view_id) for job in jobs]
    check(names == [(2, "front", "view02_front"), (0, "front", "view00_front")], f"order {names}")
    check(jobs[0].output_rel.endswith("rail_02/front"), jobs[0].output_rel)
    check(jobs[0].sample_count == 21, f"sample count {jobs[0].sample_count}")


def check_folders(scene: str) -> None:
    root = scene_dir(scene)
    new = view_folders(root, 0)
    check(set(new) == {"front"}, f"new folders {set(new)}")
    check(new["front"].name == "front", str(new["front"]))
    legacy = view_folders(root, 3)
    check(list(legacy) == ["front"], f"legacy {legacy}")
    check(legacy["front"].name.startswith("03_"), legacy["front"].name)


def check_names() -> None:
    names = [frame_name(rail, "front", frame) for rail in range(4) for frame in (0, 9, 18, 44, 80)]
    check(len(names) == len(set(names)), "frame names collide")
    check(frame_name(1, "front", 7) == "rail_01_front_f007.png", frame_name(1, "front", 7))

    class Holder:
        name = "rail_00_front_f000.png"

    check(TrainView.rail.fget(Holder()) == "rail_00_front", TrainView.rail.fget(Holder()))
    Holder.name = "rail_03_f012.png"
    check(TrainView.rail.fget(Holder()) == "rail_03", TrainView.rail.fget(Holder()))


def _job(rail: int) -> TrajJob:
    return TrajJob(rail, "look_forward", "front", f"view{rail:02d}_front", "traj0", layout="separate")


def check_resume(scene: str) -> None:
    root = scene_dir(scene)
    jobs = [_job(2), _job(0)]
    done = root / jobs[0].output_rel
    (done / "keyframes").mkdir(parents=True)
    for i in range(21):
        (done / "keyframes" / f"{i:03d}.png").write_bytes(b"x")
    (done / "meta.json").write_text(json.dumps({"layout": "separate", "frames": 21}), encoding="utf-8")
    check(trajectory_complete(done), "complete folder not detected")
    pending = pending_jobs(scene, jobs)
    check([job.view_id for job in pending] == ["view00_front"], f"pending {pending}")
    (done / "STALE.txt").write_text("stale\n", encoding="utf-8")
    check(not trajectory_complete(done), "stale counted as done")
    (done / "STALE.txt").unlink()
    (done / "meta.json").write_text(json.dumps({"layout": "joint", "frames": 21}), encoding="utf-8")
    check(not trajectory_complete(done), "joint layout counted as a per-rail clip")
    (done / "meta.json").write_text(json.dumps({"layout": "separate", "frames": 21}), encoding="utf-8")
    rerun, previous, stale = regen_plan(jobs, 0)
    check(previous == 2, f"previous {previous}")
    check([job.view_id for job in rerun] == ["view00_front"], "regen set")
    check(stale == [], "rail 0 is last; nothing later")
    rerun, previous, stale = regen_plan(jobs, 2)
    check(previous is None and len(stale) == 1 and stale[0].rail_index == 0, "first rail regen")


def check_poses() -> None:
    poses = np.repeat(np.eye(4)[None], 81, axis=0)
    for index in range(81):
        poses[index, 0, 3] = float(index)
    job = TrajJob(0, "look_forward", "front", "view00_front", "traj0")
    model, kept = select_poses(poses, job)
    check(model.shape == (21, 4, 4) and kept.shape == (21, 4, 4), f"shape {model.shape}")
    check(int(model[0, 0, 3]) == 0 and int(model[-1, 0, 3]) == 80, "ends")
    check(diffusion_num_frames(21) == 81, "21 keyframes")
    check(diffusion_num_frames(40) == 157, "40 keyframes")
    check(diffusion_num_frames(10) == 37, "10 keyframes")
    check(VIEWS == ("front",), f"views {VIEWS}")


def check_joint(scene: str) -> None:
    expected = [0, 9, 18, 27, 36, 44, 53, 62, 71, 80]
    check(sample_poses(81, 10) == expected, f"poses {sample_poses(81, 10)}")
    four = joint_order([(0, 81), (1, 81), (2, 81), (3, 81)], 10)
    check(len(four) == 40, f"four rails {len(four)}")
    check(four[0].rail == 0 and four[0].pose == 0, "frame 0 is not the star")
    check([stop.pose for stop in four[:10]] == expected, "rail A outward")
    check([stop.pose for stop in four[10:20]] == list(reversed(expected)), "rail B inward")
    check(four[20].rail == 2 and four[20].pose == 0, "rail C does not start at the star")
    check([stop.pose for stop in four[30:40]] == list(reversed(expected)), "rail D inward")
    three = joint_order([(0, 81), (1, 81), (2, 81)], 10)
    check(three[-1].rail == 2 and three[-1].pose == 80, "odd last rail should run outward")
    one = joint_plan([(0, 81), (1, 81), (2, 81), (3, 81)], 10, 40)
    check(len(one) == 1 and len(one[0].frames) == 40, "40 should be one chunk")
    two = joint_plan([(0, 81), (1, 81), (2, 81), (3, 81)], 10, 21)
    check(len(two) == 2, f"limit 21 should be two chunks, got {len(two)}")
    check(len(two[0].frames) == 21 and len(two[1].frames) == 19, "chunk sizes")

    images = np.zeros((40, 8, 8, 3), np.uint8)
    colors = []
    masks = []
    w2cs = np.repeat(np.eye(4)[None], 40, axis=0)
    for index, stop in enumerate(four):
        images[index, :, :, 1] = index
        colors.append(np.full((8, 8, 3), index, np.uint8))
        masks.append(np.full((8, 8, 3), 255 if index % 2 == 0 else 0, np.uint8))
        w2cs[index, 0, 3] = float(stop.pose)
        w2cs[index, 1, 3] = float(stop.rail)
    eye = np.eye(3)
    write_joint(
        scene,
        [0, 1, 2, 3],
        four,
        images,
        colors,
        masks,
        w2cs,
        eye,
        frames_per_rail=10,
        seed=7,
        seconds=1.5,
        peak_gb=2.0,
    )
    check(joint_complete(scene), "joint_complete missed a finished sample")
    check(joint_complete(scene, 10), "10-frame joint rejected")
    check(not joint_complete(scene, 20), "10-frame joint counted as 20")
    longer = joint_plan([(0, 81), (1, 81), (2, 81), (3, 81)])
    check(len(longer) == 4 and all(len(chunk.frames) == 20 for chunk in longer), "20 per rail is four samples")
    check(longer[0].frames[0].rail == 0 and longer[0].frames[0].pose == 0, "first sample starts at the star")
    check(longer[0].frames[-1].rail == 0 and longer[0].frames[-1].pose == 80, "first sample ends at the far end")
    check(longer[1].frames[0].rail == 1 and longer[1].frames[0].pose == 80, "second sample starts at the far end")
    (scene_dir(scene) / "worldstereo" / "joint" / "STALE.txt").write_text("stale\n", encoding="utf-8")
    check(not joint_complete(scene), "stale joint counted as done")
    (scene_dir(scene) / "worldstereo" / "joint" / "STALE.txt").unlink()
    written = split_joint(scene, eye)
    check(len(written) == 4, f"split {written}")
    names = []
    for folder in written:
        frames = sorted((folder / "keyframes").glob("[0-9][0-9][0-9].png"))
        camera = json.loads((folder / "camera.json").read_text(encoding="utf-8"))
        check(len(camera["extrinsic"]) == len(frames), f"{folder.name} camera/frame mismatch")
        meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
        check(meta.get("layout") == "joint", "split layout")
        check(list(meta.get("positions") or []) == [int(path.stem) for path in frames], "positions")
        poses = [int(row[0][3]) for row in camera["extrinsic"]]
        check(poses == sorted(poses), f"{folder.name} camera is not pose order")
        check(not trajectory_complete(folder), "split joint folder looks like a 21-frame rail")
        rail = int(folder.parent.name.split("_")[1])
        names.extend(frame_name(rail, "front", int(path.stem)) for path in frames)
    check(len(names) == len(set(names)) == 40, "dataset names collide")


def check_shared_bank() -> None:
    import torch

    from engine.worldstereo.runtime import _remember_frames

    class _Bank:
        def __init__(self) -> None:
            self.ref_w2cs = torch.zeros((1, 4, 4))
            self.ref_Ks = torch.zeros((1, 3, 3))
            self.ref_frames = ["pano"]
            self.fnames = ["pano_bank/images/0000"]
            self.mem_size = 1
            self.device = torch.device("cpu")

    bank = _Bank()
    frames = [f"f{index}" for index in range(4)]
    poses = torch.eye(4).repeat(4, 1, 1)
    eyes = torch.eye(3).repeat(4, 1, 1)
    stored = _remember_frames(bank, frames, poses, eyes, "view00_front", "traj0", 0)
    check(stored == 4 and bank.ref_frames[0] == "pano", "front kept the pano bank")
    stored = _remember_frames(bank, frames, poses, eyes, "view01_back", "traj0", 1)
    check(stored == 3, "anchor frame was not stored twice")
    check(len(bank.ref_frames) == 8, f"bank is shared, size {len(bank.ref_frames)}")
    check(bank.fnames[-1].startswith("view01_back"), "later rail landed in the same bank")


def check_hires_source(scene: str) -> None:
    folder = scene_dir(scene) / "worldstereo" / "rails" / "rail_00" / "front"
    hires = folder / "hires"
    hires.mkdir(parents=True, exist_ok=True)
    (hires / "000.png").write_bytes(b"png")
    (folder / "gate_masks").mkdir(exist_ok=True)
    (folder / "gate_masks" / "000.png").write_bytes(b"png")
    check(not hires_complete(folder, 1), "old pinhole hires is not complete")
    (hires / "source.txt").write_text(HIRES_SOURCE + "\n", encoding="utf-8")
    check(hires_complete(folder, 1), "worldstereo hires is complete")
    check(not hires_complete(folder, 2), "a short hires folder is not complete")


def check_zbuffer() -> None:
    points = np.array([[0.0, 0.0, 2.0], [0.0, 0.0, 5.0]], np.float32)
    colors = np.array([[10, 20, 30], [200, 0, 0]], np.uint8)
    K = np.array([[100.0, 0.0, 4.0], [0.0, 100.0, 4.0], [0.0, 0.0, 1.0]])
    rgb, gate = reproject(points, colors, np.eye(4), K, height=8, width=8)
    check(bool(gate[4, 4]), "pixel missed")
    check(int(rgb[4, 4, 0]) == 10, f"far point won {rgb[4, 4].tolist()}")


def main() -> int:
    scene = "_cpu_ws_check"
    root = scene_dir(scene)
    if root.exists():
        shutil.rmtree(root)
    rails = root / "rails"
    rails.mkdir(parents=True)
    pose = json.dumps([np.eye(4).tolist()])
    for index in (0, 2):
        (rails / f"rail_{index:02d}.json").write_text(pose, encoding="utf-8")
    ws = root / "worldstereo" / "rails"
    (ws / "rail_00" / "front" / "keyframes").mkdir(parents=True)
    (ws / "03_look_at_target" / "keyframes").mkdir(parents=True)
    try:
        check_list(scene)
        check_folders(scene)
        check_names()
        check_resume(scene)
        check_poses()
        check_joint(scene)
        check_shared_bank()
        check_hires_source(scene)
        check_zbuffer()
    finally:
        shutil.rmtree(root, ignore_errors=True)
    print("CPU checks ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
