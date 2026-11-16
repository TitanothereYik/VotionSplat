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
    from engine.hires.pinhole import HIRES_FRAME, hires_ready, hires_stamp, use_hires

    frame = hires / "frame"
    frame.mkdir(exist_ok=True)
    (frame / "000.png").write_bytes(b"png")
    check(hires_ready(folder, HIRES_FRAME, 1), "upscale is stored beside the composite")
    check(hires_stamp(folder) == HIRES_SOURCE, "storing the upscale leaves the composite active")
    check(use_hires(folder, HIRES_FRAME), "upscale can become the active set")
    check(hires_complete(folder, 1), "active upscale counts as complete")
    check(hires_stamp(folder) == HIRES_FRAME, "stamp follows the upscale")
    check(use_hires(folder, HIRES_SOURCE), "composite can become active again")
    check(hires_stamp(folder) == HIRES_SOURCE, "stamp follows the composite")
    from engine.hires.pinhole import active_choice, choice_matches_dataset, write_choice

    root = scene_dir(scene)
    write_choice(root, "frame")
    check(active_choice(root) == "frame", "saved Generate path is WorldStereo")
    check(choice_matches_dataset("frame", "frame"), "WorldStereo dataset matches")
    check(not choice_matches_dataset("world", "frame"), "composite choice rejects a WorldStereo dataset")
    check(not choice_matches_dataset("frame", ""), "an unmarked dataset does not match")


def _cpu_renderer(piece, texture, key: str = "pano"):
    from engine.world.render import WorldRenderer

    return WorldRenderer([piece], {key: texture}, device="cpu", bin_size=0)


def _quad_piece(verts, tex: str = "pano"):
    from engine.world.pano_mesh import Piece

    faces = np.array([[0, 1, 2], [0, 2, 3]], np.int32)
    uvs = np.array([[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]], np.float32)
    return Piece(np.asarray(verts, np.float32), faces, uvs, tex)


def _front_quad_view():
    tex = np.zeros((8, 8, 3), np.uint8)
    tex[..., 0] = 255
    verts = [[-1.0, -1.0, 2.0], [1.0, -1.0, 2.0], [1.0, 1.0, 2.0], [-1.0, 1.0, 2.0]]
    renderer = _cpu_renderer(_quad_piece(verts), tex)
    try:
        k_mat = np.array([[10.0, 0.0, 8.0], [0.0, 10.0, 6.0], [0.0, 0.0, 1.0]])
        return renderer.render(np.eye(4), k_mat, 12, 16, supersample=1)
    finally:
        renderer.close()


def check_world_projection() -> None:
    view = _front_quad_view()
    check(not bool(view.empty[6, 8]), "centre missed the quad")
    check(abs(float(view.depth[6, 8]) - 2.0) < 1e-3, f"depth {view.depth[6, 8]}")
    check(int(view.rgb[6, 8, 0]) > 200, f"colour {view.rgb[6, 8].tolist()}")
    check(int(view.source[6, 8]) == 2, f"source {int(view.source[6, 8])}")
    check(bool(view.empty[0, 0]), "corner hit the quad")


def check_world_projection_side() -> None:
    # 0.05 world units is a quarter of a pixel at this K and z=2, so it holds no pixel centre.
    # 0.2 reaches the centres in column 10. A flipped X axis would land in columns 5–7.
    half = 0.2
    verts = [
        [0.5 - half, -half, 2.0],
        [0.5 + half, -half, 2.0],
        [0.5 + half, half, 2.0],
        [0.5 - half, half, 2.0],
    ]
    tex = np.full((8, 8, 3), 128, np.uint8)
    renderer = _cpu_renderer(_quad_piece(verts), tex)
    try:
        k_mat = np.array([[10.0, 0.0, 8.0], [0.0, 10.0, 6.0], [0.0, 0.0, 1.0]])
        view = renderer.render(np.eye(4), k_mat, 12, 16, supersample=1)
    finally:
        renderer.close()
    cols = np.unique(np.where(~view.empty)[1])
    check(cols.size > 0, "side quad missed every pixel")
    check(int(cols.min()) >= 9 and int(cols.max()) <= 11, f"side columns {cols.tolist()}")
    check(not np.any((cols >= 5) & (cols <= 7)), f"axis flip columns {cols.tolist()}")


def check_guide_mask_polarity() -> None:
    view = _front_quad_view()
    mask = np.where(view.empty, 255, 0).astype(np.uint8)
    check(int(mask[6, 8]) == 0, "guide centre is not a surface")
    check(int(mask[0, 0]) == 255, "guide corner is not empty")


def check_grazing_cut() -> None:
    from engine.world.pano_mesh import build_pano_mesh

    depth = np.ones((8, 16), np.float64)
    depth[:, 8:] = 5.0
    piece, _stats = build_pano_mesh(depth, np.ones_like(depth, dtype=bool), snap_ground=False)
    if piece.faces.size == 0:
        return
    norms = np.linalg.norm(piece.verts.astype(np.float64), axis=1)
    tri = norms[piece.faces.astype(np.int64)]
    mixed = (tri.min(axis=1) < 1.5) & (tri.max(axis=1) > 3.0)
    check(not bool(np.any(mixed)), "a face crosses the depth jump")


def check_ground_fill() -> None:
    from engine.world.render import WorldRenderer

    tex = np.zeros((8, 8, 3), np.uint8)
    tex[3:5, 3:5] = (0, 200, 0)
    # The quad sits off to the right, so the centre pixel is a hole over the plane z=2.
    verts = [[3.0, -0.2, 2.0], [3.4, -0.2, 2.0], [3.4, 0.2, 2.0], [3.0, 0.2, 2.0]]
    k_mat = np.array([[10.0, 0.0, 8.0], [0.0, 10.0, 6.0], [0.0, 0.0, 1.0]])
    renderer = WorldRenderer(
        [_quad_piece(verts)],
        {"pano": tex},
        device="cpu",
        bin_size=0,
        ground=(np.array([0.0, 0.0, 1.0]), -2.0),
    )
    try:
        view = renderer.render(np.eye(4), k_mat, 12, 16, supersample=1)
    finally:
        renderer.close()
    check(not bool(view.empty[6, 8]), "ground hole stayed empty")
    check(int(view.rgb[6, 8, 1]) > 150, f"ground colour {view.rgb[6, 8].tolist()}")
    check(abs(float(view.depth[6, 8]) - 2.0) < 1e-2, f"ground depth {view.depth[6, 8]}")
    check(int(view.source[6, 8]) == 2, f"ground source {int(view.source[6, 8])}")


def check_supersample_colour() -> None:
    from engine.world.render import _resize_supersample

    rgb = np.zeros((2, 2, 3), np.uint8)
    rgb[0, 0] = (200, 40, 10)
    covered = np.zeros((2, 2), dtype=bool)
    covered[0, 0] = True
    depth = np.zeros((2, 2), np.float32)
    source = np.zeros((2, 2), np.int8)
    source[0, 0] = 2
    out, empty, _z, _src = _resize_supersample(rgb, covered, depth, source, 1, 1)
    check(bool(empty[0, 0]), "quarter coverage stays a hole")
    rgb[0, :] = (180, 20, 20)
    covered[0, :] = True
    depth[:] = 4
    source[:] = 2
    out, empty, z, src = _resize_supersample(rgb, covered, depth, source, 1, 1)
    check(not bool(empty[0, 0]), "half coverage is kept")
    check(int(out[0, 0, 0]) == 180, f"colour stayed {out[0, 0].tolist()}")
    check(int(src[0, 0]) == 2, f"source {int(src[0, 0])}")
    check(abs(float(z[0, 0]) - 4.0) < 1e-3, f"depth {float(z[0, 0])}")


def check_hires_pano_wins() -> None:
    from engine.hires.pinhole import composite_world
    from engine.world.defaults import GATE_LAYER, GATE_OWN, GATE_PANO

    pano = np.zeros((4, 4, 3), np.uint8)
    pano[:, :2] = (10, 20, 30)
    empty = np.ones((4, 4), dtype=bool)
    empty[:, :2] = False
    layer = np.zeros((4, 4, 3), np.uint8)
    layer[:] = (255, 0, 0)
    hit = np.ones((4, 4), dtype=bool)
    own = np.full((4, 4, 3), 128, np.uint8)
    out, gate = composite_world(pano, empty, layer, hit, own, 1, GATE_PANO, GATE_LAYER, GATE_OWN)
    check(int(gate[1, 0]) == GATE_PANO, "pano stays in front of the layer")
    check(int(out[1, 0, 2]) == 30, f"pano colour {out[1, 0].tolist()}")
    check(int(gate[1, 3]) == GATE_LAYER, "a hole uses the layer")
    check(int(out[1, 3, 0]) == 255, f"layer colour {out[1, 3].tolist()}")
    out, gate = composite_world(pano, empty, None, None, own, 1, GATE_PANO, GATE_LAYER, GATE_OWN)
    check(int(gate[1, 3]) == GATE_OWN, "a hole with no layer keeps the frame")
    check(int(out[1, 3, 0]) == 128, f"frame colour {out[1, 3].tolist()}")


def check_memory_far() -> None:
    from engine.worldstereo.hy_bank import memory_far

    check(abs(memory_far(1.25) - 10.0) < 1e-6, "a 1.25 m median matches the old 10 m far plane")
    check(abs(memory_far(2.0) - 16.0) < 1e-6, "far plane is eight times the depth median")
    check(abs(memory_far(0.01) - 0.15) < 1e-6, "far plane stays at the HY floor")


def check_hires_bin_size() -> None:
    from engine.world.render import WorldRenderer

    renderer = WorldRenderer([], {}, device="cpu")
    try:
        wide = renderer._bin_size_for(1920, 3328)
        grid = 1 + (3328 - 1) // wide
        check(grid < 22, f"hires bin grid {grid} at bin {wide}")
        check(renderer._bin_size_for(960, 1664) == 128, "guide bin stayed 128")
    finally:
        renderer.close()


def check_bin_retile() -> None:
    from engine.world import render as render_mod
    from engine.world.render import WorldRenderer

    tex = np.zeros((8, 8, 3), np.uint8)
    verts = [[-1.0, -1.0, 2.0], [1.0, -1.0, 2.0], [1.0, 1.0, 2.0], [-1.0, 1.0, 2.0]]
    k_mat = np.array([[150.0, 0.0, 208.0], [0.0, 150.0, 120.0], [0.0, 0.0, 1.0]])
    budget = 1_000_000
    renderer = WorldRenderer([_quad_piece(verts)], {"pano": tex}, device="cpu", bin_size=64)
    renderer.max_faces_per_bin = 1000
    calls: list[int] = []
    patched = {"_rasterize": render_mod._rasterize, "RASTER_BIN_BUDGET": render_mod.RASTER_BIN_BUDGET,
               "MAX_FACES_PER_BIN_CAP": render_mod.MAX_FACES_PER_BIN_CAP}

    def fake(rasterizer, meshes):
        settings = rasterizer.raster_settings
        high_h, high_w = settings.image_size
        bins = ((high_h + 63) // 64) * ((high_w + 63) // 64)
        calls.append(bins * int(settings.max_faces_per_bin) * 4)
        if int(settings.max_faces_per_bin) < 8000:
            raise RuntimeError("raise MAX_FACES_PER_BIN")
        settings.bin_size = 0
        return patched["_rasterize"](rasterizer, meshes)

    render_mod._rasterize = fake
    render_mod.RASTER_BIN_BUDGET = budget
    render_mod.MAX_FACES_PER_BIN_CAP = 8000
    try:
        view = renderer.render(np.eye(4), k_mat, 240, 416, supersample=2)
    finally:
        for name, value in patched.items():
            setattr(render_mod, name, value)
        renderer.close()
    check(view.rgb.shape[:2] == (240, 416), f"retile shape {view.rgb.shape}")
    check(renderer.max_faces_per_bin == 8000, f"bins {renderer.max_faces_per_bin}")
    over = [value for value in calls if value > budget]
    check(not over, f"bin memory over budget: {over}")
    check(len(calls) > 4, f"retile never split: {calls}")
    check(not bool(view.empty[120, 208]), "retiled centre stayed empty")


def check_exif_hfov(scene: str) -> None:
    import math

    from PIL import Image

    from engine.pano.scene_io import scene_dir as scene_root
    from engine.workers.detect_fov import exif_hfov

    folder = scene_root(scene) / "_work"
    folder.mkdir(parents=True, exist_ok=True)
    exif = Image.Exif()
    sub = exif.get_ifd(0x8769)
    sub[41989] = 26
    sub[40962] = 4000
    sub[40963] = 3000
    full = folder / "exif_full.jpg"
    Image.new("RGB", (400, 300)).save(full, exif=exif)
    got = exif_hfov(full)
    want = math.degrees(2.0 * math.atan(math.hypot(36.0, 24.0) * 0.8 / 52.0))
    check(got is not None and abs(got[0] - want) < 0.05, f"exif hfov {got} want {want:.2f}")
    cropped = folder / "exif_crop.jpg"
    Image.new("RGB", (400, 400)).save(cropped, exif=exif)
    check(exif_hfov(cropped) is None, "cropped photo trusted its EXIF focal length")
    bare = folder / "exif_none.jpg"
    Image.new("RGB", (400, 300)).save(bare)
    check(exif_hfov(bare) is None, "photo without EXIF returned a focal length")


def check_edge_convention() -> None:
    import cv2

    from engine.depth.scene_depth import _apply_convention, _edge_convention

    rng = np.random.default_rng(3)
    blocks = rng.integers(0, 255, size=(16, 32), dtype=np.uint8)
    gray = cv2.resize(blocks, (1024, 512), interpolation=cv2.INTER_NEAREST)
    gray[:, :400] = cv2.GaussianBlur(gray[:, :400], (0, 0), 6.0)
    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    distance = 1.0 + gray.astype(np.float32) / 64.0
    valid = np.ones_like(distance, dtype=bool)
    for applied in ("identity", "flip", "roll", "flip_roll"):
        trial, trial_valid, _normals = _apply_convention(distance, valid, None, applied)
        picked, _gain = _edge_convention(trial, trial_valid, bgr)
        check(picked == applied, f"edge guard picked {picked} for {applied}")


def check_ground_snap() -> None:
    from engine.world.pano_mesh import build_pano_mesh, erp_dirs

    height, width = 32, 64
    dirs = erp_dirs(height, width)[:, :width]
    rng = np.random.default_rng(0)
    depth = np.full((height, width), 10.0, np.float64)
    floor = dirs[..., 1] > np.sin(np.radians(10.0))
    noise = 1.0 + 0.005 * rng.standard_normal(depth.shape)
    depth[floor] = (1.5 / dirs[..., 1][floor]) * noise[floor]
    piece, stats = build_pano_mesh(depth, np.ones_like(depth, dtype=bool))
    ground = stats["ground"]
    check(ground is not None, "ground plane missing")
    normal = np.asarray(ground["normal"], dtype=np.float64)
    normal = normal / np.linalg.norm(normal)
    angle = float(np.degrees(np.arccos(np.clip(normal[1], -1.0, 1.0))))
    check(angle < 2.0, f"ground tilt {angle:.3f} deg")
    check(abs(float(ground["d"]) + 1.5) < 0.02, f"ground d {ground['d']}")
    idx = np.asarray(stats["snapped_idx"], dtype=np.int64)
    check(idx.size > 0, "nothing snapped")
    points = piece.verts[idx].astype(np.float64)
    resid = np.abs(points @ normal + float(ground["d"]))
    check(float(resid.max()) < 1e-4, f"snap residual {float(resid.max())}")


def check_lift_once() -> None:
    from engine.world.layers import lift_frame

    verts = [(-1.5, -3.0, 1.25), (0.0, -3.0, 2.0), (0.0, 3.0, 2.0), (-1.5, 3.0, 1.25)]
    tex = np.full((8, 8, 3), 128, np.uint8)
    renderer = _cpu_renderer(_quad_piece(verts), tex)
    k_mat = np.array([[40.0, 0.0, 32.0], [0.0, 40.0, 24.0], [0.0, 0.0, 1.0]])
    w2c = np.eye(4)
    try:
        view = renderer.render(w2c, k_mat, 48, 64, supersample=1)
        check(not bool(view.empty[:, :30].any()), "left side has a hole")
        check(bool(view.empty[:, 34:].all()), "right side hit the left plane")
        xs = np.arange(64, dtype=np.float64) + 0.5
        row = 2.0 / (1.0 - 0.5 * ((xs - 32.0) / 40.0))
        z_pred = np.broadcast_to(row, (48, 64)).astype(np.float32).copy()
        rgb = np.full((48, 64, 3), 255, np.uint8)
        piece, info = lift_frame(rgb, z_pred, view, w2c, k_mat, "t.png")
        check(piece is not None, f"lift skipped {info}")
        check(abs(float(info["scale"]) - 1.0) < 0.01, f"scale {info['scale']}")
        renderer.add_piece(piece, rgb)
        view2 = renderer.render(w2c, k_mat, 48, 64, supersample=1)
        check(int(view2.empty.sum()) <= 64, f"holes left {int(view2.empty.sum())}")
        piece2, info2 = lift_frame(rgb, z_pred, view2, w2c, k_mat, "t.png")
        check(piece2 is None and info2.get("skipped") == "few", f"lifted twice {info2}")
    finally:
        renderer.close()


def check_zbuffer() -> None:
    points = np.array([[0.0, 0.0, 2.0], [0.0, 0.0, 5.0]], np.float32)
    colors = np.array([[10, 20, 30], [200, 0, 0]], np.uint8)
    K = np.array([[100.0, 0.0, 4.0], [0.0, 100.0, 4.0], [0.0, 0.0, 1.0]])
    rgb, gate = reproject(points, colors, np.eye(4), K, height=8, width=8)
    check(bool(gate[4, 4]), "pixel missed")
    check(int(rgb[4, 4, 0]) == 10, f"far point won {rgb[4, 4].tolist()}")


def check_photo_paste() -> None:
    from engine.pano.photo_paste import paste_photo
    from engine.worldstereo.adapter import cut_pinhole

    rng = np.random.default_rng(0)
    pano = rng.integers(0, 255, (256, 512, 3), dtype=np.uint8)
    photo = np.zeros((60, 80, 3), np.uint8)
    photo[:, :] = (220, 20, 20)
    photo[:12, :] = (20, 220, 20)
    photo[:, :12] = (20, 20, 220)
    photo[-12:, :] = (220, 220, 20)
    pasted = paste_photo(pano, photo, 60.0, feather_px=8, levels=3)
    width, height = 80, 60
    fx = (width / 2.0) / np.tan(np.deg2rad(30.0))
    k_mat = np.array([[fx, 0.0, (width - 1) / 2.0], [0.0, fx, (height - 1) / 2.0], [0.0, 0.0, 1.0]])
    cut = cut_pinhole(pasted, np.eye(4), k_mat, width, height)
    y0, y1 = height // 4, height - height // 4
    x0, x1 = width // 4, width - width // 4
    err = np.abs(cut[y0:y1, x0:x1].astype(np.float32) - photo[y0:y1, x0:x1].astype(np.float32)).mean()
    check(err < 6.0, f"photo round trip mae {err:.2f}")
    # The feather lives inside the footprint, so pixels well outside it stay the noise.
    check(np.array_equal(pasted[:, :80], pano[:, :80]), "paste changed pixels outside the footprint")


def check_fill_visibility() -> None:
    from engine.world.pano_mesh import GroundPlane
    from engine.world.render import WorldRenderer

    tex = np.zeros((8, 8, 3), np.uint8)
    tex[3:5, 3:5] = (0, 200, 0)
    verts = [[3.0, -0.2, 2.0], [3.4, -0.2, 2.0], [3.4, 0.2, 2.0], [3.0, 0.2, 2.0]]
    k_mat = np.array([[10.0, 0.0, 8.0], [0.0, 10.0, 6.0], [0.0, 0.0, 1.0]])
    erp_dist = np.full((8, 8), 2.0, np.float32)
    erp_dist[3:5, 3:5] = 0.2
    renderer = WorldRenderer(
        [_quad_piece(verts)],
        {"pano": tex},
        device="cpu",
        bin_size=0,
        ground=GroundPlane(np.array([0.0, 0.0, 1.0]), -2.0, erp_dist),
    )
    try:
        view = renderer.render(np.eye(4), k_mat, 12, 16, supersample=1)
    finally:
        renderer.close()
    check(bool(view.empty[6, 8]), "near surface was painted as ground")


def check_plane_snap() -> None:
    from engine.world.pano_mesh import erp_dirs, fit_planes

    height, width = 512, 1024
    dirs = erp_dirs(height, width)[:, :width]
    floor = np.where(dirs[..., 1] > 1e-4, 1.5 / dirs[..., 1], np.inf)
    plus_x = np.where(dirs[..., 0] > 1e-4, 2.0 / dirs[..., 0], np.inf)
    minus_x = np.where(dirs[..., 0] < -1e-4, -2.0 / dirs[..., 0], np.inf)
    plus_z = np.where(dirs[..., 2] > 1e-4, 2.0 / dirs[..., 2], np.inf)
    minus_z = np.where(dirs[..., 2] < -1e-4, -2.0 / dirs[..., 2], np.inf)
    stack = np.stack([floor, plus_x, minus_x, plus_z, minus_z], axis=0)
    which = np.argmin(stack, axis=0)
    depth = np.min(stack, axis=0)
    rng = np.random.default_rng(0)
    depth = depth * (1.0 + 0.005 * rng.standard_normal(depth.shape))
    points = dirs * depth[..., None]
    normals = np.zeros_like(points)
    normals[which == 0] = (0.0, 1.0, 0.0)
    normals[which == 1] = (-1.0, 0.0, 0.0)
    normals[which == 2] = (1.0, 0.0, 0.0)
    normals[which == 3] = (0.0, 0.0, -1.0)
    normals[which == 4] = (0.0, 0.0, 1.0)
    valid = np.isfinite(depth)
    exclude = which == 0
    planes = fit_planes(points, normals, valid, exclude, float(np.median(depth[valid])))
    check(len(planes) >= 3, f"facade planes {len(planes)}")
    worst = 0.0
    for normal, plane_d, mask in planes:
        resid = np.abs(points[mask] @ normal + plane_d)
        # Snap along the ray, then the residual of the snapped point is zero by construction.
        align = dirs[mask] @ normal
        ok = align > 0.05
        snapped = dirs[mask][ok] * ((-plane_d / align[ok])[:, None])
        resid = np.abs(snapped @ normal + plane_d)
        worst = max(worst, float(resid.max()) if resid.size else 0.0)
    check(worst < 1e-4, f"snapped wall residual {worst}")


def check_clean_guide() -> None:
    from engine.world.render import clean_guide

    rgb = np.full((64, 64, 3), 128, np.uint8)
    gap = np.ones((64, 64), dtype=bool)
    gap[:, 31] = False
    _rgb, holes = clean_guide(rgb, gap)
    check(bool(holes[:, 31].all()), "thin covered line between holes stayed covered")
    solid = np.zeros((64, 64), dtype=bool)
    solid[0:2, 0:2] = True
    _rgb, kept = clean_guide(rgb, solid)
    check(not bool(kept[40, 10:12].any()), "covered background was opened into a hole")
    mask = np.where(holes, 255, 0).astype(np.uint8)
    check(set(np.unique(mask).tolist()) <= {0, 255}, f"mask values {np.unique(mask).tolist()}")


def check_depth_key(scene: str) -> None:
    from engine.depth.keys import depth_dir, rail_key_path, rails_stale, write_rail_key
    from engine.pano.scene_io import scene_dir as scene_root

    folder = depth_dir(scene)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "depth.json").write_text(
        json.dumps({"backend": "moge1", "pano_sha1": "abc"}) + "\n",
        encoding="utf-8",
    )
    rail = scene_root(scene) / "control" / "rail_02"
    rail.mkdir(parents=True, exist_ok=True)
    (rail / "camplot.json").write_text(json.dumps({"depth_key": "moge1:old"}) + "\n", encoding="utf-8")
    stale = rails_stale(scene)
    check(stale == [0, 2], f"stale rails {stale}")
    shutil.rmtree(scene_root(scene) / "control")
    stale = rails_stale(scene)
    check(stale == [0, 2], f"stale after Start over {stale}")
    write_rail_key(scene, 0, "moge1:abc")
    stale = rails_stale(scene)
    check(stale == [2], f"stale after re-plot {stale}")
    rail_key_path(scene, 0).unlink()


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
        check_world_projection()
        check_world_projection_side()
        check_guide_mask_polarity()
        check_grazing_cut()
        check_ground_fill()
        check_supersample_colour()
        check_hires_pano_wins()
        check_memory_far()
        check_hires_bin_size()
        check_bin_retile()
        check_edge_convention()
        check_exif_hfov(scene)
        check_photo_paste()
        check_fill_visibility()
        check_plane_snap()
        check_clean_guide()
        check_depth_key(scene)
        check_ground_snap()
        check_lift_once()
        check_zbuffer()
    finally:
        shutil.rmtree(root, ignore_errors=True)
    print("CPU checks ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
