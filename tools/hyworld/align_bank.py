"""Align generated frames to the panorama cloud. Runs inside the hyworld venv."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import cv2
import torch
from PIL import Image


def _log(msg: str) -> None:
    print(msg, flush=True)


def _frames(path: Path) -> list[Image.Image]:
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    cap.release()
    return frames


class _EmptySession:
    def __init__(self, count: int) -> None:
        self.count = count


class _NoSkyProcessor:
    def init_video_session(self, **kwargs):
        video = kwargs["video"]
        return _EmptySession(int(video.shape[0]))

    def add_text_prompt(self, inference_session=None, text: str = ""):
        del text
        return inference_session


class _NoSkyModel:
    def propagate_in_video_iterator(self, inference_session=None, **kwargs):
        del inference_session, kwargs
        return []


def _preview(save: Path, out: Path) -> None:
    """Top view: panorama cloud blue, aligned new points orange. Never fails the job."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        import trimesh

        fig, ax = plt.subplots(figsize=(8, 8), dpi=120, facecolor="#081018")
        ax.set_facecolor("#081018")
        every = []
        for name, color in (("global_pcd.ply", "#4da3ff"), ("aligned_pcd.ply", "#ff9f30")):
            path = save / name
            if not path.is_file():
                continue
            pts = np.asarray(trimesh.load(str(path)).vertices)
            if len(pts) > 200_000:
                pts = pts[np.random.default_rng(0).choice(len(pts), 200_000, replace=False)]
            ax.scatter(pts[:, 1], -pts[:, 0], s=0.05, c=color, linewidths=0)
            every.append(pts)
        if every:
            pts = np.concatenate(every)
            ax.set_xlim(np.percentile(pts[:, 1], [1, 99]))
            ax.set_ylim(np.percentile(-pts[:, 0], [1, 99]))
        ax.set_aspect("equal")
        ax.axis("off")
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out), facecolor="#081018", bbox_inches="tight")
        plt.close(fig)
        print(f"HY\taligned\t{out}", flush=True)
    except Exception as exc:
        print(f"STATUS // aligned preview skipped: {exc}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_path", required=True)
    args = parser.parse_args()
    import torch.distributed as dist

    if not dist.is_initialized():
        dist.init_process_group(backend="gloo", rank=0, world_size=1)
    root = Path(args.target_path)
    from src.retrieval_wm import PanoramaMemoryBank

    device = torch.device("cuda:0")
    sample = next((root / "render_results" / "pano_bank" / "images").glob("*.png"))
    import imagesize

    width, height = imagesize.get(str(sample))
    meta = json.loads((root / "meta_info.json").read_text(encoding="utf-8"))
    _log(f"STATUS // align scene_type={meta.get('scene_type')}")
    bank = PanoramaMemoryBank(
        root_path=str(root).replace("\\", "/"),
        image_width=width,
        image_height=height,
        device=device,
        nframe=21,
        max_reference=8,
        align_nframe=8,
        rank=0,
        world_size=1,
        sam3_model=_NoSkyModel(),
        sam3_processor=_NoSkyProcessor(),
        results_name="worldstereo-memory-dmd",
        valid_threshold=0.15,
        pts_num=2_000_000,
    )
    from engine.hyworld.paths import ordered_ids

    found = {f"{v.parent.parent.name}/{v.parent.name}": v
             for v in (root / "render_results").glob("*/*/worldstereo-memory-dmd_result.mp4")}
    videos = [found[item] for item in ordered_ids(list(found))]
    if not videos:
        _log("STATUS // no expanded trajectories to align")
        return 2
    for video in videos:
        traj = video.parent
        view = traj.parent
        camera = json.loads((traj / "camera.json").read_text(encoding="utf-8"))
        frames = _frames(video)
        w2cs = torch.tensor(camera["extrinsic"], dtype=torch.float32, device=device)
        ks = torch.tensor(camera["intrinsic"], dtype=torch.float32, device=device)
        bank.update_memory(frames, w2cs, ks, view.name, traj.name)
        _log(f"STATUS // align remembered {view.name}/{traj.name} bank={bank.mem_size}")
    save = root / "render_results" / "generation_bank_worldstereo-memory-dmd"
    bank.world_mirror_dir = str(save / "world_mirror_data").replace("\\", "/")
    bank.name_map = defaultdict(lambda: "none")
    os.makedirs(bank.world_mirror_dir, exist_ok=True)
    bank.alignment(debug_mode=False)
    bank.export_pcd(str(save).replace("\\", "/"), N_points=2_000_000)
    _preview(save, root / "_votion" / "preview" / "aligned_top.png")
    _log(f"STATUS // aligned cloud {save / 'aligned_pcd.ply'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
