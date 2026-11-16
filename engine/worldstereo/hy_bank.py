"""HY-World panorama retrieval on the vendored WorldStereo bank.

``video_gen.py`` searches with ``far = max(depth_median * 8, 0.15)``. The
public bank hardcodes ``far = 10``. The DiT, the joint chunks, and the output
folders stay this app's. WorldMirror and the SAM3 alignment pass run after
HY-World's samples for their own Gaussian trainer. This app keeps known-pose
reconstruct, so those steps are not started.
"""

from __future__ import annotations

import collections
import sys

import numpy as np
import torch

from engine.job import REPO_ROOT

_WS = REPO_ROOT / "vendor" / "worldstereo"
if str(_WS) not in sys.path:
    sys.path.insert(0, str(_WS))

from src.general_utils import rank0_log  # noqa: E402
from src.retrieval_wm import SimpleMemoryBank, find_closest_camera_in_view  # noqa: E402


def memory_far(depth_median: float) -> float:
    """Same far plane as HY-World ``PanoramaMemoryBank.retrieval``."""
    return max(float(depth_median) * 8.0, 0.15)


def scene_depth_median(scene: str) -> float:
    """Median of the cached MoGe distance. Sky fill is left out."""
    from engine.depth.scene_depth import ensure_scene_depth

    depth = ensure_scene_depth(scene, "cpu")
    valid = depth.valid & np.isfinite(depth.distance) & (depth.distance > 0)
    if not bool(valid.any()):
        raise RuntimeError("scene depth has no valid pixels for the memory far plane")
    return float(np.median(depth.distance[valid]))


class HyMemoryBank(SimpleMemoryBank):
    """Vendored bank, with HY-World's depth-scaled field of view."""

    def __init__(self, *args, depth_median: float, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.depth_median = float(depth_median)
        self.ref_depths: list = [None] * len(self.ref_frames)

    @property
    def far(self) -> float:
        return memory_far(self.depth_median)

    def retrieval(self, tar_w2cs_full, tar_Ks_full, view_id=None, traj_id=None):
        if ("wonder" in view_id or "target" in view_id) and traj_id > "traj0":
            shortcut_type = "aerial"
            shortcut_indices = [i for i, fname in enumerate(self.fnames) if fname.startswith(f"{view_id}/traj0")]
            rank0_log(f"Using {len(shortcut_indices)} history from {view_id}/traj0 ...")
            shortcut_indices = shortcut_indices[3::2]
        elif "view" in view_id and traj_id in ("traj0", "traj1"):
            shortcut_type = "regular"
            shortcut_indices = [i for i, fname in enumerate(self.fnames) if fname.startswith(f"{view_id}/traj2")]
            rank0_log(f"Using {len(shortcut_indices)} history from {view_id}/traj2 ...")
            if traj_id == "traj0":
                shortcut_indices = shortcut_indices[3::2]
            else:
                shortcut_indices = shortcut_indices[: len(shortcut_indices) // 2][3::2]
        else:
            shortcut_type = "none"
            shortcut_indices = []

        if tar_w2cs_full.shape[0] > self.nframe:
            tar_w2cs = tar_w2cs_full[0::4]
            tar_Ks = tar_Ks_full[0::4]
        else:
            tar_w2cs = tar_w2cs_full
            tar_Ks = tar_Ks_full

        retrieval_map = {}
        ref_index_dict = collections.defaultdict(dict)
        retrieved_frames = []
        retrieved_w2cs = []
        retrieved_Ks = []
        ref_w2cs = []
        retrieved_global_indices = []
        far = self.far

        for i in range(1, tar_w2cs.shape[0]):
            if len(shortcut_indices) > 0:
                if shortcut_type == "aerial" and i not in (4, 8, 12, 16):
                    shortcut_index = shortcut_indices[int(i / tar_w2cs.shape[0] * len(shortcut_indices))]
                elif shortcut_type == "regular" and i in (1, 2, 3, 5, 7, 9, 13, 17):
                    shortcut_index = shortcut_indices[int(i / tar_w2cs.shape[0] * len(shortcut_indices))]
                else:
                    shortcut_index = None
            else:
                shortcut_index = None

            best_idx, best_score, angle_diff = find_closest_camera_in_view(
                tar_w2cs[i],
                self.ref_w2cs,
                tar_Ks[i],
                self.ref_Ks,
                self.image_width,
                self.image_height,
                method="fov_overlap",
                near=0.1,
                far=far,
                angle_penalty=True,
                shortcut_index=shortcut_index,
            )
            if best_idx not in retrieval_map:
                retrieval_map[best_idx] = i - 1
            ref_index_dict[retrieval_map[best_idx]][i] = {"score": best_score, "angle_diff": angle_diff}
            retrieved_frames.append(np.array(self.ref_frames[best_idx])[None])
            retrieved_w2cs.append(self.ref_w2cs[best_idx])
            retrieved_Ks.append(self.ref_Ks[best_idx])
            retrieved_global_indices.append(best_idx)

        if len(ref_index_dict) > self.max_reference:
            rank0_log(f"Too many references. {len(ref_index_dict)} > {self.max_reference}")
            retrieved_w2cs = torch.stack(retrieved_w2cs)
            retrieved_Ks = torch.stack(retrieved_Ks)
            indices, _ = self.camera_selector.select(
                retrieved_w2cs.cpu().numpy(),
                retrieved_frames,
                topk=self.max_reference,
                camera_weight=0.3,
                image_weight=0.7,
                quality_bias=0.1,
            )
            retrieved_w2cs = retrieved_w2cs[indices]
            retrieved_Ks = retrieved_Ks[indices]
            retrieved_frames_selected = [retrieved_frames[i] for i in indices]
            retrieved_frames = []
            retrieved_global_indices_selected = [retrieved_global_indices[i] for i in indices]
            retrieved_global_indices = []
            retrieval_map = {}
            ref_index_dict = collections.defaultdict(dict)
            for i in range(1, tar_w2cs.shape[0]):
                best_idx, best_score, angle_diff = find_closest_camera_in_view(
                    tar_w2cs[i],
                    retrieved_w2cs,
                    tar_Ks[i],
                    retrieved_Ks,
                    self.image_width,
                    self.image_height,
                    method="fov_overlap",
                    near=0.1,
                    far=far,
                    angle_penalty=True,
                )
                if best_idx not in retrieval_map:
                    retrieval_map[best_idx] = i - 1
                ref_index_dict[retrieval_map[best_idx]][i] = {"score": best_score, "angle_diff": angle_diff}
                retrieved_frames.append(np.array(retrieved_frames_selected[best_idx]))
                ref_w2cs.append(retrieved_w2cs[best_idx])
                retrieved_global_indices.append(retrieved_global_indices_selected[best_idx])
        else:
            ref_w2cs = retrieved_w2cs

        ref_index = torch.tensor(list(ref_index_dict.keys()), dtype=torch.long)
        retrieved_frames = np.concatenate(retrieved_frames, axis=0)
        ref_w2cs = torch.stack(ref_w2cs)[ref_index]
        return retrieved_frames, ref_index, ref_index_dict, ref_w2cs, retrieved_global_indices
