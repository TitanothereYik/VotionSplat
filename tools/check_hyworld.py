"""CPU checks for the HY-World runner. No GPU."""

from __future__ import annotations

import glob

import numpy as np

from engine.hyworld.gs_export import M, to_ply_arrays, world_points
from engine.hyworld.paths import apply_budget, close_selection, ordered_ids
from tools.hyworld.run_stage import _patch_glob, _patch_tqdm
from tools.hyworld.votion_live import frame_matrix, to_trainer, to_view


def check(cond: bool, msg: str) -> None:
    if not cond:
        raise SystemExit(msg)


def main() -> int:
    kept = apply_budget(
        ["view0/traj0", "view1/traj0", "wonder_0/traj0", "wonder_1/traj0", "target_door_0/traj0"],
        1,
    )
    check(kept == ["view0/traj0", "view1/traj0", "wonder_0/traj0", "target_door_0/traj0"], f"budget {kept}")
    _patch_glob()
    sample = glob.glob(r"C:\Windows\*")
    check(sample and "\\" not in sample[0], "glob paths use forward slashes")
    _patch_tqdm()
    import tqdm

    bar = tqdm.tqdm(range(2), desc="cpu")
    bar.update(2)
    bar.close()

    kept = apply_budget(["wonder_0/traj0", "wonder_0/traj1", "wonder_4/traj0"], 1)
    check(kept == ["wonder_0/traj0", "wonder_0/traj1"], f"budget by view {kept}")
    order = ordered_ids(["wonder_0/traj1", "view0/traj0", "reconstruct_0/traj1", "view0/traj2",
                         "target_a_0/traj0", "wonder_0/traj0", "reconstruct_0/traj0", "view0/traj1"])
    check(order == ["view0/traj2", "view0/traj0", "view0/traj1", "target_a_0/traj0", "reconstruct_0/traj0",
                    "wonder_0/traj0", "wonder_0/traj1", "reconstruct_0/traj1"], f"order {order}")
    check(close_selection(["wonder_0/traj1"], ["wonder_0/traj0", "wonder_0/traj1"]) == ["wonder_0/traj0", "wonder_0/traj1"], "closure")

    def rot(q):
        w, x, y, z = q
        return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                         [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                         [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])

    rng = np.random.default_rng(7)
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    s, t = 2.5, rng.normal(size=3)
    big_t = np.eye(4)
    big_t[:3, :3] = s * q
    big_t[:3, 3] = t
    p_hy = rng.normal(size=(5, 3))
    p_n = p_hy @ (s * q).T + t
    check(np.allclose(world_points(p_n, big_t), p_hy @ M, atol=1e-6), "world_points")
    quats = rng.normal(size=(5, 4))
    quats /= np.linalg.norm(quats, axis=1, keepdims=True)
    scales = rng.normal(size=(5, 3)) * 0.3
    arrays = to_ply_arrays({"means": p_n, "quats": quats, "scales": scales, "opacities": np.zeros(5),
                            "sh0": np.zeros((5, 1, 3)), "shN": np.zeros((5, 0, 3))}, big_t)
    turn = M.T @ q.T
    for i in range(5):
        cov = rot(quats[i]) @ np.diag(np.exp(2 * scales[i])) @ rot(quats[i]).T
        want = turn @ cov @ turn.T / (s * s)
        got = rot(arrays["rot"][:, i]) @ np.diag(np.exp(2 * arrays["scale"][:, i])) @ rot(arrays["rot"][:, i]).T
        check(np.allclose(got, want, atol=1e-6), "export covariance")

    a = frame_matrix([0, 0, 1], [-1, 0, 0])
    check(np.allclose(a @ [0, 0, 1], [0, -1, 0]) and np.allclose(a @ [-1, 0, 0], [0, 0, 1])
          and np.allclose(a @ [0, 1, 0], [1, 0, 0]), "view frame")
    c2w = np.eye(4)
    c2w[:3, :3] = q
    c2w[:3, 3] = t
    check(np.allclose(to_trainer(a, 3.0, to_view(a, 3.0, c2w)), c2w, atol=1e-9), "view round trip")
    print("HY-World CPU checks ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
