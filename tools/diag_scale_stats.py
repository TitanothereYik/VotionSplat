"""One-off: dump Gaussian scale distribution + distance-to-nearest-camera correlation for the
current checkpoint. Diagnostic only, not part of the product CLI.

Run as ``python -m tools.diag_scale_stats --scene NAME``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from engine.pano.scene_io import scene_dir
from engine.splat import defaults as D
from engine.splat.backend.litegs_backend import LiteGSBackend, ensure_litegs


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    args = parser.parse_args(argv)

    root = Path(scene_dir(args.scene))
    ckpt_path = root / "checkpoints" / "splat_latest.pt"
    for line in ensure_litegs():
        print(line)
    device = torch.device("cuda:0")
    backend = LiteGSBackend(root, device, sh_degree=D.SH_DEGREE, resolution_scale=1.0, holdout=True, log=print)
    backend.load_colmap()
    meta, _ctl = backend.load_checkpoint(ckpt_path)
    print(f"loaded step={meta.get('step')} controller={meta.get('controller')} N={backend.n_gaussians}")
    print(f"norm_radius={backend.norm_radius:.4f} scene_scale={backend.scene_scale:.4f} cam_center={backend.cam_center}")

    with torch.no_grad():
        xyz, scale, rot, sh0, shr, opa = backend.lg_scene.cluster.uncluster(*backend.param_tuple())
        scale_ws = scale.exp()  # [3, N]
        max_scale = scale_ws.max(dim=0).values.detach().cpu().numpy()
        xyz_np = xyz.detach().cpu().numpy()  # [3, N]

    print(f"max_scale: min={max_scale.min():.5f} p50={np.percentile(max_scale,50):.5f} "
          f"p99={np.percentile(max_scale,99):.5f} p99.9={np.percentile(max_scale,99.9):.5f} "
          f"max={max_scale.max():.5f}")
    print(f"as multiple of norm_radius: p99={np.percentile(max_scale,99)/backend.norm_radius:.5f} "
          f"p99.9={np.percentile(max_scale,99.9)/backend.norm_radius:.5f} "
          f"max={max_scale.max()/backend.norm_radius:.5f}")

    # camera centers (world space)
    centers = np.stack([-(v.w2c[:3, :3].T @ v.w2c[:3, 3]) for v in backend.views], axis=0)  # [V,3]
    xyz_t = xyz_np.T  # [N,3]

    # nearest-camera distance per gaussian, sampled (full NxV would be too big)
    rng = np.random.default_rng(0)
    n = xyz_t.shape[0]
    sample_n = min(n, 200_000)
    idx = rng.choice(n, size=sample_n, replace=False)
    sample_xyz = xyz_t[idx]
    sample_scale = max_scale[idx]
    # distance to closest camera center for each sampled gaussian
    # chunk to avoid huge memory: V is likely <1000
    dmin = np.full(sample_n, np.inf, dtype=np.float64)
    chunk = 20000
    for s in range(0, sample_n, chunk):
        e = min(s + chunk, sample_n)
        diff = sample_xyz[s:e, None, :] - centers[None, :, :]
        d = np.linalg.norm(diff, axis=2).min(axis=1)
        dmin[s:e] = d

    print(f"dist-to-nearest-cam: min={dmin.min():.5f} p1={np.percentile(dmin,1):.5f} "
          f"p50={np.percentile(dmin,50):.5f}")

    # top 20 biggest gaussians: scale + dist-to-nearest-cam
    order = np.argsort(-sample_scale)[:20]
    print("top-20 biggest sampled gaussians: scale, dist_to_nearest_cam, dist/norm_radius")
    for i in order:
        print(f"  scale={sample_scale[i]:.4f} dist={dmin[i]:.4f} dist/nr={dmin[i]/backend.norm_radius:.4f} "
              f"scale/dist_ratio={sample_scale[i]/max(dmin[i],1e-6):.4f}")

    # how many gaussians are within a "near" radius of some camera (degenerate risk)
    near_thresh = 0.05 * backend.norm_radius
    n_near = int((dmin < near_thresh).sum())
    print(f"sampled gaussians within {near_thresh:.4f} (0.05*norm_radius) of a camera: {n_near}/{sample_n}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
