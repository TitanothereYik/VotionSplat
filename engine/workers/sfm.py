"""SphereSfM worker: equirect-like frames -> COLMAP images/ + sparse/0.

Qt never imports this. Run as ``python -m engine.workers.sfm``.

Uses SplatKit ``core.spheresfm_colmap.run_spheresfm`` / ``run_spheresfm_dualres`` at pin
f59de25. Windows ``colmap_sphere.exe`` SHA-256 is copied from that file's
``_BUNDLES['win64']`` — do not invent a hash. First run auto-downloads into
``vendor/splatkit/bin/``.

Two paths, chosen per-run in ``main()``:
  dualres     Every kept rail has HiRes frames on disk (``engine/workers/hires.py``, run
              from the Generate page). Poses are solved on the HiRes proxies (2K, fast,
              pixel-aligned to the full-res frames by construction), and the trainable
              cube faces are reprojected from the full-res frames (8K by default) — 972
              training images at ~2048x2048 instead of ~360x360. This is what makes
              ``splat.ply`` sharp; see ``run_spheresfm_dualres`` in spheresfm_colmap.py.
  single-res  Fallback when HiRes hasn't been run (or is partial) for the kept rails:
              poses AND cube faces both come from the WAN clip (~360x360 faces), plus
              ``pano.png`` inserted once as a single hires (2048x2048) ``initial_pano``
              anchor view — the P2-era path, before HiRes/DualRes existed. Mesh-only
              control video (no WAN) also lands here; it has huge black disocclusions,
              but SfM still must run so the plumbing is real.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SPLATKIT = REPO_ROOT / "vendor" / "splatkit"
if str(SPLATKIT) not in sys.path:
    sys.path.insert(0, str(SPLATKIT))

from engine.geo.rails import (
    confirmed_path as rails_confirmed_path,
    control_video_rel,
    hires_frames_rel,
    hires_proxies_rel,
    sync_dataset_from_confirmed,
    wan_video_rel,
)
from engine.pano.scene_io import pano_path, scene_dir
from engine.sfm import defaults as D
from engine.sfm.cubic_reproject import rewrite_pinhole_faces
from engine.job import job_progress

_HIRES_FRAME_RE = re.compile(r"_frame_(\d+)\.png$", re.IGNORECASE)


def log(msg: str) -> None:
    print(msg, flush=True)


def confirmed_path(scene: str) -> Path:
    return rails_confirmed_path(scene)


def kept_rail_rows(root: Path, scene: str) -> list[tuple[int, Path]]:
    """Kept rails as (rail_index, video_path): WAN clip if present, else mesh control video."""
    data = sync_dataset_from_confirmed(scene)
    rows: list[tuple[int, Path]] = []
    for row in data.get("rails") or []:
        if not row.get("keep", True):
            continue
        idx = int(row.get("index", 0))
        wan = root / str(row.get("wan") or wan_video_rel(idx))
        ctrl = root / str(row.get("video") or control_video_rel(idx))
        if wan.is_file() and wan.stat().st_size > 0:
            rows.append((idx, wan))
        elif ctrl.is_file() and ctrl.stat().st_size > 0:
            rows.append((idx, ctrl))
    return rows


def dataset_videos(root: Path, scene: str) -> list[Path]:
    """Kept rails: WAN clip if present, else mesh control video."""
    return [path for _idx, path in kept_rail_rows(root, scene)]


def _hires_pairs_in(dir_rel: str, root: Path, rail_index: int) -> list[tuple[int, Path]]:
    """``[(original WAN-frame index, path), ...]`` for one rail's PNGs under ``dir_rel``,
    sorted by index; ``[]`` if the directory or this rail's files are absent."""
    frames_dir = root / dir_rel
    if not frames_dir.is_dir():
        return []
    prefix = f"traj{int(rail_index):02d}_frame_"
    pairs: list[tuple[int, Path]] = []
    for p in frames_dir.glob(f"{prefix}*.png"):
        m = _HIRES_FRAME_RE.search(p.name)
        if m:
            pairs.append((int(m.group(1)), p))
    pairs.sort(key=lambda t: t[0])
    return pairs


def hires_frame_pairs(root: Path, rail_index: int) -> list[tuple[int, Path]]:
    """Full-resolution HiRes composite frames for one rail (``engine/workers/hires.py``'s
    ``frames/``) — the reprojection source for dual-res SfM. Checks the files, not
    ``dataset.json``'s ``hires`` flag — a HiRes job that was interrupted or cleaned up can
    leave the flag set with no frames behind it."""
    return _hires_pairs_in(hires_frames_rel(), root, rail_index)


def hires_proxy_pairs(root: Path, rail_index: int) -> list[tuple[int, Path]]:
    """Downscaled copies of ``hires_frame_pairs`` (same filenames, ``proxies/``) — the
    pose-solving source for dual-res SfM."""
    return _hires_pairs_in(hires_proxies_rel(), root, rail_index)


def _rmtree(path: Path) -> None:
    if not path.exists():
        return
    last: OSError | None = None
    for _ in range(8):
        try:
            shutil.rmtree(path)
        except OSError as exc:
            last = exc
            time.sleep(0.2)
            continue
        if not path.exists():
            return
        time.sleep(0.2)
    raise RuntimeError(f"could not remove previous {path}: {last}")


def wipe_previous_colmap(out_dir: Path) -> None:
    """Every Reconstruct click solves the current kept rails. Drop the last sparse model."""
    for name in ("images", "sparse", "_spheresfm_work"):
        _rmtree(out_dir / name)
    stale = out_dir / "sfm.json"
    if stale.is_file():
        stale.unlink()


def load_videos_rgb(paths: list[Path]) -> tuple[np.ndarray, list[int]]:
    stacks = [load_video_rgb(path) for path in paths]
    if not stacks:
        raise RuntimeError("no kept rail videos")
    h, w = stacks[0].shape[1], stacks[0].shape[2]
    aligned = []
    lengths = []
    for frames in stacks:
        if frames.shape[1] != h or frames.shape[2] != w:
            resized = [
                cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA) for frame in frames
            ]
            frames = np.stack(resized, axis=0)
        aligned.append(frames)
        lengths.append(int(frames.shape[0]))
    return np.concatenate(aligned, axis=0), lengths


def load_video_rgb(path: Path) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open {path}")
    frames: list[np.ndarray] = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    if len(frames) < 3:
        raise RuntimeError(f"need at least 3 frames for SfM; got {len(frames)} from {path}")
    return np.stack(frames, axis=0)


def load_pano_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"failed to read {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _prepare_dualres(
    rows: list[tuple[int, Path]],
    frames_by_rail: dict[int, list[tuple[int, Path]]],
    proxies_by_rail: dict[int, list[tuple[int, Path]]],
) -> tuple[np.ndarray, list[str], list[int]] | None:
    """Pose-frame array (the HiRes proxies), ordered full-res reprojection paths, and
    per-rail lengths for ``run_spheresfm_dualres``, one rail at a time (rail-major,
    matching ``trajectory_lengths``). Proxies and frames are the SAME composite run
    (``engine/workers/hires.py``) at two sizes, so they are pixel-aligned by construction —
    no separate WAN clip is loaded here. ``None`` if a rail's proxy/frame sets don't line up
    1:1 by original frame index (stale/partial HiRes output): the caller falls back to the
    plain low-res (WAN clip) path rather than mixing resolutions."""
    lowres_list: list[np.ndarray] = []
    hires_paths: list[str] = []
    traj_lengths: list[int] = []
    lo_h = lo_w = None
    for idx, _path in rows:
        frame_pairs = frames_by_rail[idx]
        proxy_pairs = proxies_by_rail[idx]
        frame_fi = [f for f, _ in frame_pairs]
        proxy_fi = [f for f, _ in proxy_pairs]
        if not frame_fi or frame_fi != proxy_fi:
            log(
                f"STATUS // rail {idx}: HiRes frames/ ({len(frame_fi)}) and proxies/ "
                f"({len(proxy_fi)}) don't match 1:1 — reconstructing at WAN resolution for "
                "all kept rails instead"
            )
            return None
        sub = np.stack([load_pano_rgb(p) for _, p in proxy_pairs], axis=0)
        if lo_h is None:
            lo_h, lo_w = int(sub.shape[1]), int(sub.shape[2])
        elif sub.shape[1] != lo_h or sub.shape[2] != lo_w:
            sub = np.stack(
                [cv2.resize(f, (lo_w, lo_h), interpolation=cv2.INTER_AREA) for f in sub],
                axis=0,
            )
        lowres_list.append(sub)
        hires_paths.extend(str(p) for _, p in frame_pairs)
        traj_lengths.append(int(sub.shape[0]))
    return np.concatenate(lowres_list, axis=0), hires_paths, traj_lengths


def _finish(out_dir: Path, res: dict, extra_meta: dict, run_label: str) -> int:
    """Shared tail for both SfM paths: gnomonic cube remap, ``sfm.json``, ``SFM`` protocol."""
    job_progress(4, 5, "gnomonic cube faces")
    sparse = Path(res["sparse_dir"])
    try:
        n_faces = rewrite_pinhole_faces(out_dir, log=log)
    except Exception as exc:
        log(f"STATUS // gnomonic cube remap failed: {exc}")
        return 4
    run_id = str(time.time_ns())
    for name in ("cameras.bin", "images.bin", "points3D.bin"):
        bin_path = sparse / name
        if bin_path.is_file():
            bin_path.touch()
    meta = {
        "run_id": run_id,
        "num_images": int(res["num_images"]),
        "num_points": int(res["num_points"]),
        "num_frames": int(res["num_frames"]),
        "sparse": "colmap/sparse/0",
        "images": "colmap/images",
        **extra_meta,
        "gnomonic_faces": True,
        "gnomonic_count": int(n_faces),
    }
    (out_dir / "sfm.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    log(f"SFM\tnum_images\t{res['num_images']}")
    log(f"SFM\tnum_points\t{res['num_points']}")
    log(f"SFM\tsparse\t{sparse}")
    log(f"SFM\trun\t{run_id}\t{sparse}")
    log(
        f"STATUS // spheresfm done {res['num_frames']} frames -> "
        f"{res['num_images']} images, {res['num_points']} points [{run_label}]"
    )
    job_progress(5, 5, "sfm done")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Votion SphereSfM (single trajectory)")
    parser.add_argument("--scene", required=True)
    args = parser.parse_args(argv)

    root = scene_dir(args.scene)
    gate = confirmed_path(args.scene)
    if not gate.is_file():
        log("STATUS // Reconstruct locked — Play then Confirm rails on Geometry first")
        return 2
    rows = kept_rail_rows(root, args.scene)
    if not rows:
        log("STATUS // no kept rail videos — Generate WAN or Keep a rail on Reconstruct")
        return 2
    clips = [path for _idx, path in rows]
    pano_file = pano_path(args.scene)
    if not pano_file.is_file():
        log("STATUS // missing pano.png — run Phase 1 first")
        return 2

    from core import spheresfm_colmap as ss

    log("STATUS // resolving colmap_sphere (SplatKit SHA check, auto-download if missing)")
    job_progress(1, 5, "resolve colmap_sphere")
    try:
        exe = ss.find_colmap_sphere()
    except Exception as exc:
        log(f"STATUS // {exc}")
        return 3
    log(f"STATUS // colmap_sphere {exe}")

    out_dir = root / "colmap"
    work_dir = out_dir / "_spheresfm_work"

    # DualRes: pose on the HiRes proxies (fast, reliable matching, pixel-aligned to the
    # full-res frames by construction), reproject the trainable cube faces from the HiRes
    # composite (engine/workers/hires.py, 8K by default) instead of the 720p WAN clip.
    # Only when EVERY kept rail has HiRes frames on disk — a partial set would mean mixing
    # resolutions across training views, which the plain path already does more
    # predictably via one shared low-res grid.
    frames_by_rail = {idx: hires_frame_pairs(root, idx) for idx, _ in rows}
    proxies_by_rail = {idx: hires_proxy_pairs(root, idx) for idx, _ in rows}
    missing_hires = [idx for idx, pairs in frames_by_rail.items() if not pairs]
    dualres_inputs = None
    if not missing_hires:
        log(f"STATUS // loading {len(rows)} kept rail(s) of HiRes reprojection frames")
        for idx, _path in rows:
            log(f"STATUS // reconstruct rail {idx}: {len(frames_by_rail[idx])} HiRes frames")
        job_progress(2, 5, "load HiRes proxies")
        try:
            dualres_inputs = _prepare_dualres(rows, frames_by_rail, proxies_by_rail)
        except Exception as exc:
            log(f"STATUS // {exc}")
            return 2
    elif any(frames_by_rail.values()):
        log(
            f"STATUS // HiRes frames missing for rail(s) {missing_hires} — reconstructing "
            "at WAN resolution for all kept rails (run HiRes on every kept rail for a "
            "sharper splat.ply)"
        )

    if dualres_inputs is not None:
        lowres_frames, hires_paths, traj_lengths = dualres_inputs
        matcher = "exhaustive" if len(traj_lengths) > 1 else D.MATCHER_TYPE
        log(
            f"STATUS // kept clips {len(clips)} frames {lowres_frames.shape[0]} "
            f"{lowres_frames.shape[2]}x{lowres_frames.shape[1]} dualres "
            f"({len(hires_paths)} HiRes reprojection frames) traj={traj_lengths} "
            f"matcher={matcher}"
        )
        log("STATUS // wiping previous colmap sparse so this run is the current kept rails")
        wipe_previous_colmap(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        log(
            f"STATUS // spheresfm(dualres) start scene={args.scene} matcher={matcher} "
            f"device_index={os.environ.get('CUDA_DEVICE_INDEX', '0')}"
        )
        job_progress(3, 5, "SphereSfM (dual-res)")
        try:
            res = ss.run_spheresfm_dualres(
                lowres_frames,
                hires_dir="",
                hires_paths=hires_paths,
                out_dir=str(out_dir),
                work_dir=str(work_dir),
                exe_path=exe,
                matcher_type=matcher,
                face_size=int(D.FACE_SIZE),
                max_num_features=int(D.MAX_NUM_FEATURES),
                peak_threshold=float(D.PEAK_THRESHOLD),
                edge_threshold=float(D.EDGE_THRESHOLD),
                max_num_matches=int(D.MAX_NUM_MATCHES),
                filter_max_reproj_error=float(D.FILTER_MAX_REPROJ_ERROR),
                filter_min_tri_angle=float(D.FILTER_MIN_TRI_ANGLE),
                init_min_tri_angle=float(D.INIT_MIN_TRI_ANGLE),
                init_min_num_inliers=int(D.INIT_MIN_NUM_INLIERS),
                init_max_forward_motion=float(D.INIT_MAX_FORWARD_MOTION),
                image_order=D.IMAGE_ORDER,
                trajectory_lengths=[int(x) for x in traj_lengths],
                on_split="stop",
            )
        except Exception as exc:
            log(f"STATUS // spheresfm(dualres) failed: {exc}")
            return 4
        extra_meta = {
            "num_rails": len(clips),
            "clips": [str(p) for p in clips],
            "traj_lengths": [int(x) for x in traj_lengths],
            "matcher_type": matcher,
            "dualres": True,
            "hires_frames": len(hires_paths),
        }
        return _finish(out_dir, res, extra_meta, "dualres")

    # ---- plain (low-res) path: WAN/control clips + the initial pano as one hires anchor
    log(f"STATUS // loading {len(clips)} kept rail clip(s)")
    for path in clips:
        log(f"STATUS // reconstruct clip {path}")
    job_progress(2, 5, "load rail videos")
    try:
        frames, traj_lengths = load_videos_rgb(clips)
    except Exception as exc:
        log(f"STATUS // {exc}")
        return 2
    idx = list(range(0, len(frames), max(1, int(D.FRAME_STRIDE))))
    if D.MAX_FRAMES and len(idx) > int(D.MAX_FRAMES):
        sel = np.linspace(0, len(idx) - 1, int(D.MAX_FRAMES)).round().astype(int)
        idx = [idx[i] for i in sorted(set(sel.tolist()))]
    if len(traj_lengths) == 1:
        frames = frames[idx]
        traj_lengths = [int(frames.shape[0])]
    if len(frames) < 3:
        log(f"STATUS // need at least 3 frames for SfM; got {len(frames)}")
        return 2
    matcher = "exhaustive" if len(traj_lengths) > 1 else D.MATCHER_TYPE
    log(
        f"STATUS // kept clips {len(clips)} frames {frames.shape[0]} "
        f"{frames.shape[2]}x{frames.shape[1]} traj={traj_lengths} matcher={matcher}"
    )
    ip_native = load_pano_rgb(pano_file)
    if D.INITIAL_PANO_HIRES:
        ip_for_sfm = ip_native
        log(
            f"STATUS // initial_pano native {ip_native.shape[1]}x{ip_native.shape[0]} "
            f"mode={D.INITIAL_PANO_MODE}"
        )
    else:
        th, tw = frames.shape[1], frames.shape[2]
        ip_for_sfm = (
            ip_native
            if ip_native.shape[:2] == (th, tw)
            else cv2.resize(ip_native, (tw, th), interpolation=cv2.INTER_AREA)
        )

    log("STATUS // wiping previous colmap sparse so this run is the current kept rails")
    wipe_previous_colmap(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log(
        f"STATUS // spheresfm start scene={args.scene} matcher={matcher} "
        f"stride={D.FRAME_STRIDE} device_index={os.environ.get('CUDA_DEVICE_INDEX', '0')}"
    )
    job_progress(3, 5, "SphereSfM")
    try:
        res = ss.run_spheresfm(
            frames,
            out_dir=str(out_dir),
            work_dir=str(work_dir),
            exe_path=exe,
            matcher_type=matcher,
            face_size=int(D.FACE_SIZE),
            max_num_features=int(D.MAX_NUM_FEATURES),
            peak_threshold=float(D.PEAK_THRESHOLD),
            edge_threshold=float(D.EDGE_THRESHOLD),
            max_num_matches=int(D.MAX_NUM_MATCHES),
            filter_max_reproj_error=float(D.FILTER_MAX_REPROJ_ERROR),
            filter_min_tri_angle=float(D.FILTER_MIN_TRI_ANGLE),
            init_min_tri_angle=float(D.INIT_MIN_TRI_ANGLE),
            init_min_num_inliers=int(D.INIT_MIN_NUM_INLIERS),
            init_max_forward_motion=float(D.INIT_MAX_FORWARD_MOTION),
            image_order=D.IMAGE_ORDER,
            trajectory_lengths=[int(x) for x in traj_lengths],
            initial_pano=ip_for_sfm,
            initial_pano_mode=D.INITIAL_PANO_MODE,
            reuse_solve=False,
        )
    except Exception as exc:
        log(f"STATUS // spheresfm failed: {exc}")
        return 4

    extra_meta = {
        "num_rails": len(clips),
        "clips": [str(p) for p in clips],
        "traj_lengths": [int(x) for x in traj_lengths],
        "matcher_type": matcher,
        "dualres": False,
    }
    return _finish(out_dir, res, extra_meta, "single-res")


if __name__ == "__main__":
    sys.exit(main())
