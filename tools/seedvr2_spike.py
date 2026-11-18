"""SeedVR2 test: compare upscalers on expanded WorldStereo clips.

prep        read each result clip, write source frames, bicubic 2x, and an mp4 copy for SeedVR2
ultrasharp  4x-UltraSharp per frame, then area-down to 2x (GPU, main .venv)
score       metrics, contact sheets, report.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
SCENE = REPO / "outputs" / "alpine_02" / "hyworld"
CLIPS = SCENE / "render_results"
ROOT = SCENE / "_votion" / "seedvr2_test"
PATHS = ("view1/traj0", "view1/traj1", "view1/traj2", "wonder_0/traj0")
RESULT = "worldstereo-memory-dmd_result.mp4"
SRC_W, SRC_H = 832, 480
W, H = 1664, 960
FRAMES = 21
METHODS = ("bicubic", "ultrasharp", "seedvr2_3b", "seedvr2_7b")
CROPS = ((240, 592), (400, 1100))  # (y, x) of 480x480 crops in the 1664x960 frame


def tag(path: str) -> str:
    return path.replace("/", "_")


def read_clip(path: Path) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames


def write_frames(folder: Path, frames) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for index, frame in enumerate(frames):
        cv2.imwrite(str(folder / f"{index:03d}.png"), frame)


def read_frames(folder: Path) -> list[np.ndarray]:
    return [cv2.imread(str(f), cv2.IMREAD_COLOR) for f in sorted(folder.rglob("*.png"))]


def prep() -> None:
    inputs = ROOT / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    for path in PATHS:
        clip = CLIPS / path / RESULT
        frames = read_clip(clip)
        shape = frames[0].shape if frames else None
        if len(frames) != FRAMES or shape != (SRC_H, SRC_W, 3):
            raise SystemExit(f"{path}: {len(frames)} frames of {shape}, expected {FRAMES} of {(SRC_H, SRC_W, 3)}")
        write_frames(ROOT / "src" / tag(path), frames)
        write_frames(ROOT / "bicubic" / tag(path), [cv2.resize(f, (W, H), interpolation=cv2.INTER_CUBIC) for f in frames])
        (inputs / f"{tag(path)}.mp4").write_bytes(clip.read_bytes())
        print(f"prep {path} frames={len(frames)}")
    print("prep ok")


def ultrasharp() -> None:
    import torch
    from spandrel import ModelLoader

    device = torch.device("cuda")
    model = ModelLoader().load_from_file(str(REPO / "models" / "upscale_models" / "4x-UltraSharp.pth"))
    model.to(device).eval()
    torch.cuda.reset_peak_memory_stats(device)
    seconds = {}
    for path in PATHS:
        start = time.perf_counter()
        out = []
        for frame in read_frames(ROOT / "src" / tag(path)):
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tensor = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1)[None].to(device)
            with torch.no_grad():
                big = model(tensor)
            image = (big[0].permute(1, 2, 0).clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy()
            out.append(cv2.cvtColor(cv2.resize(image, (W, H), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2BGR))
        write_frames(ROOT / "ultrasharp" / tag(path), out)
        seconds[path] = round(time.perf_counter() - start, 1)
        print(f"ultrasharp {path} seconds={seconds[path]}")
    peak = torch.cuda.max_memory_allocated(device) / 1024**3
    (ROOT / "ultrasharp_timing.json").write_text(json.dumps({"seconds": seconds, "peak_gb": round(peak, 2)}, indent=2))
    print(f"ultrasharp ok peak_gb={peak:.2f}")


def detail(img: np.ndarray) -> np.ndarray:
    f = img.astype(np.float32)
    low = cv2.resize(cv2.resize(f, (SRC_W, SRC_H), interpolation=cv2.INTER_AREA), (W, H), interpolation=cv2.INTER_CUBIC)
    return f - low


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))
    return 99.0 if mse == 0 else 10.0 * float(np.log10(255.0**2 / mse))


def lab_shift(a: np.ndarray, b: np.ndarray) -> float:
    la = cv2.cvtColor(a, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32).mean(0)
    lb = cv2.cvtColor(b, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32).mean(0)
    return float(np.abs(la - lb).max())


def sharpness(img: np.ndarray) -> float:
    return float(cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32), cv2.CV_32F).var())


def flicker(src: list[np.ndarray], frames: list[np.ndarray]) -> tuple[float, float]:
    """Share of the added detail that does not follow the motion of the source frames."""
    ys, xs = np.mgrid[0:SRC_H, 0:SRC_W].astype(np.float32)
    big_y, big_x = np.mgrid[0:H, 0:W].astype(np.float32)
    border = np.zeros((H, W), bool)
    border[16:-16, 16:-16] = True
    err = mag = 0.0
    steps = 0
    for t in range(FRAMES - 1):
        a = cv2.cvtColor(src[t], cv2.COLOR_BGR2GRAY)
        b = cv2.cvtColor(src[t + 1], cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(b, a, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        back = cv2.remap(a, xs + flow[..., 0], ys + flow[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        valid = np.abs(back.astype(np.float32) - b.astype(np.float32)) < 8.0
        valid = cv2.resize(valid.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool) & border
        if not valid.any():
            continue
        up = cv2.resize(flow, (W, H), interpolation=cv2.INTER_LINEAR) * 2.0
        moved = cv2.remap(detail(frames[t]), big_x + up[..., 0], big_y + up[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        now = detail(frames[t + 1])
        err += float(np.abs(now - moved)[valid].mean())
        mag += float(np.abs(now)[valid].mean())
        steps += 1
    return err / max(mag, 1e-6), err / max(steps, 1)


def load_method(method: str) -> dict | None:
    out = {}
    for path in PATHS:
        frames = read_frames(ROOT / method / tag(path))
        if len(frames) != FRAMES:
            print(f"score skip {method}: {path} has {len(frames)} frames")
            return None
        if frames[0].shape[:2] != (H, W):
            print(f"score {method} {path} resized from {frames[0].shape[1]}x{frames[0].shape[0]}")
            frames = [cv2.resize(f, (W, H), interpolation=cv2.INTER_AREA) for f in frames]
        out[path] = frames
    return out


def cross(methods: dict) -> dict:
    view1 = [p for p in PATHS if p.startswith("view1/")]
    first = {p: cv2.imread(str(ROOT / "src" / tag(p) / "000.png")) for p in view1}
    out = {}
    for i, a in enumerate(view1):
        for b in view1[i + 1:]:
            gap = float(np.abs(first[a].astype(np.float32) - first[b].astype(np.float32)).mean())
            row = {"src_gap": round(gap, 3)}
            if gap < 2.0:
                for method, clips in methods.items():
                    da, db = detail(clips[a][0]), detail(clips[b][0])
                    den = 0.5 * (float(np.abs(da).mean()) + float(np.abs(db).mean()))
                    row[method] = round(float(np.abs(da - db).mean()) / max(den, 1e-6), 4)
            out[f"{a}|{b}"] = row
    return out


def label(tile: np.ndarray, name: str, size: float = 0.7) -> np.ndarray:
    tile = tile.copy()
    origin = (int(12 * size / 0.7), int(34 * size / 0.7))
    thick = max(2, int(round(3 * size / 0.7)))
    cv2.putText(tile, name, origin, cv2.FONT_HERSHEY_SIMPLEX, size, (0, 0, 0), thick + 3, cv2.LINE_AA)
    cv2.putText(tile, name, origin, cv2.FONT_HERSHEY_SIMPLEX, size, (0, 255, 255), thick, cv2.LINE_AA)
    return tile


def sheet(path: str, methods: dict) -> None:
    src = cv2.imread(str(ROOT / "src" / tag(path) / "010.png"))
    rows = []
    for y, x in CROPS:
        small = src[y // 2:y // 2 + 240, x // 2:x // 2 + 240]
        tiles = [label(cv2.resize(small, (480, 480), interpolation=cv2.INTER_NEAREST), "source")]
        for method, clips in methods.items():
            tiles.append(label(clips[path][10][y:y + 480, x:x + 480], method))
        rows.append(cv2.hconcat(tiles))
    out = ROOT / "sheets" / f"{tag(path)}_f10.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), cv2.vconcat(rows))


def side_by_side(path: str, methods: dict, frame: int) -> None:
    """Whole frames in a 3-column grid: source (nearest 2x) then each method, all 1664x960."""
    src = cv2.imread(str(ROOT / "src" / tag(path) / f"{frame:03d}.png"))
    cells = [label(cv2.resize(src, (W, H), interpolation=cv2.INTER_NEAREST), "source 832x480", 2.0)]
    for method, clips in methods.items():
        cells.append(label(clips[path][frame], method, 2.0))
    while len(cells) % 3:
        cells.append(np.zeros((H, W, 3), np.uint8))
    rows = [cv2.hconcat(cells[i:i + 3]) for i in range(0, len(cells), 3)]
    out = ROOT / "compare" / f"{tag(path)}_f{frame:02d}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), cv2.vconcat(rows))
    small = cv2.resize(cv2.vconcat(rows), None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(out.with_name(out.stem + "_half.jpg")), small, [cv2.IMWRITE_JPEG_QUALITY, 92])


def score() -> None:
    src = {p: read_frames(ROOT / "src" / tag(p)) for p in PATHS}
    methods = {}
    for method in METHODS:
        clips = load_method(method)
        if clips is not None:
            methods[method] = clips
    report = {"paths": list(PATHS), "methods": {}}
    for method, clips in methods.items():
        rows = {}
        for path in PATHS:
            backs = [cv2.resize(f, (SRC_W, SRC_H), interpolation=cv2.INTER_AREA) for f in clips[path]]
            row = {
                "psnr_back": round(float(np.mean([psnr(b, s) for b, s in zip(backs, src[path])])), 2),
                "lab_shift": round(float(np.max([lab_shift(b, s) for b, s in zip(backs, src[path])])), 2),
                "sharpness": round(float(np.mean([sharpness(f) for f in clips[path]])), 1),
            }
            if method != "bicubic":
                ratio, err = flicker(src[path], clips[path])
                row["flicker"] = round(ratio, 4)
                row["flicker_err"] = round(err, 3)
            rows[path] = row
        mean = {k: round(float(np.mean([r[k] for r in rows.values()])), 4) for k in rows[PATHS[0]]}
        report["methods"][method] = {"paths": rows, "mean": mean}
        print("score " + method + " " + " ".join(f"{k}={v}" for k, v in mean.items()))
    report["cross_path"] = cross(methods)
    for key, row in report["cross_path"].items():
        print(f"cross {key} {row}")
    for path in PATHS:
        sheet(path, methods)
        for frame in (0, 10, 20):
            side_by_side(path, methods, frame)
    (ROOT / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("score ok")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=("prep", "ultrasharp", "score"))
    args = parser.parse_args()
    {"prep": prep, "ultrasharp": ultrasharp, "score": score}[args.step]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
