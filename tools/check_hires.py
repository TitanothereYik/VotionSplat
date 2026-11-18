"""CPU checks for HiRes fit, publish, and frame comparison. No GPU."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import cv2
import numpy as np

from engine.hyworld import hires as HR


def check(cond: bool, msg: str) -> None:
    if not cond:
        raise SystemExit(msg)


def check_fit() -> None:
    saved = HR.FIT_FILE
    handle, name = tempfile.mkstemp(suffix=".json")
    os.close(handle)
    HR.FIT_FILE = Path(name)
    try:
        check(HR.pick(22.0, {}, "auto") == "t1024", "22 GB should pick t1024")
        check(HR.pick(15.0, {}, "auto") == "t768", "15 GB should pick t768")
        check(HR.pick(5.0, {}, "auto") == "t512", "5 GB should pick t512")
        check(HR.pick(30.0, {}, "auto") == "whole", "30 GB should pick whole")
        check(HR.pick(22.0, {}, "t768") == "t768", "an asked level stays")
        rows = HR.fail({}, "t1024", 22.0)
        check(HR.pick(22.0, rows, "auto") == "t768", "a failed t1024 steps to t768")
        rows = HR.record({}, "t1024", 40.0, 16.0, 17.0)
        check(HR.need_gb("t1024", rows) == 17.25, f"need was {HR.need_gb('t1024', rows)}")
        check(HR.next_level("t512") is None, "t512 has no next level")
        HR.write_rows("gpu|24", rows)
        got = HR.read_rows("gpu|24")
        check(got.get("t1024", {}).get("reserved_gb") == 17.0, f"round trip was {got}")

        check(HR.pick(21.98, {}, "auto", 4) == "b13", "4x at 22 GB should start at b13")
        check(HR.pick(12.0, {}, "auto", 4) == "b5", "4x at 12 GB should pick b5")
        check(HR.next_level("b13", 4) == "b9", "b13 steps down to b9")
        check(HR.next_level("b5", 4) is None, "b5 has no next level")
        check(HR.batch("b9", 4) == (9, 3) and HR.batch("t1024") == (21, 0), "batch per level")
        for size, overlap in HR.BATCH_X4.values():
            check((21 - size) % (size - overlap) == 0, f"batch {size}/{overlap} leaves a short batch")
        rows = HR.record({}, "b13", 200.0, 18.1, 22.25, 21.98)
        check(HR.need_gb("b13", rows, 4) == 21.98, f"ok_gb should bound need, got {HR.need_gb('b13', rows, 4)}")
        check(HR.pick(21.98, rows, "auto", 4) == "b13", "a passed b13 stays at the same budget")
        check(HR.fit_key("gpu", 24.0) == "gpu|24" and HR.fit_key("gpu", 24.0, 4) == "gpu|24|x4", "fit keys")
        try:
            HR.pick(22.0, {}, "t1024", 4)
        except ValueError:
            pass
        else:
            raise SystemExit("a 2x level must be refused at 4x")
    finally:
        HR.FIT_FILE = saved
        Path(name).unlink(missing_ok=True)


def _pngs(folder: Path, count: int = 21) -> list[Path]:
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for index in range(count):
        image = np.zeros((8, 12, 3), np.uint8)
        image[:, :] = (index * 8, 40, 80)
        path = folder / f"in_{index:03d}.png"
        cv2.imwrite(str(path), image)
        paths.append(path)
    return paths


def check_publish() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        source = root / "source.mp4"
        source.write_bytes(b"source-v1")
        folder = root / "view1" / "traj0"
        meta = {"frames": 21, **HR.stamp(source), "level": "t1024"}
        HR.publish(folder, _pngs(root / "in"), meta)
        check(HR.folder_state(folder, source) == "done", "publish should be done")
        check(len(list((folder / "frames").glob("*.png"))) == 21, "21 frames")
        check((folder / "preview.mp4").is_file(), "preview missing")
        source.write_bytes(b"source-v2-changed")
        check(HR.folder_state(folder, source) == "stale", "changed source should be stale")
        meta = {"frames": 21, **HR.stamp(source), "level": "t1024"}
        HR.publish(folder, _pngs(root / "in2"), meta)
        check(HR.folder_state(folder, source) == "done", "republish should be done")
        check(not (folder.parent / "traj0.new").exists(), ".new left behind")
        check(not (folder.parent / "traj0.old").exists(), ".old left behind")
        check(source.is_file(), "source was removed")

        real = HR.hires_root
        HR.hires_root = lambda _scene, scale=2: root / ("hires" if scale == 2 else f"hires_x{scale}")
        try:
            scratch = root / "hires"
            (scratch / "_tmp").mkdir(parents=True)
            (scratch / "_tmp" / "x.txt").write_text("a", encoding="utf-8")
            (scratch / "view1" / "traj0.new").mkdir(parents=True)
            outside = root / "keep.new"
            outside.mkdir()
            (outside / "no.txt").write_text("c", encoding="utf-8")
            HR.clean_scratch("unused")
            check(not (scratch / "_tmp").exists(), "scratch _tmp should go")
            check(not (scratch / "view1" / "traj0.new").exists(), "scratch .new should go")
            check((outside / "no.txt").is_file(), "clean_scratch left the temp root")
            (root / "hires_x4" / "_tmp").mkdir(parents=True)
            HR.clean_scratch("unused", 4)
            check(not (root / "hires_x4" / "_tmp").exists(), "4x scratch _tmp should go")
        finally:
            HR.hires_root = real
        check(HR.hires_root("s", 4).name == "hires_x4" and HR.hires_root("s").name == "hires", "roots")

        big = []
        for index in range(21):
            path = root / "big" / f"{index:03d}.png"
            path.parent.mkdir(exist_ok=True)
            cv2.imwrite(str(path), np.full((1920, 3328, 3), index * 10, np.uint8))
            big.append(path)
        HR.publish(root / "x4" / "traj0", big, {"frames": 21, **HR.stamp(source)})
        cap = cv2.VideoCapture(str(root / "x4" / "traj0" / "preview.mp4"))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        cap.release()
        check(width == HR.PREVIEW_LONG_SIDE, f"4x preview width {width}")
        first = cv2.imread(str(root / "x4" / "traj0" / "frames" / "000.png"))
        check(first.shape == (1920, 3328, 3), "4x frames stay full size")


def compare(left: str, right: str) -> None:
    names = sorted(path.name for path in Path(left).glob("*.png"))
    scores = []
    for name in names:
        other = Path(right) / name
        if not other.is_file():
            continue
        a = cv2.imread(str(Path(left) / name), cv2.IMREAD_COLOR)
        b = cv2.imread(str(other), cv2.IMREAD_COLOR)
        if a is None or b is None:
            continue
        if b.shape[:2] != a.shape[:2]:
            b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA)
        mse = float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))
        scores.append(99.0 if mse == 0 else 10.0 * float(np.log10(255.0**2 / mse)))
    mean = sum(scores) / len(scores) if scores else 0.0
    print(f"psnr={mean:.2f} frames={len(scores)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compare", nargs=2, metavar=("A", "B"))
    args = parser.parse_args()
    if args.compare:
        compare(*args.compare)
        return 0
    check_fit()
    check_publish()
    print("check_hires ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
