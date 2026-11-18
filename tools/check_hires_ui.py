"""Offscreen check that the Expand page asks for HiRes and shows its lines. No GPU."""

from __future__ import annotations

import os
import sys

os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PySide6.QtWidgets import QApplication

from app.windows.expand_page import ExpandPage
from engine.hyworld.argv import hires_argv
from engine.hyworld.hires import PYTHON


def check(cond: bool, msg: str) -> None:
    if not cond:
        raise SystemExit(msg)


def main() -> int:
    app = QApplication.instance() or QApplication([])
    page = ExpandPage(lambda: "alpine_02")
    got: list = []
    page.requestJob.connect(got.append)
    page.btn_hires.setEnabled(True)
    page.btn_hires.click()
    check(got == [["__hy_hires__", "alpine_02", "", "0", "auto", "2"]], f"click payload {got}")

    page.hires_scale.setCurrentIndex(page.hires_scale.findData(4))
    levels = [page.hires_level.itemData(index) for index in range(page.hires_level.count())]
    check(levels == ["auto", "b17", "b13", "b9", "b5"], f"4x memory choices {levels}")
    check(page.hr_result._title == "HiRes 4x", page.hr_result._title)
    page.hires_level.setCurrentIndex(page.hires_level.findData("b9"))
    got.clear()
    page.btn_hires.setEnabled(True)
    page.btn_hires.click()
    check(got == [["__hy_hires__", "alpine_02", "", "0", "b9", "4"]], f"4x click payload {got}")
    page.hires_scale.setCurrentIndex(page.hires_scale.findData(2))
    check(page.hires_level.itemData(1) == "whole", "2x memory choices restored")

    page.on_job_started("hy_hires")
    page.on_hy_line(["HY", "hqueue", '["view1/traj0"]'])
    page.on_hy_line(["HY", "hires", "view1/traj0", "upscale", "0", "0"])
    page.on_hy_line(["HY", "hires", "view1/traj0", "done", "41.2", "20.10"])
    page.on_hy_line(["HY", "hfit", '{"level": "t1024", "budget_gb": 22.4, "need_gb": 21.5, "measured": false}'])
    texts = [page.rows.item(index).text() for index in range(page.rows.count())]
    matched = [text for text in texts if "view1/traj0" in text and "+HR 41.2 s" in text]
    check(bool(matched), f"row text {texts}")
    check("Decode tiles 1024" in page.hires_fit.text(), page.hires_fit.text())
    check("torch" not in sys.modules, "torch was imported into the Qt process")

    argv = hires_argv("alpine_02", only="view1/traj0", resume=False, level="t768")
    check(argv[0] == str(PYTHON), argv[0])
    check(argv[-5:] == ["--only", "view1/traj0", "--restart", "--level", "t768"], str(argv))
    argv = hires_argv("alpine_02", level="b9", scale=4)
    check(argv[-4:] == ["--level", "b9", "--scale", "4"], str(argv))
    for bad in ({"level": "t768", "scale": 4}, {"level": "b9", "scale": 2}, {"scale": 3}):
        try:
            hires_argv("alpine_02", **bad)
        except ValueError:
            continue
        raise SystemExit(f"hires_argv accepted {bad}")
    del app
    print("check_hires_ui ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
