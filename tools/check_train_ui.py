"""Offscreen check that the Splat page asks for Train and Continue. No GPU."""

from __future__ import annotations

import os
import sys

os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PySide6.QtWidgets import QApplication

from app.windows.hy_splat_page import HySplatPage
from engine.hyworld.argv import train_loop_argv


def check(cond: bool, msg: str) -> None:
    if not cond:
        raise SystemExit(msg)


def main() -> int:
    app = QApplication.instance() or QApplication([])
    page = HySplatPage(lambda: "alpine_02")
    got: list = []
    page.requestJob.connect(got.append)
    page.difix.setEnabled(True)
    page.difix.setChecked(True)
    page.btn_train.setEnabled(True)
    page.btn_train.click()
    check(len(got) == 1 and len(got[0]) == 5, f"train payload {got}")
    check(got[0][:2] == ["__hy_train__", "alpine_02"] and got[0][4] == "1", f"train payload {got}")
    page.btn_continue.setEnabled(True)
    page.btn_continue.click()
    check(got[1] == ["__hy_train_continue__", "alpine_02"], f"continue payload {got}")
    page.btn_continue.setEnabled(False)
    page.refresh()
    check(not page.btn_continue.isEnabled(), "continue should stay off without a resumable run")
    check("torch" not in sys.modules, "torch was imported into the Qt process")
    fresh = train_loop_argv("alpine_02", steps=8000, mesh=True, difix=True)
    check(fresh[-4:] == ["--steps", "8000", "--mesh", "--difix"], str(fresh))
    cont = train_loop_argv("alpine_02", cont=True)
    check(cont[-1] == "--continue" and "--steps" not in cont, str(cont))
    del app
    print("check_train_ui ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
