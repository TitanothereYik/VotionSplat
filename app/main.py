"""Launch the VotionSplat window. See docs/USER_GUIDE.md. The Qt process stays free of torch."""

from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    from app.version import APP_ID, APP_NAME, APP_VERSION, ORG_NAME
    from engine.job import apply_huggingface_cache_env

    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(APP_ID)
    apply_huggingface_cache_env(os.environ)

    from PySide6.QtGui import QFont, QIcon
    from PySide6.QtWidgets import QApplication

    from app.windows.main_window import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setApplicationVersion(APP_VERSION)
    app.setOrganizationName(ORG_NAME)
    app.setWindowIcon(QIcon(str(Path(__file__).resolve().parent / "assets" / "icon.ico")))
    app.setFont(QFont("JetBrains Mono", 10))

    qss_path = Path(__file__).with_name("theme.qss")
    app.setStyleSheet(qss_path.read_text(encoding="utf-8"))

    win = MainWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
