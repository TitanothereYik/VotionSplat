"""Launch the native Votion 3DGS window. The Qt process does not import torch."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    from PySide6.QtGui import QFont
    from PySide6.QtWidgets import QApplication

    from app.windows.main_window import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName("Votion 3DGS")
    app.setApplicationVersion("2.0")
    app.setOrganizationName("Yik")
    app.setFont(QFont("JetBrains Mono", 10))

    qss_path = Path(__file__).with_name("theme.qss")
    app.setStyleSheet(qss_path.read_text(encoding="utf-8"))

    win = MainWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
