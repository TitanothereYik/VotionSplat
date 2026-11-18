"""Shared pieces for the four HY-World pages. Qt only. Every HY page is built from these."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QButtonGroup, QHBoxLayout, QLabel, QListWidget, QPushButton, QVBoxLayout, QWidget

BLUE, ORANGE, WHITE, DIM, GREEN, RED = "#4da3ff", "#ff9f30", "#f0f0f0", "#8b9bb4", "#00ff9d", "#ff4d6d"


def family_color(item: str) -> str:
    view = item.split("/", 1)[0]
    if view.startswith("view"):
        return BLUE
    if view.startswith("wonder"):
        return ORANGE
    if view.startswith("target"):
        return GREEN
    return WHITE


def lab(text: str) -> QLabel:
    out = QLabel(text)
    out.setProperty("class", "fLab")
    return out


def hint(text: str = "") -> QLabel:
    out = QLabel(text)
    out.setWordWrap(True)
    out.setProperty("class", "fHint")
    return out


def view_label(text: str) -> QLabel:
    out = QLabel(text)
    out.setProperty("class", "viewLbl")
    return out


def button(text: str, primary: bool = False) -> QPushButton:
    out = QPushButton(text)
    out.setObjectName("btnPrimary" if primary else "btnGhost")
    return out


def rail(width: int = 290) -> tuple[QWidget, QVBoxLayout]:
    box = QWidget()
    box.setObjectName("rail")
    box.setFixedWidth(width)
    lay = QVBoxLayout(box)
    lay.setContentsMargins(12, 12, 12, 14)
    lay.setSpacing(4)
    return box, lay


def viewport() -> tuple[QWidget, QVBoxLayout]:
    box = QWidget()
    box.setObjectName("viewport")
    lay = QVBoxLayout(box)
    lay.setContentsMargins(8, 8, 8, 8)
    lay.setSpacing(6)
    return box, lay


def chips(names: list[str]) -> tuple[QWidget, QButtonGroup]:
    box = QWidget()
    lay = QHBoxLayout(box)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(0)
    group = QButtonGroup(box)
    group.setExclusive(True)
    for index, name in enumerate(names):
        btn = QPushButton(name)
        btn.setCheckable(True)
        btn.setProperty("class", "modeChip")
        group.addButton(btn, index)
        lay.addWidget(btn)
    group.button(0).setChecked(True)
    lay.addStretch(1)
    return box, group


def side_list() -> QListWidget:
    out = QListWidget()
    out.setObjectName("camList")
    out.setUniformItemSizes(True)
    out.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
    pal = out.palette()
    pal.setColor(QPalette.ColorRole.Base, QColor(8, 16, 24))
    pal.setColor(QPalette.ColorRole.AlternateBase, QColor(8, 16, 24))
    pal.setColor(QPalette.ColorRole.Text, QColor(201, 212, 227))
    pal.setColor(QPalette.ColorRole.Highlight, QColor(255, 159, 48))
    pal.setColor(QPalette.ColorRole.HighlightedText, QColor(11, 22, 35))
    out.setPalette(pal)
    out.viewport().setPalette(pal)
    out.viewport().setAutoFillBackground(True)
    return out


class HyPage(QWidget):
    """Contract with MainWindow. Pages never set ``_busy`` themselves."""

    requestJob = Signal(list)
    requestStop = Signal()
    LABELS: tuple[str, ...] = ()

    def __init__(self, scene_fn) -> None:
        super().__init__()
        self._scene = scene_fn
        self._busy = False
        self._label = ""

    def on_job_started(self, label: str) -> None:
        self._busy = True
        self._label = label
        self.refresh()

    def on_job_finished(self, label: str, code: int) -> None:
        del label, code
        self._busy = False
        self._label = ""
        self.refresh()

    def on_progress(self, label: str, current: int, total: int, pct: int, msg: str) -> None:
        del label, current, total, pct, msg

    def on_hy_line(self, parts: list[str]) -> None:
        del parts

    def refresh(self) -> None:
        return

    def restore_view(self) -> None:
        self.refresh()
