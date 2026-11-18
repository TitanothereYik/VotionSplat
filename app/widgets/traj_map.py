"""Top-down plot of the planned camera paths. Qt process; numpy only."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QSizePolicy, QWidget

_BG = QColor(8, 16, 24)
_OFF = QColor("#3a4a5e")
_DIM = QColor("#8b9bb4")
_FAMILY = (("view", QColor("#4da3ff"), "origin views"), ("wonder", QColor("#ff9f30"), "wonder"),
           ("target", QColor("#00ff9d"), "target"), ("reconstruct", QColor("#f0f0f0"), "reconstruct"))


def _color(item: str) -> QColor:
    for prefix, color, _label in _FAMILY:
        if item.startswith(prefix):
            return color
    return QColor("#f0f0f0")


class TrajMapView(QWidget):
    picked = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(240, 200)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, True)
        self._paths: dict[str, np.ndarray] = {}
        self._checked: set[str] = set()
        self._selected = ""
        self._screen: dict[str, np.ndarray] = {}

    def set_paths(self, paths: dict, checked, selected: str = "") -> None:
        self._paths = {k: np.asarray(v, dtype=np.float64).reshape(-1, 3) for k, v in paths.items() if len(v)}
        self._checked = set(checked)
        self._selected = selected
        self.update()

    def set_selected(self, item: str) -> None:
        self._selected = item
        self.update()

    def _layout(self):
        self._screen = {}
        if not self._paths:
            return None
        pts = np.concatenate(list(self._paths.values()) + [np.zeros((1, 3))])
        u, v = pts[:, 1], pts[:, 0]
        cu, cv = (u.min() + u.max()) / 2.0, (v.min() + v.max()) / 2.0
        span = max(float(u.max() - u.min()), float(v.max() - v.min()), 1e-6) * 1.2
        scale = min(self.width(), self.height()) / span

        def to_xy(p: np.ndarray) -> np.ndarray:
            return np.stack([self.width() / 2.0 + (p[:, 1] - cu) * scale,
                             self.height() / 2.0 + (p[:, 0] - cv) * scale], axis=1)

        for key, p in self._paths.items():
            self._screen[key] = to_xy(p)
        return to_xy(np.zeros((1, 3)))[0]

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.fillRect(self.rect(), _BG)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        origin = self._layout()
        if origin is None:
            p.setPen(_DIM)
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Plan trajectories to see the paths.")
            return
        order = [k for k in self._screen if k != self._selected] + ([self._selected] if self._selected in self._screen else [])
        for key in order:
            xy = self._screen[key]
            color = _color(key) if key in self._checked else _OFF
            p.setPen(QPen(color, 3.0 if key == self._selected else 1.5))
            p.drawPolyline(QPolygonF([QPointF(float(x), float(y)) for x, y in xy]))
            p.setBrush(color)
            p.drawEllipse(QPointF(float(xy[0, 0]), float(xy[0, 1])), 3.0, 3.0)
        p.setPen(QPen(QColor("#f0f0f0"), 1.5))
        ox, oy = float(origin[0]), float(origin[1])
        p.drawLine(QPointF(ox - 7, oy), QPointF(ox + 7, oy))
        p.drawLine(QPointF(ox, oy - 7), QPointF(ox, oy + 7))
        y = self.height() - 12
        x = 10
        for _prefix, color, label in _FAMILY:
            p.setPen(color)
            p.drawText(x, y, label)
            x += p.fontMetrics().horizontalAdvance(label) + 16

    def mousePressEvent(self, event) -> None:  # noqa: N802
        pos = event.position()
        best, best_d = "", 12.0
        for key, xy in self._screen.items():
            d = float(np.min(np.hypot(xy[:, 0] - pos.x(), xy[:, 1] - pos.y())))
            if d < best_d:
                best, best_d = key, d
        if best:
            self.picked.emit(best)
