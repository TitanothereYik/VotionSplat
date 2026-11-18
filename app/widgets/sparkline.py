"""Loss and PSNR sparkline for the Splat page."""

from __future__ import annotations

import math
from collections import deque

from PySide6.QtCore import QPointF, QRectF, QSize, Qt
from PySide6.QtGui import QColor, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QWidget


class Sparkline(QWidget):
    """Loss EMA (amber) and PSNR (blue) over the last reports."""

    def __init__(self, parent: QWidget | None = None, keep: int = 400) -> None:
        super().__init__(parent)
        self.setMinimumHeight(56)
        self.setMaximumHeight(72)
        self._loss: deque[float] = deque(maxlen=keep)
        self._psnr: deque[float] = deque(maxlen=keep)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, True)

    def sizeHint(self) -> QSize:
        return QSize(400, 60)

    def clear(self) -> None:
        self._loss.clear()
        self._psnr.clear()
        self.update()

    def add(self, loss: float | None, psnr: float | None) -> None:
        if loss is not None and math.isfinite(loss):
            self._loss.append(float(loss))
        if psnr is not None and math.isfinite(psnr):
            self._psnr.append(float(psnr))
        self.update()

    def _poly(self, series: deque[float], rect: QRectF, lo: float, hi: float) -> QPolygonF:
        poly = QPolygonF()
        n = len(series)
        if n < 2:
            return poly
        span = max(hi - lo, 1e-9)
        for i, v in enumerate(series):
            x = rect.left() + rect.width() * i / (n - 1)
            y = rect.bottom() - rect.height() * (v - lo) / span
            poly.append(QPointF(x, y))
        return poly

    def paintEvent(self, event) -> None:  # noqa: ANN001
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(8, 16, 24))
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        rect = QRectF(self.rect()).adjusted(6, 6, -6, -6)
        p.setPen(QColor(139, 155, 180))
        font = p.font()
        font.setPointSize(8)
        p.setFont(font)
        if not self._loss and not self._psnr:
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "loss · PSNR")
            return
        if len(self._loss) >= 2:
            lo, hi = min(self._loss), max(self._loss)
            p.setPen(QPen(QColor(255, 159, 48), 1.5))
            p.drawPolyline(self._poly(self._loss, rect, lo, hi))
            p.setPen(QColor(255, 159, 48))
            p.drawText(int(rect.left()), int(rect.top()) + 10, f"loss {self._loss[-1]:.4f}")
        if len(self._psnr) >= 2:
            lo, hi = min(self._psnr), max(self._psnr)
            p.setPen(QPen(QColor(77, 163, 255), 1.5))
            p.drawPolyline(self._poly(self._psnr, rect, lo - 0.5, hi + 0.5))
            p.setPen(QColor(77, 163, 255))
            text = f"PSNR {self._psnr[-1]:.2f} dB"
            p.drawText(int(rect.right()) - p.fontMetrics().horizontalAdvance(text), int(rect.top()) + 10, text)
