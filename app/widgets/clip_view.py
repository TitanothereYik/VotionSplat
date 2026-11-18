"""Media pane for the HY pages: loops a short mp4 or shows a still. Qt process; cv2 and numpy only."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QRect, Qt, QTimer
from PySide6.QtGui import QColor, QImage, QPainter
from PySide6.QtWidgets import QSizePolicy, QWidget

_BG = QColor(8, 16, 24)
_TITLE = QColor(77, 163, 255)
_DIM = QColor(139, 155, 180)
_MAX_SIDE = 1600
_MAX_FRAMES = 96


def _shrink(rgb: np.ndarray) -> np.ndarray:
    h, w = rgb.shape[:2]
    side = max(h, w)
    if side > _MAX_SIDE:
        k = _MAX_SIDE / side
        rgb = cv2.resize(rgb, (max(1, int(w * k)), max(1, int(h * k))), interpolation=cv2.INTER_AREA)
    return rgb


def _qimage(rgb: np.ndarray) -> QImage:
    rgb = np.ascontiguousarray(rgb[..., :3], dtype=np.uint8)
    h, w = rgb.shape[:2]
    return QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()


def _read_image(path: Path) -> list[QImage]:
    try:
        raw = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return []
    bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR) if raw.size else None
    if bgr is None:
        return []
    return [_qimage(_shrink(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))]


def _read_video(path: Path) -> list[QImage]:
    cap = cv2.VideoCapture(str(path))
    frames: list[QImage] = []
    try:
        while len(frames) < _MAX_FRAMES:
            ok, bgr = cap.read()
            if not ok:
                break
            frames.append(_qimage(_shrink(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))))
    finally:
        cap.release()
    return frames


class ClipView(QWidget):
    def __init__(self, title: str = "", empty: str = "", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(160, 100)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, True)
        self._title = title
        self._empty = empty
        self._frames: list[QImage] = []
        self._index = 0
        self._stamp: tuple | None = None
        self._timer = QTimer(self)
        self._timer.setInterval(62)
        self._timer.timeout.connect(self._advance)

    def set_title(self, text: str) -> None:
        self._title = text
        self.update()

    def set_empty(self, text: str) -> None:
        self._empty = text
        self.update()

    def clear(self) -> None:
        self._frames = []
        self._stamp = None
        self._timer.stop()
        self.update()

    def set_array(self, rgb: np.ndarray | None) -> None:
        if rgb is None:
            self.clear()
            return
        self._frames = [_qimage(_shrink(rgb))]
        self._stamp = None
        self._index = 0
        self._timer.stop()
        self.update()

    def set_path(self, path: Path | str | None) -> None:
        """Reloads only when the file changed. A file that is still being written keeps the old picture."""
        if path is None or not Path(path).is_file():
            self.clear()
            return
        path = Path(path)
        stat = path.stat()
        stamp = (str(path), stat.st_mtime_ns, stat.st_size)
        if stamp == self._stamp:
            return
        frames = _read_video(path) if path.suffix.lower() == ".mp4" else _read_image(path)
        if not frames:
            return
        self._frames = frames
        self._stamp = stamp
        self._index = 0
        if len(frames) > 1 and self.isVisible():
            self._timer.start()
        else:
            self._timer.stop()
        self.update()

    def _advance(self) -> None:
        if len(self._frames) > 1:
            self._index = (self._index + 1) % len(self._frames)
            self.update()

    def showEvent(self, event) -> None:  # noqa: N802
        if len(self._frames) > 1:
            self._timer.start()
        super().showEvent(event)

    def hideEvent(self, event) -> None:  # noqa: N802
        self._timer.stop()
        super().hideEvent(event)

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.fillRect(self.rect(), _BG)
        if self._frames:
            img = self._frames[min(self._index, len(self._frames) - 1)]
            k = min(self.width() / img.width(), self.height() / img.height())
            w, h = int(img.width() * k), int(img.height() * k)
            p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
            p.drawImage(QRect((self.width() - w) // 2, (self.height() - h) // 2, w, h), img)
        elif self._empty:
            p.setPen(_DIM)
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap, self._empty)
        if self._title:
            p.setPen(_TITLE)
            p.drawText(8, 16, self._title)
