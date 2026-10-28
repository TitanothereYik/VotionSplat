"""Panorama viewport widgets. No torch. No QWebEngine."""

from __future__ import annotations

import math
from pathlib import Path

from PIL import Image
from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QColor, QImage, QMouseEvent, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QSizePolicy, QWidget

from engine.pano.defaults import GENERATE_HEIGHT, GENERATE_WIDTH, H_FOV_DEG_DEFAULT
from engine.pano.warp_preview import (
    LOOK_ERP_MAX_WIDTH,
    LOOK_HFOV_DEG,
    LOOK_PITCH_MAX_DEG,
    erp_to_rectilinear,
    warp_photo_to_green_erp,
)


def pil_to_pixmap(im: Image.Image) -> QPixmap:
    rgb = im.convert("RGB")
    data = rgb.tobytes("raw", "RGB")
    qim = QImage(data, rgb.width, rgb.height, rgb.width * 3, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qim.copy())


def _paint_raster(painter: QPainter, rect, pix: QPixmap, fit: bool, pan: QPoint) -> None:
    if fit:
        scaled = pix.scaled(
            rect.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        x = rect.x() + (rect.width() - scaled.width()) // 2
        y = rect.y() + (rect.height() - scaled.height()) // 2
        painter.drawPixmap(x, y, scaled)
        return
    x = rect.x() - pan.x()
    y = rect.y() - pan.y()
    if pix.width() < rect.width():
        x = rect.x() + (rect.width() - pix.width()) // 2
    if pix.height() < rect.height():
        y = rect.y() + (rect.height() - pix.height()) // 2
    painter.drawPixmap(x, y, pix)


class ErpStill(QWidget):
    """Unwrapped 2:1 ERP. Fit = whole image in the pane. 100% = native pixels, drag to pan."""

    def __init__(self, parent: QWidget | None = None, empty_text: str = "2:1 ERP") -> None:
        super().__init__(parent)
        self.setMinimumHeight(140)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setStyleSheet("background: #081018; border: 1px solid #1c2b3a;")
        self._empty_text = empty_text
        self._pix: QPixmap | None = None
        self._fit = True
        self._pan = QPoint(0, 0)
        self._drag: QPoint | None = None

    def set_fit(self, fit: bool) -> None:
        self._fit = bool(fit)
        if self._fit:
            self._pan = QPoint(0, 0)
            self._drag = None
            self.setCursor(Qt.CursorShape.ArrowCursor)
        else:
            self.setCursor(Qt.CursorShape.OpenHandCursor)
            self._clamp_pan()
        self.update()

    def set_pil(self, im: Image.Image | None) -> None:
        if im is None:
            self._pix = None
            self._pan = QPoint(0, 0)
            self.update()
            return
        self._pix = pil_to_pixmap(im)
        self._clamp_pan()
        self.update()

    def set_path(self, path: Path | None) -> None:
        if path is None or not path.is_file():
            self.set_pil(None)
            return
        try:
            with Image.open(path) as im:
                im.load()
                self.set_pil(im.convert("RGB"))
        except Exception:
            return

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._clamp_pan()
        self.update()

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if self._fit or self._pix is None:
            return
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag = event.position().toPoint()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        self._drag = None
        if not self._fit and self._pix is not None:
            self.setCursor(Qt.CursorShape.OpenHandCursor)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if self._fit or self._drag is None or self._pix is None:
            return
        pos = event.position().toPoint()
        self._pan.setX(self._pan.x() - (pos.x() - self._drag.x()))
        self._pan.setY(self._pan.y() - (pos.y() - self._drag.y()))
        self._drag = pos
        self._clamp_pan()
        self.update()

    def _clamp_pan(self) -> None:
        if self._pix is None or self._pix.isNull() or self._fit:
            self._pan = QPoint(0, 0)
            return
        max_x = max(0, self._pix.width() - self.width())
        max_y = max(0, self._pix.height() - self.height())
        self._pan.setX(min(max(self._pan.x(), 0), max_x))
        self._pan.setY(min(max(self._pan.y(), 0), max_y))

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#081018"))
        if self._pix is None or self._pix.isNull():
            painter.setPen(QColor("#8b9bb4"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self._empty_text)
            return
        _paint_raster(painter, self.rect(), self._pix, self._fit, self._pan)


class WarpPreview(QWidget):
    """Green 2:1 ERP warp + dummy `+` crosshair (fov-cross at 50% / 16%)."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(140)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setStyleSheet("background: #00ff00; border: 1px solid #1c2b3a;")
        self._photo: Image.Image | None = None
        self._hfov = H_FOV_DEG_DEFAULT
        self._pix: QPixmap | None = None
        self._fit = True
        self._pan = QPoint(0, 0)
        self._drag: QPoint | None = None

    def set_fit(self, fit: bool) -> None:
        self._fit = bool(fit)
        if self._fit:
            self._pan = QPoint(0, 0)
            self._drag = None
            self.setCursor(Qt.CursorShape.ArrowCursor)
        else:
            self.setCursor(Qt.CursorShape.OpenHandCursor)
        self.refresh()
        self.updateGeometry()

    def set_photo(self, im: Image.Image | None) -> None:
        self._photo = im.convert("RGB") if im is not None else None
        self.refresh()

    def set_hfov(self, value: float) -> None:
        self._hfov = float(value)
        self.refresh()

    def refresh(self) -> None:
        if self._photo is None:
            self._pix = None
            self.update()
            return
        if self._fit:
            preview_w, preview_h = 512, 256
        else:
            preview_w, preview_h = GENERATE_WIDTH, GENERATE_HEIGHT
        warped = warp_photo_to_green_erp(
            self._photo,
            h_fov_deg=self._hfov,
            width=preview_w,
            height=preview_h,
        )
        self._pix = pil_to_pixmap(warped)
        self._clamp_pan()
        self.update()

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._clamp_pan()
        self.update()

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if self._fit or self._pix is None:
            return
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag = event.position().toPoint()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        self._drag = None
        if not self._fit and self._pix is not None:
            self.setCursor(Qt.CursorShape.OpenHandCursor)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if self._fit or self._drag is None or self._pix is None:
            return
        pos = event.position().toPoint()
        self._pan.setX(self._pan.x() - (pos.x() - self._drag.x()))
        self._pan.setY(self._pan.y() - (pos.y() - self._drag.y()))
        self._drag = pos
        self._clamp_pan()
        self.update()

    def _clamp_pan(self) -> None:
        if self._pix is None or self._pix.isNull() or self._fit:
            self._pan = QPoint(0, 0)
            return
        max_x = max(0, self._pix.width() - self.width())
        max_y = max(0, self._pix.height() - self.height())
        self._pan.setX(min(max(self._pan.x(), 0), max_x))
        self._pan.setY(min(max(self._pan.y(), 0), max_y))

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#00ff00"))
        if self._pix is None or self._pix.isNull():
            painter.setPen(QColor("#111111"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "drop a 2D image")
            return
        _paint_raster(painter, self.rect(), self._pix, self._fit, self._pan)
        if self._fit:
            scaled = self._pix.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            x = (self.width() - scaled.width()) // 2
            y = (self.height() - scaled.height()) // 2
            cx = x + scaled.width() // 2
            cy = y + int(scaled.height() * 0.16)
        else:
            x = -self._pan.x()
            y = -self._pan.y()
            if self._pix.width() < self.width():
                x = (self.width() - self._pix.width()) // 2
            if self._pix.height() < self.height():
                y = (self.height() - self._pix.height()) // 2
            cx = x + self._pix.width() // 2
            cy = y + int(self._pix.height() * 0.16)
        painter.setPen(QPen(QColor("#111111"), 2))
        painter.drawLine(cx - 6, cy, cx + 6, cy)
        painter.drawLine(cx, cy - 6, cx, cy + 6)


class Look360(QWidget):
    """Pinhole look from inside the ERP sphere (yaw + pitch). No vispy / QWebEngine."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(140)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setStyleSheet("background: #081018; border: 1px solid #1c2b3a;")
        self._erp: Image.Image | None = None
        self._yaw = 0.0
        self._pitch = 0.0
        self._drag: QPoint | None = None
        self._frame: QPixmap | None = None
        self._dirty = True
        self.setCursor(Qt.CursorShape.OpenHandCursor)

    def set_pil(self, im: Image.Image | None) -> None:
        if im is None:
            self._erp = None
            self._frame = None
            self._dirty = True
            self.update()
            return
        rgb = im.convert("RGB")
        if rgb.width > LOOK_ERP_MAX_WIDTH:
            nh = max(1, int(round(rgb.height * LOOK_ERP_MAX_WIDTH / rgb.width)))
            rgb = rgb.resize((LOOK_ERP_MAX_WIDTH, nh), Image.Resampling.BILINEAR)
        self._erp = rgb
        self._dirty = True
        self.update()

    def set_path(self, path: Path | None) -> None:
        if path is None or not path.is_file():
            self.set_pil(None)
            return
        with Image.open(path) as im:
            im.load()
            self.set_pil(im.convert("RGB"))

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._dirty = True

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag = event.position().toPoint()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        self._drag = None
        self.setCursor(Qt.CursorShape.OpenHandCursor)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if self._drag is None or self._erp is None:
            return
        pos = event.position().toPoint()
        dx = pos.x() - self._drag.x()
        dy = pos.y() - self._drag.y()
        self._drag = pos
        w = max(1, self.width())
        h = max(1, self.height())
        vfov = 2.0 * math.degrees(
            math.atan(math.tan(math.radians(LOOK_HFOV_DEG) / 2.0) * (h / w))
        )
        self._yaw = (self._yaw + dx * (LOOK_HFOV_DEG / w)) % 360.0
        self._pitch = max(
            -LOOK_PITCH_MAX_DEG,
            min(LOOK_PITCH_MAX_DEG, self._pitch - dy * (vfov / h)),
        )
        self._dirty = True
        self.update()

    def _rebuild(self) -> None:
        if self._erp is None:
            self._frame = None
            return
        cap = 640
        w, h = max(1, self.width()), max(1, self.height())
        if w >= h and w > cap:
            h = max(1, int(round(h * cap / w)))
            w = cap
        elif h > cap:
            w = max(1, int(round(w * cap / h)))
            h = cap
        look = erp_to_rectilinear(
            self._erp,
            yaw_deg=self._yaw,
            pitch_deg=self._pitch,
            h_fov_deg=LOOK_HFOV_DEG,
            width=w,
            height=h,
        )
        self._frame = pil_to_pixmap(look)

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.fillRect(self.rect(), Qt.GlobalColor.black)
        if self._erp is None:
            painter.setPen(Qt.GlobalColor.gray)
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "360 · drag to look")
            return
        if self._dirty or self._frame is None:
            self._rebuild()
            self._dirty = False
        if self._frame is None or self._frame.isNull():
            return
        scaled = self._frame.scaled(
            self.size(),
            Qt.AspectRatioMode.IgnoreAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        painter.drawPixmap(0, 0, scaled)
