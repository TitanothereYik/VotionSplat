"""Live 3DGS viewport. Qt process — no torch.

Frames arrive over shared memory (``engine.splat.viewport.SharedFrameReader``); the
worker announces the block with ``SPLAT\\tshm\\t<name>...`` and this widget attaches by
name and polls the sequence number at 60 Hz. If attach fails the PNG fallback
(``load()``) still works.

Camera: LMB look, RMB pan, wheel dolly (zoom is not product-capped). Holding LMB also
arms WASD + QE flythrough (UE-editor style: W/S forward-back, A/D strafe, E/Q up-down,
Shift sprints) driven by the same 60 Hz tick that polls shared memory. Reset is the
rails Star (lock #43). Every change is coalesced into one ``viewChanged(dict)`` per
tick; the main window writes it to the worker's stdin as ``VIEW\\t{json}``. ``cam >= 0``
jumps to a training camera, ``gt`` shows render | ground truth side by side.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from PySide6.QtCore import QRect, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QImage, QMouseEvent, QPainter, QPixmap, QWheelEvent
from PySide6.QtWidgets import QSizePolicy, QWidget

from engine.splat import defaults as D
from engine.splat.camera import ViewState
from engine.splat.viewport import SharedFrameReader

_BG = QColor(8, 16, 24)
_HUD_PLATE = QColor(6, 12, 20, 170)
_HUD_TEXT = QColor(214, 224, 236)
_HUD_DIM = QColor(139, 155, 180)
_BADGE = {
    "TRAINING": QColor(255, 159, 48),
    "PAUSED": QColor(139, 155, 180),
    "VIEW": QColor(77, 163, 255),
    "LOADING": QColor(120, 120, 140),
    "IDLE": QColor(90, 100, 120),
    "DONE": QColor(77, 163, 255),
    "SAVING": QColor(255, 159, 48),
}


_FLY_KEYS = {
    Qt.Key.Key_W,
    Qt.Key.Key_A,
    Qt.Key.Key_S,
    Qt.Key.Key_D,
    Qt.Key.Key_Q,
    Qt.Key.Key_E,
    Qt.Key.Key_Shift,
}


class SplatLiveView(QWidget):
    viewChanged = Signal(dict)  # coalesced camera request (VIEW payload)
    frameShown = Signal(int, int, int)  # seq, w, h (for the page HUD / fps)

    def __init__(self, view_path_fn, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("splatLiveView")
        self.setMinimumHeight(220)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, True)
        self.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.NoContextMenu)
        self.setCursor(Qt.CursorShape.OpenHandCursor)
        self._view_path_fn = view_path_fn
        self._img: QImage | None = None
        self._img_bytes: bytes | None = None  # keeps the QImage buffer alive
        self._pix: QPixmap | None = None  # PNG fallback
        self._stamp = ""
        self._empty = "Train to see the splat form."
        self._state = ViewState()
        self._load_persisted()
        self._drag: tuple[float, float] | None = None
        self._drag_mode: str | None = None
        self._fly_keys: set[Qt.Key] = set()
        self._fly_last_t: float | None = None
        self._reader: SharedFrameReader | None = None
        self._shm_name = ""
        self._dirty = False
        self._hud: list[str] = []
        self._badge = "IDLE"
        self._fps_n = 0
        self._fps_t0 = time.perf_counter()
        self.fps = 0.0
        self._tick = QTimer(self)
        self._tick.setInterval(16)  # 60 Hz
        self._tick.timeout.connect(self._on_tick)
        self._tick.start()

    # ------------------------------------------------------------------ camera
    def _load_persisted(self) -> None:
        try:
            path = Path(self._view_path_fn())
            if path.is_file():
                self._state = ViewState.from_dict(json.loads(path.read_text(encoding="utf-8")))
                self._state.cam = -1
                self._state.gt = False
        except Exception:
            self._state = ViewState()

    def sizeHint(self) -> QSize:
        return QSize(640, 400)

    def view_payload(self) -> dict:
        """Current camera + the frame size the worker should render."""
        w = max(64, min(int(self.width()), D.VIEWPORT_MAX_W))
        h = max(36, min(int(self.height()), D.VIEWPORT_MAX_H))
        st = self._state
        st.width, st.height = w, h
        return st.to_dict()

    def state(self) -> ViewState:
        return self._state

    def mark_dirty(self) -> None:
        self._dirty = True

    def request_view(self) -> None:
        """Send the camera now (used when a worker attaches or the page shows)."""
        self._dirty = False
        self.viewChanged.emit(self.view_payload())

    def set_camera(self, index: int) -> None:
        """Jump to training camera ``index`` (-1 = Star). Orbit offsets reset."""
        self._state.cam = int(index)
        self._state.yaw = 0.0
        self._state.pitch = 0.0
        self._state.zoom = 1.0
        self._state.pan_x = 0.0
        self._state.pan_y = 0.0
        self._state.fly_fwd = 0.0
        self._state.fly_right = 0.0
        self._state.fly_up = 0.0
        if index < 0:
            self._state.gt = False
        self.mark_dirty()
        self.update()

    def set_gt(self, on: bool) -> None:
        self._state.gt = bool(on) and self._state.cam >= 0
        self.mark_dirty()

    def camera_index(self) -> int:
        return int(self._state.cam)

    def reset_view(self) -> None:
        self.set_camera(-1)

    # ------------------------------------------------------------------ frames
    def attach(self, name: str) -> bool:
        """Attach the worker's shared-memory block. False -> caller keeps PNG fallback."""
        self.detach()
        try:
            self._reader = SharedFrameReader(name)
        except Exception:
            self._reader = None
            return False
        self._shm_name = name
        self._pix = None
        self._empty = ""
        self._fps_n = 0
        self._fps_t0 = time.perf_counter()
        return True

    def detach(self) -> None:
        if self._reader is not None:
            self._reader.close()
            self._reader = None
        self._shm_name = ""

    @property
    def attached(self) -> bool:
        return self._reader is not None

    def _advance_fly(self) -> None:
        """Integrate held WASD/QE keys into the fly accumulators. Armed only while LMB
        is held (``_drag_mode == "orbit"``); see ``keyPressEvent``."""
        now = time.perf_counter()
        if self._drag_mode != "orbit" or not self._fly_keys:
            self._fly_last_t = None
            return
        if self._fly_last_t is None:
            self._fly_last_t = now
            return
        dt = min(max(now - self._fly_last_t, 0.0), 0.1)
        self._fly_last_t = now
        if dt <= 0.0:
            return
        speed = D.VIEWPORT_FLY_SPEED
        if Qt.Key.Key_Shift in self._fly_keys:
            speed *= D.VIEWPORT_FLY_SPRINT
        step = speed * dt
        st = self._state
        moved = False
        if Qt.Key.Key_W in self._fly_keys:
            st.fly_fwd += step
            moved = True
        if Qt.Key.Key_S in self._fly_keys:
            st.fly_fwd -= step
            moved = True
        if Qt.Key.Key_D in self._fly_keys:
            st.fly_right += step
            moved = True
        if Qt.Key.Key_A in self._fly_keys:
            st.fly_right -= step
            moved = True
        if Qt.Key.Key_E in self._fly_keys:
            st.fly_up += step
            moved = True
        if Qt.Key.Key_Q in self._fly_keys:
            st.fly_up -= step
            moved = True
        if moved:
            st.gt = False
            self.mark_dirty()

    def _on_tick(self) -> None:
        self._advance_fly()
        if self._dirty:
            self._dirty = False
            self.viewChanged.emit(self.view_payload())
        if self._reader is None:
            return
        try:
            got = self._reader.poll()
        except Exception:
            self.detach()
            return
        if got is None:
            return
        seq, w, h, data = got
        self._img_bytes = data
        self._img = QImage(data, w, h, w * 3, QImage.Format.Format_RGB888)
        self._pix = None
        self._fps_n += 1
        now = time.perf_counter()
        if now - self._fps_t0 >= 1.0:
            self.fps = self._fps_n / (now - self._fps_t0)
            self._fps_n = 0
            self._fps_t0 = now
        self.frameShown.emit(seq, w, h)
        self.update()

    def clear(self, message: str = "Train to see the splat form.") -> None:
        self._img = None
        self._img_bytes = None
        self._pix = None
        self._stamp = ""
        self._empty = message
        self.update()

    def load(self, png_path: str | Path) -> None:
        """PNG fallback (worker preview file). Ignored while shared memory is attached."""
        if self._reader is not None:
            return
        path = Path(png_path)
        if not path.is_file():
            return
        try:
            st = path.stat()
            stamp = f"{path.resolve()}:{st.st_mtime_ns}:{st.st_size}"
            data = path.read_bytes()
        except OSError:
            return
        if stamp == self._stamp and self._pix is not None:
            return
        img = QImage()
        if not img.loadFromData(data):
            return
        if img.format() != QImage.Format.Format_RGB888:
            img = img.convertToFormat(QImage.Format.Format_RGB888)
        self._pix = QPixmap.fromImage(img)
        self._img = None
        self._img_bytes = None
        self._stamp = stamp
        self.update()

    # --------------------------------------------------------------------- HUD
    def set_hud(self, badge: str, lines: list[str]) -> None:
        self._badge = badge or "IDLE"
        self._hud = list(lines)
        self.update()

    # ------------------------------------------------------------------- paint
    def paintEvent(self, event) -> None:  # noqa: ANN001
        p = QPainter(self)
        p.fillRect(self.rect(), _BG)
        p.setCompositionMode(QPainter.CompositionMode.CompositionMode_Source)
        drawn = False
        if self._img is not None and not self._img.isNull():
            target = self._fit(self._img.width(), self._img.height())
            p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, target.width() != self._img.width())
            p.drawImage(target, self._img)
            drawn = True
        elif self._pix is not None and not self._pix.isNull():
            target = self._fit(self._pix.width(), self._pix.height())
            p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
            p.drawPixmap(target, self._pix)
            drawn = True
        p.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
        if not drawn and self._empty:
            p.setPen(_HUD_DIM)
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self._empty)
        self._paint_hud(p)

    def _fit(self, w: int, h: int) -> QRect:
        if w <= 0 or h <= 0:
            return self.rect()
        scale = min(self.width() / w, self.height() / h)
        tw, th = max(1, int(w * scale)), max(1, int(h * scale))
        return QRect((self.width() - tw) // 2, (self.height() - th) // 2, tw, th)

    def _paint_hud(self, p: QPainter) -> None:
        font = QFont(self.font())
        font.setPointSize(9)
        p.setFont(font)
        fm = p.fontMetrics()
        line_h = fm.height()
        badge = self._badge
        lines = self._hud
        if not lines and badge == "IDLE":
            return
        pad = 8
        width = max([fm.horizontalAdvance(t) for t in lines] + [fm.horizontalAdvance(badge) + 24])
        height = pad * 2 + line_h * (len(lines) + 1) + 4
        plate = QRect(10, 10, width + pad * 2, height)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(_HUD_PLATE)
        p.drawRoundedRect(plate, 6, 6)
        bw = fm.horizontalAdvance(badge) + 14
        p.setBrush(_BADGE.get(badge, _BADGE["IDLE"]))
        p.drawRoundedRect(QRect(plate.left() + pad, plate.top() + pad, bw, line_h), 4, 4)
        p.setPen(QColor(255, 255, 255))
        p.drawText(QRect(plate.left() + pad, plate.top() + pad, bw, line_h), Qt.AlignmentFlag.AlignCenter, badge)
        p.setPen(_HUD_TEXT)
        y = plate.top() + pad + line_h + 4
        for text in lines:
            p.drawText(plate.left() + pad, y + fm.ascent(), text)
            y += line_h

    # ------------------------------------------------------------------- input
    def resizeEvent(self, event) -> None:  # noqa: ANN001
        super().resizeEvent(event)
        self.mark_dirty()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        btn = event.button()
        if btn == Qt.MouseButton.LeftButton:
            self._drag_mode = "orbit"
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
        elif btn == Qt.MouseButton.RightButton:
            self._drag_mode = "pan"
            self.setCursor(Qt.CursorShape.SizeAllCursor)
        else:
            return
        self._drag = (float(event.position().x()), float(event.position().y()))
        self.grabMouse()
        self.setFocus(Qt.FocusReason.MouseFocusReason)
        event.accept()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._drag is None or self._drag_mode is None:
            return
        x, y = float(event.position().x()), float(event.position().y())
        dx, dy = x - self._drag[0], y - self._drag[1]
        self._drag = (x, y)
        st = self._state
        if st.gt:
            st.gt = False  # any camera motion leaves the GT compare
        if self._drag_mode == "orbit":
            # Drag right looks right (UE editor / FPS). orbit_w2c yaws around world-up
            # with the right-hand rule, so screen +x must subtract yaw. Pitch already
            # matches screen +y (drag down looks down).
            st.yaw -= dx * 0.008
            st.pitch = float(max(-1.45, min(1.45, st.pitch + dy * 0.008)))
        else:
            scale = 0.0028 / max(st.zoom, 1e-4)
            st.pan_x = float(max(-64.0, min(64.0, st.pan_x - dx * scale)))
            st.pan_y = float(max(-64.0, min(64.0, st.pan_y - dy * scale)))
        self.mark_dirty()
        event.accept()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if QWidget.mouseGrabber() is self:
            self.releaseMouse()
        self._drag = None
        self._drag_mode = None
        self._fly_keys.clear()
        self._fly_last_t = None
        self.setCursor(Qt.CursorShape.OpenHandCursor)
        self.mark_dirty()
        event.accept()

    def keyPressEvent(self, event) -> None:  # noqa: ANN001
        key = event.key()
        if key in _FLY_KEYS and self._drag_mode == "orbit" and not event.isAutoRepeat():
            self._fly_keys.add(key)
            event.accept()
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event) -> None:  # noqa: ANN001
        key = event.key()
        if key in self._fly_keys and not event.isAutoRepeat():
            self._fly_keys.discard(key)
            event.accept()
            return
        super().keyReleaseEvent(event)

    def focusOutEvent(self, event) -> None:  # noqa: ANN001
        # Defensive: alt-tab / focus loss while a fly key is held must not leave the
        # camera drifting once focus returns without a fresh keyPressEvent.
        self._fly_keys.clear()
        self._fly_last_t = None
        super().focusOutEvent(event)

    def wheelEvent(self, event: QWheelEvent) -> None:
        delta = event.angleDelta().y()
        if delta == 0:
            delta = int(event.pixelDelta().y())
        if delta == 0:
            return
        factor = 1.12 if delta > 0 else 0.89
        st = self._state
        st.gt = False
        st.zoom = float(max(1e-6, min(1e9, st.zoom * factor)))
        self.mark_dirty()
        event.accept()
