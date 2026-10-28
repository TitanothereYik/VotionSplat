"""FLOOR / SIDE editor panes. Qt process — no torch.

FLOOR: X →  Z ↑. SIDE: Z →  Y ↑. Star 0 is not draggable.

Plates are 2D SplatKit orthos. Rail gizmos are QPainter on the same widget:
embedded vispy QOpenGLWidget does not composite Line/Markers over an Image
on this Windows stack (depth + FBO). Dummy chrome is 2D; keep vispy unused here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QMouseEvent,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QPolygonF,
    QResizeEvent,
    QWheelEvent,
)
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QSizePolicy, QVBoxLayout, QWidget

from engine.geo.rail import catmull_rom, fill_targets

_CYAN = QColor(77, 242, 255)
_RED = QColor(255, 64, 64)
_ORANGE = QColor(255, 154, 50)
_YELLOW = QColor(255, 235, 56)
_PATH = QColor(126, 200, 255)
_LOOK = QColor(77, 163, 255)


class GeoCanvas(QWidget):
    """Ortho plate + rail overlay. Image pixels: origin top-left, matching the PNG."""

    knotMoved = Signal(int, float, float)
    lookMoved = Signal(float, float)
    lookTargetMoved = Signal(int, float, float)

    def __init__(self, kind: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.kind = kind
        self.setObjectName("geoCanvas")
        self.setMinimumHeight(140)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, False)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, True)
        self.setAutoFillBackground(False)
        self.setMouseTracking(True)
        self._lo = np.array([-1.0, -1.0], dtype=np.float64)
        self._hi = np.array([1.0, 1.0], dtype=np.float64)
        self._img_w = 256
        self._img_h = 256
        self._pix = QPixmap()
        self._knots = np.zeros((0, 3), dtype=np.float64)
        self._look = np.array([-0.108, 0.073, 1.953], dtype=np.float64)
        self._targets = np.zeros((0, 3), dtype=np.float64)
        self._mode = "look_at_target"
        self._n_path = 81
        self._others: list[dict] = []
        self._enabled = False
        self._show_cloud = False
        self._cloud_xy: np.ndarray | None = None
        self._cloud_rgb: np.ndarray | None = None
        self._cam = [0.0, 0.0, 256.0, 256.0]  # image-space x, y, w, h (y down)
        self._drag: dict | None = None
        self._last = QPointF()

    def sizeHint(self) -> QSize:
        return QSize(480, 280)

    def set_editor_enabled(self, on: bool) -> None:
        self._enabled = bool(on)
        self.update()

    def set_ortho(self, png_path: Path | None, lo: list[float], hi: list[float]) -> None:
        self._lo = np.array(lo, dtype=np.float64)
        self._hi = np.array(hi, dtype=np.float64)
        if png_path is None or not Path(png_path).is_file():
            self._pix = QPixmap()
            self._img_w, self._img_h = 256, 256
            self._reset_cam()
            self.update()
            return
        pix = QPixmap(str(png_path))
        self._pix = pix
        self._img_w = max(pix.width(), 1)
        self._img_h = max(pix.height(), 1)
        self._reset_cam()
        self.update()

    def set_cloud(self, xyz: np.ndarray | None, rgb: np.ndarray | None) -> None:
        if xyz is None or len(xyz) == 0:
            self._cloud_xy = None
            self._cloud_rgb = None
            self.update()
            return
        xy = self._world_to_img(xyz)
        cols = np.asarray(rgb)
        n = int(xy.shape[0])
        if n > 1200:
            pick = np.linspace(0, n - 1, 1200).astype(np.int64)
            xy = xy[pick]
            if cols.ndim == 2 and cols.shape[0] == n:
                cols = cols[pick]
        self._cloud_xy = xy
        if cols.ndim == 2 and cols.shape[1] >= 3:
            if cols.max() <= 1.0:
                cols = (cols[:, :3] * 255.0).clip(0, 255)
            self._cloud_rgb = cols[:, :3].astype(np.uint8)
        else:
            self._cloud_rgb = None
        self.update()

    def set_knots(
        self,
        knots: np.ndarray,
        look: np.ndarray,
        *,
        mode: str = "look_at_target",
        targets: np.ndarray | None = None,
        n_path: int = 81,
    ) -> None:
        self._knots = np.asarray(knots, dtype=np.float64)
        self._look = np.asarray(look, dtype=np.float64).reshape(3)
        self._mode = mode
        self._n_path = max(2, int(n_path))
        if targets is None or len(targets) != len(self._knots):
            self._targets = np.full_like(self._knots, np.nan)
        else:
            self._targets = np.asarray(targets, dtype=np.float64)
        self.update()

    def set_other_rails(self, rails: list[dict] | None) -> None:
        """Inactive rails: dimmed splines only. Not draggable."""
        self._others = list(rails or [])
        self.update()

    def toggle_cloud(self) -> None:
        self._show_cloud = not self._show_cloud
        self.update()

    def reset_view(self) -> None:
        self._reset_cam()
        self.update()

    def _reset_cam(self) -> None:
        self._cam = [0.0, 0.0, float(self._img_w), float(self._img_h)]

    def _world_to_img(self, xyz: np.ndarray) -> np.ndarray:
        """World -> PNG pixel (origin top-left). Same mapping as engine.geo.scene_ref.ortho_view."""
        pts = np.asarray(xyz, dtype=np.float64)
        w, h = self._img_w, self._img_h
        lo, hi = self._lo, self._hi
        ext = np.maximum(hi - lo, 1e-8)
        if self.kind == "floor":
            a = pts[:, 0]
            b = pts[:, 2]
        else:
            a = pts[:, 2]
            b = pts[:, 1]
        x = (a - lo[0]) / ext[0] * (w - 1)
        y = (hi[1] - b) / ext[1] * (h - 1)
        return np.column_stack([x, y])

    def _img_to_world_ab(self, ix: float, iy: float) -> tuple[float, float]:
        w, h = max(self._img_w - 1, 1), max(self._img_h - 1, 1)
        lo, hi = self._lo, self._hi
        ext = np.maximum(hi - lo, 1e-8)
        a = lo[0] + (ix / w) * ext[0]
        b = hi[1] - (iy / h) * ext[1]
        return float(a), float(b)

    def _dest(self) -> QRectF:
        cx, cy, cw, ch = self._cam
        ww, wh = max(self.width(), 1), max(self.height(), 1)
        cw, ch = max(cw, 1e-6), max(ch, 1e-6)
        scale = min(ww / cw, wh / ch)
        dw, dh = cw * scale, ch * scale
        return QRectF((ww - dw) * 0.5, (wh - dh) * 0.5, dw, dh)

    def _img_to_widget(self, img_xy: np.ndarray) -> np.ndarray:
        pts = np.asarray(img_xy, dtype=np.float64).reshape(-1, 2)
        r = self._dest()
        cx, cy, cw, ch = self._cam
        x = r.x() + (pts[:, 0] - cx) / max(cw, 1e-8) * r.width()
        y = r.y() + (pts[:, 1] - cy) / max(ch, 1e-8) * r.height()
        return np.column_stack([x, y])

    def _widget_to_img(self, qx: float, qy: float) -> tuple[float, float]:
        r = self._dest()
        cx, cy, cw, ch = self._cam
        if r.width() <= 1 or r.height() <= 1:
            return 0.0, 0.0
        ix = cx + (qx - r.x()) / r.width() * cw
        iy = cy + (qy - r.y()) / r.height() * ch
        return float(ix), float(iy)

    def _display_targets(self) -> np.ndarray:
        n = self._knots.shape[0]
        if n == 0:
            return np.zeros((0, 3), dtype=np.float64)
        if self._mode == "look_at_target":
            return np.repeat(self._look.reshape(1, 3), n, axis=0)
        return fill_targets(self._knots, self._targets)

    def _arrow_heads_img(self) -> np.ndarray:
        vis = self._world_to_img(self._knots)
        tgt = self._world_to_img(self._display_targets())
        return vis + 0.42 * (tgt - vis)

    def paintEvent(self, event) -> None:  # noqa: ANN001
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        p.fillRect(self.rect(), QColor(8, 16, 24))
        dest = self._dest()
        if not self._pix.isNull():
            src = QRectF(self._cam[0], self._cam[1], self._cam[2], self._cam[3])
            p.drawPixmap(dest, self._pix, src)
        if self._show_cloud and self._cloud_xy is not None and len(self._cloud_xy):
            wxy = self._img_to_widget(self._cloud_xy)
            rgb = self._cloud_rgb
            p.setPen(Qt.PenStyle.NoPen)
            for i, (x, y) in enumerate(wxy):
                if rgb is not None and i < len(rgb):
                    p.setBrush(QColor(int(rgb[i, 0]), int(rgb[i, 1]), int(rgb[i, 2]), 180))
                else:
                    p.setBrush(QColor(160, 180, 200, 140))
                p.drawEllipse(QPointF(float(x), float(y)), 1.6, 1.6)
        if not self._enabled:
            return
        self._paint_others(p)
        if self._knots.shape[0] == 0:
            return
        self._paint_rail(p)

    def _paint_others(self, p: QPainter) -> None:
        dim = QColor(126, 200, 255, 70)
        halo_c = QColor(8, 16, 24, 80)
        for other in self._others:
            knots = np.asarray(other.get("knots"), dtype=np.float64)
            if knots.ndim != 2 or knots.shape[0] < 2:
                continue
            n_path = max(2, int(other.get("n_path") or 81))
            spline = catmull_rom(knots, n_path)
            ws = self._img_to_widget(self._world_to_img(spline))
            path = QPainterPath(QPointF(float(ws[0, 0]), float(ws[0, 1])))
            for x, y in ws[1:]:
                path.lineTo(float(x), float(y))
            halo = QPen(halo_c, 4.0)
            halo.setCapStyle(Qt.PenCapStyle.RoundCap)
            p.setPen(halo)
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawPath(path)
            ink = QPen(dim, 2.0)
            ink.setCapStyle(Qt.PenCapStyle.RoundCap)
            p.setPen(ink)
            p.drawPath(path)

    def _paint_rail(self, p: QPainter) -> None:
        n = int(self._knots.shape[0])
        vis = self._world_to_img(self._knots)
        wvis = self._img_to_widget(vis)
        if n >= 2:
            spline = catmull_rom(self._knots, self._n_path)
            ws = self._img_to_widget(self._world_to_img(spline))
            path = QPainterPath(QPointF(float(ws[0, 0]), float(ws[0, 1])))
            for x, y in ws[1:]:
                path.lineTo(float(x), float(y))
            halo = QPen(QColor(8, 16, 24), 6.0)
            halo.setCapStyle(Qt.PenCapStyle.RoundCap)
            halo.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            p.setPen(halo)
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawPath(path)
            ink = QPen(_PATH, 3.4)
            ink.setCapStyle(Qt.PenCapStyle.RoundCap)
            ink.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            p.setPen(ink)
            p.drawPath(path)
        targets = self._display_targets()
        wtgt = self._img_to_widget(self._world_to_img(targets))
        heads = wvis + 0.42 * (wtgt - wvis)
        if self._mode == "look_at_target":
            look = self._img_to_widget(self._world_to_img(self._look.reshape(1, 3)))[0]
            dash = QPen(_ORANGE, 1.6, Qt.PenStyle.DashLine)
            p.setPen(dash)
            for i in range(n):
                p.drawLine(QPointF(float(wvis[i, 0]), float(wvis[i, 1])), QPointF(float(look[0]), float(look[1])))
        p.setPen(QPen(_LOOK, 2.8))
        for i in range(n):
            a = QPointF(float(wvis[i, 0]), float(wvis[i, 1]))
            b = QPointF(float(heads[i, 0]), float(heads[i, 1]))
            p.drawLine(a, b)
            p.setBrush(QBrush(_LOOK))
            p.drawPolygon(self._arrow_poly(a, b))
        extras = []
        extra_col = []
        if self._mode == "look_at_target":
            extras.append(self._img_to_widget(self._world_to_img(self._look.reshape(1, 3)))[0])
            extra_col.append(_ORANGE)
        if self._mode == "per_point_look":
            for i in range(n):
                extras.append(wtgt[i])
                extra_col.append(_YELLOW)
        for pt, col in zip(extras, extra_col):
            p.setPen(QPen(QColor(8, 16, 24), 2.4))
            p.setBrush(QBrush(col))
            p.drawEllipse(QPointF(float(pt[0]), float(pt[1])), 8.0, 8.0)
        font = QFont("JetBrains Mono", 10)
        font.setBold(True)
        p.setFont(font)
        for i in range(n):
            c = QPointF(float(wvis[i, 0]), float(wvis[i, 1]))
            r = 11.0 if i == 0 else 9.0
            p.setPen(QPen(QColor(8, 16, 24), 2.6))
            p.setBrush(QBrush(_CYAN if i == 0 else _RED))
            p.drawEllipse(c, r, r)
            p.setPen(QPen(QColor("white")))
            label = "★0" if i == 0 else str(i)
            p.drawText(QRectF(c.x() + r + 3, c.y() - 9, 40, 18), Qt.AlignmentFlag.AlignLeft, label)

    def _arrow_poly(self, a: QPointF, b: QPointF) -> QPolygonF:
        dx, dy = b.x() - a.x(), b.y() - a.y()
        length = max((dx * dx + dy * dy) ** 0.5, 1.0)
        ux, uy = dx / length, dy / length
        px, py = -uy, ux
        tip = QPointF(b.x() + ux * 7.0, b.y() + uy * 7.0)
        left = QPointF(b.x() - ux * 2.0 + px * 5.0, b.y() - uy * 2.0 + py * 5.0)
        right = QPointF(b.x() - ux * 2.0 - px * 5.0, b.y() - uy * 2.0 - py * 5.0)
        return QPolygonF([tip, left, right])

    def _hit_radius(self) -> float:
        return 14.0

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return
        self._last = event.position()
        if not self._enabled or self._knots.shape[0] == 0:
            self._drag = {"kind": "pan"}
            return
        qx, qy = float(event.position().x()), float(event.position().y())
        pt = np.array([qx, qy], dtype=np.float64)
        px = self._hit_radius()
        n = self._knots.shape[0]
        wvis = self._img_to_widget(self._world_to_img(self._knots))
        if self._mode == "per_point_look":
            wt = self._img_to_widget(self._world_to_img(self._display_targets()))
            d = np.linalg.norm(wt - pt, axis=1)
            i = int(np.argmin(d))
            if d[i] < px:
                self._drag = {"kind": "target", "i": i}
                return
        if self._mode == "look_at_target":
            look = self._img_to_widget(self._world_to_img(self._look.reshape(1, 3)))[0]
            if float(np.linalg.norm(pt - look)) < px:
                self._drag = {"kind": "look"}
                return
        if self._mode == "look_forward":
            heads = self._img_to_widget(self._arrow_heads_img())
            d = np.linalg.norm(heads - pt, axis=1)
            i = int(np.argmin(d))
            if d[i] < px:
                self._drag = {"kind": "arrow", "i": i}
                return
        d = np.linalg.norm(wvis - pt, axis=1)
        i = int(np.argmin(d))
        if d[i] < px and i > 0:
            self._drag = {"kind": "cam", "i": i}
            return
        self._drag = {"kind": "pan"}

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._drag is None:
            return
        pos = event.position()
        kind = self._drag["kind"]
        if kind == "pan":
            r = self._dest()
            if r.width() > 1 and r.height() > 1:
                dx = (pos.x() - self._last.x()) / r.width() * self._cam[2]
                dy = (pos.y() - self._last.y()) / r.height() * self._cam[3]
                self._cam[0] -= dx
                self._cam[1] -= dy
                self._last = pos
                self.update()
            return
        ix, iy = self._widget_to_img(float(pos.x()), float(pos.y()))
        a, b = self._img_to_world_ab(ix, iy)
        if kind == "look":
            self.lookMoved.emit(a, b)
        elif kind in ("target", "arrow"):
            self.lookTargetMoved.emit(int(self._drag["i"]), a, b)
        else:
            self.knotMoved.emit(int(self._drag["i"]), a, b)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        self._drag = None

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        self.update()

    def wheelEvent(self, event: QWheelEvent) -> None:
        delta = event.angleDelta().y()
        if delta == 0:
            return
        factor = 0.85 if delta > 0 else 1.18
        ix, iy = self._widget_to_img(float(event.position().x()), float(event.position().y()))
        cx, cy, cw, ch = self._cam
        nw, nh = cw * factor, ch * factor
        nw = min(max(nw, 8.0), self._img_w * 8.0)
        nh = min(max(nh, 8.0), self._img_h * 8.0)
        self._cam[0] = ix - (ix - cx) * (nw / cw)
        self._cam[1] = iy - (iy - cy) * (nh / ch)
        self._cam[2] = nw
        self._cam[3] = nh
        self.update()


class GeoPane(QWidget):
    """One orthographic editor pane (floor or side)."""

    knotMoved = Signal(int, float, float)
    lookMoved = Signal(float, float)
    lookTargetMoved = Signal(int, float, float)

    def __init__(self, kind: str, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.kind = kind
        tools = QHBoxLayout()
        tools.setContentsMargins(0, 0, 0, 0)
        lab = QLabel(title)
        lab.setProperty("class", "viewLbl")
        self.btn_geo = QPushButton("geo")
        self.btn_geo.setProperty("class", "rail-mini")
        self.btn_reset = QPushButton("reset view")
        self.btn_reset.setProperty("class", "rail-mini")
        self.btn_geo.setEnabled(False)
        self.btn_reset.setEnabled(False)
        self.canvas = GeoCanvas(kind)
        self.btn_geo.clicked.connect(self.canvas.toggle_cloud)
        self.btn_reset.clicked.connect(self.canvas.reset_view)
        self.canvas.knotMoved.connect(self.knotMoved)
        self.canvas.lookMoved.connect(self.lookMoved)
        self.canvas.lookTargetMoved.connect(self.lookTargetMoved)
        tools.addWidget(lab)
        tools.addStretch(1)
        tools.addWidget(self.btn_geo)
        tools.addWidget(self.btn_reset)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)
        tools_w = QWidget()
        tools_w.setLayout(tools)
        lay.addWidget(tools_w)
        lay.addWidget(self.canvas, 1)

    def set_editor_enabled(self, on: bool) -> None:
        self.canvas.set_editor_enabled(on)
        self.btn_geo.setEnabled(on)
        self.btn_reset.setEnabled(on)

    def set_ortho(self, png_path: Path | None, lo: list[float], hi: list[float]) -> None:
        self.canvas.set_ortho(png_path, lo, hi)

    def set_cloud(self, xyz: np.ndarray | None, rgb: np.ndarray | None) -> None:
        self.canvas.set_cloud(xyz, rgb)

    def set_knots(
        self,
        knots: np.ndarray,
        look: np.ndarray,
        *,
        mode: str = "look_at_target",
        targets: np.ndarray | None = None,
        n_path: int = 81,
        others: list[dict] | None = None,
    ) -> None:
        self.canvas.set_knots(knots, look, mode=mode, targets=targets, n_path=n_path)
        self.canvas.set_other_rails(others)
