"""COLMAP sparse viewport. Qt process — no torch.

Orbit of points3D.bin + camera centers. Dummy chrome is the COLMAP sparse still
(docs/media/colmap-sparse.webp); this pane shows the scene's sparse/0, not cube faces.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import QPointF, QSize, Qt
from PySide6.QtGui import (
    QColor,
    QImage,
    QMouseEvent,
    QPainter,
    QPen,
    QWheelEvent,
)
from PySide6.QtWidgets import QSizePolicy, QWidget

from engine.splat.colmap_io import load_sparse_display

_BG = (8, 16, 24)
_CYAN = QColor(77, 242, 255)
_MAX_PTS = 18000


class SparseView(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("sparseView")
        self.setMinimumHeight(200)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, True)
        self._xyz = np.zeros((0, 3), dtype=np.float32)
        self._rgb = np.zeros((0, 3), dtype=np.uint8)
        self._cams = np.zeros((0, 3), dtype=np.float32)
        self._center = np.zeros(3, dtype=np.float32)
        self._radius = 1.0
        self._yaw = 0.7
        self._pitch = 0.45
        self._zoom = 1.0
        self._drag: tuple[float, float] | None = None
        self._empty = "Empty until Reconstruct."
        self._error = ""
        self._source = ""
        self._loaded_stamp = ""

    def sizeHint(self) -> QSize:
        return QSize(640, 400)

    def clear(self, message: str = "Empty until Reconstruct.") -> None:
        self._xyz = np.zeros((0, 3), dtype=np.float32)
        self._rgb = np.zeros((0, 3), dtype=np.uint8)
        self._cams = np.zeros((0, 3), dtype=np.float32)
        self._empty = message
        self._error = ""
        self._source = ""
        self._loaded_stamp = ""
        self.update()

    def _file_stamp(self, sparse: Path) -> str:
        bits = []
        for name in ("points3D.bin", "images.bin", "cameras.bin"):
            f = sparse / name
            if f.is_file():
                st = f.stat()
                bits.append(f"{name}:{st.st_mtime_ns}:{st.st_size}")
        return "|".join(bits)

    def load(self, sparse_path: str | Path, *, force: bool = False) -> None:
        path = Path(sparse_path)
        try:
            from engine.splat.colmap_io import resolve_sparse0

            sparse = resolve_sparse0(path)
        except Exception:
            sparse = path
        key = str(sparse.resolve()) if sparse.exists() else str(path)
        stamp = self._file_stamp(sparse) if sparse.is_dir() else ""
        if (
            not force
            and key == self._source
            and stamp
            and stamp == self._loaded_stamp
            and self._xyz.shape[0] > 0
        ):
            self.update()
            return
        try:
            xyz, rgb, cams = load_sparse_display(path)
        except Exception as exc:
            self.clear(f"Could not read sparse model:\n{path}\n{exc}")
            self._error = str(exc)
            return
        n = int(xyz.shape[0])
        if n > _MAX_PTS:
            pick = np.linspace(0, n - 1, _MAX_PTS).astype(np.int64)
            xyz = xyz[pick]
            rgb = rgb[pick]
        self._xyz = np.ascontiguousarray(xyz, dtype=np.float32).copy()
        cols = np.clip(np.asarray(rgb) * 255.0, 0, 255).astype(np.uint8)
        self._rgb = np.ascontiguousarray(cols).copy()
        if cams is not None and getattr(cams, "shape", (0,))[0]:
            self._cams = np.ascontiguousarray(cams, dtype=np.float32).copy()
        else:
            self._cams = np.zeros((0, 3), np.float32)
        cloud = self._xyz if self._xyz.shape[0] else self._cams
        if cloud.shape[0] == 0:
            self.clear("sparse/0 has no points")
            return
        self._center = cloud.mean(axis=0)
        self._radius = float(np.linalg.norm(cloud - self._center, axis=1).max())
        self._radius = max(self._radius, 1e-3)
        self._error = ""
        self._source = key
        self._loaded_stamp = self._file_stamp(sparse)
        self.reset_view()

    def reset_view(self) -> None:
        self._yaw = 0.7
        self._pitch = 0.45
        self._zoom = 1.0
        self.update()

    def _rot(self) -> np.ndarray:
        cy, sy = np.cos(self._yaw), np.sin(self._yaw)
        cp, sp = np.cos(self._pitch), np.sin(self._pitch)
        ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float32)
        rx = np.array([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]], dtype=np.float32)
        return rx @ ry

    def _project(self, xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if xyz.shape[0] == 0:
            z = np.zeros((0,), dtype=np.float32)
            return np.zeros((0, 2), dtype=np.float32), z, z.astype(bool)
        rel = (xyz - self._center) @ self._rot().T
        w, h = max(self.width(), 1), max(self.height(), 1)
        scale = (0.42 * min(w, h) * self._zoom) / self._radius
        u = rel[:, 0] * scale + w * 0.5
        v = -rel[:, 1] * scale + h * 0.5
        depth = rel[:, 2]
        vis = (u >= -2) & (u < w + 2) & (v >= -2) & (v < h + 2)
        return np.column_stack([u, v]), depth, vis

    def paintEvent(self, event) -> None:  # noqa: ANN001
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(*_BG))
        if self._xyz.shape[0] == 0:
            p.setPen(QColor(139, 155, 180))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self._empty)
            return
        w, h = max(self.width(), 1), max(self.height(), 1)
        buf = np.zeros((h, w, 4), dtype=np.uint8)
        buf[:, :, 0] = _BG[2]
        buf[:, :, 1] = _BG[1]
        buf[:, :, 2] = _BG[0]
        buf[:, :, 3] = 255
        uv, depth, vis = self._project(self._xyz)
        if vis.any():
            order = np.argsort(depth[vis])
            xs = np.clip(uv[vis, 0][order].astype(np.int32), 0, w - 1)
            ys = np.clip(uv[vis, 1][order].astype(np.int32), 0, h - 1)
            cols = self._rgb[vis][order]
            # BGRA for QImage Format_ARGB32 on little-endian
            buf[ys, xs, 0] = cols[:, 2]
            buf[ys, xs, 1] = cols[:, 1]
            buf[ys, xs, 2] = cols[:, 0]
            buf[ys, xs, 3] = 255
            if w > 2 and h > 2:
                buf[ys, np.clip(xs + 1, 0, w - 1), :3] = buf[ys, xs, :3]
                buf[np.clip(ys + 1, 0, h - 1), xs, :3] = buf[ys, xs, :3]
        qimg = QImage(buf.data, w, h, w * 4, QImage.Format.Format_ARGB32).copy()
        p.drawImage(0, 0, qimg)
        if self._cams.shape[0]:
            cuv, _, cvis = self._project(self._cams)
            p.setPen(QPen(_CYAN, 1.6))
            p.setBrush(_CYAN)
            for (x, y), ok in zip(cuv, cvis):
                if not ok:
                    continue
                p.drawEllipse(QPointF(float(x), float(y)), 3.2, 3.2)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag = (float(event.position().x()), float(event.position().y()))

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._drag is None:
            return
        x, y = float(event.position().x()), float(event.position().y())
        dx, dy = x - self._drag[0], y - self._drag[1]
        self._drag = (x, y)
        self._yaw += dx * 0.008
        self._pitch = float(np.clip(self._pitch + dy * 0.008, -1.2, 1.2))
        self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        self._drag = None

    def wheelEvent(self, event: QWheelEvent) -> None:
        delta = event.angleDelta().y()
        if delta == 0:
            return
        factor = 1.12 if delta > 0 else 0.89
        self._zoom = float(np.clip(self._zoom * factor, 0.15, 12.0))
        self.update()
