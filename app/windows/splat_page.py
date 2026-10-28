"""Phase 2 Splat page (Postshot-style, 2026-09-07): LiteGS | MCMC Compact, live viewport.

Left rail: controller chips, max steps (any integer >= 1), Gaussian budget (editable),
resolution scale, SH degree, hold-out eval, Train | Pause | Stop | Continue | Retrain,
Export splat.ply | Export compact, export colour space (Unreal / MLSLabs or Unreal / Luma), Open COLMAP folder.
Viewport: shared-memory frames with HUD (state, step / epoch, loss EMA, PSNR,
Gaussians / budget, it/s, viewport fps, VRAM), loss / PSNR sparkline, training camera
list (grouped by rail) with GT compare. Reset returns to the rails Star.

Qt process — no torch. Worker protocol: ``engine.splat.trainer`` docstring.
"""

from __future__ import annotations

import json
import math
import re
from collections import deque
from pathlib import Path

from PySide6.QtCore import QPointF, QRectF, QSize, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QColor, QDesktopServices, QPainter, QPalette, QPen, QPolygonF
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from app.widgets.splat_view import SplatLiveView
from engine.pano.scene_io import scene_dir
from engine.splat import defaults as D
from engine.splat.status import litegs_spike_detail, litegs_spike_ok

_FRAME_RE = re.compile(r"frame_(\d+)_perspective_(\d+)")
_RAIL_NUM_RE = re.compile(r"^(?:rail[_-]?)?(\d+)$", re.IGNORECASE)


def _lab(text: str) -> QLabel:
    lab = QLabel(text)
    lab.setProperty("class", "fLab")
    return lab


def _hint(text: str) -> QLabel:
    lab = QLabel(text)
    lab.setWordWrap(True)
    lab.setProperty("class", "fHint")
    return lab


def _cam_rail_key(cam: dict) -> str:
    """Worker ``rail`` is a string (``rail_03``, ``view``, ``frame``), never an int."""
    raw = cam.get("rail")
    if raw is None or raw == "":
        return "0"
    return str(raw)


def _rail_sort_key(key: str) -> tuple:
    m = _RAIL_NUM_RE.match(key)
    if m:
        return (0, int(m.group(1)), key.lower())
    return (1, 0, key.lower())


def _rail_heading(key: str, n: int) -> str:
    m = _RAIL_NUM_RE.match(key)
    if m:
        return f"rail {int(m.group(1))} · {n} views"
    return f"{key.replace('_', ' ')} · {n} views"


def _int_text(text: str, default: int, lo: int = 1) -> int:
    raw = "".join(ch for ch in text if ch.isdigit())
    if not raw:
        return default
    try:
        return max(lo, int(raw))
    except ValueError:
        return default


class Sparkline(QWidget):
    """Loss EMA (amber) and PSNR (green) over the last reports."""

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
            p.setPen(QPen(QColor(46, 200, 120), 1.5))
            p.drawPolyline(self._poly(self._psnr, rect, lo - 0.5, hi + 0.5))
            p.setPen(QColor(46, 200, 120))
            text = f"PSNR {self._psnr[-1]:.2f} dB"
            p.drawText(int(rect.right()) - p.fontMetrics().horizontalAdvance(text), int(rect.top()) + 10, text)


class SplatPage(QWidget):
    requestJob = Signal(list)  # ["__splat__", scene, strategy_key, mode, steps, budget, res_scale, holdout, color_space]
    requestStop = Signal()
    controlLine = Signal(str)  # "PAUSE" | "RESUME" | "VIEW\t{json}" -> active splat worker stdin
    logLine = Signal(str)

    def __init__(self, scene_name_fn, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._scene_name = scene_name_fn
        self._strategy = "litegs"
        self._res_scale = 1.0
        self._state = "IDLE"
        self._paused = False
        self._worker_kind = ""  # "train" | "view" | ""
        self._cameras: list[dict] = []
        self._stats: dict[str, str] = {}
        self._shm_pending = ""
        body = QHBoxLayout(self)
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)
        body.addWidget(self._build_rail())
        body.addWidget(self._build_view(), 1)
        body.addWidget(self._build_cameras())
        self._timer = QTimer(self)
        self._timer.setInterval(200)
        self._timer.timeout.connect(self._poll_preview)
        self._timer.start()
        self.refresh()

    # ----------------------------------------------------------------- public
    def strategy_key(self) -> str:
        return "mcmc_compact" if self._strategy == "mcmc_compact" else "litegs"

    def strategy_label(self) -> str:
        return D.strategy_from_key(self.strategy_key())

    def _view_json(self) -> Path:
        return scene_dir(self._scene_name()) / D.VIEW_REL

    def _preview_png(self) -> Path:
        return scene_dir(self._scene_name()) / D.PREVIEW_REL

    # ------------------------------------------------------------------ rail
    def _build_rail(self) -> QWidget:
        rail = QWidget()
        rail.setObjectName("rail")
        rail.setFixedWidth(290)
        r = QVBoxLayout(rail)
        r.setContentsMargins(12, 12, 12, 14)
        r.setSpacing(4)

        r.addWidget(_lab("Density controller"))
        tog = QWidget()
        tog_l = QHBoxLayout(tog)
        tog_l.setContentsMargins(0, 0, 0, 0)
        tog_l.setSpacing(0)
        self.strat_group = QButtonGroup(self)
        self.strat_group.setExclusive(True)
        self.btn_litegs = QPushButton("LiteGS")
        self.btn_mcmc = QPushButton("MCMC Compact")
        for i, btn in enumerate((self.btn_litegs, self.btn_mcmc)):
            btn.setCheckable(True)
            btn.setProperty("class", "modeChip")
            self.strat_group.addButton(btn, i)
            tog_l.addWidget(btn)
        self.btn_litegs.setChecked(True)
        self.strat_group.idClicked.connect(self._on_strat)
        r.addWidget(tog)
        self.strat_hint = _hint("")
        r.addWidget(self.strat_hint)

        r.addWidget(_lab("Max steps"))
        self.max_steps = QLineEdit(str(D.MAX_STEPS))
        self.max_steps.setPlaceholderText(str(D.MAX_STEPS))
        self.max_steps.editingFinished.connect(self._normalize_fields)
        r.addWidget(self.max_steps)
        r.addWidget(_hint("Any integer ≥ 1, no product cap. Epoch = steps / training views."))

        r.addWidget(_lab("Gaussian budget"))
        self.budget = QLineEdit(str(D.GAUSSIAN_BUDGET))
        self.budget.setPlaceholderText(str(D.GAUSSIAN_BUDGET))
        self.budget.editingFinished.connect(self._normalize_fields)
        r.addWidget(self.budget)
        r.addWidget(_hint("LiteGS: TamingGS target primitives. MCMC Compact: importance-prune target."))

        r.addWidget(_lab("Resolution scale"))
        res = QWidget()
        res_l = QHBoxLayout(res)
        res_l.setContentsMargins(0, 0, 0, 0)
        res_l.setSpacing(0)
        self.res_group = QButtonGroup(self)
        self.res_group.setExclusive(True)
        self.res_btns: list[QPushButton] = []
        for i, s in enumerate(D.RESOLUTION_SCALES):
            btn = QPushButton(f"{s:g}")
            btn.setCheckable(True)
            btn.setProperty("class", "modeChip")
            self.res_group.addButton(btn, i)
            res_l.addWidget(btn)
            self.res_btns.append(btn)
        self.res_btns[0].setChecked(True)
        self.res_group.idClicked.connect(self._on_res)
        r.addWidget(res)

        sh_row = QHBoxLayout()
        sh_row.addWidget(_lab(f"SH degree {D.SH_DEGREE}"))
        sh_row.addStretch(1)
        self.holdout = QCheckBox("Hold-out eval (every 8th view)")
        self.holdout.setChecked(True)
        sh_row.addWidget(self.holdout)
        r.addLayout(sh_row)

        row1 = QHBoxLayout()
        self.btn_train = QPushButton("Train")
        self.btn_train.setObjectName("btnPrimary")
        self.btn_train.clicked.connect(lambda: self._emit("train"))
        self.btn_pause = QPushButton("Pause")
        self.btn_pause.setObjectName("btnGhost")
        self.btn_pause.clicked.connect(self._on_pause)
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setObjectName("btnGhost")
        self.btn_stop.clicked.connect(self.requestStop.emit)
        row1.addWidget(self.btn_train)
        row1.addWidget(self.btn_pause)
        row1.addWidget(self.btn_stop)
        r.addLayout(row1)

        row2 = QHBoxLayout()
        self.btn_continue = QPushButton("Continue")
        self.btn_continue.setObjectName("btnGhost")
        self.btn_continue.clicked.connect(lambda: self._emit("continue"))
        self.btn_retrain = QPushButton("Retrain")
        self.btn_retrain.setObjectName("btnGhost")
        self.btn_retrain.clicked.connect(lambda: self._emit("retrain"))
        row2.addWidget(self.btn_continue)
        row2.addWidget(self.btn_retrain)
        r.addLayout(row2)

        r.addWidget(_lab("Export color space"))
        self.color_space = QComboBox()
        for key, label in D.COLOR_SPACE_CHOICES:
            self.color_space.addItem(label, key)
        self.color_space.setCurrentIndex(0)
        self.color_space.setToolTip(
            "Unreal / MLSLabs (default): full SH, prune faint Gaussians, fatten needles. "
            "Unreal / Luma: DC only, same prune/needles, cap 1.9M so Niagara does not "
            "chunk — place the _Baked blueprint, not _Dynamic. 3DGS / sRGB is the raw "
            "trained file for Brush/Postshot. Linear is sRGB-to-linear for other engines."
        )
        r.addWidget(self.color_space)
        r.addWidget(
            _hint(
                "Unreal / MLSLabs: import the PLY in MLSLabsRenderer, actor scale 1. "
                "Unreal / Luma: drag into the Content Browser, place _Baked, scale 1."
            )
        )

        row3 = QHBoxLayout()
        self.btn_export = QPushButton("Export splat.ply")
        self.btn_export.setObjectName("btnGhost")
        self.btn_export.clicked.connect(lambda: self._emit("export"))
        self.btn_export_compact = QPushButton("Export compact")
        self.btn_export_compact.setObjectName("btnGhost")
        self.btn_export_compact.clicked.connect(lambda: self._emit("export_compact"))
        row3.addWidget(self.btn_export)
        row3.addWidget(self.btn_export_compact)
        r.addLayout(row3)
        self.drop_sh = QCheckBox("Compact export drops f_rest bands (DC only)")
        r.addWidget(self.drop_sh)

        self.btn_colmap = QPushButton("Open COLMAP folder")
        self.btn_colmap.setObjectName("btnGhost")
        self.btn_colmap.clicked.connect(self._open_colmap)
        r.addWidget(self.btn_colmap)

        r.addWidget(_hint("If the LiteGS spike failed: Train disabled; Open COLMAP folder still works (Brush)."))
        self.gate_hint = _hint("")
        r.addWidget(self.gate_hint)
        r.addStretch(1)
        r.addWidget(_hint(D.LICENSE_LINE))
        self._set_strat_hint()
        return rail

    # -------------------------------------------------------------- viewport
    def _build_view(self) -> QWidget:
        view = QWidget()
        view.setObjectName("viewport")
        v = QVBoxLayout(view)
        v.setContentsMargins(8, 8, 8, 8)
        v.setSpacing(6)
        hdr = QHBoxLayout()
        lbl = QLabel("Live splat")
        lbl.setProperty("class", "viewLbl")
        self.cam_lbl = QLabel("Star")
        self.cam_lbl.setProperty("class", "fHint")
        self.btn_gt = QPushButton("GT compare")
        self.btn_gt.setCheckable(True)
        self.btn_gt.setProperty("class", "rail-mini")
        self.btn_gt.setEnabled(False)
        self.btn_gt.toggled.connect(self._on_gt)
        self.btn_reset = QPushButton("reset view")
        self.btn_reset.setProperty("class", "rail-mini")
        hdr.addWidget(lbl)
        hdr.addWidget(self.cam_lbl)
        hdr.addStretch(1)
        hdr.addWidget(self.btn_gt)
        hdr.addWidget(self.btn_reset)
        v.addLayout(hdr)
        self.live = SplatLiveView(self._view_json)
        self.live.viewChanged.connect(self._on_view_changed)
        self.live.frameShown.connect(lambda *_a: self._update_hud())
        self.btn_reset.clicked.connect(self._reset_view)
        v.addWidget(self.live, 1)
        self.spark = Sparkline()
        v.addWidget(self.spark)
        self.view_hint = _hint(
            "Reset is the rails Star. LMB look · RMB pan · wheel dolly (uncapped) · "
            "hold LMB + WASD/QE to fly (Shift sprints). Click a training camera to jump there."
        )
        v.addWidget(self.view_hint)
        return view

    def _build_cameras(self) -> QWidget:
        panel = QWidget()
        panel.setObjectName("rail")
        panel.setFixedWidth(230)
        c = QVBoxLayout(panel)
        c.setContentsMargins(10, 12, 10, 12)
        c.setSpacing(4)
        c.addWidget(_lab("Training cameras"))
        self.cam_list = QListWidget()
        self.cam_list.setObjectName("camList")
        self.cam_list.setUniformItemSizes(True)
        self.cam_list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        pal = self.cam_list.palette()
        pal.setColor(QPalette.ColorRole.Base, QColor(8, 16, 24))
        pal.setColor(QPalette.ColorRole.AlternateBase, QColor(8, 16, 24))
        pal.setColor(QPalette.ColorRole.Text, QColor(201, 212, 227))
        pal.setColor(QPalette.ColorRole.Highlight, QColor(255, 159, 48))
        pal.setColor(QPalette.ColorRole.HighlightedText, QColor(11, 22, 35))
        self.cam_list.setPalette(pal)
        self.cam_list.viewport().setPalette(pal)
        self.cam_list.viewport().setAutoFillBackground(True)
        self.cam_list.itemClicked.connect(self._on_camera_item)
        c.addWidget(self.cam_list, 1)
        self.cam_hint = _hint("Filled by the worker (SPLAT cameras). Hold-out views are marked.")
        c.addWidget(self.cam_hint)
        return panel

    # ----------------------------------------------------------- rail events
    def _steps(self) -> int:
        return _int_text(self.max_steps.text(), int(D.MAX_STEPS))

    def _budget(self) -> int:
        return _int_text(self.budget.text(), int(D.GAUSSIAN_BUDGET))

    def _normalize_fields(self) -> None:
        n = self._steps()
        if self.max_steps.text().strip() != str(n):
            self.max_steps.setText(str(n))
        b = self._budget()
        if self.budget.text().strip() != str(b):
            self.budget.setText(str(b))
        self._update_hud()

    def _set_strat_hint(self) -> None:
        if self._strategy == "mcmc_compact":
            self.strat_hint.setText(
                "MCMC Compact (Jung & Hong 2025, 3.1): Metropolis-Hastings adaptive split "
                "threshold keeps the split ratio at target; stochastic importance prune to the budget. "
                "Same Gaussians in the viewport. Switching controller needs Retrain."
            )
        else:
            self.strat_hint.setText(
                "LiteGS (default): TamingGS densify scores, budget = target primitives, "
                "opacity decay, weight prune. Continue keeps this checkpoint; switching controller needs Retrain."
            )

    def _on_strat(self, idx: int) -> None:
        self._strategy = "mcmc_compact" if idx == 1 else "litegs"
        self._set_strat_hint()
        self.refresh()

    def _on_res(self, idx: int) -> None:
        try:
            self._res_scale = float(D.RESOLUTION_SCALES[idx])
        except (IndexError, ValueError):
            self._res_scale = 1.0

    def _on_pause(self) -> None:
        if self._worker_kind != "train":
            return
        if self._paused:
            self.controlLine.emit("RESUME")
        else:
            self.controlLine.emit("PAUSE")

    def _color_space_key(self) -> str:
        data = self.color_space.currentData()
        if data:
            return D.color_space_from_key(str(data))
        return D.COLOR_SPACE_DEFAULT

    def _emit(self, mode: str) -> None:
        self._normalize_fields()
        if mode == "export_compact" and self.drop_sh.isChecked():
            mode = "export_compact_dc"
        self.requestJob.emit(
            [
                "__splat__",
                self._scene_name(),
                self.strategy_key(),
                mode,
                str(self._steps()),
                str(self._budget()),
                f"{self._res_scale:g}",
                "1" if self.holdout.isChecked() else "0",
                self._color_space_key(),
            ]
        )

    def _open_colmap(self) -> None:
        folder = scene_dir(self._scene_name()) / "colmap"
        if folder.is_dir():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))

    # ------------------------------------------------------- viewport events
    def _on_view_changed(self, payload: dict) -> None:
        if self._worker_kind:
            self.controlLine.emit("VIEW\t" + json.dumps(payload))
        self._update_cam_label()

    def _reset_view(self) -> None:
        self.btn_gt.setChecked(False)
        self.live.reset_view()
        self.cam_list.clearSelection()
        self._update_cam_label()

    def _on_gt(self, on: bool) -> None:
        self.live.set_gt(on)
        self._update_cam_label()

    def _on_camera_item(self, item: QListWidgetItem) -> None:
        idx = item.data(Qt.ItemDataRole.UserRole)
        if idx is None:
            return
        self.live.set_camera(int(idx))
        self.btn_gt.setEnabled(True)
        if self.btn_gt.isChecked():
            self.live.set_gt(True)
        self._update_cam_label()

    def _update_cam_label(self) -> None:
        idx = self.live.camera_index()
        if idx < 0:
            self.cam_lbl.setText("Star")
            self.btn_gt.setEnabled(False)
            if self.btn_gt.isChecked():
                self.btn_gt.blockSignals(True)
                self.btn_gt.setChecked(False)
                self.btn_gt.blockSignals(False)
            return
        name = ""
        if 0 <= idx < len(self._cameras):
            name = str(self._cameras[idx].get("name", ""))
        self.cam_lbl.setText(f"cam {idx} {name}" + (" · render | GT" if self.live.state().gt else ""))

    def _fill_cameras(self, cams: list[dict]) -> None:
        self._cameras = cams
        self.cam_list.clear()
        by_rail: dict[str, list[dict]] = {}
        for cam in cams:
            by_rail.setdefault(_cam_rail_key(cam), []).append(cam)
        holdout = 0
        ink = QColor(201, 212, 227)
        for rail in sorted(by_rail, key=_rail_sort_key):
            rows = by_rail[rail]
            head = QListWidgetItem(_rail_heading(rail, len(rows)))
            head.setFlags(Qt.ItemFlag.NoItemFlags)
            head.setForeground(QColor(255, 159, 48))
            self.cam_list.addItem(head)
            for cam in rows:
                name = str(cam.get("name", ""))
                m = _FRAME_RE.search(name)
                label = f"frame {int(m.group(1))} · face {int(m.group(2))}" if m else name
                if cam.get("holdout"):
                    label += "  [hold-out]"
                    holdout += 1
                item = QListWidgetItem(label)
                item.setData(Qt.ItemDataRole.UserRole, int(cam.get("i", 0)))
                item.setForeground(QColor(46, 200, 120) if cam.get("holdout") else ink)
                self.cam_list.addItem(item)
        self.cam_hint.setText(
            f"{len(cams)} views, {holdout} hold-out. Click to jump; GT compare shows render | ground truth."
        )

    # ------------------------------------------------------------- worker IO
    def on_worker_started(self, kind: str) -> None:
        """``kind``: 'train' (job) or 'view' (idle viewer)."""
        self._worker_kind = kind
        self._paused = False
        self._shm_pending = ""
        if kind == "train":
            self.spark.clear()
            self._stats = {}
        self._set_state("LOADING")

    def on_worker_finished(self) -> None:
        self._worker_kind = ""
        self._paused = False
        self.live.detach()
        self._set_state("IDLE")

    def on_splat_line(self, parts: list[str]) -> None:
        """One ``SPLAT\\t<kind>\\t...`` line from either splat worker."""
        kind = parts[1] if len(parts) > 1 else ""
        args = parts[2:]
        if kind == "shm" and args:
            if not self.live.attach(args[0]):
                self._shm_pending = args[0]
                self.logLine.emit(f"STATUS // viewport: shared memory {args[0]} attach failed; PNG fallback")
            else:
                self.live.request_view()
        elif kind == "cameras" and args:
            try:
                cams = json.loads("\t".join(args))
            except ValueError as exc:
                self.logLine.emit(f"STATUS // training cameras JSON failed: {exc}")
                return
            if isinstance(cams, list):
                self._fill_cameras(cams)
        elif kind == "state" and args:
            self._set_state(args[0])
        elif kind == "step" and len(args) >= 2:
            self._stats["step"] = args[0]
            self._stats["max"] = args[1]
        elif kind == "epoch" and len(args) >= 2:
            self._stats["epoch"] = args[0]
            self._stats["epochs"] = args[1]
        elif kind == "loss" and args:
            self._stats["loss"] = args[0]
            try:
                self.spark.add(float(args[0]), None)
            except ValueError:
                pass
        elif kind == "psnr" and args:
            self._stats["psnr"] = args[0]
            self._stats["psnr_eval"] = args[1] if len(args) > 1 else "-"
            try:
                self.spark.add(None, float(args[0]))
            except ValueError:
                pass
        elif kind == "gaussians" and args:
            self._stats["gaussians"] = args[0]
        elif kind == "budget" and args:
            self._stats["budget"] = args[0]
        elif kind == "its" and args:
            self._stats["its"] = args[0]
        elif kind == "vram" and args:
            self._stats["vram"] = args[0]
        elif kind == "fps" and args:
            self._stats["fps"] = args[0]
        elif kind == "strategy" and args:
            self._stats["strategy"] = args[0]
        elif kind == "preview" and args:
            self.live.load(args[0])
        elif kind == "ply" and args:
            self.view_hint.setText(f"Exported {args[0]}")
        elif kind == "error" and args:
            self.gate_hint.setText(args[0])
        self._update_hud()

    def _set_state(self, state: str) -> None:
        self._state = state
        self._paused = state == "PAUSED"
        self.btn_pause.setText("Resume" if self._paused else "Pause")
        self.btn_pause.setEnabled(self._worker_kind == "train" and state in ("TRAINING", "PAUSED"))
        self._update_hud()

    def _update_hud(self) -> None:
        s = self._stats
        lines: list[str] = []
        step = s.get("step")
        if step:
            ep = f"  epoch {s.get('epoch', '?')}/{s.get('epochs', '?')}" if s.get("epoch") else ""
            lines.append(f"step {step} / {s.get('max', self._steps())}{ep}")
        if s.get("loss"):
            psnr = s.get("psnr")
            ev = s.get("psnr_eval", "-")
            text = f"loss {s['loss']}"
            if psnr:
                text += f"   PSNR {psnr}" + (f" (eval {ev})" if ev and ev != "-" else "")
            lines.append(text)
        if s.get("gaussians"):
            lines.append(f"gaussians {int(s['gaussians']):,} / {int(s.get('budget', self._budget())):,}")
        perf = []
        if s.get("its"):
            perf.append(f"{s['its']} it/s")
        fps = self.live.fps if self.live.attached else 0.0
        if fps > 0:
            perf.append(f"{fps:.0f} fps")
        if s.get("vram"):
            perf.append(f"{s['vram']} MB")
        if s.get("strategy"):
            perf.append(s["strategy"])
        if perf:
            lines.append("   ".join(perf))
        self.live.set_hud(self._state, lines)

    # ------------------------------------------------------------- compat
    def apply_stats(self, **kw) -> None:
        """Old entry point (step/loss/gaussians/ply/strategy/preview keywords)."""
        for key, val in kw.items():
            if not val:
                continue
            if key == "step":
                self.on_splat_line(["SPLAT", "step", str(val), str(kw.get("max_steps") or self._steps())])
            elif key in ("loss", "gaussians", "ply", "strategy", "preview"):
                self.on_splat_line(["SPLAT", key, str(val)])

    # ------------------------------------------------------------- refresh
    def _colmap_ok(self, root: Path) -> bool:
        return (root / "colmap" / "sparse" / "0" / "cameras.bin").is_file()

    def _meta(self, root: Path) -> dict:
        path = root / "checkpoints" / "splat_meta.json"
        if not path.is_file():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _poll_preview(self) -> None:
        if not self.isVisible() or self.live.attached:
            return
        png = self._preview_png()
        if png.is_file():
            self.live.load(png)

    def refresh(self) -> None:
        root = scene_dir(self._scene_name())
        colmap_ok = self._colmap_ok(root)
        spike_ok = litegs_spike_ok()
        self.btn_colmap.setEnabled(colmap_ok)
        meta = self._meta(root)
        legacy = bool(meta) and meta.get("format") != "votion-litegs-1"
        ck_strat = str(meta.get("controller") or meta.get("strategy") or "")
        same = bool(ck_strat) and ck_strat == self.strategy_label() and not legacy
        ply = (root / "splat.ply").is_file()
        can_train = colmap_ok and spike_ok
        busy = self._worker_kind == "train"
        self.btn_train.setEnabled(can_train and not busy)
        self.btn_retrain.setEnabled(can_train and not busy)
        self.btn_continue.setEnabled(can_train and same and not busy and not bool(meta.get("done")))
        self.btn_export.setEnabled(colmap_ok and (ply or same) and not busy)
        self.btn_export_compact.setEnabled(colmap_ok and same and not busy)
        self.color_space.setEnabled(colmap_ok and (ply or same) and not busy)
        self.drop_sh.setEnabled(colmap_ok and same and not busy)
        steps = self._steps()
        if not colmap_ok:
            self.gate_hint.setText("Reconstruct first — Train needs colmap/sparse/0.")
        elif not spike_ok:
            self.gate_hint.setText(
                "LiteGS spike not OK — Train disabled. Home → Build LiteGS. "
                "Open COLMAP folder still works (Brush). " + litegs_spike_detail()
            )
        elif legacy:
            self.gate_hint.setText("Checkpoint is from the old gsplat trainer (params only) — Retrain required.")
        elif ck_strat and not same:
            self.gate_hint.setText(f"Checkpoint is {ck_strat}. Continue stays off until Retrain as {self.strategy_label()}.")
        elif meta.get("done"):
            self.gate_hint.setText(
                f"{ck_strat} finished at step {meta.get('step')} · N={meta.get('gaussians')}"
                + (f" · eval PSNR {meta.get('psnr_eval')}" if meta.get("psnr_eval") else "")
                + ". Raise Max steps and Continue, or Retrain."
            )
        else:
            self.gate_hint.setText(f"{self.strategy_label()} · max {steps} · budget {self._budget():,}.")
        if meta.get("step") and not busy:
            self._stats["step"] = str(int(meta["step"]))
            self._stats["max"] = str(int(meta.get("max_steps") or steps))
            if meta.get("gaussians"):
                self._stats["gaussians"] = str(int(meta["gaussians"]))
            if meta.get("budget"):
                self._stats["budget"] = str(int(meta["budget"]))
            if meta.get("psnr"):
                self._stats["psnr"] = f"{float(meta['psnr']):.2f}"
                self._stats["psnr_eval"] = f"{float(meta['psnr_eval']):.2f}" if meta.get("psnr_eval") else "-"
            if ck_strat:
                self._stats["strategy"] = ck_strat
        if not self.live.attached:
            png = self._preview_png()
            if png.is_file():
                self.live.load(png)
            elif not self._worker_kind:
                self.live.clear("Train to see the splat form.")
        if ply and not busy:
            self.view_hint.setText(
                "Reset is the rails Star. LMB look · RMB pan · wheel dolly (uncapped) · "
                "hold LMB + WASD/QE to fly.  " + str(root / "splat.ply")
            )
        self._update_hud()
