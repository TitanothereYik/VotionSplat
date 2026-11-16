"""Phase 3 Geometry page — dummy docs/Votion_3DGS_2.0_Phase3.html §geo-ui."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from app.widgets.geo_view import GeoPane
from engine.depth.keys import rails_stale
from engine.geo import defaults as D
from engine.geo.anchors import format_anchors, format_point, parse_anchors_ext
from engine.geo.rail import fill_targets, snap_length
from engine.geo.rails import (
    RailSpec,
    all_played,
    camplot_meta_rel,
    confirmed_matches,
    confirmed_path,
    control_video_rel,
    default_rails,
    first_knot_off_origin,
    load_editor,
    next_free_rail_index,
    rail_clips_wall,
    save_editor,
    seed_rails_for_scene,
    straight_rail,
    suggestion_to_rail,
    write_confirmed,
)
from engine.geo.suggest import load_scene_cloud, suggest_paths
from engine.pano.scene_io import pano_path, scene_dir


def _lab(text: str) -> QLabel:
    lab = QLabel(text)
    lab.setProperty("class", "fLab")
    return lab


class GeometryPage(QWidget):
    requestJob = Signal(list)
    requestStop = Signal()
    railsChanged = Signal()

    def __init__(self, scene_name_fn, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._scene_name = scene_name_fn
        self._geo_on = False
        self._rails: list[RailSpec] = default_rails()
        self._cloud_pts: np.ndarray | None = None
        self._active = 0
        self._updating = False
        self._loaded_scene = ""
        self._video_path: Path | None = None
        self._preview_cap = None
        self._preview_timer = QTimer(self)
        self._preview_timer.setInterval(100)
        self._preview_timer.timeout.connect(self._preview_tick)

        body = QHBoxLayout(self)
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)
        body.addWidget(self._build_rail())
        body.addWidget(self._build_view(), 1)
        self._set_geo_state(False)
        self.try_load_existing()

    def _rail(self) -> RailSpec:
        if not self._rails:
            self._rails = [straight_rail(0)]
            self._active = 0
        self._active = max(0, min(self._active, len(self._rails) - 1))
        return self._rails[self._active]

    def _build_rail(self) -> QWidget:
        rail = QWidget()
        rail.setObjectName("rail")
        rail.setFixedWidth(280)
        r = QVBoxLayout(rail)
        r.setContentsMargins(12, 12, 12, 14)
        r.setSpacing(4)

        r.addWidget(_lab("Scene pano"))
        self.pano_thumb = QLabel("No pano.png — run Phase 1 first")
        self.pano_thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.pano_thumb.setMinimumHeight(48)
        self.pano_thumb.setMaximumHeight(72)
        self.pano_thumb.setStyleSheet(
            "border: 1px dashed #2a3d52; background: #081018; color: #8b9bb4; font-size: 10px;"
        )
        self.pano_thumb.setScaledContents(True)
        r.addWidget(self.pano_thumb)

        row = QHBoxLayout()
        self.btn_compute = QPushButton("Compute geometry")
        self.btn_compute.setObjectName("btnGhost")
        self.btn_compute.clicked.connect(self._on_compute)
        self.btn_suggest = QPushButton("Suggest paths")
        self.btn_suggest.setObjectName("btnGhost")
        self.btn_suggest.setEnabled(False)
        self.btn_suggest.clicked.connect(self._on_suggest)
        row.addWidget(self.btn_compute)
        row.addWidget(self.btn_suggest)
        r.addLayout(row)

        r.addWidget(_lab("Active rail"))
        self.active_combo = QComboBox()
        self.active_combo.currentIndexChanged.connect(self._on_active_combo)
        r.addWidget(self.active_combo)

        r.addWidget(_lab("Look mode"))
        self.look_mode = QComboBox()
        for name in D.LOOK_MODES:
            self.look_mode.addItem(name)
        self.look_mode.setCurrentText(D.LOOK_DEFAULT)
        self.look_mode.currentTextChanged.connect(self._on_look_mode)
        r.addWidget(self.look_mode)

        r.addWidget(_lab("look_at_target"))
        self.look_edit = QLineEdit(D.LOOK_AT_DEFAULT)
        self.look_edit.editingFinished.connect(self._on_look_text)
        r.addWidget(self.look_edit)

        r.addWidget(_lab("Anchors (xyz)"))
        self.anchors = QTextEdit()
        self.anchors.setPlainText(D.ANCHORS_DEFAULT)
        self.anchors.setMinimumHeight(84)
        self.anchors.textChanged.connect(self._on_anchors_text)
        r.addWidget(self.anchors)
        self.origin_warn = QLabel("")
        self.origin_warn.setWordWrap(True)
        self.origin_warn.setProperty("class", "fHint")
        self.origin_warn.setStyleSheet("color: #ff9a32;")
        r.addWidget(self.origin_warn)
        self.wall_hint = QLabel("")
        self.wall_hint.setWordWrap(True)
        self.wall_hint.setProperty("class", "fHint")
        self.wall_hint.setStyleSheet("color: #ff9a32;")
        r.addWidget(self.wall_hint)

        r.addWidget(_lab("length"))
        self.length = QSpinBox()
        self.length.setRange(D.LENGTH_MIN, D.LENGTH_MAX)
        self.length.setSingleStep(D.LENGTH_STEP)
        self.length.setValue(D.LENGTH_DEFAULT)
        self.length.valueChanged.connect(self._mark_stale)
        r.addWidget(self.length)

        r.addWidget(_lab("moge_level (shared, not fps)"))
        self.moge_level = QSpinBox()
        self.moge_level.setRange(0, 9)
        self.moge_level.setValue(D.MOGE_LEVEL)
        self.moge_level.setEnabled(False)
        r.addWidget(self.moge_level)

        add_row = QHBoxLayout()
        self.btn_add = QPushButton("+ Add rail")
        self.btn_add.setObjectName("btnGhost")
        self.btn_add.setEnabled(False)
        self.btn_add.clicked.connect(self._on_add_rail)
        add_row.addWidget(self.btn_add)
        self.btn_remove = QPushButton("- Remove rail")
        self.btn_remove.setObjectName("btnGhost")
        self.btn_remove.setEnabled(False)
        self.btn_remove.setToolTip(
            "Remove the active rail. Rail 1 (default) stays. Minimum is 1; Add fills back up to 5."
        )
        self.btn_remove.clicked.connect(self._on_remove_rail)
        add_row.addWidget(self.btn_remove)
        r.addLayout(add_row)

        self.btn_confirm = QPushButton("Confirm rails")
        self.btn_confirm.setObjectName("btnPrimary")
        self.btn_confirm.setEnabled(False)
        self.btn_confirm.clicked.connect(self._on_confirm)
        r.addWidget(self.btn_confirm)
        note = QLabel(
            "Active rail solid; others dimmed. Star locked. Play each rail on the "
            "Preview pane, then Confirm rails gates Generate. Remove the active rail "
            "down to rail 1 (default); Add fills back up to 5."
        )
        note.setWordWrap(True)
        note.setProperty("class", "fHint")
        r.addWidget(note)
        r.addStretch(1)
        self._fill_active_combo()
        return rail

    def _build_view(self) -> QWidget:
        view = QWidget()
        view.setObjectName("viewport")
        v = QVBoxLayout(view)
        v.setContentsMargins(8, 8, 8, 8)
        self.floor = GeoPane("floor", "FLOOR X → Z ↑ · drag 1–3 + orange look-at")
        self.side = GeoPane("side", "SIDE Z → Y ↑")
        self.floor.knotMoved.connect(self._on_floor_knot)
        self.side.knotMoved.connect(self._on_side_knot)
        self.floor.lookMoved.connect(self._on_floor_look)
        self.side.lookMoved.connect(self._on_side_look)
        self.floor.lookTargetMoved.connect(self._on_floor_look_target)
        self.side.lookTargetMoved.connect(self._on_side_look_target)
        self.floor.dragFinished.connect(self._on_gizmo_done)
        self.side.dragFinished.connect(self._on_gizmo_done)
        preview = QWidget()
        preview.setObjectName("previewFlight")
        pv = QVBoxLayout(preview)
        pv.setContentsMargins(6, 4, 6, 6)
        row = QHBoxLayout()
        lab = QLabel("Preview flight")
        lab.setProperty("class", "viewLbl")
        self.btn_play = QPushButton("Play")
        self.btn_play.setObjectName("btnGhost")
        self.btn_play.setEnabled(False)
        self.btn_play.clicked.connect(self._on_play)
        row.addWidget(lab)
        row.addStretch(1)
        row.addWidget(self.btn_play)
        pv.addLayout(row)
        self.preview_hint = QLabel("Empty until Compute geometry. Play lives on this pane, not the rail.")
        self.preview_hint.setWordWrap(True)
        self.preview_hint.setProperty("class", "fHint")
        pv.addWidget(self.preview_hint)
        self.preview_frame = QLabel()
        self.preview_frame.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_frame.setMinimumHeight(96)
        self.preview_frame.setStyleSheet("background: #081018; border: 1px solid #1c2b3a;")
        self.preview_frame.setScaledContents(False)
        pv.addWidget(self.preview_frame, 1)
        pair = QSplitter(Qt.Orientation.Horizontal)
        pair.addWidget(self.floor)
        pair.addWidget(self.side)
        pair.setStretchFactor(0, 1)
        pair.setStretchFactor(1, 1)
        pair.setChildrenCollapsible(False)
        split = QSplitter(Qt.Orientation.Vertical)
        split.addWidget(pair)
        split.addWidget(preview)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 1)
        split.setChildrenCollapsible(False)
        v.addWidget(split, 1)
        return view

    def generate_unlocked(self) -> bool:
        return confirmed_matches(self._scene_name(), self._rails, int(self.moge_level.value()))

    def refresh_pano(self) -> None:
        path = pano_path(self._scene_name())
        if path.is_file():
            pix = QPixmap(str(path))
            if not pix.isNull():
                self.pano_thumb.setPixmap(
                    pix.scaled(
                        self.pano_thumb.size() if self.pano_thumb.width() > 8 else pix.size(),
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation,
                    )
                )
                self.pano_thumb.setText("")
            else:
                self.pano_thumb.setText(path.name)
            self.btn_compute.setEnabled(True)
        else:
            self.pano_thumb.setPixmap(QPixmap())
            self.pano_thumb.setText("No pano.png — run Phase 1 first")
            self.btn_compute.setEnabled(False)

    def _clear_plates(self) -> None:
        self.floor.set_ortho(None, [-1.0, -1.0], [1.0, 1.0])
        self.side.set_ortho(None, [-1.0, -1.0], [1.0, 1.0])
        self.floor.set_cloud(None, None)
        self.side.set_cloud(None, None)
        self._cloud_pts = None
        self.wall_hint.setText("")

    def release_open_media(self) -> None:
        """Close the flight video and drop the plates so Start over can delete them."""
        self._stop_preview()
        self.preview_frame.clear()
        self._video_path = None
        self._clear_plates()

    def _set_geo_state(self, on: bool) -> None:
        self._geo_on = on
        self.look_mode.setEnabled(on)
        self.look_edit.setEnabled(on)
        self.anchors.setEnabled(on)
        self.length.setEnabled(on)
        self.active_combo.setEnabled(on)
        self.btn_suggest.setEnabled(on)
        self._sync_rail_count_buttons()
        self.floor.set_editor_enabled(on)
        self.side.set_editor_enabled(on)
        self.btn_play.setEnabled(on)
        if not on:
            self.btn_confirm.setEnabled(False)
            self._stop_preview()

    def _on_compute(self) -> None:
        scene = self._scene_name()
        path = pano_path(scene)
        if not path.is_file():
            self.requestJob.emit(["__geo_error__", "missing pano.png — run Phase 1 first"])
            return
        self.requestJob.emit(["__moge__", scene])

    def apply_geometry(self, json_path: str) -> None:
        meta_path = Path(json_path)
        if not meta_path.is_file():
            return
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        moge_dir = meta_path.parent
        floor = moge_dir / "floor.png"
        side = moge_dir / "side.png"
        fv = meta.get("views", {}).get("floor") or {}
        sv = meta.get("views", {}).get("side") or {}
        if fv:
            self.floor.set_ortho(floor if floor.is_file() else None, fv.get("lo", [-1, -1]), fv.get("hi", [1, 1]))
        if sv:
            self.side.set_ortho(side if side.is_file() else None, sv.get("lo", [-1, -1]), sv.get("hi", [1, 1]))
        npz = moge_dir / "cloud.npz"
        if npz.is_file():
            with np.load(npz) as data:
                pts = np.array(data["points"], copy=True)
                cols = np.array(data["colors"], copy=True)
            self.floor.set_cloud(pts, cols)
            self.side.set_cloud(pts, cols)
            self._cloud_pts = pts
        self._set_geo_state(True)
        self._push_gizmos()
        self._sync_confirm()
        if self.generate_unlocked():
            self.preview_hint.setText("Confirm rails is on. Generate is unlocked.")
        else:
            self.preview_hint.setText(
                "Geometry on. Play each rail on this pane (black holes OK), then Confirm rails."
            )

    def try_load_existing(self) -> None:
        scene = self._scene_name()
        scene_changed = scene != self._loaded_scene
        if scene_changed:
            self._stop_preview()
            self.preview_frame.clear()
            self._video_path = None
            self._load_rails(scene)
        self._loaded_scene = scene
        meta = scene_dir(scene) / "moge" / "scene_ref.json"
        if meta.is_file():
            self.apply_geometry(str(meta))
            self._restore_preview_pane()
        else:
            self._clear_plates()
            self.preview_frame.clear()
            self._video_path = None
            self._set_geo_state(False)
            self.preview_hint.setText(
                "Empty until Compute geometry. Play lives on this pane, not the rail."
            )
        self.refresh_pano()
        self._show_stale_rails(scene)
        self.railsChanged.emit()

    def _show_stale_rails(self, scene: str) -> None:
        stale = rails_stale(scene)
        if not stale:
            return
        listed = ", ".join(str(index + 1) for index in stale)
        self.preview_hint.setText(
            f"Rails were plotted on an older depth: re-plot rails {listed} in Geometry."
        )

    def restore_view(self) -> None:
        scene = self._scene_name()
        if scene != self._loaded_scene or not self._geo_on:
            self.try_load_existing()
            return
        self.refresh_pano()
        self._push_gizmos()
        self._restore_preview_pane()

    def _load_rails(self, scene: str) -> None:
        loaded = load_editor(scene)
        if loaded is not None:
            self._rails, moge, active = loaded
            self._updating = True
            self.moge_level.setValue(moge)
            self._updating = False
            self._active = active
        else:
            self._rails = seed_rails_for_scene(scene)
            self._active = 0
        self._fill_active_combo()
        self._push_rail_to_widgets()
        self._persist()

    def _fill_active_combo(self) -> None:
        self._updating = True
        self.active_combo.blockSignals(True)
        self.active_combo.clear()
        for spec in self._rails:
            self.active_combo.addItem(spec.combo_text(), spec.index)
        self.active_combo.setCurrentIndex(self._active)
        self.active_combo.blockSignals(False)
        self._updating = False
        self._sync_rail_count_buttons()

    def _sync_rail_count_buttons(self) -> None:
        self.btn_add.setEnabled(self._geo_on and len(self._rails) < D.MAX_RAILS)
        spec = self._rails[self._active] if self._rails else None
        self.btn_remove.setEnabled(
            self._geo_on
            and spec is not None
            and len(self._rails) > D.MIN_RAILS
            and spec.index != 0
        )

    def _select_rail_index(self, rail_index: int) -> None:
        for i, spec in enumerate(self._rails):
            if spec.index == rail_index:
                self._active = i
                return
        self._active = 0

    def _on_active_combo(self, index: int) -> None:
        if self._updating or index < 0:
            return
        self._pull_widgets_to_rail()
        self._stop_preview()
        self._active = index
        self._push_rail_to_widgets()
        self._push_gizmos()
        self._sync_rail_count_buttons()
        self._save_editor()
        self._restore_preview_pane()

    def _pull_widgets_to_rail(self) -> None:
        spec = self._rail()
        spec.orientation = self.look_mode.currentText()
        spec.anchors = self.anchors.toPlainText()
        spec.look_at = self.look_edit.text()
        spec.length = snap_length(int(self.length.value()))

    def _push_rail_to_widgets(self) -> None:
        spec = self._rail()
        self._updating = True
        self.look_mode.setCurrentText(spec.orientation)
        self.anchors.setPlainText(spec.anchors)
        self.look_edit.setText(spec.look_at)
        self.length.setValue(snap_length(spec.length))
        self._updating = False
        self._refresh_origin_warn()
        self._refresh_wall_hint()

    def _save_editor(self) -> None:
        save_editor(self._scene_name(), self._rails, int(self.moge_level.value()), self._active)

    def _persist(self) -> None:
        self._save_editor()
        self.railsChanged.emit()

    def _refresh_origin_warn(self) -> None:
        if first_knot_off_origin(self.anchors.toPlainText()):
            self.origin_warn.setText(
                "First knot is far from origin — editor locks knot 0 to the star. "
                "Coverage will be mostly the generated frames if a rail does not start at the pano center."
            )
        else:
            self.origin_warn.setText("")

    def _refresh_wall_hint(self) -> None:
        points = self._cloud_pts
        if points is None or getattr(points, "size", 0) == 0:
            self.wall_hint.setText("")
            return
        clipping = [spec for spec in self._rails if rail_clips_wall(spec, points)]
        if not clipping:
            self.wall_hint.setText("")
            return
        names = ", ".join(f"rail {spec.index + 1}" for spec in clipping)
        self.wall_hint.setText(
            f"Clips wall: {names}. A sampled point is past the cloud. Confirm still works; shorten the rail if you want."
        )

    def _show_video_still(self, path: Path) -> None:
        import cv2

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            return
        ok, frame = cap.read()
        cap.release()
        if not ok:
            return
        height, width, channels = frame.shape
        qimg = QImage(
            frame.data, width, height, channels * width, QImage.Format.Format_BGR888
        ).copy()
        pix = QPixmap.fromImage(qimg)
        target = self.preview_frame.size()
        if target.width() > 8 and target.height() > 8:
            pix = pix.scaled(
                target,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        self.preview_frame.setPixmap(pix)

    def _restore_preview_pane(self) -> None:
        if self._preview_timer.isActive():
            return
        video = self._matching_video()
        if video is None:
            return
        pix = self.preview_frame.pixmap()
        if pix is None or pix.isNull():
            self._show_video_still(video)
            self._video_path = video
        spec = self._rail()
        if spec.played or video is not None:
            if self.generate_unlocked():
                self.preview_hint.setText("Confirm rails is on. Generate is unlocked.")
            elif spec.played:
                self.preview_hint.setText(
                    f"Rail {spec.index + 1} previewed. Play the other rails, then Confirm rails."
                )

    def _on_anchors_text(self) -> None:
        if self._updating:
            return
        try:
            parse_anchors_ext(self.anchors.toPlainText())
        except ValueError:
            return
        self._rail().anchors = self.anchors.toPlainText()
        self._refresh_origin_warn()
        self._push_gizmos()
        self._mark_stale()

    def _on_look_text(self) -> None:
        if self._updating:
            return
        self._rail().look_at = self.look_edit.text()
        self._push_gizmos()
        self._mark_stale()

    def _other_rails(self) -> list[dict]:
        out: list[dict] = []
        for i, spec in enumerate(self._rails):
            if i == self._active:
                continue
            try:
                knots, _ = spec.knots_and_targets()
            except ValueError:
                continue
            out.append({"knots": knots, "n_path": snap_length(spec.length)})
        return out

    def _push_gizmos(self, *, live: bool = False) -> None:
        spec = self._rail()
        try:
            knots, targets = spec.knots_and_targets()
        except ValueError:
            return
        look = spec.look_point()
        mode = spec.orientation
        n_path = snap_length(int(self.length.value()))
        others = self._other_rails()
        self.floor.set_knots(knots, look, mode=mode, targets=targets, n_path=n_path, others=others)
        self.side.set_knots(knots, look, mode=mode, targets=targets, n_path=n_path, others=others)
        if not live:
            self._refresh_wall_hint()

    def _write_anchor_box(self) -> None:
        spec = self._rail()
        self._updating = True
        six = spec.orientation == "per_point_look"
        try:
            knots, targets = spec.knots_and_targets()
        except ValueError:
            self._updating = False
            return
        spec.anchors = format_anchors(knots, targets, six=six)
        spec.look_at = format_point(spec.look_point())
        self.anchors.setPlainText(spec.anchors)
        self.look_edit.setText(spec.look_at)
        self._updating = False

    def _on_look_mode(self, _mode: str) -> None:
        if self._updating:
            return
        spec = self._rail()
        spec.orientation = self.look_mode.currentText()
        if spec.orientation != "look_at_target" and spec.label == "pond":
            spec.label = ""
        if spec.orientation == "per_point_look":
            try:
                knots, targets = spec.knots_and_targets()
            except ValueError:
                return
            if targets.shape[0] != knots.shape[0] or not np.any(np.isfinite(targets)):
                targets = fill_targets(knots, np.full_like(knots, np.nan))
                spec.anchors = format_anchors(knots, targets, six=True)
        self._write_anchor_box()
        self._fill_active_combo()
        self._push_gizmos()
        self._mark_stale()

    def _ensure_per_point_look(self) -> None:
        spec = self._rail()
        if spec.orientation == "per_point_look":
            return
        try:
            knots, _ = spec.knots_and_targets()
        except ValueError:
            return
        targets = fill_targets(knots, np.full_like(knots, np.nan))
        spec.orientation = "per_point_look"
        spec.anchors = format_anchors(knots, targets, six=True)
        if spec.label == "pond":
            spec.label = ""
        self._updating = True
        self.look_mode.setCurrentText("per_point_look")
        self._updating = False
        self._write_anchor_box()
        self._fill_active_combo()

    def _on_floor_knot(self, index: int, x: float, z: float) -> None:
        spec = self._rail()
        try:
            knots, targets = spec.knots_and_targets()
        except ValueError:
            return
        if index <= 0 or index >= len(knots):
            return
        knots[index, 0] = x
        knots[index, 2] = z
        spec.anchors = format_anchors(knots, targets, six=spec.orientation == "per_point_look")
        self._push_gizmos(live=True)

    def _on_side_knot(self, index: int, z: float, y: float) -> None:
        spec = self._rail()
        try:
            knots, targets = spec.knots_and_targets()
        except ValueError:
            return
        if index <= 0 or index >= len(knots):
            return
        knots[index, 2] = z
        knots[index, 1] = y
        spec.anchors = format_anchors(knots, targets, six=spec.orientation == "per_point_look")
        self._push_gizmos(live=True)

    def _on_floor_look(self, x: float, z: float) -> None:
        spec = self._rail()
        look = spec.look_point()
        look[0] = x
        look[2] = z
        spec.look_at = format_point(look)
        self._push_gizmos(live=True)

    def _on_side_look(self, z: float, y: float) -> None:
        spec = self._rail()
        look = spec.look_point()
        look[2] = z
        look[1] = y
        spec.look_at = format_point(look)
        self._push_gizmos(live=True)

    def _on_floor_look_target(self, index: int, x: float, z: float) -> None:
        self._ensure_per_point_look()
        spec = self._rail()
        try:
            knots, targets = spec.knots_and_targets()
        except ValueError:
            return
        if index < 0 or index >= len(knots):
            return
        if targets.shape[0] != knots.shape[0]:
            targets = fill_targets(knots, np.full_like(knots, np.nan))
        targets[index, 0] = x
        targets[index, 2] = z
        spec.anchors = format_anchors(knots, targets, six=True)
        self._push_gizmos(live=True)

    def _on_side_look_target(self, index: int, z: float, y: float) -> None:
        self._ensure_per_point_look()
        spec = self._rail()
        try:
            knots, targets = spec.knots_and_targets()
        except ValueError:
            return
        if index < 0 or index >= len(knots):
            return
        if targets.shape[0] != knots.shape[0]:
            targets = fill_targets(knots, np.full_like(knots, np.nan))
        targets[index, 2] = z
        targets[index, 1] = y
        spec.anchors = format_anchors(knots, targets, six=True)
        self._push_gizmos(live=True)

    def _on_gizmo_done(self) -> None:
        """Sync the text. Play saves. Release does not rebuild Generate or Reconstruct."""
        self._write_anchor_box()
        self._refresh_origin_warn()
        self._refresh_wall_hint()
        spec = self._rail()
        spec.played = False
        spec.dirty = True
        self._sync_confirm()
        if self._geo_on:
            self.preview_hint.setText(
                f"Rail {spec.index + 1} changed. Play the mesh flight on this pane before Confirm rails."
            )

    def _plot_payload(self) -> dict:
        self._pull_widgets_to_rail()
        return self._rail().payload(int(self.moge_level.value()))

    def _write_request(self) -> Path:
        root = scene_dir(self._scene_name())
        work = root / "_work"
        work.mkdir(parents=True, exist_ok=True)
        path = work / "camplot_request.json"
        path.write_text(json.dumps(self._plot_payload(), indent=2) + "\n", encoding="utf-8")
        return path

    def _matching_video(self) -> Path | None:
        spec = self._rail()
        root = scene_dir(self._scene_name())
        meta_path = root / camplot_meta_rel(spec.index)
        video = root / control_video_rel(spec.index)
        if spec.index == 0 and (not meta_path.is_file() or not video.is_file()):
            legacy_meta = root / "control" / "camplot.json"
            legacy_video = root / "control" / "control_video.mp4"
            if legacy_meta.is_file() and legacy_video.is_file():
                meta_path, video = legacy_meta, legacy_video
        if not meta_path.is_file() or not video.is_file():
            return None
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        want = spec.payload(int(self.moge_level.value()))
        for key in ("orientation", "length", "moge_level", "anchors", "look_at"):
            if str(meta.get(key, "")).strip() != str(want[key]).strip():
                return None
        return video

    def _mark_stale(self) -> None:
        if self._updating:
            return
        self._pull_widgets_to_rail()
        spec = self._rail()
        spec.played = False
        spec.dirty = True
        self._sync_confirm()
        self._persist()
        if self._geo_on:
            self.preview_hint.setText(
                f"Rail {spec.index + 1} changed. Play the mesh flight on this pane before Confirm rails."
            )

    def _sync_confirm(self) -> None:
        self.btn_confirm.setEnabled(self._geo_on and all_played(self._rails))
        self._sync_rail_count_buttons()

    def _on_suggest(self) -> None:
        if not self._geo_on:
            return
        if confirmed_path(self._scene_name()).is_file():
            reply = QMessageBox.question(
                self,
                "Suggest paths",
                "Replace the four default rails with Suggest paths? "
                "Confirm’d rails will need Play + Confirm rails again. The optional 5th rail is kept.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
        try:
            pts = load_scene_cloud(scene_dir(self._scene_name()))
            items = suggest_paths(pts, count=D.SUGGEST_COUNT)
        except Exception as exc:
            self.requestJob.emit(["__geo_error__", str(exc)])
            return
        extra = [r for r in self._rails if r.index >= D.DEFAULT_RAIL_COUNT]
        self._rails = [suggestion_to_rail(i, item) for i, item in enumerate(items[: D.SUGGEST_COUNT])]
        for spec in extra:
            spec.played = False
            spec.dirty = True
            self._rails.append(spec)
        self._rails.sort(key=lambda r: r.index)
        self._active = 0
        self._fill_active_combo()
        self._push_rail_to_widgets()
        self._push_gizmos()
        self._sync_confirm()
        self._persist()
        self.preview_hint.setText(
            "Suggest paths filled four archetypes (all start at the star). Play each, then Confirm rails."
        )

    def _on_add_rail(self) -> None:
        nxt = next_free_rail_index(self._rails)
        if nxt is None:
            return
        spec = straight_rail(nxt)
        spec.played = False
        spec.dirty = True
        self._pull_widgets_to_rail()
        self._rails.append(spec)
        self._rails.sort(key=lambda r: r.index)
        self._select_rail_index(nxt)
        self._fill_active_combo()
        self._push_rail_to_widgets()
        self._push_gizmos()
        self._sync_confirm()
        self._persist()
        extra = " (optional 5th uses Add-to-dataset)" if nxt == D.MAX_RAILS - 1 else ""
        self.preview_hint.setText(f"Added rail {nxt + 1}{extra}. Play it, then Confirm rails.")

    def _on_remove_rail(self) -> None:
        spec = self._rail()
        if spec.index == 0 or len(self._rails) <= D.MIN_RAILS:
            self.preview_hint.setText("Rail 1 is the default and stays. Switch to another rail to remove it.")
            return
        gone = spec.index
        self._pull_widgets_to_rail()
        self._stop_preview()
        del self._rails[self._active]
        if self._active >= len(self._rails):
            self._active = len(self._rails) - 1
        self._fill_active_combo()
        self._push_rail_to_widgets()
        self._push_gizmos()
        self._sync_confirm()
        self._persist()
        self.preview_hint.setText(
            f"Removed rail {gone + 1}. Rail 1 stays. Add rail to fill a slot, then Confirm rails."
        )

    def _on_play(self) -> None:
        if not self._geo_on:
            return
        try:
            parse_anchors_ext(self.anchors.toPlainText())
        except ValueError as exc:
            self.requestJob.emit(["__geo_error__", str(exc)])
            return
        self._pull_widgets_to_rail()
        self._save_editor()
        self._write_request()
        cached = self._matching_video()
        if cached is not None:
            self._start_preview(cached)
            return
        self.btn_play.setEnabled(False)
        spec = self._rail()
        self.preview_hint.setText(
            f"Rendering rail {spec.index + 1} mesh flight (MoGe + mesh, black holes OK). Stop kills the worker."
        )
        self.requestJob.emit(["__camplot__", self._scene_name(), str(spec.index)])

    def on_camplot_busy(self) -> None:
        self.btn_play.setEnabled(self._geo_on)
        self.preview_hint.setText("Busy — Stop the current job first, then Play.")

    def apply_camplot(self, kind: str, path: str) -> None:
        if kind != "video":
            return
        video = Path(path)
        if video.is_file():
            self._start_preview(video)

    def on_job_finished(self, label: str, code: int) -> None:
        if label != "camplot":
            return
        self.btn_play.setEnabled(self._geo_on)
        if code != 0 and not self._rail().played:
            self.preview_hint.setText("Mesh flight failed. Check Log, then Play again.")

    def _start_preview(self, path: Path) -> None:
        import cv2

        self._stop_preview()
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            self.requestJob.emit(["__geo_error__", f"could not open {path}"])
            self.btn_play.setEnabled(self._geo_on)
            return
        self._preview_cap = cap
        self._video_path = path
        spec = self._rail()
        spec.played = True
        spec.dirty = False
        self._sync_confirm()
        self._save_editor()
        self.btn_play.setEnabled(self._geo_on)
        self.preview_hint.setText(
            f"Playing rail {spec.index + 1} {path.name} (black holes OK). "
            "Confirm rails when every rail has been previewed."
        )
        self._show_preview_frame()
        self._preview_timer.start()

    def _show_preview_frame(self) -> None:
        import cv2

        cap = self._preview_cap
        if cap is None:
            return
        ok, frame = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self._preview_timer.stop()
            pending = [r.index + 1 for r in self._rails if not r.played or r.dirty]
            if pending:
                self.preview_hint.setText(
                    f"Flight finished. Still need Play on rail(s) {pending} then Confirm rails."
                )
            else:
                self.preview_hint.setText("Flight finished. Confirm rails to unlock Generate.")
            return
        height, width, channels = frame.shape
        qimg = QImage(
            frame.data, width, height, channels * width, QImage.Format.Format_BGR888
        ).copy()
        pix = QPixmap.fromImage(qimg)
        target = self.preview_frame.size()
        if target.width() > 8 and target.height() > 8:
            pix = pix.scaled(
                target,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        self.preview_frame.setPixmap(pix)

    def _preview_tick(self) -> None:
        self._show_preview_frame()

    def _stop_preview(self) -> None:
        self._preview_timer.stop()
        cap = self._preview_cap
        if cap is not None:
            cap.release()
            self._preview_cap = None

    def _on_confirm(self) -> None:
        if not all_played(self._rails):
            return
        self._pull_widgets_to_rail()
        write_confirmed(self._scene_name(), self._rails, int(self.moge_level.value()))
        self._persist()
        self.requestJob.emit(["__confirmed__", self._scene_name()])
        self.preview_hint.setText("Confirm rails is on. Generate is unlocked.")
        self.railsChanged.emit()
