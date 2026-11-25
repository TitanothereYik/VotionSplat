"""Splat page for the HY-World trainer: live viewport, pause, stop-with-save, export."""

from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import Qt, QUrl, Signal
from PySide6.QtGui import QColor, QDesktopServices, QPalette, QStandardItem, QStandardItemModel
from PySide6.QtWidgets import (
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

from app.widgets.sparkline import Sparkline
from app.widgets.splat_view import SplatLiveView
from engine.hyworld.frames import CROP, frame_options
from engine.hyworld.paths import gs_data, gs_result, hy_root, latest_checkpoint, votion_dir
from engine.hyworld.train_run import DIFIX_PY, DIFIX_SRC, continue_info
from engine.splat import defaults as D


def _lab(text: str) -> QLabel:
    lab = QLabel(text)
    lab.setProperty("class", "fLab")
    return lab


def _hint(text: str) -> QLabel:
    lab = QLabel(text)
    lab.setWordWrap(True)
    lab.setProperty("class", "fHint")
    return lab


def _int_text(text: str, default: int, lo: int = 1) -> int:
    raw = "".join(ch for ch in text if ch.isdigit())
    if not raw:
        return default
    try:
        return max(lo, int(raw))
    except ValueError:
        return default


class HySplatPage(QWidget):
    requestJob = Signal(list)
    requestStop = Signal()
    controlLine = Signal(str)
    logLine = Signal(str)

    def __init__(self, scene_fn, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._scene_name = scene_fn
        self._state = "IDLE"
        self._paused = False
        self._worker_kind = ""
        self._cameras: list[dict] = []
        self._stats: dict[str, str] = {}
        self._shm_pending = ""
        self._busy = False
        body = QHBoxLayout(self)
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)
        body.addWidget(self._build_rail())
        body.addWidget(self._build_view(), 1)
        body.addWidget(self._build_cameras())
        self.refresh()

    def _view_json(self) -> Path:
        return votion_dir(self._scene_name()) / "splat_view.json"

    def _build_rail(self) -> QWidget:
        rail = QWidget()
        rail.setObjectName("rail")
        rail.setFixedWidth(290)
        r = QVBoxLayout(rail)
        r.setContentsMargins(12, 12, 12, 14)
        r.setSpacing(4)
        r.addWidget(_lab("Max steps"))
        self.max_steps = QLineEdit("8000")
        self.max_steps.setPlaceholderText("8000")
        self.max_steps.editingFinished.connect(self._normalize_fields)
        r.addWidget(self.max_steps)
        r.addWidget(_hint("Train starts over from the dataset. Continue finishes the last run with its own step count."))
        r.addWidget(_lab("Training frames"))
        self._frames_model = QStandardItemModel(self)
        self._frame_rows: list[dict] = []
        self.frames = QComboBox()
        self.frames.setModel(self._frames_model)
        self.frames.currentIndexChanged.connect(lambda _i: self._frames_note())
        r.addWidget(self.frames)
        self.frames_hint = _hint("")
        r.addWidget(self.frames_hint)
        self.mesh = QCheckBox("Export mesh at the end (slow)")
        r.addWidget(self.mesh)
        self.difix = QCheckBox("Difix3D+ refine (evaluation only)")
        self.difix.setToolTip(
            "Renders views between the paths, cleans them with NVIDIA Difix and trains on them. "
            "NVIDIA licence: research and evaluation only, not for commercial use. "
            "One round every 2000 steps, about 2 minutes each."
        )
        if not DIFIX_PY.is_file() or not DIFIX_SRC.is_dir():
            self.difix.setEnabled(False)
            self.difix.setToolTip("Difix is not installed")
        r.addWidget(self.difix)
        row1 = QHBoxLayout()
        self.btn_train = QPushButton("Train")
        self.btn_train.setObjectName("btnPrimary")
        self.btn_train.clicked.connect(lambda: self._emit("train"))
        self.btn_continue = QPushButton("Continue training")
        self.btn_continue.setObjectName("btnGhost")
        self.btn_continue.clicked.connect(lambda: self._emit("continue"))
        for btn in (self.btn_train, self.btn_continue):
            row1.addWidget(btn)
        r.addLayout(row1)
        row2 = QHBoxLayout()
        self.btn_pause = QPushButton("Pause")
        self.btn_pause.setObjectName("btnGhost")
        self.btn_pause.clicked.connect(self._on_pause)
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setObjectName("btnGhost")
        self.btn_stop.clicked.connect(self.requestStop.emit)
        for btn in (self.btn_pause, self.btn_stop):
            row2.addWidget(btn)
        r.addLayout(row2)
        r.addWidget(_hint("Stop saves a checkpoint first. Press Stop twice to kill at once."))
        r.addWidget(_lab("Export color space"))
        self.color_space = QComboBox()
        for key, label in D.COLOR_SPACE_CHOICES:
            self.color_space.addItem(label, key)
        default = self.color_space.findData(D.COLOR_SPACE_DEFAULT)
        self.color_space.setCurrentIndex(default if default >= 0 else 0)
        self.color_space.setToolTip(
            D.VOTION_UE_SETUP + "\n\n" + D.WALLGS_UE_SETUP + "\n\n" + D.WALLGS_ACESCG_UE_SETUP + "\n\n" + D.LUMA_UE_SETUP
        )
        self.color_space.currentIndexChanged.connect(self._sync_soften)
        r.addWidget(self.color_space)
        self.soften = QCheckBox("Soften like the viewport (no needles)")
        self.soften.setChecked(True)
        self.soften.setToolTip(D.HELP_SOFTEN)
        r.addWidget(self.soften)
        self.btn_export = QPushButton("Export splat.ply")
        self.btn_export.setObjectName("btnGhost")
        self.btn_export.clicked.connect(lambda: self._emit("export"))
        r.addWidget(self.btn_export)
        self.btn_folder = QPushButton("Open result folder")
        self.btn_folder.setObjectName("btnGhost")
        self.btn_folder.clicked.connect(self._open_folder)
        r.addWidget(self.btn_folder)
        self.gate_hint = _hint("")
        r.addWidget(self.gate_hint)
        r.addStretch(1)
        return rail

    def _open_folder(self) -> None:
        folder = hy_root(self._scene_name())
        if folder.is_dir():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))

    def _build_view(self) -> QWidget:
        view = QWidget()
        view.setObjectName("viewport")
        v = QVBoxLayout(view)
        v.setContentsMargins(8, 8, 8, 8)
        v.setSpacing(6)
        hdr = QHBoxLayout()
        lbl = QLabel("Live splat")
        lbl.setProperty("class", "viewLbl")
        self.cam_lbl = QLabel("Centre")
        self.cam_lbl.setProperty("class", "fHint")
        self.btn_reset = QPushButton("reset view")
        self.btn_reset.setProperty("class", "rail-mini")
        hdr.addWidget(lbl)
        hdr.addWidget(self.cam_lbl)
        hdr.addStretch(1)
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
            "Reset is the panorama centre. LMB look, RMB pan, wheel dolly, "
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
        self.cam_hint = _hint("Filled when training starts.")
        c.addWidget(self.cam_hint)
        return panel

    def _steps(self) -> int:
        return _int_text(self.max_steps.text(), 8000, lo=100)

    def _normalize_fields(self) -> None:
        n = self._steps()
        if self.max_steps.text().strip() != str(n):
            self.max_steps.setText(str(n))
        self._update_hud()

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

    def _sync_soften(self) -> None:
        votion = self._color_space_key() == D.COLOR_SPACE_VOTION
        self.soften.setEnabled(self.color_space.isEnabled() and not votion)
        self.soften.setToolTip("Not used by the Votion plugin preset" if votion else D.HELP_SOFTEN)

    def _emit(self, mode: str) -> None:
        self._normalize_fields()
        scene = self._scene_name()
        if mode == "train":
            self.requestJob.emit([
                "__hy_train__", scene, str(self._steps()),
                "1" if self.mesh.isChecked() else "0",
                "1" if self.difix.isChecked() else "0",
                self._frames_key(),
            ])
        elif mode == "continue":
            self.requestJob.emit(["__hy_train_continue__", scene])
        elif mode == "export":
            self.requestJob.emit([
                "__hy_export__", scene, self._color_space_key(),
                "0" if self._color_space_key() == D.COLOR_SPACE_VOTION else ("1" if self.soften.isChecked() else "0"),
            ])

    def _frames_key(self) -> str:
        return str(self.frames.currentData() or "expand")

    def _fill_frames(self, scene: str) -> None:
        keep = self._frames_key()
        self._frame_rows = frame_options(scene)
        self.frames.blockSignals(True)
        self._frames_model.clear()
        pick = 0
        for index, row in enumerate(self._frame_rows):
            item = QStandardItem(row["label"])
            item.setData(row["key"], Qt.ItemDataRole.UserRole)
            item.setEnabled(bool(row["ok"]))
            if row["note"]:
                item.setToolTip(row["note"])
            self._frames_model.appendRow(item)
            if row["key"] == keep and row["ok"]:
                pick = index
        self.frames.setCurrentIndex(pick)
        self.frames.blockSignals(False)
        self._frames_note()

    def _frames_note(self) -> None:
        key = self._frames_key()
        lines = []
        if key != "expand":
            lines.append(
                "Upscaled frames replace the Expand frames. Panorama and polar views stay at the Expand size. "
                "Each step renders more pixels, so training is slower."
            )
        if key in CROP:
            lines.append(
                f"Full 4x frames do not fit in 24 GB, so each step trains on a random {CROP[key]}x{CROP[key]} crop "
                "of the frame. Use more steps so every part of the frame is covered."
            )
        lines.extend(row["note"] for row in self._frame_rows if row["note"] and row["key"] != "expand")
        self.frames_hint.setText(" ".join(lines))

    def _on_view_changed(self, payload: dict) -> None:
        if self._worker_kind:
            self.controlLine.emit("VIEW\t" + json.dumps(payload))
        self._update_cam_label()

    def _reset_view(self) -> None:
        self.live.reset_view()
        self.cam_list.clearSelection()
        self._update_cam_label()

    def _on_camera_item(self, item: QListWidgetItem) -> None:
        idx = item.data(Qt.ItemDataRole.UserRole)
        if idx is None:
            return
        self.live.set_camera(int(idx))
        self._update_cam_label()

    def _update_cam_label(self) -> None:
        idx = self.live.camera_index()
        if idx < 0:
            self.cam_lbl.setText("Centre")
            return
        name = ""
        if 0 <= idx < len(self._cameras):
            name = str(self._cameras[idx].get("name", ""))
        self.cam_lbl.setText(f"cam {idx} {name}")

    def _fill_cameras(self, cams: list[dict]) -> None:
        self._cameras = cams
        self.cam_list.clear()
        groups: dict[str, list[dict]] = {}
        for cam in cams:
            groups.setdefault(str(cam.get("rail") or "views"), []).append(cam)
        order = sorted(groups, key=lambda key: (0 if key == "panorama" else 1 if key == "polar" else 2, key))
        for key in order:
            head = QListWidgetItem(f"{key} · {len(groups[key])} views")
            head.setFlags(Qt.ItemFlag.NoItemFlags)
            head.setForeground(QColor(255, 159, 48))
            self.cam_list.addItem(head)
            for cam in groups[key]:
                item = QListWidgetItem(str(cam.get("name", "")))
                item.setData(Qt.ItemDataRole.UserRole, int(cam.get("i", 0)))
                item.setForeground(QColor(201, 212, 227))
                self.cam_list.addItem(item)
        self.cam_hint.setText(f"{len(cams)} training views. Click one to jump there.")

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
        elif kind == "its" and args:
            self._stats["its"] = args[0]
        elif kind == "vram" and args:
            self._stats["vram"] = args[0]
        elif kind == "fps" and args:
            self._stats["fps"] = args[0]
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
            lines.append(f"step {step} / {s.get('max', self._steps())}")
        if s.get("loss"):
            psnr = s.get("psnr")
            ev = s.get("psnr_eval", "-")
            text = f"loss {s['loss']}"
            if psnr:
                text += f"   PSNR {psnr}" + (f" (eval {ev})" if ev and ev != "-" else "")
            lines.append(text)
        if s.get("gaussians"):
            lines.append(f"gaussians {int(s['gaussians']):,}")
        perf = []
        if s.get("its"):
            perf.append(f"{s['its']} it/s")
        fps = self.live.fps if self.live.attached else 0.0
        if fps > 0:
            perf.append(f"{fps:.0f} fps")
        if s.get("vram"):
            perf.append(f"{s['vram']} MB")
        if perf:
            lines.append("   ".join(perf))
        self.live.set_hud(self._state, lines)

    def refresh(self) -> None:
        scene = self._scene_name()
        ready = (gs_data(scene) / "cameras.json").is_file()
        ckpt = latest_checkpoint(scene)
        ply = hy_root(scene) / "splat.ply"
        info = continue_info(gs_result(scene))
        self.btn_train.setEnabled(ready and not self._busy)
        self.btn_continue.setEnabled(ready and info["ok"] and not self._busy)
        self.btn_stop.setEnabled(self._busy)
        self.btn_export.setEnabled(ckpt is not None and not self._busy)
        self.color_space.setEnabled(ckpt is not None and not self._busy)
        self._sync_soften()
        installed = DIFIX_PY.is_file() and DIFIX_SRC.is_dir()
        self.difix.setEnabled(not self._busy and installed)
        self._fill_frames(scene)
        self.frames.setEnabled(ready and not self._busy)
        lines = []
        if not ready:
            lines.append("Build the dataset first.")
        elif ckpt is None:
            lines.append("Dataset is ready. Press Train.")
        else:
            lines.append(f"Checkpoint {ckpt.name}." + (f" Exported {ply}." if ply.is_file() else ""))
        if info["ok"]:
            extra = f"Continue picks up at step {info['next_step']} of {info['max_steps']}"
            if info["frames"] != "expand":
                extra += f" on HiRes {info['frames'][1:]}x frames"
            if info["difix"]:
                extra += " with Difix"
            lines.append(extra + ".")
        elif info["reason"] != "No run to continue.":
            lines.append(info["reason"])
        run_file = gs_result(scene) / "votion_run.json"
        if run_file.is_file():
            try:
                last = json.loads(run_file.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                last = {}
            if last.get("difix"):
                lines.append("Last run used Difix3D+ (evaluation only).")
        self.gate_hint.setText(" ".join(lines))
        if not self.live.attached and not self._worker_kind:
            self.live.clear("Train to see the splat form.")
        self._update_hud()

    def restore_view(self) -> None:
        self.refresh()

    def on_job_started(self, label: str) -> None:
        self._busy = True
        if label == "hy_train":
            self.on_worker_started("train")
        self.refresh()

    def on_job_finished(self, label: str, code: int) -> None:
        del code
        self._busy = False
        if label == "hy_train":
            self.on_worker_finished()
        self.refresh()

    def on_progress(self, label: str, current: int, total: int, pct: int, msg: str) -> None:
        del label, current, total, pct, msg

    def on_hy_line(self, parts: list[str]) -> None:
        del parts
