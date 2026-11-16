"""Generate page — one WorldStereo sample across the rails, or one rail at a time."""

from __future__ import annotations

import gc
import json
import shutil
import time
from pathlib import Path

import cv2
from PIL import Image
from PySide6.QtCore import Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from app.widgets.erp_view import ErpStill
from engine.geo.rails import sync_dataset_from_confirmed
from engine.hires.pinhole import (
    HIRES_COMPOSITE,
    HIRES_FRAME,
    active_choice,
    hires_dir_for,
    hires_ready,
    write_choice,
    use_hires,
)
from engine.job import REPO_ROOT
from engine.pano.defaults import wan_prompt_from_stored
from engine.pano.scene_io import prompt_path, scene_dir, wan_prompt_path
from engine.worldstereo import defaults as WS
from engine.worldstereo.views import rail_view_rel


def _lab(text: str) -> QLabel:
    lab = QLabel(text)
    lab.setProperty("class", "fLab")
    return lab


def _hint(text: str) -> QLabel:
    lab = QLabel(text)
    lab.setWordWrap(True)
    lab.setProperty("class", "fHint")
    return lab


def _remove_tree(path: Path) -> str | None:
    """Delete a cache folder. Retry briefly: Windows keeps a video locked until the capture is gone."""
    if not path.exists():
        return None
    last = "busy"
    for _ in range(8):
        try:
            shutil.rmtree(path)
            return None
        except OSError as exc:
            last = str(exc)
            gc.collect()
            time.sleep(0.05)
    return f"{path.name}: {last}"


def _ws_weights_ready() -> bool:
    weights = REPO_ROOT / "models" / "diffusion_models" / WS.FP8_FILE
    dino = REPO_ROOT / "models" / "dinov2-base" / "model.safetensors"
    return (
        weights.is_file()
        and weights.stat().st_size == WS.FP8_BYTES
        and dino.is_file()
        and dino.stat().st_size > 0
    )


class GeneratePage(QWidget):
    requestJob = Signal(list)
    requestStop = Signal()

    def __init__(
        self,
        scene_name_fn,
        rails_ready_fn,
        parent: QWidget | None = None,
        *,
        release_media=None,
        after_clear=None,
    ) -> None:
        super().__init__(parent)
        self._scene_name = scene_name_fn
        self._rails_ready = rails_ready_fn
        self._release_media = release_media
        self._after_clear = after_clear
        self._loading_prompt = False
        self._busy = False
        self._job_kind = ""
        self._job_rail = -1
        self._job_view = "rail"
        self._view_rail = -1
        self._active_view = "rail"
        self._follow_live = True
        self._user_picked_view = False
        self._quality = True
        self._rail_rows: dict[int, dict] = {}
        self._rail_pick = QButtonGroup(self)
        self._rail_pick.setExclusive(True)
        self._ws_order: list[int] = []
        self._view_chips: dict[str, QPushButton] = {}
        self._mask_cap = None
        self._render_cap = None
        self._kf_dir: Path | None = None
        self._hires_dir: Path | None = None
        self._frame_positions: list[int] | None = None
        self._retrieved_count = 0
        self._preview_timer = QTimer(self)
        self._preview_timer.timeout.connect(self._preview_tick)
        self._preview_timer.setInterval(83)
        body = QHBoxLayout(self)
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)
        body.addWidget(self._build_rail())
        body.addWidget(self._build_view(), 1)
        self.refresh()

    def _build_rail(self) -> QWidget:
        rail = QWidget()
        rail.setObjectName("rail")
        rail.setFixedWidth(280)
        r = QVBoxLayout(rail)
        r.setContentsMargins(12, 12, 12, 14)
        r.setSpacing(4)

        r.addWidget(_lab("Generation"))
        self.gen_mode = QComboBox()
        self.gen_mode.addItem("Joint (all rails, one sample)", "joint")
        self.gen_mode.addItem("Per rail (21 frames each)", "rail")
        self.gen_mode.setToolTip(
            "Joint generates every kept rail in one sample. "
            "Per rail generates a 21-frame front view for each rail."
        )
        self.gen_mode.currentIndexChanged.connect(lambda _index: self._sync_buttons())
        r.addWidget(self.gen_mode)
        self.frames_hint = _hint("")
        r.addWidget(self.frames_hint)

        r.addWidget(_lab("Rail order"))
        self.rail_order = QListWidget()
        self.rail_order.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.rail_order.setDefaultDropAction(Qt.DropAction.MoveAction)
        self.rail_order.setMaximumHeight(110)
        self.rail_order.model().rowsMoved.connect(self._on_rail_order_moved)
        r.addWidget(self.rail_order)
        r.addWidget(_hint("Drag to set the order memory is carried across rails."))

        r.addWidget(_lab("Prompt (required)"))
        self.prompt_box = QTextEdit()
        self.prompt_box.setMinimumHeight(72)
        self.prompt_box.setMaximumHeight(120)
        self.prompt_box.textChanged.connect(self._on_prompt_changed)
        r.addWidget(self.prompt_box)
        r.addWidget(_hint("Describe the actual pano. An empty prompt blocks Generate."))

        r.addWidget(_lab("Seed"))
        self.seed = QSpinBox()
        self.seed.setRange(0, 2_147_483_647)
        self.seed.setValue(WS.SEED)
        r.addWidget(self.seed)

        row = QHBoxLayout()
        self.btn_gen = QPushButton("Generate")
        self.btn_gen.setObjectName("btnPrimary")
        self.btn_gen.clicked.connect(self._on_generate)
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setObjectName("btnGhost")
        self.btn_stop.clicked.connect(self.requestStop.emit)
        row.addWidget(self.btn_gen)
        row.addWidget(self.btn_stop)
        r.addLayout(row)

        self.btn_restart = QPushButton("Start over")
        self.btn_restart.setObjectName("btnGhost")
        self.btn_restart.clicked.connect(self._on_start_over)
        r.addWidget(self.btn_restart)

        self.btn_regen = QPushButton("Re-gen selected rail")
        self.btn_regen.setObjectName("btnGhost")
        self.btn_regen.clicked.connect(self._on_regen_selected)
        r.addWidget(self.btn_regen)

        r.addWidget(_lab("HiRes"))
        self.hires_mode = QComboBox()
        self.hires_mode.addItem("Composite", HIRES_COMPOSITE)
        self.hires_mode.addItem("WorldStereo", HIRES_FRAME)
        self.hires_mode.setToolTip(
            "Composite paints the 8K panorama where the mesh is visible. "
            "WorldStereo upscales the generated frames and does not paint the panorama. "
            "HiRes all writes the selected one. Reconstruct uses whichever is active."
        )
        self.hires_mode.currentIndexChanged.connect(self._on_hires_mode)
        r.addWidget(self.hires_mode)
        self.hires_hint = _hint("")
        r.addWidget(self.hires_hint)

        self.btn_hires = QPushButton("HiRes all")
        self.btn_hires.setObjectName("btnGhost")
        self.btn_hires.clicked.connect(self._on_hires)
        r.addWidget(self.btn_hires)

        self.btn_frames = QPushButton("Open frames folder")
        self.btn_frames.setObjectName("btnGhost")
        self.btn_frames.clicked.connect(self._on_open_frames)
        r.addWidget(self.btn_frames)
        self.gate_hint = _hint("")
        r.addWidget(self.gate_hint)
        r.addStretch(1)
        return rail

    def _build_view(self) -> QWidget:
        view = QWidget()
        view.setObjectName("viewport")
        v = QVBoxLayout(view)
        v.setContentsMargins(8, 8, 8, 8)
        v.setSpacing(6)
        hdr = QLabel("Rails · click a number to preview")
        hdr.setProperty("class", "viewLbl")
        v.addWidget(hdr)
        self.rail_host = QWidget()
        self.rail_lay = QVBoxLayout(self.rail_host)
        self.rail_lay.setContentsMargins(0, 0, 0, 0)
        self.rail_lay.setSpacing(4)
        v.addWidget(self.rail_host)
        self.progress = QProgressBar()
        self.progress.setObjectName("jobBar")
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        v.addWidget(self.progress)
        preview_head = QHBoxLayout()
        self.preview_lab = QLabel("Live preview · render | mask | keyframe")
        self.preview_lab.setProperty("class", "viewLbl")
        preview_head.addWidget(self.preview_lab)
        preview_head.addStretch(1)
        self.btn_mask_fit = QPushButton("Fit")
        self.btn_mask_100 = QPushButton("100%")
        self.mask_view_group = QButtonGroup(self)
        self.mask_view_group.setExclusive(True)
        for btn, checked in ((self.btn_mask_fit, True), (self.btn_mask_100, False)):
            btn.setCheckable(True)
            btn.setChecked(checked)
            btn.setProperty("class", "modeChip")
            self.mask_view_group.addButton(btn)
            preview_head.addWidget(btn)
        self.btn_mask_fit.clicked.connect(lambda: self._set_preview_fit(True))
        self.btn_mask_100.clicked.connect(lambda: self._set_preview_fit(False))
        v.addLayout(preview_head)
        self.legend = QLabel("Render · mask (white = empty) · keyframe")
        self.legend.setProperty("class", "fHint")
        v.addWidget(self.legend)
        self.retrieved_label = QLabel("retrieved 0 memory frames")
        self.retrieved_label.setProperty("class", "fHint")
        v.addWidget(self.retrieved_label)
        split = QSplitter(Qt.Orientation.Horizontal)
        split.setChildrenCollapsible(False)
        split.setHandleWidth(8)
        self.render_view = ErpStill(empty_text="Render")
        self.mask_view = ErpStill(empty_text="Mask")
        key_col = QWidget()
        key_lay = QVBoxLayout(key_col)
        key_lay.setContentsMargins(0, 0, 0, 0)
        key_lay.setSpacing(4)
        chip_row = QHBoxLayout()
        chip_row.setSpacing(4)
        key_lab = QLabel("Keyframe")
        key_lab.setProperty("class", "viewLbl")
        chip_row.addWidget(key_lab)
        self._chip_group = QButtonGroup(self)
        self._chip_group.setExclusive(True)
        for name, label in (("rail", "Rail"), ("joint", "Joint")):
            chip = QPushButton(label)
            chip.setCheckable(True)
            chip.setProperty("class", "modeChip")
            chip.clicked.connect(lambda _=False, view=name: self._on_view_chip(view))
            self._chip_group.addButton(chip)
            self._view_chips[name] = chip
            chip_row.addWidget(chip)
        chip_row.addStretch(1)
        key_lay.addLayout(chip_row)
        self.frame_view = ErpStill(empty_text="Keyframe")
        self.render_view.setMinimumHeight(180)
        self.mask_view.setMinimumHeight(180)
        self.frame_view.setMinimumHeight(180)
        key_lay.addWidget(self.frame_view, 1)
        split.addWidget(self.render_view)
        split.addWidget(self.mask_view)
        split.addWidget(key_col)
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 1)
        split.setStretchFactor(2, 1)
        split.setSizes([280, 280, 280])
        v.addWidget(split, 1)
        v.addWidget(
            _hint(
                "Rail shows the selected rail. Joint plays the whole sample. "
                "Generate resumes. Start over closes the preview, then clears rails, memory, and the geometry cache."
            )
        )
        self._view_chips["rail"].setChecked(True)
        return view

    def set_quality(self, quality: bool) -> None:
        self._quality = bool(quality)
        self._sync_buttons()

    def _gen_mode(self) -> str:
        return str(self.gen_mode.currentData() or "joint")

    def _on_view_chip(self, view: str) -> None:
        self._user_picked_view = True
        self._follow_live = False
        self._active_view = view
        if self._view_rail >= 0:
            self._show_ws_preview(self._view_rail, view)

    def _set_preview_fit(self, fit: bool) -> None:
        self.render_view.set_fit(fit)
        self.mask_view.set_fit(fit)
        self.frame_view.set_fit(fit)
        self.btn_mask_fit.setChecked(fit)
        self.btn_mask_100.setChecked(not fit)

    def _on_rail_order_moved(self, *_args) -> None:
        order = []
        for row in range(self.rail_order.count()):
            item = self.rail_order.item(row)
            if item is not None:
                order.append(int(item.data(Qt.ItemDataRole.UserRole)))
        if order:
            self._ws_order = order

    def _sync_ws_order(self) -> None:
        rows = self._kept_rows()
        current = [int(row["index"]) for row in rows] or [0, 1, 2, 3]
        kept = [index for index in self._ws_order if index in current]
        for index in current:
            if index not in kept:
                kept.append(index)
        self._ws_order = kept
        self.rail_order.blockSignals(True)
        self.rail_order.clear()
        for index in self._ws_order:
            self.rail_order.addItem(f"Rail {index + 1}")
            self.rail_order.item(self.rail_order.count() - 1).setData(Qt.ItemDataRole.UserRole, index)
        self.rail_order.blockSignals(False)

    def rail_view_dir(self, rail_index: int, view: str) -> Path:
        if view == "joint":
            return scene_dir(self._scene_name()) / "worldstereo" / "joint"
        return scene_dir(self._scene_name()) / rail_view_rel(rail_index, "front")

    def _expected_names(self, folder: Path) -> list[str]:
        meta_path = folder / "meta.json"
        if meta_path.is_file():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                meta = {}
            positions = meta.get("positions") if isinstance(meta, dict) else None
            if isinstance(positions, list) and positions:
                return [f"{int(pos):03d}.png" for pos in positions]
            frames = int(meta.get("frames") or 0) if isinstance(meta, dict) else 0
            if frames > 0:
                return [f"{index:03d}.png" for index in range(frames)]
        joint = folder / "joint.json"
        if joint.is_file():
            try:
                payload = json.loads(joint.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                payload = {}
            rows = payload.get("frames") if isinstance(payload, dict) else None
            if isinstance(rows, list) and rows:
                return [f"{int(row.get('index', index)):03d}.png" for index, row in enumerate(rows)]
        return [f"{index:03d}.png" for index in range(WS.NFRAME)]

    def _view_ready(self, rail_index: int, view: str = "rail") -> bool:
        folder = self.rail_view_dir(rail_index, view)
        frames = folder / "keyframes"
        if (folder / "STALE.txt").is_file() or not frames.is_dir():
            return False
        if view == "joint":
            return (folder / "joint.json").is_file() and all(
                (frames / name).is_file() for name in self._expected_names(folder)
            )
        if not (folder / "meta.json").is_file():
            return False
        return all((frames / name).is_file() for name in self._expected_names(folder))

    def _preview_ready(self, rail_index: int, view: str) -> bool:
        folder = self.rail_view_dir(rail_index, view)
        return (folder / "render" / "render.mp4").is_file() and (folder / "mask" / "render_mask.mp4").is_file()

    def _guide_preview_only(self) -> bool:
        folder = scene_dir(self._scene_name()) / "worldstereo" / "joint"
        keyframes = folder / "keyframes"
        has_frames = keyframes.is_dir() and any(keyframes.glob("[0-9][0-9][0-9].png"))
        return self._gen_mode() == "joint" and self._preview_ready(0, "joint") and not has_frames

    def _sync_view_chips(self) -> None:
        rail = self._view_rail
        for name, chip in self._view_chips.items():
            if name == "rail":
                ready = rail >= 0 and (self._view_ready(rail, name) or self._preview_ready(rail, name))
            else:
                ready = self._view_ready(0, "joint") or self._preview_ready(0, "joint")
            live = self._busy and self._job_view == name
            chip.setEnabled(ready or live or name == self._active_view)

    def _apply_preview_caption(self, view: str, count: int) -> None:
        label = "Joint" if view == "joint" else "Rail"
        self.retrieved_label.setText(f"{label} · retrieved {int(count)} memory frames")

    def restore_view(self) -> None:
        self.refresh(keep_preview=True)

    def _prompt_text(self) -> str:
        return wan_prompt_from_stored(self.prompt_box.toPlainText())

    def _load_prompt(self, scene: str) -> str:
        extra = wan_prompt_path(scene)
        if extra.is_file():
            return wan_prompt_from_stored(extra.read_text(encoding="utf-8"))
        stored = prompt_path(scene)
        raw = stored.read_text(encoding="utf-8") if stored.is_file() else ""
        return wan_prompt_from_stored(raw)

    def _on_prompt_changed(self) -> None:
        if self._loading_prompt:
            return
        text = self._prompt_text()
        path = wan_prompt_path(self._scene_name())
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + ("\n" if text else ""), encoding="utf-8")
        self._sync_buttons()

    def _dataset(self) -> dict:
        return sync_dataset_from_confirmed(self._scene_name())

    def _kept_rows(self) -> list[dict]:
        return [row for row in self._dataset().get("rails") or [] if row.get("keep", True)]

    def _any_view_ready(self) -> bool:
        if self._view_ready(0, "joint"):
            return True
        return any(self._view_ready(int(row["index"]), "rail") for row in self._kept_rows())

    def _sync_buttons(self) -> None:
        empty = not self._prompt_text()
        ready = bool(self._rails_ready())
        weights = _ws_weights_ready()
        can_job = not empty and ready and weights and self._quality and not self._busy
        joint = self._gen_mode() == "joint"
        kept = len(self._ws_order)
        per = int(WS.JOINT_FRAMES_PER_RAIL)
        cap = int(WS.JOINT_MAX_FRAMES_PER_GEN)
        total = kept * per
        passes = (total + cap - 1) // cap if total else 1
        if joint and passes > 1:
            self.frames_hint.setText(
                f"{total} frames, {per} per kept rail: {passes} samples of {cap}, sharing memory"
            )
        elif joint:
            self.frames_hint.setText(f"{total} frames in one sample: {per} per kept rail")
        else:
            self.frames_hint.setText(f"{int(WS.NFRAME)} frames per kept rail")
        self.btn_gen.setEnabled(can_job and bool(self._ws_order))
        self.btn_regen.setEnabled(can_job and self._view_rail >= 0 and not joint)
        self.btn_regen.setToolTip(
            "Joint is one sample; Start over or Generate again."
            if joint
            else "Resample the selected rail. Later rails are marked stale."
        )
        self.btn_restart.setEnabled(not self._busy)
        self.btn_hires.setEnabled(can_job and self._any_view_ready())
        self.hires_mode.setEnabled(not self._busy)
        selected = str(self.hires_mode.currentData() or HIRES_COMPOSITE)
        if selected == HIRES_FRAME:
            self.btn_hires.setToolTip("Upscale the WorldStereo frames. Reconstruct uses those images.")
        else:
            self.btn_hires.setToolTip("Paint the 8K panorama where the mesh is visible. Reconstruct uses those images.")
        if self._hires_missing():
            self.hires_hint.setText("HiRes all builds this choice. Reconstruct keeps the other set until then.")
        elif selected == HIRES_FRAME:
            self.hires_hint.setText("Reconstruct uses the upscaled WorldStereo frames.")
        else:
            self.hires_hint.setText("Reconstruct uses the panorama composite.")
        if self._busy:
            self.gate_hint.setText("WorldStereo running — one process, shared memory. Stop frees the GPU.")
        elif empty:
            self.gate_hint.setText("Empty prompt blocks Generate.")
        elif not ready:
            self.gate_hint.setText("Confirm rails on Geometry first (Play each rail, then Confirm rails).")
        elif not self._quality:
            self.gate_hint.setText("WorldStereo is the Quality route. Fast and Scout stay off.")
        elif not weights:
            self.gate_hint.setText("WorldStereo stays off until the FP8 weights and DINOv2 are READY.")
        elif joint:
            self.gate_hint.setText(
                "Joint covers every kept rail, one sample per rail, and the next sample continues from the shared memory. "
                "Generate resumes a matching run. Start over closes the preview, then clears rails, memory, and the geometry cache."
            )
        else:
            self.gate_hint.setText(
                "Per rail generates a 21-frame front view. Generate resumes. Start over closes the preview, then clears rails, memory, and the geometry cache."
            )
        self._sync_view_chips()

    def refresh(self, keep_preview: bool = False) -> None:
        scene = self._scene_name()
        text = self._load_prompt(scene)
        self._loading_prompt = True
        self.prompt_box.setPlainText(text)
        self._loading_prompt = False
        self._fill_rail_rows()
        self._sync_ws_order()
        self._load_hires_choice()
        self._sync_buttons()
        if not self._busy and not keep_preview and self._guide_preview_only():
            self._active_view = "joint"
            chip = self._view_chips.get("joint")
            if chip is not None:
                chip.setChecked(True)
            self._show_ws_preview(0, "joint")
            return
        if keep_preview and self._view_rail >= 0:
            self._show_ws_preview(self._view_rail, self._active_view)
        elif not self._busy:
            if self._view_rail >= 0:
                self._show_ws_preview(self._view_rail, self._active_view)
            else:
                rows = self._kept_rows() or (self._dataset().get("rails") or [])
                if rows:
                    self._select_rail(int(rows[0]["index"]))

    def _fill_rail_rows(self) -> None:
        while self.rail_lay.count():
            item = self.rail_lay.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._rail_rows.clear()
        for btn in list(self._rail_pick.buttons()):
            self._rail_pick.removeButton(btn)
        rows = self._dataset().get("rails") or []
        if not rows:
            rows = [{"index": i, "keep": True, "status": "queued"} for i in range(4)]
        for row in rows:
            rail_index = int(row.get("index", 0))
            line = QWidget()
            hl = QHBoxLayout(line)
            hl.setContentsMargins(0, 0, 0, 0)
            pick = QPushButton(f"Rail {rail_index + 1}")
            pick.setCheckable(True)
            pick.setProperty("class", "railPick")
            pick.clicked.connect(lambda _=False, idx=rail_index: self._select_rail(idx))
            self._rail_pick.addButton(pick)
            status = QLabel(self._status_text(row))
            status.setProperty("class", "fHint")
            status.setAlignment(Qt.AlignmentFlag.AlignRight)
            hl.addWidget(pick)
            hl.addWidget(status, 1)
            self.rail_lay.addWidget(line)
            self._rail_rows[rail_index] = {"pick": pick, "status": status, "row": row}
            if rail_index == self._view_rail:
                pick.setChecked(True)

    def _status_text(self, row: dict) -> str:
        if not row.get("keep", True):
            return "ditched"
        index = int(row.get("index", 0))
        folder = self.rail_view_dir(index, "rail")
        frames = folder / "keyframes"
        count = len(list(frames.glob("[0-9][0-9][0-9].png"))) if frames.is_dir() else 0
        if count and not (folder / "STALE.txt").is_file():
            return f"{count} keyframes"
        return str(row.get("status") or "queued")

    def set_rail_status(self, rail_index: int, text: str) -> None:
        slot = self._rail_rows.get(int(rail_index))
        if slot is not None:
            slot["status"].setText(text)

    def _select_rail(self, rail_index: int) -> None:
        self._view_rail = int(rail_index)
        self._active_view = "rail"
        chip = self._view_chips.get("rail")
        if chip is not None:
            chip.setChecked(True)
        slot = self._rail_rows.get(self._view_rail)
        if slot is not None:
            slot["pick"].setChecked(True)
        self._sync_buttons()
        self._show_ws_preview(self._view_rail, "rail")

    def _show_ws_preview(self, rail_index: int, view: str | None = None) -> None:
        view = view or self._active_view
        rail_dir = self.rail_view_dir(rail_index, view)
        render = rail_dir / "render" / "render.mp4"
        mask = rail_dir / "mask" / "render_mask.mp4"
        keyframes = rail_dir / "keyframes"
        retrieved = rail_dir / "retrieved.json"
        count = 0
        if retrieved.is_file():
            try:
                payload = json.loads(retrieved.read_text(encoding="utf-8"))
                count = len(payload.get("fnames") or [])
            except json.JSONDecodeError:
                count = 0
        self.retrieved_label.setText(f"{view} · retrieved {count} memory frames")
        self._retrieved_count = count
        self._apply_preview_caption(view, count)
        self._frame_positions = None
        meta_path = rail_dir / "meta.json"
        if view == "rail" and meta_path.is_file():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                meta = {}
            positions = meta.get("positions") if isinstance(meta, dict) else None
            if isinstance(positions, list) and positions:
                self._frame_positions = [int(pos) for pos in positions]
        self._stop_preview_play()
        self._kf_dir = keyframes if keyframes.is_dir() else None
        choice = active_choice(scene_dir(self._scene_name()))
        shown = hires_dir_for(rail_dir, choice)
        self._hires_dir = shown if self._kf_dir is not None else None
        if self._hires_dir is not None:
            self.preview_lab.setText("Live preview · render | mask | HiRes")
            if choice == HIRES_FRAME:
                self.legend.setText("Render · mask (white = empty) · HiRes (upscaled WorldStereo)")
            else:
                self.legend.setText("Render · mask (white = empty) · HiRes (8K pano, WorldStereo in the holes)")
        elif self._kf_dir is None:
            self.preview_lab.setText("Live preview · guide | mask")
            self.legend.setText("Guide · mask (white = empty)")
        else:
            self.preview_lab.setText("Live preview · render | mask | keyframe")
            self.legend.setText("Render · mask (white = empty) · keyframe")
        self._sync_view_chips()
        if render.is_file() and mask.is_file():
            render_cap = cv2.VideoCapture(str(render))
            mask_cap = cv2.VideoCapture(str(mask))
            if render_cap.isOpened() and mask_cap.isOpened():
                self._render_cap = render_cap
                self._mask_cap = mask_cap
                self._preview_tick()
                self._preview_timer.start()
                return
            render_cap.release()
            mask_cap.release()
        self.render_view.set_pil(None)
        self.mask_view.set_pil(None)
        self.frame_view.set_pil(None)

    def show_ws(self, rail_index: int, phase: str, retrieved: int | None = None, view: str | None = None) -> None:
        self._job_rail = int(rail_index)
        if view:
            self._job_view = view
            if not self._user_picked_view:
                self._active_view = view
        label = f"{self._job_view} {phase}" if view else phase
        self.set_rail_status(self._job_rail, label)
        if retrieved is not None:
            self.retrieved_label.setText(f"{self._active_view} · retrieved {int(retrieved)} memory frames")
        if self._follow_live or self._view_rail == self._job_rail:
            self._view_rail = self._job_rail
            self._show_ws_preview(self._job_rail, self._active_view)
            chip = self._view_chips.get(self._active_view)
            if chip is not None and chip.isEnabled():
                chip.setChecked(True)

    def set_job_progress(self, current: int, total: int, pct: int, msg: str = "") -> None:
        if total > 0:
            self.progress.setRange(0, total)
            self.progress.setValue(min(current, total))
        elif pct >= 0:
            self.progress.setRange(0, 100)
            self.progress.setValue(min(100, max(0, pct)))

    def on_generate_start(self, rail_indices: list[int], kind: str = "worldstereo") -> None:
        self._busy = True
        self._job_kind = kind
        self._job_rail = int(rail_indices[0]) if rail_indices else -1
        self._follow_live = not self._user_picked_view
        self._fill_rail_rows()
        for idx in {int(i) for i in rail_indices}:
            self.set_rail_status(idx, "queued")
        if self._job_rail >= 0:
            self._select_rail(self._job_rail)
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self._sync_buttons()

    def on_job_finished(self, label: str, code: int, *, more: bool) -> None:
        if label not in ("worldstereo", "pinhole_hires"):
            return
        if more:
            return
        self._busy = False
        self._job_kind = ""
        self._user_picked_view = False
        self._follow_live = True
        self._fill_rail_rows()
        self._sync_buttons()
        if code != 0 and self._job_rail >= 0:
            self.set_rail_status(self._job_rail, "failed")
        elif self._view_rail >= 0:
            self._show_ws_preview(self._view_rail, self._active_view)

    def _read_preview_pair(self):
        if self._mask_cap is None:
            return None
        ok_m, frame_m = self._mask_cap.read()
        ok_s, frame_s = (False, None)
        if self._render_cap is not None:
            ok_s, frame_s = self._render_cap.read()
        if not ok_m or (self._render_cap is not None and not ok_s):
            self._mask_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            if self._render_cap is not None:
                self._render_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok_m, frame_m = self._mask_cap.read()
            if self._render_cap is not None:
                ok_s, frame_s = self._render_cap.read()
            if not ok_m:
                return None
        index = max(0, int(self._mask_cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1)
        return index, frame_m, frame_s if ok_s else None

    def _preview_tick(self) -> None:
        if self._mask_cap is None:
            return
        pair = self._read_preview_pair()
        if pair is None:
            return
        frame_index, frame_m, frame_s = pair
        if len(frame_m.shape) == 3:
            gray = cv2.cvtColor(frame_m, cv2.COLOR_BGR2GRAY)
            mask_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        else:
            mask_rgb = cv2.cvtColor(frame_m, cv2.COLOR_GRAY2RGB)
        self.mask_view.set_pil(Image.fromarray(mask_rgb))
        if frame_s is not None:
            self.render_view.set_pil(Image.fromarray(cv2.cvtColor(frame_s, cv2.COLOR_BGR2RGB)))
        shot = None
        if self._kf_dir is not None:
            if self._frame_positions is None:
                shot = self._kf_dir / f"{frame_index:03d}.png"
            elif frame_index < len(self._frame_positions):
                shot = self._kf_dir / f"{self._frame_positions[frame_index]:03d}.png"
        if self._hires_dir is not None and shot is not None:
            hi = self._hires_dir / shot.name
            if hi.is_file():
                shot = hi
        if shot is not None and shot.is_file():
            self.frame_view.set_path(shot)
        else:
            self.frame_view.set_pil(None)

    def _release_captures(self) -> None:
        if self._mask_cap is not None:
            self._mask_cap.release()
            self._mask_cap = None
        if self._render_cap is not None:
            self._render_cap.release()
            self._render_cap = None

    def _stop_preview_play(self) -> None:
        self._preview_timer.stop()
        self._release_captures()

    def _on_open_frames(self) -> None:
        folder = scene_dir(self._scene_name()) / "worldstereo" / "rails"
        folder.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))

    def _emit_worldstereo(self, rails: list[int], *, restart: bool, regen_rail: int | None) -> None:
        scene = self._scene_name()
        text = self._prompt_text()
        if not text:
            self.requestJob.emit(["__gen_error__", "empty prompt blocks Generate"])
            return
        if not self._rails_ready():
            self.requestJob.emit(["__gen_error__", "Confirm rails on Geometry first"])
            return
        if not rails:
            self.requestJob.emit(["__gen_error__", "no rails selected"])
            return
        if not _ws_weights_ready():
            self.requestJob.emit(["__gen_error__", "WorldStereo weights are not READY"])
            return
        if regen_rail is not None and self._gen_mode() == "joint":
            self.requestJob.emit(["__gen_error__", "Joint is one sample; Start over or Generate again."])
            return
        work = scene_dir(scene) / "_work"
        work.mkdir(parents=True, exist_ok=True)
        req = work / "worldstereo_request.json"
        gen_mode = "rail" if regen_rail is not None else self._gen_mode()
        req.write_text(
            json.dumps(
                {
                    "scene": scene,
                    "prompt": text,
                    "seed": int(self.seed.value()),
                    "rails": [int(i) for i in rails],
                    "mode": gen_mode,
                    "frames_per_rail": int(WS.JOINT_FRAMES_PER_RAIL),
                    "generator": "worldstereo",
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        mode = "regen" if regen_rail is not None else ("restart" if restart else "resume")
        self._user_picked_view = False
        self._follow_live = True
        self._active_view = "joint" if gen_mode == "joint" else "rail"
        chip = self._view_chips.get(self._active_view)
        if chip is not None:
            chip.setChecked(True)
        self.requestJob.emit(
            [
                "__worldstereo__",
                scene,
                str(req),
                ",".join(str(int(i)) for i in rails),
                mode,
                str(int(self.seed.value())),
                "" if regen_rail is None else str(int(regen_rail)),
                gen_mode,
            ]
        )

    def _on_generate(self) -> None:
        self._emit_worldstereo(self._ws_order, restart=False, regen_rail=None)

    def _on_start_over(self) -> None:
        if self._busy:
            self.requestJob.emit(["__gen_error__", "Stop the current job before Start over"])
            return
        answer = QMessageBox.question(
            self,
            "Start over",
            "Close the open preview, then clear WorldStereo output, the depth cache, "
            "and the geometry cache for this scene? Floor, side, and the flight video "
            "are removed. The panorama and your rails stay.",
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        self._stop_preview_play()
        self.render_view.set_pil(None)
        self.mask_view.set_pil(None)
        self.frame_view.set_pil(None)
        if self._release_media is not None:
            self._release_media()
        gc.collect()
        scene = scene_dir(self._scene_name())
        folders = [scene / "worldstereo" / name for name in ("rails", "memory", "joint", "input", "world")]
        folders.extend((scene / "moge", scene / "control", scene / "depth"))
        errors = [msg for msg in (_remove_tree(folder) for folder in folders) if msg]
        if self._after_clear is not None:
            self._after_clear()
        self.refresh()
        if errors:
            self.requestJob.emit(
                ["__gen_error__", "Start over could not remove " + "; ".join(errors)]
            )

    def _on_regen_selected(self) -> None:
        if self._gen_mode() == "joint":
            self.requestJob.emit(["__gen_error__", "Joint is one sample; Start over or Generate again."])
            return
        if self._view_rail < 0:
            self.requestJob.emit(["__gen_error__", "click a rail to Re-gen"])
            return
        self._emit_worldstereo(self._ws_order, restart=False, regen_rail=self._view_rail)

    def _load_hires_choice(self) -> None:
        root = scene_dir(self._scene_name())
        saved = active_choice(root)
        index = self.hires_mode.findData(saved)
        if index >= 0 and index != self.hires_mode.currentIndex():
            self.hires_mode.blockSignals(True)
            self.hires_mode.setCurrentIndex(index)
            self.hires_mode.blockSignals(False)
        for rail in self._ws_order:
            folder = self.rail_view_dir(rail, "front")
            frames = folder / "keyframes"
            count = len(list(frames.glob("[0-9][0-9][0-9].png"))) if frames.is_dir() else 0
            if count and hires_ready(folder, saved, count):
                use_hires(folder, saved)

    def _hires_missing(self) -> bool:
        source = str(self.hires_mode.currentData() or HIRES_COMPOSITE)
        for index in self._ws_order:
            folder = self.rail_view_dir(index, "front")
            frames = folder / "keyframes"
            count = len(list(frames.glob("[0-9][0-9][0-9].png"))) if frames.is_dir() else 0
            if count and not hires_ready(folder, source, count):
                return True
        return False

    def _on_hires_mode(self) -> None:
        source = str(self.hires_mode.currentData() or HIRES_COMPOSITE)
        write_choice(scene_dir(self._scene_name()), source)
        switched = False
        for index in self._ws_order:
            folder = self.rail_view_dir(index, "front")
            frames = folder / "keyframes"
            count = len(list(frames.glob("[0-9][0-9][0-9].png"))) if frames.is_dir() else 0
            if count and hires_ready(folder, source, count) and use_hires(folder, source):
                switched = True
        self._sync_buttons()
        if switched and self._view_rail >= 0:
            self._show_ws_preview(self._view_rail, self._active_view)

    def _on_hires(self) -> None:
        if not self._any_view_ready():
            self.requestJob.emit(["__gen_error__", "Generate keyframes before HiRes all"])
            return
        source = str(self.hires_mode.currentData() or HIRES_COMPOSITE)
        self.requestJob.emit(["__ws_hires__", self._scene_name(), source])

