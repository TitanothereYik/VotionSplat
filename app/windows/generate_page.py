"""Phase 3 Generate page — WAN review, then HiRes.

Empty WAN prompt blocks Generate. The 2D green-outpaint instruction is stripped;
WAN fills black holes, not green. HiRes waits until the user confirms the 720p clips.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
from PIL import Image
from PySide6.QtCore import Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from app.widgets.erp_view import ErpStill
from engine.geo.rails import (
    camplot_meta_rel,
    control_mask_rel,
    dataset_path,
    sync_dataset_from_confirmed,
    wan_preview_dir_rel,
    wan_video_rel,
)
from engine.job import REPO_ROOT
from engine.worldstereo import defaults as WS
from engine.pano.defaults import wan_prompt_from_stored
from engine.pano.scene_io import prompt_path, scene_dir, wan_prompt_path
from engine.video import defaults as V


def _lab(text: str) -> QLabel:
    lab = QLabel(text)
    lab.setProperty("class", "fLab")
    return lab


def _hint(text: str) -> QLabel:
    lab = QLabel(text)
    lab.setWordWrap(True)
    lab.setProperty("class", "fHint")
    return lab


def _wan_weight_rows() -> list[tuple[str, Path]]:
    models = REPO_ROOT / "models"
    return [
        ("wan14", models / "diffusion_models" / V.UNET_FILE),
        ("lora_pano", models / "loras" / V.PANO_LORA_FILE),
        ("lora_lightx2v", models / "loras" / V.LIGHTX2V_FILE),
        ("umt5", models / "text_encoders" / V.UMT5_FILE),
        ("clip_vision", models / "clip_vision" / V.CLIP_VISION_FILE),
        ("vae", models / "vae" / V.VAE_FILE),
        ("ultrasharp", models / "upscale_models" / V.HOLE_UPSCALER),
    ]


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

    def __init__(self, scene_name_fn, rails_ready_fn, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._scene_name = scene_name_fn
        self._rails_ready = rails_ready_fn
        self._loading_prompt = False
        self._busy = False
        self._job_kind = ""
        self._job_rail = -1
        self._view_rail = -1
        self._follow_live = True
        self._rail_rows: dict[int, dict] = {}
        self._rail_pick = QButtonGroup(self)
        self._rail_pick.setExclusive(True)
        self._mask_path: Path | None = None
        self._rgb_path: Path | None = None
        self._mask_cap = None
        self._rgb_cap = None
        self._render_cap = None
        self._kf_dir: Path | None = None
        self._quality = True
        self._ws_order: list[int] = []
        self._saved_wan_seed = V.WAN_SEED
        self._preview_timer = QTimer(self)
        self._preview_timer.timeout.connect(self._preview_tick)
        self._preview_timer.setInterval(83)
        body = QHBoxLayout(self)
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)
        body.addWidget(self._build_rail())
        body.addWidget(self._build_view(), 1)
        self._apply_mode()
        self.refresh()

    def _build_rail(self) -> QWidget:
        rail = QWidget()
        rail.setObjectName("rail")
        rail.setFixedWidth(280)
        r = QVBoxLayout(rail)
        r.setContentsMargins(12, 12, 12, 14)
        r.setSpacing(4)

        r.addWidget(_lab("Generator"))
        self.generator = QComboBox()
        self.generator.addItem("WAN 2.1 + pano LoRA", "wan")
        self.generator.addItem("WorldStereo 2.0 (experimental)", "worldstereo")
        self.generator.currentIndexChanged.connect(self._on_generator)
        r.addWidget(self.generator)
        self._apply_generator_gate()

        self.views_lab = _lab("Views per rail")
        self.views = QSpinBox()
        self.views.setRange(1, 4)
        self.views.setValue(1)
        self.views.setToolTip("This build samples the rail direction only (1 view).")
        r.addWidget(self.views_lab)
        r.addWidget(self.views)

        self.order_lab = _lab("Rail order")
        self.rail_order = QListWidget()
        self.rail_order.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.rail_order.setDefaultDropAction(Qt.DropAction.MoveAction)
        self.rail_order.setMaximumHeight(110)
        self.rail_order.model().rowsMoved.connect(self._on_rail_order_moved)
        r.addWidget(self.order_lab)
        r.addWidget(self.rail_order)
        self.order_hint = _hint("Drag to set the order memory is carried across rails.")
        r.addWidget(self.order_hint)

        self.prompt_lab = _lab("WAN prompt (required)")
        r.addWidget(self.prompt_lab)
        self.prompt_box = QTextEdit()
        self.prompt_box.setMinimumHeight(72)
        self.prompt_box.setMaximumHeight(120)
        self.prompt_box.textChanged.connect(self._on_prompt_changed)
        r.addWidget(self.prompt_box)
        r.addWidget(
            _hint(
                "Describe the actual pano. WAN fills black holes in the mesh flight. "
                "Review 1440×720 clips before HiRes. Re-gen a rail after editing the prompt."
            )
        )

        self.seed_lab = _lab("Seed (WAN)")
        r.addWidget(self.seed_lab)
        self.seed = QSpinBox()
        self.seed.setRange(0, 2_147_483_647)
        self.seed.setValue(V.WAN_SEED)
        r.addWidget(self.seed)

        self.style_lab = _lab("Style suffix (graph id 10, optional)")
        r.addWidget(self.style_lab)
        self.style = QTextEdit()
        self.style.setPlainText(V.STYLE_SUFFIX)
        self.style.setMinimumHeight(56)
        r.addWidget(self.style)

        self.negative_lab = _lab("Negative (graph id 11)")
        r.addWidget(self.negative_lab)
        self.negative = QTextEdit()
        self.negative.setPlainText(V.NEGATIVE)
        self.negative.setMinimumHeight(48)
        r.addWidget(self.negative)

        self.frames_lab = _lab("HiRes frames (after Confirm WAN)")
        r.addWidget(self.frames_lab)
        self.frames = QComboBox()
        for value, label in V.HIRES_FRAMES_PRESETS:
            self.frames.addItem(label, value)
        self.frames.setCurrentIndex(0)
        r.addWidget(self.frames)

        self.seam_lab = _lab("HiRes seam feather (px at 8K)")
        r.addWidget(self.seam_lab)
        self.seam = QDoubleSpinBox()
        self.seam.setRange(0.0, 32.0)
        self.seam.setSingleStep(0.5)
        self.seam.setDecimals(1)
        self.seam.setValue(float(V.HIRES_GATE_EDGE))
        self.seam.setToolTip(
            "Gaussian sigma on the 8K photograph | WAN join. 0 is a hard mask edge. "
            "The 720p WAN clip has no seam; this only affects Confirm WAN → HiRes."
        )
        r.addWidget(self.seam)
        self.seam_hint = _hint("0 = hard cut. Default 16 blends the mask so the 8K paste is not a clean edge.")
        r.addWidget(self.seam_hint)

        self.debug_lab = _lab("Debug save")
        r.addWidget(self.debug_lab)
        self.dbg_off = QRadioButton("off (product default)")
        self.dbg_all = QRadioButton("all (graph widget; disk-heavy)")
        self.dbg_group = QButtonGroup(self)
        self.dbg_group.addButton(self.dbg_off)
        self.dbg_group.addButton(self.dbg_all)
        self.dbg_off.setChecked(True)
        r.addWidget(self.dbg_off)
        r.addWidget(self.dbg_all)

        row = QHBoxLayout()
        self.btn_gen = QPushButton("Generate WAN")
        self.btn_gen.setObjectName("btnPrimary")
        self.btn_gen.clicked.connect(self._on_generate)
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setObjectName("btnGhost")
        self.btn_stop.clicked.connect(self.requestStop.emit)
        row.addWidget(self.btn_gen)
        row.addWidget(self.btn_stop)
        r.addLayout(row)

        self.btn_regen = QPushButton("Re-gen selected rail")
        self.btn_regen.setObjectName("btnGhost")
        self.btn_regen.clicked.connect(self._on_regen_selected)
        r.addWidget(self.btn_regen)

        self.btn_hires = QPushButton("Confirm WAN → HiRes")
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
        self.preview_lab = QLabel("Live preview · mask | WAN fill")
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
        self.legend = QLabel("Left: validity mask (white = known, black = hole) · Right: WAN output")
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
        self.frame_view = ErpStill(empty_text="WAN frame")
        self.render_view.setMinimumHeight(180)
        self.mask_view.setMinimumHeight(180)
        self.frame_view.setMinimumHeight(180)
        split.addWidget(self.render_view)
        split.addWidget(self.mask_view)
        split.addWidget(self.frame_view)
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 1)
        split.setStretchFactor(2, 1)
        split.setSizes([280, 280, 280])
        v.addWidget(split, 1)
        self.view_hint = _hint(
            "Stage 1 is 1440×720 WAN. The right pane updates as soon as a frame exists, "
            "including a low-res latent preview during sampling. Confirm WAN before 8K HiRes."
        )
        v.addWidget(self.view_hint)
        return view

    def _set_preview_fit(self, fit: bool) -> None:
        self.render_view.set_fit(fit)
        self.mask_view.set_fit(fit)
        self.frame_view.set_fit(fit)
        self.btn_mask_fit.setChecked(fit)
        self.btn_mask_100.setChecked(not fit)

    def _worldstereo(self) -> bool:
        return self.generator.currentData() == "worldstereo"

    def set_quality(self, quality: bool) -> None:
        self._quality = bool(quality)
        self._apply_generator_gate()
        self._apply_mode()
        self._sync_buttons()

    def _apply_generator_gate(self) -> None:
        item = self.generator.model().item(1)
        ready = _ws_weights_ready()
        item.setEnabled(ready and self._quality)
        if not ready:
            tip = "Disabled until the WorldStereo FP8 weights and DINOv2 are READY."
        elif not self._quality:
            tip = "WorldStereo is Quality only. Fast and Scout do not offer it."
        else:
            tip = "Experimental. Off by default. WAN stays installed."
        item.setToolTip(tip)
        self.generator.setToolTip(tip)
        if not item.isEnabled() and self.generator.currentIndex() != 0:
            self.generator.setCurrentIndex(0)

    def _on_generator(self, _index: int) -> None:
        if self._worldstereo():
            self._saved_wan_seed = int(self.seed.value())
            if self.seed.value() == V.WAN_SEED:
                self.seed.setValue(WS.SEED)
        else:
            self.seed.setValue(self._saved_wan_seed)
        self._apply_mode()
        self._sync_buttons()
        if self._view_rail >= 0:
            self._show_rail_preview(self._view_rail, follow_live=False)

    def _apply_mode(self) -> None:
        ws = self._worldstereo()
        for widget in (
            self.views_lab,
            self.views,
            self.order_lab,
            self.rail_order,
            self.order_hint,
            self.render_view,
            self.retrieved_label,
        ):
            widget.setVisible(ws)
        for widget in (
            self.style_lab,
            self.style,
            self.negative_lab,
            self.negative,
            self.frames_lab,
            self.frames,
            self.seam_lab,
            self.seam,
            self.seam_hint,
            self.debug_lab,
            self.dbg_off,
            self.dbg_all,
        ):
            widget.setVisible(not ws)
        self.prompt_lab.setText("Prompt (required)" if ws else "WAN prompt (required)")
        self.seed_lab.setText("Seed" if ws else "Seed (WAN)")
        self.btn_gen.setText("Generate WorldStereo" if ws else "Generate WAN")
        self.btn_hires.setText("Confirm → HiRes" if ws else "Confirm WAN → HiRes")
        self.preview_lab.setText(
            "Live preview · render | mask | keyframe" if ws else "Live preview · mask | WAN fill"
        )
        self.legend.setText(
            "Render · mask (white = empty) · keyframe"
            if ws
            else "Left: validity mask (white = known, black = hole) · Right: WAN output"
        )
        self.frame_view._empty_text = "Keyframe" if ws else "WAN frame"
        self.frame_view.update()

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

    def _ws_folder(self, rail_index: int) -> str:
        path = scene_dir(self._scene_name()) / camplot_meta_rel(int(rail_index))
        direction = "look_forward"
        if path.is_file():
            try:
                name = str(json.loads(path.read_text(encoding="utf-8")).get("orientation") or "").strip()
            except json.JSONDecodeError:
                name = ""
            if name:
                direction = name
        return f"{int(rail_index):02d}_{direction}"

    def _ws_rail_dir(self, rail_index: int) -> Path:
        return scene_dir(self._scene_name()) / "worldstereo" / "rails" / self._ws_folder(rail_index)

    def restore_view(self) -> None:
        self.refresh(keep_preview=True)

    def _wan_prompt_text(self) -> str:
        return wan_prompt_from_stored(self.prompt_box.toPlainText())

    def _load_wan_prompt(self, scene: str) -> str:
        extra = wan_prompt_path(scene)
        if extra.is_file():
            return wan_prompt_from_stored(extra.read_text(encoding="utf-8"))
        stored = prompt_path(scene)
        raw = stored.read_text(encoding="utf-8") if stored.is_file() else ""
        return wan_prompt_from_stored(raw)

    def _on_prompt_changed(self) -> None:
        if self._loading_prompt:
            return
        text = self._wan_prompt_text()
        path = wan_prompt_path(self._scene_name())
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + ("\n" if text else ""), encoding="utf-8")
        self._sync_buttons()

    def _dataset(self) -> dict:
        return sync_dataset_from_confirmed(self._scene_name())

    def _kept_rows(self) -> list[dict]:
        return [row for row in self._dataset().get("rails") or [] if row.get("keep", True)]

    def _wan_ready_indices(self) -> list[int]:
        root = scene_dir(self._scene_name())
        ready = []
        for row in self._kept_rows():
            idx = int(row["index"])
            path = root / str(row.get("wan") or wan_video_rel(idx))
            if path.is_file() and path.stat().st_size > 0:
                ready.append(idx)
        return ready

    def _missing_wan_indices(self) -> list[int]:
        have = set(self._wan_ready_indices())
        return [int(row["index"]) for row in self._kept_rows() if int(row["index"]) not in have]

    def _sync_buttons(self) -> None:
        empty = not self._wan_prompt_text()
        ready = bool(self._rails_ready())
        missing = self._missing_wan_indices()
        wan_ready = self._wan_ready_indices()
        can_job = not empty and ready and not self._busy
        if self._worldstereo():
            weights = _ws_weights_ready()
            self.btn_gen.setEnabled(can_job and weights and bool(self._ws_order))
            self.btn_regen.setEnabled(can_job and weights and self._view_rail >= 0)
            self.btn_hires.setEnabled(False)
            self.btn_hires.setToolTip("Pinhole HiRes is not in this build. WAN HiRes is unchanged.")
        else:
            self.btn_gen.setEnabled(can_job and bool(self._kept_rows()))
            self.btn_regen.setEnabled(can_job and self._view_rail >= 0)
            self.btn_hires.setEnabled(can_job and bool(wan_ready))
            self.btn_hires.setToolTip("")
        if self._worldstereo() and self._busy:
            self.gate_hint.setText("WorldStereo running — one process for every rail. Stop frees the GPU.")
            return
        if self._busy:
            if self._job_kind == "hires":
                self.gate_hint.setText("HiRes running — Stop kills the current rail.")
            else:
                self.gate_hint.setText(
                    "WAN running — Stop kills the current rail. Click another rail to inspect it."
                )
        elif empty:
            self.gate_hint.setText(
                "Empty prompt blocks Generate."
                if self._worldstereo()
                else "Empty WAN prompt blocks Generate. Describe the actual pano "
                "(WAN fills black holes, not green)."
            )
        elif not ready:
            self.gate_hint.setText("Confirm rails on Geometry first (Play each rail, then Confirm rails).")
        elif self._worldstereo() and not _ws_weights_ready():
            self.gate_hint.setText("WorldStereo stays off until the FP8 weights and DINOv2 are READY.")
        elif self._worldstereo():
            self.gate_hint.setText(
                "Experimental Quality generator. WAN stays installed. "
                "One view per rail. Confirm → HiRes stays on the WAN path."
            )
        elif missing:
            self.gate_hint.setText(
                f"Generate WAN for {len(missing)} missing rail(s). "
                "Then click a rail to preview. Confirm WAN before HiRes."
            )
        elif wan_ready:
            self.gate_hint.setText(
                "WAN clips are ready. Re-gen a rail if it hallucinated, or Confirm WAN → HiRes."
            )
        else:
            self.gate_hint.setText("Confirm rails is on. Generate WAN first (720p). HiRes is a second step.")

    def refresh(self, keep_preview: bool = False) -> None:
        scene = self._scene_name()
        text = self._load_wan_prompt(scene)
        self._loading_prompt = True
        self.prompt_box.setPlainText(text)
        self._loading_prompt = False
        self._fill_rail_rows()
        self._sync_ws_order()
        self._sync_buttons()
        if keep_preview and self._view_rail >= 0:
            self._show_rail_preview(self._view_rail, follow_live=self._follow_live)
        elif not self._busy:
            if self._view_rail >= 0:
                self._show_rail_preview(self._view_rail, follow_live=False)
            else:
                rows = self._kept_rows() or (self._dataset().get("rails") or [])
                if rows:
                    self._select_rail(int(rows[0]["index"]), follow_live=False)

    def _fill_rail_rows(self) -> None:
        while self.rail_lay.count():
            item = self.rail_lay.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self._rail_rows.clear()
        for btn in list(self._rail_pick.buttons()):
            self._rail_pick.removeButton(btn)
        data = self._dataset()
        rows = data.get("rails") or []
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
            pick.clicked.connect(lambda _=False, idx=rail_index: self._select_rail(idx, follow_live=False))
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
        if row.get("hires"):
            return "HiRes"
        return str(row.get("status") or "queued")

    def set_rail_status(self, rail_index: int, text: str) -> None:
        slot = self._rail_rows.get(int(rail_index))
        if slot is not None:
            slot["status"].setText(text)

    def _select_rail(self, rail_index: int, *, follow_live: bool) -> None:
        self._view_rail = int(rail_index)
        self._follow_live = bool(follow_live) or (self._view_rail == self._job_rail)
        slot = self._rail_rows.get(self._view_rail)
        if slot is not None:
            slot["pick"].setChecked(True)
        self._sync_buttons()
        self._show_rail_preview(self._view_rail, follow_live=self._follow_live)

    def _paths_for_rail(self, rail_index: int) -> tuple[Path, Path]:
        root = scene_dir(self._scene_name())
        preview = root / wan_preview_dir_rel(rail_index)
        mask_still = preview / "live_mask.png"
        live_rgb = preview / "live_rgb.png"
        wan = root / wan_video_rel(rail_index)
        mask_vid = root / control_mask_rel(rail_index)
        data = self._dataset()
        for row in data.get("rails") or []:
            if int(row.get("index", -1)) == int(rail_index):
                wan = root / str(row.get("wan") or wan_video_rel(rail_index))
                mask_vid = root / str(row.get("mask") or control_mask_rel(rail_index))
                break
        mask = mask_still if mask_still.is_file() else mask_vid
        rgb = wan if wan.is_file() else live_rgb
        return mask, rgb

    def _show_ws_preview(self, rail_index: int) -> None:
        rail_dir = self._ws_rail_dir(rail_index)
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
        self.retrieved_label.setText(f"retrieved {count} memory frames")
        self._stop_preview_play()
        self._kf_dir = keyframes if keyframes.is_dir() else None
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

    def show_ws(self, rail_index: int, phase: str, retrieved: int | None = None) -> None:
        self._job_rail = int(rail_index)
        self.set_rail_status(self._job_rail, phase)
        if retrieved is not None:
            self.retrieved_label.setText(f"retrieved {int(retrieved)} memory frames")
        if self._follow_live or self._view_rail == self._job_rail:
            self._show_ws_preview(self._job_rail)

    def _show_rail_preview(self, rail_index: int, *, follow_live: bool) -> None:
        if self._worldstereo():
            self._show_ws_preview(rail_index)
            return
        root = scene_dir(self._scene_name())
        preview = root / wan_preview_dir_rel(rail_index)
        mask_still = preview / "live_mask.png"
        live_rgb = preview / "live_rgb.png"
        mask_vid = root / control_mask_rel(rail_index)
        wan = root / wan_video_rel(rail_index)
        if follow_live:
            if mask_still.is_file():
                self._stop_preview_play()
                self.mask_view.set_path(mask_still)
            elif mask_vid.is_file():
                self._play_mask_only(mask_vid)
            else:
                self.mask_view.set_pil(None)
            if live_rgb.is_file():
                self.frame_view.set_path(live_rgb)
            else:
                self.frame_view.set_pil(None)
            return
        if wan.is_file() and mask_vid.is_file():
            self._start_video_preview(mask_vid, wan)
            return
        if mask_still.is_file():
            self._stop_preview_play()
            self.mask_view.set_path(mask_still)
        elif mask_vid.is_file():
            self._play_mask_only(mask_vid)
        else:
            self.mask_view.set_pil(None)
        if live_rgb.is_file():
            self.frame_view.set_path(live_rgb)
        else:
            self.frame_view.set_pil(None)

    def set_job_progress(self, current: int, total: int, pct: int, msg: str = "") -> None:
        if total > 0:
            self.progress.setRange(0, total)
            self.progress.setValue(min(current, total))
        elif pct >= 0:
            self.progress.setRange(0, 100)
            self.progress.setValue(min(100, max(0, pct)))
        if self._job_rail < 0 or not msg:
            return
        if "wan step" in msg:
            step = msg.split("wan step", 1)[1].strip()
            self.set_rail_status(self._job_rail, f"WAN {step}")
        elif msg.startswith("wan "):
            self.set_rail_status(self._job_rail, msg.replace("wan ", "WAN ", 1))
        elif msg.startswith("hires "):
            self.set_rail_status(self._job_rail, msg.replace("hires ", "HiRes ", 1))

    def on_generate_start(self, rail_indices: list[int], kind: str = "wan") -> None:
        self._busy = True
        self._job_kind = kind
        self._job_rail = int(rail_indices[0]) if rail_indices else -1
        self._follow_live = True
        self._fill_rail_rows()
        queued = {int(i) for i in rail_indices}
        for idx in queued:
            self.set_rail_status(idx, "queued")
        if self._job_rail >= 0:
            self._select_rail(self._job_rail, follow_live=True)
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self._sync_buttons()

    def on_job_finished(self, label: str, code: int, *, more: bool) -> None:
        if label not in ("wan", "hires", "worldstereo"):
            return
        if more:
            return
        self._busy = False
        self._job_kind = ""
        self._fill_rail_rows()
        self._sync_buttons()
        if code != 0 and self._job_rail >= 0:
            self.set_rail_status(self._job_rail, "failed")
        elif self._view_rail >= 0:
            self._show_rail_preview(self._view_rail, follow_live=False)

    def apply_wan_rail(self, rail_index: int, mask_path: str, rgb_path: str) -> None:
        self._job_rail = int(rail_index)
        self.set_rail_status(self._job_rail, "WAN loading…")
        if self._follow_live or self._view_rail == self._job_rail:
            self._view_rail = self._job_rail
            slot = self._rail_rows.get(self._view_rail)
            if slot is not None:
                slot["pick"].setChecked(True)
            mask = Path(mask_path)
            if mask.is_file() and mask.suffix.lower() == ".mp4":
                self._play_mask_only(mask)
            elif mask.is_file():
                self._stop_preview_play()
                self.mask_view.set_path(mask)
            self.frame_view.set_pil(None)

    def apply_wan_preview(
        self, rail_index: int, step: int, mask_path: str, rgb_path: str, steps: int = 0
    ) -> None:
        self._job_rail = int(rail_index)
        label = f"WAN {int(step)}/{int(steps)}" if steps else f"WAN step {int(step)}"
        self.set_rail_status(self._job_rail, label)
        if not (self._follow_live or self._view_rail == self._job_rail):
            return
        self._preview_timer.stop()
        self._release_captures()
        mask_file = Path(mask_path)
        rgb_file = Path(rgb_path)
        if mask_file.is_file():
            if mask_file.suffix.lower() == ".mp4":
                self._play_mask_only(mask_file)
            else:
                self.mask_view.set_path(mask_file)
        if rgb_file.is_file():
            self.frame_view.set_path(rgb_file)

    def apply_wan_frame(self, rail_index: int, frame_idx: int, mask_path: str, rgb_path: str) -> None:
        self._job_rail = int(rail_index)
        self.set_rail_status(self._job_rail, f"WAN frame {int(frame_idx) + 1}")
        if not (self._follow_live or self._view_rail == self._job_rail):
            return
        self._preview_timer.stop()
        self._release_captures()
        mask_file = Path(mask_path)
        rgb_file = Path(rgb_path)
        if mask_file.is_file() and mask_file.suffix.lower() != ".mp4":
            self.mask_view.set_path(mask_file)
        if rgb_file.is_file():
            self.frame_view.set_path(rgb_file)

    def apply_wan_done(self, rail_index: int, video_path: str) -> None:
        self._job_rail = int(rail_index)
        self.set_rail_status(self._job_rail, "WAN")
        if self._follow_live or self._view_rail == self._job_rail:
            root = scene_dir(self._scene_name())
            mask_vid = root / control_mask_rel(int(rail_index))
            video = Path(video_path)
            if mask_vid.is_file() and video.is_file():
                self._start_video_preview(mask_vid, video)

    def apply_hires_frame(self, rail_index: int, frame_idx: int, mask_path: str, rgb_path: str) -> None:
        self._job_rail = int(rail_index)
        self.set_rail_status(self._job_rail, f"HiRes {int(frame_idx) + 1}")
        if not (self._follow_live or self._view_rail == self._job_rail):
            return
        self._stop_preview_play()
        if mask_path:
            self.mask_view.set_path(Path(mask_path))
        if rgb_path:
            self.frame_view.set_path(Path(rgb_path))

    def _play_mask_only(self, mask_path: Path) -> None:
        self._preview_timer.stop()
        self._release_captures()
        self._mask_path = mask_path
        self._rgb_path = None
        cap = cv2.VideoCapture(str(mask_path))
        if not cap.isOpened():
            cap.release()
            return
        self._mask_cap = cap
        self._preview_tick()
        self._preview_timer.start()

    def _start_video_preview(self, mask_path: Path, rgb_path: Path) -> None:
        self._preview_timer.stop()
        self._release_captures()
        self._mask_path = mask_path
        self._rgb_path = rgb_path
        mask_cap = cv2.VideoCapture(str(mask_path))
        rgb_cap = cv2.VideoCapture(str(rgb_path))
        if not mask_cap.isOpened() or not rgb_cap.isOpened():
            mask_cap.release()
            rgb_cap.release()
            return
        self._mask_cap = mask_cap
        self._rgb_cap = rgb_cap
        self._preview_tick()
        self._preview_timer.start()

    def _preview_tick(self) -> None:
        if self._mask_cap is None:
            return
        ok_m, frame_m = self._mask_cap.read()
        if not ok_m:
            self._mask_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok_m, frame_m = self._mask_cap.read()
            if not ok_m:
                return
        if len(frame_m.shape) == 3:
            gray = cv2.cvtColor(frame_m, cv2.COLOR_BGR2GRAY)
            mask_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        else:
            mask_rgb = cv2.cvtColor(frame_m, cv2.COLOR_GRAY2RGB)
        self.mask_view.set_pil(Image.fromarray(mask_rgb))
        if self._render_cap is not None:
            ok_s, frame_s = self._render_cap.read()
            if not ok_s:
                self._render_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok_s, frame_s = self._render_cap.read()
            if ok_s:
                self.render_view.set_pil(Image.fromarray(cv2.cvtColor(frame_s, cv2.COLOR_BGR2RGB)))
            frame_index = max(0, int(self._render_cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1)
            shot = None if self._kf_dir is None else self._kf_dir / f"{frame_index:03d}.png"
            if shot is not None and shot.is_file():
                self.frame_view.set_path(shot)
            else:
                self.frame_view.set_pil(None)
            return
        if self._rgb_cap is None:
            return
        ok_r, frame_r = self._rgb_cap.read()
        if not ok_r:
            self._rgb_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok_r, frame_r = self._rgb_cap.read()
            if not ok_r:
                return
        frame_rgb = cv2.cvtColor(frame_r, cv2.COLOR_BGR2RGB)
        self.frame_view.set_pil(Image.fromarray(frame_rgb))

    def _release_captures(self) -> None:
        if self._mask_cap is not None:
            self._mask_cap.release()
            self._mask_cap = None
        if self._rgb_cap is not None:
            self._rgb_cap.release()
            self._rgb_cap = None
        if self._render_cap is not None:
            self._render_cap.release()
            self._render_cap = None

    def _stop_preview_play(self) -> None:
        self._preview_timer.stop()
        self._release_captures()

    def _on_open_frames(self) -> None:
        if self._worldstereo():
            folder = scene_dir(self._scene_name()) / "worldstereo" / "rails"
        else:
            folder = scene_dir(self._scene_name()) / "hires" / "frames"
        folder.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))

    def _write_request(self, rails: list[int]) -> Path | None:
        scene = self._scene_name()
        text = self._wan_prompt_text()
        if not text:
            self.requestJob.emit(["__gen_error__", "empty WAN prompt blocks Generate"])
            return None
        if not self._rails_ready():
            self.requestJob.emit(["__gen_error__", "Confirm rails on Geometry first"])
            return None
        if not rails:
            self.requestJob.emit(["__gen_error__", "no rails selected"])
            return None
        debug = "all" if self.dbg_all.isChecked() else "off"
        frames = str(self.frames.currentData() or V.HIRES_FRAMES)
        payload = {
            "scene": scene,
            "prompt": text,
            "seed": int(self.seed.value()),
            "style_suffix": self.style.toPlainText().strip(),
            "negative": self.negative.toPlainText().strip(),
            "frames": frames,
            "debug_save": debug,
            "rails": [int(i) for i in rails],
            "wan_width": V.WAN_WIDTH,
            "wan_height": V.WAN_HEIGHT,
            "wan_length": V.WAN_LENGTH,
            "wan_steps": V.WAN_STEPS,
            "hires_output_width": V.HIRES_OUTPUT_WIDTH,
            "hires_base_mode": V.HIRES_BASE_MODE,
            "hires_moge_level": V.HIRES_MOGE_LEVEL,
            "hires_gate_edge": float(self.seam.value()),
            "hires_edge_erode": int(V.HIRES_EDGE_ERODE),
            "hires_tone_mode": V.HIRES_TONE_MODE,
            "create_video_fps": V.CREATE_VIDEO_FPS,
            "create_video_bit_depth": V.CREATE_VIDEO_BIT_DEPTH,
        }
        work = scene_dir(scene) / "_work"
        work.mkdir(parents=True, exist_ok=True)
        req = work / "generate_request.json"
        req.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return req

    def _missing_weights(self) -> str:
        missing = []
        for ident, path in _wan_weight_rows():
            if not path.is_file() or path.stat().st_size <= 0:
                missing.append(ident)
        return ",".join(missing)

    def _on_generate(self) -> None:
        if self._worldstereo():
            self._emit_worldstereo(self._ws_order, only=False, start_rail=0)
            return
        missing = self._missing_wan_indices()
        if not missing:
            self.requestJob.emit(
                [
                    "__gen_error__",
                    "all kept rails already have WAN — Re-gen selected, or ditch a rail on Reconstruct",
                ]
            )
            return
        self._emit_wan(missing)

    def _on_regen_selected(self) -> None:
        if self._worldstereo():
            if self._view_rail < 0:
                self.requestJob.emit(["__gen_error__", "click a rail to Re-gen"])
                return
            self._emit_worldstereo([self._view_rail], only=True, start_rail=self._view_rail)
            return
        if self._view_rail < 0:
            self.requestJob.emit(["__gen_error__", "click a rail to Re-gen"])
            return
        self._emit_wan([self._view_rail])

    def request_wan_rails(self, rails: list[int]) -> None:
        self._emit_wan([int(i) for i in rails])

    def _emit_wan(self, rails: list[int]) -> None:
        req = self._write_request(rails)
        if req is None:
            return
        scene = self._scene_name()
        self.requestJob.emit(["__generate__", scene, str(req), self._missing_weights()])

    def _emit_worldstereo(self, rails: list[int], *, only: bool, start_rail: int) -> None:
        scene = self._scene_name()
        text = self._wan_prompt_text()
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
        work = scene_dir(scene) / "_work"
        work.mkdir(parents=True, exist_ok=True)
        req = work / "worldstereo_request.json"
        req.write_text(
            json.dumps(
                {
                    "scene": scene,
                    "prompt": text,
                    "seed": int(self.seed.value()),
                    "rails": [int(i) for i in rails],
                    "views": int(self.views.value()),
                    "generator": "worldstereo",
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        self.requestJob.emit(
            [
                "__worldstereo__",
                scene,
                str(req),
                ",".join(str(int(i)) for i in rails),
                "1" if only else "0",
                str(int(start_rail)),
                str(int(self.seed.value())),
            ]
        )

    def _on_hires(self) -> None:
        if self._worldstereo():
            self.requestJob.emit(
                ["__gen_error__", "Pinhole HiRes is not in this build. WAN outputs were not changed."]
            )
            return
        rails = self._wan_ready_indices()
        if not rails:
            self.requestJob.emit(["__gen_error__", "Generate WAN and keep at least one rail first"])
            return
        req = self._write_request(rails)
        if req is None:
            return
        data = self._dataset()
        data["wan_confirmed"] = True
        dataset_path(self._scene_name()).write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        self.requestJob.emit(["__hires__", self._scene_name(), str(req), ""])
