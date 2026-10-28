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
    QButtonGroup,
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
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
    control_mask_rel,
    dataset_path,
    sync_dataset_from_confirmed,
    wan_preview_dir_rel,
    wan_video_rel,
)
from engine.job import REPO_ROOT
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

        r.addWidget(_lab("WAN prompt (required)"))
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

        r.addWidget(_lab("Seed (WAN)"))
        self.seed = QSpinBox()
        self.seed.setRange(0, 2_147_483_647)
        self.seed.setValue(V.WAN_SEED)
        r.addWidget(self.seed)

        r.addWidget(_lab("Style suffix (graph id 10, optional)"))
        self.style = QTextEdit()
        self.style.setPlainText(V.STYLE_SUFFIX)
        self.style.setMinimumHeight(56)
        r.addWidget(self.style)

        r.addWidget(_lab("Negative (graph id 11)"))
        self.negative = QTextEdit()
        self.negative.setPlainText(V.NEGATIVE)
        self.negative.setMinimumHeight(48)
        r.addWidget(self.negative)

        r.addWidget(_lab("HiRes frames (after Confirm WAN)"))
        self.frames = QComboBox()
        for value, label in V.HIRES_FRAMES_PRESETS:
            self.frames.addItem(label, value)
        self.frames.setCurrentIndex(0)
        r.addWidget(self.frames)

        r.addWidget(_lab("HiRes seam feather (px at 8K)"))
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
        r.addWidget(_hint("0 = hard cut. Default 16 blends the mask so the 8K paste is not a clean edge."))

        r.addWidget(_lab("Debug save"))
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
        preview_lab = QLabel("Live preview · mask | WAN fill")
        preview_lab.setProperty("class", "viewLbl")
        preview_head.addWidget(preview_lab)
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
        legend = QLabel("Left: validity mask (white = known, black = hole) · Right: WAN output")
        legend.setProperty("class", "fHint")
        v.addWidget(legend)
        split = QSplitter(Qt.Orientation.Horizontal)
        split.setChildrenCollapsible(False)
        split.setHandleWidth(8)
        self.mask_view = ErpStill(empty_text="Mask")
        self.frame_view = ErpStill(empty_text="WAN frame")
        self.mask_view.setMinimumHeight(180)
        self.frame_view.setMinimumHeight(180)
        split.addWidget(self.mask_view)
        split.addWidget(self.frame_view)
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 1)
        split.setSizes([420, 420])
        v.addWidget(split, 1)
        self.view_hint = _hint(
            "Stage 1 is 1440×720 WAN. The right pane updates as soon as a frame exists, "
            "including a low-res latent preview during sampling. Confirm WAN before 8K HiRes."
        )
        v.addWidget(self.view_hint)
        return view

    def _set_preview_fit(self, fit: bool) -> None:
        self.mask_view.set_fit(fit)
        self.frame_view.set_fit(fit)
        self.btn_mask_fit.setChecked(fit)
        self.btn_mask_100.setChecked(not fit)

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
        self.btn_gen.setEnabled(can_job and bool(self._kept_rows()))
        self.btn_regen.setEnabled(can_job and self._view_rail >= 0)
        self.btn_hires.setEnabled(can_job and bool(wan_ready))
        if self._busy:
            if self._job_kind == "hires":
                self.gate_hint.setText("HiRes running — Stop kills the current rail.")
            else:
                self.gate_hint.setText(
                    "WAN running — Stop kills the current rail. Click another rail to inspect it."
                )
        elif empty:
            self.gate_hint.setText(
                "Empty WAN prompt blocks Generate. Describe the actual pano "
                "(WAN fills black holes, not green)."
            )
        elif not ready:
            self.gate_hint.setText("Confirm rails on Geometry first (Play each rail, then Confirm rails).")
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

    def _show_rail_preview(self, rail_index: int, *, follow_live: bool) -> None:
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
        if label not in ("wan", "hires"):
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

    def _stop_preview_play(self) -> None:
        self._preview_timer.stop()
        self._release_captures()

    def _on_open_frames(self) -> None:
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

    def _on_hires(self) -> None:
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
