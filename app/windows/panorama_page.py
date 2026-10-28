"""Phase 1 Panorama page — dummy docs/Votion_3DGS_2.0_Phase1.html §01."""

from __future__ import annotations

from pathlib import Path

from PIL import Image
from PySide6.QtCore import QEvent, Qt, Signal
from PySide6.QtGui import QDragEnterEvent, QDropEvent, QPixmap
from PySide6.QtWidgets import (
    QButtonGroup,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSlider,
    QSplitter,
    QStackedWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from app.widgets.erp_view import ErpStill, Look360, WarpPreview
from engine.pano import defaults as D
from engine.pano.aspect import aspect_error, is_2to1
from engine.pano.scene_io import pano_path


class DropZone(QLabel):
    pathDropped = Signal(str)

    def __init__(self, hint: str, parent: QWidget | None = None) -> None:
        super().__init__(hint, parent)
        self.setAcceptDrops(True)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setWordWrap(True)
        self.setMinimumHeight(96)
        self.setScaledContents(False)
        self.setProperty("class", "dropZone")
        self.setStyleSheet(
            "border: 1px dashed #2a3d52; background: #081018; color: #8b9bb4; font-size: 11px;"
        )
        self._hint = hint
        self.path: str = ""
        self._thumb: QPixmap | None = None

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # noqa: N802
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:  # noqa: N802
        urls = event.mimeData().urls()
        if not urls:
            return
        path = urls[0].toLocalFile()
        if path:
            self.set_path(path)
            self.pathDropped.emit(path)

    def set_path(self, path: str) -> None:
        self.path = path
        name = Path(path).name
        self.setToolTip(name)
        pix = QPixmap(path)
        if pix.isNull():
            self._thumb = None
            self.setPixmap(QPixmap())
            self.setText(name or self._hint)
            return
        self._thumb = pix
        self.setText("")
        self._rescale_thumb()

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._rescale_thumb()

    def _rescale_thumb(self) -> None:
        if self._thumb is None or self._thumb.isNull():
            return
        inner = self.contentsRect().adjusted(4, 4, -4, -4)
        if inner.width() < 8 or inner.height() < 8:
            return
        self.setPixmap(
            self._thumb.scaled(
                inner.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802
        from PySide6.QtWidgets import QFileDialog

        chosen, _ = QFileDialog.getOpenFileName(
            self,
            "Choose image",
            "",
            "Images (*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff);;All files (*.*)",
        )
        if chosen:
            self.set_path(chosen)
            self.pathDropped.emit(chosen)


def _lab(text: str) -> QLabel:
    lab = QLabel(text)
    lab.setProperty("class", "fLab")
    return lab


def _hint(text: str) -> QLabel:
    lab = QLabel(text)
    lab.setWordWrap(True)
    lab.setProperty("class", "fHint")
    return lab


class PanoramaPage(QWidget):
    requestJob = Signal(list)
    requestStop = Signal()

    def __init__(self, scene_name_fn, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._scene_name = scene_name_fn
        self._mode = "2d"
        self._photo: Image.Image | None = None
        self._erp_path: str = ""
        self._last_pano: Path | None = None
        self._showing_result = False

        body = QHBoxLayout(self)
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)
        body.addWidget(self._build_rail())
        body.addWidget(self._build_view(), 1)
        self._show_mode("2d")
        self._sync_generate()

    def _build_rail(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(300)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        rail = QWidget()
        rail.setObjectName("rail")
        r = QVBoxLayout(rail)
        r.setContentsMargins(12, 12, 12, 14)
        r.setSpacing(4)

        r.addWidget(_lab("Input"))
        tog = QWidget()
        tog_l = QHBoxLayout(tog)
        tog_l.setContentsMargins(0, 0, 0, 0)
        tog_l.setSpacing(0)
        self.mode_group = QButtonGroup(self)
        self.mode_group.setExclusive(True)
        self.btn_2d = QPushButton("2D Images")
        self.btn_360 = QPushButton("360 Images")
        self.btn_text = QPushButton("text")
        for i, btn in enumerate((self.btn_2d, self.btn_360, self.btn_text)):
            btn.setCheckable(True)
            btn.setProperty("class", "modeChip")
            self.mode_group.addButton(btn, i)
            tog_l.addWidget(btn)
        self.btn_2d.setChecked(True)
        self.mode_group.idClicked.connect(self._on_mode)
        r.addWidget(tog)

        self.mode_stack = QStackedWidget()
        self.mode_stack.addWidget(self._pane_2d())
        self.mode_stack.addWidget(self._pane_360())
        self.mode_stack.addWidget(self._pane_text())
        r.addWidget(self.mode_stack)
        r.addStretch(1)
        scroll.setWidget(rail)
        return scroll

    def _pane_2d(self) -> QWidget:
        w = QWidget()
        r = QVBoxLayout(w)
        r.setContentsMargins(0, 0, 0, 0)
        r.setSpacing(4)
        r.addWidget(_lab("2D image"))
        self.drop_2d = DropZone("Drop 2D photo · double-click to browse")
        self.drop_2d.pathDropped.connect(self._on_2d_image)
        r.addWidget(self.drop_2d)

        hfov_row = QHBoxLayout()
        hfov_row.addWidget(_lab("h_fov"))
        hfov_row.addStretch(1)
        self.hfov_edit = QDoubleSpinBox()
        self.hfov_edit.setRange(D.H_FOV_DEG_MIN, D.H_FOV_DEG_MAX)
        self.hfov_edit.setSingleStep(D.H_FOV_DEG_STEP)
        self.hfov_edit.setDecimals(1)
        self.hfov_edit.setValue(D.H_FOV_DEG_DEFAULT)
        self.hfov_edit.setKeyboardTracking(False)
        self.hfov_edit.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.hfov_edit.setFixedWidth(72)
        hfov_row.addWidget(self.hfov_edit)
        r.addLayout(hfov_row)
        self._hfov_ticks = int(round(1.0 / D.H_FOV_DEG_STEP))
        self.hfov = QSlider(Qt.Orientation.Horizontal)
        self.hfov.setRange(
            int(round(D.H_FOV_DEG_MIN * self._hfov_ticks)),
            int(round(D.H_FOV_DEG_MAX * self._hfov_ticks)),
        )
        self.hfov.setValue(int(round(D.H_FOV_DEG_DEFAULT * self._hfov_ticks)))
        self.hfov.valueChanged.connect(self._on_hfov_slider)
        self.hfov_edit.valueChanged.connect(self._on_hfov_edit)
        r.addWidget(self.hfov)
        r.addWidget(_hint("Type or drag. MickmumpitzPanoWarp INPUT_TYPES 10–170, step 0.5."))

        r.addWidget(_lab("Surroundings"))
        self.surr_type = QRadioButton("Type a sentence")
        self.surr_flo = QRadioButton("Florence-2 caption, then edit")
        self.surr_type.setChecked(True)
        surr_g = QButtonGroup(self)
        surr_g.addButton(self.surr_type)
        surr_g.addButton(self.surr_flo)
        r.addWidget(self.surr_type)
        r.addWidget(self.surr_flo)
        self.btn_caption = QPushButton("Caption")
        self.btn_caption.setObjectName("btnGhost")
        self.btn_caption.clicked.connect(self._caption)
        self.btn_caption.setVisible(False)
        self.surr_flo.toggled.connect(self.btn_caption.setVisible)
        r.addWidget(self.btn_caption)
        self.surr_box = QTextEdit()
        self.surr_box.setPlaceholderText("Empty → block Generate")
        self.surr_box.setFixedHeight(64)
        self.surr_box.textChanged.connect(self._sync_generate)
        self.surr_box.installEventFilter(self)
        r.addWidget(self.surr_box)
        r.addWidget(_hint(f"Always appended: {D.SURROUNDINGS_SUFFIX}"))
        r.addWidget(_hint("Graph demo is not a silent default. Caption runs engine.workers.caption."))

        r.addWidget(_lab("Seed (IMAGE→PANO)"))
        self.seed_2d = QLineEdit(str(D.IMAGE_PANO_SEED_DEFAULT))
        r.addWidget(self.seed_2d)
        r.addWidget(_lab("Seed (seam INPAINT, node 63)"))
        self.seed_2d_seam = QLineEdit(str(D.IMAGE_SEAM_SEED_DEFAULT))
        r.addWidget(self.seed_2d_seam)
        r.addWidget(_lab("Upscaler prompt"))
        self.upscaler_2d = QLineEdit(D.UPSCALER_PROMPT_DEFAULT)
        r.addWidget(self.upscaler_2d)

        self.warn_ostris = QLabel("Ostris Edit not READY — Generate disabled. Use 360 Images or text.")
        self.warn_ostris.setWordWrap(True)
        self.warn_ostris.setProperty("class", "fWarn")
        self.warn_ostris.setStyleSheet(
            "border-left: 3px solid #ff4d6d; background: rgba(255,77,109,0.08); "
            "color: #e8b4be; font-size: 11px; padding: 6px;"
        )
        self.warn_ostris.setVisible(not D.OSTRIS_READY)
        r.addWidget(self.warn_ostris)

        row = QHBoxLayout()
        self.btn_gen_2d = QPushButton("Generate")
        self.btn_gen_2d.setObjectName("btnPrimary")
        self.btn_gen_2d.setEnabled(False)
        self.btn_gen_2d.clicked.connect(self._generate_2d)
        self.btn_stop_2d = QPushButton("Stop")
        self.btn_stop_2d.setObjectName("btnGhost")
        self.btn_stop_2d.clicked.connect(self.requestStop.emit)
        row.addWidget(self.btn_gen_2d)
        row.addWidget(self.btn_stop_2d)
        r.addLayout(row)
        return w

    def _pane_360(self) -> QWidget:
        w = QWidget()
        r = QVBoxLayout(w)
        r.setContentsMargins(0, 0, 0, 0)
        r.setSpacing(4)
        r.addWidget(_lab("360 file (2:1 only)"))
        self.drop_360 = DropZone("Drop ERP · reject non-2:1 · do not letterbox")
        self.drop_360.pathDropped.connect(self._on_360_image)
        r.addWidget(self.drop_360)
        r.addWidget(_lab("prompt.txt (WAN later)"))
        self.prompt_360 = QTextEdit()
        self.prompt_360.setPlaceholderText("Empty OK in P1")
        self.prompt_360.setFixedHeight(56)
        r.addWidget(self.prompt_360)
        r.addWidget(_hint("Store as-is even if not 8192. Quality HiRes refuses other widths unless you Upscale to 8K."))
        r.addWidget(_lab("Upscaler prompt"))
        self.upscaler_360 = QLineEdit(D.UPSCALER_PROMPT_DEFAULT)
        r.addWidget(self.upscaler_360)
        row = QHBoxLayout()
        self.btn_use = QPushButton("Use as pano.png")
        self.btn_use.setObjectName("btnPrimary")
        self.btn_use.clicked.connect(self._use_360)
        self.btn_up_8k = QPushButton("Upscale to 8K")
        self.btn_up_8k.setObjectName("btnGhost")
        self.btn_up_8k.clicked.connect(self._upscale_360)
        self.btn_stop_360 = QPushButton("Stop")
        self.btn_stop_360.setObjectName("btnGhost")
        self.btn_stop_360.clicked.connect(self.requestStop.emit)
        row.addWidget(self.btn_use)
        row.addWidget(self.btn_up_8k)
        row.addWidget(self.btn_stop_360)
        r.addLayout(row)
        return w

    def _pane_text(self) -> QWidget:
        w = QWidget()
        r = QVBoxLayout(w)
        r.setContentsMargins(0, 0, 0, 0)
        r.setSpacing(4)
        r.addWidget(_lab("Trigger prefix (locked)"))
        prefix = QLabel(D.TRIGGER_PREFIX)
        prefix.setWordWrap(True)
        prefix.setStyleSheet("color: #4da3ff; font-size: 10px; background: #081018; padding: 6px; border: 1px solid #1c2b3a;")
        r.addWidget(prefix)
        self.sentence = QTextEdit()
        self.sentence.setPlaceholderText("scene sentence after the comma")
        self.sentence.setFixedHeight(64)
        self.sentence.textChanged.connect(self._on_sentence)
        r.addWidget(self.sentence)
        r.addWidget(_hint("Empty scene → block Generate. No 2D image in this mode."))
        r.addWidget(_lab("prompt.txt (WAN later)"))
        self.prompt_preview = QTextEdit()
        self.prompt_preview.setReadOnly(True)
        self.prompt_preview.setFixedHeight(56)
        r.addWidget(self.prompt_preview)
        r.addWidget(_lab("Seed (TEXT→PANO)"))
        self.seed_text = QLineEdit(str(D.TEXT_PANO_SEED_DEFAULT))
        r.addWidget(self.seed_text)
        r.addWidget(_lab("Seed (seam INPAINT, node 31)"))
        self.seed_text_seam = QLineEdit(str(D.TEXT_SEAM_SEED_DEFAULT))
        r.addWidget(self.seed_text_seam)
        r.addWidget(_lab("Upscaler prompt"))
        self.upscaler_text = QLineEdit(D.UPSCALER_PROMPT_DEFAULT)
        r.addWidget(self.upscaler_text)
        row = QHBoxLayout()
        self.btn_gen_text = QPushButton("Generate")
        self.btn_gen_text.setObjectName("btnPrimary")
        self.btn_gen_text.clicked.connect(self._generate_text)
        self.btn_stop_text = QPushButton("Stop")
        self.btn_stop_text.setObjectName("btnGhost")
        self.btn_stop_text.clicked.connect(self.requestStop.emit)
        row.addWidget(self.btn_gen_text)
        row.addWidget(self.btn_stop_text)
        r.addLayout(row)
        self._on_sentence()
        return w

    def _build_view(self) -> QWidget:
        view = QWidget()
        view.setObjectName("viewport")
        v = QVBoxLayout(view)
        v.setContentsMargins(12, 10, 12, 12)

        left = QWidget()
        left_l = QVBoxLayout(left)
        left_l.setContentsMargins(0, 0, 0, 0)
        left_l.setSpacing(4)
        head = QHBoxLayout()
        self.left_lbl = QLabel("2:1 · live h_fov warp")
        self.left_lbl.setProperty("class", "viewLbl")
        head.addWidget(self.left_lbl)
        head.addStretch(1)
        self.btn_erp_fit = QPushButton("Fit")
        self.btn_erp_100 = QPushButton("100%")
        self.erp_view_group = QButtonGroup(self)
        self.erp_view_group.setExclusive(True)
        for btn, checked in ((self.btn_erp_fit, True), (self.btn_erp_100, False)):
            btn.setCheckable(True)
            btn.setChecked(checked)
            btn.setProperty("class", "modeChip")
            self.erp_view_group.addButton(btn)
            head.addWidget(btn)
        self.btn_erp_fit.clicked.connect(lambda: self._set_erp_fit(True))
        self.btn_erp_100.clicked.connect(lambda: self._set_erp_fit(False))
        left_l.addLayout(head)
        self.warp = WarpPreview()
        self.erp = ErpStill()
        self.left_stack = QStackedWidget()
        self.left_stack.addWidget(self.warp)
        self.left_stack.addWidget(self.erp)
        left_l.addWidget(self.left_stack, 1)
        self.fov_bar = QLabel(f"{D.GENERATE_WIDTH} × {D.GENERATE_HEIGHT}")
        self.fov_bar.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.fov_bar.setStyleSheet("background: #2a2a2e; color: #c8c8c8; font-size: 11px; padding: 4px;")
        left_l.addWidget(self.fov_bar)

        right = QWidget()
        right_l = QVBoxLayout(right)
        right_l.setContentsMargins(0, 0, 0, 0)
        right_l.setSpacing(4)
        rlbl = QLabel("360 · drag to look")
        rlbl.setProperty("class", "viewLbl")
        right_l.addWidget(rlbl)
        self.look = Look360()
        right_l.addWidget(self.look, 1)

        self.view_split = QSplitter(Qt.Orientation.Horizontal)
        self.view_split.setChildrenCollapsible(False)
        self.view_split.setHandleWidth(8)
        self.view_split.addWidget(left)
        self.view_split.addWidget(right)
        self.view_split.setStretchFactor(0, 1)
        self.view_split.setStretchFactor(1, 1)
        self.view_split.setSizes([520, 520])
        left.setMinimumWidth(160)
        right.setMinimumWidth(160)
        v.addWidget(self.view_split, 1)
        self.view_hint = QLabel(
            "Left Fit shows the full 2:1 in the pane. 100% is native pixels — drag to slide. "
            "Drag the split to resize. Right is a spherical look — drag to pan and tilt."
        )
        self.view_hint.setWordWrap(True)
        self.view_hint.setProperty("class", "fHint")
        v.addWidget(self.view_hint)
        return view

    def _set_erp_fit(self, fit: bool) -> None:
        self.warp.set_fit(fit)
        self.erp.set_fit(fit)
        self.btn_erp_fit.setChecked(fit)
        self.btn_erp_100.setChecked(not fit)

    def _on_mode(self, index: int) -> None:
        mode = ("2d", "360", "text")[index]
        self._show_mode(mode)

    def _show_generated_erp(self, path: Path) -> None:
        self._last_pano = path
        self._showing_result = True
        self.left_stack.setCurrentWidget(self.erp)
        self.erp.set_path(path)
        self.look.set_path(path)
        self.left_lbl.setText("2:1 ERP")
        with Image.open(path) as im:
            self.fov_bar.setText(f"{im.size[0]} × {im.size[1]}")

    def _show_mode(self, mode: str) -> None:
        self._mode = mode
        self.mode_stack.setCurrentIndex({"2d": 0, "360": 1, "text": 2}[mode])
        pano = self._last_pano if self._last_pano and self._last_pano.is_file() else None
        if pano is None:
            disk = pano_path(self._scene())
            if disk.is_file():
                pano = disk
        if mode == "2d" and pano is None:
            self.left_stack.setCurrentWidget(self.warp)
            self.left_lbl.setText("2:1 · live h_fov warp")
            self.fov_bar.setText(f"{D.GENERATE_WIDTH} × {D.GENERATE_HEIGHT}")
            self.view_hint.setText(
                "2D Images: left is the 2:1 ERP (photo on green). "
                "Right is a spherical look — drag to pan and tilt."
            )
            if self._photo is not None:
                self.warp.refresh()
                self.look.set_pil(self._preview_look())
            return
        self.view_hint.setText(
            "360 Images: store as pano.png. Optional Upscale to 8K. "
            "Right is a spherical look — drag to pan and tilt."
            if mode == "360"
            else (
                "text: no 2D image. Prefix stays locked. After Generate, prompt.txt = prefix + scene sentence. "
                "Right is a spherical look — drag to pan and tilt."
                if mode == "text"
                else "Left is the generated 2:1 ERP. Right is a spherical look — drag to pan and tilt."
            )
        )
        if pano is not None:
            self._show_generated_erp(pano)
        elif mode == "360" and self._erp_path:
            self.left_stack.setCurrentWidget(self.erp)
            self.erp.set_path(Path(self._erp_path))
            self.look.set_path(Path(self._erp_path))
            self.left_lbl.setText("2:1 ERP")
        else:
            self.left_stack.setCurrentWidget(self.erp)
            self.erp.set_pil(None)
            self.look.set_pil(None)
            self.left_lbl.setText("2:1 ERP")

    def restore_view(self) -> None:
        """Keep generated ERP/look when the Panorama chip is clicked again."""
        disk = pano_path(self._scene())
        if disk.is_file():
            if self._showing_result and self._last_pano and self._last_pano.resolve() == disk.resolve():
                self.erp.update()
                self.look.update()
                return
            self._show_generated_erp(disk)
            return
        if self._mode == "2d" and self._photo is not None:
            self.left_stack.setCurrentWidget(self.warp)
            self.warp.update()
            self.look.update()

    def _preview_look(self) -> Image.Image | None:
        if self._photo is None:
            return None
        from engine.pano.warp_preview import warp_photo_to_green_erp

        return warp_photo_to_green_erp(self._photo, h_fov_deg=self._hfov_deg(), width=512, height=256)

    def _hfov_deg(self) -> float:
        return float(self.hfov_edit.value())

    def _on_hfov_slider(self, ticks: int) -> None:
        deg = ticks / self._hfov_ticks
        self.hfov_edit.blockSignals(True)
        self.hfov_edit.setValue(deg)
        self.hfov_edit.blockSignals(False)
        self._apply_hfov(deg)

    def _on_hfov_edit(self, deg: float) -> None:
        self.hfov.blockSignals(True)
        self.hfov.setValue(int(round(float(deg) * self._hfov_ticks)))
        self.hfov.blockSignals(False)
        self._apply_hfov(float(deg))

    def _apply_hfov(self, deg: float) -> None:
        self.warp.set_hfov(deg)
        if self._mode == "2d" and not self._showing_result:
            self.look.set_pil(self._preview_look())

    def _on_2d_image(self, path: str) -> None:
        with Image.open(path) as im:
            im.load()
            self._photo = im.convert("RGB")
        self.warp.set_photo(self._photo)
        self.warp.set_hfov(self._hfov_deg())
        self._showing_result = False
        self.left_stack.setCurrentWidget(self.warp)
        self.left_lbl.setText("2:1 · live h_fov warp")
        self.look.set_pil(self._preview_look())
        self._sync_generate()

    def _on_360_image(self, path: str) -> None:
        with Image.open(path) as im:
            im.load()
            w, h = im.size
            if not is_2to1(w, h):
                self.drop_360.setText(aspect_error(w, h))
                self._erp_path = ""
                return
            rgb = im.convert("RGB")
        self._erp_path = path
        self.erp.set_pil(rgb)
        self.look.set_pil(rgb)
        self.fov_bar.setText(f"{w} × {h}")

    def _on_sentence(self) -> None:
        sentence = self.sentence.toPlainText().strip()
        if sentence:
            self.prompt_preview.setPlainText(D.TRIGGER_PREFIX + " " + sentence)
        else:
            self.prompt_preview.setPlainText(D.TRIGGER_PREFIX)
        self._sync_generate()

    def _sync_generate(self) -> None:
        if D.OSTRIS_READY:
            surr = D.with_surroundings_suffix(self.surr_box.toPlainText())
            self.btn_gen_2d.setEnabled(bool(self.drop_2d.path) and bool(surr))
        else:
            self.btn_gen_2d.setEnabled(False)
        sentence = self.sentence.toPlainText().strip()
        self.btn_gen_text.setEnabled(bool(sentence))

    def _scene(self) -> str:
        return self._scene_name() or "alpine_01"

    def _int_field(self, edit: QLineEdit, default: int) -> int:
        try:
            return int(edit.text().strip())
        except ValueError:
            return default

    def eventFilter(self, watched, event):  # noqa: N802
        if watched is getattr(self, "surr_box", None) and event.type() == QEvent.Type.FocusOut:
            self._commit_surroundings_suffix()
        return super().eventFilter(watched, event)

    def _commit_surroundings_suffix(self) -> None:
        body = self.surr_box.toPlainText()
        surr = D.with_surroundings_suffix(body)
        if surr and surr != body:
            self.surr_box.blockSignals(True)
            self.surr_box.setPlainText(surr)
            self.surr_box.blockSignals(False)

    def _generate_2d(self) -> None:
        if not D.OSTRIS_READY:
            return
        path = self.drop_2d.path
        self._commit_surroundings_suffix()
        surr = D.with_surroundings_suffix(self.surr_box.toPlainText())
        if not path or not surr:
            return
        self.requestJob.emit(
            [
                "--mode",
                "2d",
                "--action",
                "generate",
                "--scene",
                self._scene(),
                "--image",
                path,
                "--sentence",
                surr,
                "--seed",
                str(self._int_field(self.seed_2d, D.IMAGE_PANO_SEED_DEFAULT)),
                "--seam-seed",
                str(self._int_field(self.seed_2d_seam, D.IMAGE_SEAM_SEED_DEFAULT)),
                "--upscaler-prompt",
                self.upscaler_2d.text() or D.UPSCALER_PROMPT_DEFAULT,
                "--hfov",
                str(self._hfov_deg()),
            ]
        )

    def _use_360(self) -> None:
        if not self._erp_path:
            return
        self.requestJob.emit(
            [
                "--mode",
                "360",
                "--action",
                "use",
                "--scene",
                self._scene(),
                "--image",
                self._erp_path,
                "--prompt",
                self.prompt_360.toPlainText(),
            ]
        )

    def _upscale_360(self) -> None:
        if not self._erp_path:
            return
        self.requestJob.emit(
            [
                "--mode",
                "360",
                "--action",
                "upscale",
                "--scene",
                self._scene(),
                "--image",
                self._erp_path,
                "--prompt",
                self.prompt_360.toPlainText(),
                "--upscaler-prompt",
                self.upscaler_360.text(),
            ]
        )

    def _generate_text(self) -> None:
        sentence = self.sentence.toPlainText().strip()
        if not sentence:
            return
        self.requestJob.emit(
            [
                "--mode",
                "text",
                "--action",
                "generate",
                "--scene",
                self._scene(),
                "--sentence",
                sentence,
                "--seed",
                str(self._int_field(self.seed_text, D.TEXT_PANO_SEED_DEFAULT)),
                "--seam-seed",
                str(self._int_field(self.seed_text_seam, D.TEXT_SEAM_SEED_DEFAULT)),
                "--upscaler-prompt",
                self.upscaler_text.text() or D.UPSCALER_PROMPT_DEFAULT,
            ]
        )

    def _caption(self) -> None:
        if not self.drop_2d.path:
            return
        self.requestJob.emit(["__caption__", self.drop_2d.path])

    def apply_caption(self, text: str) -> None:
        self.surr_box.setPlainText(D.with_surroundings_suffix(text))

    def apply_pano(self, path: str) -> None:
        self._show_generated_erp(Path(path))
