"""Phase 0 Home window — chrome from docs/Votion_3DGS_2.0_Phase0.html §02."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from PySide6.QtCore import QProcess, QProcessEnvironment, Qt, QTimer, Signal
from PySide6.QtGui import QCloseEvent, QColor, QFont, QTextBlockFormat, QTextCharFormat, QTextCursor
from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from app.windows.generate_page import GeneratePage
from app.windows.geometry_page import GeometryPage
from app.windows.panorama_page import PanoramaPage
from app.windows.reconstruct_page import ReconstructPage
from app.windows.splat_page import SplatPage
from engine.geo.defaults import HELP_GEOMETRY
from engine.job import (
    REPO_ROOT,
    camplot_argv,
    caption_argv,
    detect_fov_argv,
    download_argv,
    dummy_argv,
    known_pose_argv,
    litegs_build_argv,
    moge_argv,
    pano_argv,
    pinhole_hires_argv,
    splat_argv,
    worker_env,
    worldstereo_argv,
)
from engine.pano.defaults import HELP_PANORAMA
from engine.pano.scene_io import scene_dir
from engine.splat.status import litegs_spike_ok
from engine.sfm.defaults import HELP_RECONSTRUCT
from engine.splat.defaults import HELP_SPLAT
from engine.worldstereo.defaults import HELP_GENERATE

STAGES = (
    "Home",
    "Panorama",
    "Geometry",
    "Generate",
    "Reconstruct",
    "Splat",
)

HELP_HOME = """FIELD GUIDE
2D Images (default) or 360 / text → 8K ERP → Compute geometry → rails → Generate → splat.ply (LiteGS or MCMC Compact).
Ostris blocks 2D Generate until READY. h_fov 70. text keeps the trigger prefix.
After 2D Images, prompt.txt = outpaint instruction + surroundings. Generate strips that green instruction.
Preview the mesh flight. Four straight rails usually enough. Do not skip 8K. Extra rails do not fix glass.
Download models: Hub cache is the install folder .hf_cache, ready files in models.
Build LiteGS: compiles the vendored LiteGS CUDA kernels into this venv once (MSVC + CUDA 12.8), then records the 1-step spike in docs/env_report.txt. Train on the Splat tab is gated on that line. Never shipped in the installer (Inria non-commercial licence).

Votion 3DGS  2.0  Yik  MIT"""

LOG_IDLE = """STATUS // idle
Qt never imports torch. Dummy worker: 20 lines then exit. Stop kills the subprocess."""


def _stops_the_job(text: str) -> bool:
    """True when this log line is why a running job ended."""
    low = text.lower()
    if "out of memory" in low or "cuda error" in low or "vram //" in low:
        return True
    if low.startswith("traceback"):
        return True
    if "acceleratorerror:" in low or "outofmemoryerror:" in low or "runtimeerror:" in low:
        return True
    if text.startswith("STATUS // stopped"):
        return True
    if text.startswith("STATUS // crashed") or text.startswith("STATUS // process error"):
        return True
    if text.startswith("STATUS // exit "):
        tail = text.removeprefix("STATUS // exit ").strip().split()[0]
        try:
            return int(tail) != 0
        except ValueError:
            return False
    return False

MANIFEST_PATH = REPO_ROOT / "tools" / "model_manifest.json"


def _read_manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def _is_ready(root: Path, entry: dict) -> bool:
    base = root if entry.get("under") == "repo" else root / "models"
    path = base / entry["local_relative_path"]
    if not path.is_file():
        return False
    size = path.stat().st_size
    expected = entry.get("expected_bytes")
    if expected is not None:
        return size == int(expected)
    return size > 0


def gpu_names() -> list[str]:
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=8,
            check=False,
        )
        return [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
    except Exception:
        return []


def _fmt_bytes(n: int) -> str:
    """Format a measured byte count. Does not invent a size."""
    value = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            if unit == "B":
                return f"{int(value)} B"
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{n} B"


class StatCell(QFrame):
    def __init__(
        self,
        entry_id: str,
        name: str,
        value: str,
        state: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.entry_id = entry_id
        self._idle_value = value
        self._idle_state = state
        self.setObjectName("statCell")
        self.setProperty("class", "statCell")
        self.setFrameShape(QFrame.Shape.Box)
        self.setAccessibleName(f"{name} {value}")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(4)
        title = QLabel(name)
        title.setProperty("class", "statName")
        self.val = QLabel(value)
        self.val.setProperty("class", "statVal")
        self.bar = QProgressBar()
        self.bar.setObjectName("statBar")
        self.bar.setRange(0, 100)
        self.bar.setValue(0)
        self.bar.setTextVisible(False)
        self.bar.setFixedHeight(6)
        self.bar.hide()
        layout.addWidget(title)
        layout.addWidget(self.val)
        layout.addWidget(self.bar)
        self._apply_state(state)

    def _apply_state(self, state: str) -> None:
        self.val.setProperty("state", state)
        self.val.style().unpolish(self.val)
        self.val.style().polish(self.val)

    def set_status(self, value: str, state: str) -> None:
        self.val.setText(value)
        self.setAccessibleName(f"{self.entry_id} {value}")
        self._apply_state(state)

    def show_queued(self) -> None:
        self.bar.show()
        self.bar.setRange(0, 100)
        self.bar.setValue(0)
        self.set_status("queued", "MISSING")

    def show_progress(self, n: int, total: int, pct: int) -> None:
        self.bar.show()
        if pct < 0 or total <= 0:
            self.bar.setRange(0, 0)
            if n > 0:
                self.set_status(f"downloading  {_fmt_bytes(n)}", "MISSING")
            else:
                self.set_status("downloading", "MISSING")
            return
        self.bar.setRange(0, 100)
        self.bar.setValue(min(100, max(0, pct)))
        self.set_status(f"{pct}%  {_fmt_bytes(n)} / {_fmt_bytes(total)}", "MISSING")

    def show_done(self, ready: bool) -> None:
        self.bar.setRange(0, 100)
        self.bar.setValue(100 if ready else 0)
        if ready:
            self.set_status("READY", "READY")
            self.bar.show()
        else:
            self.set_status(self._idle_value, self._idle_state)
            self.bar.hide()

    def hide_bar(self) -> None:
        self.bar.hide()
        self.set_status(self._idle_value, self._idle_state)


class JobStrip(QWidget):
    """Always-on background job bar — visible on every stage chip, not log-only."""

    requestStop = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("jobStrip")
        row = QHBoxLayout(self)
        row.setContentsMargins(10, 4, 10, 4)
        row.setSpacing(8)
        self.lab = QLabel("idle")
        self.lab.setObjectName("jobStripLab")
        self.bar = QProgressBar()
        self.bar.setObjectName("jobBar")
        self.bar.setRange(0, 100)
        self.bar.setValue(0)
        self.bar.setTextVisible(False)
        self.bar.setFixedHeight(8)
        self.pct = QLabel("")
        self.pct.setObjectName("jobStripPct")
        self.pct.setMinimumWidth(36)
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setObjectName("jobStripStop")
        self.btn_stop.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_stop.clicked.connect(self.requestStop.emit)
        row.addWidget(self.lab, 1)
        row.addWidget(self.bar, 2)
        row.addWidget(self.pct)
        row.addWidget(self.btn_stop)

    def set_idle(self) -> None:
        self.lab.setText("idle")
        self.bar.setRange(0, 100)
        self.bar.setValue(0)
        self.pct.setText("")

    def set_busy(self, label: str, message: str = "") -> None:
        self.lab.setText(message or label)
        self.bar.setRange(0, 0)
        self.pct.setText("")

    def set_caption(self, message: str) -> None:
        text = message.strip()
        if text.startswith("STATUS //"):
            text = text[9:].strip()
        if text:
            self.lab.setText(text)

    def set_progress(self, current: int, total: int, pct: int, message: str = "") -> None:
        if message:
            self.set_caption(message)
        if pct < 0 or total <= 0:
            self.bar.setRange(0, 0)
            self.pct.setText("")
            return
        self.bar.setRange(0, 100)
        self.bar.setValue(min(100, max(0, pct)))
        self.pct.setText(f"{min(100, max(0, pct))}%")


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Votion 3DGS")
        self.resize(1180, 760)
        self._proc: QProcess | None = None
        self._splat_view_proc: QProcess | None = None
        self._splat_view_buf = ""
        self._want_splat_view = False
        self._splat_view_stopping = False
        self._expect_kill = False
        self._out_buf = ""
        self._stat_cells: dict[str, StatCell] = {}
        self._downloading = False
        self._florence_hub = ""
        self._florence_rev = ""
        self._job_label = ""
        self._sfm_images = "0"
        self._sfm_points = "0"
        self._sfm_sparse = ""
        self._sfm_preview = ""
        self._gpu_names = gpu_names()
        self._manifest = _read_manifest()

        root = QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        outer.addWidget(self._build_brand())
        outer.addWidget(self._build_stages())
        self.job_strip = JobStrip()
        self.job_strip.requestStop.connect(self._stop_job)
        outer.addWidget(self.job_strip)

        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self._build_pages())
        splitter.addWidget(self._build_drawers())
        splitter.setStretchFactor(0, 5)
        splitter.setStretchFactor(1, 0)
        splitter.setSizes([560, 180])
        outer.addWidget(splitter, 1)

        self._set_live("idle")
        self._refresh_status()
        self._append_log(LOG_IDLE)

    def _build_brand(self) -> QWidget:
        bar = QWidget()
        bar.setObjectName("brandBar")
        row = QHBoxLayout(bar)
        row.setContentsMargins(10, 8, 10, 8)
        title = QLabel("Votion 3DGS")
        title.setObjectName("brandTitle")
        self.live = QLabel("idle")
        self.live.setObjectName("liveBadge")
        self.gpu_chip = QLabel(self._gpu_label())
        self.gpu_chip.setObjectName("metaChip")
        self.idx_chip = QLabel("GPU 0")
        self.idx_chip.setObjectName("metaChip")
        row.addWidget(title)
        row.addWidget(self.live)
        row.addStretch(1)
        row.addWidget(self.gpu_chip)
        row.addWidget(self.idx_chip)
        return bar

    def _gpu_label(self) -> str:
        idx = int(self.cuda.value()) if hasattr(self, "cuda") else 0
        if 0 <= idx < len(self._gpu_names):
            return self._gpu_names[idx]
        if self._gpu_names:
            return self._gpu_names[0]
        return "GPU"

    def _build_stages(self) -> QWidget:
        bar = QWidget()
        bar.setObjectName("stageBar")
        row = QHBoxLayout(bar)
        row.setContentsMargins(8, 6, 8, 6)
        row.setSpacing(6)
        self.stage_group = QButtonGroup(self)
        self.stage_group.setExclusive(True)
        self.stage_btns: list[QPushButton] = []
        for i, name in enumerate(STAGES):
            btn = QPushButton(name)
            btn.setCheckable(True)
            btn.setProperty("class", "stageChip")
            font = btn.font()
            font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 1)
            btn.setFont(font)
            self.stage_group.addButton(btn, i)
            self.stage_btns.append(btn)
            row.addWidget(btn)
        row.addStretch(1)
        self.stage_btns[0].setChecked(True)
        self.stage_group.idClicked.connect(self._on_stage)
        for btn in self.stage_btns:
            btn.style().unpolish(btn)
            btn.style().polish(btn)
        return bar

    def _build_pages(self) -> QWidget:
        self.pages = QStackedWidget()
        self.pages.addWidget(self._build_home())
        self.pano_page = PanoramaPage(lambda: self.scene.text())
        self.pano_page.requestJob.connect(self._on_pano_job)
        self.pano_page.requestStop.connect(self._stop_job)
        self.pages.addWidget(self.pano_page)
        self.geo_page = GeometryPage(lambda: self.scene.text())
        self.geo_page.requestJob.connect(self._on_geo_job)
        self.geo_page.requestStop.connect(self._stop_job)
        self.pages.addWidget(self.geo_page)
        self.gen_page = GeneratePage(
            lambda: self.scene.text(),
            self.geo_page.generate_unlocked,
            release_media=self.geo_page.release_open_media,
            after_clear=self.geo_page.try_load_existing,
        )
        self.gen_page.requestJob.connect(self._on_gen_job)
        self.gen_page.requestStop.connect(self._stop_job)
        self.geo_page.railsChanged.connect(self.gen_page.refresh)
        self.pages.addWidget(self.gen_page)
        self.recon_page = ReconstructPage(lambda: self.scene.text())
        self.recon_page.requestJob.connect(self._on_recon_job)
        self.recon_page.requestStop.connect(self._stop_job)
        self.geo_page.railsChanged.connect(self.recon_page.refresh)
        self.pages.addWidget(self.recon_page)
        self.splat_page = SplatPage(lambda: self.scene.text())
        self.splat_page.requestJob.connect(self._on_splat_job)
        self.splat_page.requestStop.connect(self._stop_job)
        self.splat_page.controlLine.connect(self._on_splat_control)
        self.splat_page.logLine.connect(self._append_log)
        self.pages.addWidget(self.splat_page)
        return self.pages

    def _build_home(self) -> QWidget:
        page = QWidget()
        body = QHBoxLayout(page)
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)

        rail = QWidget()
        rail.setObjectName("rail")
        rail.setFixedWidth(280)
        r = QVBoxLayout(rail)
        r.setContentsMargins(12, 12, 12, 14)
        r.setSpacing(4)

        r.addWidget(self._lab("Scene name"))
        self.scene = QLineEdit("alpine_01")
        self.scene.textChanged.connect(self._on_scene)
        r.addWidget(self.scene)

        r.addWidget(self._lab("Profile"))
        self.profile = QComboBox()
        self.profile.addItem("Quality")
        self.profile.addItem("Fast (P4)")
        self.profile.addItem("Scout (P4)")
        model = self.profile.model()
        model.item(1).setEnabled(False)
        model.item(2).setEnabled(False)
        model.item(1).setToolTip("Not in this build.")
        model.item(2).setToolTip("Not in this build.")
        r.addWidget(self.profile)
        self.profile.currentIndexChanged.connect(self._on_profile)

        r.addWidget(self._lab("CUDA device"))
        self.cuda = QSpinBox()
        self.cuda.setRange(0, max(0, len(self._gpu_names) - 1) if self._gpu_names else 7)
        self.cuda.setValue(0)
        self.cuda.valueChanged.connect(self._on_cuda)
        r.addWidget(self.cuda)

        r.addWidget(self._lab("Download set"))
        self.radio_p1 = QRadioButton("P1 subset (default)")
        self.radio_all = QRadioButton("All (includes WAN 14B)")
        self.radio_p1.setChecked(True)
        self.radio_p1.toggled.connect(self._on_download_set)
        radios = QVBoxLayout()
        radios.setSpacing(4)
        radios.addWidget(self.radio_p1)
        radios.addWidget(self.radio_all)
        r.addLayout(radios)

        hint = QLabel(
            "Florence-2 only if captioning is enabled. MoGe is a P2 subset — not this default."
        )
        hint.setWordWrap(True)
        hint.setProperty("class", "fHint")
        r.addWidget(hint)

        btns = QHBoxLayout()
        self.btn_dl = QPushButton("Download models")
        self.btn_dl.setObjectName("btnPrimary")
        self.btn_dummy = QPushButton("Dummy job")
        self.btn_dummy.setObjectName("btnGhost")
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setObjectName("btnGhost")
        self.btn_dl.clicked.connect(self._start_download)
        self.btn_dummy.clicked.connect(self._start_dummy)
        self.btn_stop.clicked.connect(self._stop_job)
        btns.addWidget(self.btn_dl)
        btns.addWidget(self.btn_dummy)
        btns.addWidget(self.btn_stop)
        r.addLayout(btns)

        r.addWidget(self._lab("Splat trainer"))
        self.btn_litegs = QPushButton("Build LiteGS")
        self.btn_litegs.setObjectName("btnGhost")
        self.btn_litegs.clicked.connect(self._start_litegs_build)
        r.addWidget(self.btn_litegs)
        self.litegs_hint = QLabel("")
        self.litegs_hint.setWordWrap(True)
        self.litegs_hint.setProperty("class", "fHint")
        r.addWidget(self.litegs_hint)
        self._refresh_litegs_hint()
        r.addStretch(1)

        view = QWidget()
        view.setObjectName("viewport")
        v = QVBoxLayout(view)
        v.setContentsMargins(12, 10, 12, 12)
        lbl = QLabel("Model status")
        lbl.setProperty("class", "viewLbl")
        v.addWidget(lbl)
        self.stat_host = QWidget()
        self.stat_grid = QGridLayout(self.stat_host)
        self.stat_grid.setContentsMargins(0, 4, 0, 0)
        self.stat_grid.setSpacing(6)
        v.addWidget(self.stat_host)
        note = QLabel("Status machine: MISSING_* | READY. Subset READY is OK in P0.")
        note.setWordWrap(True)
        note.setProperty("class", "fHint")
        v.addWidget(note)
        v.addStretch(1)

        body.addWidget(rail)
        body.addWidget(view, 1)
        return page

    def _lab(self, text: str) -> QLabel:
        lab = QLabel(text)
        lab.setProperty("class", "fLab")
        return lab

    def _build_drawers(self) -> QWidget:
        draws = QWidget()
        draws.setObjectName("draws")
        row = QHBoxLayout(draws)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(0)

        log_col = QVBoxLayout()
        log_col.setContentsMargins(0, 0, 0, 0)
        log_col.setSpacing(0)
        self.btn_log = QPushButton("Log")
        self.btn_log.setProperty("class", "drawToggle")
        self.btn_log.setCheckable(True)
        self.btn_log.setChecked(True)
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setProperty("class", "logPane")
        self.log.setMinimumHeight(90)
        self.log.setMaximumHeight(168)
        self.btn_log.toggled.connect(self.log.setVisible)
        log_col.addWidget(self.btn_log)
        log_col.addWidget(self.log, 1)

        help_col = QVBoxLayout()
        help_col.setContentsMargins(0, 0, 0, 0)
        help_col.setSpacing(0)
        self.btn_help = QPushButton("Help")
        self.btn_help.setProperty("class", "drawToggle")
        self.btn_help.setCheckable(True)
        self.btn_help.setChecked(True)
        self.help = QTextEdit()
        self.help.setReadOnly(True)
        self.help.setProperty("class", "logPane")
        self.help.setMaximumHeight(168)
        self.help.setPlainText(HELP_HOME)
        self.btn_help.toggled.connect(self.help.setVisible)
        help_col.addWidget(self.btn_help)
        help_col.addWidget(self.help, 1)

        left = QWidget()
        left.setLayout(log_col)
        right = QWidget()
        right.setLayout(help_col)
        row.addWidget(left, 1)
        row.addWidget(right, 1)
        return draws

    def _on_stage(self, index: int) -> None:
        self.pages.setCurrentIndex(index)
        name = STAGES[index]
        if name == "Home":
            self.help.setPlainText(HELP_HOME)
        elif name == "Panorama":
            self.help.setPlainText(HELP_PANORAMA)
            self.pano_page.restore_view()
        elif name == "Geometry":
            self.help.setPlainText(HELP_GEOMETRY)
            self.geo_page.restore_view()
        elif name == "Generate":
            self.help.setPlainText(HELP_GENERATE)
            self.gen_page.restore_view()
        elif name == "Reconstruct":
            self.help.setPlainText(HELP_RECONSTRUCT)
            self.recon_page.refresh()
        elif name == "Splat":
            self.help.setPlainText(HELP_SPLAT)
            self.splat_page.refresh()
            self._want_splat_view = True
            self._ensure_splat_view()
        if name != "Splat":
            self._want_splat_view = False
            self._stop_splat_view()

    def _on_download_set(self, _checked: bool) -> None:
        if self._downloading:
            return
        self._refresh_status()

    def _on_profile(self, index: int) -> None:
        if hasattr(self, "gen_page"):
            self.gen_page.set_quality(int(index) == 0)

    def _on_cuda(self, value: int) -> None:
        self.idx_chip.setText(f"GPU {value}")
        self.gpu_chip.setText(self._gpu_label())

    def _set_live(self, text: str) -> None:
        self.live.setText(text)
        self.live.setProperty("busy", "true" if text != "idle" else "false")
        self.live.style().unpolish(self.live)
        self.live.style().polish(self.live)

    def _append_log(self, text: str) -> None:
        body = text.rstrip("\n")
        if not body:
            return
        fatal = _stops_the_job(body)
        cursor = self.log.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        if self.log.toPlainText():
            cursor.insertBlock()
        block = QTextBlockFormat()
        char = QTextCharFormat()
        if fatal:
            block.setBackground(QColor("#6b2418"))
            char.setForeground(QColor("#ffe8e0"))
            char.setFontWeight(QFont.Weight.DemiBold)
            self.btn_log.setChecked(True)
        else:
            char.setForeground(QColor("#8b9bb4"))
        lines = body.splitlines() or [body]
        cursor.setBlockFormat(block)
        cursor.setCharFormat(char)
        cursor.insertText(lines[0])
        for line in lines[1:]:
            cursor.insertBlock()
            cursor.setBlockFormat(block)
            cursor.setCharFormat(char)
            cursor.insertText(line)
        self.log.setTextCursor(cursor)
        self.log.ensureCursorVisible()

    def _download_set(self) -> str:
        return "p1" if self.radio_p1.isChecked() else "all"

    def _status_rows(self) -> list[tuple[str, str, str, str]]:
        download_set = self._download_set()
        wanted = "p1" if download_set == "p1" else "all"
        rows: list[tuple[str, str, str, str]] = []
        shown = {e["id"]: e for e in self._manifest["files"] if wanted in e.get("sets", [])}
        order = [
            "krea",
            "clip",
            "vae",
            "lora",
            "lora_outpaint",
            "realesrgan",
            "umt5",
            "clip_vision",
            "ultrasharp",
            "worldstereo_fp8",
            "worldstereo_config",
            "dinov2",
            "wan480_config",
        ]
        for key in order:
            entry = shown.get(key)
            if entry is None:
                continue
            if _is_ready(REPO_ROOT, entry):
                rows.append((entry["id"], entry["label"], "READY", "READY"))
            else:
                rows.append(
                    (entry["id"], entry["label"], f"MISSING_{entry['id']}", "MISSING")
                )
        if download_set == "p1":
            moge = next(e for e in self._manifest["files"] if e["id"] == "moge")
            if _is_ready(REPO_ROOT, moge):
                rows.append((moge["id"], moge["label"], "READY", "READY"))
            else:
                rows.append((moge["id"], moge["label"], "P2 subset", "NOTE"))
        for info in self._manifest.get("info_rows", []):
            rows.append((info["id"], info["label"], info["note"], "NOTE"))
        return rows

    def _refresh_status(self) -> None:
        while self.stat_grid.count():
            item = self.stat_grid.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self._stat_cells = {}
        for i, (entry_id, name, value, state) in enumerate(self._status_rows()):
            cell = StatCell(entry_id, name, value, state)
            self._stat_cells[entry_id] = cell
            self.stat_grid.addWidget(cell, i // 2, i % 2)

    def _set_download_controls(self, enabled: bool) -> None:
        self.radio_p1.setEnabled(enabled)
        self.radio_all.setEnabled(enabled)
        self.btn_dl.setEnabled(enabled)
        self.btn_dummy.setEnabled(enabled)

    def _arm_download_bars(self) -> None:
        wanted = "p1" if self._download_set() == "p1" else "all"
        for entry in self._manifest["files"]:
            if wanted not in entry.get("sets", []):
                continue
            cell = self._stat_cells.get(entry["id"])
            if cell is None:
                continue
            if cell._idle_state == "READY":
                continue
            cell.show_queued()

    def _handle_progress_line(self, line: str) -> None:
        parts = line.split("\t")
        if len(parts) < 3:
            return
        if parts[1] == "job":
            try:
                current = int(parts[2])
                total = int(parts[3]) if len(parts) > 3 else 0
                pct = int(parts[4]) if len(parts) > 4 else (-1 if total <= 0 else 0)
            except ValueError:
                return
            msg = parts[5] if len(parts) > 5 else ""
            self.job_strip.set_progress(current, total, pct, msg)
            if self._job_label in ("wan", "hires", "worldstereo") and hasattr(self, "gen_page"):
                self.gen_page.set_job_progress(current, total, pct, msg)
            return
        file_id = parts[1]
        cell = self._stat_cells.get(file_id)
        kind = parts[2]
        if kind == "start":
            total = int(parts[3]) if len(parts) > 3 else 0
            if cell is not None:
                cell.show_progress(0, total, 0 if total else -1)
            self.job_strip.set_progress(0, total, 0 if total else -1, f"download {file_id}")
            return
        if kind == "done":
            if cell is not None:
                cell.show_done(ready=True)
            self.job_strip.set_progress(1, 1, 100, f"download {file_id}")
            return
        if kind == "fail":
            if cell is not None:
                cell.show_done(ready=False)
            self.job_strip.set_caption(f"download {file_id} failed")
            return
        if len(parts) < 5:
            return
        try:
            n = int(parts[2])
            total = int(parts[3])
            pct = int(parts[4])
        except ValueError:
            return
        if cell is not None:
            cell.show_progress(n, total, pct)
        self.job_strip.set_progress(n, total, pct, f"download {file_id}")

    def _handle_worker_line(self, line: str) -> None:
        raw = line.rstrip("\r")
        if not raw:
            return
        if raw.startswith("PROGRESS\t"):
            self._handle_progress_line(raw)
            return
        if raw.startswith("FLORENCE_HUB\t"):
            self._florence_hub = raw.split("\t", 1)[1].strip()
            self._append_log(raw)
            return
        if raw.startswith("FLORENCE_REV\t"):
            self._florence_rev = raw.split("\t", 1)[1].strip()
            self._append_log(raw)
            return
        if raw.startswith("CAPTION\t"):
            self.pano_page.apply_caption(raw.split("\t", 1)[1])
            self._write_florence_pin()
            self._append_log(raw)
            return
        if raw.startswith("HFOV\t"):
            try:
                self.pano_page.apply_hfov(float(raw.split("\t", 1)[1]))
            except ValueError:
                self._append_log(f"STATUS // bad h_fov line {raw}")
                return
            self._append_log(raw)
            return
        if raw.startswith("PANO\t"):
            self.pano_page.apply_pano(raw.split("\t", 1)[1])
            self._append_log(raw)
            return
        if raw.startswith("GEO\t"):
            parts = raw.split("\t")
            if len(parts) >= 3 and parts[1] == "json":
                self.geo_page.apply_geometry(parts[2])
            self._append_log(raw)
            return
        if raw.startswith("CAMPLOT\t"):
            parts = raw.split("\t")
            if len(parts) >= 3:
                self.geo_page.apply_camplot(parts[1], parts[2])
            self._append_log(raw)
            return
        if raw.startswith("SFM\t"):
            parts = raw.split("\t")
            kind = parts[1] if len(parts) > 1 else ""
            if kind == "num_images" and len(parts) >= 3:
                self._sfm_images = parts[2]
            elif kind == "num_points" and len(parts) >= 3:
                self._sfm_points = parts[2]
            elif kind == "sparse" and len(parts) >= 3:
                self._sfm_sparse = parts[2]
            elif kind == "preview" and len(parts) >= 3:
                self._sfm_preview = parts[2]
            imgs = int(getattr(self, "_sfm_images", "0") or 0)
            pts = int(getattr(self, "_sfm_points", "0") or 0)
            # Counts only while SphereSfM is still exiting. Viewport loads in
            # on_sfm_finished so it always reads the files this run just wrote.
            self.recon_page.apply_sfm(imgs, pts, "", getattr(self, "_sfm_preview", ""))
            self._append_log(raw)
            return
        if raw.startswith("SPLAT\t"):
            parts = raw.split("\t")
            kind = parts[1] if len(parts) > 1 else ""
            self.splat_page.on_splat_line(parts)
            if kind == "step" and len(parts) >= 4:
                try:
                    cur, total = int(parts[2]), int(parts[3])
                    self.job_strip.set_progress(cur, total, int(round(100 * cur / max(total, 1))), f"splat {cur}/{total}")
                except ValueError:
                    pass
            self._log_splat_line(parts, raw)
            return
        if raw.startswith("WAN\t") or raw.startswith("HIRES\t"):
            self._append_log(raw)
            return
        if raw.startswith("WS\t"):
            parts = raw.split("\t")
            if hasattr(self, "gen_page") and len(parts) >= 2:
                kind = parts[1]
                if kind == "view" and len(parts) >= 4:
                    self.gen_page.show_ws(int(parts[2]), "view", view=parts[3])
                elif kind == "retrieved" and len(parts) >= 4:
                    self.gen_page.show_ws(int(parts[2]), "retrieved", int(parts[3]))
                elif kind == "rail" and len(parts) >= 4:
                    self.gen_page.show_ws(int(parts[2]), parts[3])
                elif kind == "inputs":
                    self.gen_page.refresh(keep_preview=True)
            self._append_log(raw)
            return
        if raw.startswith("STATUS //") and self._job_running():
            self.job_strip.set_caption(raw)
        self._append_log(raw)

    def _write_florence_pin(self) -> None:
        """Record Hub id + revision after the first successful caption. Do not invent a fork."""
        hub = (self._florence_hub or "").strip()
        if not hub:
            return
        try:
            data = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
            data["florence"] = {
                "hub_id": hub,
                "revision": (self._florence_rev or "").strip(),
                "task": "<CAPTION>",
                "note": "Pinned after first successful engine.workers.caption. Do not invent a quantized fork.",
            }
            MANIFEST_PATH.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
            self._append_log(f"STATUS // florence pin written {hub} rev={self._florence_rev}")
        except Exception as exc:
            self._append_log(f"STATUS // florence pin write failed: {exc}")

    def _on_pano_job(self, args: list[str]) -> None:
        if args and args[0] == "__caption__":
            self._start_job(caption_argv(args[1]), "caption")
            return
        if args and args[0] == "__detect_fov__":
            self._start_job(detect_fov_argv(args[1]), "detect_fov")
            return
        self._start_job(pano_argv(args), "pano")

    def _on_geo_job(self, args: list[str]) -> None:
        if not args:
            return
        if args[0] == "__geo_error__":
            self._append_log(f"STATUS // {args[1] if len(args) > 1 else 'geometry error'}")
            return
        if args[0] == "__moge__":
            scene = args[1] if len(args) > 1 else self.scene.text()
            self._start_job(moge_argv(scene), "moge")
            return
        if args[0] == "__camplot__":
            scene = args[1] if len(args) > 1 else self.scene.text()
            rail = int(args[2]) if len(args) > 2 else 0
            if not self._start_job(camplot_argv(scene, rail), "camplot"):
                self.geo_page.on_camplot_busy()
            return
        if args[0] == "__geo_log__":
            self._append_log(args[1] if len(args) > 1 else "")
            return
        if args[0] == "__confirmed__":
            self._append_log("STATUS // Confirm rails — Generate unlocked")
            if hasattr(self, "recon_page"):
                self.recon_page.refresh()
            if hasattr(self, "gen_page"):
                self.gen_page.refresh()
            return

    def _rails_from_request(self, scene: str, req: str) -> list[int]:
        path = Path(req)
        if path.is_file():
            data = json.loads(path.read_text(encoding="utf-8"))
            saved = data.get("rails")
            if isinstance(saved, list) and saved:
                return [int(i) for i in saved]
        from engine.geo.rails import kept_rail_indices

        kept = kept_rail_indices(scene)
        return kept or [0]

    def _on_gen_job(self, args: list[str]) -> None:
        if not args:
            return
        if args[0] == "__gen_error__":
            self._append_log(f"STATUS // {args[1] if len(args) > 1 else 'generate error'}")
            return
        if args[0] == "__ws_hires__":
            if self._job_running():
                self._append_log("STATUS // busy — Stop the current job first")
                return
            scene = args[1] if len(args) > 1 else self.scene.text()
            source = args[2] if len(args) > 2 and args[2] in ("world", "frame") else "world"
            kind = "WorldStereo upscale" if source == "frame" else "panorama composite"
            self._append_log(f"STATUS // HiRes all for {scene} — {kind}")
            if hasattr(self, "gen_page"):
                rails = getattr(self.gen_page, "_ws_order", None) or []
                self.gen_page.on_generate_start(rails, kind="pinhole_hires")
            self._start_job(pinhole_hires_argv(scene, source), "pinhole_hires")
            return
        if args[0] == "__worldstereo__":
            if self._job_running():
                self._append_log("STATUS // busy — Stop the current job first")
                return
            scene = args[1] if len(args) > 1 else self.scene.text()
            req = args[2] if len(args) > 2 else ""
            rails = [int(part) for part in (args[3] if len(args) > 3 else "").split(",") if part.strip()]
            run_mode = args[4] if len(args) > 4 else "resume"
            seed = int(args[5]) if len(args) > 5 and str(args[5]).strip() else None
            regen_text = args[6] if len(args) > 6 else ""
            regen = int(regen_text) if str(regen_text).strip() else None
            gen_mode = args[7].strip() if len(args) > 7 and str(args[7]).strip() else "joint"
            self._append_log(
                f"STATUS // WorldStereo request for {scene} rails={rails} mode={run_mode} generation={gen_mode}"
            )
            if hasattr(self, "gen_page"):
                self.gen_page.on_generate_start(rails, kind="worldstereo")
            self._start_job(
                worldstereo_argv(
                    scene,
                    mode=gen_mode,
                    resume=run_mode != "restart",
                    regen_rail=regen,
                    request=req or None,
                    rails=rails or None,
                    seed=seed,
                ),
                "worldstereo",
            )
            return

    def _on_recon_job(self, args: list[str]) -> None:
        if not args:
            return
        if args[0] == "__dataset__":
            if hasattr(self, "gen_page"):
                self.gen_page.refresh(keep_preview=True)
            if hasattr(self, "recon_page"):
                self.recon_page.refresh()
            return
        if args[0] == "__known_pose__":
            scene = args[1] if len(args) > 1 else self.scene.text()
            self._sfm_images = "0"
            self._sfm_points = "0"
            self._sfm_sparse = ""
            self._sfm_preview = ""
            self._start_job(known_pose_argv(scene), "known_pose")

    # 10-step telemetry (step / loss / psnr / its / vram / fps / gaussians) stays in the
    # HUD; the log drawer gets controller events, checkpoints, exports, errors, state.
    _SPLAT_LOG_KINDS = ("ctl", "ckpt", "ply", "error", "state", "strategy", "budget", "shm")

    def _log_splat_line(self, parts: list[str], raw: str) -> None:
        kind = parts[1] if len(parts) > 1 else ""
        if kind == "cameras":
            try:
                payload = "\t".join(parts[2:]) if len(parts) > 2 else "[]"
                n = len(json.loads(payload))
            except ValueError:
                n = 0
            self._append_log(f"SPLAT\tcameras\t{n} training views")
        elif kind in self._SPLAT_LOG_KINDS:
            self._append_log(raw)

    def _on_splat_control(self, line: str) -> None:
        """Splat page -> active splat worker stdin (VIEW / PAUSE / RESUME)."""
        proc = None
        if self._job_running() and self._job_label == "splat":
            proc = self._proc
        elif self._splat_view_running():
            proc = self._splat_view_proc
        if proc is None:
            return
        try:
            proc.write((line.rstrip("\n") + "\n").encode("utf-8"))
        except Exception as exc:
            self._append_log(f"STATUS // splat control write failed: {exc}")

    def _splat_view_running(self) -> bool:
        return (
            self._splat_view_proc is not None
            and self._splat_view_proc.state() != QProcess.ProcessState.NotRunning
        )

    def _stop_splat_view(self) -> None:
        proc = self._splat_view_proc
        if proc is None:
            return
        self._splat_view_stopping = True
        self._splat_view_proc = None
        self._splat_view_buf = ""
        if hasattr(self, "splat_page"):
            self.splat_page.on_worker_finished()
        if proc.state() != QProcess.ProcessState.NotRunning:
            try:
                proc.write(b"STOP\n")  # lets the worker unlink its shared memory and persist the camera
            except Exception:
                pass
            if not proc.waitForFinished(1500):
                proc.kill()
                proc.waitForFinished(4000)
        self._splat_view_stopping = False

    def _ensure_splat_view(self) -> None:
        if not self._want_splat_view or self._job_running() or self._splat_view_running():
            return
        if not litegs_spike_ok():
            return
        scene = self.scene.text()
        ckpt = scene_dir(scene) / "checkpoints" / "splat_latest.pt"
        if not ckpt.is_file():
            return
        strategy = self.splat_page.strategy_key() if hasattr(self, "splat_page") else "litegs"
        argv = splat_argv(scene, strategy, "view")
        proc = QProcess(self)
        proc.setWorkingDirectory(str(REPO_ROOT))
        proc.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        env = QProcessEnvironment.systemEnvironment()
        for key, val in worker_env(int(self.cuda.value())).items():
            env.insert(key, val)
        proc.setProcessEnvironment(env)
        proc.readyReadStandardOutput.connect(self._on_splat_view_out)
        proc.finished.connect(self._on_splat_view_finished)
        self._splat_view_proc = proc
        self._splat_view_buf = ""
        self.splat_page.on_worker_started("view")
        proc.start(argv[0], argv[1:])

    def _on_splat_view_out(self) -> None:
        if self._splat_view_proc is None:
            return
        data = bytes(self._splat_view_proc.readAllStandardOutput()).decode(
            "utf-8", errors="replace"
        )
        self._splat_view_buf += data.replace("\r\n", "\n").replace("\r", "\n")
        while "\n" in self._splat_view_buf:
            line, self._splat_view_buf = self._splat_view_buf.split("\n", 1)
            raw = line.strip()
            if not raw:
                continue
            if raw.startswith("SPLAT\t"):
                parts = raw.split("\t")
                self.splat_page.on_splat_line(parts)
                if parts[1:2] != ["fps"]:
                    self._log_splat_line(parts, raw)
            elif raw.startswith("STATUS //") and (
                "view" in raw or "Retrain" in raw or "failed" in raw or "skipped" in raw or "reload" in raw
            ):
                self._append_log(raw)

    def _on_splat_view_finished(self, _code: int, _status: QProcess.ExitStatus) -> None:
        self._splat_view_proc = None
        self._splat_view_buf = ""
        if hasattr(self, "splat_page"):
            self.splat_page.on_worker_finished()
        if self._splat_view_stopping:
            return
        if self._want_splat_view and not self._job_running():
            QTimer.singleShot(500, self._ensure_splat_view)

    def _on_splat_job(self, args: list[str]) -> None:
        """["__splat__", scene, strategy_key, mode, steps, budget, res_scale, holdout, color_space]"""
        if not args or args[0] != "__splat__":
            return
        scene = args[1] if len(args) > 1 else self.scene.text()
        strategy = args[2] if len(args) > 2 else "litegs"
        mode = args[3] if len(args) > 3 else "train"

        def _int(i: int):
            try:
                return int(args[i]) if len(args) > i else None
            except ValueError:
                return None

        max_steps = _int(4)
        budget = _int(5)
        try:
            res_scale = float(args[6]) if len(args) > 6 else None
        except ValueError:
            res_scale = None
        holdout = len(args) > 7 and args[7] == "1"
        color_space = args[8] if len(args) > 8 and args[8] else None
        argv = splat_argv(
            scene,
            strategy,
            mode,
            max_steps=max_steps,
            budget=budget,
            res_scale=res_scale,
            holdout=holdout,
            color_space=color_space,
        )
        if self._start_job(argv, "splat"):
            self.splat_page.on_worker_started("train")

    def _show_stage(self, name: str) -> None:
        if name not in STAGES:
            return
        index = STAGES.index(name)
        if index < len(self.stage_btns):
            self.stage_btns[index].setChecked(True)
        self._on_stage(index)

    def _on_scene(self, _text: str) -> None:
        if hasattr(self, "pano_page"):
            self.pano_page.restore_view()
        if hasattr(self, "geo_page"):
            self.geo_page.try_load_existing()
        if hasattr(self, "gen_page"):
            self.gen_page.refresh()
        if hasattr(self, "recon_page"):
            self.recon_page.refresh()
        if hasattr(self, "splat_page"):
            self.splat_page.refresh()
        self._stop_splat_view()
        self._ensure_splat_view()

    def _job_running(self) -> bool:
        return self._proc is not None and self._proc.state() != QProcess.ProcessState.NotRunning

    def _start_download(self) -> None:
        self._start_job(download_argv(self._download_set()), "download")

    def _start_dummy(self) -> None:
        self._start_job(dummy_argv(), "dummy")

    def _start_litegs_build(self) -> None:
        self._start_job(litegs_build_argv(), "litegs")

    def _refresh_litegs_hint(self) -> None:
        if not hasattr(self, "litegs_hint"):
            return
        from engine.splat.status import litegs_spike_detail

        if litegs_spike_ok():
            self.litegs_hint.setText("LiteGS spike OK — Splat Train enabled. " + litegs_spike_detail())
        else:
            self.litegs_hint.setText(
                "Splat Train is off until this passes. Builds simple-knn, fused_ssim and "
                "litegs_fused with MSVC + CUDA 12.8 (2 to 5 min). " + litegs_spike_detail()
            )

    def _start_job(self, argv: list[str], label: str) -> bool:
        if self._job_running():
            self._append_log("STATUS // busy — Stop the current job first")
            return False
        self._stop_splat_view()
        self._out_buf = ""
        self._refresh_status()
        self._downloading = label == "download"
        if self._downloading:
            self._arm_download_bars()
            self._set_download_controls(False)
        proc = QProcess(self)
        proc.setWorkingDirectory(str(REPO_ROOT))
        proc.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        env = QProcessEnvironment.systemEnvironment()
        for key, val in worker_env(int(self.cuda.value())).items():
            env.insert(key, val)
        proc.setProcessEnvironment(env)
        proc.readyReadStandardOutput.connect(self._on_proc_out)
        proc.finished.connect(self._on_proc_finished)
        proc.errorOccurred.connect(self._on_proc_error)
        self._proc = proc
        self._job_label = label
        program, args = argv[0], argv[1:]
        self._set_live(label)
        self._expect_kill = False
        self.job_strip.set_busy(label, f"{label} running")
        self._append_log(f"STATUS // start {label}: {program} {' '.join(args)}")
        proc.start(program, args)
        return True

    def _on_proc_out(self) -> None:
        if self._proc is None:
            return
        data = bytes(self._proc.readAllStandardOutput()).decode("utf-8", errors="replace")
        self._out_buf += data.replace("\r\n", "\n").replace("\r", "\n")
        while "\n" in self._out_buf:
            line, self._out_buf = self._out_buf.split("\n", 1)
            self._handle_worker_line(line)

    def _on_proc_error(self, error: QProcess.ProcessError) -> None:
        if self._expect_kill:
            return
        self._append_log(f"STATUS // stopped — process error {error.name}")

    def _on_proc_finished(self, code: int, status: QProcess.ExitStatus) -> None:
        if self._out_buf:
            self._handle_worker_line(self._out_buf)
            self._out_buf = ""
        if self._expect_kill:
            self._append_log("STATUS // cancelled")
        elif status == QProcess.ExitStatus.CrashExit:
            self._append_log("STATUS // stopped — crashed")
        elif code != 0:
            self._append_log(f"STATUS // stopped — exit {code}")
        else:
            self._append_log("STATUS // exit 0")
        label = self._job_label
        cancelled = self._expect_kill
        self._expect_kill = False
        self._downloading = False
        self._job_label = ""
        self._set_download_controls(True)
        self._set_live("idle")
        self.job_strip.set_idle()
        self._refresh_status()
        self._proc = None
        if hasattr(self, "gen_page"):
            self.gen_page.on_job_finished(label, code, more=False)
        if hasattr(self, "recon_page") and label in ("worldstereo", "pinhole_hires"):
            self.recon_page.refresh()
        if hasattr(self, "geo_page"):
            self.geo_page.on_job_finished(label, code)
        if label in ("sfm", "known_pose") and hasattr(self, "recon_page"):
            self.recon_page.on_sfm_finished(code)
        if label == "litegs":
            self._refresh_litegs_hint()
        if hasattr(self, "splat_page"):
            if label == "splat":
                self.splat_page.on_worker_finished()
            self.splat_page.refresh()
        self._ensure_splat_view()

    def _stop_job(self) -> None:
        if not self._job_running() or self._proc is None:
            self._append_log("STATUS // idle (nothing to stop)")
            return
        self._expect_kill = True
        self._append_log("STATUS // cancel — killing subprocess")
        self._proc.kill()

    def closeEvent(self, event: QCloseEvent) -> None:
        self._want_splat_view = False
        self._stop_splat_view()
        if self._job_running() and self._proc is not None:
            self._proc.kill()
            self._proc.waitForFinished(2000)
        super().closeEvent(event)
