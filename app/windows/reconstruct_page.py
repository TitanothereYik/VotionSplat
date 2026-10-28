"""Phase 2/3 Reconstruct page — keep or ditch rails, then SphereSfM."""

from __future__ import annotations

import json

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from app.widgets.sparse_view import SparseView
from engine.geo.rails import (
    confirmed_path,
    set_dataset_rail,
    sync_dataset_from_confirmed,
    wan_video_rel,
)
from engine.pano.scene_io import scene_dir


def _lab(text: str) -> QLabel:
    lab = QLabel(text)
    lab.setProperty("class", "fLab")
    return lab


def _hint(text: str) -> QLabel:
    lab = QLabel(text)
    lab.setWordWrap(True)
    lab.setProperty("class", "fHint")
    return lab


class ReconstructPage(QWidget):
    requestJob = Signal(list)
    requestStop = Signal()

    def __init__(self, scene_name_fn, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._scene_name = scene_name_fn
        self._row_status: dict[int, QLabel] = {}
        self._sfm_busy = False

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

        r.addWidget(_lab("Rails in dataset"))
        r.addWidget(
            _hint(
                "Ditch a rail if WAN invented something you do not want. "
                "Re-gen WAN after editing the prompt on Generate, or after a new Play on Geometry."
            )
        )
        self.rail_host = QWidget()
        self.rail_lay = QVBoxLayout(self.rail_host)
        self.rail_lay.setContentsMargins(0, 0, 0, 0)
        self.rail_lay.setSpacing(6)
        r.addWidget(self.rail_host)

        r.addWidget(_lab("SphereSfM"))
        self.btn_recon = QPushButton("Reconstruct")
        self.btn_recon.setObjectName("btnPrimary")
        self.btn_recon.setEnabled(False)
        self.btn_recon.clicked.connect(self._on_reconstruct)
        r.addWidget(self.btn_recon)

        grid_host = QFrame()
        grid_host.setProperty("class", "statCell")
        grid = QGridLayout(grid_host)
        grid.setContentsMargins(8, 8, 8, 8)
        grid.setSpacing(6)
        self.num_images = QLabel("—")
        self.num_images.setProperty("class", "statVal")
        self.num_points = QLabel("—")
        self.num_points.setProperty("class", "statVal")
        n1 = QLabel("num_images")
        n1.setProperty("class", "statName")
        n2 = QLabel("num_points")
        n2.setProperty("class", "statName")
        grid.addWidget(n1, 0, 0)
        grid.addWidget(self.num_images, 1, 0)
        grid.addWidget(n2, 0, 1)
        grid.addWidget(self.num_points, 1, 1)
        r.addWidget(grid_host)

        self.gate_hint = _hint("")
        r.addWidget(self.gate_hint)
        r.addStretch(1)
        return rail

    def _build_view(self) -> QWidget:
        view = QWidget()
        view.setObjectName("viewport")
        v = QVBoxLayout(view)
        v.setContentsMargins(8, 8, 8, 8)
        hdr = QHBoxLayout()
        lbl = QLabel("COLMAP sparse")
        lbl.setProperty("class", "viewLbl")
        self.btn_reset = QPushButton("reset view")
        self.btn_reset.setProperty("class", "rail-mini")
        self.sparse = SparseView()
        self.btn_reset.clicked.connect(self.sparse.reset_view)
        hdr.addWidget(lbl)
        hdr.addStretch(1)
        hdr.addWidget(self.btn_reset)
        v.addLayout(hdr)
        v.addWidget(self.sparse, 1)
        return view

    def refresh(self, *, load_sparse: bool = True) -> None:
        while self.rail_lay.count():
            item = self.rail_lay.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self._row_status.clear()
        scene = self._scene_name()
        root = scene_dir(scene)
        gate = confirmed_path(scene)
        confirmed = gate.is_file()
        data = sync_dataset_from_confirmed(scene) if confirmed else {"rails": []}
        rows = data.get("rails") or []
        kept_ready = 0
        for row in rows:
            idx = int(row.get("index", 0))
            keep = bool(row.get("keep", True))
            wan = root / str(row.get("wan") or wan_video_rel(idx))
            has_wan = wan.is_file() and wan.stat().st_size > 0
            if keep and has_wan:
                kept_ready += 1
            block = QWidget()
            col = QVBoxLayout(block)
            col.setContentsMargins(0, 0, 0, 0)
            col.setSpacing(2)
            top = QHBoxLayout()
            name = QLabel(f"Rail {idx + 1}")
            name.setProperty("class", "fLab")
            state = QLabel("ditched" if not keep else ("WAN" if has_wan else "no WAN"))
            state.setProperty("class", "fHint")
            state.setAlignment(Qt.AlignmentFlag.AlignRight)
            self._row_status[idx] = state
            top.addWidget(name)
            top.addWidget(state, 1)
            col.addLayout(top)
            btns = QHBoxLayout()
            ditch = QPushButton("Keep" if keep else "Ditched")
            ditch.setCheckable(True)
            ditch.setChecked(keep)
            ditch.setProperty("class", "modeChip")
            ditch.clicked.connect(lambda on, rail=idx: self._on_keep(rail, bool(on)))
            regen = QPushButton("Re-gen WAN")
            regen.setProperty("class", "rail-mini")
            regen.clicked.connect(lambda _=False, rail=idx: self._on_regen(rail))
            btns.addWidget(ditch)
            btns.addWidget(regen)
            col.addLayout(btns)
            self.rail_lay.addWidget(block)

        self.btn_recon.setEnabled(confirmed and kept_ready > 0)
        if not confirmed:
            self.gate_hint.setText("Play each rail on Geometry, then Confirm rails.")
        elif kept_ready == 0:
            self.gate_hint.setText("Keep at least one rail with a WAN clip, or Generate WAN first.")
        else:
            self.gate_hint.setText(
                f"{kept_ready} kept rail(s) with WAN. Reconstruct uses those clips. Dual-res HiRes frames if present."
            )
        if self._sfm_busy or not load_sparse:
            return
        self.show_latest_sparse(announce=False)

    def apply_sfm(
        self,
        num_images: int,
        num_points: int,
        sparse: str,
        preview: str,  # still-image path from older workers; ignored
        *,
        force: bool = True,
    ) -> None:
        """Counts from the worker log. Sparse itself loads only after the process exits."""
        if num_images:
            self.num_images.setText(str(num_images))
        if num_points:
            self.num_points.setText(str(num_points))
        if sparse and not self._sfm_busy:
            self.sparse.load(sparse, force=force)

    def show_latest_sparse(self, *, announce: bool = True) -> None:
        """Read colmap/sparse/0 and sfm.json from disk. Always the current solve."""
        root = scene_dir(self._scene_name())
        meta_path = root / "colmap" / "sfm.json"
        sparse = root / "colmap" / "sparse" / "0"
        n_img, n_pts, n_rails = 0, 0, 0
        if meta_path.is_file():
            data = json.loads(meta_path.read_text(encoding="utf-8"))
            n_img = int(data.get("num_images") or 0)
            n_pts = int(data.get("num_points") or 0)
            n_rails = int(data.get("num_rails") or 0)
        self.num_images.setText(str(n_img) if n_img else "—")
        self.num_points.setText(str(n_pts) if n_pts else "—")
        if not (sparse / "points3D.bin").is_file():
            self.sparse.clear("Empty until Reconstruct.")
            return
        self.sparse.clear("Loading newest sparse…")
        self.sparse.load(str(sparse), force=True)
        if announce:
            rails_bit = f"{n_rails} kept rail(s). " if n_rails else ""
            self.gate_hint.setText(
                f"{rails_bit}Showing current sparse ({n_img} images, {n_pts} points)."
            )

    def begin_reconstruct(self) -> None:
        self._sfm_busy = True
        self.num_images.setText("—")
        self.num_points.setText("—")
        self.sparse.clear("Reconstructing…")
        self.gate_hint.setText("SphereSfM running. Viewport updates when Reconstruct finishes.")

    def on_sfm_finished(self, code: int) -> None:
        self._sfm_busy = False
        self.refresh(load_sparse=False)
        if code != 0:
            self.gate_hint.setText("Reconstruct failed. Check Log, then Reconstruct again.")
            root = scene_dir(self._scene_name())
            if not (root / "colmap" / "sparse" / "0" / "points3D.bin").is_file():
                self.num_images.setText("—")
                self.num_points.setText("—")
                self.sparse.clear("Reconstruct failed.")
            return
        self.show_latest_sparse(announce=True)

    def _on_keep(self, rail_index: int, keep: bool) -> None:
        set_dataset_rail(self._scene_name(), rail_index, keep=bool(keep))
        self.requestJob.emit(["__dataset__"])
        self.refresh()

    def _on_regen(self, rail_index: int) -> None:
        set_dataset_rail(self._scene_name(), rail_index, keep=True)
        self.requestJob.emit(["__wan_rails__", self._scene_name(), str(int(rail_index))])

    def _on_reconstruct(self) -> None:
        if not self.btn_recon.isEnabled():
            return
        self.begin_reconstruct()
        self.requestJob.emit(["__sfm__", self._scene_name()])
