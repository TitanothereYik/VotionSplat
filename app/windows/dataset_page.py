"""Dataset page: build the training folder and browse image, depth and normals."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QHBoxLayout, QListWidgetItem, QSlider, QSpinBox

from app.widgets.clip_view import ClipView
from app.windows.hy_common import HyPage, button, chips, hint, lab, rail, side_list, view_label, viewport
from engine.hyworld.paths import gs_data, render_root


def depth_rgb(path: Path):
    try:
        raw = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    depth = cv2.imdecode(raw, cv2.IMREAD_UNCHANGED) if raw.size else None
    if depth is None:
        return None
    depth = depth.astype(np.float32)
    if depth.ndim == 3:
        depth = depth[..., 0]
    valid = depth > 0
    if not valid.any():
        return np.zeros((*depth.shape, 3), dtype=np.uint8)
    lo, hi = np.percentile(depth[valid], [2, 98])
    near = 1.0 - np.clip((depth - lo) / max(float(hi - lo), 1e-6), 0.0, 1.0)
    rgb = cv2.cvtColor(cv2.applyColorMap((255 * near).astype(np.uint8), cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
    rgb[~valid] = (8, 16, 24)
    return rgb


def _group(name: str) -> str:
    if name.startswith("panorama"):
        return "panorama"
    if name.startswith("polar"):
        return "polar"
    return name.rsplit("_", 1)[0]


class DatasetPage(HyPage):
    LABELS = ("hy_gsdata",)

    def __init__(self, scene_fn) -> None:
        super().__init__(scene_fn)
        self._names: list[str] = []
        self._filter = "all"
        body = QHBoxLayout(self)
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)
        body.addWidget(self._build_rail())
        body.addWidget(self._build_view(), 1)
        body.addWidget(self._build_groups())
        self._timer = QTimer(self)
        self._timer.setInterval(1000)
        self._timer.timeout.connect(self._poll)
        self.refresh()

    def _build_rail(self):
        box, lay = rail()
        lay.addWidget(lab("Panorama density"))
        self.pano_mult = QSpinBox()
        self.pano_mult.setRange(1, 4)
        self.pano_mult.setValue(2)
        lay.addWidget(self.pano_mult)
        lay.addWidget(lab("Upper polar density"))
        self.up_mult = QSpinBox()
        self.up_mult.setRange(1, 8)
        self.up_mult.setValue(4)
        lay.addWidget(self.up_mult)
        self.btn = button("Build dataset", primary=True)
        self.btn_stop = button("Stop")
        self.btn.clicked.connect(self._build)
        self.btn_stop.clicked.connect(self.requestStop.emit)
        lay.addWidget(self.btn)
        lay.addWidget(self.btn_stop)
        self.progress = hint("")
        self.hint_lab = hint("")
        lay.addWidget(self.progress)
        lay.addWidget(self.hint_lab)
        lay.addStretch(1)
        return box

    def _build_view(self):
        box, lay = viewport()
        head = QHBoxLayout()
        head.addWidget(view_label("Dataset"))
        chip_box, self.chips = chips(["Image", "Depth", "Normal"])
        self.chips.idClicked.connect(lambda _i: self._show())
        head.addWidget(chip_box, 1)
        lay.addLayout(head)
        self.view = ClipView(empty="Build the dataset to browse it.")
        lay.addWidget(self.view, 1)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 0)
        self.slider.valueChanged.connect(lambda _v: self._show())
        lay.addWidget(self.slider)
        self.name = hint("")
        lay.addWidget(self.name)
        return box

    def _build_groups(self):
        box, lay = rail(230)
        lay.addWidget(lab("Groups"))
        self.groups = side_list()
        self.groups.currentItemChanged.connect(self._on_group)
        lay.addWidget(self.groups, 1)
        return box

    def _list_names(self, scene: str) -> list[str]:
        folder = gs_data(scene) / "images"
        if not folder.is_dir():
            return []
        return sorted(path.stem for path in folder.glob("*.png"))

    def _filtered(self) -> list[str]:
        if self._filter == "all":
            return list(self._names)
        return [name for name in self._names if _group(name) == self._filter]

    def _fill_groups(self) -> None:
        counts: dict[str, int] = {}
        for name in self._names:
            key = _group(name)
            counts[key] = counts.get(key, 0) + 1
        order = ["all", "panorama", "polar"]
        order += sorted(key for key in counts if key not in ("panorama", "polar"))
        labels = {"all": f"All ({len(self._names)})", "panorama": f"panorama ({counts.get('panorama', 0)})",
                  "polar": f"polar ({counts.get('polar', 0)})"}
        current = self._filter
        self.groups.blockSignals(True)
        self.groups.clear()
        for key in order:
            if key not in ("all", "panorama", "polar") and key not in counts:
                continue
            if key in ("panorama", "polar") and counts.get(key, 0) == 0 and key != current:
                text = labels[key]
            elif key == "all":
                text = labels["all"]
            elif key in labels and key in ("panorama", "polar"):
                text = labels[key]
            else:
                text = f"{key} ({counts.get(key, 0)})"
            item = QListWidgetItem(text)
            item.setData(Qt.ItemDataRole.UserRole, key)
            self.groups.addItem(item)
            if key == current:
                self.groups.setCurrentItem(item)
        if self.groups.currentItem() is None and self.groups.count():
            self.groups.setCurrentRow(0)
            self._filter = str(self.groups.currentItem().data(Qt.ItemDataRole.UserRole))
        self.groups.blockSignals(False)

    def _show(self) -> None:
        names = self._filtered()
        if not names:
            self.view.clear()
            self.view.set_empty("Build the dataset to browse it.")
            self.name.setText("")
            return
        index = min(self.slider.value(), len(names) - 1)
        name = names[index]
        self.name.setText(f"{index + 1} / {len(names)}   {name}")
        root = gs_data(self._scene())
        kind = self.chips.checkedId()
        if kind == 1:
            self.view.set_array(depth_rgb(root / "depths" / f"{name}.png"))
        elif kind == 2:
            self.view.set_path(root / "normals" / f"{name}.png")
        else:
            self.view.set_path(root / "images" / f"{name}.png")

    def refresh(self) -> None:
        scene = self._scene()
        self._names = self._list_names(scene)
        self._fill_groups()
        shown = self._filtered()
        self.slider.blockSignals(True)
        self.slider.setRange(0, max(0, len(shown) - 1))
        self.slider.blockSignals(False)
        aligned = render_root(scene) / "generation_bank_worldstereo-memory-dmd" / "aligned_pcd.ply"
        self.btn.setEnabled(aligned.is_file() and not self._busy)
        self.btn_stop.setEnabled(self._busy)
        self.hint_lab.setText(
            f"Aligned cloud {'is' if aligned.is_file() else 'is not'} ready. Dataset images: {len(self._names)}."
        )
        self._show()

    def _on_group(self, current, _prev) -> None:
        if current is None:
            return
        self._filter = str(current.data(Qt.ItemDataRole.UserRole) or "all")
        shown = self._filtered()
        self.slider.blockSignals(True)
        self.slider.setRange(0, max(0, len(shown) - 1))
        self.slider.setValue(0)
        self.slider.blockSignals(False)
        self._show()

    def _build(self) -> None:
        self.requestJob.emit(
            [
                "__hy_gsdata__",
                self._scene(),
                str(int(self.pano_mult.value())),
                str(int(self.up_mult.value())),
            ]
        )

    def on_job_started(self, label: str) -> None:
        super().on_job_started(label)
        if label in self.LABELS:
            self._timer.start()

    def on_job_finished(self, label: str, code: int) -> None:
        self._timer.stop()
        super().on_job_finished(label, code)

    def on_progress(self, label: str, current: int, total: int, pct: int, msg: str) -> None:
        if label in self.LABELS:
            self.progress.setText(f"{pct}%  {current}/{total}  {msg}")

    def _poll(self) -> None:
        scene = self._scene()
        names = self._list_names(scene)
        if names == self._names:
            self._show()
            return
        self._names = names
        self._fill_groups()
        shown = self._filtered()
        self.slider.blockSignals(True)
        self.slider.setRange(0, max(0, len(shown) - 1))
        self.slider.setValue(max(0, len(shown) - 1))
        self.slider.blockSignals(False)
        self.hint_lab.setText(
            f"Aligned cloud is ready. Dataset images: {len(self._names)}."
        )
        self._show()
