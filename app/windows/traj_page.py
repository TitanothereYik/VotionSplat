"""Trajectories page: plan the paths, then watch each step appear."""

from __future__ import annotations

import json

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QCheckBox, QComboBox, QHBoxLayout, QListWidgetItem, QSpinBox, QStackedWidget

from app.widgets.clip_view import ClipView
from app.widgets.traj_map import TrajMapView
from app.windows.hy_common import (
    HyPage,
    button,
    chips,
    family_color,
    hint,
    lab,
    rail,
    side_list,
    view_label,
    viewport,
)
from engine.hyworld.paths import (
    apply_budget,
    camera_centers,
    close_selection,
    hy_root,
    list_trajectories,
    ordered_ids,
    read_selection,
    render_root,
    traj_state,
    write_selection,
)
from engine.pano.scene_io import pano_path

_EMPTY = "Plan trajectories to see this step."


class TrajPage(HyPage):
    LABELS = ("hy_plan", "hy_render")

    def __init__(self, scene_fn) -> None:
        super().__init__(scene_fn)
        self._ids: list[str] = []
        self._paths: dict = {}
        self._user_chip = False
        self._setting_chip = False
        self._cam_count = -1
        body = QHBoxLayout(self)
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)
        body.addWidget(self._build_rail())
        body.addWidget(self._build_view(), 1)
        body.addWidget(self._build_list())
        self._timer = QTimer(self)
        self._timer.setInterval(1000)
        self._timer.timeout.connect(self._poll)
        self.refresh()

    def _build_rail(self):
        box, lay = rail()
        lay.addWidget(lab("Scene type"))
        self.scene_type = QComboBox()
        self.scene_type.addItem("Auto", "")
        self.scene_type.addItem("Outdoor", "outdoor")
        self.scene_type.addItem("Indoor", "indoor")
        lay.addWidget(self.scene_type)
        self.nav = QCheckBox("Navigation paths")
        self.up = QCheckBox("Upward routes")
        self.recon = QCheckBox("Reconstruction loops")
        for box_ in (self.nav, self.up, self.recon):
            box_.setChecked(True)
            lay.addWidget(box_)
        self.fresh = QCheckBox("Start over (archive the old plan)")
        lay.addWidget(self.fresh)
        lay.addWidget(lab("Extra views per kind"))
        self.budget = QSpinBox()
        self.budget.setRange(0, 20)
        self.budget.setValue(2)
        self.budget.valueChanged.connect(self._on_budget)
        lay.addWidget(self.budget)
        lay.addWidget(hint("Origin views are always kept. A kept wonder, target or reconstruct view keeps all of its paths."))
        self.btn_plan = button("Plan trajectories", primary=True)
        self.btn_render = button("Render guides")
        self.btn_stop = button("Stop")
        self.btn_plan.clicked.connect(self._plan)
        self.btn_render.clicked.connect(self._render)
        self.btn_stop.clicked.connect(self.requestStop.emit)
        lay.addWidget(self.btn_plan)
        lay.addWidget(self.btn_render)
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
        head.addWidget(view_label("Trajectories"))
        chip_box, self.chips = chips(["Panorama", "Sky mask", "Nav map", "Objects", "Paths", "Guide"])
        self.chips.idClicked.connect(self._on_chip)
        head.addWidget(chip_box, 1)
        lay.addLayout(head)
        self.stack = QStackedWidget()
        self.still = ClipView(empty=_EMPTY)
        self.stack.addWidget(self.still)
        self.map = TrajMapView()
        self.map.picked.connect(self._on_map_pick)
        self.stack.addWidget(self.map)
        guide = QHBoxLayout()
        guide.setContentsMargins(0, 0, 0, 0)
        self.clip_start = ClipView("Start frame", "Select a path.")
        self.clip_guide = ClipView("Guide", "Select a path.")
        self.clip_mask = ClipView("Mask", "Select a path.")
        for clip in (self.clip_start, self.clip_guide, self.clip_mask):
            guide.addWidget(clip, 1)
        guide_box_holder = viewport()[0]
        guide_box_holder.layout().addLayout(guide)
        self.stack.addWidget(guide_box_holder)
        lay.addWidget(self.stack, 1)
        self.caption = hint("")
        lay.addWidget(self.caption)
        return box

    def _build_list(self):
        box, lay = rail(230)
        lay.addWidget(lab("Paths"))
        self.rows = side_list()
        self.rows.itemChanged.connect(self._on_item)
        self.rows.currentItemChanged.connect(self._on_current)
        lay.addWidget(self.rows, 1)
        self.count = hint("")
        lay.addWidget(self.count)
        return box

    def _selected(self) -> str:
        item = self.rows.currentItem()
        if item is None:
            return ""
        return str(item.data(Qt.ItemDataRole.UserRole) or "")

    def _checked(self) -> list[str]:
        kept = []
        for index in range(self.rows.count()):
            item = self.rows.item(index)
            if item.checkState() == Qt.CheckState.Checked:
                kept.append(str(item.data(Qt.ItemDataRole.UserRole) or ""))
        return [item for item in kept if item]

    def _select_id(self, ident: str) -> None:
        for index in range(self.rows.count()):
            item = self.rows.item(index)
            if item.data(Qt.ItemDataRole.UserRole) == ident:
                self.rows.setCurrentItem(item)
                return

    def refresh(self) -> None:
        scene = self._scene()
        ids = ordered_ids(list_trajectories(scene))
        self._ids = ids
        saved = read_selection(scene)
        keep = set(saved) if saved is not None else set(apply_budget(ids, int(self.budget.value())))
        previous = self._selected()
        self.rows.blockSignals(True)
        self.rows.clear()
        for ident in ids:
            item = QListWidgetItem(f"{ident}   {traj_state(scene, ident)}")
            item.setData(Qt.ItemDataRole.UserRole, ident)
            item.setFlags(
                Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
            )
            item.setCheckState(Qt.CheckState.Checked if ident in keep else Qt.CheckState.Unchecked)
            item.setForeground(QColor(family_color(ident)))
            self.rows.addItem(item)
        self.rows.blockSignals(False)
        paths = {}
        for ident in ids:
            try:
                paths[ident] = camera_centers(scene, ident)
            except (OSError, ValueError, KeyError):
                continue
        self._paths = paths
        if previous:
            self._select_id(previous)
        elif ids:
            self.rows.setCurrentRow(0)
        self.map.set_paths(paths, self._checked(), self._selected())
        ready = pano_path(scene).is_file()
        self.btn_plan.setEnabled(ready and not self._busy)
        self.btn_render.setEnabled(ready and bool(ids) and not self._busy)
        self.btn_stop.setEnabled(self._busy)
        self.count.setText(f"{len(ids)} paths, {len(self._checked())} kept")
        if not ready:
            self.hint_lab.setText("The panorama is missing. Finish Panorama first.")
        self._cam_count = len(ids)
        self._show_chip()

    def _on_budget(self, value: int) -> None:
        keep = set(apply_budget(self._ids, int(value)))
        self.rows.blockSignals(True)
        for index in range(self.rows.count()):
            item = self.rows.item(index)
            ident = str(item.data(Qt.ItemDataRole.UserRole) or "")
            item.setCheckState(Qt.CheckState.Checked if ident in keep else Qt.CheckState.Unchecked)
        self.rows.blockSignals(False)
        self.map.set_paths(self._paths, self._checked(), self._selected())
        self.count.setText(f"{len(self._ids)} paths, {len(self._checked())} kept")

    def _on_item(self, _item) -> None:
        self.map.set_paths(self._paths, self._checked(), self._selected())
        self.count.setText(f"{len(self._ids)} paths, {len(self._checked())} kept")

    def _on_current(self, _cur, _prev) -> None:
        self.map.set_selected(self._selected())
        if self.chips.checkedId() == 5:
            self._show_chip()

    def _on_map_pick(self, ident: str) -> None:
        self._select_id(ident)

    def _activate(self, index: int) -> None:
        if self.chips.checkedId() == index:
            return
        button_ = self.chips.button(index)
        if button_ is None:
            return
        self._setting_chip = True
        button_.setChecked(True)
        self._setting_chip = False

    def _on_chip(self, _index: int) -> None:
        if not self._setting_chip:
            self._user_chip = True
        self._show_chip()

    def _show_chip(self) -> None:
        scene = self._scene()
        root = hy_root(scene)
        index = self.chips.checkedId()
        if index < 0:
            index = 0
        if index <= 3:
            self.stack.setCurrentIndex(0)
            files = {
                0: root / "panorama.png",
                1: root / "render_results" / "sky_mask.png",
                2: root / "navmesh" / "exploration" / "map.png",
                3: root / "render_results" / "segmentation_vis.png",
            }
            self.still.set_empty(_EMPTY if index != 3 else "No objects found, or the vision model failed.")
            path = files[index]
            self.still.set_path(path if path.is_file() else None)
            self.caption.setText("")
            return
        if index == 4:
            self.stack.setCurrentIndex(1)
            self.caption.setText("")
            return
        self.stack.setCurrentIndex(2)
        ident = self._selected()
        view = ident.split("/", 1)[0] if ident else ""
        folder = render_root(scene) / ident if ident else None
        start = render_root(scene) / view / "start_frame.png" if view else None
        self.clip_start.set_path(start if start is not None and start.is_file() else None)
        self.clip_guide.set_path(folder / "render.mp4" if folder is not None else None)
        self.clip_mask.set_path(folder / "render_mask.mp4" if folder is not None else None)
        text = ""
        caption = folder / "traj_caption.json" if folder is not None else None
        if caption is not None and caption.is_file():
            try:
                text = str(json.loads(caption.read_text(encoding="utf-8")).get("prompt") or "")
            except (OSError, json.JSONDecodeError, UnicodeError):
                text = ""
        self.caption.setText(text)

    def _plan(self) -> None:
        scene = self._scene()
        flags = ["--apply_nav_traj"] if self.nav.isChecked() else []
        if self.up.isChecked():
            flags.append("--apply_up_route")
        if self.recon.isChecked():
            flags.append("--apply_recon_iteration")
        flags.append("--force_vlm")
        flags.append("--skip_exist")
        fresh = "1" if self.fresh.isChecked() else "0"
        self.requestJob.emit(["__hy_plan__", scene, self.scene_type.currentData() or "", fresh, *flags])

    def _render(self) -> None:
        scene = self._scene()
        keep = close_selection(self._checked(), self._ids)
        if not keep:
            self.hint_lab.setText("Keep at least one path.")
            return
        write_selection(scene, keep)
        self.requestJob.emit(["__hy_render__", scene])

    def on_job_started(self, label: str) -> None:
        super().on_job_started(label)
        if label in self.LABELS:
            self._user_chip = False
            self._timer.start()

    def on_job_finished(self, label: str, code: int) -> None:
        self._timer.stop()
        self.fresh.setChecked(False)
        super().on_job_finished(label, code)
        if label == "hy_plan" and code == 0:
            self._activate(4)
            self._show_chip()

    def on_progress(self, label: str, current: int, total: int, pct: int, msg: str) -> None:
        if label in self.LABELS:
            self.progress.setText(f"{pct}%  {current}/{total}  {msg}")

    def on_hy_line(self, parts: list[str]) -> None:
        if len(parts) > 2 and parts[1] == "warn":
            self.hint_lab.setText(parts[2] + " Pick Outdoor or Indoor to plan without the vision model.")

    def _poll(self) -> None:
        scene = self._scene()
        root = hy_root(scene)
        if not self._user_chip and self._label == "hy_plan":
            files = [
                root / "panorama.png",
                root / "render_results" / "sky_mask.png",
                root / "navmesh" / "exploration" / "map.png",
                root / "render_results" / "segmentation_vis.png",
            ]
            last = 0
            for index, path in enumerate(files):
                if path.is_file():
                    last = index
            count = len(list_trajectories(scene))
            if count:
                last = 4
            if count != self._cam_count:
                self.refresh()
            self._activate(last)
        elif not self._user_chip and self._label == "hy_render":
            newest = ""
            newest_t = -1.0
            for index in range(self.rows.count()):
                ident = str(self.rows.item(index).data(Qt.ItemDataRole.UserRole) or "")
                video = render_root(scene) / ident / "render.mp4"
                if video.is_file():
                    stamp = video.stat().st_mtime
                    if stamp >= newest_t:
                        newest, newest_t = ident, stamp
            if newest:
                self._select_id(newest)
                self._activate(5)
        self._show_chip()
