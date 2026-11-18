"""Expand page: one WorldStereo trajectory at a time, then alignment."""

from __future__ import annotations

import json

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QComboBox,
    QGridLayout,
    QHBoxLayout,
    QListWidgetItem,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from app.widgets.clip_view import ClipView
from app.windows.hy_common import BLUE, DIM, ORANGE, RED, HyPage, button, chips, hint, lab, rail, side_list, view_label, viewport
from engine.hyworld import hires as HR
from engine.hyworld.paths import hy_root, list_trajectories, ordered_ids, read_selection, render_root, traj_state, votion_dir

_RESULT = "worldstereo-memory-dmd_result.mp4"
_MEMORY = "worldstereo-memory-dmd.mp4"
_LEVELS = (
    ("auto", "Auto (fit this machine)"),
    ("card", "Everything on the card"),
    ("blocks", "DiT blocks in RAM"),
    ("states", "Blocks and controlnet outputs in RAM"),
    ("controlnet", "Blocks, outputs and controlnet in RAM"),
)


class ExpandPage(HyPage):
    LABELS = ("hy_expand", "hy_align", "hy_hires")

    def __init__(self, scene_fn) -> None:
        super().__init__(scene_fn)
        self._ids: list[str] = []
        self._status: dict[str, str] = {}
        self._times: list[float] = []
        self._hr_left: set[str] = set()
        self._hr_times: list[float] = []
        self._hr_shown = False
        self._setting_chip = False
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
        self.btn_one = button("Profile first path")
        self.btn_all = button("Expand kept paths", primary=True)
        self.btn_again = button("Regenerate selected")
        self.btn_align = button("Align")
        self.btn_hires = button("HiRes all")
        self.btn_hires_one = button("HiRes selected")
        self.btn_stop = button("Stop")
        self.btn_one.clicked.connect(lambda: self._expand(first=True, restart=False))
        self.btn_all.clicked.connect(lambda: self._expand(first=False, restart=False))
        self.btn_again.clicked.connect(self._again)
        self.btn_align.clicked.connect(self._align)
        self.btn_hires.clicked.connect(lambda: self._hires(first=False))
        self.btn_hires_one.clicked.connect(lambda: self._hires(first=True))
        self.btn_stop.clicked.connect(self.requestStop.emit)
        for widget in (
            self.btn_one, self.btn_all, self.btn_again, self.btn_align,
            self.btn_hires, self.btn_hires_one, self.btn_stop,
        ):
            lay.addWidget(widget)
        lay.addWidget(lab("GPU memory"))
        self.level = QComboBox()
        for key, text in _LEVELS:
            self.level.addItem(text, key)
        lay.addWidget(self.level)
        self.fit_lab = hint("")
        lay.addWidget(self.fit_lab)
        lay.addWidget(lab("HiRes scale"))
        self.hires_scale = QComboBox()
        for scale in HR.SCALES:
            self.hires_scale.addItem(HR.SCALE_NAMES[scale], scale)
        self.hires_scale.currentIndexChanged.connect(lambda _i: self._scale_changed())
        lay.addWidget(self.hires_scale)
        lay.addWidget(lab("HiRes memory"))
        self.hires_level = QComboBox()
        self._fill_hires_levels()
        lay.addWidget(self.hires_level)
        self.hires_fit = hint("")
        lay.addWidget(self.hires_fit)
        self.progress = hint("")
        self.hint_lab = hint("")
        lay.addWidget(self.progress)
        lay.addWidget(self.hint_lab)
        lay.addStretch(1)
        return box

    def _build_view(self):
        box, lay = viewport()
        head = QHBoxLayout()
        head.addWidget(view_label("Expand"))
        chip_box, self.chips = chips(["Clips", "Aligned cloud", "HiRes"])
        self.chips.idClicked.connect(lambda _i: self.stack.setCurrentIndex(max(0, self.chips.checkedId())))
        head.addWidget(chip_box, 1)
        lay.addLayout(head)
        self.stack = QStackedWidget()
        grid_host = QWidget()
        grid = QGridLayout(grid_host)
        grid.setContentsMargins(0, 0, 0, 0)
        self.clip_guide = ClipView("Guide", "Render guides first.")
        self.clip_mask = ClipView("Mask", "Render guides first.")
        self.clip_memory = ClipView("Memory", "Retrieval frames appear here.")
        self.clip_result = ClipView("Result", "The generated frames appear here.")
        grid.addWidget(self.clip_guide, 0, 0)
        grid.addWidget(self.clip_mask, 0, 1)
        grid.addWidget(self.clip_memory, 1, 0)
        grid.addWidget(self.clip_result, 1, 1)
        self.stack.addWidget(grid_host)
        cloud = QWidget()
        cloud_lay = QVBoxLayout(cloud)
        cloud_lay.setContentsMargins(0, 0, 0, 0)
        self.cloud = ClipView(empty="Run Align to see the fitted cloud.")
        self.cloud_hint = hint("")
        cloud_lay.addWidget(self.cloud, 1)
        cloud_lay.addWidget(self.cloud_hint)
        self.stack.addWidget(cloud)
        hires = QWidget()
        hires_lay = QHBoxLayout(hires)
        hires_lay.setContentsMargins(0, 0, 0, 0)
        self.hr_source = ClipView("Result 832x480", "Expand this path first.")
        self.hr_result = ClipView(self._hr_title(), "Run HiRes all to see the upscaled frames.")
        hires_lay.addWidget(self.hr_source, 1)
        hires_lay.addWidget(self.hr_result, 1)
        self.stack.addWidget(hires)
        lay.addWidget(self.stack, 1)
        return box

    def _build_list(self):
        box, lay = rail(230)
        lay.addWidget(lab("Queue"))
        self.rows = side_list()
        self.rows.currentItemChanged.connect(lambda *_a: self._load_clips())
        lay.addWidget(self.rows, 1)
        self.count = hint("")
        lay.addWidget(self.count)
        return box

    def _selected(self) -> str:
        item = self.rows.currentItem()
        if item is None:
            return ""
        return str(item.data(Qt.ItemDataRole.UserRole) or "")

    def _scale(self) -> int:
        return int(self.hires_scale.currentData() or HR.SCALE)

    def _hr_title(self) -> str:
        return f"HiRes {self._scale()}x"

    def _fill_hires_levels(self) -> None:
        self.hires_level.clear()
        for key in HR.choices(self._scale()):
            self.hires_level.addItem(HR.NAMES[key], key)

    def _scale_changed(self) -> None:
        self._fill_hires_levels()
        self.hires_fit.setText("")
        self.hr_result.set_title(self._hr_title())
        if not self._busy:
            self.refresh()

    def _disk_status(self, scene: str, ident: str) -> str:
        state = traj_state(scene, ident)
        if state == "expanded":
            hires = HR.hires_state(scene, ident, self._scale())
            if hires == "done":
                return "done +HR"
            if hires == "stale":
                return "done +HR old"
            return "done"
        if state == "guide":
            return "guide"
        return "no guide"

    def _color(self, status: str) -> str:
        if status == "done" or status.startswith("done"):
            return BLUE
        if status in ("retrieve", "sample"):
            return ORANGE
        if status == "failed" or status.endswith("failed"):
            return RED
        if status.startswith("hr "):
            return ORANGE
        return DIM

    def _fill(self, ids: list[str], scene: str) -> None:
        previous = self._selected()
        self._ids = list(ids)
        self.rows.blockSignals(True)
        self.rows.clear()
        for ident in self._ids:
            status = self._status.get(ident) or self._disk_status(scene, ident)
            self._status[ident] = status
            item = QListWidgetItem(f"{ident}   {status}")
            item.setData(Qt.ItemDataRole.UserRole, ident)
            item.setForeground(QColor(self._color(status)))
            self.rows.addItem(item)
        self.rows.blockSignals(False)
        if previous:
            for index in range(self.rows.count()):
                if self.rows.item(index).data(Qt.ItemDataRole.UserRole) == previous:
                    self.rows.setCurrentRow(index)
                    break
        elif self.rows.count():
            self.rows.setCurrentRow(0)
        self._refresh_hint(scene)

    def _refresh_hint(self, scene: str) -> None:
        done = sum(1 for ident in self._ids if str(self._status.get(ident, "")).startswith("done"))
        if self._times:
            waiting = sum(
                1
                for ident in self._ids
                if self._status.get(ident) not in ("done", "skip", "failed") and not str(self._status.get(ident, "")).startswith("done")
            )
            minutes = (sum(self._times) / len(self._times)) * waiting / 60.0
            self.hint_lab.setText(f"About {minutes:.0f} min left for {waiting} paths.")
        else:
            self.hint_lab.setText(f"{len(self._ids)} paths kept, {done} expanded. One 21-frame sample is the profile.")
        self.count.setText(f"{done} / {len(self._ids)} expanded")

    def _has_guide(self, scene: str) -> bool:
        return any(traj_state(scene, ident) in ("guide", "expanded") for ident in self._ids)

    def _has_expanded(self, scene: str) -> bool:
        return any(traj_state(scene, ident) == "expanded" for ident in self._ids)

    def refresh(self) -> None:
        scene = self._scene()
        ids = ordered_ids(read_selection(scene) or list_trajectories(scene))
        if not self._busy:
            self._status = {ident: self._disk_status(scene, ident) for ident in ids}
        self._fill(ids, scene)
        guides = self._has_guide(scene)
        self.btn_one.setEnabled(guides and not self._busy)
        self.btn_all.setEnabled(guides and not self._busy)
        self.btn_again.setEnabled(guides and not self._busy)
        self.btn_align.setEnabled(self._has_expanded(scene) and not self._busy)
        self.btn_hires.setEnabled(self._has_expanded(scene) and not self._busy)
        selected = self._selected()
        self.btn_hires_one.setEnabled(
            bool(selected) and traj_state(scene, selected) == "expanded" and not self._busy
        )
        self.btn_stop.setEnabled(self._busy)
        self.level.setEnabled(not self._busy)
        self.hires_level.setEnabled(not self._busy)
        self.hires_scale.setEnabled(not self._busy)
        self._load_clips()
        self._load_cloud()

    def _load_clips(self) -> None:
        ident = self._selected()
        folder = render_root(self._scene()) / ident if ident else None
        self.clip_guide.set_path(folder / "render.mp4" if folder is not None else None)
        self.clip_mask.set_path(folder / "render_mask.mp4" if folder is not None else None)
        memory = folder / "memory_inputs" / _MEMORY if folder is not None else None
        self.clip_memory.set_path(memory)
        self.clip_result.set_path(folder / _RESULT if folder is not None else None)
        self.hr_source.set_path(folder / _RESULT if folder is not None else None)
        self.hr_result.set_path(HR.preview_path(self._scene(), ident, self._scale()) if ident else None)

    def _load_cloud(self) -> None:
        scene = self._scene()
        self.cloud.set_path(votion_dir(scene) / "preview" / "aligned_top.png")
        info = render_root(scene) / "generation_bank_worldstereo-memory-dmd" / "pcd_info.json"
        text = ""
        if info.is_file():
            try:
                text = info.read_text(encoding="utf-8")[:800]
            except OSError:
                text = ""
        self.cloud_hint.setText(text)

    def _expand(self, first: bool, restart: bool) -> None:
        scene = self._scene()
        if not self._ids:
            self.hint_lab.setText("Plan and render trajectories first.")
            return
        target = self._ids[0] if first else ""
        self.requestJob.emit(["__hy_expand__", scene, target, "1" if restart else "0", self._level()])

    def _again(self) -> None:
        ident = self._selected()
        if not ident:
            self.hint_lab.setText("Select a path first.")
            return
        self.requestJob.emit(["__hy_expand__", self._scene(), ident, "1", self._level()])

    def _level(self) -> str:
        return str(self.level.currentData() or "auto")

    def _show_fit(self, fit: dict) -> None:
        names = dict(_LEVELS)
        level = str(fit.get("level", ""))
        gpu, _sep, vram = str(fit.get("key", "")).partition("|")
        text = f"{gpu} {vram} GB\n" if gpu else ""
        text += f"Using: {names.get(level, level)}"
        try:
            text += (
                f"\nBudget {float(fit.get('budget_gb', 0)):.1f} GB, on card {float(fit.get('on_card_gb', 0)):.1f} GB, "
                f"in RAM {float(fit.get('parked_gb', 0)):.1f} GB"
            )
            floor = float(fit.get("floor_gb", 0))
            if 0 < float(fit.get("budget_gb", 0)) < floor:
                text += f"\nBelow the {floor:g} GB tested minimum; this will likely run out."
        except (TypeError, ValueError):
            pass
        self.fit_lab.setText(text)

    def _align(self) -> None:
        self.requestJob.emit(["__hy_align__", self._scene()])

    def _hires(self, first: bool) -> None:
        ident = self._selected() if first else ""
        if first and not ident:
            self.hint_lab.setText("Select a path first.")
            return
        self.requestJob.emit(["__hy_hires__", self._scene(), ident, "1" if first else "0",
                              str(self.hires_level.currentData() or "auto"), str(self._scale())])

    def _show_hires_fit(self, fit: dict) -> None:
        level = str(fit.get("level", ""))
        text = f"Using: {fit.get('scale', HR.SCALE)}x, {HR.NAMES.get(level, level)}"
        try:
            text += f"\nBudget {float(fit.get('budget_gb', 0)):.1f} GB, needs {float(fit.get('need_gb', 0)):.1f} GB"
            text += " (measured)" if fit.get("measured") else " (estimate)"
        except (TypeError, ValueError):
            pass
        self.hires_fit.setText(text)

    def on_job_started(self, label: str) -> None:
        if label in self.LABELS:
            self._times = []
        if label == "hy_hires":
            self._hr_shown = False
            self._hr_times = []
            self._hr_left = set()
        super().on_job_started(label)
        if label in self.LABELS:
            self._timer.start()

    def on_job_finished(self, label: str, code: int) -> None:
        self._timer.stop()
        super().on_job_finished(label, code)

    def on_progress(self, label: str, current: int, total: int, pct: int, msg: str) -> None:
        if label in self.LABELS:
            self.progress.setText(f"{pct}%  {current}/{total}  {msg}")

    def on_hy_line(self, parts: list[str]) -> None:
        if len(parts) < 2:
            return
        kind = parts[1]
        scene = self._scene()
        if kind == "queue" and len(parts) > 2:
            try:
                ids = json.loads(parts[2])
            except json.JSONDecodeError:
                return
            if isinstance(ids, list):
                self._status = {str(ident): self._disk_status(scene, str(ident)) for ident in ids}
                self._fill([str(ident) for ident in ids], scene)
        elif kind == "state" and len(parts) > 3:
            ident = parts[2]
            state = parts[3]
            seconds = parts[4] if len(parts) > 4 else "0"
            peak = parts[5] if len(parts) > 5 else "0"
            if state == "done":
                label = f"done {seconds} s  {peak} GB"
                try:
                    self._times.append(float(seconds))
                except ValueError:
                    pass
            else:
                label = state
            self._status[ident] = label if state == "done" else state
            for index in range(self.rows.count()):
                item = self.rows.item(index)
                if item.data(Qt.ItemDataRole.UserRole) == ident:
                    item.setText(f"{ident}   {label}")
                    item.setForeground(QColor(self._color(label if state == "done" else state)))
                    if state in ("retrieve", "sample"):
                        self.rows.setCurrentRow(index)
                    break
            self._refresh_hint(scene)
            self._load_clips()
        elif kind == "hqueue" and len(parts) > 2:
            try:
                queued = json.loads(parts[2])
            except json.JSONDecodeError:
                return
            if isinstance(queued, list):
                self._hr_left = {str(ident) for ident in queued}
                self._hr_times = []
        elif kind == "hires" and len(parts) > 4:
            ident, state, seconds = parts[2], parts[3], parts[4]
            reserved = parts[5] if len(parts) > 5 else "0"
            if state == "upscale":
                label = "hr upscale"
                text = f"{ident}   HiRes..."
                if not self._hr_shown:
                    self._hr_shown = True
                    self._setting_chip = True
                    chip = self.chips.button(2)
                    if chip is not None:
                        chip.setChecked(True)
                    self._setting_chip = False
                    self.stack.setCurrentIndex(2)
            elif state == "done":
                label = "done +HR"
                text = f"{ident}   done +HR {seconds} s  {reserved} GB"
                try:
                    self._hr_times.append(float(seconds))
                except ValueError:
                    pass
                self._hr_left.discard(ident)
            elif state == "skip":
                label = "done +HR"
                text = f"{ident}   {label}"
                self._hr_left.discard(ident)
            elif state == "failed":
                label = "hr failed"
                text = f"{ident}   {label}"
                self._hr_left.discard(ident)
            else:
                return
            self._status[ident] = label
            for index in range(self.rows.count()):
                item = self.rows.item(index)
                if item.data(Qt.ItemDataRole.UserRole) == ident:
                    item.setText(text)
                    item.setForeground(QColor(self._color(label)))
                    if state == "upscale":
                        self.rows.setCurrentRow(index)
                    break
            if self._hr_times:
                average = sum(self._hr_times) / len(self._hr_times)
                waiting = len(self._hr_left)
                self.hint_lab.setText(f"HiRes: about {average * waiting / 60:.0f} min left for {waiting} paths.")
            self._load_clips()
        elif kind == "hfit" and len(parts) > 2:
            try:
                fit = json.loads(parts[2])
            except json.JSONDecodeError:
                return
            if isinstance(fit, dict):
                self._show_hires_fit(fit)
        elif kind == "fit" and len(parts) > 2:
            try:
                fit = json.loads(parts[2])
            except json.JSONDecodeError:
                return
            if isinstance(fit, dict):
                self._show_fit(fit)
        elif kind == "aligned":
            self._setting_chip = True
            aligned = self.chips.button(1)
            if aligned is not None:
                aligned.setChecked(True)
            self._setting_chip = False
            self.stack.setCurrentIndex(1)
            self._load_cloud()

    def _poll(self) -> None:
        self._load_clips()
        self._load_cloud()
