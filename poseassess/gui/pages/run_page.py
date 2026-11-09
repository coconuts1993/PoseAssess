"""Run page: execute pipeline stages with a live log."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QMessageBox, QPlainTextEdit,
    QPushButton, QVBoxLayout, QWidget,
)

from poseassess.core.pipeline import Pipeline
from ..workers import PipelineWorker, run_in_thread
from .base_page import BasePage


STAGE_LABELS = {
    "calibration": "Calibration",
    "pose": "2D pose",
    "synchronization": "Synchronize",
    "personAssociation": "Person assoc.",
    "triangulation": "Triangulate 3D",
    "filtering": "Filter 3D",
    "markerAugmentation": "Marker augment",
    "kinematics": "OpenSim IK",
}


class RunPage(BasePage):
    nav_title = "4. Run"

    def __init__(self, state):
        super().__init__(state)
        self._thread = None
        self._worker = None

        root = QVBoxLayout(self)

        stages_box = QGroupBox("Stages to run")
        grid = QGridLayout(stages_box)
        # Stages off by default: 'calibration' (usually a ready Calib.toml is
        # imported/reused) and 'synchronization' (multi-GoPro capture is
        # pre-synced; re-syncing wastes time and can misalign).
        off_by_default = {"calibration", "synchronization"}
        self.stage_checks: dict[str, QCheckBox] = {}
        for idx, name in enumerate(Pipeline.STAGES):
            cb = QCheckBox(STAGE_LABELS.get(name, name))
            cb.setChecked(name not in off_by_default)
            if name == "synchronization":
                cb.setToolTip("Leave off if your cameras are already synced "
                              "(hardware timecode / synced clocks).")
            if name == "calibration":
                cb.setToolTip("Leave off if a Calib.toml is already in the "
                              "project's calibration folder.")
            self.stage_checks[name] = cb
            grid.addWidget(cb, idx // 4, idx % 4)
        root.addWidget(stages_box)

        ctl = QHBoxLayout()
        self.keep_going = QCheckBox("Continue on error")
        self.run_btn = QPushButton("▶ Run pipeline")
        self.run_btn.clicked.connect(self._run)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self._cancel)
        ctl.addWidget(self.keep_going)
        ctl.addStretch(1)
        ctl.addWidget(self.cancel_btn)
        ctl.addWidget(self.run_btn)
        root.addLayout(ctl)

        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(5000)
        root.addWidget(self.log_view, 1)

        self.summary = QLabel("")
        self.summary.setWordWrap(True)
        root.addWidget(self.summary)

    def _selected_stages(self) -> list[str]:
        return [n for n in Pipeline.STAGES if self.stage_checks[n].isChecked()]

    def _run(self):
        proj = self.state.project
        if not proj:
            return
        # Guard against starting a second run while one is live: replacing
        # self._thread/self._worker would orphan the running QThread and crash
        # ("QThread: Destroyed while thread is still running").
        if self._thread is not None and self._thread.isRunning():
            self._append("[a run is already in progress — cancel it first]")
            return
        stages = self._selected_stages()
        if not stages:
            return
        export = self.state.busy("export")
        if export:  # it replaces videos/ and Config.toml while the pipeline would read them
            QMessageBox.warning(self, "Run pipeline", f"Wait until {export} has finished: it "
                                "replaces the videos and Config.toml that the pipeline reads.")
            return
        self.log_view.clear()
        self.summary.setText("")
        self._set_running(True)

        self._worker = PipelineWorker(
            proj, stages=stages, stop_on_error=not self.keep_going.isChecked()
        )
        self._worker.log.connect(self._append)
        self._worker.stage_done.connect(self._on_stage)
        self._worker.finished.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._thread = run_in_thread(self._worker)
        self.state.set_busy("pipeline", "the pipeline run (4. Run)")

    def _cancel(self):
        if self._worker:
            self._worker.cancel()
            self._append("[cancel requested…]")

    def _set_running(self, running: bool):
        self.run_btn.setEnabled(not running)
        self.cancel_btn.setEnabled(running)

    def _append(self, line: str):
        self.log_view.appendPlainText(line)

    def _on_stage(self, res):
        mark = "OK" if res.ok else "FAIL"
        self._append(f"  → {res.name}: {mark} {res.message}")

    def _on_finished(self, results):
        self.state.set_busy("pipeline", None)
        self._set_running(False)
        n_ok = sum(1 for r in results if r.ok)
        self.summary.setText(f"Done: {n_ok}/{len(results)} stages ok.")

    def _on_failed(self, msg):
        self.state.set_busy("pipeline", None)
        self._set_running(False)
        self._append(f"[pipeline error] {msg}")
        self.summary.setText(f"Pipeline error: {msg}")
