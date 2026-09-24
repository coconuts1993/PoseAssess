"""Benchmark page: compare 2D backends on the same trial by joint-angle error.

Runs the identical downstream pipeline (triangulation -> ... -> OpenSim IK) for
each selected 2D backend, then tabulates per-joint RMSE of every backend against
a chosen reference. This is the harness that makes "which pose model is more
accurate for clinical kinematics?" answerable — and reportable.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QObject, QThread, Signal
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QGroupBox, QHBoxLayout, QHeaderView, QLabel,
    QMessageBox, QPlainTextEdit, QPushButton, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget,
)

from poseassess.plugins import base
from poseassess.core.assessment.metrics import CLINICAL_JOINTS
from poseassess.core.pose_backends import POSE_BACKENDS
from .base_page import BasePage

ALL_BACKENDS = list(POSE_BACKENDS)
DEFAULT_CHECKED = ["rtmpose_lite", "rtmpose", "rtmpose_perf"]


class _BenchWorker(QObject):
    log = Signal(str)
    done = Signal(object, str)   # (BenchmarkResult|None, error_str)

    def __init__(self, project, backends, reference):
        super().__init__()
        self.project = project
        self.backends = backends
        self.reference = reference

    def run(self):
        try:
            from ..workers import silence_matplotlib_gui
            silence_matplotlib_gui()
            from poseassess.core.benchmark import run_backends
            result = run_backends(self.project, self.backends, progress=self.log.emit)
            self.done.emit(result, "")
        except Exception as e:  # noqa: BLE001
            self.done.emit(None, f"{type(e).__name__}: {e}")


class BenchmarkPage(BasePage):
    nav_title = "7. Benchmark"

    def __init__(self, state):
        super().__init__(state)
        self._thread = None
        self._worker = None

        root = QVBoxLayout(self)

        cfg = QGroupBox("Backends to compare (same calibration + videos + downstream)")
        cl = QHBoxLayout(cfg)
        self.checks: dict[str, QCheckBox] = {}
        for b in ALL_BACKENDS:
            cb = QCheckBox(b)
            cb.setToolTip(POSE_BACKENDS[b].display)
            if b in DEFAULT_CHECKED:
                cb.setChecked(True)
            self.checks[b] = cb
            cl.addWidget(cb)
        cl.addStretch(1)
        cl.addWidget(QLabel("Reference:"))
        self.ref_combo = QComboBox()
        self.ref_combo.addItems(ALL_BACKENDS)
        cl.addWidget(self.ref_combo)
        root.addWidget(cfg)

        ctl = QHBoxLayout()
        self.run_btn = QPushButton("▶ Run benchmark")
        self.run_btn.clicked.connect(self._run)
        self.load_btn = QPushButton("Load existing results")
        self.load_btn.clicked.connect(self._load_existing)
        ctl.addWidget(self.run_btn)
        ctl.addWidget(self.load_btn)
        ctl.addStretch(1)
        root.addLayout(ctl)

        self.log = QPlainTextEdit(); self.log.setReadOnly(True); self.log.setMaximumHeight(120)
        root.addWidget(self.log)

        self.table = QTableWidget(0, 1)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        root.addWidget(self.table, 1)

        self.note = QLabel(
            "<small>Cells = RMSE (deg) of each backend's joint angle vs the "
            "reference backend, per joint. Lower = closer to reference. "
            "Use a marker-based export as reference for true accuracy.</small>")
        self.note.setWordWrap(True)
        root.addWidget(self.note)

    # ---- run ----
    def _selected(self) -> list[str]:
        return [b for b in ALL_BACKENDS if self.checks[b].isChecked()]

    def _run(self):
        proj = self.state.project
        if not proj:
            return
        backends = self._selected()
        if len(backends) < 2:
            QMessageBox.information(self, "Benchmark", "Select at least two backends.")
            return
        if not list(proj.calibration_dir.glob("Calib*.toml")):
            QMessageBox.warning(self, "Benchmark",
                                "No calibration found. Calibrate the project first.")
            return
        self.log.clear()
        self.run_btn.setEnabled(False)
        self._worker = _BenchWorker(proj, backends, self.ref_combo.currentText())
        self._worker.log.connect(self.log.appendPlainText)
        self._worker.done.connect(self._on_done)
        self._thread = QThread()
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.done.connect(self._thread.quit)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.start()

    def _on_done(self, result, err):
        self.run_btn.setEnabled(True)
        if err:
            self.log.appendPlainText(f"[error] {err}")
            QMessageBox.critical(self, "Benchmark failed", err)
            return
        self._show_result(result)

    def _load_existing(self):
        proj = self.state.project
        if not proj:
            return
        from poseassess.core.benchmark import BenchmarkResult, BackendRun
        bench_dir = proj.root / "benchmark"
        runs = []
        for sub in sorted(bench_dir.glob("*/result.mot")):
            runs.append(BackendRun(sub.parent.name, True, sub, "loaded"))
        if len(runs) < 2:
            QMessageBox.information(
                self, "Benchmark",
                f"Need >=2 results in {bench_dir}/<backend>/result.mot. "
                f"Found {len(runs)}. Run the benchmark first.")
            return
        self._show_result(BenchmarkResult(runs=runs))

    def _show_result(self, result):
        from poseassess.core.benchmark import compare_runs
        try:
            ref, comparisons = compare_runs(result, reference=self.ref_combo.currentText())
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, "Benchmark", str(e))
            return
        self._fill_table(ref, comparisons)

    def _fill_table(self, reference: str, comparisons: list):
        # rows = clinical coords present; columns = one per compared backend
        others = [c.other for c in comparisons]
        coord_order = []
        for base_name in CLINICAL_JOINTS:
            for suf in ("_r", "_l"):
                coord_order.append(base_name + suf)

        # build lookup: (backend, coord) -> rmse
        lut = {}
        present = set()
        for c in comparisons:
            for cc in c.coords:
                lut[(c.other, cc.coord)] = cc.rmse
                present.add(cc.coord)
        rows = [c for c in coord_order if c in present]

        self.table.clear()
        self.table.setColumnCount(1 + len(others))
        self.table.setHorizontalHeaderLabels(
            [f"Joint (vs {reference})"] + [f"{o} RMSE°" for o in others])
        self.table.setRowCount(len(rows) + 1)

        for i, coord in enumerate(rows):
            self.table.setItem(i, 0, QTableWidgetItem(coord))
            for j, o in enumerate(others):
                v = lut.get((o, coord))
                item = QTableWidgetItem(f"{v:.2f}" if v is not None else "—")
                item.setTextAlignment(Qt.AlignCenter)
                self.table.setItem(i, j + 1, item)

        # mean row
        mrow = len(rows)
        mean_item = QTableWidgetItem("MEAN RMSE")
        self.table.setItem(mrow, 0, mean_item)
        for j, c in enumerate(comparisons):
            it = QTableWidgetItem(f"{c.mean_rmse:.2f}")
            it.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(mrow, j + 1, it)
        self.log.appendPlainText(f"compared {len(others)} backend(s) vs {reference}")
