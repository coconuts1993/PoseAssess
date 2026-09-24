"""Results page: clinical assessment of the OpenSim joint angles.

Loads a kinematics/*.mot file, shows range-of-motion + left/right symmetry in a
table (asymmetries flagged), plots each joint's R/L angle curve, and exports the
report to CSV/JSON. This is the clinical layer that the underlying engines
(Pose2Sim/OpenSim) don't provide.

The joint angles are in the "Joint angles" tab (unchanged). The "Balance (Wii)" tab holds the
Wii Balance Board analysis (``BalancePanel``: recording + alignment, COP / COM sway metrics,
force plots, fused-table export).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QComboBox, QFileDialog, QHBoxLayout, QHeaderView, QLabel, QMessageBox,
    QPushButton, QSplitter, QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget,
)

from poseassess.core.assessment import read_mot, analyze, CLINICAL_JOINTS, AXIAL_JOINTS
from ..widgets.balance_panel import BalancePanel
from ..widgets.plot_canvas import PlotCanvas
from .base_page import BasePage


class ResultsPage(BasePage):
    nav_title = "6. Results"

    def __init__(self, state):
        super().__init__(state)
        self._motion = None
        self._report = None

        root = QVBoxLayout(self)

        # ---- top bar ----
        bar = QHBoxLayout()
        self.file_label = QLabel("No motion loaded.")
        self.file_label.setWordWrap(True)
        load_btn = QPushButton("Load latest result")
        load_btn.clicked.connect(self._load_latest)
        browse_btn = QPushButton("Load .mot…")
        browse_btn.clicked.connect(self._browse)
        self.export_btn = QPushButton("Export report…")
        self.export_btn.clicked.connect(self._export)
        self.export_btn.setEnabled(False)
        self.export_csv_btn = QPushButton("Export all angles (CSV)…")
        self.export_csv_btn.setToolTip("Write every joint coordinate: a full "
                                       "time-series CSV + a per-coordinate summary CSV.")
        self.export_csv_btn.clicked.connect(self._export_all_csv)
        self.export_csv_btn.setEnabled(False)
        bar.addWidget(self.file_label, 1)
        bar.addWidget(load_btn)
        bar.addWidget(browse_btn)
        bar.addWidget(self.export_btn)
        bar.addWidget(self.export_csv_btn)
        root.addLayout(bar)

        # ---- tabs: joint angles (split: table | plot) | balance (Wii) ----
        self.tabs = QTabWidget()
        root.addWidget(self.tabs, 1)
        split = QSplitter(Qt.Horizontal)
        self.tabs.addTab(split, "Joint angles")

        left = QWidget(); lv = QVBoxLayout(left)
        lv.addWidget(QLabel("<b>Range of motion & symmetry</b>"))
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["Joint", "R ROM°", "L ROM°", "Sym %", "Flag"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        lv.addWidget(self.table)
        split.addWidget(left)

        right = QWidget(); rv = QVBoxLayout(right)
        sel = QHBoxLayout()
        sel.addWidget(QLabel("Joint:"))
        self.joint_combo = QComboBox()
        self.joint_combo.currentIndexChanged.connect(self._plot_selected)
        sel.addWidget(self.joint_combo, 1)
        rv.addLayout(sel)
        self.canvas = PlotCanvas()
        rv.addWidget(self.canvas, 1)
        split.addWidget(right)
        split.setSizes([460, 540])

        self.balance = BalancePanel(state)
        self.tabs.addTab(self.balance, "Balance (Wii)")

        self.note = QLabel(
            "<small>Symmetry index = |R−L| ROM / mean ROM. "
            "Flagged (⚠) when > 10%. Not a medical diagnosis.</small>")
        self.note.setWordWrap(True)
        root.addWidget(self.note)

    def shutdown(self) -> None:
        self.balance.shutdown()

    # ---- loading ---------------------------------------------------------- #
    def on_project_changed(self, project):
        self.table.setRowCount(0)
        self.joint_combo.clear()
        self.canvas.clear()
        self._motion = self._report = None
        self.export_btn.setEnabled(False)
        self.export_csv_btn.setEnabled(False)
        if project:
            self.file_label.setText("No motion loaded. Click “Load latest result”.")
        else:
            self.file_label.setText("No project open.")

    def _load_latest(self):
        proj = self.state.project
        if not proj:
            return
        mots = sorted(proj.kinematics_dir.glob("*.mot"))
        if not mots:
            QMessageBox.information(self, "Results",
                                    f"No .mot files in {proj.kinematics_dir}.\n"
                                    f"Run the pipeline through the kinematics stage first.")
            return
        self._load(mots[-1])

    def _browse(self):
        start = str(self.state.project.kinematics_dir) if self.state.project else ""
        f, _ = QFileDialog.getOpenFileName(self, "Load OpenSim motion", start,
                                           "OpenSim motion (*.mot *.sto)")
        if f:
            self._load(Path(f))

    def _load(self, path: Path):
        try:
            self._motion = read_mot(path)
            self._report = analyze(self._motion)
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "Load failed", str(e))
            return
        self._mot_path = path
        self.file_label.setText(
            f"{path.name} — {self._motion.n_frames} frames, {self._motion.duration:.2f}s")
        self._fill_table()
        self._fill_joint_combo()
        self.export_btn.setEnabled(True)
        self.export_csv_btn.setEnabled(True)

    # ---- views ------------------------------------------------------------ #
    def _fill_table(self):
        rep = self._report
        self.table.setRowCount(0)
        for b in rep.bilateral:
            self._add_row(b.label,
                          f"{b.right.rom:.1f}" if b.right else "—",
                          f"{b.left.rom:.1f}" if b.left else "—",
                          f"{b.rom_symmetry_index:.1f}" if b.rom_symmetry_index is not None else "—",
                          "⚠" if b.flagged else "", b.flagged)
        for a in rep.axial:
            self._add_row(a.coord, f"{a.rom:.1f}", "—", "—", "", False)

    def _add_row(self, joint, r, l, sym, flag, flagged):
        row = self.table.rowCount()
        self.table.insertRow(row)
        for col, val in enumerate([joint, r, l, sym, flag]):
            item = QTableWidgetItem(val)
            if col > 0:
                item.setTextAlignment(Qt.AlignCenter)
            if flagged:
                item.setForeground(QColor("#e06c6c"))
            self.table.setItem(row, col, item)

    def _fill_joint_combo(self):
        self.joint_combo.blockSignals(True)
        self.joint_combo.clear()
        cols = set(self._motion.df.columns)
        for base, label in CLINICAL_JOINTS.items():
            if f"{base}_r" in cols or f"{base}_l" in cols:
                self.joint_combo.addItem(label, base)
        for coord, label in AXIAL_JOINTS.items():
            if coord in cols:
                self.joint_combo.addItem(label, coord)
        self.joint_combo.blockSignals(False)
        if self.joint_combo.count():
            self.joint_combo.setCurrentIndex(0)
            self._plot_selected()

    def _plot_selected(self):
        if not self._motion or self.joint_combo.count() == 0:
            return
        base = self.joint_combo.currentData()
        label = self.joint_combo.currentText()
        self.canvas.plot_bilateral(self._motion, base, label)

    # ---- export ----------------------------------------------------------- #
    def _export(self):
        if not self._report:
            return
        proj = self.state.project
        start = str(proj.root / "assessment_report") if proj else "assessment_report"
        path, _ = QFileDialog.getSaveFileName(
            self, "Export report", start, "JSON (*.json);;CSV (*.csv)")
        if not path:
            return
        p = Path(path)
        if p.suffix.lower() == ".csv":
            self._export_csv(p)
        else:
            p.write_text(json.dumps(self._report.to_dict(), indent=2))
        QMessageBox.information(self, "Export", f"Saved {p.name}")

    def _export_all_csv(self):
        if not self._motion:
            return
        proj = self.state.project
        stem = Path(getattr(self, "_mot_path", "result")).stem or "result"
        start = str((proj.root if proj else Path.cwd()) / stem)
        # user picks a base location; we write <base>_angles.csv + <base>_summary.csv
        base, _ = QFileDialog.getSaveFileName(
            self, "Export all joint angles (CSV base name)", start, "CSV (*.csv)")
        if not base:
            return
        base_path = Path(base)
        out_dir = base_path.parent
        out_stem = base_path.stem
        from poseassess.core.assessment import export_all
        try:
            out = export_all(self._motion, out_dir, out_stem)
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "Export failed", str(e))
            return
        n = len(self._motion.df.columns)
        QMessageBox.information(
            self, "Export all angles",
            f"Wrote {n} coordinates:\n• {out['angles'].name} (time series)\n"
            f"• {out['summary'].name} (per-coordinate summary)")

    def _export_csv(self, path: Path):
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["joint", "side", "min", "max", "rom", "mean",
                        "rom_symmetry_index", "flagged"])
            for b in self._report.bilateral:
                if b.right:
                    w.writerow([b.label, "R", f"{b.right.min:.2f}", f"{b.right.max:.2f}",
                                f"{b.right.rom:.2f}", f"{b.right.mean:.2f}",
                                f"{b.rom_symmetry_index:.1f}" if b.rom_symmetry_index else "",
                                b.flagged])
                if b.left:
                    w.writerow([b.label, "L", f"{b.left.min:.2f}", f"{b.left.max:.2f}",
                                f"{b.left.rom:.2f}", f"{b.left.mean:.2f}", "", ""])
            for a in self._report.axial:
                w.writerow([a.coord, "-", f"{a.min:.2f}", f"{a.max:.2f}",
                            f"{a.rom:.2f}", f"{a.mean:.2f}", "", ""])
