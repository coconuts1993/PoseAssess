"""3D view page: watch the reconstructed skeleton move in 3D.

Loads a triangulated .trc from the project's pose-3d/ folder and animates it, so
the clinician can visually verify tracking quality and see the movement (and
left/right asymmetry) before trusting the joint-angle numbers.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QComboBox, QFileDialog, QHBoxLayout, QLabel, QPushButton, QVBoxLayout,
)

from ..widgets.skeleton_3d import Skeleton3DViewer
from .base_page import BasePage


def _trc_sort_key(p: Path):
    # prefer filtered (butterworth) over raw, but LSTM-augmented last (different
    # marker set); clinicians want the clean keypoint skeleton.
    name = p.name.lower()
    return ("lstm" in name, "filt" not in name, name)


class Viz3DPage(BasePage):
    nav_title = "5. 3D View"

    def __init__(self, state):
        super().__init__(state)
        root = QVBoxLayout(self)

        bar = QHBoxLayout()
        self.file_combo = QComboBox()
        self.file_combo.currentIndexChanged.connect(self._load_selected)
        reload_btn = QPushButton("Reload")
        reload_btn.clicked.connect(self._refresh_list)
        browse_btn = QPushButton("Load .trc…")
        browse_btn.clicked.connect(self._browse)
        bar.addWidget(QLabel("Trajectory:"))
        bar.addWidget(self.file_combo, 1)
        bar.addWidget(reload_btn)
        bar.addWidget(browse_btn)
        root.addLayout(bar)

        self.viewer = Skeleton3DViewer()
        root.addWidget(self.viewer, 1)

        self.status = QLabel("")
        self.status.setWordWrap(True)
        root.addWidget(self.status)

    def on_project_changed(self, project):
        self._refresh_list()

    def _refresh_list(self):
        self.file_combo.blockSignals(True)
        self.file_combo.clear()
        proj = self.state.project
        if proj and proj.pose3d_dir.exists():
            trcs = sorted(proj.pose3d_dir.glob("*.trc"), key=_trc_sort_key)
            for t in trcs:
                n = t.name.lower()
                if "lstm" in n:
                    tag = "augmented markers"
                elif "filt" in n:
                    tag = "FILTERED — recommended"
                else:
                    tag = "raw (unfiltered — will look jittery)"
                self.file_combo.addItem(f"{t.name}   ·  {tag}", str(t))
        self.file_combo.blockSignals(False)
        if self.file_combo.count():
            self.file_combo.setCurrentIndex(0)
            self._load_selected()
        else:
            self.viewer.clear()
            self.status.setText("No .trc in pose-3d/. Run the pipeline through "
                                "triangulation first, or load one manually.")

    def _load_selected(self, *_):
        path = self.file_combo.currentData()
        if not path:
            return
        ok = self.viewer.load_trc(path)
        if ok:
            trc = self.viewer._trc
            msg = f"{Path(path).name} — {trc.n_frames} frames, {len(trc.markers)} markers"
            # overlay camera poses from the project's calibration, if present
            proj = self.state.project
            if proj:
                calib = next(iter(sorted(proj.calibration_dir.glob("Calib*.toml"))), None)
                if calib:
                    try:
                        from poseassess.core.calib_read import read_camera_poses
                        poses = read_camera_poses(calib)
                        self.viewer.set_cameras(poses)
                        msg += f" · {len(poses)} cameras"
                    except Exception as e:  # noqa: BLE001
                        msg += f" · (cameras unavailable: {e})"
            self.status.setText(msg)
        else:
            self.status.setText(f"Could not load {Path(path).name}")

    def _browse(self):
        start = str(self.state.project.pose3d_dir) if self.state.project else ""
        f, _ = QFileDialog.getOpenFileName(self, "Load .trc", start, "TRC (*.trc)")
        if f and self.viewer.load_trc(f):
            self.status.setText(Path(f).name)
