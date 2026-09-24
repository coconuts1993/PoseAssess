"""Project page: create or open a workspace and edit its settings."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox,
    QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPushButton, QSpinBox,
    QVBoxLayout, QWidget,
)

from poseassess.core.project import Project, ProjectConfig, CheckerboardConfig
from poseassess.core.config_gen import generate_config
from .base_page import BasePage


class ProjectPage(BasePage):
    nav_title = "1. Project"
    needs_project = False

    def __init__(self, state):
        super().__init__(state)
        root = QVBoxLayout(self)

        # ---- open / create row ----
        open_box = QGroupBox("Workspace")
        ob = QHBoxLayout(open_box)
        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("choose or create a project folder...")
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse)
        create = QPushButton("Create New")
        create.clicked.connect(self._create)
        openb = QPushButton("Open")
        openb.clicked.connect(self._open)
        example = QPushButton("Open Example")
        example.setToolTip("Load the bundled sit-to-stand demo (3D view + results, "
                           "no processing needed).")
        example.clicked.connect(self._open_example)
        ob.addWidget(self.path_edit, 1)
        ob.addWidget(browse)
        ob.addWidget(create)
        ob.addWidget(openb)
        ob.addWidget(example)
        root.addWidget(open_box)

        # ---- settings form ----
        self.settings_box = QGroupBox("Session settings")
        form = QFormLayout(self.settings_box)
        self.name_edit = QLineEdit()
        self.cameras_spin = QSpinBox(); self.cameras_spin.setRange(1, 32); self.cameras_spin.setValue(6)
        self.fps_spin = QSpinBox(); self.fps_spin.setRange(1, 1000); self.fps_spin.setValue(60)
        self.backend_combo = QComboBox()
        from poseassess.core.pose_backends import POSE_BACKENDS
        for name, spec in POSE_BACKENDS.items():
            self.backend_combo.addItem(spec.display, name)
        self.backend_combo.setToolTip(
            "2D pose model. RTMPose t/m/x share the HALPE_26 skeleton, so they "
            "give a clean same-downstream accuracy comparison.")
        self.corners_h = QSpinBox(); self.corners_h.setRange(2, 30); self.corners_h.setValue(4)
        self.corners_w = QSpinBox(); self.corners_w.setRange(2, 30); self.corners_w.setValue(7)
        self.square = QDoubleSpinBox(); self.square.setRange(1, 1000); self.square.setValue(60.0); self.square.setSuffix(" mm")

        form.addRow("Session name", self.name_edit)
        form.addRow("Cameras", self.cameras_spin)
        form.addRow("Frame rate", self.fps_spin)
        form.addRow("2D backend", self.backend_combo)
        corners_row = QWidget(); cr = QHBoxLayout(corners_row); cr.setContentsMargins(0, 0, 0, 0)
        cr.addWidget(QLabel("H")); cr.addWidget(self.corners_h)
        cr.addWidget(QLabel("W")); cr.addWidget(self.corners_w); cr.addStretch(1)
        form.addRow("Board inner corners", corners_row)
        form.addRow("Square size", self.square)

        save = QPushButton("Save settings")
        save.clicked.connect(self._save)
        form.addRow(save)
        root.addWidget(self.settings_box)

        self.status = QLabel("No project open.")
        self.status.setWordWrap(True)
        root.addWidget(self.status)
        root.addStretch(1)

        self.settings_box.setEnabled(False)

    # ---- actions ----
    def _browse(self):
        d = QFileDialog.getExistingDirectory(self, "Select project folder")
        if d:
            self.path_edit.setText(d)

    def _create(self):
        root = self.path_edit.text().strip()
        if not root:
            QMessageBox.warning(self, "Create", "Choose a folder first.")
            return
        cfg = ProjectConfig(
            name=self.name_edit.text().strip() or Path(root).name,
            num_cameras=self.cameras_spin.value(),
            frame_rate=self.fps_spin.value(),
            pose2d_backend=self.backend_combo.currentData(),
            checkerboard=CheckerboardConfig(
                corners_nb=[self.corners_h.value(), self.corners_w.value()],
                square_size=self.square.value(),
            ),
        )
        try:
            proj = Project(root, cfg).create()
            generate_config(proj)
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "Create failed", str(e))
            return
        self.state.set_project(proj)

    def _open(self):
        root = self.path_edit.text().strip()
        if not root or not Project.is_project(root):
            QMessageBox.warning(self, "Open", "That folder has no project.toml.")
            return
        try:
            self.state.open_project(root)
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "Open failed", str(e))

    def _open_example(self):
        # bundled at <repo>/examples/Demo_SitToStand
        demo = Path(__file__).resolve().parents[3] / "examples" / "Demo_SitToStand"
        if not Project.is_project(demo):
            QMessageBox.warning(self, "Example",
                                f"Example project not found at\n{demo}")
            return
        try:
            self.state.open_project(demo)
            self.path_edit.setText(str(demo))
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "Example failed", str(e))

    def _save(self):
        proj = self.state.project
        if not proj:
            return
        c = proj.config
        c.name = self.name_edit.text().strip() or c.name
        c.num_cameras = self.cameras_spin.value()
        c.frame_rate = self.fps_spin.value()
        c.pose2d_backend = self.backend_combo.currentData()
        c.checkerboard = CheckerboardConfig(
            corners_nb=[self.corners_h.value(), self.corners_w.value()],
            square_size=self.square.value(),
        )
        proj.save()
        generate_config(proj)
        self.status.setText(f"Saved. Config regenerated at {proj.config_file}")

    # ---- react to project change ----
    def on_project_changed(self, project):
        self.settings_box.setEnabled(project is not None)
        if project is None:
            self.status.setText("No project open.")
            return
        c = project.config
        self.path_edit.setText(str(project.root))
        self.name_edit.setText(c.name)
        self.cameras_spin.setValue(c.num_cameras)
        self.fps_spin.setValue(c.frame_rate)
        i = self.backend_combo.findData(c.pose2d_backend)
        if i >= 0:
            self.backend_combo.setCurrentIndex(i)
        self.corners_h.setValue(int(c.checkerboard.corners_nb[0]))
        self.corners_w.setValue(int(c.checkerboard.corners_nb[1]))
        self.square.setValue(float(c.checkerboard.square_size))
        self.status.setText(f"Open: {project.root}")
