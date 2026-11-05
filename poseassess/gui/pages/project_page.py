"""Project page: create or open a workspace and edit its settings."""
from __future__ import annotations

import threading
from pathlib import Path

from PySide6.QtCore import QObject, Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from poseassess.core import openpose_setup as ops
from poseassess.core.config_gen import generate_config
from poseassess.core.project import CheckerboardConfig, Project, ProjectConfig

from ..widgets.openpose_dialog import (
    MUTED,
    WARN,
    OpenPoseDialog,
    OpenPoseReport,
    clean_setting,
    openpose_report,
    status_html,
    stop_jobs,
)
from .base_page import BasePage

OPENPOSE_BACKEND = "openpose"
CHECK_DELAY_MS = 300   # re-check the OpenPose folder this long after the last keystroke
#: Dynamic property of the folder field while no usable OpenPose install is found.
MISSING_PROP = "openposeMissing"
_MISSING_STYLE = (f'QLineEdit[{MISSING_PROP}="true"] {{ border: 2px solid {WARN}; '
                  "border-radius: 3px; }")
AUTO_PLACEHOLDER = "auto-detect (C:\\openpose, next to PoseAssess, ...)"
MISSING_PLACEHOLDER = "e.g. C:\\openpose (none found automatically)"


class _CheckBridge(QObject):
    """Delivers results of the background OpenPose / GPU checks to the GUI thread."""
    report = Signal(int, object)   # sequence number, OpenPoseReport
    gpu = Signal(object)           # list[str] (gpu_warnings)


class BackendCombo(QComboBox):
    """Combo box that tells a choice made in its popup list (`chosen_in_popup`) from a
    change with the arrow keys or the mouse wheel on the closed box: Qt emits `activated`
    for all of them, and only a choice in the list may move the keyboard focus (an arrow
    key press must stay in the box, a wheel turn over it is often meant for the page)."""
    chosen_in_popup = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._closing_popup = False
        self.activated.connect(self._on_activated)

    def hidePopup(self) -> None:
        # a click or Enter in the list calls hidePopup() and then emits activated
        self._closing_popup = True
        QTimer.singleShot(0, self._popup_closed)
        super().hidePopup()

    def _popup_closed(self) -> None:
        self._closing_popup = False

    def _on_activated(self, index: int) -> None:
        in_popup = self._closing_popup or self.view().isVisible()
        self._closing_popup = False
        if in_popup:
            self.chosen_in_popup.emit(index)


class ProjectPage(BasePage):
    nav_title = "1. Project"
    needs_project = False

    def __init__(self, state):
        super().__init__(state)
        # OpenPose folder checks (background threads, results via the bridge)
        self._check_seq = 0      # number of the newest check
        self._report_seq = 0     # number of the check shown
        self._op_report: OpenPoseReport | None = None
        self._gpu_warnings: list[str] | None = None   # checked once (runs nvidia-smi)
        self._gpu_check_started = False
        self._focus_if_missing = False   # the user just chose OpenPose in the combo box
        self._remember_on_check: str | None = None   # saved folder to remember once checked
        self._dialog: OpenPoseDialog | None = None    # "Get OpenPose…" while open
        self._bridge = _CheckBridge()
        self._bridge.report.connect(self._on_openpose_report)
        self._bridge.gpu.connect(self._on_gpu_checked)
        self._check_timer = QTimer(self)
        self._check_timer.setSingleShot(True)
        self._check_timer.setInterval(CHECK_DELAY_MS)
        self._check_timer.timeout.connect(self._check_openpose)

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
        self.backend_combo = BackendCombo()
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
        self.openpose_row = self._build_openpose_row()
        form.addRow("OpenPose folder", self.openpose_row)
        self._form = form
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
        self.backend_combo.currentIndexChanged.connect(self._update_openpose_row)
        self.backend_combo.chosen_in_popup.connect(self._on_backend_activated)  # user choice
        self._update_openpose_row()

    def _build_openpose_row(self) -> QWidget:
        """Folder field + Browse / Get OpenPose + status line (OpenPose backend only)."""
        row = QWidget()
        v = QVBoxLayout(row)
        v.setContentsMargins(0, 0, 0, 0)
        h = QHBoxLayout()
        self.openpose_edit = QLineEdit()
        self.openpose_edit.setPlaceholderText(AUTO_PLACEHOLDER)
        self.openpose_edit.setToolTip(
            "Folder of your own OpenPose install (with bin\\OpenPoseDemo.exe and models\\). "
            "Empty: PoseAssess looks in C:\\openpose, D:\\openpose, %USERPROFILE%\\openpose, "
            "an openpose folder next to run.bat and the last folder used on this PC.")
        self.openpose_edit.setObjectName("openposeFolder")
        self.openpose_edit.setProperty(MISSING_PROP, False)
        self.openpose_edit.setStyleSheet(_MISSING_STYLE)
        self.openpose_edit.textChanged.connect(self._schedule_openpose_check)
        self.openpose_edit.editingFinished.connect(self._check_if_scheduled)
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse_openpose)
        get = QPushButton("Get OpenPose…")
        get.setToolTip("Download the official OpenPose, choose an existing install or import "
                       "the BODY_25B model file.")
        get.clicked.connect(self._get_openpose)
        h.addWidget(self.openpose_edit, 1)
        h.addWidget(browse)
        h.addWidget(get)
        v.addLayout(h)
        self.openpose_status = QLabel()
        self.openpose_status.setWordWrap(True)
        self.openpose_status.setTextFormat(Qt.RichText)
        sp = self.openpose_status.sizePolicy()
        sp.setHorizontalPolicy(QSizePolicy.Ignored)  # long paths wrap, never widen the page
        self.openpose_status.setSizePolicy(sp)
        v.addWidget(self.openpose_status)
        return row

    # ---- OpenPose folder ----
    def _openpose_shown(self) -> bool:
        return self.backend_combo.currentData() == OPENPOSE_BACKEND

    def _update_openpose_row(self, *_):
        shown = self._openpose_shown()
        self._form.setRowVisible(self.openpose_row, shown)
        self._set_missing_highlight(self.openpose_missing)
        if shown:
            self._start_gpu_check()
            self._check_openpose()
        else:
            self._check_timer.stop()

    def _schedule_openpose_check(self, *_):
        if self._openpose_shown():
            self._check_timer.start()

    def _check_if_scheduled(self) -> None:
        """Enter / leaving the field: check now instead of after the typing delay."""
        if self._check_timer.isActive():
            self._check_openpose()

    def _on_backend_activated(self, _index: int) -> None:
        """The user chose a backend in the list: for OpenPose without a usable install, put
        the keyboard focus into the folder field (once the check of the folder says so)."""
        self._focus_if_missing = self._openpose_shown()
        if self._focus_if_missing and not self.openpose_check_pending():
            self._apply_missing_focus()

    def _apply_missing_focus(self) -> None:
        if not self._focus_if_missing:
            return
        self._focus_if_missing = False
        if not self.openpose_missing or not self._openpose_shown():
            return
        focus = self.window().focusWidget()
        if focus is None or focus is self.backend_combo or focus is self.openpose_edit:
            self.openpose_edit.setFocus(Qt.OtherFocusReason)  # unless the user moved on

    def _check_openpose(self) -> None:
        """Check the folder setting in a background thread (it looks into several folders,
        some may be on slow or disconnected drives); `_on_openpose_report` shows it."""
        self._check_timer.stop()
        self._check_seq += 1
        seq, text, bridge = self._check_seq, self.openpose_edit.text(), self._bridge
        if self._op_report is None:
            self._show_openpose_status()

        def work():
            try:
                report = openpose_report(text)
            except Exception as e:  # noqa: BLE001 - shown in the status line
                report = OpenPoseReport(False, None, [(f"OpenPose check failed: {e}", WARN)])
            bridge.report.emit(seq, report)

        threading.Thread(target=work, name="openpose-check", daemon=True).start()

    def _on_openpose_report(self, seq: int, report: OpenPoseReport) -> None:
        if seq != self._check_seq:
            return  # an older check; a newer one is running
        self._report_seq = seq
        self._op_report = report
        self._show_openpose_status()
        if self._remember_on_check is not None and report.setting == self._remember_on_check:
            self._remember_on_check = None
            self._remember_checked(report)
        if self._dialog is not None and not self.openpose_check_pending():
            self._dialog.offer_status(self._dialog_status(report))
        if not self.openpose_check_pending():
            self._apply_missing_focus()

    def _start_gpu_check(self) -> None:
        if self._gpu_check_started:
            return
        self._gpu_check_started = True
        bridge = self._bridge

        def work():
            try:
                warnings = ops.gpu_warnings(ops.gpu_info())
            except Exception as e:  # noqa: BLE001
                warnings = [f"could not check the NVIDIA GPU: {e}"]
            bridge.gpu.emit(warnings)

        threading.Thread(target=work, name="gpu-check", daemon=True).start()

    def _on_gpu_checked(self, warnings: list[str]) -> None:
        self._gpu_warnings = list(warnings)
        self._show_openpose_status()

    def _show_openpose_status(self) -> None:
        if self._op_report is None:
            lines = [("Checking the OpenPose folder…", MUTED)]
        else:
            lines = list(self._op_report.lines)
        lines += [(f"GPU: {w}", WARN) for w in self._gpu_warnings or []]
        self.openpose_status.setText(status_html(lines))
        self._set_missing_highlight(self.openpose_missing)

    def _set_missing_highlight(self, on: bool) -> None:
        """Amber frame around the folder field while no usable install is found."""
        edit = self.openpose_edit
        if bool(edit.property(MISSING_PROP)) == on:
            return
        edit.setPlaceholderText(MISSING_PLACEHOLDER if on else AUTO_PLACEHOLDER)
        edit.setProperty(MISSING_PROP, on)
        edit.style().unpolish(edit)   # re-evaluate the [openposeMissing] style rule
        edit.style().polish(edit)
        edit.update()

    @property
    def openpose_ok(self) -> bool | None:
        """Result of the last folder check (None before the first one)."""
        return None if self._op_report is None else self._op_report.ok

    @property
    def openpose_missing(self) -> bool:
        """OpenPose is the backend and the last check found no usable install (none at
        all, or an incomplete one): the folder field is highlighted."""
        return self._openpose_shown() and self._op_report is not None and not self._op_report.ok

    def openpose_check_pending(self) -> bool:
        """True while the shown OpenPose status is not that of the current folder text."""
        return self._check_timer.isActive() or self._report_seq != self._check_seq

    def _openpose_setting(self) -> str:
        return clean_setting(self.openpose_edit.text())

    def _current_report(self) -> OpenPoseReport | None:
        """Result of the check of the current folder text (None while it runs)."""
        report = self._op_report
        if (report is None or self.openpose_check_pending()
                or report.setting != self._openpose_setting()):
            return None
        return report

    def _remember_openpose(self, setting: str) -> None:
        """Remember a working OpenPose folder on this computer (for every project) once it
        checks OK. Uses the background check (no file access here: the folder may be on a
        slow or disconnected drive); still running: remembered when it ends."""
        self._remember_on_check = None
        if not setting:
            return
        report = self._current_report()
        if report is not None and report.setting == setting:
            self._remember_checked(report)
            return
        self._remember_on_check = setting
        self._check_openpose()   # now; replaces a check that may be of an older text

    @staticmethod
    def _remember_checked(report: OpenPoseReport) -> None:
        st = report.typed_status
        if st is not None and st.ok and st.root is not None:
            ops.remember_dir(st.root)

    def _known_openpose_root(self) -> Path | None:
        """Install found by the last check (None before it / when none was found)."""
        return self._op_report.root if self._op_report is not None else None

    def _browse_openpose(self):
        start = self._openpose_setting() or str(self._known_openpose_root() or "")
        d = QFileDialog.getExistingDirectory(self, "Select the OpenPose folder", start)
        if d:
            root = ops.find_root(d)
            self.openpose_edit.setText(str(root or Path(d)))
            self._check_openpose()   # now, not after the typing delay

    def _dialog_status(self, report: OpenPoseReport):
        """Install the dialog starts with: the folder entered (to complete it), else the
        auto-detected one."""
        return report.typed_status if report.setting else report.status

    def _get_openpose(self):
        if self._check_timer.isActive():
            self._check_openpose()
        report = self._current_report()
        # known from the background check; still running: offered to the dialog when done
        dlg = OpenPoseDialog(self, status=self._dialog_status(report) if report else None)
        self._dialog = dlg
        try:
            dlg.exec()
        finally:
            self._dialog = None
        picked = dlg.result_root()
        dlg.deleteLater()
        if picked is not None and str(picked) != self._openpose_setting():
            self.openpose_edit.setText(str(picked))
        self._check_openpose()  # now; also after weights were imported into the same folder

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
            openpose_dir=self._openpose_setting(),
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
        self._remember_openpose(cfg.openpose_dir)
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
        c.openpose_dir = self._openpose_setting()
        c.checkerboard = CheckerboardConfig(
            corners_nb=[self.corners_h.value(), self.corners_w.value()],
            square_size=self.square.value(),
        )
        proj.save()
        generate_config(proj)
        self._remember_openpose(c.openpose_dir)
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
        self.openpose_edit.setText(c.openpose_dir or "")
        i = self.backend_combo.findData(c.pose2d_backend)
        if i >= 0:
            self.backend_combo.setCurrentIndex(i)
        self.corners_h.setValue(int(c.checkerboard.corners_nb[0]))
        self.corners_w.setValue(int(c.checkerboard.corners_nb[1]))
        self.square.setValue(float(c.checkerboard.square_size))
        self.status.setText(f"Open: {project.root}")

    def shutdown(self) -> None:
        self._check_timer.stop()
        # OpenPose downloads cancelled by closing their dialog: told to stop now; waited for
        # at exit, after the window has gone (they remove their partial files)
        stop_jobs()
