"""Calibration page.

Two tabs:
  • Select frames  — scrub each camera's video and capture calibration frames
                     (intrinsic: many; extrinsic: one reference frame).
  • Corners & run  — click/auto-detect the board corners on the captured
                     extrinsic frame, then run intrinsic+extrinsic calibration
                     to produce Calib.toml.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QThread, QObject, Signal, QRectF
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QButtonGroup, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QGraphicsScene,
    QGraphicsView, QGroupBox, QHBoxLayout, QLabel, QListWidget, QMessageBox,
    QPlainTextEdit, QPushButton, QRadioButton, QTabWidget, QVBoxLayout, QWidget,
)


class _ZoomView(QGraphicsView):
    """Read-only image view with wheel-zoom and drag-pan (for calib QC)."""
    def __init__(self):
        super().__init__()
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self._item = None

    def set_bgr(self, img):
        import numpy as np
        h, w = img.shape[:2]
        rgb = img[:, :, ::-1].copy()
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
        self._scene.clear()
        self._item = self._scene.addPixmap(QPixmap.fromImage(qimg))
        self._scene.setSceneRect(QRectF(0, 0, w, h))
        self.resetTransform()
        self.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio)

    def clear(self):
        self._scene.clear(); self._item = None

    def wheelEvent(self, ev):
        self.scale(1.25 if ev.angleDelta().y() > 0 else 0.8,
                   1.25 if ev.angleDelta().y() > 0 else 0.8)

from poseassess.core.calibration import save_clicked_points
from ..widgets.frame_selector import FrameSelector
from ..widgets.corner_picker import CornerPicker
from .base_page import BasePage

VIDEO_EXTS = ("*.mp4", "*.MP4", "*.avi", "*.AVI", "*.mov", "*.MOV", "*.mkv")


class _CalibWorker(QObject):
    log = Signal(str)
    done = Signal(bool, str)

    def __init__(self, project):
        super().__init__()
        self.project = project

    def run(self):
        try:
            from ..workers import silence_matplotlib_gui
            silence_matplotlib_gui()
            from poseassess.core.calibration import calibrate_project
            outcome = calibrate_project(self.project, progress=self.log.emit)
            self.done.emit(True, f"Calib.toml written: {outcome.calib_toml}")
        except Exception as e:  # noqa: BLE001
            self.done.emit(False, f"{type(e).__name__}: {e}")


class _ExtractWorker(QObject):
    log = Signal(str)
    done = Signal(int)

    def __init__(self, video_path, out_dir, corners_nb, max_frames=20, step=15):
        super().__init__()
        self.video_path = video_path
        self.out_dir = out_dir
        self.corners_nb = corners_nb
        self.max_frames = max_frames
        self.step = step

    def run(self):
        try:
            from poseassess.core.calibration import auto_extract_intrinsic_frames
            saved = auto_extract_intrinsic_frames(
                self.video_path, self.out_dir, self.corners_nb,
                max_frames=self.max_frames, step=self.step, progress=self.log.emit)
            self.done.emit(len(saved))
        except Exception as e:  # noqa: BLE001
            self.log.emit(f"extract failed: {type(e).__name__}: {e}")
            self.done.emit(0)


class CalibrationPage(BasePage):
    nav_title = "2. Calibration"

    def __init__(self, state):
        super().__init__(state)
        self._thread = None
        self._worker = None

        root = QVBoxLayout(self)
        self.tabs = QTabWidget()
        root.addWidget(self.tabs)
        self.tabs.addTab(self._build_frames_tab(), "Select frames")
        self._corners_tab = self._build_corners_tab()
        self.tabs.addTab(self._corners_tab, "Corners & run")
        self._verify_tab = self._build_verify_tab()
        self.tabs.addTab(self._verify_tab, "Verify (cube)")
        # auto-refresh Verify whenever it's opened, so it always reflects the
        # latest Calib.toml + corners (re-running calibration won't leave a stale
        # cube behind).
        self.tabs.currentChanged.connect(self._on_tab_changed)

    # ================= Tab 1: frame selection ============================== #
    def _build_frames_tab(self) -> QWidget:
        w = QWidget()
        root = QHBoxLayout(w)

        left = QWidget(); left.setFixedWidth(230)
        lv = QVBoxLayout(left)

        cam_box = QGroupBox("Camera")
        cb = QVBoxLayout(cam_box)
        self.cam_combo = QComboBox()
        self.cam_combo.currentIndexChanged.connect(self._reload_video)
        cb.addWidget(self.cam_combo)
        lv.addWidget(cam_box)

        mode_box = QGroupBox("Mode")
        mb = QVBoxLayout(mode_box)
        self.rb_intri = QRadioButton("Intrinsic (many frames)")
        self.rb_extri = QRadioButton("Extrinsic (one frame)")
        self.rb_extri.setChecked(True)
        grp = QButtonGroup(self)
        grp.addButton(self.rb_intri); grp.addButton(self.rb_extri)
        self.rb_intri.toggled.connect(self._reload_video)
        mb.addWidget(self.rb_intri); mb.addWidget(self.rb_extri)
        lv.addWidget(mode_box)

        browse = QPushButton("Load video manually…")
        browse.clicked.connect(self._browse_video)
        lv.addWidget(browse)

        self.auto_extract_btn = QPushButton("⚙ Auto-extract intrinsic frames")
        self.auto_extract_btn.setToolTip(
            "Scan the current camera's video and save well-spread frames where "
            "the checkerboard is detected (Intrinsic mode).")
        self.auto_extract_btn.clicked.connect(self._auto_extract_intrinsic)
        lv.addWidget(self.auto_extract_btn)

        lv.addWidget(QLabel("Captured frames:"))
        self.captured = QListWidget()
        lv.addWidget(self.captured, 1)
        clear = QPushButton("Clear captured")
        clear.clicked.connect(self._clear_captured)
        lv.addWidget(clear)

        self.info = QLabel(""); self.info.setWordWrap(True)
        lv.addWidget(self.info)
        root.addWidget(left)

        self.selector = FrameSelector()
        self.selector.frame_captured.connect(self._on_captured)
        root.addWidget(self.selector, 1)
        return w

    # ================= Tab 2: corners + calibrate ========================== #
    def _build_corners_tab(self) -> QWidget:
        w = QWidget()
        root = QHBoxLayout(w)

        left = QWidget(); left.setFixedWidth(230)
        lv = QVBoxLayout(left)

        cam_box = QGroupBox("Camera")
        cb = QVBoxLayout(cam_box)
        self.corner_cam = QComboBox()
        self.corner_cam.currentIndexChanged.connect(self._load_corner_image)
        cb.addWidget(self.corner_cam)
        lv.addWidget(cam_box)

        self.corner_count = QLabel("points: 0 / 0")
        lv.addWidget(self.corner_count)

        auto = QPushButton("Auto-detect corners")
        auto.clicked.connect(self._auto_detect)
        lv.addWidget(auto)
        rev = QPushButton("↺ Reverse corner order (180°)")
        rev.setToolTip("Flip the origin to the opposite end. Use so the GREEN "
                       "origin corner is the SAME physical corner on every camera "
                       "— required for extrinsics, or the world frame breaks.")
        rev.clicked.connect(lambda: self.picker.reverse_order())
        lv.addWidget(rev)
        rotrow = QHBoxLayout()
        rot_cw = QPushButton("⟳ Rotate 90°")
        rot_cw.setToolTip("Turn the corner numbering a quarter turn clockwise. "
                          "Use when a camera saw the board rotated ~90° so its "
                          "grid is transposed vs the others (rows/cols swapped). "
                          "The dots don't move — only their order/origin.")
        rot_cw.clicked.connect(lambda: self._rotate_corners(True))
        rot_ccw = QPushButton("⟲ Rotate -90°")
        rot_ccw.setToolTip("Turn the corner numbering a quarter turn "
                           "counter-clockwise.")
        rot_ccw.clicked.connect(lambda: self._rotate_corners(False))
        rotrow.addWidget(rot_cw)
        rotrow.addWidget(rot_ccw)
        lv.addLayout(rotrow)
        self.snap_chk = QCheckBox("Snap clicks to corner (sub-pixel)")
        self.snap_chk.setChecked(False)
        self.snap_chk.setToolTip("Optional: refine a click to the exact nearby "
                                 "corner. Helps on boards that fill the frame; "
                                 "leave OFF for small/dense boards where it can "
                                 "jump to the wrong corner. Test it on your board.")
        self.snap_chk.toggled.connect(lambda on: setattr(self.picker, "snap_enabled", on))
        lv.addWidget(self.snap_chk)
        clearc = QPushButton("Clear corners")
        clearc.clicked.connect(lambda: self.picker.clear_points())
        lv.addWidget(clearc)
        savep = QPushButton("Save corners for this cam")
        savep.clicked.connect(self._save_points)
        lv.addWidget(savep)

        import_btn = QPushButton("Import intrinsics / calibration…")
        import_btn.setToolTip(
            "Copy an existing intri.yml / extri.yml or Calib.toml into this "
            "project's calibration folder (e.g. from a previous session).")
        import_btn.clicked.connect(self._import_calibration)
        lv.addWidget(import_btn)

        lv.addWidget(QLabel(
            "<small><b>The GREEN dot is the origin (corner 0).</b> It must be the "
            "SAME physical board corner on every camera. If auto-detect put it at "
            "the wrong end, click <b>Reverse corner order</b>.<br>"
            "Left-click: add · drag: move · right-click: remove · wheel: zoom.</small>"))

        lv.addStretch(1)
        self.invert_z_chk = QCheckBox("Flip Z axis (subject upright)")
        self.invert_z_chk.setToolTip(
            "Flip the world Z direction. Turn ON if the calibrated world comes "
            "out Z-down and the reconstructed person ends up upside-down. Takes "
            "effect on the next Run calibration; also updates the Verify cube.")
        self.invert_z_chk.toggled.connect(self._toggle_invert_z)
        lv.addWidget(self.invert_z_chk)
        run = QPushButton("▶ Run calibration (all cams)")
        run.clicked.connect(self._run_calibration)
        lv.addWidget(run)
        self.calib_log = QPlainTextEdit(); self.calib_log.setReadOnly(True)
        self.calib_log.setMaximumHeight(150)
        lv.addWidget(self.calib_log)
        root.addWidget(left)

        self.picker = CornerPicker()
        self.picker.points_changed.connect(self._on_points_changed)
        root.addWidget(self.picker, 1)
        return w

    # ================= Tab 3: verify calibration ========================== #
    def _build_verify_tab(self) -> QWidget:
        w = QWidget()
        root = QHBoxLayout(w)
        left = QWidget(); left.setFixedWidth(260)
        lv = QVBoxLayout(left)

        run = QPushButton("✓ Verify calibration (all cameras)")
        run.setToolTip("Triangulate the board corners from every camera and "
                       "measure reprojection error — objective, no assumptions "
                       "about the world origin or square size.")
        run.clicked.connect(self._verify_run)
        lv.addWidget(run)

        self.verify_report = QLabel("Not verified yet.")
        self.verify_report.setWordWrap(True)
        self.verify_report.setStyleSheet("font-family: monospace; font-size: 12px;")
        lv.addWidget(self.verify_report)

        cam_box = QGroupBox("Show camera")
        cb = QVBoxLayout(cam_box)
        self.verify_cam = QComboBox()
        self.verify_cam.currentIndexChanged.connect(self._verify_show)
        cb.addWidget(self.verify_cam)
        lv.addWidget(cam_box)

        self.verify_all = QCheckBox("Show all cameras (grid)")
        self.verify_all.setChecked(True)
        self.verify_all.setToolTip("See every camera at once to spot at a glance "
                                   "which one (if any) doesn't line up.")
        self.verify_all.toggled.connect(self._verify_show)
        lv.addWidget(self.verify_all)
        self.verify_undistort = QCheckBox("Show undistorted image")
        self.verify_undistort.setToolTip("Straight lines (door frames, tiles) should "
                                         "become straight if distortion is right.")
        self.verify_undistort.toggled.connect(self._verify_show)
        lv.addWidget(self.verify_undistort)
        self.verify_cube = QCheckBox("Also draw axes/cube at marked origin")
        self.verify_cube.setChecked(True)  # calibration usually passes; show it by default
        self.verify_cube.toggled.connect(self._verify_show)
        lv.addWidget(self.verify_cube)

        lv.addWidget(QLabel(
            "<small><b>Green</b> = your clicked/marked corners.<br>"
            "<b>Red</b> = the same corners triangulated from all cameras and "
            "reprojected. If green and red overlap, the calibration is consistent."
            "<br><br>Needs each camera's extrinsic frame + saved corners.</small>"))
        lv.addStretch(1)
        self.verify_status = QLabel(""); self.verify_status.setWordWrap(True)
        lv.addWidget(self.verify_status)
        root.addWidget(left)

        self.verify_view = _ZoomView()
        root.addWidget(self.verify_view, 1)
        self._verify_report_data = None   # consistency_report dict
        self._verify_cams = None          # calib per camera
        return w

    def _on_tab_changed(self, idx):
        w = self.tabs.widget(idx)
        if not self.state.project:
            return
        if w is self._verify_tab:
            self._verify_run()
        elif w is self._corners_tab and self.picker._pix_item is None:
            # picker is blank — reload the extrinsic frame in case it was captured
            # after this tab was first shown. (Only when blank, so we never wipe
            # corners you're in the middle of marking.)
            self._load_corner_image()

    def _verify_gather(self):
        """Per-camera (frame_path, corners_2d) from the project; None if incomplete."""
        import numpy as np
        from poseassess.core.calibration import load_clicked_points
        proj = self.state.project
        frames, corners = [], []
        for i in range(1, proj.config.num_cameras + 1):
            d = proj.extrinsic_cam_dir(i)
            jpg = next(iter(sorted(d.glob("*.jpg"))), None)
            # the user's own live hand-marked corners — no cached copy
            js = proj.extrinsic_points_file(i)
            if not js.exists():
                js = next(iter(sorted(d.glob("*.json"))), None)
            if jpg is None or js is None:
                return None, f"cam{i:02d}: missing extrinsic frame or corners"
            frames.append(jpg)
            corners.append(np.asarray(load_clicked_points(js), float))
        # all cameras must have the same number of corners to triangulate
        if len({len(c) for c in corners}) != 1:
            return None, "cameras have different corner counts — re-mark consistently"
        return (frames, corners), None

    def _verify_run(self):
        proj = self.state.project
        if not proj or self.verify_cam.count() == 0:
            return
        calib = next(iter(sorted(proj.calibration_dir.glob("Calib*.toml"))), None)
        if not calib:
            self.verify_report.setText("No Calib.toml — run calibration first.")
            return
        # --- THE cube source of truth: the project's Calib.toml, read fresh. --
        import os, datetime
        from poseassess.core.calib_check import read_calib_for_projection, consistency_report
        try:
            cams = read_calib_for_projection(calib)
        except Exception as e:  # noqa: BLE001
            self.verify_report.setText(f"Cannot read {calib.name}: {e}")
            return
        if not cams:
            self.verify_report.setText(f"{calib.name} has no camera poses.")
            return
        frames = [next(iter(sorted(proj.extrinsic_cam_dir(i).glob("*.jpg"))), None)
                  for i in range(1, len(cams) + 1)]
        self._verify_cams = cams
        self._verify_frames = frames
        mt = datetime.datetime.fromtimestamp(os.path.getmtime(calib)).strftime("%Y-%m-%d %H:%M:%S")

        # Corner QC (numeric verdict + green/red overlay). The live points.json
        # marks are first auto-aligned to the SAME canonical grid order the
        # Calib.toml poses were solved with (auto-detect returns different grid
        # traversals per camera when the board is seen rotated) — otherwise a
        # corner would triangulate against the wrong pose and green/red would fly
        # apart. Pixel positions are untouched; only the 2D<->3D correspondence is
        # resolved. The CUBE itself is drawn straight from Calib.toml regardless.
        rep, corners = None, None
        gathered, err = self._verify_gather()
        if gathered is not None and len(gathered[1]) == len(cams):
            try:
                from types import SimpleNamespace
                from poseassess.core.calibration.extrinsic import align_extrinsics
                cbn = proj.config.checkerboard
                intr = {i + 1: SimpleNamespace(K=cams[i]["K"], dist=cams[i]["dist"])
                        for i in range(len(cams))}
                _extr, aligned = align_extrinsics(
                    intr, {i + 1: gathered[1][i] for i in range(len(cams))},
                    cbn.corners_nb, cbn.square_size / 1000.0,
                    invert_z=getattr(cbn, "invert_z", False))
                corners = [aligned[i + 1] for i in range(len(cams))]
                rep = consistency_report(cams, corners)
            except Exception:  # noqa: BLE001
                rep, corners = None, None
        self._verify_report_data = rep
        self._verify_corners = corners

        head = f"<small>cube from <b>{calib.name}</b> · updated {mt}</small><br>"
        if rep is not None:
            color = {"GOOD": "#3ad35a", "CHECK": "#e0a94c", "FAILED": "#e05c5c"}[rep["verdict"]]
            msg = (head +
                f"<b style='color:{color}'>VERDICT: {rep['verdict']}</b><br>"
                f"reproj error: <b>{rep['mean_px']:.2f} px</b> (median {rep['median_px']:.2f}, "
                f"max {rep['max_px']:.2f})<br>"
                f"board flatness: {rep['planarity_mm']:.1f} mm<br>"
                f"corner spacing: {rep['spacing_mm']:.1f} ± {rep['spacing_std_mm']:.1f} mm<br>"
                f"<small>&lt;1 px = excellent · &lt;3 px = good</small>")
        else:
            msg = head + ("<small>Cube drawn from the calibration poses. Corner "
                          "overlay/verdict unavailable"
                          + (f" ({err})" if err else "") + ".</small>")
        self.verify_report.setText(msg)
        self._verify_show()

    def _render_cam(self, idx: int):
        """Draw the calibration cube (straight from Calib.toml) on this frame."""
        import cv2
        import numpy as np
        cam = self._verify_cams[idx]
        frame = self._verify_frames[idx]
        img = cv2.imread(str(frame)) if frame else None
        if img is None:
            img = np.zeros((720, 1280, 3), np.uint8)
            cv2.putText(img, f"cam{idx+1}: no extrinsic frame", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (60, 160, 240), 2, cv2.LINE_AA)
            return img
        if self.verify_undistort.isChecked():
            from poseassess.core.calib_check import undistort
            return undistort(img, cam)
        from poseassess.core.calib_check import draw_overlay, reproject, world_unit_scale
        # Cube = XYZ axes + wireframe cube at the world origin (our calibration
        # puts it at board corner 0), projected with THIS camera's stored
        # K/dist/R/T from Calib.toml. A pure function of the calibration file — it
        # updates whenever Calib.toml changes, nothing cached.
        if self.verify_cube.isChecked():
            sq_mm = self.state.project.config.checkerboard.square_size
            sq = sq_mm / 1000.0 if world_unit_scale(self._verify_cams) == "m" else sq_mm
            img = draw_overlay(img, cam, square_size_world=sq, n_squares=20, draw_cube=True)
        rep = self._verify_report_data
        if rep is not None and self._verify_corners is not None:
            for (gx, gy) in self._verify_corners[idx]:          # green = marked
                cv2.circle(img, (int(gx), int(gy)), 7, (0, 220, 0), 2, cv2.LINE_AA)
            for (rx, ry) in reproject(cam, rep["points_3d"]):   # red = reprojected
                cv2.drawMarker(img, (int(rx), int(ry)), (0, 0, 255),
                               cv2.MARKER_CROSS, 10, 2, cv2.LINE_AA)
        return img

    def _verify_show(self, *_):
        import cv2, numpy as np
        proj = self.state.project
        if not proj or self._verify_cams is None:
            return
        rep = self._verify_report_data
        n = len(self._verify_cams)
        if self.verify_all.isChecked():
            # 3-column grid of every camera, labelled with its reproj error
            cells = []
            for i in range(n):
                c = cv2.resize(self._render_cam(i), (640, 360))
                label = f"cam{i+1}"
                if rep is not None:
                    label += f"  {rep['per_cam_px'][i]:.2f}px"
                    col = (90, 220, 90) if rep["per_cam_px"][i] < 3.0 else (60, 160, 240)
                else:
                    col = (210, 210, 210)
                cv2.putText(c, label, (16, 44),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, col, 2, cv2.LINE_AA)
                cells.append(c)
            while len(cells) % 3 != 0:
                cells.append(np.zeros((360, 640, 3), np.uint8))
            rows = [np.hstack(cells[r:r + 3]) for r in range(0, len(cells), 3)]
            self.verify_view.set_bgr(np.vstack(rows))
            self.verify_status.setText(
                f"all {n} cameras" + (f" · overall {rep['mean_px']:.2f} px" if rep else ""))
        else:
            idx = min(self.verify_cam.currentIndex(), n - 1)
            self.verify_view.set_bgr(self._render_cam(idx))
            self.verify_status.setText(
                f"{proj.cam_name(idx + 1)}" +
                (f" · reproj {rep['per_cam_px'][idx]:.2f} px" if rep else ""))

    # ================= shared helpers ====================================== #
    def _cur_cam_frames(self) -> int:
        return self.cam_combo.currentIndex() + 1

    def _is_intrinsic(self) -> bool:
        return self.rb_intri.isChecked()

    def _cam_folder(self, cam: int) -> Path:
        p = self.state.project
        return p.intrinsic_cam_dir(cam) if self._is_intrinsic() else p.extrinsic_cam_dir(cam)

    def _find_video(self, folder: Path):
        for pat in VIDEO_EXTS:
            hits = sorted(folder.glob(pat))
            if hits:
                return hits[0]
        return None

    def _expected_corners(self) -> int:
        p = self.state.project
        if not p:
            return 0
        r, c = p.config.checkerboard.corners_nb
        return int(r) * int(c)

    # ================= project change ====================================== #
    def on_project_changed(self, project):
        combos = (self.cam_combo, self.corner_cam, self.verify_cam)
        for combo in combos:
            combo.blockSignals(True); combo.clear()
        if project:
            for i in range(1, project.config.num_cameras + 1):
                self.cam_combo.addItem(project.cam_name(i))
                self.corner_cam.addItem(project.cam_name(i))
                self.verify_cam.addItem(project.cam_name(i))
            self.selector.set_board(project.config.checkerboard.corners_nb)
            self.invert_z_chk.blockSignals(True)
            self.invert_z_chk.setChecked(
                getattr(project.config.checkerboard, "invert_z", False))
            self.invert_z_chk.blockSignals(False)
        for combo in combos:
            combo.blockSignals(False)
        self._verify_report_data = None
        self.verify_report.setText("Not verified yet. Click 'Verify calibration'.")
        self.verify_view.clear()
        self._reload_video()
        self._load_corner_image()

    # ================= Tab 1 logic ========================================= #
    def _reload_video(self, *_):
        proj = self.state.project
        if not proj or self.cam_combo.count() == 0:
            return
        cam = self._cur_cam_frames()
        folder = self._cam_folder(cam)
        self.selector.set_output_dir(folder)
        self.selector.set_board(proj.config.checkerboard.corners_nb)
        vid = self._find_video(folder)
        mode = "intrinsic" if self._is_intrinsic() else "extrinsic"
        if vid:
            self.selector.load_video(vid)
            self.info.setText(f"{proj.cam_name(cam)} · {mode}\nvideo: {vid.name}")
        else:
            self.selector.close_video()
            self.selector.view.setText(f"No video in\n{folder}\n\nUse “Load video manually…”.")
            self.info.setText(f"{proj.cam_name(cam)} · {mode}\nno video in folder")
        self._refresh_captured()

    def _auto_extract_intrinsic(self):
        proj = self.state.project
        if not proj or self.cam_combo.count() == 0:
            return
        cam = self._cur_cam_frames()
        # Auto-extract always writes into the intrinsic folder. Source video:
        # prefer the one you loaded (via "Load video manually…" in Intrinsic
        # mode), else a video already sitting in the intrinsic folder.
        intr_dir = proj.intrinsic_cam_dir(cam)
        video = None
        if self.rb_intri.isChecked():
            video = self.selector.current_video()
        if not video:
            video = self._find_video(intr_dir)
        if not video:
            QMessageBox.warning(
                self, "Auto-extract",
                f"No intrinsic video for {proj.cam_name(cam)}.\n\n"
                f"Switch to Intrinsic mode and use “Load video manually…” to "
                f"pick this camera's checkerboard-waving clip first.")
            return
        self.auto_extract_btn.setEnabled(False)
        self.info.setText(f"scanning {video.name} for board frames…")
        self._ex_worker = _ExtractWorker(
            video, intr_dir, proj.config.checkerboard.corners_nb, max_frames=20, step=15)
        self._ex_worker.log.connect(lambda m: self.info.setText(m))
        self._ex_worker.done.connect(self._on_extract_done)
        self._ex_thread = QThread()
        self._ex_worker.moveToThread(self._ex_thread)
        self._ex_thread.started.connect(self._ex_worker.run)
        self._ex_worker.done.connect(self._ex_thread.quit)
        self._ex_thread.finished.connect(self._ex_thread.deleteLater)
        self._ex_thread.start()

    def _on_extract_done(self, n):
        self.auto_extract_btn.setEnabled(True)
        self.info.setText(f"extracted {n} intrinsic frames.")
        if self.rb_intri.isChecked():
            self._refresh_captured()

    def _browse_video(self):
        if not self.state.project:
            return
        src, _ = QFileDialog.getOpenFileName(self, "Select video", "",
                                             "Videos (*.mp4 *.MP4 *.avi *.mov *.mkv)")
        if src:
            self.selector.set_output_dir(self._cam_folder(self._cur_cam_frames()))
            self.selector.load_video(Path(src))

    def _on_captured(self, index, path):
        self.info.setText(f"captured frame {index}\n→ {Path(path).name}")
        self._refresh_captured()

    def _refresh_captured(self):
        self.captured.clear()
        proj = self.state.project
        if not proj or self.cam_combo.count() == 0:
            return
        for img in sorted(self._cam_folder(self._cur_cam_frames()).glob("*.jpg")):
            self.captured.addItem(img.name)

    def _clear_captured(self):
        if not self.state.project or self.cam_combo.count() == 0:
            return
        for img in self._cam_folder(self._cur_cam_frames()).glob("*.jpg"):
            img.unlink()
        self._refresh_captured()

    # ================= Tab 2 logic ========================================= #
    def _load_corner_image(self, *_):
        proj = self.state.project
        if not proj or self.corner_cam.count() == 0:
            return
        cam = self.corner_cam.currentIndex() + 1
        folder = proj.extrinsic_cam_dir(cam)
        jpgs = sorted(folder.glob("*.jpg"))
        if not jpgs:
            self.picker._scene.clear()
            self.corner_count.setText("no captured extrinsic frame — capture one first")
            return
        self.picker.load_image(jpgs[0])
        self.picker.set_grid_shape(proj.config.checkerboard.corners_nb)
        # if points already saved, show them
        pts_file = proj.extrinsic_points_file(cam)
        if pts_file.exists():
            from poseassess.core.calibration import load_clicked_points
            self.picker.set_points([tuple(p) for p in load_clicked_points(pts_file)])
        self._on_points_changed(len(self.picker.points()))

    def _auto_detect(self):
        proj = self.state.project
        if not proj:
            return
        ok = self.picker.auto_detect(proj.config.checkerboard.corners_nb)
        if ok:
            self.picker.set_grid_shape(proj.config.checkerboard.corners_nb)
        else:
            QMessageBox.information(
                self, "Auto-detect",
                "Board not auto-detected on this frame. Click the corners "
                "manually (a different reference frame may detect better).")

    def _rotate_corners(self, clockwise: bool):
        proj = self.state.project
        if proj:
            # (re)assert the grid shape from the board config before rotating
            if not hasattr(self.picker, "_grid_rows"):
                self.picker.set_grid_shape(proj.config.checkerboard.corners_nb)
        if not self.picker.rotate_90(clockwise):
            QMessageBox.information(
                self, "Rotate 90°",
                f"Rotation needs the full board grid ({self._expected_corners()} "
                "points). Auto-detect or finish marking all corners first.")

    def _on_points_changed(self, n):
        self.corner_count.setText(f"points: {n} / {self._expected_corners()}")

    def _save_points(self):
        proj = self.state.project
        if not proj:
            return
        cam = self.corner_cam.currentIndex() + 1
        pts = self.picker.points()
        expected = self._expected_corners()
        if len(pts) != expected:
            QMessageBox.warning(
                self, "Save corners",
                f"Have {len(pts)} points but board needs {expected}. "
                f"Add/remove points to match before saving.")
            return
        save_clicked_points(proj.extrinsic_points_file(cam), pts)
        self.calib_log.appendPlainText(f"saved {len(pts)} corners for {proj.cam_name(cam)}")

    def _import_calibration(self):
        proj = self.state.project
        if not proj:
            return
        files, _ = QFileDialog.getOpenFileNames(
            self, "Import calibration files", "",
            "Calibration (*.yml *.toml);;All files (*.*)")
        if not files:
            return
        import shutil
        proj.calibration_dir.mkdir(parents=True, exist_ok=True)
        copied = []
        for f in files:
            src = Path(f)
            # normalize common names so downstream stages find them
            name = src.name
            low = name.lower()
            if low.endswith(".toml"):
                dst = proj.calibration_dir / ("Calib.toml" if "calib" in low else name)
            elif "intri" in low:
                dst = proj.intri_yml
            elif "extri" in low:
                dst = proj.extri_yml
            else:
                dst = proj.calibration_dir / name
            shutil.copy(src, dst)
            copied.append(dst.name)
        self.calib_log.appendPlainText(f"imported: {', '.join(copied)}")
        QMessageBox.information(self, "Import", f"Imported: {', '.join(copied)}")

    def _toggle_invert_z(self, on):
        proj = self.state.project
        if not proj:
            return
        proj.config.checkerboard.invert_z = bool(on)
        try:
            proj.save()
        except Exception as e:  # noqa: BLE001
            self.calib_log.appendPlainText(f"could not save project: {e}")
        # apply instantly if a calibration already exists: re-solve extrinsics
        # only (fast, no intrinsic recompute) and patch Calib.toml, then refresh
        # the Verify cube. If there's no Calib.toml yet it just takes effect on
        # the next full Run calibration.
        try:
            from poseassess.core.calibration.orchestrator import resolve_extrinsics_only
            resolve_extrinsics_only(proj, progress=self.calib_log.appendPlainText)
            self._verify_run()
        except FileNotFoundError:
            self.calib_log.appendPlainText(
                "Flip Z will apply on the next Run calibration.")
        except Exception as e:  # noqa: BLE001
            self.calib_log.appendPlainText(f"Flip Z: {e}")

    def _run_calibration(self):
        proj = self.state.project
        if not proj:
            return
        self.calib_log.appendPlainText("=== running calibration ===")
        self._worker = _CalibWorker(proj)
        self._worker.log.connect(self.calib_log.appendPlainText)
        self._worker.done.connect(self._calib_done)
        self._thread = QThread()
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.done.connect(self._thread.quit)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.start()

    def _calib_done(self, ok, msg):
        self.calib_log.appendPlainText(("OK: " if ok else "FAILED: ") + msg)
        if ok:
            # the calibration/ folder just changed on disk — refresh Verify from
            # it so the cube/verdict reflect the new Calib.toml immediately.
            self._verify_run()
            QMessageBox.information(self, "Calibration", msg)
        else:
            QMessageBox.critical(self, "Calibration failed", msg)
