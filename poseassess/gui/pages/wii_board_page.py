"""Page "2b. Wii Board": the Wii Balance Board device and its position in the camera world.

Tab "Device && live COP" (works without a project):
  auto-connect (plug-and-play, on by default), device list + Scan / Connect / Use simulator /
  Disconnect, Tare (blocked while recording), COP min. load, status (battery, connections,
  last error, hidapi / pairing hints), live ``CopView`` (3 s COP trail, total kg, % of body
  mass when known).

Tab "Board location" (needs a project; the same idea as clicking the checkerboard corners):
  1. per camera pick a frame where the board is visible (``wii/board/camNN/*.jpg``: grab one
     from ``videos/camNN.*`` or another video, copy the calibration frame, or load an image);
  2. click TL, TR, BR, BL, C (``BoardPointPicker``) and save them (``board.save_clicks``);
  3. "Compute board position" (``board.compute_board``): result text + QC grid of every
     camera (``overlay.board_check_image``), stale banner after a recalibration;
  4. corner check with the connected board: the operator presses the corner that was CLICKED
     as TL (marked "PRESS" in the check view; not found via the power button, which would
     make the check meaningless). TL sensor = ok, BR = front/back swapped -> fixed
     automatically, TR/BL = redo the clicks;
  5. board geometry (top-surface height, dimensions).

Every change of ``wii/board/board.json`` is followed by ``state.notify_balance_changed("board")``.
"""

from __future__ import annotations

import logging
import shutil
import time
from pathlib import Path

import numpy as np
from PySide6.QtCore import QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QFont, QImage, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGraphicsScene,
    QGraphicsView,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..widgets.board_picker import COLORS, HINTS, LABELS, BoardPointPicker
from ..widgets.cop_view import CopView
from .base_page import BasePage

log = logging.getLogger(__name__)

GOOD, WARN, BAD, ACCENT = "#3ad35a", "#e0a94c", "#e05c5c", "#3d8bfd"
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")
VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv")
LIVE_INTERVAL_MS = 50  # live COP / status refresh while the page is visible
COP_TRAIL_S = 3.0
CORNER_POLL_MS = 100
CORNER_CHECK_TIMEOUT_S = 15.0
CORNER_BASELINE_S = 0.5
CORNER_WINDOW_S = 0.3
CORNER_MIN_RISE_KG = 3.0
QC_CELL = (640, 360)
LEFT_WIDTH = 260

CORNER_NAMES = {"TL": "front-left (edge opposite the power button, left)",
                "TR": "front-right (edge opposite the power button, right)",
                "BL": "back-left (power-button edge, left)",
                "BR": "back-right (power-button edge, right)"}
CLICK_HELP = ("<small><b>Order:</b> 1 TL front-left → 2 TR front-right → 3 BR back-right → "
              "4 BL back-left → 5 C centre.<br>"
              "<b>Front</b> = the long edge OPPOSITE the power button (TL/TR sensors); the "
              "subject stands facing it. Left/right are the <i>subject's</i>, not the image's."
              "<br>Left-click: add · drag a point: move · right-click: undo last · wheel: zoom "
              "· drag elsewhere: pan.<br>Clicking the board in 2+ cameras triangulates it "
              "(more accurate); one camera uses PnP.</small>")
CORNER_TIP = ("Press the corner you clicked as TL (marked in the check view), found from the "
              "camera images, not from the power button: the sensor that responds shows "
              "whether the clicks match the board, and a front/back swap is fixed by itself.")
AUTO_TIP = ("Connect as soon as a paired board is switched on and reconnect automatically "
            "after a Bluetooth drop. The tare of each board is remembered while PoseAssess is "
            "running (tare again after a restart).")
SIM_TIP = "Synthetic data of a 70 kg person swaying (no hardware needed)."
DISCONNECT_TIP = "Release the board and switch Auto-connect off."
MIN_KG_TIP = ("Below this total load the centre of pressure is undefined (nobody on the "
              "board).")
TARE_TIP = ("Zero the four sensors with nobody on the board. The tare of each board is "
            "remembered while PoseAssess is running (also across reconnects); tare again at the "
            "start of every session. Not possible while recording.")
PAIRING_HELP = ("<small>Pair the board once via Bluetooth (Windows: Settings &gt; Bluetooth &gt; "
                "Add device, press the red <b>SYNC</b> button in the battery compartment, leave "
                "the PIN empty). With Auto-connect on, it connects as soon as it is switched on "
                "and reconnects after a drop. The board is optional.</small>")
HIDAPI_HELP = ("<small><b style='color:#e05c5c'>hidapi is not available</b> — the Balance "
               "Board cannot be read. Install it into the bundle: <tt>venv\\Scripts\\python -m "
               "pip install hidapi</tt>, then restart. <i>Use simulator</i> still works.</small>")


def _bgr_to_pixmap(img: np.ndarray) -> QPixmap:
    h, w = img.shape[:2]
    rgb = np.ascontiguousarray(img[:, :, ::-1])
    qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())


class _ImageView(QGraphicsView):
    """Read-only image view with wheel zoom and drag pan (QC grid)."""

    def __init__(self):
        super().__init__()
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setStyleSheet("background:#222;")
        self._has_image = False
        self._user_zoom = False

    def set_bgr(self, img: np.ndarray) -> None:
        pix = _bgr_to_pixmap(img)
        self._scene.clear()
        self._scene.addPixmap(pix)
        self._scene.setSceneRect(QRectF(pix.rect()))
        self._has_image = True
        self.fit()

    def fit(self) -> None:
        self._user_zoom = False
        if self._has_image:
            self.resetTransform()
            self.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio)

    def clear(self) -> None:
        self._scene.clear()
        self._has_image = False

    def has_image(self) -> bool:
        return self._has_image

    def wheelEvent(self, ev):
        self._user_zoom = True
        f = 1.25 if ev.angleDelta().y() > 0 else 0.8
        self.scale(f, f)

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        if not self._user_zoom:
            self.fit()

    def showEvent(self, ev):
        super().showEvent(ev)
        if not self._user_zoom:
            self.fit()

    def mouseDoubleClickEvent(self, ev):  # double-click: whole grid again
        self.fit()


class _GrabFrameDialog(QDialog):
    """Scrub a video (``FrameSelector``) and save one frame into ``out_dir``."""

    frame_captured = Signal(int, str, str)  # (frame index, image path, video path)

    def __init__(self, parent, video: Path | None, out_dir: Path, title: str):
        super().__init__(parent)
        from ..widgets.frame_selector import FrameSelector

        self.setWindowTitle(title)
        self.resize(960, 640)
        self.video: Path | None = None
        v = QVBoxLayout(self)
        self.info = QLabel("")
        self.info.setWordWrap(True)
        v.addWidget(self.info)
        self.selector = FrameSelector()
        self.selector.set_board(None)
        self.selector.detect_cb.hide()
        self.selector.detect_status.hide()
        self.selector.capture_btn.setText("📸 Use this frame")
        self.selector.set_output_dir(out_dir)
        self.selector.frame_captured.connect(self._on_captured)
        v.addWidget(self.selector, 1)
        row = QHBoxLayout()
        other = QPushButton("Open another video…")
        other.clicked.connect(self._open_other)
        row.addWidget(other)
        row.addStretch(1)
        close = QPushButton("Close")
        close.clicked.connect(self.reject)
        row.addWidget(close)
        v.addLayout(row)
        if video is not None:
            self.load(video)
        else:
            self.info.setText("Open a video in which the board is visible.")

    def load(self, video: Path) -> bool:
        ok = self.selector.load_video(Path(video))
        if ok:
            self.video = Path(video)
            self.info.setText(f"<b>{self.video.name}</b>: find a frame where the whole board is "
                              "visible and nobody stands on it, then click “Use this frame”.")
        return ok

    def _open_other(self):
        f, _ = QFileDialog.getOpenFileName(self, "Select video", "",
                                           "Videos (*.mp4 *.MP4 *.avi *.mov *.mkv)")
        if f:
            self.load(Path(f))

    def _on_captured(self, index: int, path: str):
        self.frame_captured.emit(int(index), str(path), str(self.video or ""))
        self.accept()

    def done(self, r):
        self.selector.close_video()
        super().done(r)


def _scroll(widget: QWidget, width: int = LEFT_WIDTH) -> QScrollArea:
    """Fixed-width, frameless vertical scroll area around a control column (keeps the page's
    minimum height small on 680 px high windows)."""
    sa = QScrollArea()
    sa.setWidget(widget)
    sa.setWidgetResizable(True)
    sa.setFrameShape(QFrame.NoFrame)
    sa.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    sa.setFixedWidth(width + 14)
    # same background as the tab page (the viewport would otherwise paint its own)
    sa.viewport().setAutoFillBackground(False)
    widget.setAutoFillBackground(False)
    sa.setStyleSheet("QScrollArea { background: transparent; }")
    return sa


def _small(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setWordWrap(True)
    lbl.setTextFormat(Qt.RichText)
    return lbl


class WiiBoardPage(BasePage):
    nav_title = "2b. Wii Board"
    needs_project = False

    def __init__(self, state):
        super().__init__(state)
        self._wii = state.wii
        self._project = None          # project shown (the clicks below belong to it)
        self._cams: dict = {}         # camNN -> CameraCalibration (Calib.toml)
        self._board = None            # BoardRegistration | None
        self._qc_dirty = True
        self._dirty = False           # picker points differ from points.json
        self._loading = False
        self._cam_index = 0           # 1-based camera shown in the picker (0: none)
        self._image: Path | None = None
        self._image_sources: dict[str, str] = {}
        self._corner: dict | None = None
        self._body_mass: float | None = None
        self._emitting = False
        self._grab_dlg = None
        self._last_detail = ""
        self._flash_msg: str | None = None
        self._flash_prev = ""

        root = QVBoxLayout(self)
        self.tabs = QTabWidget()
        self.device_tab = self._build_device_tab()
        self.board_tab = self._build_board_tab()
        self.tabs.addTab(self.device_tab, "Device && live COP")
        self.tabs.addTab(self.board_tab, "Board location")
        # Replay of a recording file (original Wii program / wii.csv): no board needed
        from ..widgets.wii_replay import WiiReplayWidget
        self.replay_tab = WiiReplayWidget()
        self.tabs.addTab(self.replay_tab, "Replay file")
        root.addWidget(self.tabs)

        self._live = QTimer(self)
        self._live.setInterval(LIVE_INTERVAL_MS)
        self._live.timeout.connect(self._tick_live)
        self._corner_timer = QTimer(self)
        self._corner_timer.setInterval(CORNER_POLL_MS)
        self._corner_timer.timeout.connect(self._tick_corner)

        c = self._wii
        for sig in (c.source_changed, c.state_changed, c.tare_lock_changed, c.tare_changed):
            sig.connect(self._on_wii_signal)
        state.balance_changed.connect(self._on_balance_changed)
        self.min_kg.setValue(c.min_load)
        self._sync_device_controls()
        self.on_project_changed(state.project)

    # ================================================================ UI
    def _build_device_tab(self) -> QWidget:
        w = QWidget()
        root = QHBoxLayout(w)
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 4, 0)

        g = QGroupBox("Balance Board")
        gl = QVBoxLayout(g)
        self.chk_auto = QCheckBox("Auto-connect (plug and play)")
        self.chk_auto.setToolTip(AUTO_TIP)
        self.chk_auto.toggled.connect(self._toggle_auto)
        gl.addWidget(self.chk_auto)
        row = QHBoxLayout()
        self.dev_combo = QComboBox()
        self.dev_combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.dev_combo.setMinimumContentsLength(8)
        self.dev_combo.setPlaceholderText("first board found")
        self.dev_combo.setToolTip("Balance Boards found by the last Scan (empty: the first "
                                  "board found is used).")
        row.addWidget(self.dev_combo, 1)
        self.btn_scan = QPushButton("Scan")
        self.btn_scan.setToolTip("List the paired Balance Boards (HID devices).")
        self.btn_scan.clicked.connect(self._scan)
        row.addWidget(self.btn_scan)
        gl.addLayout(row)
        grid = QGridLayout()
        self.btn_connect = QPushButton("Connect")
        self.btn_connect.setToolTip("Connect the board selected after Scan (or the first one "
                                    "found). With Auto-connect ticked it keeps reconnecting to "
                                    "that board.")
        self.btn_connect.clicked.connect(self._connect)
        self.btn_sim = QPushButton("Use simulator")
        self.btn_sim.setToolTip(SIM_TIP)
        self.btn_sim.clicked.connect(self._use_simulator)
        self.btn_disconnect = QPushButton("Disconnect")
        self.btn_disconnect.setToolTip(DISCONNECT_TIP)
        self.btn_disconnect.clicked.connect(self._disconnect)
        grid.addWidget(self.btn_connect, 0, 0)
        grid.addWidget(self.btn_sim, 0, 1)
        grid.addWidget(self.btn_disconnect, 1, 0, 1, 2)
        gl.addLayout(grid)
        self.scan_lbl = _small("")
        self.scan_lbl.hide()
        gl.addWidget(self.scan_lbl)
        lv.addWidget(g)

        g = QGroupBox("Zero and COP")
        gl = QVBoxLayout(g)
        self.btn_tare = QPushButton("Tare (board empty)")
        self.btn_tare.clicked.connect(self._tare)
        gl.addWidget(self.btn_tare)
        form = QFormLayout()
        self.min_kg = QDoubleSpinBox()
        self.min_kg.setRange(0.0, 50.0)
        self.min_kg.setDecimals(1)
        self.min_kg.setSuffix(" kg")
        self.min_kg.setToolTip(MIN_KG_TIP)
        self.min_kg.valueChanged.connect(self._set_min_load)
        form.addRow("COP min. load", self.min_kg)
        gl.addLayout(form)
        lv.addWidget(g)

        self.hint_lbl = _small(PAIRING_HELP)
        lv.addWidget(self.hint_lbl)
        self.status_lbl = QLabel("")
        self.status_lbl.setWordWrap(True)
        self.status_lbl.setStyleSheet("font-family: monospace; font-size: 12px;")
        self.status_lbl.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.status_lbl.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        lv.addWidget(self.status_lbl)
        lv.addStretch(1)
        root.addWidget(_scroll(left))

        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)
        self.cop_view = CopView()
        rv.addWidget(self.cop_view, 1)
        rv.addWidget(_small("<small>Live top view of the board (front = TL/TR edge at the top): "
                            f"centre of pressure of the last {COP_TRAIL_S:g} s. The same COP "
                            "is recorded on the Capture page and shown in the 3D View.</small>"))
        root.addWidget(right, 1)
        return w

    def _build_board_tab(self) -> QWidget:
        w = QWidget()
        root = QHBoxLayout(w)

        # ---- left column: camera, frame, clicks
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 4, 0)
        g = QGroupBox("1. Camera and frame")
        gl = QVBoxLayout(g)
        self.cam_combo = QComboBox()
        self.cam_combo.currentIndexChanged.connect(self._on_camera_changed)
        gl.addWidget(self.cam_combo)
        self.frame_list = QListWidget()
        self.frame_list.setMaximumHeight(96)
        self.frame_list.setToolTip("Frames of this camera in wii/board/camNN (the board must be "
                                   "visible; nobody needs to stand on it).")
        self.frame_list.currentItemChanged.connect(self._on_frame_selected)
        gl.addWidget(self.frame_list)
        self.btn_grab = QPushButton("Grab frame from video…")
        self.btn_grab.setToolTip("Scrub this camera's trial video (videos/camNN) or another "
                                 "video of the same camera and save one frame.")
        self.btn_grab.clicked.connect(self._grab_from_video)
        gl.addWidget(self.btn_grab)
        self.btn_calib_frame = QPushButton("Use calibration frame")
        self.btn_calib_frame.setToolTip("Copy the extrinsic (checkerboard) frame of this camera: "
                                        "useful when the board was already in place during the "
                                        "calibration.")
        self.btn_calib_frame.clicked.connect(self._use_calibration_frame)
        gl.addWidget(self.btn_calib_frame)
        self.btn_load_img = QPushButton("Load image…")
        self.btn_load_img.setToolTip("Copy an image of this camera (same resolution as its "
                                     "calibration) into the project.")
        self.btn_load_img.clicked.connect(self._load_image_file)
        gl.addWidget(self.btn_load_img)
        self.frame_info = _small("")
        gl.addWidget(self.frame_info)
        lv.addWidget(g)

        g = QGroupBox("2. Click TL, TR, BR, BL, C")
        gl = QVBoxLayout(g)
        self.click_hint = _small("")
        self.click_hint.setStyleSheet("font-weight: 600;")
        gl.addWidget(self.click_hint)
        row = QHBoxLayout()
        self.btn_undo = QPushButton("Undo last")
        self.btn_undo.setToolTip("Remove the last point (same as a right-click).")
        self.btn_undo.clicked.connect(lambda: self.picker.undo_last())
        self.btn_clear = QPushButton("Clear")
        self.btn_clear.setToolTip("Remove all points of this camera.")
        self.btn_clear.clicked.connect(self._clear_clicks)
        row.addWidget(self.btn_undo)
        row.addWidget(self.btn_clear)
        gl.addLayout(row)
        self.btn_swap = QPushButton("↺ Swap front/back (180°)")
        self.btn_swap.setToolTip("Exchange TL↔BR and TR↔BL of this camera: use when you clicked "
                                 "the power-button edge as the front.")
        self.btn_swap.clicked.connect(self._swap_front_back)
        gl.addWidget(self.btn_swap)
        self.btn_save = QPushButton("Save clicks for this camera")
        self.btn_save.setToolTip("Write wii/board/camNN/points.json (clicks are also saved "
                                 "automatically when you switch camera, frame or page).")
        self.btn_save.clicked.connect(lambda: self._save_clicks(quiet=False))
        gl.addWidget(self.btn_save)
        self.clicks_status = _small("")
        gl.addWidget(self.clicks_status)
        gl.addWidget(_small(CLICK_HELP))
        lv.addWidget(g)
        lv.addStretch(1)
        root.addWidget(_scroll(left))

        # ---- centre: picker / QC grid
        self.view_tabs = QTabWidget()
        self.picker = BoardPointPicker()
        self.picker.points_changed.connect(self._on_points_changed)
        self.view_tabs.addTab(self.picker, "Click the board")
        qc = QWidget()
        qv = QVBoxLayout(qc)
        qv.setContentsMargins(0, 2, 0, 0)
        qrow = QHBoxLayout()
        self.chk_qc_zoom = QCheckBox("Zoom to the board")
        self.chk_qc_zoom.setChecked(True)
        self.chk_qc_zoom.setToolTip("Show the region around the board in every camera (untick "
                                    "for the whole frames).")
        self.chk_qc_zoom.toggled.connect(lambda _on: self._build_qc())
        qrow.addWidget(self.chk_qc_zoom)
        qrow.addWidget(_small("<small>Cyan outline = computed board (orange: front edge TL-TR),"
                              " dots = your clicks. They should coincide.</small>"), 1)
        qv.addLayout(qrow)
        self.qc_view = _ImageView()
        qv.addWidget(self.qc_view, 1)
        self.qc_tab = qc
        self.view_tabs.addTab(qc, "Check (all cameras)")
        self.view_tabs.currentChanged.connect(self._on_view_tab)
        root.addWidget(self.view_tabs, 1)

        # ---- right column: result, corner check, geometry
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(4, 0, 0, 0)
        g = QGroupBox("3. Board position")
        gl = QVBoxLayout(g)
        self.btn_compute = QPushButton("✓ Compute board position")
        self.btn_compute.setToolTip("Locate the board in the Calib.toml world from every camera "
                                    "with 5 saved clicks (2+ cameras: triangulation, 1 camera: "
                                    "PnP), fitted flat on the floor.")
        self.btn_compute.clicked.connect(self._compute_board)
        gl.addWidget(self.btn_compute)
        self.stale_lbl = _small("")
        self.stale_lbl.setStyleSheet(f"color: {BAD}; font-weight: 600;")
        self.stale_lbl.hide()
        gl.addWidget(self.stale_lbl)
        self.btn_recompute = QPushButton("Recompute")
        self.btn_recompute.setToolTip("Compute the board position again with the current "
                                      "calibration and the saved clicks.")
        self.btn_recompute.clicked.connect(self._compute_board)
        self.btn_recompute.hide()
        gl.addWidget(self.btn_recompute)
        self.result_txt = QPlainTextEdit()
        self.result_txt.setReadOnly(True)
        self.result_txt.setFont(QFont("monospace", 9))
        self.result_txt.setLineWrapMode(QPlainTextEdit.WidgetWidth)
        self.result_txt.setMinimumHeight(120)
        gl.addWidget(self.result_txt, 1)
        rv.addWidget(g, 1)

        g = QGroupBox("4. Corner check (live board)")
        gl = QVBoxLayout(g)
        self.btn_corner = QPushButton("▶ Start corner check")
        self.btn_corner.setToolTip(CORNER_TIP)
        self.btn_corner.clicked.connect(self._corner_button)
        gl.addWidget(self.btn_corner)
        self.corner_lbl = _small("<small>Needs a computed board position and the connected "
                                 "Balance Board.</small>")
        gl.addWidget(self.corner_lbl)
        rv.addWidget(g)

        g = QGroupBox("Board geometry")
        gl = QVBoxLayout(g)
        form = QFormLayout()
        self.geo_spins: dict[str, QDoubleSpinBox] = {}
        for key, label, lo, tip in (
                ("height_mm", "Top height", 0.0,
                 "Height of the standing surface above the checkerboard plane (floor). Subtract "
                 "the checkerboard's own thickness if it lay on a mat."),
                ("length_mm", "Length (L-R)", 10.0, "Outer length, left-right."),
                ("width_mm", "Width (F-B)", 10.0, "Outer width, front-back."),
                ("sensor_dx_mm", "Sensors L-R", 10.0, "Sensor centre spacing, left-right."),
                ("sensor_dy_mm", "Sensors F-B", 10.0, "Sensor centre spacing, front-back.")):
            s = QDoubleSpinBox()
            s.setRange(lo, 2000.0)
            s.setDecimals(1)
            s.setSuffix(" mm")
            s.setToolTip(tip)
            self.geo_spins[key] = s
            form.addRow(label, s)
        gl.addLayout(form)
        self._geo_advanced_rows = [k for k in self.geo_spins if k != "height_mm"]
        self.chk_geo_adv = QCheckBox("Show board dimensions")
        self.chk_geo_adv.setToolTip("The defaults are the official Wii Balance Board sizes "
                                    "(511 x 316 mm, sensors 433 x 238 mm).")
        self.chk_geo_adv.toggled.connect(self._show_geo_advanced)
        gl.addWidget(self.chk_geo_adv)
        self._geo_form = form
        self.btn_geo = QPushButton("Apply geometry")
        self.btn_geo.setToolTip("Use these dimensions (recomputes the board position when one "
                                "was computed).")
        self.btn_geo.clicked.connect(self._apply_geometry)
        gl.addWidget(self.btn_geo)
        rv.addWidget(g)
        root.addWidget(_scroll(right))
        self._show_geo_advanced(False)
        self._set_geometry_controls(None)
        return w

    # ============================================================ helpers
    def _warn(self, title: str, text: str) -> None:
        QMessageBox.warning(self, title, text)

    def _info(self, title: str, text: str) -> None:
        QMessageBox.information(self, title, text)

    def _status_bar(self):
        sb = getattr(self.window(), "statusBar", None)
        if not callable(sb):
            return None
        try:
            return sb()
        except Exception:  # noqa: BLE001
            return None

    def _flash(self, msg: str, ms: int = 6000) -> None:
        """Show ``msg`` in the main window's status bar for ``ms``, then restore the previous
        (project) message: the main window never re-shows it by itself."""
        bar = self._status_bar()
        if bar is None:
            return
        cur = bar.currentMessage()
        if cur != self._flash_msg:
            self._flash_prev = cur
        self._flash_msg = msg
        bar.showMessage(msg)
        QTimer.singleShot(ms, self, self._flash_restore)

    def _flash_restore(self) -> None:
        bar = self._status_bar()
        if bar is not None and self._flash_msg is not None \
                and bar.currentMessage() == self._flash_msg:
            bar.showMessage(self._flash_prev)
        self._flash_msg = None

    def _notify_board(self) -> None:
        self._emitting = True
        try:
            self.state.notify_balance_changed("board")
        finally:
            self._emitting = False

    def _paths(self):
        from poseassess.core.balance.paths import WiiPaths

        return WiiPaths(self._project)

    def _cam_name(self, i: int | None = None) -> str:
        from poseassess.core.balance.paths import cam_name

        return cam_name(self._cam_index if i is None else i)

    # ============================================================ device tab
    def _on_wii_signal(self, *_args) -> None:
        self._sync_device_controls()

    def _sync_device_controls(self) -> None:
        c = self._wii
        self.chk_auto.blockSignals(True)
        self.chk_auto.setChecked(c.auto_connect)
        self.chk_auto.blockSignals(False)
        lock = c.tare_locked
        self.btn_disconnect.setEnabled(c.source is not None and not lock)
        self.btn_tare.setEnabled(c.connected and not lock)
        # While a take is recorded the source must not change: simulated or another board's
        # samples would be mixed into the take's wii.csv (Connect asks first: a board that
        # dropped out may need it).
        for w in (self.chk_auto, self.btn_sim, self.min_kg):
            w.setEnabled(not lock)
        if lock:
            self.btn_tare.setToolTip(f"Tare is not possible while {lock} is running (it would "
                                     "change the zero in the middle of the data).")
            why = f"Not possible while {lock} is running: the take would mix two sources."
            for w in (self.chk_auto, self.btn_sim, self.btn_disconnect, self.min_kg):
                w.setToolTip(why)
        else:
            self.btn_tare.setToolTip(TARE_TIP)
            self.chk_auto.setToolTip(AUTO_TIP)
            self.btn_sim.setToolTip(SIM_TIP)
            self.btn_disconnect.setToolTip(DISCONNECT_TIP)
            self.min_kg.setToolTip(MIN_KG_TIP)
        self.hint_lbl.setText(HIDAPI_HELP if c.fatal_error else PAIRING_HELP)
        self._update_detail()
        self._update_corner_button()

    def _update_detail(self, sample=None) -> None:
        try:
            text = self._wii.detail_text(sample)
        except Exception as e:  # noqa: BLE001
            text = f"{type(e).__name__}: {e}"
        if text != self._last_detail:
            self._last_detail = text
            self.status_lbl.setText(text)

    def _toggle_auto(self, on: bool) -> None:
        try:
            if on:
                if not self._wii.auto_connect:
                    self._wii.start_auto()
            elif self._wii.auto_connect:
                self._wii.disconnect()
        except Exception as e:  # noqa: BLE001
            log.exception("auto-connect toggle failed")
            self._warn("Wii Balance Board", f"Auto-connect failed: {e}")
        self._sync_device_controls()

    def _scan(self) -> None:
        devs = self._wii.list_devices()
        self.dev_combo.clear()
        for d in devs:
            label = d.get("product_string") or "Balance Board"
            serial = d.get("serial_number") or ""
            self.dev_combo.addItem(f"{label} {serial}".strip(), d.get("path"))
            self.dev_combo.setItemData(self.dev_combo.count() - 1, repr(d.get("path")),
                                       Qt.ToolTipRole)
        if devs:
            self.dev_combo.setCurrentIndex(0)  # (with a placeholder Qt keeps index -1)
        err = self._wii.last_scan_error
        if err:
            msg = f"<small style='color:{BAD}'>Cannot list HID devices: {err}</small>"
        elif not devs:
            msg = ("<small>No Balance Board found. Pair it via Bluetooth (red SYNC button; "
                   "usually needed again after it was switched off) and switch it on.</small>")
        else:
            msg = f"<small>{len(devs)} board(s) found.</small>"
        self.scan_lbl.setText(msg)
        self.scan_lbl.show()

    def _connect(self) -> None:
        path = self.dev_combo.currentData()
        lock = self._wii.tare_locked
        if lock and QMessageBox.question(
                self, "Wii Balance Board", f"{lock.capitalize()} is running. Connect this board "
                "and record its data into the running take?\n\nOnly do this when the take's "
                "board dropped out and does not reconnect by itself.",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
            return
        try:
            self._wii.connect_device(path, reconnect=self.chk_auto.isChecked(),
                                     allow_during_lock=bool(lock))
        except Exception as e:  # noqa: BLE001
            log.exception("connect failed")
            self._warn("Wii Balance Board", f"Connect failed: {e}")
        self._sync_device_controls()

    def _use_simulator(self) -> None:
        try:
            self._wii.start_simulator()
        except Exception as e:  # noqa: BLE001
            self._warn("Wii Balance Board", f"Simulator failed: {e}")
        self._sync_device_controls()

    def _disconnect(self) -> None:
        try:
            self._wii.disconnect()
        except RuntimeError as e:  # recording
            self._warn("Wii Balance Board", str(e))
        self._sync_device_controls()

    def _set_min_load(self, kg: float) -> None:
        try:
            self._wii.set_min_load(kg)
        except RuntimeError as e:  # recording: keep the threshold of the take
            self.min_kg.blockSignals(True)
            self.min_kg.setValue(self._wii.min_load)
            self.min_kg.blockSignals(False)
            self._warn("COP min. load", str(e))

    def _tare(self) -> None:
        try:
            t = self._wii.tare(1.0)
        except Exception as e:  # noqa: BLE001
            self._warn("Tare", str(e))
            return
        self._flash("Tare done: " + " ".join(f"{v:.2f}" for v in t) + " kg")

    def _tick_live(self) -> None:
        try:
            latest = self._wii.latest()
            self._update_detail(latest)
            if self.tabs.currentWidget() is self.device_tab:
                self._update_cop_view(latest)
            self._update_corner_button()
        except Exception:  # noqa: BLE001  (a live view must never raise into the event loop)
            log.exception("live Wii update failed")

    def _update_cop_view(self, latest) -> None:
        if latest is None:
            self.cop_view.set_data([], [], 0.0, "no live data")
            return
        samples = self._wii.recent(COP_TRAIL_S)
        trail = np.array([s.cop_board for s in samples], float).reshape(-1, 2)
        text = ""
        bm = self._body_mass
        if bm and bm > 0 and np.isfinite(latest.total_kg):
            text = f"{latest.total_kg / bm * 100:.0f}% of body mass ({bm:.1f} kg)"
        self.cop_view.set_data(trail, [], float(latest.total_kg), text)

    # ============================================================ project
    def on_project_changed(self, project) -> None:
        if project is not None:  # open dialog of the replay starts in the project's Wii folder
            self.replay_tab.start_dir = str(Path(project.root) / "wii" / "recordings")
        self._autosave()
        self._cancel_corner("Corner check cancelled (project changed).")
        old = self._project
        same = (old is not None and project is not None
                and Path(old.root).resolve() == Path(project.root).resolve())
        keep_cam = self._cam_index if same else 0  # e.g. the project reloaded after an export
        self._project = project
        self._dirty = False
        self._image = None
        if not same:
            self._image_sources = {}
        self._cam_index = 0
        self.board_tab.setEnabled(project is not None)
        self._reload_calibration()
        self._reload_board()
        self._reload_trial()
        self.cam_combo.blockSignals(True)
        self.cam_combo.clear()
        if project is not None:
            for i in range(1, int(project.config.num_cameras) + 1):
                self.cam_combo.addItem(self._cam_name(i))
            if 0 < keep_cam <= self.cam_combo.count():
                self.cam_combo.setCurrentIndex(keep_cam - 1)
        self.cam_combo.blockSignals(False)
        if project is None:
            self.frame_list.clear()
            self.picker.clear_image()
            self.qc_view.clear()
            self.frame_info.setText("")
            self.clicks_status.setText("")
            self.result_txt.setPlainText("Open a project to locate the board.")
            self._update_hint()
            return
        self._on_camera_changed()

    def _reload_calibration(self) -> None:
        self._cams = {}
        if self._project is None:
            return
        try:
            from poseassess.core.balance.calib import project_cameras

            self._cams = project_cameras(self._project)
        except Exception as e:  # noqa: BLE001  (malformed Calib.toml: shown when computing)
            log.warning("cannot read the calibration: %s", e)

    def _reload_board(self) -> None:
        self._board = None
        if self._project is not None:
            try:
                from poseassess.core.balance.board import load_board

                self._board = load_board(self._project)
            except Exception:  # noqa: BLE001
                log.exception("cannot read board.json")
        self._set_geometry_controls(self._board.geometry if self._board else None)
        self._qc_dirty = True
        self._show_board()

    def _reload_trial(self) -> None:
        self._body_mass = None
        if self._project is None:
            return
        try:
            from poseassess.core.balance.trial import load_trial

            bm = load_trial(self._project).body_mass_kg
            self._body_mass = float(bm) if bm else None
        except Exception:  # noqa: BLE001
            pass

    def _on_balance_changed(self, what: str) -> None:
        if self._emitting or self._project is None:
            return
        if what == "board":
            self._reload_board()
            if not self._dirty:
                self._reload_points()
            self._refresh_clicks_status()
        elif what == "trial":
            self._reload_trial()

    # ============================================================ board tab: frames
    def _on_camera_changed(self, *_):
        self._autosave()
        self._cam_index = self.cam_combo.currentIndex() + 1 if self.cam_combo.count() else 0
        self._image = None
        self._refresh_frames(select=None)
        self._refresh_clicks_status()

    def _board_dir(self) -> Path:
        return self._paths().board_cam_dir(self._cam_index)

    def _list_frames(self) -> list[Path]:
        d = self._board_dir()
        if not d.is_dir():
            return []
        return sorted(p for p in d.iterdir() if p.suffix.lower() in IMAGE_EXTS and p.is_file())

    def _saved_clicks(self):
        from poseassess.core.balance.board import load_clicks

        return load_clicks(self._project, self._cam_index)

    def _resolve_image(self, name: str | None) -> Path | None:
        if not name:
            return None
        p = self._board_dir() / name
        if p.is_file():
            return p
        p = self._paths().resolve(name)
        return p if p.is_file() else None

    def _refresh_frames(self, select: Path | None = None) -> None:
        """Fill the frame list of the current camera and show ``select`` (default: the image
        of the saved clicks, else the first frame)."""
        if self._project is None or not self._cam_index:
            return
        frames = self._list_frames()
        saved = self._saved_clicks()
        saved_img = self._resolve_image(saved.image) if saved else None
        if saved_img is not None and saved_img not in frames:
            frames.append(saved_img)
        if select is None:
            select = saved_img or (frames[0] if frames else None)
        self.frame_list.blockSignals(True)
        self.frame_list.clear()
        cur = None
        for p in frames:
            text = p.name if p.parent == self._board_dir() else self._paths().relative(p)
            if saved_img is not None and p == saved_img:
                n = len(saved.points)
                text += f"   ({n}/5 clicked)"
            it = QListWidgetItem(text)
            it.setData(Qt.UserRole, str(p))
            self.frame_list.addItem(it)
            if select is not None and p == Path(select):
                cur = it
        if cur is not None:
            self.frame_list.setCurrentItem(cur)
        self.frame_list.blockSignals(False)
        self._show_image(Path(select) if select is not None else None)

    def _on_frame_selected(self, cur, _prev=None) -> None:
        if cur is None:
            return
        p = Path(cur.data(Qt.UserRole))
        if self._image is not None and p == self._image:
            return
        self._autosave()
        self._show_image(p)

    def _show_image(self, path: Path | None) -> None:
        self._loading = True
        try:
            self._image = None
            if path is None or not self.picker.load_image(path):
                self.picker.clear_image()
                cam = self._cam_name()
                self.frame_info.setText(
                    f"<small>No frame for {cam} yet: grab one from a video, use the "
                    "calibration frame or load an image.</small>" if path is None else
                    f"<small style='color:{BAD}'>Cannot read {path.name}.</small>")
            else:
                self._image = path
                saved = self._saved_clicks()
                if saved is not None and self._resolve_image(saved.image) == path:
                    self.picker.set_points(saved.points)
                self._update_frame_info()
            self._dirty = False
        finally:
            self._loading = False
        self._update_hint()

    def _update_frame_info(self) -> None:
        size = self.picker.image_size()
        if size is None:
            self.frame_info.setText("")
            return
        cal = self._cams.get(self._cam_name())
        text = f"{size[0]}x{size[1]}"
        if cal is None:
            msg = (f"<small>{text} · no calibration for this camera yet "
                   "(2. Calibration).</small>")
        elif tuple(int(v) for v in cal.image_size) != tuple(size):
            cw, ch = (int(v) for v in cal.image_size)
            msg = (f"<small style='color:{BAD}'>{text}, but {self._cam_name()} is calibrated at "
                   f"{cw}x{ch}: clicks on this frame cannot be used. Use a frame of the "
                   "calibrated resolution.</small>")
        else:
            msg = f"<small style='color:{GOOD}'>{text} · matches the calibration</small>"
        self.frame_info.setText(msg)

    def _select_new_frame(self, path: Path, source: str) -> None:
        self._autosave()
        self._image_sources[str(Path(path))] = source
        self._refresh_frames(select=Path(path))
        self.view_tabs.setCurrentWidget(self.picker)

    def _grab_from_video(self) -> None:
        proj = self.state.project
        if not proj or not self._cam_index:
            return
        self._autosave()
        video = self._trial_video(proj)
        if video is None:
            f, _ = QFileDialog.getOpenFileName(
                self, f"Select a video of {self._cam_name()}", "",
                "Videos (*.mp4 *.MP4 *.avi *.mov *.mkv)")
            if not f:
                return
            video = Path(f)
        out = self._board_dir()
        out.mkdir(parents=True, exist_ok=True)
        dlg = _GrabFrameDialog(self, video, out, f"Grab a board frame — {self._cam_name()}")
        cam = self._cam_index

        def captured(index: int, path: str, vid: str):
            if cam != self._cam_index or self._project is not proj:
                return
            src = f"{self._paths().relative(vid)}#frame={index}" if vid else ""
            self._select_new_frame(Path(path), src)

        dlg.frame_captured.connect(captured)
        self._grab_dlg = dlg
        dlg.open()

    def _trial_video(self, proj) -> Path | None:
        """``videos/camNN.<video ext>`` of the current camera, or None."""
        d = proj.videos_dir
        if not d.is_dir():
            return None
        vids = sorted((p for p in d.iterdir() if p.is_file() and p.stem == self._cam_name()
                       and p.suffix.lower() in VIDEO_EXTS),
                      key=lambda p: VIDEO_EXTS.index(p.suffix.lower()))
        return vids[0] if vids else None

    def _use_calibration_frame(self) -> None:
        proj = self.state.project
        if not proj or not self._cam_index:
            return
        d = proj.extrinsic_cam_dir(self._cam_index)
        frames = sorted(p for p in d.glob("*") if p.suffix.lower() in IMAGE_EXTS) \
            if d.is_dir() else []
        if not frames:
            self._info("Use calibration frame",
                       f"No calibration (extrinsic) frame for {self._cam_name()}. Capture one "
                       "on “2. Calibration” (Select frames, Extrinsic), or grab a frame from a "
                       "video.")
            return
        self._autosave()
        src = frames[0]
        out = self._board_dir()
        out.mkdir(parents=True, exist_ok=True)
        dst = out / f"calib_{src.name}"
        shutil.copy2(src, dst)
        self._select_new_frame(dst, self._paths().relative(src))

    def _load_image_file(self) -> None:
        proj = self.state.project
        if not proj or not self._cam_index:
            return
        f, _ = QFileDialog.getOpenFileName(
            self, f"Load an image of {self._cam_name()}", "",
            "Images (*.jpg *.jpeg *.png *.bmp)")
        if not f:
            return
        src = Path(f)
        self._autosave()
        out = self._board_dir()
        out.mkdir(parents=True, exist_ok=True)
        dst = out / src.name
        if dst.resolve() != src.resolve():
            k = 2
            while dst.exists():
                dst = out / f"{src.stem}_{k}{src.suffix}"
                k += 1
            shutil.copy2(src, dst)
        self._select_new_frame(dst, str(src))

    # ============================================================ board tab: clicks
    def _on_points_changed(self, _n: int = 0) -> None:
        if not self._loading:
            self._dirty = True
        self._update_hint()

    def _update_hint(self) -> None:
        if self._project is None:
            self.click_hint.setText("")
            return
        if not self.picker.has_image():
            self.click_hint.setText("<small>Load a frame of this camera first.</small>")
            return
        n = len(self.picker.points())
        unsaved = " <span style='color:#e0a94c'>(unsaved)</span>" if self._dirty else ""
        if n < len(LABELS):
            lab = LABELS[n]
            col = COLORS[lab].name()
            self.click_hint.setText(f"Click {n + 1}/5: <span style='color:{col}'>{lab}</span> — "
                                    f"<span style='font-weight:400'>{HINTS[lab]}</span>{unsaved}")
        else:
            self.click_hint.setText(f"All 5 points set.{unsaved}<br><span style='font-weight:400'>"
                                    "Save them, click the other cameras, then compute the board "
                                    "position.</span>")

    def _clear_clicks(self) -> None:
        self.picker.clear_points()
        self._on_points_changed(0)

    def _swap_front_back(self) -> None:
        if not self.picker.swap_front_back():
            self._info("Swap front/back", "Click all 5 points first.")

    def _autosave(self) -> None:
        """Save unsaved clicks before the shown camera / frame / project changes or the page is
        left. Incomplete clicks on another frame never replace the complete clicks saved for a
        different frame (the user was only exploring); they are dropped instead."""
        if not self._dirty or self._project is None:
            return
        saved = self._saved_clicks()
        if (len(self.picker.points()) < len(LABELS) and saved is not None and saved.complete
                and self._resolve_image(saved.image) != self._image):
            log.info("incomplete clicks of %s not saved: the complete clicks of %s are kept",
                     self._cam_name(), saved.image)
            self._dirty = False
            return
        self._save_clicks(quiet=True)

    def _save_clicks(self, quiet: bool = True) -> bool:
        """Write the picker points of the shown camera/frame to points.json."""
        proj = self._project
        if proj is None or not self._cam_index:
            return False
        if self._image is None:
            if not quiet:
                self._info("Save clicks", "Load a frame of this camera first.")
            return False
        from poseassess.core.balance.board import CameraClicks, save_clicks

        paths = self._paths()
        img = self._image
        name = img.name if img.parent == paths.board_cam_dir(self._cam_index) else \
            paths.relative(img)
        source = self._image_sources.get(str(img)) or self._previous_source(img) or \
            paths.relative(img)
        clicks = CameraClicks(self._cam_name(), [tuple(map(float, p)) for p in
                                                 self.picker.points()],
                              name, self.picker.image_size(), source)
        try:
            save_clicks(proj, self._cam_index, clicks)
        except OSError as e:
            if not quiet:
                self._warn("Save clicks", f"Cannot save the clicks: {e}")
            log.warning("cannot save the clicks of %s: %s", clicks.cam, e)
            return False
        self._dirty = False
        self._update_hint()
        self._refresh_clicks_status()
        self._mark_saved_item()
        if not quiet:
            self._flash(f"Saved {len(clicks.points)} board points for {clicks.cam}.")
        return True

    def _previous_source(self, img: Path) -> str | None:
        saved = self._saved_clicks()
        if saved is not None and saved.source and self._resolve_image(saved.image) == img:
            return saved.source
        return None

    def _mark_saved_item(self) -> None:
        cur = self.frame_list.currentItem()
        if cur is None or self._image is None:
            return
        n = len(self.picker.points())
        for i in range(self.frame_list.count()):
            it = self.frame_list.item(i)
            p = Path(it.data(Qt.UserRole))
            base = p.name if p.parent == self._board_dir() else self._paths().relative(p)
            it.setText(base + (f"   ({n}/5 clicked)" if p == self._image else ""))

    def _reload_points(self) -> None:
        """Show the saved clicks again (after a front/back swap of the files)."""
        if self._image is None:
            return
        self._loading = True
        try:
            saved = self._saved_clicks()
            if saved is not None and self._resolve_image(saved.image) == self._image:
                self.picker.set_points(saved.points)
            self._dirty = False
        finally:
            self._loading = False
        self._update_hint()

    def _refresh_clicks_status(self) -> None:
        proj = self._project
        if proj is None:
            self.clicks_status.setText("")
            return
        from poseassess.core.balance.board import load_all_clicks

        try:
            clicks = load_all_clicks(proj)
        except Exception:  # noqa: BLE001
            clicks = {}
        used = set(self._board.cameras_used) if self._board is not None else set()
        parts = []
        for i in range(1, int(proj.config.num_cameras) + 1):
            name = self._cam_name(i)
            c = clicks.get(name)
            n = len(c.points) if c else 0
            if n == 5:
                mark = f"<span style='color:{GOOD}'>{name}&nbsp;✓</span>"
            elif n:
                mark = f"<span style='color:{WARN}'>{name}&nbsp;{n}/5</span>"
            else:
                mark = f"<span style='color:#888'>{name}&nbsp;–</span>"
            if name in used:
                mark = f"<b>{mark}</b>"
            parts.append(mark)
        legend = " (bold: used for the board position)" if used else ""
        self.clicks_status.setText("<small>Saved clicks: " + " · ".join(parts) +
                                   f"<br><span style='color:#888'>✓ = all 5 points{legend}"
                                   "</span></small>")

    # ============================================================ board tab: compute
    def _compute_board(self) -> None:
        proj = self.state.project
        if not proj:
            return
        self._autosave()
        self._reload_calibration()
        try:
            from poseassess.core.balance.board import compute_board

            reg = compute_board(proj, geometry=self._geometry())
        except NotImplementedError:
            self._warn("Board position", "Computing the board position is not available yet "
                                         "in this build.")
            return
        except ValueError as e:
            self._warn("Board position", str(e))
            return
        except Exception as e:  # noqa: BLE001
            log.exception("compute_board failed")
            self._warn("Board position", f"The board position could not be computed:\n"
                                         f"{type(e).__name__}: {e}")
            return
        self._board = reg
        self._qc_dirty = True
        self._show_board()
        self._refresh_clicks_status()
        self._notify_board()
        self.view_tabs.setCurrentWidget(self.qc_tab)
        self._flash("Board position computed (" + ", ".join(reg.cameras_used) + ").")
        if reg.pose.warnings:
            self._warn("Board position",
                       "Board position computed, but please check:\n\n" +
                       "\n\n".join(reg.pose.warnings))

    def _show_board(self) -> None:
        """Result text, stale banner, geometry of the COP view / controller."""
        b = self._board
        geo = b.geometry if b is not None else self._geometry()
        self.cop_view.board_mm = (geo.length_mm, geo.width_mm)
        self.cop_view.sensor_mm = (geo.sensor_dx_mm, geo.sensor_dy_mm)
        if b is not None:
            self._wii.set_sensor_spacing(geo.sensor_dx_mm / 1000.0, geo.sensor_dy_mm / 1000.0)
        if b is None:
            self.stale_lbl.hide()
            self.btn_recompute.hide()
            self.result_txt.setPlainText(
                "No board position yet.\n\nClick TL, TR, BR, BL, C in one or more cameras, "
                "save the clicks, then click “Compute board position”."
                if self._project is not None else "Open a project to locate the board.")
            self._update_corner_button()
            if self.view_tabs.currentWidget() is self.qc_tab:
                self._build_qc()
            return
        self.stale_lbl.setVisible(bool(b.stale))
        self.btn_recompute.setVisible(bool(b.stale))
        if b.stale:
            self.stale_lbl.setText(f"OUTDATED: {b.stale}")
        text = self._board_text(b)
        try:
            from poseassess.core.balance.calib import calib_mismatch

            mismatch = calib_mismatch(self._project) if self._project is not None else None
        except Exception:  # noqa: BLE001
            mismatch = None
        if mismatch and mismatch not in b.pose.warnings:
            text += f"\nWarning: {mismatch}"  # appeared after the board was computed
        self.result_txt.setPlainText(text)
        self._update_corner_button()
        if self.view_tabs.currentWidget() is self.qc_tab:
            self._build_qc()

    @staticmethod
    def _board_text(b) -> str:
        p = b.pose
        T = p.board_to_world
        method = ("multi-camera triangulation" if p.method == "triangulation"
                  else "single-camera PnP" if p.method == "pnp" else p.method)
        lines = [f"Method: {method}" + (", flat on the floor" if p.floor_constrained else ""),
                 "Cameras: " + (", ".join(b.cameras_used) or "-")]
        if p.reproj_error_px:
            lines.append("Reprojection error:")
            lines += [f"  {k}: {float(v):.2f} px" for k, v in sorted(p.reproj_error_px.items())]
        c = np.round(T.t, 3)
        u = np.round(b.up_world, 3)
        lines.append(f"Centre (m): {c[0]:+.3f} {c[1]:+.3f} {c[2]:+.3f}")
        lines.append(f"Up:         {u[0]:+.3f} {u[1]:+.3f} {u[2]:+.3f}")
        if b.world_up_sign < 0:
            lines.append("World +Z points down (Calib.toml); handled.")
        if p.tilt_deg is not None:
            lines.append(f"Tilt of the free fit: {p.tilt_deg:.1f}°"
                         + (" (removed)" if p.floor_constrained else ""))
        if p.world_camera:
            lines.append(f"World = camera {p.world_camera} (no extrinsics)")
        g = b.geometry
        lines.append(f"Geometry: {g.length_mm:g} x {g.width_mm:g} mm, top {g.height_mm:g} mm")
        cc = b.corner_check
        if cc:
            res = {"ok": "OK", "swapped": "front/back swapped (fixed)",
                   "redo": "clicks do not match (redo)"}.get(cc.get("result"), cc.get("result"))
            lines.append(f"Corner check: {res} ({cc.get('pressed') or '-'} responded)")
        else:
            lines.append("Corner check: not done")
        if b.created:
            lines.append(f"Computed: {b.created}")
        if b.stale:
            lines.append(f"OUTDATED: {b.stale}")
        lines += [f"Warning: {w}" for w in p.warnings]
        lines += [f"Note: {n}" for n in p.notes]
        return "\n".join(lines)

    def _on_view_tab(self, _idx: int) -> None:
        if self.view_tabs.currentWidget() is self.qc_tab and self._qc_dirty:
            self._build_qc()

    def _qc_image(self, i: int):
        """Image of camera ``i`` for the QC grid: the clicked frame, else any board frame,
        else the calibration frame; None when there is none."""
        from poseassess.core.balance.board import load_clicks

        paths = self._paths()
        cands = []
        name = self._cam_name(i)
        c = (self._board.clicks.get(name) if self._board is not None else None) or \
            load_clicks(self._project, i)
        if c is not None and c.image:
            d = paths.board_cam_dir(i)
            cands += [d / c.image, paths.resolve(c.image)]
        d = paths.board_cam_dir(i)
        if d.is_dir():
            cands += sorted(p for p in d.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        e = self._project.extrinsic_cam_dir(i)
        if e.is_dir():
            cands += sorted(p for p in e.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        for p in cands:
            if p.is_file():
                img = _imread(p)
                if img is not None:
                    pts = c.as_array() if (c is not None and c.complete and c.image and
                                           p.name == Path(c.image).name) else None
                    return img, pts
        return None, None

    def _qc_cell(self, i: int, zoom: bool):
        """QC image of camera ``i`` (the registered board over its frame, + clicks) and the
        caption ``(text, bgr colour)`` to draw on the final cell (None: already drawn)."""
        name = self._cam_name(i)
        img, pts = self._qc_image(i)
        cam = self._cams.get(name)
        if img is None:
            return _placeholder(f"{name}: no frame"), None
        if self._board is None or cam is None:
            cell = img.copy()
            if pts is not None:
                from poseassess.core.balance.overlay import draw_clicks

                draw_clicks(cell, [tuple(p) for p in pts])
            if zoom:
                cell = _crop_around(cell, [pts])
            text = "board not computed" if self._board is None else "no calibration"
            return cell, (f"{name}: {text}", (220, 220, 220))
        from poseassess.core.balance.overlay import board_check_image, project_world

        cell = board_check_image(img, cam, self._board, pts,
                                 highlight="TL" if self._corner is not None else None)
        if not zoom:
            return cell, None
        cell = _crop_around(cell, [project_world(cam, self._board.corners_world), pts])
        err = self._board.pose.reproj_error_px.get(name)
        if err is None:
            return cell, (f"{name}: not used for the board position", (200, 200, 200))
        col = (0, 200, 0) if err <= 5 else (0, 165, 255) if err <= 15 else (0, 0, 255)
        text = f"{name}: reprojection {float(err):.1f} px"
        if self._board.stale:
            text += "  (STALE)"
        return cell, (text, col)

    def _build_qc(self) -> None:
        self._qc_dirty = False
        if self._project is None:
            self.qc_view.clear()
            return
        cells = []
        zoom = self.chk_qc_zoom.isChecked()
        n = int(self._project.config.num_cameras)
        for i in range(1, n + 1):
            name = self._cam_name(i)
            try:
                cell, caption = self._qc_cell(i, zoom)
            except NotImplementedError:
                cell, caption = _placeholder(f"{name}: QC view not available yet"), None
            except Exception as e:  # noqa: BLE001
                log.exception("QC image of %s failed", name)
                cell, caption = _placeholder(f"{name}: {type(e).__name__}"), None
            cell = _letterbox(cell, QC_CELL)
            if caption is not None:
                _label(cell, *caption)
            cells.append(cell)
        if not cells:
            self.qc_view.clear()
            return
        cols = _grid_columns(len(cells), self.qc_view.width(), self.qc_view.height())
        while len(cells) % cols:
            cells.append(np.zeros((QC_CELL[1], QC_CELL[0], 3), np.uint8))
        rows = [np.hstack(cells[r:r + cols]) for r in range(0, len(cells), cols)]
        grid = np.vstack(rows)
        self.qc_view.set_bgr(np.ascontiguousarray(grid))
        self._qc_grid = grid

    # ============================================================ geometry
    def _show_geo_advanced(self, on: bool) -> None:
        for key in self._geo_advanced_rows:
            s = self.geo_spins[key]
            s.setVisible(on)
            lbl = self._geo_form.labelForField(s)
            if lbl is not None:
                lbl.setVisible(on)

    def _set_geometry_controls(self, geo) -> None:
        from poseassess.core.balance.geometry import BoardGeometry

        geo = geo or BoardGeometry()
        for key, s in self.geo_spins.items():
            s.blockSignals(True)
            s.setValue(float(getattr(geo, key)))
            s.blockSignals(False)

    def _geometry(self):
        from poseassess.core.balance.geometry import BoardGeometry

        return BoardGeometry(**{k: float(s.value()) for k, s in self.geo_spins.items()})

    def _apply_geometry(self) -> None:
        geo = self._geometry()
        self._wii.set_sensor_spacing(geo.sensor_dx_mm / 1000.0, geo.sensor_dy_mm / 1000.0)
        self.cop_view.board_mm = (geo.length_mm, geo.width_mm)
        self.cop_view.sensor_mm = (geo.sensor_dx_mm, geo.sensor_dy_mm)
        proj = self.state.project
        if not proj or self._board is None:
            self._flash("Board geometry set; it is used when the board position is computed.")
            return
        try:
            from poseassess.core.balance.board import compute_board

            reg = compute_board(proj, geometry=geo, keep_corner_check=True)
        except NotImplementedError:
            self._warn("Board geometry", "Recomputing the board is not available yet.")
            return
        except Exception as e:  # noqa: BLE001
            self._warn("Board geometry", f"The board position could not be recomputed with the "
                                         f"new geometry:\n{e}")
            return
        self._board = reg
        self._qc_dirty = True
        self._show_board()
        self._notify_board()
        self._flash("Board position recomputed with the new geometry.")

    # ============================================================ corner check
    def _update_corner_button(self) -> None:
        if not hasattr(self, "btn_corner"):
            return
        running = self._corner is not None
        ok = running or (self._board is not None and self._wii.connected
                         and self._project is not None)
        self.btn_corner.setEnabled(bool(ok))
        self.btn_corner.setText("■ Cancel corner check" if running else "▶ Start corner check")
        if not running:
            if self._board is None:
                tip = "Compute the board position first."
            elif not self._wii.connected:
                tip = "Connect the Balance Board first (tab “Device & live COP”)."
            else:
                tip = CORNER_TIP
            self.btn_corner.setToolTip(tip)

    def _corner_button(self) -> None:
        if self._corner is not None:
            self._cancel_corner("Corner check cancelled.")
        else:
            self.start_corner_check()

    def start_corner_check(self) -> bool:
        """Start the corner check (the subject presses TL). Returns whether it started."""
        if self._project is None:
            return False
        if self._board is None:
            self._warn("Corner check", "Compute the board position first.")
            return False
        samples = self._wii.recent(CORNER_BASELINE_S) if self._wii.connected else []
        if not samples:
            self._warn("Corner check", "The corner check needs the connected Wii Balance Board: "
                                       "connect it first (tab “Device & live COP”).")
            return False
        self._corner = {"t": time.perf_counter(), "src": self._wii.source,
                        "baseline": np.mean([s.kg for s in samples], axis=0)}
        # The check only means something when the operator presses the corner the CLICKS call
        # TL: with front and back swapped that is the physical BR corner. So the target is
        # shown in the camera images, never described via the power button.
        self.corner_lbl.setText(
            "<b>Press firmly on the corner marked “PRESS” (yellow rings) in the check view</b> "
            "— the corner you clicked first (TL), at the left end of the orange “front” edge. "
            "Find it "
            "in the room from the camera images; do <b>not</b> go by the power button for this "
            "test. Keep pressing for a second: the sensor that responds shows whether your "
            f"clicks match the board. (timeout {CORNER_CHECK_TIMEOUT_S:g} s)")
        self._corner_timer.start()
        self._update_corner_button()
        if self.view_tabs.currentWidget() is self.qc_tab:
            self._refresh_qc()  # now with the "PRESS" mark
        else:
            self._qc_dirty = True
            self.view_tabs.setCurrentWidget(self.qc_tab)  # builds it
        return True

    def _refresh_qc(self) -> None:
        """Rebuild the check view now if it is shown (else when it is shown)."""
        self._qc_dirty = True
        if self.view_tabs.currentWidget() is self.qc_tab:
            self._build_qc()

    def _cancel_corner(self, msg: str) -> None:
        if self._corner is None:
            return
        self._corner = None
        self._corner_timer.stop()
        self.corner_lbl.setText(f"<small>{msg}</small>")
        self._update_corner_button()
        self._refresh_qc()  # drop the "PRESS" mark

    def _tick_corner(self) -> None:
        c = self._corner
        if c is None:
            self._corner_timer.stop()
            return
        try:
            now = time.perf_counter()
            if c["src"] is not self._wii.source or not self._wii.connected:
                self._cancel_corner("Corner check cancelled (the board was disconnected).")
                return
            if now - c["t"] > CORNER_CHECK_TIMEOUT_S:
                self._cancel_corner("Corner check: no clear press detected. Try again and press "
                                    f"harder (at least {CORNER_MIN_RISE_KG:g} kg).")
                return
            recent = self._wii.recent(CORNER_WINDOW_S)
            if now - c["t"] < CORNER_WINDOW_S or not recent:
                return
            from poseassess.wii.protocol import pressed_sensor

            name = pressed_sensor(c["baseline"], np.mean([s.kg for s in recent], axis=0),
                                  CORNER_MIN_RISE_KG)
            if name is not None:
                self._corner = None
                self._corner_timer.stop()
                self._corner_result(name)
                self._update_corner_button()
                self._refresh_qc()  # drop the "PRESS" mark
        except Exception as e:  # noqa: BLE001
            log.exception("corner check failed")
            self._cancel_corner(f"Corner check failed: {e}")

    def _corner_result(self, name: str) -> None:
        proj = self._project
        if proj is None:
            return
        from poseassess.core.balance import board as B

        if name == "TL":
            result = "ok"
            msg = (f"<b style='color:{GOOD}'>OK</b>: the TL sensor responded — the board frame "
                   "matches the sensors (+y = front = edge opposite the power button, +x = "
                   "right).")
        elif name == "BR":
            result = "swapped"
            try:
                B.swap_front_back_clicks(proj)
                try:
                    B.compute_board(proj, geometry=(self._board.geometry if self._board
                                                    is not None else self._geometry()))
                except Exception:
                    B.swap_front_back_clicks(proj)  # undo: keep clicks and board consistent
                    raise
            except Exception as e:  # noqa: BLE001  (ValueError, NotImplementedError...)
                self.corner_lbl.setText(
                    f"<b style='color:{BAD}'>The BR sensor responded</b>: front and back are "
                    f"swapped, but the board could not be recomputed ({e}). Use “Swap "
                    "front/back” in every camera and compute again.")
                return
            msg = (f"<b style='color:{WARN}'>Front and back were swapped; fixed.</b> The BR "
                   "sensor responded, so the clicks were exchanged (TL↔BR, TR↔BL) in every "
                   "camera and the board position was recomputed. The subject must stand "
                   "facing the TL/TR edge (opposite the power button). Run the check again to "
                   "confirm.")
        else:
            result = "redo"
            msg = (f"<b style='color:{BAD}'>Redo the clicks</b>: the {name} sensor responded "
                   f"({CORNER_NAMES[name]}), so the clicks do not match the sensors (left/right "
                   "mirrored or turned 90°). TL/TR is the long edge opposite the power button; "
                   "left/right as seen by the subject facing that edge.")
        try:
            B.set_corner_check(proj, result, name)
        except Exception as e:  # noqa: BLE001
            log.warning("cannot record the corner check: %s", e)
        self.corner_lbl.setText(msg)
        self._reload_board()
        if name == "BR":
            if self._dirty:
                self.picker.swap_front_back()
            else:
                self._reload_points()
        self._refresh_clicks_status()
        self._notify_board()
        self._flash(f"Corner check: {name} sensor responded")

    # ============================================================ lifecycle
    def showEvent(self, ev):
        super().showEvent(ev)
        self._live.start()
        if self._project is not None:
            # the calibration may have changed on "2. Calibration": refresh stale state
            self._reload_calibration()
            self._reload_board()
            self._refresh_frames_if_changed()
            self._update_frame_info()
        self._sync_device_controls()

    def _refresh_frames_if_changed(self) -> None:
        """Frames may have been added or removed elsewhere meanwhile (e.g. "📸 Board snapshots"
        on 3b. Capture): refresh the frame list of the shown camera, keeping the shown frame
        and its clicks. Never while there are unsaved clicks."""
        if self._project is None or not self._cam_index or self._dirty:
            return
        listed = {self.frame_list.item(i).data(Qt.UserRole)
                  for i in range(self.frame_list.count())}
        on_disk = {str(p) for p in self._list_frames()}
        gone = {p for p in listed if not Path(p).is_file()}
        if not (on_disk - listed) and not gone:
            return
        keep = self._image if self._image is not None and self._image.is_file() else None
        self._refresh_frames(select=keep)

    def hideEvent(self, ev):
        super().hideEvent(ev)
        self._live.stop()
        self._autosave()

    def shutdown(self) -> None:
        self.replay_tab.shutdown()
        self._live.stop()
        self._corner = None
        self._corner_timer.stop()
        self._autosave()
        if self._grab_dlg is not None:
            try:
                self._grab_dlg.reject()
            except RuntimeError:
                pass
            self._grab_dlg = None


# ------------------------------------------------------------ image helpers
def _imread(path: Path) -> np.ndarray | None:
    """BGR image or None. Reads the bytes with numpy and decodes them with OpenCV: unlike
    ``cv2.imread`` this works with non-ASCII paths on Windows (e.g. a folder with Chinese characters)."""
    import cv2

    try:
        buf = np.fromfile(str(path), dtype=np.uint8)
        return cv2.imdecode(buf, cv2.IMREAD_COLOR) if buf.size else None
    except (OSError, ValueError, cv2.error):
        return None


def _placeholder(text: str) -> np.ndarray:
    import cv2

    img = np.full((QC_CELL[1], QC_CELL[0], 3), 40, np.uint8)
    cv2.putText(img, text, (20, QC_CELL[1] // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (200, 200, 200), 2, cv2.LINE_AA)
    return img


def _label(img: np.ndarray, text: str, color=(220, 220, 220)) -> None:
    """Caption with a dark outline at the bottom-left of ``img`` (in place)."""
    import cv2

    k = max(1.4, min(3.0, img.shape[1] / 1280.0))
    org = (int(12 * k), img.shape[0] - int(14 * k))
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.7 * k, (0, 0, 0),
                max(2, int(round(5 * k))), cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.7 * k, color,
                max(1, int(round(2 * k))), cv2.LINE_AA)


def _crop_around(img: np.ndarray, point_sets, margin: float = 0.6,
                 aspect: float = QC_CELL[0] / QC_CELL[1]) -> np.ndarray:
    """Crop ``img`` to the region around the given pixel points (+ ``margin`` of their extent
    on each side) with the cell aspect ratio; the whole image when there are no points."""
    pts = [np.asarray(p, float).reshape(-1, 2) for p in point_sets if p is not None]
    pts = np.vstack(pts) if pts else np.zeros((0, 2))
    pts = pts[np.all(np.isfinite(pts), axis=1)] if len(pts) else pts
    h, w = img.shape[:2]
    if not len(pts):
        return img
    (x0, y0), (x1, y1) = pts.min(axis=0), pts.max(axis=0)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    bw = max((x1 - x0) * (1 + 2 * margin), 160.0)
    bh = max((y1 - y0) * (1 + 2 * margin), 90.0)
    if bw / bh < aspect:
        bw = bh * aspect
    else:
        bh = bw / aspect
    s = min(1.0, w / bw, h / bh)
    bw, bh = bw * s, bh * s
    x0 = int(round(min(max(cx - bw / 2, 0), w - bw)))
    y0 = int(round(min(max(cy - bh / 2, 0), h - bh)))
    out = img[y0:y0 + int(round(bh)), x0:x0 + int(round(bw))]
    return np.ascontiguousarray(out) if out.size else img


def _grid_columns(n: int, view_w: int, view_h: int, max_cols: int = 3) -> int:
    """Number of grid columns (1..max_cols) that shows ``n`` 16:9 cells largest in a
    ``view_w`` x ``view_h`` view."""
    if n <= 1 or view_w <= 0 or view_h <= 0:
        return 1
    cw, ch = QC_CELL
    best, best_s = 1, -1.0
    for cols in range(1, min(max_cols, n) + 1):
        rows = -(-n // cols)
        s = min(view_w / (cols * cw), view_h / (rows * ch))
        if s > best_s + 1e-9:
            best, best_s = cols, s
    return best


def _letterbox(img: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize ``img`` into ``size`` (w, h) keeping its aspect ratio (black borders)."""
    import cv2

    w, h = size
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    ih, iw = img.shape[:2]
    s = min(w / iw, h / ih)
    nw, nh = max(1, int(round(iw * s))), max(1, int(round(ih * s)))
    small = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    out = np.zeros((h, w, 3), np.uint8)
    x0, y0 = (w - nw) // 2, (h - nh) // 2
    out[y0:y0 + nh, x0:x0 + nw] = small
    return out
