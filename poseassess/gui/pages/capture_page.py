"""Page "3b. Capture": record the trial videos of all cameras AND the Wii data together, straight
into the project.

Owner: GUI-CAPTURE (docs/WII_INTEGRATION.md §7). Two workflows:

* record here: every project camera (``cam01..camNN``, sources in ``wii/capture.json``) and the
  shared Wii Balance Board (``state.wii``) are recorded into ``wii/recordings/<take>/`` on one
  clock; the take is then exported (worker thread) to ``videos/camNN.mp4`` + ``frames.csv``, so
  "4. Run" works unchanged and the Wii data are aligned by the frame timestamps (trial.json
  method ``recorded``; minus the camera latency set under "Export to videos/");
* Wii only: uncheck "Record cameras", record the board while another system films, import the
  videos on "3. Videos" and align on "6. Results" (2 small jumps at the start).

Nothing here runs unless the page is used: cameras open only on "Open cameras", timers run
while the page is visible (or recording), the Wii is optional.
"""

from __future__ import annotations

import logging
import re
import threading
from pathlib import Path

from PySide6.QtCore import QObject, Qt, QThread, QTimer, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..widgets.camera_grid import CameraGrid
from .base_page import BasePage

log = logging.getLogger(__name__)
_ORPHANS: list = []  # export threads still running at shutdown (kept alive until exit)

GOOD, WARN, BAD, ACCENT = "#3ad35a", "#e0a94c", "#e05c5c", "#3d8bfd"
PREVIEW_MS = 66           # ~15 Hz previews / live COP
MAX_DEVICE_INDEX = 8      # "Scan cameras" tries device indices 0..7
RESOLUTIONS = ["default", "640x480", "1280x720", "1920x1080"]
FPS_CHOICES = ["default", "25", "30", "50", "60"]

HELP_HTML = """
<h3>Two ways to collect a trial with the Wii Balance Board</h3>
<p><b>A. Record here (recommended).</b> Set the source of every camera (device index or a video
file) on the <i>Cameras</i> tab, click <b>▶ Open cameras</b>, check that every preview runs and
that the board shows as connected, then <b>● Record</b>. Mark events with <b>F9</b> (or
<b>M</b>). After <b>■ Stop</b> the take is exported to <code>videos/cam01.mp4 … camNN.mp4</code>
on one common frame grid (the cameras are free-running; every exported frame k shows the same
instant) and the Wii data are aligned by the frame timestamps (<code>frames.csv</code>). Then
run the pipeline on <b>4. Run</b> as usual.</p>
<p><b>B. Videos recorded by another system.</b> Uncheck <b>Record cameras</b> and record the
Wii only (or use <code>run_wii.bat</code>). Ask the subject to do <b>2 small jumps (or 2
stomps)</b> at the start. Import the videos on <b>3. Videos</b>, run the pipeline, then align
the Wii recording on <b>6. Results → Balance (Wii)</b> (sync event or manual offset).</p>
<p><b>Tips.</b> Record at the resolution of the calibration (the intrinsics are
resolution-specific). Keep Pose2Sim's synchronization stage off for takes recorded here (the
export already synchronizes the cameras). <b>📸 Board snapshots</b> saves the current frame of
every camera for clicking the board corners on <b>2b. Wii Board</b>. Tare the board while it is
empty, before the subject steps on (tare is locked while recording).</p>
<p><b>Timing.</b> A frame is timestamped when the camera delivers it, after exposure, USB
transfer and driver buffering (USB webcams: typically 30-100 ms), so the Wii data would lead
the video by that much. Record 2 small jumps once, export, run the pipeline and use <i>Check
timing with a sync event</i> on 6. Results → Balance (Wii): it measures this delay. Enter it as
<b>Camera latency</b> (Export to videos/) and export again; it is kept for later takes.</p>
"""


# ---------------------------------------------------------------------------- workers
class _ExportWorker(QObject):
    progress = Signal(int, int, str)
    done = Signal(object, str)  # (ExportResult | None, error text | "cancelled" | "")

    def __init__(self, project, rec_id: str, fps: int | None, keep_raw: bool,
                 latency_s: dict | None = None):
        super().__init__()
        self.project, self.rec_id, self.fps, self.keep_raw = project, rec_id, fps, keep_raw
        self.latency_s = latency_s
        self._cancel = threading.Event()

    def cancel(self) -> None:
        self._cancel.set()

    def run(self):
        try:
            from poseassess.core.capture.export import (
                ExportCancelled,
                export_recording_to_project,
            )
        except Exception as e:  # noqa: BLE001
            self.done.emit(None, f"{type(e).__name__}: {e}")
            return
        try:
            # update_project=False: the shared Project / Config.toml are updated on the GUI
            # thread (_on_export_done), never from this worker
            res = export_recording_to_project(self.project, self.rec_id, fps=self.fps,
                                              keep_raw=self.keep_raw, update_project=False,
                                              progress=self.progress.emit,
                                              cancel=self._cancel.is_set,
                                              latency_s=self.latency_s)
            self.done.emit(res, "")
        except ExportCancelled:
            self.done.emit(None, "cancelled")
        except (ValueError, RuntimeError) as e:
            self.done.emit(None, str(e))
        except Exception as e:  # noqa: BLE001
            log.exception("export failed")
            self.done.emit(None, f"{type(e).__name__}: {e}")


class _ProbeBridge(QObject):
    """Delivers ``probe_cameras`` results from its (daemon) thread to the GUI thread."""

    done = Signal(object, str)


class _OpenBridge(QObject):
    """Delivers ``session.open_streams`` results ``(session, started, results)`` from its
    (daemon) thread to the GUI thread."""

    done = Signal(object, object, object)


def _parse_source(text: str):
    """Source cell text -> device index (int) or path / URL (str); "" -> None."""
    t = text.strip()
    if not t:
        return None
    m = re.match(r"^(\d+)(\s|·|$)", t)
    if m:
        return int(m.group(1))
    return t


def _parse_size(text: str) -> tuple[int | None, int | None]:
    m = re.match(r"^\s*(\d+)\s*[x×*]\s*(\d+)\s*$", text or "")
    return (int(m.group(1)), int(m.group(2))) if m else (None, None)


def _parse_fps(text: str) -> float | None:
    try:
        v = float((text or "").strip())
    except ValueError:
        return None
    return v if v > 0 else None


def _format_error(e: Exception) -> str:
    return str(e) or type(e).__name__


class CapturePage(BasePage):
    nav_title = "3b. Capture"

    def __init__(self, state):
        super().__init__(state)
        self._session = None
        self._settings = None
        self._thread = None      # export QThread (kept until its finished signal)
        self._worker = None
        self._export_active = False
        self._probe_bridge = None
        self._probe_thread = None
        self._open_bridge = None
        self._opening = None     # (session, {cam: CameraStream}) while cameras are opening
        self._found: list[dict] = []
        self._marks = 0
        self._loading = False
        self._status_tick = 0
        self._export_rec_id: str | None = None
        self._wii_link: str | None = None  # last board link state written to the take

        root = QHBoxLayout(self)
        root.addWidget(self._build_left())

        right = QSplitter(Qt.Vertical)
        top = QWidget()
        th = QHBoxLayout(top)
        th.setContentsMargins(0, 0, 0, 0)
        self.grid = CameraGrid()
        th.addWidget(self.grid, 1)
        th.addWidget(self._build_wii_box())
        right.addWidget(top)
        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_cameras_tab(), "Cameras")
        self.tabs.addTab(self._build_takes_tab(), "Takes")
        help_lbl = QLabel(HELP_HTML)
        help_lbl.setWordWrap(True)
        help_lbl.setTextFormat(Qt.RichText)
        help_lbl.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        help_scroll = QScrollArea()
        help_scroll.setWidgetResizable(True)
        help_scroll.setWidget(help_lbl)
        self.tabs.addTab(help_scroll, "Help")
        right.addWidget(self.tabs)
        right.setStretchFactor(0, 3)
        right.setStretchFactor(1, 2)
        root.addWidget(right, 1)

        self._timer = QTimer(self)
        self._timer.setInterval(PREVIEW_MS)
        self._timer.timeout.connect(self._tick_safe)

        for key in ("F9", "M"):
            sc = QShortcut(QKeySequence(key), self)
            sc.setContext(Qt.WindowShortcut)
            sc.activated.connect(self._mark_event)
            sc.setEnabled(False)
            setattr(self, f"_sc_{key.lower()}", sc)

        wii = self.state.wii
        wii.status_changed.connect(self._on_wii_status)
        wii.source_changed.connect(self._on_wii_source)
        wii.connection_event.connect(self._on_wii_connection)
        self.state.balance_changed.connect(self._on_balance_changed)
        self.state.busy_changed.connect(self._update_controls)
        self._set_recording_ui(False)

    # ================================================================= UI build
    def _build_left(self) -> QWidget:
        left = QWidget()
        left.setFixedWidth(240)
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 4, 0)

        take = QGroupBox("Take")
        tf = QVBoxLayout(take)
        form = QFormLayout()
        self.subject_edit = QLineEdit()
        self.subject_edit.setPlaceholderText("optional")
        self.subject_edit.setToolTip("Subject or trial name; it is added to the take's folder "
                                     "name (letters and digits only) and stored in session.json.")
        self.notes_edit = QLineEdit()
        self.notes_edit.setPlaceholderText("optional")
        form.addRow("Subject", self.subject_edit)
        form.addRow("Notes", self.notes_edit)
        tf.addLayout(form)
        self.chk_record_cams = QCheckBox("Record cameras")
        self.chk_record_cams.setChecked(True)
        self.chk_record_cams.setToolTip(
            "Checked: record every camera of the project together with the Wii data (the take "
            "can then replace the trial videos). Unchecked: record the Wii Balance Board only, "
            "for videos filmed by another system.")
        tf.addWidget(self.chk_record_cams)
        self.b_record = QPushButton("● Record")
        self.b_record.setMinimumHeight(34)
        self.b_record.clicked.connect(self._toggle_record)
        tf.addWidget(self.b_record)
        self.rec_label = QLabel("Not recording")
        self.rec_label.setWordWrap(True)
        tf.addWidget(self.rec_label)
        er = QHBoxLayout()
        self.event_edit = QLineEdit()
        self.event_edit.setPlaceholderText("event label")
        self.event_edit.setToolTip("Label of the next event marker (empty: mark_<n>). Use "
                                   "\"sync\" for a jump or stomp made for synchronization.")
        self.b_mark = QPushButton("⚑ Mark")
        self.b_mark.setToolTip("Write an event marker into events.csv now (F9, or M when no "
                               "text field has the focus).")
        self.b_mark.clicked.connect(self._mark_event)
        er.addWidget(self.event_edit, 1)
        er.addWidget(self.b_mark)
        tf.addLayout(er)
        self.events_list = QListWidget()
        self.events_list.setMaximumHeight(80)
        self.events_list.setToolTip("Events of the current take (time since the start).")
        tf.addWidget(self.events_list)
        lv.addWidget(take)

        cams = QGroupBox("Cameras")
        cv = QVBoxLayout(cams)
        self.b_open = QPushButton("▶ Open cameras")
        self.b_open.setToolTip("Start every enabled camera of the Cameras tab (preview only; "
                               "nothing is recorded yet). Also applies changed camera settings.")
        self.b_open.clicked.connect(self._open_cameras)
        self.b_close = QPushButton("Close cameras")
        self.b_close.clicked.connect(self._close_cameras)
        self.b_scan = QPushButton("Scan cameras")
        self.b_scan.setToolTip("Look for cameras on device indices 0-7 (takes a few seconds; "
                               "cameras that are open here are not probed).")
        self.b_scan.clicked.connect(self._scan_cameras)
        cv.addWidget(self.b_open)
        row = QHBoxLayout()
        row.addWidget(self.b_close)
        row.addWidget(self.b_scan)
        cv.addLayout(row)
        self.cams_label = QLabel("<small>Cameras closed.</small>")
        self.cams_label.setWordWrap(True)
        cv.addWidget(self.cams_label)
        self.b_snap = QPushButton("📸 Board snapshots")
        self.b_snap.setToolTip("Save the current frame of every open camera into "
                               "wii/board/camNN/ for clicking the board corners on 2b. Wii Board.")
        self.b_snap.clicked.connect(self._board_snapshots)
        cv.addWidget(self.b_snap)
        lv.addWidget(cams)

        exp = QGroupBox("Export to videos/")
        ef = QFormLayout(exp)
        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(0, 240)
        self.fps_spin.setSpecialValueText("project")
        self.fps_spin.setSuffix(" fps")
        self.fps_spin.setToolTip("Frame rate of the exported videos (the common frame grid). "
                                 "\"project\" uses the project's frame rate. When it differs "
                                 "from the project's, the project frame rate is updated.")
        self.fps_spin.valueChanged.connect(self._save_settings)
        self.chk_keep_raw = QCheckBox("Keep raw camera files")
        self.chk_keep_raw.setChecked(True)
        self.chk_keep_raw.setToolTip("Keep camNN.mkv in the take's folder after the export (needed "
                                     "to export the take again, e.g. at another frame rate).")
        self.chk_keep_raw.toggled.connect(self._save_settings)
        self.latency_spin = QSpinBox()
        self.latency_spin.setRange(0, 1000)
        self.latency_spin.setSuffix(" ms")
        self.latency_spin.setToolTip(
            "Camera latency: a frame is timestamped when the camera delivers it, after "
            "exposure, USB transfer and driver buffering (USB webcams: typically 30-100 ms). "
            "The export subtracts it from the frame timestamps, so the Wii data line up with "
            "the video. Measure it once per camera setup with \"Check timing with a sync "
            "event\" on 6. Results → Balance (Wii) (a take with 2 small jumps), then export "
            "again.")
        self.latency_spin.valueChanged.connect(self._save_settings)
        ef.addRow("Output", self.fps_spin)
        ef.addRow("Camera latency", self.latency_spin)
        ef.addRow(self.chk_keep_raw)
        lv.addWidget(exp)
        lv.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidget(left)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setFixedWidth(258)
        return scroll

    def _build_wii_box(self) -> QWidget:
        from ..widgets.cop_view import CopView

        box = QGroupBox("Wii Balance Board")
        box.setFixedWidth(350)
        v = QVBoxLayout(box)
        self.wii_label = QLabel("Wii: off")
        self.wii_label.setWordWrap(True)
        v.addWidget(self.wii_label)
        self.cop = CopView()
        self.cop.setMinimumSize(300, 170)
        v.addWidget(self.cop, 1)
        row = QHBoxLayout()
        self.b_tare = QPushButton("Tare (board empty)")
        self.b_tare.setToolTip("Zero the board with the last second of data. Nobody may stand "
                               "on the board.")
        self.b_tare.clicked.connect(self._tare)
        row.addWidget(self.b_tare)
        row.addStretch(1)
        v.addLayout(row)
        self.wii_hint = QLabel("")
        self.wii_hint.setWordWrap(True)
        v.addWidget(self.wii_hint)
        return box

    def _build_cameras_tab(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        self.cam_table = QTableWidget(0, 7)
        self.cam_table.setHorizontalHeaderLabels(
            ["Camera", "Use", "Source (device index / file)", "", "Resolution", "FPS", "Status"])
        hh = self.cam_table.horizontalHeader()
        for c, width in ((0, 60), (1, 36), (2, 190), (3, 30), (4, 100), (5, 70)):
            hh.setSectionResizeMode(c, QHeaderView.Interactive)
            self.cam_table.setColumnWidth(c, width)
        hh.setStretchLastSection(True)
        self.cam_table.verticalHeader().setVisible(False)
        self.cam_table.setSelectionMode(QAbstractItemView.NoSelection)
        v.addWidget(self.cam_table, 1)
        self.scan_label = QLabel("<small>Source: a device index (0 = first camera; use "
                                 "\"Scan cameras\") or a video file (for testing). Changes are "
                                 "saved in wii/capture.json and applied by \"▶ Open cameras\"."
                                 "</small>")
        self.scan_label.setWordWrap(True)
        v.addWidget(self.scan_label)
        return w

    def _build_takes_tab(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        self.takes = QTableWidget(0, 7)
        self.takes.setHorizontalHeaderLabels(
            ["Take", "Started", "Duration", "Cameras", "Wii", "Exported", "Trial"])
        self.takes.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.takes.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.takes.setSelectionMode(QAbstractItemView.SingleSelection)
        self.takes.verticalHeader().setVisible(False)
        self.takes.horizontalHeader().setStretchLastSection(True)
        self.takes.setColumnWidth(0, 190)
        self.takes.setColumnWidth(1, 130)
        self.takes.itemSelectionChanged.connect(self._update_take_buttons)
        v.addWidget(self.takes, 1)
        row = QHBoxLayout()
        self.b_export = QPushButton("⇪ Export to videos/")
        self.b_export.setToolTip("Resample the take's camera videos onto one common frame grid "
                                 "and write them as videos/camNN.mp4 (replaces the trial videos).")
        self.b_export.clicked.connect(self._export_selected)
        self.b_use = QPushButton("Use as trial's Wii recording")
        self.b_use.setToolTip("Make this take the Wii recording of the trial (for videos filmed "
                              "elsewhere: align it on 6. Results → Balance (Wii)).")
        self.b_use.clicked.connect(self._use_selected)
        self.b_folder = QPushButton("Open folder")
        self.b_folder.clicked.connect(self._open_selected_folder)
        refresh = QPushButton("Refresh")
        refresh.clicked.connect(self._refresh_takes)
        for b in (self.b_export, self.b_use, self.b_folder):
            row.addWidget(b)
        row.addStretch(1)
        row.addWidget(refresh)
        v.addLayout(row)
        prow = QHBoxLayout()
        self.progress = QProgressBar()
        self.progress.setTextVisible(True)
        self.b_cancel = QPushButton("Cancel export")
        self.b_cancel.clicked.connect(self._cancel_export)
        self.export_label = QLabel("")
        self.export_label.setWordWrap(True)
        self.export_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        prow.addWidget(self.progress, 1)
        prow.addWidget(self.b_cancel)
        v.addLayout(prow)
        v.addWidget(self.export_label)
        self._show_progress(False)
        return w

    # ============================================================ project / settings
    def on_project_changed(self, project):
        same = (project is not None and self._session is not None
                and Path(self._session.project.root) == Path(project.root))
        if not same:
            self._close_session(reason="another project was opened" if project is not None
                                else "the project was closed")
        if project is None:
            self._settings = None
            self.cam_table.setRowCount(0)
            self.grid.set_cameras([])
            self.takes.setRowCount(0)
            return
        from poseassess.core.capture.session import CaptureSession
        from poseassess.core.capture.settings import load_capture_settings

        if same:
            self._session.project = project
            if not self._session.recording:
                self._settings = load_capture_settings(project)
                self._session.settings = self._settings
        else:
            self._settings = load_capture_settings(project)
            self._session = CaptureSession(project, self._settings)
        self._load_settings_ui()
        self._refresh_takes()
        self._update_controls()

    def _close_session(self, reason: str = "") -> None:
        s = self._session
        self._session = None
        if s is None:
            return
        folder = None
        if s.recording:
            folder = self._finish_recording(s)
        try:
            s.close()
        except Exception:  # noqa: BLE001
            log.exception("closing the cameras failed")
        self.grid.set_recording(False)
        if folder is not None:
            QMessageBox.information(self, "Capture", f"The recording was stopped because "
                                    f"{reason}. The take was saved in\n{folder}")

    def _load_settings_ui(self) -> None:
        proj, s = self.state.project, self._settings
        if proj is None or s is None:
            return
        self._loading = True
        try:
            self.fps_spin.setSpecialValueText(f"project ({proj.config.frame_rate})")
            self.fps_spin.setValue(int(s.output_fps or 0))
            self.latency_spin.setValue(int(round(s.latency_ms or 0)))
            self.chk_keep_raw.setChecked(bool(s.keep_raw))
            if not self.subject_edit.text():
                self.subject_edit.setText(s.subject)
            names = sorted(s.cameras)
            self.grid.set_cameras(names)
            self.cam_table.setRowCount(len(names))
            for r, cam in enumerate(names):
                cs = s.cameras[cam]
                item = QTableWidgetItem(cam)
                item.setFlags(Qt.ItemIsEnabled)
                self.cam_table.setItem(r, 0, item)
                chk = QCheckBox()
                chk.setChecked(cs.enabled)
                chk.setToolTip(f"Use {cam}")
                chk.toggled.connect(self._save_settings)
                cw = QWidget()
                cl = QHBoxLayout(cw)
                cl.setContentsMargins(6, 0, 0, 0)
                cl.addWidget(chk)
                self.cam_table.setCellWidget(r, 1, cw)
                src = QComboBox()
                src.setEditable(True)
                src.setInsertPolicy(QComboBox.NoInsert)
                self._fill_sources(src, cs.source)
                src.setToolTip(str(cs.source))
                src.currentTextChanged.connect(self._save_settings)
                src.currentTextChanged.connect(src.setToolTip)
                self.cam_table.setCellWidget(r, 2, src)
                pick = QToolButton()
                pick.setText("…")
                pick.setToolTip(f"Use a video file as {cam} (for testing the workflow).")
                pick.clicked.connect(lambda _=False, row=r: self._pick_file(row))
                self.cam_table.setCellWidget(r, 3, pick)
                res = QComboBox()
                res.setEditable(True)
                res.addItems(RESOLUTIONS)
                res.setCurrentText(f"{cs.width}x{cs.height}" if cs.width and cs.height
                                   else "default")
                res.setToolTip("Requested resolution (WxH). Use the resolution of the "
                               "calibration; \"default\" = what the camera delivers.")
                res.currentTextChanged.connect(self._save_settings)
                self.cam_table.setCellWidget(r, 4, res)
                fps = QComboBox()
                fps.setEditable(True)
                fps.addItems(FPS_CHOICES)
                fps.setCurrentText(f"{cs.fps:g}" if cs.fps else "default")
                fps.currentTextChanged.connect(self._save_settings)
                self.cam_table.setCellWidget(r, 5, fps)
                st = QTableWidgetItem("closed")
                st.setFlags(Qt.ItemIsEnabled)
                self.cam_table.setItem(r, 6, st)
        finally:
            self._loading = False

    def _fill_sources(self, combo: QComboBox, current) -> None:
        combo.blockSignals(True)
        combo.clear()
        found = {d["index"]: d for d in self._found}
        for i in range(MAX_DEVICE_INDEX):
            d = found.get(i)
            if d is not None:
                fps = f" @{d['fps']:.0f}" if d.get("fps") else ""
                combo.addItem(f"{i} · found {d['width']}x{d['height']}{fps}")
            else:
                combo.addItem(str(i))
        if isinstance(current, int):
            idx = current if 0 <= current < MAX_DEVICE_INDEX else -1
            if idx >= 0:
                combo.setCurrentIndex(idx)
            else:
                combo.setEditText(str(current))
        else:
            combo.setEditText(str(current))
        combo.blockSignals(False)

    def _cam_rows(self) -> list[str]:
        return [self.cam_table.item(r, 0).text() for r in range(self.cam_table.rowCount())
                if self.cam_table.item(r, 0) is not None]

    def _read_table(self) -> None:
        """Camera table -> self._settings (in place)."""
        from poseassess.core.capture.settings import CameraSetting

        s = self._settings
        if s is None:
            return
        for r, cam in enumerate(self._cam_rows()):
            cw = self.cam_table.cellWidget(r, 1)
            chk = cw.findChild(QCheckBox) if cw is not None else None
            src = self.cam_table.cellWidget(r, 2)
            res = self.cam_table.cellWidget(r, 4)
            fps = self.cam_table.cellWidget(r, 5)
            source = _parse_source(src.currentText()) if src is not None else None
            old = s.cameras.get(cam)
            if source is None:
                source = old.source if old is not None else r
            w, h = _parse_size(res.currentText() if res is not None else "")
            s.cameras[cam] = CameraSetting(source, w, h,
                                           _parse_fps(fps.currentText() if fps else ""),
                                           bool(chk.isChecked()) if chk is not None else True,
                                           old.latency_ms if old is not None else None)

    def _save_settings(self, *_):
        if self._loading or self._settings is None:
            return
        proj = self.state.project
        if not proj:
            return
        if not (self._session is not None and self._session.recording):
            self._read_table()
        self._settings.output_fps = self.fps_spin.value() or None
        self._settings.latency_ms = float(self.latency_spin.value())
        self._settings.keep_raw = self.chk_keep_raw.isChecked()
        self._settings.subject = self.subject_edit.text().strip()
        try:
            from poseassess.core.capture.settings import save_capture_settings

            save_capture_settings(proj, self._settings)
        except OSError as e:
            log.error("cannot save wii/capture.json: %s", e)

    def _pick_file(self, row: int):
        proj = self.state.project
        if not proj:
            return
        f, _ = QFileDialog.getOpenFileName(self, "Video file as camera source", str(proj.root),
                                           "Videos (*.mp4 *.avi *.mov *.mkv)")
        if not f:
            return
        src = self.cam_table.cellWidget(row, 2)
        if src is not None:
            src.setEditText(f)
        self._save_settings()

    # =================================================================== cameras
    def _open_cameras(self):
        """Open every enabled camera on worker threads (a camera driver can block for seconds:
        the window must stay responsive); ``_on_open_done`` registers them."""
        proj = self.state.project
        s = self._session
        if not proj or s is None or s.recording or self._opening is not None:
            return
        self._read_table()
        self._save_settings()
        try:
            started = s.begin_open()
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "Open cameras", _format_error(e))
            return
        self._timer.start()
        if not started:
            self._on_open_done(s, {}, {})
            return
        from poseassess.core.capture import session as session_mod

        timeout = session_mod.OPEN_TIMEOUT_S
        bridge = self._open_bridge = _OpenBridge()
        bridge.done.connect(self._on_open_done)
        self._opening = (s, started)
        for cam in started:
            self.grid.set_message(cam, "opening…")
        self.cams_label.setText(f"<small>Opening {len(started)} camera(s)…</small>")

        def work():
            try:
                results = session_mod.open_streams(started, timeout)
            except Exception as e:  # noqa: BLE001
                results = {c: _format_error(e) for c in started}
            try:
                bridge.done.emit(s, started, results)
            except RuntimeError:  # the app closed meanwhile: release the cameras here
                for st in started.values():
                    st.request_stop()

        threading.Thread(target=work, name="open-cameras", daemon=True).start()
        self._update_controls()

    def _on_open_done(self, session, started, results):
        self._opening = None
        if session is not self._session:  # project changed / page closed meanwhile
            for st in started.values():
                st.request_stop()  # its capture thread releases the device
            self._update_controls()
            return
        try:
            errors = session.finish_open(started, results)
        except Exception as e:
            log.exception("registering the opened cameras failed")
            errors = {c: _format_error(e) for c in started}
        self._update_controls()
        if errors:
            QMessageBox.warning(self, "Open cameras", "These cameras could not be opened:\n\n"
                                + "\n".join(f"• {c}: {e}" for c, e in sorted(errors.items()))
                                + "\n\nCheck the source (device index / file) on the Cameras "
                                  "tab, or close other programs that use the camera.")

    def _close_cameras(self):
        s = self._session
        if s is None or s.recording or self._opening is not None:
            return
        try:
            s.stop_cameras()  # in parallel, one shared timeout
        except Exception:
            log.exception("closing the cameras failed")
        for cam in self.grid.names():
            self.grid.set_message(cam, "closed")
        self._update_controls()

    def _scan_cameras(self):
        if self._probe_thread is not None and self._probe_thread.is_alive():
            return
        skip = set()
        s = self._session
        if s is not None:
            for c, st in s.streams.items():
                if st.running and isinstance(st.source, int):
                    skip.add(st.source)
        self._probe_bridge = _ProbeBridge()
        self._probe_bridge.done.connect(self._on_scan_done)
        bridge = self._probe_bridge

        def work():
            try:
                from poseassess.core.capture.session import probe_cameras

                found = probe_cameras(MAX_DEVICE_INDEX, 3.0, skip=skip)
                bridge.done.emit(found, "")
            except Exception as e:  # noqa: BLE001
                bridge.done.emit([], _format_error(e))

        # A daemon thread (not a QThread): opening a camera driver can block for seconds and
        # must never keep the app from closing.
        self._probe_thread = threading.Thread(target=work, name="probe-cameras", daemon=True)
        self._probe_skip = skip
        self.b_scan.setEnabled(False)
        self.b_scan.setText("Scanning…")
        self._probe_thread.start()

    def _on_scan_done(self, found, error):
        self.b_scan.setText("Scan cameras")
        self._probe_thread = None
        self._update_controls()
        if error:
            QMessageBox.warning(self, "Scan cameras", error)
            return
        self._found = list(found)
        for r in range(self.cam_table.rowCount()):
            src = self.cam_table.cellWidget(r, 2)
            if src is not None:
                cur = _parse_source(src.currentText())
                self._loading = True
                try:
                    self._fill_sources(src, cur if cur is not None else r)
                finally:
                    self._loading = False
        parts = [f"{d['index']} ({d['width']}x{d['height']}"
                 + (f", {d['fps']:.0f} fps)" if d.get("fps") else ")") for d in self._found]
        skipped = sorted(getattr(self, "_probe_skip", set()))
        txt = ("Found: " + ", ".join(parts)) if parts else \
            f"No camera found on device indices 0-{MAX_DEVICE_INDEX - 1}."
        if skipped:
            txt += f" In use here (not probed): {', '.join(map(str, skipped))}."
        self.scan_label.setText(f"<small>{txt}</small>")
        self.tabs.setCurrentIndex(0)

    def _board_snapshots(self):
        proj = self.state.project
        s = self._session
        if not proj or s is None:
            return
        files = s.save_board_snapshots()
        if not files:
            QMessageBox.warning(self, "Board snapshots", "No camera delivers frames: open the "
                                "cameras first (▶ Open cameras).")
            return
        rel = "\n".join(f"• {f.relative_to(proj.root).as_posix()}" for f in files)
        QMessageBox.information(self, "Board snapshots", f"Saved {len(files)} snapshot(s):\n"
                                f"{rel}\n\nClick the board corners on 2b. Wii Board.")

    # ======================================================================= Wii
    def _wii_connected(self) -> bool:
        try:
            return bool(self.state.wii.connected)
        except Exception:  # noqa: BLE001
            return False

    def _on_wii_status(self, _text: str = ""):
        self._update_wii_label()

    def _update_wii_label(self):
        try:
            text = self.state.wii.status_text()
        except Exception:  # noqa: BLE001
            text = "Wii: unavailable"
        ok = self._wii_connected()
        try:
            from ..widgets.wii_status import state_color

            color = state_color(self.state.wii)
        except Exception:  # noqa: BLE001 - same colours as the status-bar item, else a guess
            low = text.lower()
            color = GOOD if ok else (BAD if ("error" in low or "hidapi" in low) else
                                     "#8a8f98" if low.startswith("wii: off") else WARN)
        html = f"<span style='color:{color}'>●</span> {text}"
        if self.wii_label.text() != html:
            self.wii_label.setText(html)
        hint = "" if ok else ("<small>No board connected: switch it on (it connects "
                              "automatically once paired) or connect it on 2b. Wii Board. "
                              "Video takes can be recorded without the board.</small>")
        if self.wii_hint.text() != hint:
            self.wii_hint.setText(hint)

    def _on_wii_source(self, src):
        s = self._session
        if s is not None and s.recording:
            try:
                s.attach_force(src)
            except Exception:  # noqa: BLE001
                log.exception("attaching the new force source failed")

    def _on_wii_connection(self, label: str):
        """Board connected / lost (GUI thread). During a take, a board that (re)connects is
        added to it and link CHANGES are written to events.csv (a late signal about the state
        the take started with is not an event)."""
        s = self._session
        if s is None or not s.recording:
            return
        src = self.state.wii.source
        try:
            if label == "wii_connected" and src is not None:
                s.attach_force(src)
                if label != self._wii_link:
                    s.log_force_source(src, "connected")  # device and tare in effect
            if label == self._wii_link:
                return
            self._wii_link = label
            ev = s.add_event(label)
        except Exception:  # noqa: BLE001
            log.exception("logging %s failed", label)
            return
        if ev is not None:
            self._add_event_item(ev)

    def _tare(self):
        wii = self.state.wii
        if wii.tare_locked:
            QMessageBox.information(self, "Tare", "Tare is locked while recording.")
            return
        try:
            wii.tare(1.0)
        except NotImplementedError:
            QMessageBox.warning(self, "Tare", "Taring is not available yet.")
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, "Tare", f"Cannot tare: {_format_error(e)}\n\nConnect the "
                                "board and leave it empty for a second, then try again.")

    def _update_cop(self):
        wii = self.state.wii
        try:
            samples = wii.recent(3.0) or []
            latest = wii.latest()
        except Exception:  # noqa: BLE001
            samples, latest = [], None
        cop = [s.cop_board for s in samples]
        total = float(latest.total_kg) if latest is not None else 0.0
        self.cop.set_data(cop, [], total, "" if latest is not None else "no data")

    # ================================================================= recording
    def _toggle_record(self):
        s = self._session
        if s is not None and s.recording:
            self._stop_recording()
        else:
            self._start_recording()

    def _start_recording(self):
        proj = self.state.project
        s = self._session
        if not proj or s is None or s.recording:
            return
        if self._export_running():
            QMessageBox.warning(self, "Record", "Wait until the export has finished.")
            return
        record_cams = self.chk_record_cams.isChecked()
        wii = self.state.wii
        connected = self._wii_connected()
        if record_cams:
            problems = s.video_take_problems()
            if problems:
                QMessageBox.warning(self, "Record", "Every camera of the project must deliver "
                                    "frames before recording:\n\n"
                                    + "\n".join(f"• {p}" for p in problems)
                                    + "\n\nSet the sources on the Cameras tab and click ▶ Open "
                                      "cameras, or uncheck \"Record cameras\" to record the "
                                      "Wii Balance Board only.")
                return
            warn = s.size_warnings()
            if warn and QMessageBox.question(
                    self, "Record", "\n\n".join(warn) + "\n\nRecord anyway?",
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
                return
            if not connected and QMessageBox.question(
                    self, "Record", "No Wii Balance Board is connected. Record the videos "
                    "without force data?\n\n(A board that connects during the take is added to "
                    "it automatically.)",
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
                return
        elif not connected:
            QMessageBox.warning(self, "Record", "Nothing to record: no Wii Balance Board is "
                                "connected and \"Record cameras\" is unchecked.\n\nSwitch the "
                                "board on (or connect it on 2b. Wii Board) and wait until it "
                                "shows as connected.")
            return
        pose = geometry = None
        try:
            from poseassess.core.balance.board import load_board

            reg = load_board(proj)
            if reg is not None:
                geometry = reg.geometry
                if not reg.stale and reg.pose.world_camera is None:
                    pose = reg.pose  # only fills the (informative) COP world columns
        except Exception:  # noqa: BLE001
            log.exception("board.json not readable")
        if geometry is None:
            try:
                from poseassess.core.balance.geometry import BoardGeometry

                geometry = BoardGeometry()
            except Exception:  # noqa: BLE001
                geometry = None
        self._save_settings()
        try:
            s.start_recording(force=wii.source if connected else None, board=pose,
                              geometry=geometry, subject=self.subject_edit.text().strip(),
                              notes=self.notes_edit.text().strip(),
                              record_cameras=record_cams)
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "Record", f"Cannot start recording:\n{_format_error(e)}")
            return
        try:
            wii.lock_tare("recording")
        except Exception:  # noqa: BLE001
            log.exception("tare lock failed")
        self._wii_link = "wii_connected" if connected else "wii_disconnected"
        self._marks = 0
        self.events_list.clear()
        self._set_recording_ui(True)
        self._timer.start()
        self._update_rec_label()

    def _finish_recording(self, s):
        """Stop ``s``'s take and unlock tare; returns the folder (None if not recording)."""
        try:
            folder = s.stop_recording()
        except Exception:  # noqa: BLE001
            log.exception("stopping the recording failed")
            folder = None
        try:
            self.state.wii.lock_tare(None)
        except Exception:  # noqa: BLE001
            log.exception("tare unlock failed")
        self._set_recording_ui(False)
        return folder

    def _stop_recording(self):
        s = self._session
        if s is None or not s.recording:
            return
        counts = dict(s.recorder.counts)
        folder = self._finish_recording(s)
        if not self.isVisible():
            self._timer.stop()
        self.state.notify_balance_changed("recording")
        self._refresh_takes()
        if folder is None:
            return
        from poseassess.wii import io as wio

        meta = wio.read_session_json(folder)
        dur = meta.get("duration_s") or 0.0
        self.rec_label.setText(f"Saved {folder.name}: {dur:.1f} s, Wii {counts.get('wii', 0)} "
                               f"samples, {counts.get('events', 0)} events")
        err = s.recorder.error
        if err:
            QMessageBox.warning(self, "Recording", f"The take was saved, but a file could not "
                                f"be written completely:\n{err}")
        self._select_take(folder.name)
        if meta.get("has_video"):
            proj = self.state.project
            existing = [f for f in proj.videos_dir.glob("cam*.*")] if proj else []
            if not existing:
                self._start_export(folder.name)
            elif QMessageBox.question(
                    self, "Export", f"Replace the trial videos in videos/ with the take "
                    f"{folder.name}?\n\n(You can also export it later from the Takes tab.)",
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes) == QMessageBox.Yes:
                self._start_export(folder.name)
        elif meta.get("has_wii"):
            if QMessageBox.question(
                    self, "Wii recording", f"Use the take {folder.name} as the trial's Wii "
                    "recording?\n\nImport the videos of the other system on 3. Videos, run the "
                    "pipeline, then align the Wii data on 6. Results → Balance (Wii).",
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes) == QMessageBox.Yes:
                self._use_take(folder.name, ask=False)

    def _set_recording_ui(self, on: bool):
        self.b_record.setText("■ Stop" if on else "● Record")
        self.b_record.setStyleSheet(f"background:{BAD}; color:white; font-weight:600;" if on
                                    else "font-weight:600;")
        self.b_record.setToolTip("Stop the take (the cameras keep running)." if on else
                                 "Start a take: every camera of the project and the Wii "
                                 "Balance Board on one clock.")
        self.b_mark.setEnabled(on)
        self._sc_f9.setEnabled(on)
        self._sc_m.setEnabled(on)
        self.grid.set_recording(on and bool(self._session and self._session.recorder.meta
                                            .get("has_video")))
        for w in (self.subject_edit, self.notes_edit, self.chk_record_cams, self.cam_table):
            w.setEnabled(not on)
        if not on:
            self.rec_label.setText("Not recording")
        self._update_controls()

    def _update_rec_label(self):
        s = self._session
        if s is None or not s.recording:
            return
        c = s.recorder.counts
        e = s.elapsed_s
        txt = (f"<span style='color:{BAD}'>● REC</span> {int(e // 60):02d}:{e % 60:04.1f} · "
               f"Wii {c.get('wii', 0)} · events {c.get('events', 0)}")
        err = s.recorder.error
        if err:
            txt += f"<br><span style='color:{BAD}'>{err}</span>"
        if self.rec_label.text() != txt:
            self.rec_label.setText(txt)

    def _mark_event(self):
        s = self._session
        if s is None or not s.recording:
            return
        label = self.event_edit.text().strip() or f"mark_{self._marks + 1}"
        try:
            ev = s.add_event(label)
        except OSError as e:
            QMessageBox.warning(self, "Mark event", f"Cannot write events.csv: {e}")
            return
        if ev is None:
            return
        self._marks += 1
        self._add_event_item(ev)

    def _add_event_item(self, ev: dict):
        self.events_list.addItem(f"{ev['t_rel']:8.2f} s   {ev['label']}")
        self.events_list.scrollToBottom()
        self._update_rec_label()

    # ===================================================================== takes
    def _refresh_takes(self):
        proj = self.state.project
        self.takes.setRowCount(0)
        if not proj:
            return
        from poseassess.core.balance.paths import WiiPaths
        from poseassess.core.balance.trial import load_trial
        from poseassess.core.capture.export import take_cameras
        from poseassess.wii import io as wio

        paths = WiiPaths(proj)
        trial = load_trial(proj)
        keep = self._selected_take()
        ids = list(reversed(paths.list_recordings()))  # newest first
        self.takes.setRowCount(len(ids))
        running = self._session.folder if self._session and self._session.recording else None
        for r, rec_id in enumerate(ids):
            folder = paths.recording_dir(rec_id)
            meta = wio.read_session_json(folder)
            cams = meta.get("camera_names") or sorted(take_cameras(folder))
            has_wii = (folder / wio.WII_CSV).is_file()
            n_wii = (meta.get("samples") or {}).get("wii")
            ex = meta.get("export") or {}
            started = str(meta.get("start_time_iso") or meta.get("created") or "")
            started = started.replace("T", " ")[:19]
            dur = meta.get("duration_s")
            if running is not None and folder == running:
                dur_txt = "recording…"
            else:
                dur_txt = f"{dur:.1f} s" if isinstance(dur, (int, float)) else "?"
            if ex:
                exp_txt = f"{ex.get('n_frames', '?')} fr @ {ex.get('fps', '?')} fps"
            elif cams:
                raw = all(v is not None for v, _ in take_cameras(folder).values())
                exp_txt = "no" if raw else "no (raw missing)"
            else:
                exp_txt = "—"
            vals = [rec_id, started, dur_txt, ", ".join(cams) if cams else "Wii only",
                    (f"yes ({n_wii})" if n_wii else "yes") if has_wii else "no", exp_txt,
                    "✓ " + trial.alignment.method if trial.recording == rec_id else ""]
            for c, v in enumerate(vals):
                item = QTableWidgetItem(str(v))
                if c == 0:
                    item.setData(Qt.UserRole, rec_id)
                    if meta.get("imported_from"):
                        item.setToolTip(f"Imported from {meta['imported_from']}")
                self.takes.setItem(r, c, item)
        if keep:
            self._select_take(keep)
        self._update_take_buttons()

    def _selected_take(self) -> str | None:
        r = self.takes.currentRow()
        if r < 0 or not self.takes.selectedItems():
            return None
        item = self.takes.item(r, 0)
        return item.data(Qt.UserRole) if item is not None else None

    def _select_take(self, rec_id: str):
        for r in range(self.takes.rowCount()):
            item = self.takes.item(r, 0)
            if item is not None and item.data(Qt.UserRole) == rec_id:
                self.takes.selectRow(r)
                return

    def _update_take_buttons(self):
        sel = self._selected_take()
        busy = self._export_running()
        rec = bool(self._session and self._session.recording)
        is_running = bool(rec and sel and self._session.folder
                          and self._session.folder.name == sel)
        run = self.state.busy("pipeline")
        self.b_export.setEnabled(bool(sel) and not busy and not rec and not run)
        self.b_export.setToolTip(
            f"Not possible during {run}: it reads the videos this would replace." if run else
            "Resample the take's camera videos onto one common frame grid and write them as "
            "videos/camNN.mp4 (replaces the trial videos).")
        self.b_use.setEnabled(bool(sel) and not busy and not is_running)
        self.b_folder.setEnabled(bool(sel))

    def _open_selected_folder(self):
        proj, rec_id = self.state.project, self._selected_take()
        if not proj or not rec_id:
            return
        from poseassess.core.balance.paths import WiiPaths

        QDesktopServices.openUrl(QUrl.fromLocalFile(str(WiiPaths(proj).recording_dir(rec_id))))

    def _use_selected(self):
        rec_id = self._selected_take()
        if rec_id:
            self._use_take(rec_id, ask=True)

    def _use_take(self, rec_id: str, ask: bool = True):
        proj = self.state.project
        if not proj:
            return
        from poseassess.core.balance.paths import WiiPaths
        from poseassess.core.balance.trial import load_trial, set_recording
        from poseassess.wii import io as wio

        folder = WiiPaths(proj).recording_dir(rec_id)
        if not (folder / wio.WII_CSV).is_file():
            QMessageBox.warning(self, "Wii recording", f"The take {rec_id} has no Wii data.")
            return
        meta = wio.read_session_json(folder)
        if ask and meta.get("has_video"):
            box = QMessageBox(self)
            box.setWindowTitle("Wii recording")
            box.setText(f"The take {rec_id} also has camera videos. Exporting them to videos/ "
                        "aligns the Wii data by the frame timestamps.\n\nUse only its Wii data "
                        "instead (align on 6. Results → Balance (Wii))?")
            b_exp = box.addButton("Export videos", QMessageBox.AcceptRole)
            b_wii = box.addButton("Use Wii data only", QMessageBox.DestructiveRole)
            box.addButton(QMessageBox.Cancel)
            box.exec()
            if box.clickedButton() is b_exp:
                self._start_export(rec_id)
                return
            if box.clickedButton() is not b_wii:
                return
        trial = load_trial(proj)
        if ask and trial.recording and trial.recording != rec_id and QMessageBox.question(
                self, "Wii recording", f"The trial currently uses the Wii recording "
                f"{trial.recording} ({trial.alignment.method}). Replace it with {rec_id}?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
            return
        try:
            set_recording(proj, rec_id, source="external")
        except OSError as e:
            QMessageBox.critical(self, "Wii recording", f"Cannot write wii/trial.json: {e}")
            return
        self.state.notify_balance_changed("trial")
        self._refresh_takes()

    # ==================================================================== export
    def _export_running(self) -> bool:
        return self._export_active or self._thread is not None

    def _on_thread_finished(self):
        # Only now may the QThread be released ("Destroyed while thread is still running").
        self._thread = None
        self._worker = None
        self.state.set_busy("export", None)
        self._update_controls()

    def _export_selected(self):
        rec_id = self._selected_take()
        if rec_id:
            self._start_export(rec_id)

    def _start_export(self, rec_id: str) -> bool:
        proj = self.state.project
        if not proj or self._export_running():
            return False
        s = self._session
        if s is not None and s.recording and s.folder and s.folder.name == rec_id:
            return False
        run = self.state.busy("pipeline")
        if run:  # it reads videos/ and Config.toml, which the export replaces
            QMessageBox.warning(self, "Export", f"Wait until {run} has finished, then export "
                                f"the take {rec_id} from the Takes tab (⇪ Export to videos/).")
            return False
        from poseassess.core.balance.paths import WiiPaths, cam_name
        from poseassess.core.capture.export import plan_export

        fps = int(self.fps_spin.value() or proj.config.frame_rate)
        cams = [cam_name(i) for i in range(1, proj.config.num_cameras + 1)]
        latency = self._settings.latency_s(cams) if self._settings is not None else None
        try:
            plan = plan_export(WiiPaths(proj).recording_dir(rec_id), fps, cams=cams,
                               latency_s=latency)
        except (ValueError, OSError) as e:
            QMessageBox.warning(self, "Export", f"The take {rec_id} cannot be exported:\n"
                                f"{_format_error(e)}")
            return False
        measured = [m for m in plan.measured_fps.values() if m > 0]
        low = min(measured) if measured else fps
        if low < 0.9 * fps:
            alt = max(1, int(round(low)))
            box = QMessageBox(self)
            box.setIcon(QMessageBox.Warning)
            box.setWindowTitle("Export")
            box.setText(f"The cameras delivered only about {low:.1f} fps, but the output frame "
                        f"rate is {fps} fps: many frames would be duplicates.\n\nExport at "
                        f"{alt} fps instead? (The project frame rate is set to the exported "
                        "frame rate.)")
            b_alt = box.addButton(f"Export at {alt} fps", QMessageBox.AcceptRole)
            b_keep = box.addButton(f"Export at {fps} fps", QMessageBox.DestructiveRole)
            box.addButton(QMessageBox.Cancel)
            box.exec()
            if box.clickedButton() is b_alt:
                fps = alt
            elif box.clickedButton() is not b_keep:
                return False
        self._export_rec_id = rec_id
        self._worker = _ExportWorker(proj, rec_id, fps, self.chk_keep_raw.isChecked(), latency)
        self._worker.progress.connect(self._on_export_progress)
        self._worker.done.connect(self._on_export_done)
        self._thread = QThread()
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.done.connect(self._thread.quit)
        self._thread.finished.connect(self._on_thread_finished)
        self._export_active = True
        self.state.set_busy("export", f"the export of the take {rec_id} to videos/ "
                                      "(3b. Capture)")
        self._show_progress(True)
        self.progress.setRange(0, 0)
        self.export_label.setText(f"Exporting {rec_id} at {fps} fps…")
        self.tabs.setCurrentIndex(1)
        self._thread.start()
        self._update_controls()
        return True

    def _show_progress(self, on: bool):
        self.progress.setVisible(on)
        self.b_cancel.setVisible(on)
        self.b_cancel.setEnabled(on)

    def _on_export_progress(self, done: int, total: int, msg: str):
        if self.progress.maximum() != total:
            self.progress.setRange(0, max(1, total))
        self.progress.setValue(done)
        self.export_label.setText(msg)

    def _cancel_export(self):
        if self._worker is not None:
            self._worker.cancel()
            self.b_cancel.setEnabled(False)
            self.export_label.setText("Cancelling…")

    def _on_export_done(self, res, error: str):
        self._export_active = False
        self.state.set_busy("export", None)
        self._show_progress(False)
        proj = self.state.project
        wproj = self._worker.project if self._worker is not None else proj
        same = (proj is not None and wproj is not None
                and Path(proj.root) == Path(wproj.root))
        if res is not None and not error and wproj is not None:
            # the project frame rate + Config.toml follow the exported videos (GUI thread)
            from poseassess.core.capture.export import apply_export_fps

            try:
                res.fps_changed, more = apply_export_fps(wproj, res.fps)
                res.warnings += more
            except Exception as e:  # noqa: BLE001
                log.exception("updating the project frame rate failed")
                res.warnings.append(f"The project frame rate could not be set to {res.fps} fps "
                                    f"({e}): set it on 1. Project and click Save settings.")
        if error == "cancelled":
            self.export_label.setText("Export cancelled; videos/ is unchanged.")
        elif error or res is None:
            self.export_label.setText(f"<span style='color:{BAD}'>Export failed.</span>")
            QMessageBox.critical(self, "Export", f"The export failed; videos/ is unchanged.\n\n"
                                 f"{error}")
        else:
            if same:
                if res.fps_changed:
                    self.state.set_project(wproj)  # other pages reload (new frame rate)
                self.state.notify_balance_changed("trial")
            worst = ", ".join(f"{c} {v:.0f} ms" for c, v in sorted(res.max_dt_ms.items()))
            names = res.videos[0].name if len(res.videos) == 1 else \
                f"{res.videos[0].name} … {res.videos[-1].name}"
            msg = (f"Exported {res.n_frames} frames at {res.fps} fps to videos/{names}.\n"
                   f"Largest time difference to the common frame grid: {worst}.")
            if res.fps_changed:
                msg += f"\nThe project frame rate is now {res.fps} fps (Config.toml updated)."
            msg += "\n\nNext: run the pipeline on 4. Run (existing results belong to the " \
                   "previous videos)."
            self.export_label.setText(f"<span style='color:{GOOD}'>✓</span> Exported "
                                      f"{res.rec_id}: {res.n_frames} frames at {res.fps} fps.")
            if res.warnings:
                QMessageBox.warning(self, "Export", msg + "\n\nWarnings:\n"
                                    + "\n".join(f"• {w}" for w in res.warnings))
            else:
                QMessageBox.information(self, "Export", msg)
        self._export_rec_id = None
        self._refresh_takes()
        self._update_controls()

    # ==================================================================== timers
    def showEvent(self, event):
        super().showEvent(event)
        self._timer.start()
        self._update_wii_label()
        self._update_cop()

    def hideEvent(self, event):
        super().hideEvent(event)
        if not (self._session is not None and self._session.recording):
            self._timer.stop()

    def _tick_safe(self):
        try:
            self._tick()
        except Exception:  # noqa: BLE001 - a display error must not spam dialogs at 15 Hz
            log.exception("capture page update failed")

    def _tick(self):
        s = self._session
        if s is not None and s.streams:
            s.check_cameras()
        if s is not None and s.recording:
            self._update_rec_label()
        if not self.isVisible():
            return
        if s is not None:
            status = s.camera_status()
            opening = set(self._opening[1]) if self._opening is not None else set()
            for cam in self.grid.names():
                st = status.get(cam)
                f = s.latest(cam)
                if cam in opening:
                    self.grid.set_message(cam, "opening…")
                elif st is None or not st["running"]:
                    msg = "closed" if st is None or not st.get("error") else st["error"]
                    self.grid.set_message(cam, msg)
                elif f is None:
                    self.grid.set_message(cam, "no frames yet")
                else:
                    text = f"{st['measured_fps']:.0f} fps · {st['size'][0]}x{st['size'][1]}"
                    if st["lost"]:
                        text += " · NO NEW FRAMES"
                    self.grid.set_frame(cam, f.image, text)
            self._status_tick += 1
            if self._status_tick % 6 == 0:
                self._update_cam_status(status)
        self._update_wii_label()
        self._update_cop()

    def _update_cam_status(self, status: dict):
        n_run = 0
        for r, cam in enumerate(self._cam_rows()):
            st = status.get(cam) or {}
            if st.get("running"):
                n_run += 1
                if st.get("lost"):
                    txt = "NO NEW FRAMES"
                elif st.get("size") is None:
                    txt = "no frames yet"
                else:
                    txt = f"{st['measured_fps']:.1f} fps · {st['size'][0]}x{st['size'][1]}"
                if st.get("recording"):
                    txt = "● REC · " + txt
                if st.get("error"):
                    txt += f" · {st['error']}"
            else:
                txt = "closed" if not st.get("error") else f"error: {st['error']}"
            item = self.cam_table.item(r, 6)
            if item is not None and item.text() != txt:
                item.setText(txt)
        total = len(self._cam_rows())
        html = f"<small>{n_run}/{total} cameras open.</small>"
        if self.cams_label.text() != html:
            self.cams_label.setText(html)

    def _update_controls(self):
        s = self._session
        rec = bool(s is not None and s.recording)
        busy = self._export_running()
        has_proj = self.state.project is not None
        open_n = sum(1 for st in (s.streams.values() if s else []) if st.running)
        opening = self._opening is not None
        self.b_open.setEnabled(has_proj and not rec and not opening)
        self.b_open.setText("Opening cameras…" if opening else "▶ Open cameras")
        self.b_close.setEnabled(has_proj and not rec and open_n > 0 and not opening)
        scanning = self._probe_thread is not None
        self.b_scan.setEnabled(has_proj and not rec and not scanning)
        self.b_snap.setEnabled(has_proj and open_n > 0 and not opening)
        self.b_record.setEnabled(has_proj and (rec or not (busy or opening)))
        locked = rec
        self.b_tare.setEnabled(not locked)
        self.b_tare.setToolTip("Tare is locked while recording." if locked else
                               "Zero the board with the last second of data. Nobody may stand "
                               "on the board.")
        if s is not None and not rec and not opening:
            self.cams_label.setText(f"<small>{open_n}/{len(s.settings.cameras)} cameras open."
                                    "</small>" if open_n else "<small>Cameras closed.</small>")
        self._update_take_buttons()

    def _on_balance_changed(self, what: str):
        if what in ("recording", "trial"):
            self._refresh_takes()

    # ================================================================= lifecycle
    def can_close(self) -> bool:
        s = self._session
        if s is not None and s.recording:
            return QMessageBox.question(
                self, "Recording", "A take is being recorded. Stop the recording and quit?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No) == QMessageBox.Yes
        if self._export_running():
            return QMessageBox.question(
                self, "Export", "A take is being exported to videos/. Cancel the export and "
                "quit? (videos/ stays unchanged.)",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No) == QMessageBox.Yes
        return True

    def shutdown(self) -> None:
        self._timer.stop()
        self.state.set_busy("export", None)
        s, self._session = self._session, None
        if s is not None:
            if s.recording:
                self._finish_recording(s)
            try:
                s.close()
            except Exception:  # noqa: BLE001
                log.exception("closing the cameras failed")
        th, w = self._thread, self._worker
        if th is not None:
            if w is not None:
                w.cancel()
            th.quit()
            if not th.wait(15000):
                _ORPHANS.append((th, w))  # never let Python destroy a running QThread
