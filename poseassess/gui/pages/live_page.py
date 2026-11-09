"""Page "3c. Live 3D": real-time 3D skeleton from the project's webcams (prototype).

Uses the cameras of "3b. Capture" (``wii/capture.json``): when they are already open there, the
same streams are used; otherwise this page opens them itself (a webcam cannot be opened twice,
so close them here before recording on 3b). 2D keypoints come from RTMPose (rtmlib, onnxruntime
GPU when available), 3D from ``calibration/Calib.toml``; see ``poseassess.core.realtime``.

Nothing is written into the project unless "Save CSV" is checked (``live/live_<time>.csv``).
The offline pipeline (4. Run: Pose2Sim + OpenSim IK) is unchanged.
"""
from __future__ import annotations

import logging
import threading
import time
from datetime import datetime

import numpy as np
from PySide6.QtCore import QObject, Qt, QTimer, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QFormLayout, QGroupBox, QHBoxLayout, QHeaderView, QLabel,
    QMessageBox, QPushButton, QSizePolicy, QSpinBox, QSplitter, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget,
)

from .base_page import BasePage

log = logging.getLogger(__name__)

SIZES = ["As in 3b. Capture", "1280x720", "960x540", "640x480"]
ANGLE_LABELS = {"RKnee": "Knee R", "LKnee": "Knee L", "RHip": "Hip R", "LHip": "Hip L",
                "RElbow": "Elbow R", "LElbow": "Elbow L", "RShoulder": "Shoulder R",
                "LShoulder": "Shoulder L", "Trunk": "Trunk lean"}


class _Bridge(QObject):
    done = Signal(object)


class LivePage(BasePage):
    nav_title = "3c. Live 3D"

    def __init__(self, state):
        super().__init__(state)
        self._engine = None
        self._own_session = None     # CaptureSession opened by this page (None: 3b's is used)
        self._streams: dict = {}
        self._starting = False
        self._bridge = None
        self._last_seq = -1
        self._last_new = 0.0
        self._disp_lat: list[float] = []
        self._preview_t = 0.0
        self._build()
        self._timer = QTimer(self)
        self._timer.setInterval(33)
        self._timer.timeout.connect(self._tick)

    # ------------------------------------------------------------------- UI
    def _build(self):
        root = QHBoxLayout(self)
        split = QSplitter(Qt.Horizontal)
        root.addWidget(split)

        left = QWidget()
        lv = QVBoxLayout(left)
        box = QGroupBox("Live 3D skeleton (prototype)")
        form = QFormLayout(box)
        self.mode = QComboBox()
        self.mode.addItem("Lightweight (RTMPose-s, fastest)", "lightweight")
        self.mode.addItem("Balanced (RTMPose-m)", "balanced")
        self.mode.addItem("Performance (RTMPose-x, slow)", "performance")
        self.mode.setCurrentIndex(1)
        self.device = QComboBox()
        self.device.addItem("Auto", "auto")
        try:
            from poseassess.core.realtime.pose2d import available_devices
            for d in available_devices():
                self.device.addItem({"cuda": "NVIDIA CUDA", "dml": "DirectML GPU",
                                     "cpu": "CPU"}[d], d)
        except Exception:  # noqa: BLE001
            log.exception("onnxruntime unavailable")
        self.det_every = QSpinBox()
        self.det_every.setRange(1, 30)
        self.det_every.setValue(3)
        self.det_every.setToolTip("Run the person detector every N frames (it is the slowest "
                                  "step); in between the box follows the skeleton.")
        self.size = QComboBox()
        self.size.addItems(SIZES)
        self.size.setCurrentIndex(1)
        self.smooth = QCheckBox("One-Euro smoothing")
        self.smooth.setChecked(True)
        self.only2d = QCheckBox("2D only (no calibration needed)")
        self.save_csv = QCheckBox("Save CSV (live/ in the project)")
        form.addRow("Model:", self.mode)
        form.addRow("Device:", self.device)
        form.addRow("Detector every:", self.det_every)
        form.addRow("Camera size:", self.size)
        form.addRow(self.smooth)
        form.addRow(self.only2d)
        form.addRow(self.save_csv)
        self.start_btn = QPushButton("Start")
        self.start_btn.setMinimumHeight(34)
        self.start_btn.clicked.connect(self._toggle)
        form.addRow(self.start_btn)
        lv.addWidget(box)

        self.stats = QLabel("<small>Stopped.</small>")
        self.stats.setWordWrap(True)
        self.stats.setTextFormat(Qt.RichText)
        self.stats.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        lv.addWidget(self.stats)

        self.angles = QTableWidget(len(ANGLE_LABELS), 1)
        self.angles.setHorizontalHeaderLabels(["deg"])
        self.angles.setVerticalHeaderLabels(list(ANGLE_LABELS.values()))
        self.angles.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.angles.setEditTriggers(QTableWidget.NoEditTriggers)
        for i in range(len(ANGLE_LABELS)):
            self.angles.setItem(i, 0, QTableWidgetItem("-"))
        lv.addWidget(self.angles, 1)
        note = QLabel("<small>Geometric angles for live feedback (knee/elbow: 0 = straight). "
                      "Clinical joint angles: 4. Run (OpenSim IK).</small>")
        note.setWordWrap(True)
        lv.addWidget(note)
        split.addWidget(left)

        right = QWidget()
        rv = QVBoxLayout(right)
        self.previews_row = QHBoxLayout()
        rv.addLayout(self.previews_row)
        self._previews: list[QLabel] = []
        from ..widgets.live_skeleton import LiveSkeletonView
        self.view3d = LiveSkeletonView()
        rv.addWidget(self.view3d, 1)
        split.addWidget(right)
        split.setSizes([300, 800])

    def _set_previews(self, n: int):
        for lbl in self._previews:
            self.previews_row.removeWidget(lbl)
            lbl.deleteLater()
        self._previews = []
        for _ in range(n):
            lbl = QLabel()
            lbl.setFixedHeight(180)
            lbl.setMinimumWidth(240)
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet("background:#1a1c20;color:#888")
            lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            self.previews_row.addWidget(lbl)
            self._previews.append(lbl)

    def _set_controls(self):
        running = self._engine is not None
        self.start_btn.setText("Stop" if running else ("Starting…" if self._starting else "Start"))
        self.start_btn.setEnabled(not self._starting)
        for w in (self.mode, self.device, self.det_every, self.size, self.smooth, self.only2d,
                  self.save_csv):
            w.setEnabled(not running and not self._starting)

    # ------------------------------------------------------------- lifecycle
    def on_project_changed(self, project):
        self._stop()

    def shutdown(self):
        self._stop()

    def _toggle(self):
        if self._engine is not None:
            self._stop()
        else:
            self._start()

    def _start(self):
        proj = self.state.project
        if proj is None or self._starting:
            return
        from poseassess.core.capture.settings import load_capture_settings

        settings = load_capture_settings(proj)
        cams = [c for c, cs in sorted(settings.cameras.items()) if cs.enabled]
        if not cams:
            QMessageBox.warning(self, "Live 3D", "No enabled camera on 3b. Capture.")
            return
        only2d = self.only2d.isChecked()
        if not only2d and len(cams) < 2:
            QMessageBox.warning(self, "Live 3D", "3D needs at least 2 cameras (or check '2D only').")
            return
        if not only2d and not proj.calib_toml.exists():
            QMessageBox.warning(self, "Live 3D", f"No calibration ({proj.calib_toml}).\n\nCalibrate "
                                "on 2. Calibration, or check '2D only' to measure the 2D speed.")
            return
        size = None if self.size.currentIndex() == 0 else tuple(
            int(v) for v in self.size.currentText().split("x"))

        shared = self.state.capture_session
        shared_running = {c for c, s in (shared.streams.items() if shared else []) if s.running}
        started = {}
        if shared_running:
            if not set(cams) <= shared_running or (size and any(
                    (settings.cameras[c].width, settings.cameras[c].height) != size for c in cams)):
                QMessageBox.warning(self, "Live 3D", "Cameras are open on 3b. Capture with other "
                                    "settings. Close them there, or choose 'As in 3b. Capture'.")
                return
            self._streams = {c: shared.streams[c] for c in cams}
        else:
            from poseassess.core.capture.session import CaptureSession

            for c in cams:
                if size:
                    settings.cameras[c].width, settings.cameras[c].height = size
            self._own_session = CaptureSession(proj, settings)
            try:
                started = self._own_session.begin_open()
            except Exception as e:  # noqa: BLE001
                self._own_session = None
                QMessageBox.critical(self, "Live 3D", str(e))
                return

        mode, device, det_every = self.mode.currentData(), self.device.currentData(), self.det_every.value()
        calib_path = None if only2d else proj.calib_toml
        warm_size = size or (settings.cameras[cams[0]].width or 1280,
                             settings.cameras[cams[0]].height or 720)
        self._starting = True
        self._set_controls()
        self.stats.setText("<small>Opening cameras and loading the pose model "
                           "(first start on a GPU can take ~10-30 s)…</small>")
        bridge = self._bridge = _Bridge()
        bridge.done.connect(self._on_started)
        session = self._own_session

        def work():
            out = {"session": session, "started": started, "cams": cams}
            try:
                from poseassess.core.capture.session import open_streams
                from poseassess.core.realtime.pose2d import PoseEstimator, resolve_device

                if started:
                    out["open"] = open_streams(started)
                out["device"] = resolve_device(device)
                if calib_path is not None:
                    from poseassess.core.calib_read import read_camera_poses
                    from poseassess.core.realtime.calib import load_calib

                    allc = load_calib(calib_path)
                    idx = [int(c[3:]) - 1 for c in cams]
                    if max(idx) >= len(allc):
                        raise RuntimeError(f"Calib.toml has {len(allc)} cameras, "
                                           f"{cams[idx.index(max(idx))]} is not calibrated")
                    out["calib"] = [allc[i] for i in idx]
                    poses = read_camera_poses(calib_path)
                    out["poses"] = [poses[i] for i in idx if i < len(poses)]
                ests = []
                for _ in cams:
                    e = PoseEstimator(mode, out["device"], det_every)
                    e.warmup(warm_size)
                    ests.append(e)
                out["ests"] = ests
                active = getattr(ests[0], "active_device", out["device"])
                if active != out["device"]:
                    out["fallback"] = f"{out['device']} not usable, running on {active}"
                    out["device"] = active
            except Exception as e:  # noqa: BLE001
                log.exception("starting the live view failed")
                out["error"] = f"{type(e).__name__}: {e}"
            try:
                bridge.done.emit(out)
            except RuntimeError:
                for st in started.values():
                    st.request_stop()

        threading.Thread(target=work, name="live-start", daemon=True).start()

    def _on_started(self, out):
        self._starting = False
        session = out["session"]
        if session is not None and session is self._own_session:
            errors = session.finish_open(out["started"], out.get("open", {}))
            if errors and "error" not in out:
                out["error"] = "Cameras could not be opened:\n" + "\n".join(
                    f"• {c}: {e}" for c, e in sorted(errors.items())) + (
                    "\n\nClose other programs that use them (or the cameras on 3b. Capture).")
            self._streams = {c: session.streams[c] for c in out["cams"] if c in session.streams}
        elif session is not None:  # project changed meanwhile
            for st in out["started"].values():
                st.request_stop()
            out.setdefault("error", "cancelled")
        if "error" in out:
            self._release()
            self._set_controls()
            self.stats.setText("<small>Stopped.</small>")
            if out["error"] != "cancelled":
                QMessageBox.critical(self, "Live 3D", out["error"])
            return
        from poseassess.core.realtime.engine import EngineConfig, RealtimeEngine

        csv_path = None
        if self.save_csv.isChecked() and self.state.project is not None:
            csv_path = (self.state.project.root / "live"
                        / f"live_{datetime.now():%Y%m%d_%H%M%S}.csv")
        self._csv_path = csv_path
        self._device = out["device"]
        self._engine = RealtimeEngine(self._streams, out["ests"], out.get("calib"),
                                      EngineConfig(smooth=self.smooth.isChecked()),
                                      csv_path=csv_path)
        self._engine.start()
        if out.get("fallback"):
            self._engine.warnings.append(out["fallback"] + " (see README_LIVE.md)")
        self._set_previews(len(self._streams))
        self.view3d.set_cameras(out.get("poses", []))
        self.view3d.reset_view()
        self.view3d.show_points(None)
        self._last_seq = -1
        self._last_new = time.perf_counter()
        self._disp_lat = []
        self._timer.start()
        self._set_controls()

    def _release(self):
        if self._own_session is not None:
            try:
                self._own_session.close()
            except Exception:  # noqa: BLE001
                log.exception("closing the live cameras failed")
        self._own_session = None
        self._streams = {}

    def _stop(self):
        self._timer.stop()
        eng, self._engine = self._engine, None
        if eng is not None:
            eng.stop()
            s = eng.stats.summary()
            self.stats.setText(self._stats_html(s, final=True))
        self._release()
        self._set_controls()

    # ---------------------------------------------------------------- update
    def _tick(self):
        eng = self._engine
        if eng is None:
            return
        if eng.error:
            err = eng.error
            self._stop()
            QMessageBox.critical(self, "Live 3D", err)
            return
        now = time.perf_counter()
        r = eng.latest()
        if r is not None and r.seq != self._last_seq:
            self._last_seq = r.seq
            self._last_new = now
            self._disp_lat.append((now - min(r.t_frames)) * 1000.0)
            self._disp_lat = self._disp_lat[-90:]
            if r.X is not None:
                self.view3d.show_points(r.X)
                for i, k in enumerate(ANGLE_LABELS):
                    v = r.angles.get(k, float("nan"))
                    self.angles.item(i, 0).setText("-" if v != v else f"{v:.0f}")
        if now - self._preview_t >= 0.1:  # previews at ~10 Hz: cheap
            self._preview_t = now
            self._update_previews(r)
            s = eng.stats.summary()
            html = self._stats_html(s)
            if now - self._last_new > 2.0:
                html += "<br><b style='color:#e0a94c'>No new frames - camera stopped?</b>"
            if eng.warnings:
                html += "<br><small style='color:#e0a94c'>" + "<br>".join(eng.warnings) + "</small>"
            self.stats.setText(html)

    def _stats_html(self, s: dict, final: bool = False) -> str:
        def f(v, nd=0):
            return "-" if v != v else f"{v:.{nd}f}"
        disp = float(np.mean(self._disp_lat)) if self._disp_lat else float("nan")
        cams = " · ".join(f"{c} {st.measured_fps:.0f} fps" for c, st in self._streams.items())
        head = "Last run" if final else "Running"
        txt = (f"<b>{head}</b> on {getattr(self, '_device', '?')}<br>"
               f"<span style='font-size:15px'><b>{f(s['fps'], 1)} fps</b> · latency "
               f"<b>{f(s['latency_ms'])} ms</b> (p95 {f(s['latency_p95_ms'])})</span><br>"
               f"<small>on screen ≈ {f(disp)} ms · 2D {f(s['pose2d_ms'])} ms · "
               f"3D {f(s['triangulate_ms'], 1)} ms · camera skew {f(s['skew_ms'])} ms · "
               f"3D joints {f(s['joints_3d'])}/26<br>{cams}<br>"
               "Latency = frame grabbed → 3D ready; add ~30-100 ms of camera exposure/USB."
               "</small>")
        if getattr(self, "_csv_path", None):
            txt += f"<br><small>CSV: {self._csv_path}</small>"
        return txt

    def _update_previews(self, r):
        if not self._previews:
            return
        import cv2

        from poseassess.core.realtime.skeleton import BONE_IDX

        col = {"R": (255, 150, 60), "L": (40, 140, 255), "C": (200, 200, 200)}
        for i, (lbl, (c, st)) in enumerate(zip(self._previews, self._streams.items())):
            fr = st.latest()
            if fr is None:
                lbl.setText(f"{c}: no frame")
                continue
            h, w = fr.image.shape[:2]
            th = lbl.height()
            sc = th / float(h)
            img = cv2.resize(fr.image, (max(1, int(w * sc)), th))
            if r is not None and i < len(r.kpts2d):
                k, s = r.kpts2d[i] * sc, r.scores[i]
                for a, b, sd in BONE_IDX:
                    if s[a] >= 0.3 and s[b] >= 0.3:
                        cv2.line(img, (int(k[a, 0]), int(k[a, 1])), (int(k[b, 0]), int(k[b, 1])),
                                 col[sd], 2, cv2.LINE_AA)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            qimg = QImage(img.data, img.shape[1], img.shape[0], img.strides[0],
                          QImage.Format_RGB888).copy()
            lbl.setPixmap(QPixmap.fromImage(qimg))
