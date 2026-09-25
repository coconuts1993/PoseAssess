"""Replay of a Wii Balance Board recording: COP trajectory in real time on a top-down board
view, load / COP curves with a cursor, play / pause / speed / seek.

Used by the "2b. Wii Board" page ("Replay file" tab) and by the stand-alone viewer
(``python -m poseassess.wii.viewer``, run_wii_viewer.bat). Files: see
``poseassess.wii.viewer.load_trace``. Opening a file jumps to 1 s before the subject steps on
the board. Keys (while the replay has focus): Space = play / pause, Left / Right = -/+ 1 s,
Home = start.
"""

from __future__ import annotations

import time

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (QComboBox, QFileDialog, QHBoxLayout, QLabel, QMessageBox,
                               QPushButton, QSlider, QSplitter, QVBoxLayout, QWidget)

from poseassess.gui.widgets.cop_view import CopView
from poseassess.wii import io as wio
from poseassess.wii.viewer import SPEEDS, TRAILS, WiiTrace, load_trace


class WiiReplayWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.trace: WiiTrace | None = None
        self.pos = 0.0  # playback time (s)
        self.playing = False
        self._last = time.perf_counter()
        self.start_dir = ""

        root = QVBoxLayout(self)

        bar = QHBoxLayout()
        self.open_btn = QPushButton("Open recording…")
        self.open_btn.setToolTip("A file of the original Wii program, a wii.csv or a recording "
                                 "folder's wii.csv. You can also drop a file here.")
        self.open_btn.clicked.connect(self.open_dialog)
        self.play_btn = QPushButton("▶ Play")
        self.play_btn.clicked.connect(self.toggle)
        self.speed = QComboBox()
        self.speed.addItems(SPEEDS)
        self.speed.setCurrentText("1x")
        self.trail = QComboBox()
        self.trail.addItems(list(TRAILS))
        self.trail.setCurrentText("5 s")
        self.trail.currentTextChanged.connect(lambda _: self.show_time(self.pos))
        self.file_lbl = QLabel("Open a Wii recording (or drop a file here).")
        bar.addWidget(self.open_btn)
        bar.addWidget(self.play_btn)
        bar.addWidget(QLabel("Speed"))
        bar.addWidget(self.speed)
        bar.addWidget(QLabel("Trail"))
        bar.addWidget(self.trail)
        bar.addWidget(self.file_lbl, 1)
        root.addLayout(bar)

        split = QSplitter(Qt.Horizontal)
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        self.board = CopView()
        self.board.setMinimumSize(300, 220)
        lv.addWidget(self.board, 1)
        self.readout = QLabel("")
        self.readout.setTextFormat(Qt.RichText)
        lv.addWidget(self.readout)
        split.addWidget(left)

        self.plots = pg.GraphicsLayoutWidget()
        self.plots.setBackground("w")
        self.p_load = self.plots.addPlot(row=0, col=0, title="Load (kg)")
        self.p_x = self.plots.addPlot(row=1, col=0, title="COP medio-lateral (mm, + right)")
        self.p_y = self.plots.addPlot(row=2, col=0, title="COP antero-posterior (mm, + front)")
        self.p_y.setLabel("bottom", "time (s)", color="#333333")
        for p in (self.p_x, self.p_y):
            p.setXLink(self.p_load)
        self.curves = [self.p_load.plot(pen=pg.mkPen((40, 150, 60), width=1.5)),
                       self.p_x.plot(pen=pg.mkPen((200, 50, 50), width=1.5)),
                       self.p_y.plot(pen=pg.mkPen((40, 90, 200), width=1.5))]
        self.cursors = [pg.InfiniteLine(angle=90, pen=pg.mkPen((120, 120, 120), width=1))
                        for _ in range(3)]
        for p, c in zip((self.p_load, self.p_x, self.p_y), self.cursors):
            p.addItem(c)
            p.showGrid(x=True, y=True, alpha=0.3)
            p.setTitle(p.titleLabel.text, color="#333333")  # dark text on the white background
            for ax in ("left", "bottom"):
                p.getAxis(ax).setTextPen("#333333")
                p.getAxis(ax).setPen("#888888")
        split.addWidget(self.plots)
        split.setSizes([460, 700])
        root.addWidget(split, 1)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 1000)
        self.slider.sliderMoved.connect(self._on_slider)
        self.slider.sliderPressed.connect(lambda: self._on_slider(self.slider.value()))
        self.time_lbl = QLabel("")
        srow = QHBoxLayout()
        srow.addWidget(self.slider, 1)
        srow.addWidget(self.time_lbl)
        root.addLayout(srow)
        self.stats_lbl = QLabel("")
        self.stats_lbl.setWordWrap(True)
        root.addWidget(self.stats_lbl)

        for key, fn in (("Space", self.toggle), ("Left", lambda: self.seek(self.pos - 1)),
                        ("Right", lambda: self.seek(self.pos + 1)),
                        ("Home", lambda: self.seek(0.0))):
            sc = QShortcut(QKeySequence(key), self)
            sc.setContext(Qt.WidgetWithChildrenShortcut)  # only while the replay has focus
            sc.activated.connect(fn)

        self.timer = QTimer(self)
        self.timer.setInterval(30)
        self.timer.timeout.connect(self._tick)
        self._set_enabled(False)

    # ---- loading --------------------------------------------------------------- #
    def open_dialog(self):
        f, _ = QFileDialog.getOpenFileName(
            self, "Open Wii recording", self.start_dir,
            "Wii recordings (*.csv *.txt *.dat);;All files (*)")
        if f:
            self.load(f)

    def load(self, path) -> bool:
        try:
            tr = load_trace(path)
        except (OSError, ValueError, KeyError) as e:
            QMessageBox.warning(self, "Wii replay", str(e))
            return False
        self.trace = tr
        self.pause()
        self.file_lbl.setText(f"<b>{tr.path.name}</b> · {tr.kind} · {len(tr.t)} samples · "
                              f"{tr.duration:.1f} s · "
                              f"{(len(tr.t) - 1) / max(tr.duration, 1e-9):.0f} Hz")
        self.curves[0].setData(tr.t, tr.total_kg)
        self.curves[1].setData(tr.t, tr.cop_x * 1000, connect="finite")
        self.curves[2].setData(tr.t, tr.cop_y * 1000, connect="finite")
        self.p_load.setXRange(0, tr.duration, padding=0.01)
        self.stats_lbl.setText(self._stats(tr))
        self._set_enabled(True)
        self.seek(self.start_time(tr))
        self.setFocus()
        return True

    @staticmethod
    def start_time(tr: WiiTrace) -> float:
        """1 s before the subject first steps on the board (0 when on it from the start)."""
        on = np.flatnonzero(np.isfinite(tr.cop_x))
        return max(0.0, float(tr.t[on[0]]) - 1.0) if len(on) else 0.0

    @staticmethod
    def _stats(tr: WiiTrace) -> str:
        on = np.isfinite(tr.cop_x)
        if on.sum() < 10:
            return "Nobody on the board in this recording (load < 5 kg)."
        from poseassess.core.balance.analysis import sway_metrics
        m = sway_metrics(tr.t[on], tr.cop_x[on], tr.cop_y[on])
        load = np.nanmedian(tr.total_kg[on])
        txt = (f"On the board {on.mean() * 100:.0f} % of the time · median load {load:.1f} kg"
               f" · COP range ML {m.get('range_ml_mm', np.nan):.0f} mm, "
               f"AP {m.get('range_ap_mm', np.nan):.0f} mm · path "
               f"{m.get('path_length_mm', np.nan):.0f} mm · mean velocity "
               f"{m.get('mean_velocity_mm_s', np.nan):.1f} mm/s")
        if tr.clock_note:
            txt += f" · {tr.clock_note}"
        return txt

    def _set_enabled(self, on: bool):
        for w in (self.play_btn, self.slider, self.speed, self.trail):
            w.setEnabled(on)

    # ---- playback -------------------------------------------------------------- #
    def toggle(self):
        if self.trace is None:
            return
        self.pause() if self.playing else self.play()

    def play(self):
        if self.trace is None:
            return
        if self.pos >= self.trace.duration:
            self.pos = 0.0
        self.playing = True
        self._last = time.perf_counter()
        self.play_btn.setText("⏸ Pause")
        self.timer.start()

    def pause(self):
        self.playing = False
        self.timer.stop()
        self.play_btn.setText("▶ Play")

    def _tick(self):
        now = time.perf_counter()
        dt, self._last = now - self._last, now
        speed = float(self.speed.currentText().rstrip("x"))
        self.pos += dt * speed
        if self.pos >= self.trace.duration:
            self.pos = self.trace.duration
            self.pause()
        self.show_time(self.pos)

    def seek(self, t: float):
        if self.trace is None:
            return
        self.pos = float(np.clip(t, 0.0, self.trace.duration))
        self._last = time.perf_counter()
        self.show_time(self.pos)

    def _on_slider(self, v: int):
        if self.trace is not None:
            self.seek(v / 1000 * self.trace.duration)

    def show_time(self, t: float):
        tr = self.trace
        if tr is None:
            return
        i = tr.index_at(t)
        span = TRAILS.get(self.trail.currentText())
        j0 = 0 if span is None else tr.index_at(t - span)
        trail = np.column_stack([tr.cop_x[j0:i + 1], tr.cop_y[j0:i + 1]])
        load = float(tr.total_kg[i]) if np.isfinite(tr.total_kg[i]) else 0.0
        on = np.isfinite(tr.cop_x[i])
        self.board.set_data(trail, np.zeros((0, 2)), load, "" if on else "(nobody on the board)")
        for c in self.cursors:
            c.setValue(t)
        self.slider.blockSignals(True)
        self.slider.setValue(int(round(t / max(tr.duration, 1e-9) * 1000)))
        self.slider.blockSignals(False)
        clock = wio.local_time(tr.t_unix[i])
        self.time_lbl.setText(f"{t:7.2f} / {tr.duration:.2f} s" + (f"  ·  {clock}" if clock else ""))
        s = tr.sensors
        cop = f"x {tr.cop_x[i] * 1000:+6.1f} mm, y {tr.cop_y[i] * 1000:+6.1f} mm" if on else "—"
        self.readout.setText(
            f"<table><tr><td>Load</td><td><b>{load:6.1f} kg</b></td><td width=20></td>"
            f"<td>COP</td><td><b>{cop}</b></td></tr>"
            f"<tr><td>TL</td><td>{s['TL_kg'][i]:5.1f} kg</td><td></td>"
            f"<td>TR</td><td>{s['TR_kg'][i]:5.1f} kg</td></tr>"
            f"<tr><td>BL</td><td>{s['BL_kg'][i]:5.1f} kg</td><td></td>"
            f"<td>BR</td><td>{s['BR_kg'][i]:5.1f} kg</td></tr></table>")

    # ---- drag & drop / lifetime ---------------------------------------------------- #
    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        urls = e.mimeData().urls()
        if urls:
            self.load(urls[0].toLocalFile())

    def hideEvent(self, e):
        self.pause()  # page switched away / window closed
        super().hideEvent(e)

    def shutdown(self):
        self.pause()
