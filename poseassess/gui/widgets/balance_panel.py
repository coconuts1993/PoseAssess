"""``BalancePanel``: the "Balance (Wii)" tab of the Results page.

Owner: GUI-VIEW. Content: ``AlignmentPanel`` (which recording, how it is aligned), balance
metrics table (COP sway, COM sway, COM-COP), ``BalancePlotCanvas`` (force, COP, COM vs .trc
time, events), time window selection, export (fused CSV + summary JSON + GRF .mot).

Layout: a fixed-width left column (recording + alignment, analysis window, export) and, on the
right, the metrics table next to the plot tabs ("Signals", "Sway path", "Alignment check").
Without usable Wii data the right side shows what to do instead.

The panel recomputes (``fusion.fuse_trial`` + ``balance_summary``) on ``refresh()`` (also the
"Reload" button); project and ``balance_changed`` notifications only mark it dirty while it is
hidden, it refreshes when shown. When shown it also refreshes if the files it depends on changed
on disk meanwhile (a pipeline run wrote or replaced a .trc, ``run_wii.bat --project`` added a
recording, ...): ``_disk_signature``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMenu,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSplitter,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .balance_plots import BalancePlotCanvas
from .wii_alignment import WARN, AlignmentPanel

log = logging.getLogger(__name__)

LEFT_WIDTH = 290

HELP_HTML = """
<h3>Balance analysis with the Wii Balance Board</h3>
<p style='color:{color}'><b>{reason}</b></p>
<p><b>A. Record here (recommended):</b> locate the board on <i>2b. Wii Board</i>, then record
the trial videos and the Wii data together on <i>3b. Capture</i> and export the take to
videos/. They are aligned by the frame timestamps (set the camera latency on 3b. Capture; check it
once with a jump: <i>Check timing with a sync event</i>).</p>
<p><b>B. Videos recorded elsewhere:</b> record the Wii data only (<i>3b. Capture</i> with
"Record cameras" off, or <tt>run_wii.bat</tt>), import the videos on <i>3. Videos</i> and the
Wii recording here (<i>Import Wii recording…</i>). Ask the subject for <b>2 small jumps or 2
stomps</b> on the board at the start of the trial, then click <i>Detect sync event</i> (or set
the offset by eye in <i>Alignment check</i>).</p>
<p>Then run the pipeline (<i>4. Run</i>). This tab shows the COP sway, the COM–COP relation
and the force plots, and exports the fused per-frame table. <i>5. 3D View</i> shows the board,
COP, force and COM together with the skeleton.</p>
"""

# (label, key) rows of the metrics table; COP and COM columns (sway_metrics keys)
SWAY_ROWS = [
    ("Mean ML (mm)", "mean_x_mm"), ("Mean AP (mm)", "mean_y_mm"),
    ("Range ML (mm)", "range_ml_mm"), ("Range AP (mm)", "range_ap_mm"),
    ("RMS ML (mm)", "rms_ml_mm"), ("RMS AP (mm)", "rms_ap_mm"),
    ("Path length (mm)", "path_length_mm"), ("Mean velocity (mm/s)", "mean_velocity_mm_s"),
    ("95 % ellipse (mm²)", "ellipse95_area_mm2"), ("Samples", "samples"),
]
COM_COP_ROWS = [
    ("RMS distance (mm)", "rms_distance_mm"), ("RMS ML (mm)", "rms_ml_mm"),
    ("RMS AP (mm)", "rms_ap_mm"), ("Mean ML (mm)", "mean_ml_mm"),
    ("Mean AP (mm)", "mean_ap_mm"), ("Correlation ML (r)", "corr_ml"),
    ("Correlation AP (r)", "corr_ap"), ("Frames", "frames"),
]


def _fmt(v, key: str = "") -> str:
    if v is None:
        return "—"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if not np.isfinite(f):
        return "—"
    if key in ("samples", "frames"):
        return f"{int(round(f))}"
    if key.startswith("corr"):
        return f"{f:.2f}"
    if key in ("ellipse95_area_mm2", "path_length_mm"):
        return f"{f:.0f}"
    return f"{f:.1f}"


class BalancePanel(QWidget):
    """Results page tab. ``state`` is the shared ``AppState``."""

    def __init__(self, state, parent=None):
        super().__init__(parent)
        self.state = state
        self._fused = None
        self._summary: dict | None = None
        self._dirty = True
        self._loaded_sig = None  # _disk_signature() of the last refresh
        self._trc_key: str | None = None
        self._all_force_events: list[dict] = []
        self._trial_range = (0.0, 0.0)  # exact .trc time range (the spin boxes round to 0.01 s)

        root = QHBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)

        # ---- left column ------------------------------------------------
        col = QWidget()
        cl = QVBoxLayout(col)
        cl.setContentsMargins(0, 0, 6, 0)
        g1 = QGroupBox("Wii recording && alignment")
        g1l = QVBoxLayout(g1)
        self.alignment = AlignmentPanel(state)
        g1l.addWidget(self.alignment)
        cl.addWidget(g1)

        g2 = QGroupBox("Analysis")
        grid = QGridLayout(g2)
        grid.addWidget(QLabel("Trajectory:"), 0, 0)
        self.trc_combo = QComboBox()
        self.trc_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.trc_combo.setMinimumContentsLength(6)  # the column is LEFT_WIDTH wide
        self.trc_combo.setToolTip("The .trc of pose-3d/ to fuse with the Wii data (same default "
                                  "as the 3D View: filtered first).")
        self.trc_combo.currentIndexChanged.connect(self._on_trc_changed)
        self.reload_btn = QPushButton("Reload")
        self.reload_btn.setToolTip("Read the .trc files, the Wii recordings and the alignment "
                                   "from disk again (e.g. after running the pipeline).")
        self.reload_btn.clicked.connect(self.refresh)
        trow = QHBoxLayout()
        trow.setContentsMargins(0, 0, 0, 0)
        trow.addWidget(self.trc_combo, 1)
        trow.addWidget(self.reload_btn)
        grid.addLayout(trow, 0, 1, 1, 2)
        grid.addWidget(QLabel("From:"), 1, 0)
        self.from_spin = self._time_spin("Start of the analysis window (.trc time).")
        grid.addWidget(self.from_spin, 1, 1, 1, 2)
        grid.addWidget(QLabel("To:"), 2, 0)
        self.to_spin = self._time_spin("End of the analysis window (.trc time).")
        grid.addWidget(self.to_spin, 2, 1, 1, 2)
        self.whole_btn = QPushButton("Whole trial")
        self.whole_btn.setToolTip("Analyse every frame of the .trc.")
        self.whole_btn.clicked.connect(self._whole_trial)
        self.events_btn = QPushButton("Events")
        self.events_btn.setToolTip("Start or end the window at a marked or detected event.")
        self.events_menu = QMenu(self.events_btn)
        self.events_btn.setMenu(self.events_menu)
        grid.addWidget(self.whole_btn, 3, 1)
        grid.addWidget(self.events_btn, 3, 2)
        grid.addWidget(QLabel("Body mass:"), 4, 0)
        self.mass_spin = QDoubleSpinBox()
        self.mass_spin.setRange(0.0, 400.0)
        self.mass_spin.setDecimals(1)
        self.mass_spin.setSingleStep(0.5)
        self.mass_spin.setSuffix(" kg")
        self.mass_spin.setSpecialValueText("auto (measured)")
        self.mass_spin.setKeyboardTracking(False)
        self.mass_spin.setToolTip("Body mass for the % body weight and the body-weight line. "
                                  "'auto' = measured from the quiet standing load.")
        self.mass_spin.editingFinished.connect(self._save_body_mass)
        grid.addWidget(self.mass_spin, 4, 1, 1, 2)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(2, 1)
        cl.addWidget(g2)

        g3 = QGroupBox("Export")
        g3l = QVBoxLayout(g3)
        self.export_btn = QPushButton("Export fused table…")
        self.export_btn.setToolTip("Write the fused table (one row per .trc frame: time, "
                                   "force, COP, COM), the summary (JSON) and the ground reaction "
                                   "force for OpenSim (.mot) to a folder.")
        self.export_btn.clicked.connect(self._export)
        g3l.addWidget(self.export_btn)
        lbl = QLabel("<small>Default folder: wii/exports/. The summary covers the analysis "
                     "window; the table covers every frame.</small>")
        lbl.setWordWrap(True)
        g3l.addWidget(lbl)
        cl.addWidget(g3)
        cl.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidget(col)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFixedWidth(LEFT_WIDTH)
        root.addWidget(scroll)

        # ---- right side -------------------------------------------------
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)
        self.info = QLabel("")
        self.info.setWordWrap(True)
        self.info.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        rv.addWidget(self.info)

        self.stack = QStackedWidget()
        self.help = QLabel("")
        self.help.setWordWrap(True)
        self.help.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        help_scroll = QScrollArea()
        help_scroll.setWidget(self.help)
        help_scroll.setWidgetResizable(True)
        help_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.stack.addWidget(help_scroll)

        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Metric", "COP (Wii)", "COM (markers)"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.table.setWordWrap(False)
        self.table.verticalHeader().setDefaultSectionSize(22)
        self.table.setToolTip("Board frame: ML = medio-lateral (+ = subject's right), AP = "
                              "antero-posterior (+ = front). COP: Wii data resampled to 100 Hz "
                              "and low-pass filtered at 10 Hz (samples = raw Wii samples); COM "
                              "at the .trc rate. COM − COP: the COM plumb point minus the COP; "
                              "r = correlation of the COM and COP positions.")
        self.plot = BalancePlotCanvas(6, 5)
        self.path_plot = BalancePlotCanvas(5, 5)
        self.plot_tabs = QTabWidget()
        self.plot_tabs.addTab(self.plot, "Signals")
        self.plot_tabs.addTab(self.path_plot, "Sway path")
        self.plot_tabs.addTab(self._with_toolbar(self.alignment.plot), "Alignment check")
        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(self.table)
        split.addWidget(self.plot_tabs)
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 2)
        split.setSizes([320, 600])
        self.stack.addWidget(split)
        rv.addWidget(self.stack, 1)
        root.addWidget(right, 1)

        self._window_timer = QTimer(self)
        self._window_timer.setSingleShot(True)
        self._window_timer.setInterval(250)
        self._window_timer.timeout.connect(self._update_summary)
        self.from_spin.valueChanged.connect(self._window_changed)
        self.to_spin.valueChanged.connect(self._window_changed)

        self.alignment.plot_requested.connect(lambda: self.plot_tabs.setCurrentIndex(2))
        state.project_changed.connect(self._mark_dirty)
        state.balance_changed.connect(self._mark_dirty)
        self._show_help("Open a project first.", color="#8a8f98")
        self._set_controls(False)

    # ---- helpers ------------------------------------------------------------ #
    @staticmethod
    def _with_toolbar(canvas) -> QWidget:
        """``canvas`` below a matplotlib zoom / pan toolbar."""
        from matplotlib.backends.backend_qtagg import NavigationToolbar2QT

        box = QWidget()
        lay = QVBoxLayout(box)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        bar = NavigationToolbar2QT(canvas, box, coordinates=False)
        bar.setIconSize(bar.iconSize() * 0.75)
        lay.addWidget(bar)
        lay.addWidget(canvas, 1)
        return box

    @staticmethod
    def _time_spin(tip: str) -> QDoubleSpinBox:
        s = QDoubleSpinBox()
        s.setDecimals(2)
        s.setSingleStep(0.1)
        s.setSuffix(" s")
        s.setRange(0.0, 0.0)
        s.setKeyboardTracking(False)
        s.setToolTip(tip)
        return s

    def _set_controls(self, on: bool) -> None:
        for w in (self.from_spin, self.to_spin, self.whole_btn, self.events_btn,
                  self.export_btn):
            w.setEnabled(on)

    def _mark_dirty(self, *_):
        self._dirty = True
        if self.isVisible():
            self.refresh()

    def showEvent(self, e):
        super().showEvent(e)
        if self._dirty or self._disk_signature() != self._loaded_sig:
            self.refresh()

    def _disk_signature(self):
        """What ``refresh`` read from disk: the .trc files of pose-3d/ (name, mtime), the Wii
        recordings, trial.json, board.json and the calibration file (mtimes). A pipeline run,
        ``run_wii.bat --project`` or another program changes it without any notification."""
        proj = self.state.project
        if proj is None:
            return None

        def mt(p):
            try:
                return Path(p).stat().st_mtime_ns
            except (OSError, TypeError):
                return None

        try:
            from poseassess.core.balance.calib import find_calib_toml
            from poseassess.core.balance.paths import WiiPaths
            from poseassess.core.balance.trc import find_trc_files

            paths = WiiPaths(proj)
            return (str(proj.root), tuple((t.name, mt(t)) for t in find_trc_files(proj)),
                    tuple(paths.list_recordings()), mt(paths.trial_json), mt(paths.board_json),
                    mt(find_calib_toml(proj)))
        except OSError:
            return None

    def shutdown(self) -> None:
        """Stop the alignment worker (app closing)."""
        self._window_timer.stop()
        self.alignment.shutdown()

    @property
    def fused(self):
        """The current ``FusedTrial`` (None when unavailable)."""
        return self._fused

    @property
    def summary(self) -> dict | None:
        """The ``balance_summary`` of the current window."""
        return self._summary

    # ---- refresh ------------------------------------------------------------ #
    def refresh(self) -> None:
        """Reload trial / board / recording from disk and recompute (project or
        ``state.balance_changed``)."""
        self._dirty = False
        self._loaded_sig = self._disk_signature()
        proj = self.state.project
        self._fill_trc_combo(proj)
        trc = self.trc_combo.currentData()
        self.alignment.set_trc(trc)
        try:
            self.alignment.refresh()
        except Exception:  # noqa: BLE001
            log.exception("alignment panel refresh failed")
        self._fused = None
        self._summary = None
        if proj is None:
            self._show_help("Open a project first.", color="#8a8f98")
            self._set_controls(False)
            return
        self._load_body_mass(proj)
        try:
            from poseassess.core.balance.fusion import fuse_trial
            fused = fuse_trial(proj, trc_path=trc)
        except NotImplementedError:
            self._show_help("The balance analysis is not available yet.")
            return
        except (ValueError, OSError) as e:
            self._show_help(str(e))
            return
        except Exception as e:  # noqa: BLE001
            log.exception("fuse_trial failed")
            self._show_help(f"The Wii data could not be combined with the .trc: "
                            f"{type(e).__name__}: {e}")
            return
        self._fused = fused
        self._setup_window(fused, same_trc=(str(fused.trc_path) == self._trc_key))
        self._trc_key = str(fused.trc_path)
        self._all_force_events = []
        try:
            from poseassess.core.balance.fusion import balance_summary
            self._all_force_events = balance_summary(fused).get("force_events") or []
        except Exception:  # noqa: BLE001
            log.exception("balance_summary failed")
        self._fill_events_menu()
        self._set_controls(True)
        self.stack.setCurrentIndex(1)
        self._update_summary()

    def _fill_trc_combo(self, proj) -> None:
        keep = self.trc_combo.currentData()
        self.trc_combo.blockSignals(True)
        self.trc_combo.clear()
        if proj is not None:
            from poseassess.core.balance.trc import find_trc_files
            for t in find_trc_files(proj):
                n = t.name.lower()
                tag = "augmented" if "lstm" in n else ("filtered" if "filt" in n else "raw")
                self.trc_combo.addItem(f"{t.name}  ·  {tag}", str(t))
            i = self.trc_combo.findData(keep) if keep else -1
            self.trc_combo.setCurrentIndex(i if i >= 0 else (0 if self.trc_combo.count() else -1))
        self.trc_combo.setEnabled(self.trc_combo.count() > 0)
        self.trc_combo.blockSignals(False)

    def _on_trc_changed(self, *_):
        self._mark_dirty()

    def _show_help(self, reason: str, color: str = WARN) -> None:
        self._fused = None
        self._summary = None
        self.help.setText(HELP_HTML.format(reason=reason, color=color))
        self.stack.setCurrentIndex(0)
        self.info.clear()
        self.table.setRowCount(0)
        self.plot.clear()
        self.path_plot.clear()
        self._set_controls(False)

    # ---- window --------------------------------------------------------------- #
    def _setup_window(self, fused, same_trc: bool) -> None:
        tt = np.asarray(fused.trc_time, float)
        tt = tt[np.isfinite(tt)]
        lo, hi = (float(tt.min()), float(tt.max())) if len(tt) else (0.0, 0.0)
        self._trial_range = (lo, hi)
        old = (self.from_spin.value(), self.to_spin.value())
        for s in (self.from_spin, self.to_spin):
            s.blockSignals(True)
            s.setRange(lo, hi)
        if same_trc and old[1] > old[0]:
            self.from_spin.setValue(max(lo, min(old[0], hi)))
            self.to_spin.setValue(max(lo, min(old[1], hi)))
        else:
            self.from_spin.setValue(lo)
            self.to_spin.setValue(hi)
        for s in (self.from_spin, self.to_spin):
            s.blockSignals(False)

    def window(self) -> tuple[float, float]:
        a, b = self.from_spin.value(), self.to_spin.value()
        a, b = (a, b) if a <= b else (b, a)
        # A spin box at its end means the end of the trial: use the exact .trc time (the spin
        # boxes show 0.01 s, which would drop the first or last frame).
        lo, hi = self._trial_range
        if a <= self.from_spin.minimum() + 1e-9:
            a = min(a, lo)
        if b >= self.to_spin.maximum() - 1e-9:
            b = max(b, hi)
        return a, b

    def set_window(self, t_from: float, t_to: float) -> None:
        self.from_spin.setValue(t_from)
        self.to_spin.setValue(t_to)
        self._window_timer.stop()
        self._update_summary()

    def _whole_trial(self) -> None:
        self.set_window(self.from_spin.minimum(), self.to_spin.maximum())

    def _window_changed(self, *_):
        self._window_timer.start()

    def _event_list(self) -> list[tuple[str, float]]:
        out = []
        f = self._fused
        for e in (f.events if f is not None else []):
            t = e.get("trc_time")
            if t is not None and np.isfinite(t):
                out.append((f"{e.get('label', 'event')}", float(t)))
        for e in self._all_force_events:
            t = e.get("trc_time")
            if t is not None and np.isfinite(t):
                out.append((f"{e.get('type', 'event')} (detected)", float(t)))
        lo, hi = self.from_spin.minimum(), self.to_spin.maximum()
        return sorted((x for x in out if lo - 1e-6 <= x[1] <= hi + 1e-6), key=lambda x: x[1])

    def _fill_events_menu(self) -> None:
        self.events_menu.clear()
        evs = self._event_list()
        if not evs:
            a = self.events_menu.addAction("No events in this trial")
            a.setEnabled(False)
            return
        start = self.events_menu.addMenu("Start at")
        end = self.events_menu.addMenu("End at")
        for label, t in evs:
            start.addAction(f"{label}  ({t:.2f} s)",
                            lambda t=t: self.set_window(t, max(t, self.to_spin.value())))
            end.addAction(f"{label}  ({t:.2f} s)",
                          lambda t=t: self.set_window(min(t, self.from_spin.value()), t))

    # ---- body mass -------------------------------------------------------------- #
    def _load_body_mass(self, proj) -> None:
        from poseassess.core.balance.trial import load_trial
        bm = load_trial(proj).body_mass_kg
        self.mass_spin.blockSignals(True)
        self.mass_spin.setValue(float(bm) if bm else 0.0)
        self.mass_spin.blockSignals(False)

    def _save_body_mass(self) -> None:
        proj = self.state.project
        if proj is None:
            return
        from poseassess.core.balance.trial import load_trial, save_trial
        v = float(self.mass_spin.value())
        new = v if v > 0 else None
        info = load_trial(proj)
        if (info.body_mass_kg or None) == new:
            return
        info.body_mass_kg = new
        try:
            save_trial(proj, info)
        except OSError as e:
            QMessageBox.warning(self, "Body mass", str(e))
            return
        self.state.notify_balance_changed("trial")

    # ---- summary ------------------------------------------------------------ #
    def _update_summary(self) -> None:
        f = self._fused
        if f is None:
            return
        a, b = self.window()
        try:
            from poseassess.core.balance.fusion import balance_summary
            s = balance_summary(f, a, b)
        except Exception as e:  # noqa: BLE001
            log.exception("balance_summary failed")
            self.info.setText(f"<span style='color:{WARN}'>⚠ Cannot compute the metrics: "
                              f"{e}</span>")
            return
        self._summary = s
        self._fill_table(s)
        try:
            self.plot.plot_fused(f, a, b, force_events=self._all_force_events)
            self.path_plot.plot_path(f, a, b)
        except Exception:  # noqa: BLE001
            log.exception("balance plot failed")
        self._set_info(f, s)

    def _set_info(self, f, s: dict) -> None:
        al = f.alignment
        head = [f"<b>{Path(f.trc_path).name}</b>", f"recording {f.recording}",
                f"alignment: {al.method}" + (f" {al.offset_s:+.3f} s"
                                             if al.method not in ("none", "recorded") else ""),
                f"window {s['window'][0]:.2f}–{s['window'][1]:.2f} s ({s['frames']} frames)"]
        lines = [" · ".join(head)]
        for w in s.get("warnings") or []:
            lines.append(f"<span style='color:{WARN}'>⚠ {w}</span>")
        if not s.get("cop") and al.method != "none":
            lines.append(f"<span style='color:{WARN}'>⚠ Too few force samples with someone on "
                         "the board in this window for COP metrics.</span>")
        self.info.setText("<br>".join(lines))

    def _fill_table(self, s: dict) -> None:
        t = self.table
        t.setRowCount(0)
        cop, com = s.get("cop") or {}, s.get("com") or {}
        self._section("Sway (board frame)")
        for label, key in SWAY_ROWS:
            self._row(label, _fmt(cop.get(key), key), _fmt(com.get(key), key))
        cc = s.get("com_cop") or {}
        self._section("COM − COP (board frame)")
        for label, key in COM_COP_ROWS:
            self._row(label, _fmt(cc.get(key), key))
        self._section("Load")
        mean = s.get("mean_total_kg")
        bm = s.get("body_mass_kg")
        entered = bool(self.mass_spin.value() > 0)
        self._row("Mean load (kg)", _fmt(mean))
        self._row("Body mass (kg)", _fmt(bm) + ("  (entered)" if entered else "  (measured)"
                                               if bm is not None else ""))
        self._row("Mean load (% BW)",
                  _fmt(100 * mean / bm if mean is not None and bm else None))
        self._section("Window")
        w = s.get("window") or [None, None]
        self._row("From – to (s)", f"{w[0]:.2f} – {w[1]:.2f}" if None not in w[:2] else "—")
        self._row("Duration (s)", _fmt(s.get("duration_s")))
        self._row("Frames", _fmt(s.get("frames"), "frames"))
        self._row("Marked events", f"{len(s.get('events') or [])}")
        self._row("Detected events", f"{len(s.get('force_events') or [])}")
        t.resizeColumnToContents(1)
        t.resizeColumnToContents(2)

    def _section(self, title: str) -> None:
        t = self.table
        r = t.rowCount()
        t.insertRow(r)
        it = QTableWidgetItem(title)
        f = it.font()
        f.setBold(True)
        it.setFont(f)
        it.setBackground(Qt.GlobalColor.lightGray)
        t.setItem(r, 0, it)
        t.setSpan(r, 0, 1, 3)

    def _row(self, label: str, a: str, b: str | None = None) -> None:
        t = self.table
        r = t.rowCount()
        t.insertRow(r)
        t.setItem(r, 0, QTableWidgetItem(label))
        for c, v in ((1, a), (2, b)):
            if v is None:
                continue
            it = QTableWidgetItem(v)
            it.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            t.setItem(r, c, it)
        if b is None:
            t.setSpan(r, 1, 1, 2)

    # ---- export --------------------------------------------------------------- #
    def _export(self) -> None:
        proj = self.state.project
        if proj is None or self._fused is None:
            return
        from poseassess.core.balance.paths import WiiPaths
        default = WiiPaths(proj).exports_dir
        try:
            default.mkdir(parents=True, exist_ok=True)
        except OSError:
            default = proj.root
        out = QFileDialog.getExistingDirectory(self, "Export the fused table to a folder",
                                               str(default))
        if out:
            self.export_to(out)

    def export_to(self, out_dir) -> dict | None:
        """``fusion.export_all`` of the current trial and window into ``out_dir``."""
        proj = self.state.project
        if proj is None or self._fused is None:
            return None
        try:
            from poseassess.core.balance.fusion import export_all
            paths = export_all(proj, self._fused, out_dir, *self.window())
        except NotImplementedError:
            QMessageBox.information(self, "Export", "The export is not available yet.")
            return None
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "Export failed", str(e))
            return None
        lines = "\n".join(f"• {Path(p).name}" for p in paths.values())
        extra = "" if "grf" in paths else ("\n\nNo ground reaction force .mot: the board "
                                           "position is not known (2b. Wii Board).")
        QMessageBox.information(self, "Export", f"Wrote to {out_dir}:\n{lines}{extra}")
        return paths
