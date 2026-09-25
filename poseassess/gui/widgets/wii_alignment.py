"""``AlignmentPanel``: choose / import the trial's Wii recording and align it with the .trc.

Owner: GUI-VIEW. Content: recording combo (``WiiPaths.list_recordings``) + "Import Wii
recording…" (``trial.import_recording``), status (``alignment.alignment_status``), method:
recorded (read-only when the take was captured with the videos; "Check timing with a sync
event" = ``alignment.check_recorded_alignment`` measures the camera latency) / "Detect sync event"
(``alignment.auto_align`` in a worker thread) / manual offset spin box (s, 3 decimals), a
small plot of force vs marker height with the matched events, "Save alignment". Emits
``state.notify_balance_changed("alignment")`` after saving.

The plot (``self.plot``, a ``BalancePlotCanvas``) is NOT in this widget's layout: the host
places it (``BalancePanel`` puts it in its "Alignment check" tab). It previews the offset in the
spin box live (Wii time = .trc time + offset), so a manual offset can be set by eye.
The host sets the .trc used for detection and the plot with ``set_trc(path)``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PySide6.QtCore import QEvent, QObject, QThread, QTimer, QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMenu,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .balance_plots import BalancePlotCanvas

log = logging.getLogger(__name__)

GOOD, WARN, BAD, ACCENT, GREY = "#3ad35a", "#e0a94c", "#e05c5c", "#3d8bfd", "#8a8f98"
LEVEL_COLOR = {"ok": GOOD, "warn": WARN, "none": GREY}
DETECT_TIP = ("Find the jumps / stomps in the force and in the marker motion of the .trc and "
              "match them (refined by cross-correlation). Check the result in 'Alignment "
              "check', then save it.")


def _mtime(p) -> int | None:
    try:
        return Path(p).stat().st_mtime_ns
    except (OSError, TypeError):
        return None


def up_vector(project) -> np.ndarray:
    """World up for marker heights: ``alignment.world_up`` (the registered board's normal when
    it is current, else ``world_up_sign`` x Z), the same rule as ``alignment.auto_align``."""
    from poseassess.core.balance.alignment import world_up

    return world_up(project)[0]


class _AlignWorker(QObject):
    # (AlignmentResult | check_recorded_alignment dict (mode "check") | None, error text)
    done = Signal(object, str)

    def __init__(self, project, trc_path, rec_id, mode: str = "align"):
        super().__init__()
        self.project = project
        self.trc_path = trc_path
        self.rec_id = rec_id
        self.mode = mode

    def run(self):
        try:
            from poseassess.core.balance.alignment import (
                auto_align,
                check_recorded_alignment,
            )
            if self.mode == "check":
                res = check_recorded_alignment(self.project, self.trc_path)
            else:
                res = auto_align(self.project, self.trc_path, self.rec_id)
            self.done.emit(res, "")
        except NotImplementedError:
            self.done.emit(None, "automatic alignment is not available yet")
        except Exception as e:  # noqa: BLE001
            self.done.emit(None, f"{type(e).__name__}: {e}")


class AlignmentPanel(QWidget):
    alignment_saved = Signal()
    plot_requested = Signal()   # a detection finished: the host should show ``self.plot``

    def __init__(self, state, parent=None):
        super().__init__(parent)
        self.state = state
        self._trc_path: str | None = None
        self._trial = None          # TrialInfo read by refresh()
        self._root = None           # project root of that TrialInfo
        self._pending = None        # Alignment of the last successful detection (unsaved)
        self._result = None         # AlignmentResult of the last detection (plot events)
        self._result_key = None     # (project root, recording, trc) of that result
        self._override = False      # "Re-align manually…" on a recorded take
        self._signals = None
        self._signals_key = None
        self._thread = None
        self._worker = None
        self._job_key = None
        self._job_mode = "align"
        self._plot_dirty = True

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)

        row = QHBoxLayout()
        row.addWidget(QLabel("Recording:"))
        self.rec_combo = QComboBox()
        self.rec_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.rec_combo.setMinimumContentsLength(10)
        self.rec_combo.setToolTip("The Wii recording (wii/recordings/) that belongs to this "
                                  "trial.")
        self.rec_combo.activated.connect(self._on_recording_chosen)
        row.addWidget(self.rec_combo, 1)
        lay.addLayout(row)

        row = QHBoxLayout()
        self.import_btn = QPushButton("Import Wii recording…")
        self.import_btn.setToolTip(
            "Copy a Wii recording made elsewhere (a recording folder from run_wii.bat / "
            "python -m poseassess.wii, or a wii.csv file) into this project and use it for "
            "this trial.")
        menu = QMenu(self.import_btn)
        menu.addAction("Recording folder…", lambda: self._import("folder"))
        menu.addAction("wii.csv or original Wii program file…", lambda: self._import("file"))
        self.import_btn.setMenu(menu)
        self.open_btn = QPushButton("Open folder")
        self.open_btn.setToolTip("Open the recording folder in the file manager.")
        self.open_btn.clicked.connect(self._open_folder)
        row.addWidget(self.import_btn, 1)
        row.addWidget(self.open_btn)
        lay.addLayout(row)

        self.status_lbl = QLabel("")
        self.status_lbl.setWordWrap(True)
        lay.addWidget(self.status_lbl)

        # recorded together with the videos: nothing to do
        self.recorded_box = QWidget()
        rl = QVBoxLayout(self.recorded_box)
        rl.setContentsMargins(0, 0, 0, 0)
        lbl = QLabel("<small>The Wii data was recorded together with these videos: they are "
                     "aligned by the frame timestamps (frames.csv). The camera latency is "
                     "included only when it was set before the export (3b. Capture); if "
                     "timing matters, check it once with a jump.</small>")
        lbl.setWordWrap(True)
        rl.addWidget(lbl)
        self.check_btn = QPushButton("Check timing with a sync event")
        self.check_btn.setToolTip(
            "Find the jumps / stomps of this take in the force and in the markers and compare "
            "them with the frame timestamps: measures the camera latency (the delay between "
            "exposure and the frame timestamp) to enter on 3b. Capture.")
        self.check_btn.clicked.connect(self._check_timing)
        rl.addWidget(self.check_btn)
        self.check_lbl = QLabel("")
        self.check_lbl.setWordWrap(True)
        self.check_lbl.setVisible(False)
        rl.addWidget(self.check_lbl)
        self.realign_btn = QPushButton("Re-align manually…")
        self.realign_btn.setToolTip("Override the recorded timestamps with a detected or manual "
                                    "offset (only needed when videos/ was replaced or trimmed).")
        self.realign_btn.clicked.connect(self._realign)
        rl.addWidget(self.realign_btn)
        lay.addWidget(self.recorded_box)

        # externally recorded videos: detect or set the offset
        self.align_box = QWidget()
        al = QVBoxLayout(self.align_box)
        al.setContentsMargins(0, 0, 0, 0)
        self.detect_btn = QPushButton("▶ Detect sync event")
        self.detect_btn.clicked.connect(self._detect)
        al.addWidget(self.detect_btn)
        self.result_lbl = QLabel("")
        self.result_lbl.setWordWrap(True)
        self.result_lbl.setVisible(False)
        al.addWidget(self.result_lbl)
        row = QHBoxLayout()
        row.addWidget(QLabel("Offset:"))
        self.offset_spin = QDoubleSpinBox()
        self.offset_spin.setRange(-600.0, 600.0)
        self.offset_spin.setDecimals(3)
        self.offset_spin.setSingleStep(0.001)
        self.offset_spin.setSuffix(" s")
        self.offset_spin.setKeyboardTracking(False)
        self.offset_spin.setToolTip("Wii time = .trc time + offset. Positive when the Wii "
                                    "recording started before the video.")
        self.offset_spin.valueChanged.connect(self._schedule_plot)
        row.addWidget(self.offset_spin, 1)
        al.addLayout(row)
        self.save_btn = QPushButton("✓ Save alignment")
        self.save_btn.setToolTip("Store this offset for the trial (wii/trial.json). The 3D View "
                                 "and the balance results use it.")
        self.save_btn.clicked.connect(self._save)
        al.addWidget(self.save_btn)
        hint = QLabel("<small>For videos recorded elsewhere, ask the subject for <b>2 small "
                      "jumps or 2 stomps</b> on the board at the start of the trial. The "
                      "offset can also be set by eye in 'Alignment check'.</small>")
        hint.setWordWrap(True)
        al.addWidget(hint)
        lay.addWidget(self.align_box)

        self.plot = BalancePlotCanvas(6, 3.5)
        self.plot.installEventFilter(self)
        self._plot_timer = QTimer(self)
        self._plot_timer.setSingleShot(True)
        self._plot_timer.setInterval(150)
        self._plot_timer.timeout.connect(self._update_plot)
        self._update_controls()

    # ---- host API --------------------------------------------------------- #
    def set_trc(self, path) -> None:
        """The .trc used for "Detect sync event" and the marker heights of the plot."""
        path = str(path) if path else None
        if path != self._trc_path:
            self._trc_path = path
            self._pending = None
            self._signals = None
            self.plot.reset_view()
            self._update_controls()
            self._schedule_plot()

    @property
    def busy(self) -> bool:
        return self._thread is not None

    def refresh(self) -> None:
        """Reload recordings and trial.json."""
        proj = self.state.project
        self.rec_combo.clear()
        if proj is None:
            self._trial = self._root = None
            self._pending = self._result = None
            self._override = False
            self._set_status("none", "No project open")
            self._set_result(None, "")
            self._update_controls()
            self.plot.clear()
            return
        from poseassess.core.balance.paths import WiiPaths
        from poseassess.core.balance.trial import load_trial

        trial = load_trial(proj)
        root = str(proj.root)
        if self._trial is None or trial.recording != self._trial.recording or \
                root != self._root:
            self._pending = None
            self._override = False
            self.plot.reset_view()
            self._set_result(None, "")
            self._set_check(None, "")
        self._trial = trial
        self._root = root
        paths = WiiPaths(proj)
        recs = paths.list_recordings()
        self.rec_combo.addItem("(none)", None)
        for r in recs:
            self.rec_combo.addItem(self._describe(paths.recording_dir(r), r), r)
        if trial.recording and trial.recording not in recs:
            self.rec_combo.addItem(f"{trial.recording} (missing)", trial.recording)
        i = self.rec_combo.findData(trial.recording) if trial.recording else 0
        self.rec_combo.setCurrentIndex(max(i, 0))
        try:
            from poseassess.core.balance.alignment import alignment_status
            level, text = alignment_status(proj)
        except NotImplementedError:
            level, text = "warn", "Alignment status not available yet"
        except Exception as e:  # noqa: BLE001
            log.exception("alignment_status failed")
            level, text = "warn", f"Cannot read the alignment: {e}"
        self._set_status(level, text)
        al = trial.alignment
        if al.method != "none" and self._pending is None:
            self.offset_spin.blockSignals(True)
            self.offset_spin.setValue(float(al.offset_s))
            self.offset_spin.blockSignals(False)
        self._update_controls()
        self._schedule_plot()

    def shutdown(self) -> None:
        """Wait for a running detection (called when the app closes)."""
        th = self._thread
        if th is not None:
            th.quit()
            th.wait(10000)

    # ---- helpers ------------------------------------------------------------ #
    @staticmethod
    def _describe(folder: Path, rec: str) -> str:
        from poseassess.wii import io as wio

        try:
            meta = wio.read_session_json(folder)
        except Exception:  # noqa: BLE001
            meta = {}
        bits = []
        dur = meta.get("duration_s")
        if isinstance(dur, (int, float)) and np.isfinite(dur):
            bits.append(f"{dur:.0f} s")
        if meta.get("has_video"):
            bits.append("video + Wii")
        if meta.get("export"):
            bits.append("exported")
        if meta.get("imported_from"):
            bits.append("imported")
        return f"{rec}  ({', '.join(bits)})" if bits else rec

    def _set_status(self, level: str, text: str) -> None:
        color = LEVEL_COLOR.get(level, GREY)
        self.status_lbl.setText(f"<span style='color:{color}'>●</span> {text}")

    def _set_result(self, color: str | None, text: str) -> None:
        """Result line under "Detect sync event" (a coloured dot + the text; hidden when
        empty)."""
        if not text:
            self.result_lbl.clear()
        elif color:
            self.result_lbl.setText(f"<small><span style='color:{color}'>●</span> "
                                    f"{text}</small>")
        else:
            self.result_lbl.setText(f"<small>{text}</small>")
        self.result_lbl.setVisible(bool(text))

    def _set_check(self, color: str | None, text: str) -> None:
        """Result line of "Check timing with a sync event" (hidden when empty)."""
        self.check_lbl.setText(f"<small><span style='color:{color}'>●</span> {text}</small>"
                               if text and color else f"<small>{text}</small>" if text else "")
        self.check_lbl.setVisible(bool(text))

    def _recorded_mode(self) -> bool:
        t = self._trial
        return bool(t and t.recording and t.alignment.method == "recorded" and not self._override)

    def _update_controls(self) -> None:
        proj = self.state.project
        trial = self._trial
        has_rec = bool(proj is not None and trial is not None and trial.recording)
        busy = self.busy
        recorded = self._recorded_mode()
        self.recorded_box.setVisible(has_rec and recorded)
        self.align_box.setVisible(has_rec and not recorded)
        self.detect_btn.setEnabled(has_rec and bool(self._trc_path) and not busy)
        self.detect_btn.setText("Detecting…" if busy else "▶ Detect sync event")
        self.check_btn.setEnabled(has_rec and recorded and bool(self._trc_path) and not busy)
        self.check_btn.setText("Checking…" if busy else "Check timing with a sync event")
        self.save_btn.setEnabled(has_rec and not busy)
        self.offset_spin.setEnabled(has_rec and not busy)
        self.import_btn.setEnabled(proj is not None and not busy)
        self.rec_combo.setEnabled(proj is not None and not busy)
        self.open_btn.setEnabled(has_rec)
        self.detect_btn.setToolTip(
            DETECT_TIP if self._trc_path else
            "Needs a .trc in pose-3d/: run the pipeline first (4. Run).")

    def _key(self):
        proj = self.state.project
        rec = self._trial.recording if self._trial is not None else None
        return (str(proj.root) if proj else None, rec, self._trc_path)

    def _confirm_reset(self, title: str, question: str) -> bool:
        t = self._trial
        if not (t and t.recording and t.alignment.method != "none"):
            return True
        ans = QMessageBox.question(
            self, title, f"{question}\n\nThe current alignment of {t.recording} "
            f"({t.alignment.method}, {t.alignment.offset_s:+.3f} s) will be reset.")
        return ans == QMessageBox.StandardButton.Yes

    # ---- recording choice / import ----------------------------------------- #
    def _on_recording_chosen(self, index: int) -> None:
        proj = self.state.project
        trial = self._trial
        if proj is None or trial is None:
            return
        rec = self.rec_combo.itemData(index)
        if rec == trial.recording:
            return
        if not self._confirm_reset("Wii recording",
                                   f"Use {rec or 'no Wii recording'} for this trial?"):
            i = self.rec_combo.findData(trial.recording) if trial.recording else 0
            self.rec_combo.setCurrentIndex(max(i, 0))
            return
        from poseassess.core.balance.paths import WiiPaths
        from poseassess.core.balance.trial import (
            save_trial,
            set_recording,
            videos_fingerprint,
        )
        from poseassess.wii import io as wio

        try:
            if rec is None:
                set_recording(proj, None, "external")
            elif (WiiPaths(proj).recording_dir(rec) / wio.FRAMES_CSV).is_file() and \
                    QMessageBox.question(
                        self, "Wii recording",
                        f"{rec} was exported to videos/ by 3b. Capture.\n\nUse its frame "
                        "timestamps as the alignment? Answer Yes only if videos/ still holds "
                        "the videos exported from this take.") == QMessageBox.StandardButton.Yes:
                from poseassess.core.balance.alignment import recorded_alignment
                info = set_recording(proj, rec, "capture", recorded_alignment(proj, rec))
                info.videos_fingerprint = videos_fingerprint(proj)
                save_trial(proj, info)
            else:
                set_recording(proj, rec, "external")
        except NotImplementedError:
            QMessageBox.information(self, "Wii recording", "Not available yet.")
            return
        except (OSError, ValueError) as e:
            QMessageBox.warning(self, "Wii recording", str(e))
            self.refresh()
            return
        self._pending = self._result = None
        self._override = False
        self.refresh()
        self.state.notify_balance_changed("trial")

    def _import(self, kind: str) -> None:
        proj = self.state.project
        if proj is None:
            return
        start = str(proj.root.parent)
        if kind == "folder":
            src = QFileDialog.getExistingDirectory(self, "Import Wii recording (folder)", start)
        else:
            src, _ = QFileDialog.getOpenFileName(
                self, "Import Wii recording (wii.csv or a file of the original Wii program)",
                start, "Wii data (*.csv *.txt *.dat);;All files (*)")
        if not src:
            return
        self.import_path(src)

    def import_path(self, src) -> str | None:
        """Import ``src`` (folder or wii.csv) and make it the trial's recording."""
        proj = self.state.project
        if proj is None:
            return None
        if not self._confirm_reset("Import Wii recording",
                                   f"Import {Path(src).name} and use it for this trial?"):
            return None
        try:
            from poseassess.core.balance.trial import import_recording
            rec_id = import_recording(proj, src)
        except NotImplementedError:
            QMessageBox.information(self, "Import Wii recording", "Not available yet.")
            return None
        except (OSError, ValueError) as e:
            QMessageBox.warning(self, "Import Wii recording", str(e))
            return None
        self._pending = self._result = None
        self._override = False
        self.refresh()
        self._set_result(GOOD, f"Imported as {rec_id}. Now detect the sync event or set the "
                               "offset.")
        self.state.notify_balance_changed("trial")
        return rec_id

    def _open_folder(self) -> None:
        proj = self.state.project
        if proj is None or not self._trial or not self._trial.recording:
            return
        from poseassess.core.balance.paths import WiiPaths
        QDesktopServices.openUrl(QUrl.fromLocalFile(
            str(WiiPaths(proj).recording_dir(self._trial.recording))))

    # ---- alignment -------------------------------------------------------- #
    def _realign(self) -> None:
        ans = QMessageBox.question(
            self, "Re-align",
            "This Wii recording was made together with the videos: its frame timestamps align "
            "them (without the camera latency unless it was set before the export).\n\n"
            "Override it with a detected or manual offset? (Needed when the videos in videos/ "
            "were replaced or trimmed, or when \"Check timing\" found a delay you do not want "
            "to fix by exporting again.)")
        if ans != QMessageBox.StandardButton.Yes:
            return
        self._override = True
        self._update_controls()
        self._schedule_plot()

    def _check_timing(self) -> None:
        """Recorded take: compare a detected sync event with the frame timestamps."""
        self._start_job("check")

    def _detect(self) -> None:
        self._start_job("align")

    def _start_job(self, mode: str) -> None:
        """Run ``auto_align`` ("align") or ``check_recorded_alignment`` ("check") in a worker
        thread; ``_on_detect_done`` gets the result."""
        proj = self.state.project
        trial = self._trial
        if proj is None or trial is None or not trial.recording or self.busy:
            return
        if not self._trc_path:
            QMessageBox.information(self, "Detect sync event",
                                    "No .trc file: run the pipeline first (4. Run).")
            return
        self._job_key = self._key()
        self._job_mode = mode
        msg = "Looking for jumps and stomps in the force and the markers…"
        if mode == "check":
            self._set_check(None, msg)
        else:
            self._set_result(None, msg)
        self._worker = _AlignWorker(proj, self._trc_path, trial.recording, mode)
        self._thread = QThread()
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.done.connect(self._on_detect_done)
        self._worker.done.connect(self._thread.quit)
        self._thread.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.finished.connect(self._on_thread_finished)
        self._update_controls()
        self._thread.start()

    def _on_thread_finished(self) -> None:
        self._thread = self._worker = None
        self._update_controls()

    def _on_detect_done(self, res, err: str) -> None:
        key, self._job_key = self._job_key, None
        if key is None or key != self._key():
            return  # the project, recording or .trc changed meanwhile
        if self._job_mode == "check":
            self._on_check_done(res, err, key)
            return
        if err:
            self._set_result(BAD, f"Detection failed: {err}")
            return
        self._result, self._result_key = res, key
        self.plot.reset_view()
        if res.signals:
            self._signals, self._signals_key = dict(res.signals), self._signals_stamp()
        if res.ok:
            self._pending = res.alignment
            self.offset_spin.setValue(float(res.alignment.offset_s))
            self._set_result(GOOD, f"{res.message}. Check 'Alignment check', then click Save "
                                   "alignment.")
        else:
            self._pending = None
            self._set_result(WARN, res.message)
        self._schedule_plot()
        self.plot_requested.emit()

    def _on_check_done(self, chk, err: str, key) -> None:
        if err or chk is None:
            self._set_check(BAD, f"Check failed: {err or 'no result'}")
            return
        res = chk.get("result")
        if res is not None:
            self._result, self._result_key = res, key  # the events go into the plot
            if res.signals:
                self._signals, self._signals_key = dict(res.signals), self._signals_stamp()
            self.plot.reset_view()
        color = GOOD if chk.get("ok") and chk.get("agrees") else WARN
        self._set_check(color, chk.get("message") or "")
        self._schedule_plot()
        self.plot_requested.emit()

    def save_alignment(self) -> bool:
        """Store the offset of the spin box (as the detected method when it is the detected
        offset, else "manual"). Returns True when saved."""
        proj = self.state.project
        trial = self._trial
        if proj is None or trial is None or not trial.recording:
            return False
        off = float(self.offset_spin.value())
        try:
            from poseassess.core.balance.alignment import (
                apply_alignment,
                set_manual_offset,
            )
            if self._pending is not None and abs(float(self._pending.offset_s) - off) < 5e-4:
                apply_alignment(proj, self._pending)
            else:
                set_manual_offset(proj, off, trc_path=self._trc_path)
        except NotImplementedError:
            QMessageBox.information(self, "Save alignment", "Not available yet.")
            return False
        except (OSError, ValueError) as e:
            QMessageBox.warning(self, "Save alignment", str(e))
            return False
        self._pending = None
        self._override = False
        self.refresh()
        self._set_result(GOOD, f"Alignment saved (offset {off:+.3f} s).")
        self.alignment_saved.emit()
        self.state.notify_balance_changed("alignment")
        return True

    def _save(self) -> None:
        self.save_alignment()

    # ---- plot ---------------------------------------------------------------- #
    def eventFilter(self, obj, event):
        if obj is self.plot and event.type() == QEvent.Type.Show and self._plot_dirty:
            QTimer.singleShot(0, self._update_plot)
        return super().eventFilter(obj, event)

    def _schedule_plot(self, *_) -> None:
        self._plot_dirty = True
        self._plot_timer.start()

    def _signals_stamp(self):
        proj = self.state.project
        if proj is None or self._trial is None or not self._trial.recording:
            return None
        from poseassess.core.balance.calib import find_calib_toml
        from poseassess.core.balance.paths import WiiPaths
        from poseassess.wii import io as wio
        paths = WiiPaths(proj)
        # the up axis of the marker heights depends on board.json AND the calibration
        return (str(proj.root), self._trial.recording, self._trc_path,
                _mtime(paths.recording_dir(self._trial.recording) / wio.WII_CSV),
                _mtime(self._trc_path), _mtime(paths.board_json),
                _mtime(find_calib_toml(proj)))

    def _load_signals(self) -> dict:
        """Force and marker-height curves (same keys as ``AlignmentResult.signals``)."""
        stamp = self._signals_stamp()
        if self._signals is not None and self._signals_key == stamp:
            return self._signals
        from poseassess.core.balance.paths import WiiPaths
        from poseassess.wii import io as wio

        proj = self.state.project
        sig: dict[str, np.ndarray] = {}
        rec_dir = WiiPaths(proj).recording_dir(self._trial.recording)
        try:
            wii = wio.read_wii_csv(rec_dir)
            t_rel = np.asarray(wii["t_rel"], float)
            if not np.isfinite(t_rel).any():
                t_rel = wii["t"] - np.nanmin(wii["t"])
            total = np.asarray(wii["total_kg"], float)
            if not np.isfinite(total).any():
                total = np.sum([wii[c] for c in wio.SENSOR_COLS], axis=0)
            sig["force_t_rel"], sig["force_total_kg"] = t_rel, total
        except (OSError, ValueError, KeyError) as e:
            log.info("cannot read the Wii data: %s", e)
        if self._trc_path:
            try:
                from poseassess.core.balance.analysis import trc_heights
                from poseassess.core.balance.trc import read_trc_full
                trc = read_trc_full(self._trc_path)
                hl, hr, com = trc_heights(trc.names, trc.world(), up_vector(proj))
                sig["trc_t"] = np.asarray(trc.times, float)
                sig["trc_foot_up_m"] = np.fmin(hl, hr)
                sig["trc_com_up_m"] = com
            except (OSError, ValueError) as e:
                log.info("cannot read the .trc: %s", e)
        self._signals, self._signals_key = sig, stamp
        return sig

    def _to_trc_time(self, t_rel: np.ndarray) -> np.ndarray:
        t_rel = np.asarray(t_rel, float)
        if self._recorded_mode():
            from poseassess.core.balance.alignment import wii_time_to_trc_time
            return wii_time_to_trc_time(self.state.project, t_rel, self._trial)
        return t_rel - float(self.offset_spin.value())

    def _update_plot(self) -> None:
        if not self.plot.isVisible():
            self._plot_dirty = True
            return
        self._plot_dirty = False
        proj = self.state.project
        trial = self._trial
        if proj is None or trial is None or not trial.recording:
            self.plot.show_message("No Wii recording for this trial.")
            return
        try:
            sig = self._load_signals()
            ft = sig.get("force_t_rel")
            force_t = self._to_trc_time(ft) if ft is not None else None
            fev, tev, focus = [], [], None
            if self._result is not None and self._result_key == self._key():
                for e in self._result.force_events:
                    t = e.get("t_rel_onset") if e.get("type") == "stomp" else None
                    t = e.get("t_rel") if t is None or not np.isfinite(t) else t
                    fev.append({**e, "trc_time": float(self._to_trc_time(np.array([t]))[0])})
                tev = list(self._result.trc_events)
                ts = [e["trc_time"] for e in fev] + [e.get("t") for e in tev]
                ts = [float(t) for t in ts if t is not None and np.isfinite(t)]
                if ts:
                    focus = (min(ts) - 1.5, max(ts) + 1.5)
        except NotImplementedError:
            self.plot.show_message("Alignment check not available yet.")
            return
        except Exception as e:  # noqa: BLE001
            log.exception("alignment plot failed")
            self.plot.show_message(f"Cannot show the alignment: {e}")
            return
        if self._recorded_mode():
            title = "Recorded with the videos (frame timestamps)"
        else:
            title = (f"Offset {self.offset_spin.value():+.3f} s "
                     "(Wii time = .trc time + offset)")
        if sig.get("trc_t") is None:
            title += " · no .trc"
        self.plot.plot_alignment(force_t, sig.get("force_total_kg"), sig.get("trc_t"),
                                 sig.get("trc_foot_up_m"), sig.get("trc_com_up_m"),
                                 force_events=fev, trc_events=tev, title=title,
                                 focus=focus)
