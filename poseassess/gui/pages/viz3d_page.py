"""3D view page: watch the reconstructed skeleton move in 3D.

Loads a triangulated .trc from the project's pose-3d/ folder and animates it, so
the clinician can visually verify tracking quality and see the movement (and
left/right asymmetry) before trusting the joint-angle numbers.

With Wii Balance Board data in the project (wii/), the viewer also shows the board, the centre
of pressure, the ground reaction force and the whole-body centre of mass, synchronized with the
.trc playback (``BalanceSkeleton3DViewer``), and the world / board coordinate frames. The
"Wii data" menu loads a Wii file for the trial (a wii.csv or a file of the original Wii program)
and aligns it (jumps / stomps, or the "Wii offset" box: Wii time = .trc time + offset, applied
live so the COP can be matched to the feet by eye).
"""
from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMenu,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from ..widgets.balance_3d import BalanceSkeleton3DViewer
from .base_page import BasePage

log = logging.getLogger(__name__)
_WII_COLORS = {"ok": "#3ad35a", "warn": "#e0a94c", "none": "#8a8f98"}


def _mtime_ns(path) -> int | None:
    try:
        return Path(path).stat().st_mtime_ns
    except (OSError, TypeError):
        return None


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

        # Wii data of the trial, straight from the 3D View
        self.wii_btn = QPushButton("Wii data")
        self.wii_btn.setToolTip("Load a Wii Balance Board recording for this trial and align "
                                "it with the .trc, to see the COP and the force in 3D.")
        self.wii_menu = QMenu(self.wii_btn)
        self.act_wii_load = self.wii_menu.addAction(
            "Load Wii file for this trial…", self._wii_load)
        self.found_menu = QMenu("Wii files found in this project", self.wii_menu)
        self.wii_menu.addMenu(self.found_menu)
        self.wii_menu.aboutToShow.connect(self._fill_found_menu)
        self.wii_menu.addSeparator()
        self.act_wii_align = self.wii_menu.addAction(
            "Auto-align (jumps / stomps on the board)", self._wii_auto_align)
        self.act_wii_clock = self.wii_menu.addAction(
            "Align by the computer clock (file times, estimate)", self._wii_clock_align)
        self.act_wii_zero = self.wii_menu.addAction(
            "Set the offset to 0 (Wii start = video start)", lambda: self._wii_set_offset(0.0))
        self.wii_btn.setMenu(self.wii_menu)
        self.wii_offset = QDoubleSpinBox()
        self.wii_offset.setRange(-36000.0, 36000.0)
        self.wii_offset.setDecimals(3)
        self.wii_offset.setSingleStep(0.05)
        self.wii_offset.setSuffix(" s")
        self.wii_offset.setKeyboardTracking(False)
        self.wii_offset.setToolTip(
            "Wii time = .trc time + offset. Change it while watching the COP (red) under the "
            "feet; it is saved as a manual alignment. Disabled when the Wii data was recorded "
            "together with the videos (frame timestamps).")
        self._offset_timer = QTimer(self)
        self._offset_timer.setSingleShot(True)
        self._offset_timer.setInterval(350)
        self._offset_timer.timeout.connect(self._wii_offset_committed)
        self.wii_offset.valueChanged.connect(lambda _v: self._offset_timer.start())
        self.wii_offset_lbl = QLabel("Wii offset:")
        bar.addWidget(self.wii_btn)
        bar.addWidget(self.wii_offset_lbl)
        bar.addWidget(self.wii_offset)
        root.addLayout(bar)

        self.viewer = BalanceSkeleton3DViewer()
        root.addWidget(self.viewer, 1)

        self.status = QLabel("")
        self.status.setWordWrap(True)
        root.addWidget(self.status)

        # Wii Balance Board overlays (optional): one compact status line, hidden when the
        # project has no Wii data. The overlays are computed when the page is visible.
        self.wii_status = QLabel("")
        self.wii_status.setWordWrap(True)
        self.wii_status.setVisible(False)
        root.addWidget(self.wii_status)
        self._balance_path: str | None = None
        self._balance_mtime: int | None = None  # of the .trc the viewer holds in memory
        self._balance_dirty = False
        state.balance_changed.connect(self._on_balance_changed)
        self._update_wii_controls()

    def on_project_changed(self, project):
        self._refresh_list()
        self._update_wii_controls()

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
            self._set_balance_source(None)

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
            self._set_balance_source(path)
        else:
            self.status.setText(f"Could not load {Path(path).name}")

    def _browse(self):
        start = str(self.state.project.pose3d_dir) if self.state.project else ""
        f, _ = QFileDialog.getOpenFileName(self, "Load .trc", start, "TRC (*.trc)")
        if f and self.viewer.load_trc(f):
            self.status.setText(Path(f).name)
            self._set_balance_source(f)

    # ---- Wii Balance Board overlays ---------------------------------------- #
    def _set_balance_source(self, path) -> None:
        """A new .trc is shown: drop the old overlays, compute the new ones (now if the page
        is visible, else when it is shown)."""
        self._balance_path = str(path) if path else None
        self._balance_mtime = _mtime_ns(path) if path else None
        self.viewer.clear_balance()
        self._show_wii_status(None, "")
        self._request_balance()

    def _on_balance_changed(self, _what: str = "") -> None:
        if self._balance_path:
            self._request_balance()

    def _request_balance(self) -> None:
        self._balance_dirty = True
        if self.isVisible():
            self._reload_balance()

    def showEvent(self, e):
        super().showEvent(e)
        if self._balance_dirty:
            self._reload_balance()

    def _reload_balance(self) -> None:
        """Fuse the shown .trc with the trial's Wii recording and draw the overlays."""
        self._balance_dirty = False
        proj = self.state.project
        path = self._balance_path
        if proj is None or not path or self.viewer._trc is None:
            self.viewer.clear_balance()
            self._show_wii_status(None, "")
            return
        if _mtime_ns(path) != self._balance_mtime:
            # the pipeline wrote a new .trc, but the viewer still plays the old one from
            # memory: fusing the new file would not match the skeleton on screen
            from poseassess.core.balance.paths import WiiPaths

            self.viewer.clear_balance()
            if WiiPaths(proj).wii_dir.is_dir():
                self._show_wii_status("warn", f"Wii: {Path(path).name} changed on disk (pipeline "
                                              "run again?): press Reload to show the new "
                                              "trajectory with its Wii overlays.")
            else:
                self._show_wii_status(None, "")
            return
        try:
            level, text, fused, board = self._compute_balance(proj, Path(path))
        except Exception as e:  # noqa: BLE001  (the Wii must never break the 3D View)
            log.exception("Wii overlays failed")
            level, text, fused, board = "warn", f"Wii: overlays unavailable ({e})", None, None
        if fused is None and board is None:
            self.viewer.clear_balance()
        else:
            self.viewer.set_balance(fused, board)
        self._show_wii_status(level, text)
        self._update_wii_controls()

    def _compute_balance(self, proj, path: Path):
        """``(level, status text, FusedTrial | None, BoardRegistration | None)``; level None
        hides the status line (no Wii data in this project)."""
        from poseassess.core.balance.paths import WiiPaths

        paths = WiiPaths(proj)
        if not paths.wii_dir.is_dir():
            return self._found_status(proj)
        try:
            inside = path.resolve().parent == proj.pose3d_dir.resolve()
        except OSError:
            inside = False
        if not inside:
            return ("none", "Wii: overlays are shown only for the .trc files of this "
                    "project's pose-3d/ folder", None, None)
        from poseassess.core.balance.board import load_board
        from poseassess.core.balance.trial import load_trial

        trial = load_trial(proj)
        reg = load_board(proj)
        if reg is None and not trial.recording:
            found = self._found_status(proj)
            return found if found[0] else ("none", "Wii: no data for this trial", None, None)
        fused, err = None, ""
        if trial.recording:
            try:
                from poseassess.core.balance.fusion import fuse_trial
                fused = fuse_trial(proj, trc_path=path)
            except NotImplementedError:
                err = "the Wii overlays are not available yet"
            except (ValueError, OSError) as e:
                err = str(e)
        if fused is not None:
            board = fused.board
        else:
            board = reg if reg is not None and reg.pose.world_camera is None else None

        parts, warns = [], []
        level = "ok"
        if trial.recording:
            parts.append(f"Wii: {trial.recording}")
            try:
                from poseassess.core.balance.alignment import alignment_status
                level, al_text = alignment_status(proj)
                parts.append(al_text)
            except NotImplementedError:
                pass
        else:
            found = self._found_files(proj)
            if found:
                level = "warn"
                parts.append(f"Wii: board only. {self._found_text(proj, found)}")
            else:
                parts.append("Wii: board only, no recording for this trial (Wii data > Load "
                             "Wii file for this trial…)")
        if err:
            warns.append(err)
        if fused is not None:
            for w in fused.warnings:
                low = w.lower()
                if level == "warn" and ("not aligned" in low or "videos changed" in low):
                    continue  # already said by the alignment status
                warns.append(w)
        elif reg is not None:
            if reg.stale:
                warns.append(reg.stale)
            if reg.pose.world_camera is not None:
                warns.append("The board was located without camera extrinsics: recompute it "
                             "(2b. Wii Board).")
        if warns and level == "ok":
            level = "warn"
        return level, " · ".join(parts + [f"⚠ {w}" for w in warns]), fused, board

    def _show_wii_status(self, level, text: str) -> None:
        if level is None:
            self.wii_status.setVisible(False)
            self.wii_status.clear()
            return
        color = _WII_COLORS.get(level, _WII_COLORS["none"])
        self.wii_status.setText(f"<small><span style='color:{color}'>●</span> {text}</small>")
        self.wii_status.setVisible(True)

    # ---- Wii data menu / offset ---------------------------------------------- #
    def _trial(self):
        proj = self.state.project
        if proj is None:
            return None
        from poseassess.core.balance.trial import load_trial

        return load_trial(proj)

    def _update_wii_controls(self) -> None:
        proj = self.state.project
        trial = self._trial()
        has_rec = bool(trial and trial.recording)
        has_trc = bool(self._balance_path) and self.viewer._trc is not None
        self.wii_btn.setEnabled(proj is not None)
        self.act_wii_align.setEnabled(has_rec and has_trc)
        self.act_wii_zero.setEnabled(has_rec)
        self.act_wii_clock.setEnabled(has_rec)
        recorded = has_rec and trial.alignment.method == "recorded"
        self.wii_offset.setEnabled(has_rec and not recorded)
        self.wii_offset_lbl.setEnabled(has_rec and not recorded)
        if has_rec and not self._offset_timer.isActive():
            self.wii_offset.blockSignals(True)
            self.wii_offset.setValue(float(trial.alignment.offset_s))
            self.wii_offset.blockSignals(False)

    def _found_files(self, proj) -> list:
        from poseassess.core.balance.discover import find_wii_files

        try:
            return find_wii_files(proj)
        except OSError:
            return []

    @staticmethod
    def _found_text(proj, found) -> str:
        names = []
        for p in found[:3]:
            try:
                names.append(str(Path(p).relative_to(proj.root)))
            except ValueError:
                names.append(Path(p).name)
        more = f" (+{len(found) - 3} more)" if len(found) > 3 else ""
        return (f"Wii file(s) found in the project folder but not loaded: {', '.join(names)}"
                f"{more}. Load one with Wii data > Wii files found in this project.")

    def _found_status(self, proj):
        found = self._found_files(proj)
        if not found:
            return None, "", None, None
        return "warn", "Wii: " + self._found_text(proj, found), None, None

    def _fill_found_menu(self) -> None:
        self.found_menu.clear()
        proj = self.state.project
        found = self._found_files(proj) if proj is not None else []
        for p in found:
            try:
                label = str(Path(p).relative_to(proj.root))
            except ValueError:
                label = str(p)
            self.found_menu.addAction(label, lambda p=p: self.load_wii_file(p))
        if not found:
            self.found_menu.addAction("(none: put the Wii file in the project folder, or use "
                                      "Load Wii file…)").setEnabled(False)

    def _wii_load(self) -> None:
        proj = self.state.project
        if proj is None:
            return
        f, _ = QFileDialog.getOpenFileName(
            self, "Load Wii file for this trial (wii.csv or a file of the original Wii program)",
            str(proj.root.parent), "Wii data (*.csv *.txt *.dat);;All files (*)")
        if f:
            self.load_wii_file(f)

    def load_wii_file(self, path, auto_align: bool = True) -> str | None:
        """Import ``path`` as the trial's Wii recording, then try the automatic alignment
        (else offset 0, to be adjusted in the "Wii offset" box). Returns the recording id."""
        proj = self.state.project
        if proj is None:
            return None
        from poseassess.core.balance.trial import import_recording

        try:
            rec_id = import_recording(proj, path)
        except (OSError, ValueError) as e:
            QMessageBox.warning(self, "Load Wii file", str(e))
            return None
        msg = None
        if auto_align and self._balance_path:
            msg = self._run_auto_align(quiet=True)
        if msg is None and auto_align:
            msg = self._run_clock_align(quiet=True)
            if msg is not None:
                msg = (f"Wii file loaded as {rec_id}. No jump / stomp found: {msg} Fine-tune the "
                       "Wii offset until the COP follows the feet.")
        if msg is None:
            self._wii_set_offset(0.0, notify=False)
            if auto_align:
                msg = (f"Wii file loaded as {rec_id}. No automatic alignment possible (no jump "
                       "/ stomp, no computer time): offset 0 - adjust the Wii offset until the "
                       "COP follows the feet.")
        self.state.notify_balance_changed("trial")  # -> _on_balance_changed redraws
        if msg:
            self.status.setText(msg)
        return rec_id

    def _run_clock_align(self, quiet: bool = False) -> str | None:
        """Offset from the computer times of the Wii recording and the videos (estimate)."""
        proj = self.state.project
        from poseassess.core.balance.alignment import Alignment, apply_alignment
        from poseassess.core.balance.discover import clock_offset
        from poseassess.core.balance.paths import WiiPaths

        try:
            off, details = clock_offset(proj)
        except (OSError, ValueError, KeyError) as e:
            if not quiet:
                QMessageBox.information(self, "Align by the computer clock", str(e))
            return None
        details["estimated_from"] = "computer clock (file times)"
        trc = WiiPaths(proj).relative(self._balance_path) if self._balance_path else None
        apply_alignment(proj, Alignment("manual", float(off), None, trc, details))
        spread = details.get("video_start_spread_s") or 0.0
        return (f"Offset {off:+.3f} s estimated from the computer clock (Wii file and video "
                f"file times" + (f"; the cameras differ by {spread:.1f} s" if spread > 1 else "")
                + ").")

    def _wii_clock_align(self) -> None:
        msg = self._run_clock_align()
        if msg is not None:
            self.status.setText(msg + " Fine-tune the Wii offset if needed.")
            self.state.notify_balance_changed("alignment")

    def _run_auto_align(self, quiet: bool = False) -> str | None:
        """Automatic alignment of the trial's recording with the shown .trc; the result text,
        or None when it failed (message box unless ``quiet``)."""
        proj = self.state.project
        from poseassess.core.balance.alignment import apply_alignment, auto_align

        err = ""
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            res = auto_align(proj, trc_path=self._balance_path)
        except Exception as e:  # noqa: BLE001
            log.exception("auto-align failed")
            res, err = None, str(e) or type(e).__name__
        finally:
            QApplication.restoreOverrideCursor()
        if res is None or not res.ok:
            if not quiet:
                QMessageBox.information(self, "Auto-align",
                                        err if res is None else res.message)
            return None
        apply_alignment(proj, res.alignment)
        self.status.setText(res.message)
        return res.message

    def _wii_auto_align(self) -> None:
        if self._run_auto_align() is not None:
            self.state.notify_balance_changed("alignment")

    def _wii_set_offset(self, offset: float, notify: bool = True) -> None:
        proj = self.state.project
        trial = self._trial()
        if proj is None or not (trial and trial.recording):
            return
        from poseassess.core.balance.alignment import set_manual_offset

        set_manual_offset(proj, float(offset), trc_path=self._balance_path)
        if notify:
            self.state.notify_balance_changed("alignment")

    def _wii_offset_committed(self) -> None:
        self._wii_set_offset(self.wii_offset.value())
