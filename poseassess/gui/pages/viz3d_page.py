"""3D view page: watch the reconstructed skeleton move in 3D.

Loads a triangulated .trc from the project's pose-3d/ folder and animates it, so
the clinician can visually verify tracking quality and see the movement (and
left/right asymmetry) before trusting the joint-angle numbers.

With Wii Balance Board data in the project (wii/), the viewer also shows the board, the centre
of pressure, the ground reaction force and the whole-body centre of mass, synchronized with the
.trc playback (``BalanceSkeleton3DViewer``). Without Wii data the page is unchanged.
"""
from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtWidgets import (
    QComboBox, QFileDialog, QHBoxLayout, QLabel, QPushButton, QVBoxLayout,
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

    def on_project_changed(self, project):
        self._refresh_list()

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

    def _compute_balance(self, proj, path: Path):
        """``(level, status text, FusedTrial | None, BoardRegistration | None)``; level None
        hides the status line (no Wii data in this project)."""
        from poseassess.core.balance.paths import WiiPaths

        paths = WiiPaths(proj)
        if not paths.wii_dir.is_dir():
            return None, "", None, None
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
            return "none", "Wii: no data for this trial", None, None
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
            parts.append("Wii: board only, no recording for this trial (record one on "
                         "3b. Capture or import one on 6. Results > Balance (Wii))")
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
