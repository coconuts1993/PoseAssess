"""Interactive 3D skeleton viewer for a reconstructed .trc trajectory.

A pyqtgraph OpenGL viewport that animates the triangulated skeleton: joints as
points, bones as lines coloured by body side (right=blue, left=orange, centre=
grey) so a clinician can spot left/right asymmetry at a glance. Orbit with the
mouse, scrub the timeline, play/pause. The scene is auto-oriented so the subject
stands upright regardless of the capture frame's axis convention.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QHBoxLayout, QLabel, QPushButton, QSlider,
    QVBoxLayout, QWidget,
)

import pyqtgraph.opengl as gl

from poseassess.core.trc_io import read_trc, TrcData

try:
    from pyqtgraph.opengl import GLTextItem
    _HAS_TEXT = True
except Exception:  # pragma: no cover
    _HAS_TEXT = False

_CAM_COLOR = (0.20, 0.85, 0.80, 1.0)  # teal, distinct from the skeleton


def _zup2yup(pts: np.ndarray) -> np.ndarray:
    """Pose2Sim writes the .trc in Y-up but Calib.toml is Z-up. Match the trc's
    reorientation (new [X,Y,Z] = old [Y,Z,X]) so cameras land in the skeleton frame."""
    p = np.atleast_2d(pts)
    return p[:, [1, 2, 0]]

# Skeleton connectivity (marker pairs) with side, for the Pose2Sim base markers.
_R = (0.30, 0.60, 1.00, 1.0)   # right = blue
_L = (1.00, 0.55, 0.20, 1.0)   # left  = orange
_C = (0.75, 0.75, 0.75, 1.0)   # centre = grey
BONES = [
    ("Hip", "Neck", _C), ("Neck", "Head", _C), ("Head", "Nose", _C),
    ("Hip", "RHip", _R), ("RHip", "RKnee", _R), ("RKnee", "RAnkle", _R),
    ("RAnkle", "RHeel", _R), ("RAnkle", "RBigToe", _R), ("RBigToe", "RSmallToe", _R),
    ("Neck", "RShoulder", _R), ("RShoulder", "RElbow", _R), ("RElbow", "RWrist", _R),
    ("Hip", "LHip", _L), ("LHip", "LKnee", _L), ("LKnee", "LAnkle", _L),
    ("LAnkle", "LHeel", _L), ("LAnkle", "LBigToe", _L), ("LBigToe", "LSmallToe", _L),
    ("Neck", "LShoulder", _L), ("LShoulder", "LElbow", _L), ("LElbow", "LWrist", _L),
]


def _orient_matrix(trc: TrcData) -> np.ndarray:
    """3x3 transform mapping the capture frame so the subject's up axis -> +Z."""
    def mean_of(*names):
        got = [trc.markers[n] for n in names if n in trc.markers]
        return np.nanmean(np.stack(got), axis=(0, 1)) if got else None
    top = mean_of("Head", "Neck")
    bot = mean_of("RAnkle", "LAnkle", "RHeel", "LHeel")
    if top is None or bot is None:
        return np.eye(3)
    v = top - bot
    up = int(np.argmax(np.abs(v)))
    s = float(np.sign(v[up]) or 1.0)
    others = [i for i in range(3) if i != up]
    M = np.zeros((3, 3))
    M[0, others[0]] = 1.0
    M[1, others[1]] = 1.0
    M[2, up] = s
    # Must be a PROPER rotation (det=+1). A reflection (det=-1) would mirror
    # left<->right in the view, which is clinically wrong and breaks the L/R
    # colour coding. Flip a horizontal axis to restore det=+1 without changing up.
    if np.linalg.det(M) < 0:
        M[0] *= -1.0
    return M


class Skeleton3DViewer(QWidget):
    def __init__(self):
        super().__init__()
        self._trc: TrcData | None = None
        self._M = np.eye(3)
        self._center = np.zeros(3)
        self._frame = 0

        root = QVBoxLayout(self)

        self.view = gl.GLViewWidget()
        self.view.setBackgroundColor(0.10, 0.11, 0.13, 1.0)
        self.grid = gl.GLGridItem()
        self.grid.setSize(3, 3)
        self.grid.setSpacing(0.25, 0.25)
        self.view.addItem(self.grid)
        self._joints = gl.GLScatterPlotItem(size=9.0, pxMode=True)
        self.view.addItem(self._joints)
        self._bone_items: list[gl.GLLinePlotItem] = []
        self._cam_items: list = []
        self._cam_poses = []
        root.addWidget(self.view, 1)

        # ---- transport controls ----
        ctl = QHBoxLayout()
        self.play_btn = QPushButton("▶")
        self.play_btn.setFixedWidth(36)
        self.play_btn.clicked.connect(self._toggle_play)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.valueChanged.connect(self._on_slider)
        self.frame_lbl = QLabel("0 / 0")
        self.speed = QComboBox()
        self.speed.addItems(["0.25×", "0.5×", "1×", "2×"])
        self.speed.setCurrentIndex(2)
        self.cam_check = QCheckBox("Cameras")
        self.cam_check.setChecked(True)
        self.cam_check.toggled.connect(self._toggle_cameras)
        ctl.addWidget(self.play_btn)
        ctl.addWidget(self.slider, 1)
        ctl.addWidget(self.frame_lbl)
        ctl.addWidget(QLabel("Speed:"))
        ctl.addWidget(self.speed)
        ctl.addWidget(self.cam_check)
        root.addLayout(ctl)

        legend = QLabel("<span style='color:#4c9bff'>■ Right</span>&nbsp;&nbsp;"
                        "<span style='color:#ff8c33'>■ Left</span>&nbsp;&nbsp;"
                        "<span style='color:#bfbfbf'>■ Centre</span>&nbsp;&nbsp;"
                        "<small>drag to orbit · scroll to zoom</small>")
        root.addWidget(legend)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._advance)

    # ---- loading ---------------------------------------------------------- #
    def load_trc(self, path: str | Path) -> bool:
        try:
            trc = read_trc(path)
        except Exception:
            return False
        if trc.n_frames == 0 or not trc.markers:
            return False
        self._trc = trc
        self._M = _orient_matrix(trc)
        pts = trc.all_points() @ self._M.T
        finite = pts[np.isfinite(pts).all(axis=1)]
        self._center = finite.mean(axis=0) if len(finite) else np.zeros(3)
        # rebuild bone line items for the bones whose markers exist
        for it in self._bone_items:
            self.view.removeItem(it)
        self._bone_items = []
        self._active_bones = [(a, b, c) for a, b, c in BONES
                              if a in trc.markers and b in trc.markers]
        for _a, _b, color in self._active_bones:
            it = gl.GLLinePlotItem(width=3.0, antialias=True, color=color)
            self.view.addItem(it)
            self._bone_items.append(it)
        # camera distance from subject height
        span = float(np.nanmax(finite, axis=0)[2] - np.nanmin(finite, axis=0)[2]) if len(finite) else 1.7
        self.view.setCameraPosition(distance=max(2.5, span * 2.2), elevation=12, azimuth=-70)
        self.slider.setRange(0, trc.n_frames - 1)
        self._show_frame(0)
        return True

    def clear(self):
        self._trc = None
        self._joints.setData(pos=np.zeros((0, 3)))
        for it in self._bone_items:
            it.setData(pos=np.zeros((2, 3)))
        self.frame_lbl.setText("0 / 0")
        self._clear_cameras()

    # ---- cameras ---------------------------------------------------------- #
    def _tf(self, world: np.ndarray) -> np.ndarray:
        """World point(s) -> display frame (same transform as the skeleton)."""
        return world @ self._M.T - self._center

    def _clear_cameras(self):
        for it in self._cam_items:
            self.view.removeItem(it)
        self._cam_items = []

    def set_cameras(self, poses):
        """Draw a small frustum + label per camera, in the skeleton's frame."""
        self._clear_cameras()
        self._cam_poses = list(poses)
        if not self._cam_poses:
            return
        for p in self._cam_poses:
            # Calib.toml is Z-up; match the trc's zup->yup reorientation first,
            # then apply the skeleton display transform.
            apex = self._tf(_zup2yup(p.center.reshape(1, 3)))[0]
            corners = self._tf(_zup2yup(p.image_corner_rays(depth=0.30)))  # (4,3)
            # 8 segments (4 rays + rectangle) as point pairs for 'lines' mode
            segs = []
            for c in corners:
                segs += [apex, c]
            for i in range(4):
                segs += [corners[i], corners[(i + 1) % 4]]
            item = gl.GLLinePlotItem(pos=np.array(segs), color=_CAM_COLOR,
                                     width=1.6, antialias=True, mode="lines")
            item.setVisible(self.cam_check.isChecked())
            self.view.addItem(item)
            self._cam_items.append(item)
            if _HAS_TEXT:
                lbl = GLTextItem(pos=apex, text=str(p.name), color=(120, 230, 220, 255))
                lbl.setVisible(self.cam_check.isChecked())
                self.view.addItem(lbl)
                self._cam_items.append(lbl)
        # zoom out so the whole camera ring is visible
        cams_disp = self._tf(_zup2yup(np.array([p.center for p in self._cam_poses])))
        reach = float(np.linalg.norm(cams_disp, axis=1).max())
        self.view.setCameraPosition(distance=max(2.5, reach * 1.9))

    def _toggle_cameras(self, on: bool):
        for it in self._cam_items:
            it.setVisible(on)

    # ---- rendering -------------------------------------------------------- #
    def _pt(self, name: str) -> np.ndarray:
        p = self._trc.markers[name][self._frame]
        return (self._M @ p) - self._center

    def _show_frame(self, i: int):
        if self._trc is None:
            return
        self._frame = max(0, min(i, self._trc.n_frames - 1))
        names = [n for n in self._trc.markers]
        pos = np.array([self._pt(n) for n in names])
        # colour joints by side
        cols = []
        for n in names:
            if n.startswith("R"):
                cols.append(_R)
            elif n.startswith("L"):
                cols.append(_L)
            else:
                cols.append(_C)
        self._joints.setData(pos=pos, color=np.array(cols))
        for it, (a, b, _c) in zip(self._bone_items, self._active_bones):
            seg = np.array([self._pt(a), self._pt(b)])
            it.setData(pos=seg)
        self.frame_lbl.setText(f"{self._frame} / {self._trc.n_frames - 1}")
        self.slider.blockSignals(True)
        self.slider.setValue(self._frame)
        self.slider.blockSignals(False)

    # ---- transport -------------------------------------------------------- #
    def _on_slider(self, v: int):
        self._show_frame(v)

    def _toggle_play(self):
        if self._trc is None:
            return
        if self._timer.isActive():
            self._timer.stop()
            self.play_btn.setText("▶")
        else:
            self._timer.start(int(1000 / 60))
            self.play_btn.setText("⏸")

    def _speed_factor(self) -> float:
        return {0: 0.25, 1: 0.5, 2: 1.0, 3: 2.0}[self.speed.currentIndex()]

    def _advance(self):
        if self._trc is None:
            return
        step = max(1, int(round(self._speed_factor())))
        nxt = self._frame + step
        if nxt >= self._trc.n_frames:
            nxt = 0
        self._show_frame(nxt)
