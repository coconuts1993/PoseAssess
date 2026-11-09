"""Live 3D skeleton view (pyqtgraph OpenGL) for ``core.realtime`` results.

Points are in the calibration world frame of ``Calib.toml`` (metres, Z up), which is also
pyqtgraph's up axis, so no reorientation is needed (unlike the .trc viewer, which is Y-up).
"""
from __future__ import annotations

import numpy as np
import pyqtgraph.opengl as gl
from PySide6.QtWidgets import QVBoxLayout, QWidget

from poseassess.core.realtime.skeleton import BONE_IDX, HALPE26, side

_COL = {"R": (0.30, 0.60, 1.00, 1.0), "L": (1.00, 0.55, 0.20, 1.0), "C": (0.75, 0.75, 0.75, 1.0)}
_CAM_COLOR = (0.20, 0.85, 0.80, 1.0)


class LiveSkeletonView(QWidget):
    def __init__(self):
        super().__init__()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.view = gl.GLViewWidget()
        self.view.setBackgroundColor(0.10, 0.11, 0.13, 1.0)
        grid = gl.GLGridItem()
        grid.setSize(4, 4)
        grid.setSpacing(0.25, 0.25)
        self.view.addItem(grid)
        self.view.addItem(gl.GLAxisItem(size=None))
        self._joints = gl.GLScatterPlotItem(size=9.0, pxMode=True)
        self.view.addItem(self._joints)
        self._joint_cols = np.array([_COL[side(n)] for n in HALPE26])
        self._bones = gl.GLLinePlotItem(width=3.0, antialias=True, mode="lines")
        self.view.addItem(self._bones)
        self._cam_items: list = []
        self._centered = False
        self.view.setCameraPosition(distance=4.5, elevation=15, azimuth=-60)
        lay.addWidget(self.view)

    def set_cameras(self, poses) -> None:
        """``core.calib_read.CameraPose`` list: a small frustum per camera."""
        for it in self._cam_items:
            self.view.removeItem(it)
        self._cam_items = []
        for p in poses:
            apex = p.center
            corners = p.image_corner_rays(depth=0.3)
            segs = []
            for c in corners:
                segs += [apex, c]
            for i in range(4):
                segs += [corners[i], corners[(i + 1) % 4]]
            it = gl.GLLinePlotItem(pos=np.array(segs), color=_CAM_COLOR, width=1.6,
                                   antialias=True, mode="lines")
            self.view.addItem(it)
            self._cam_items.append(it)

    def reset_view(self) -> None:
        self._centered = False

    def show_points(self, X: np.ndarray | None) -> None:
        if X is None:
            self._joints.setData(pos=np.zeros((0, 3)))
            self._bones.setData(pos=np.zeros((0, 3)))
            return
        ok = np.isfinite(X).all(axis=1)
        self._joints.setData(pos=X[ok], color=self._joint_cols[ok])
        segs, cols = [], []
        for a, b, s in BONE_IDX:
            if ok[a] and ok[b]:
                segs += [X[a], X[b]]
                cols += [_COL[s], _COL[s]]
        if segs:
            self._bones.setData(pos=np.array(segs), color=np.array(cols))
        else:
            self._bones.setData(pos=np.zeros((0, 3)))
        if not self._centered and ok.sum() >= 10:
            from pyqtgraph import Vector

            c = X[ok].mean(axis=0)
            self.view.opts["center"] = Vector(float(c[0]), float(c[1]), float(c[2]))
            self.view.update()
            self._centered = True
