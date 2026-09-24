"""``BalanceSkeleton3DViewer``: the 3D View skeleton player + Wii overlays (board, COP point and
trail, ground reaction force arrow, whole-body COM with plumb line), synchronized with the .trc
playback.

Owner: GUI-VIEW. Subclass of ``Skeleton3DViewer`` so the original viewer keeps its behaviour:
* ``load_trc``: super(), then re-transform static items (the display transform ``_M``/``_center``
  changes per .trc);
* ``_show_frame(i)``: super(), then update per-frame overlays from ``FusedTrial`` row
  ``fused.index_for_time(self._trc.times[i])`` (never by wall clock or row index);
* ``clear``: super(), then hide overlays.
Transforms (docs/WII_INTEGRATION.md, "Coordinate frames"): world point P ->
``self._tf(zup_to_yup(P))``; world direction d -> ``zup_to_yup(d) @ self._M.T``.
Toggles ("Board", "COP", "Force", "COM") go in the transport row next to "Cameras".
The GL background renders LIGHT (see the 3D View quirk): use dark / saturated overlay colours.

Without balance data (``set_balance`` never called, or ``clear_balance``) no GL item is created
and the toggles stay hidden, so the viewer looks exactly like ``Skeleton3DViewer``.
"""

from __future__ import annotations

import logging

import numpy as np
import pyqtgraph.opengl as gl
from PySide6.QtWidgets import QCheckBox, QHBoxLayout, QLabel

from .skeleton_3d import _HAS_TEXT, Skeleton3DViewer, _zup2yup

log = logging.getLogger(__name__)

BOARD_COLOR = "#333333"
COP_COLOR = "#e0303a"
FORCE_COLOR = "#1a9a3a"
COM_COLOR = "#8a2be2"
TRAIL_S = 2.0            # COP trail length (s of .trc rows)
LIFT_M = 0.002           # COP / force drawn this far above the top surface (no z-fighting)
ARROW_M_PER_KG = 0.005   # = fusion.COP_ARROW_M_PER_KG (local copy: no core import at startup)
HEAD_LEN_M = 0.06        # force arrow head length (at most 40 % of the arrow)
HEAD_RADIUS_M = 0.022
FRAME_ITEMS = ("cop", "cop_trail", "force", "force_head", "com", "plumb", "plumb_foot")
BOARD_ITEMS = ("surface", "outline", "front", "sensors", "front_label")


def _rgba(hex_color: str, alpha: float = 1.0) -> tuple[float, float, float, float]:
    h = hex_color.lstrip("#")
    return (int(h[0:2], 16) / 255, int(h[2:4], 16) / 255, int(h[4:6], 16) / 255, float(alpha))


def _unit(v) -> np.ndarray | None:
    v = np.asarray(v, np.float64).ravel()
    if v.size != 3 or not np.isfinite(v).all():
        return None
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else None


def _cone(apex: np.ndarray, axis: np.ndarray, length: float, radius: float,
          n: int = 14) -> tuple[np.ndarray, np.ndarray]:
    """Vertices / faces of a closed cone with its tip at ``apex``, pointing along ``axis``."""
    a = _unit(axis)
    ref = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(a, ref)
    u /= np.linalg.norm(u)
    v = np.cross(a, u)
    base = apex - a * length
    ang = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    ring = base + radius * (np.cos(ang)[:, None] * u + np.sin(ang)[:, None] * v)
    verts = np.vstack([apex, base, ring])
    faces = []
    for k in range(n):
        i, j = 2 + k, 2 + (k + 1) % n
        faces.append((0, i, j))
        faces.append((1, j, i))
    return verts, np.asarray(faces, np.int32)


class BalanceSkeleton3DViewer(Skeleton3DViewer):
    """Skeleton3DViewer + balance overlays (see module docstring)."""

    def __init__(self):
        super().__init__()
        self._fused = None
        self._board = None
        self._bal_items: dict[str, object] = {}
        self._bal_row = -1
        self._trail_rows = 60

        # toggles next to "Cameras" (hidden until there is balance data)
        row = self._transport_row()
        self.board_check = self._make_toggle(
            "Board", BOARD_COLOR, "Show the Wii Balance Board: outline (thick edge = front, "
            "opposite the power button) and its four sensors.")
        self.cop_check = self._make_toggle(
            "COP", COP_COLOR, "Show the centre of pressure measured by the board, with its "
            f"trail over the last {TRAIL_S:.0f} seconds.")
        self.force_check = self._make_toggle(
            "Force", FORCE_COLOR, "Show the vertical ground reaction force as an arrow at the "
            "centre of pressure (0.35 m for 70 kg). The board measures vertical load only.")
        self.com_check = self._make_toggle(
            "COM", COM_COLOR, "Show the whole-body centre of mass computed from the markers, "
            "with its plumb line down to the board surface.")
        self._toggles = (self.board_check, self.cop_check, self.force_check, self.com_check)
        for chk in self._toggles:
            row.addWidget(chk)

        # per-frame read-out under the legend (hidden until there is fused data)
        self.balance_info = QLabel("")
        self.balance_info.setVisible(False)
        self.balance_info.setToolTip(
            "Wii data of the current frame. ML = medio-lateral (+ = subject's right), AP = "
            "antero-posterior (+ = front), both in the board frame.")
        self.layout().addWidget(self.balance_info)

    # ---- construction helpers ------------------------------------------- #
    def _transport_row(self) -> QHBoxLayout:
        """The transport row of ``Skeleton3DViewer`` (the layout that holds ``cam_check``)."""
        root = self.layout()
        for k in range(root.count()):
            lay = root.itemAt(k).layout()
            if lay is not None and lay.indexOf(self.cam_check) >= 0:
                return lay
        lay = QHBoxLayout()  # fallback, not expected
        lay.addStretch(1)
        root.addLayout(lay)
        return lay

    def _make_toggle(self, text: str, color: str, tip: str) -> QCheckBox:
        chk = QCheckBox(text)
        chk.setChecked(True)
        chk.setVisible(False)
        chk.setToolTip(tip)
        chk.setStyleSheet(f"QCheckBox {{ color: {color}; font-weight: 600; }}")
        chk.toggled.connect(self._on_toggle)
        return chk

    def _ensure_items(self) -> None:
        """Create the GL items once (lazily, so a viewer without Wii data is untouched).

        Like the skeleton (pyqtgraph "additive"), the overlays are drawn without depth test
        (the 3D View grid at hip height would otherwise cut gaps into them), with normal alpha
        blending; the insertion order below is the drawing order."""
        if self._bal_items:
            return
        from OpenGL import GL

        tr = {GL.GL_DEPTH_TEST: False, GL.GL_BLEND: True, GL.GL_CULL_FACE: False,
              "glBlendFuncSeparate": (GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA,
                                      GL.GL_ONE, GL.GL_ONE_MINUS_SRC_ALPHA)}
        it: dict[str, object] = {}
        surf = gl.GLMeshItem(vertexes=np.zeros((3, 3)), faces=np.array([[0, 1, 2]]),
                             color=_rgba(BOARD_COLOR, 0.16), smooth=False, drawEdges=False,
                             glOptions=tr)
        surf.setDepthValue(-1)  # drawn before the skeleton: the feet stay on top of it
        it["surface"] = surf
        it["outline"] = gl.GLLinePlotItem(mode="line_strip", width=2.0, antialias=True,
                                          color=_rgba(BOARD_COLOR), glOptions=tr)
        it["front"] = gl.GLLinePlotItem(mode="lines", width=5.0, antialias=True,
                                        color=_rgba(BOARD_COLOR), glOptions=tr)
        it["sensors"] = gl.GLScatterPlotItem(size=8.0, pxMode=True, color=_rgba(BOARD_COLOR),
                                             glOptions=tr)
        if _HAS_TEXT:
            from pyqtgraph.opengl import GLTextItem
            it["front_label"] = GLTextItem(text="front", color=(51, 51, 51, 255))
        it["cop_trail"] = gl.GLLinePlotItem(mode="lines", width=3.0, antialias=True,
                                            color=_rgba(COP_COLOR), glOptions=tr)
        it["force"] = gl.GLLinePlotItem(mode="lines", width=5.0, antialias=True,
                                        color=_rgba(FORCE_COLOR), glOptions=tr)
        it["force_head"] = gl.GLMeshItem(vertexes=np.zeros((3, 3)), faces=np.array([[0, 1, 2]]),
                                         color=_rgba(FORCE_COLOR), smooth=False,
                                         drawEdges=False, glOptions=tr)
        it["cop"] = gl.GLScatterPlotItem(size=15.0, pxMode=True, color=_rgba(COP_COLOR),
                                         glOptions=tr)
        it["plumb"] = gl.GLLinePlotItem(mode="lines", width=2.0, antialias=True,
                                        color=_rgba(COM_COLOR), glOptions=tr)
        it["plumb_foot"] = gl.GLScatterPlotItem(size=7.0, pxMode=True,
                                                color=_rgba(COM_COLOR), glOptions=tr)
        it["com"] = gl.GLScatterPlotItem(size=16.0, pxMode=True, color=_rgba(COM_COLOR),
                                         glOptions=tr)
        for item in it.values():
            item.setVisible(False)
            self.view.addItem(item)
        self._bal_items = it

    # ---- public API ------------------------------------------------------- #
    def set_balance(self, fused, board) -> None:
        """Show ``fused`` (``core.balance.fusion.FusedTrial`` or None) and ``board``
        (``core.balance.board.BoardRegistration`` or None; board drawn even without fused data).
        Call after ``load_trc`` + ``set_cameras``.

        ``board`` None falls back to ``fused.board``. COP and force need a board (they are drawn
        in the Calib world); the COM does not (its plumb line does)."""
        if board is None and fused is not None:
            board = getattr(fused, "board", None)
        if fused is None and board is None:
            self.clear_balance()
            return
        self._fused = fused
        self._board = board
        self._trail_rows = 60
        if fused is not None:
            dt = np.diff(np.asarray(fused.trc_time, float))
            dt = dt[np.isfinite(dt) & (dt > 0)]
            step = float(np.median(dt)) if len(dt) else 1 / 30
            self._trail_rows = max(1, int(round(TRAIL_S / step)))
        self._ensure_items()
        has_cop = has_force = has_com = False
        if fused is not None:
            has_com = bool(np.isfinite(fused.com_world).all(axis=1).any())
            if board is not None:
                has_cop = bool(np.isfinite(fused.cop_world).all(axis=1).any())
                has_force = bool(np.isfinite(fused.force_world).all(axis=1).any())
        self.board_check.setVisible(board is not None)
        self.cop_check.setVisible(has_cop)
        self.force_check.setVisible(has_force)
        self.com_check.setVisible(has_com)
        self.balance_info.setVisible(fused is not None)
        self._update_board_items()
        self._update_balance_frame()

    def clear_balance(self) -> None:
        """Remove all balance overlays."""
        self._fused = None
        self._board = None
        self._bal_row = -1
        for item in self._bal_items.values():
            item.setVisible(False)
        for chk in self._toggles:
            chk.setVisible(False)
        self.balance_info.setVisible(False)
        self.balance_info.setText("")

    @property
    def balance_row(self) -> int:
        """``FusedTrial`` row shown for the current frame (-1: none)."""
        return self._bal_row

    def has_balance(self) -> bool:
        return self._fused is not None or self._board is not None

    # ---- Skeleton3DViewer overrides -------------------------------------- #
    def load_trc(self, path) -> bool:
        ok = super().load_trc(path)
        if ok and self._bal_items:
            self._update_board_items()  # the display transform changed
            self._update_balance_frame()
        return ok

    def clear(self):
        super().clear()
        self.clear_balance()

    def _show_frame(self, i: int):
        super()._show_frame(i)
        if self._bal_items:
            self._update_balance_frame()

    # ---- transforms --------------------------------------------------------- #
    def _disp(self, p_world) -> np.ndarray:
        """World (Calib.toml, Z-up) point(s) -> display (N,3), exactly like ``set_cameras``."""
        return self._tf(_zup2yup(np.asarray(p_world, np.float64).reshape(-1, 3)))

    def _disp_dir(self, d_world) -> np.ndarray:
        """World direction -> display direction (3,) (no centre subtraction)."""
        return _zup2yup(np.asarray(d_world, np.float64).reshape(-1, 3))[0] @ self._M.T

    # ---- drawing ------------------------------------------------------------ #
    def _on_toggle(self, _on: bool = True):
        self._update_board_items()
        self._update_balance_frame()

    def _set_visible(self, names, on: bool) -> None:
        for n in names:
            if n in self._bal_items:
                self._bal_items[n].setVisible(on)

    def _update_board_items(self) -> None:
        it = self._bal_items
        if not it:
            return
        reg = self._board
        if reg is None or not self.board_check.isChecked():
            self._set_visible(BOARD_ITEMS, False)
            return
        try:
            corners = self._disp(reg.corners_world)          # TL, TR, BR, BL
            sensors = self._disp(reg.sensors_world)
            up = self._disp_dir(reg.up_world)
            center = self._disp(reg.center_world)[0]
        except Exception:  # noqa: BLE001
            log.exception("cannot draw the board")
            self._set_visible(BOARD_ITEMS, False)
            return
        below = corners - up * LIFT_M
        it["surface"].setMeshData(vertexes=below, faces=np.array([[0, 1, 2], [0, 2, 3]]),
                                  color=_rgba(BOARD_COLOR, 0.16))
        it["outline"].setData(pos=np.vstack([corners, corners[:1]]))
        it["front"].setData(pos=corners[:2])
        it["sensors"].setData(pos=sensors)
        if "front_label" in it:  # outside the front-left (TL) corner, clear of the feet
            out = _unit(corners[0] - center)
            pos = corners[0] + (out * 0.06 if out is not None else 0.0) + up * 0.01
            it["front_label"].setData(pos=pos)
        self._set_visible(BOARD_ITEMS, True)

    def _update_balance_frame(self) -> None:
        if not self._bal_items:
            return
        f = self._fused
        row = -1
        if f is not None and self._trc is not None and self._trc.n_frames:
            try:
                row = int(f.index_for_time(float(self._trc.times[self._frame])))
            except Exception:  # noqa: BLE001
                row = -1
        self._bal_row = row
        self._set_visible(FRAME_ITEMS, False)
        if f is None:
            return
        if row >= 0:
            up_w = _unit(self._board.up_world) if self._board is not None else None
            up_d = self._disp_dir(up_w) if up_w is not None else np.zeros(3)
            self._draw_cop(f, row, up_d)
            self._draw_force(f, row, up_d)
            self._draw_com(f, row, up_w)
        self._update_info(f, row)

    def _draw_cop(self, f, row: int, up_d: np.ndarray) -> None:
        if self._board is None or not self.cop_check.isChecked():
            return
        it = self._bal_items
        cop = f.cop_world[row]
        if np.isfinite(cop).all():
            it["cop"].setData(pos=self._disp(cop) + up_d * LIFT_M)
            it["cop"].setVisible(True)
        # trail: the last TRAIL_S seconds of rows, broken where the COP is unknown
        lo = max(0, row - self._trail_rows)
        pts = f.cop_world[lo:row + 1]
        ok = np.isfinite(pts).all(axis=1)
        pair = ok[:-1] & ok[1:]
        if not pair.any():
            return
        disp = self._disp(np.where(ok[:, None], pts, 0.0)) + up_d * LIFT_M
        idx = np.nonzero(pair)[0]
        seg = np.empty((2 * len(idx), 3))
        seg[0::2], seg[1::2] = disp[idx], disp[idx + 1]
        alpha = 0.15 + 0.85 * (np.arange(len(pts)) / max(len(pts) - 1, 1))
        col = np.tile(np.array(_rgba(COP_COLOR)), (len(seg), 1))
        col[0::2, 3], col[1::2, 3] = alpha[idx], alpha[idx + 1]
        it["cop_trail"].setData(pos=seg, color=col)
        it["cop_trail"].setVisible(True)

    def _draw_force(self, f, row: int, up_d: np.ndarray) -> None:
        if self._board is None or not self.force_check.isChecked():
            return
        cop = f.cop_world[row]
        total = float(f.total_kg[row])
        d = _unit(f.force_world[row])
        if d is None or not np.isfinite(cop).all() or not np.isfinite(total) or total <= 0:
            return
        axis = _unit(self._disp_dir(d))
        if axis is None:
            return
        length = total * ARROW_M_PER_KG
        start = self._disp(cop)[0] + up_d * LIFT_M
        head = min(HEAD_LEN_M, 0.4 * length)
        tip = start + axis * length
        it = self._bal_items
        it["force"].setData(pos=np.vstack([start, tip - axis * head * 0.5]))
        verts, faces = _cone(tip, axis, head, HEAD_RADIUS_M * head / HEAD_LEN_M)
        it["force_head"].setMeshData(vertexes=verts, faces=faces, color=_rgba(FORCE_COLOR))
        it["force"].setVisible(True)
        it["force_head"].setVisible(True)

    def _draw_com(self, f, row: int, up_w: np.ndarray | None) -> None:
        if not self.com_check.isChecked():
            return
        com = f.com_world[row]
        if not np.isfinite(com).all():
            return
        it = self._bal_items
        c = self._disp(com)
        it["com"].setData(pos=c)
        it["com"].setVisible(True)
        if self._board is None or up_w is None:
            return
        t = np.asarray(self._board.pose.board_to_world.t, np.float64)
        foot_w = com - float((com - t) @ up_w) * up_w  # = fusion.plumb_point
        foot = self._disp(foot_w)
        it["plumb"].setData(pos=np.vstack([c, foot]))
        it["plumb_foot"].setData(pos=foot)
        it["plumb"].setVisible(True)
        it["plumb_foot"].setVisible(True)

    def _update_info(self, f, row: int) -> None:
        if row < 0:
            self.balance_info.setText("<small>Wii: no data for this frame</small>")
            return
        t_rel = float(f.t_rel[row])
        if not np.isfinite(t_rel):
            self.balance_info.setText("<small>Wii: not aligned with the videos yet "
                                      "(6. Results &gt; Balance (Wii))</small>")
            return
        parts = [f"Wii t {t_rel:.3f} s"]
        total = float(f.total_kg[row])
        if np.isfinite(total):
            bm = f.body_mass_kg
            parts.append(f"<span style='color:{FORCE_COLOR}'>■</span> load {total:.1f} kg"
                         + (f" ({100 * total / bm:.0f} % BW)" if bm else ""))
        else:
            parts.append("no force data")
        cop = f.cop_board[row]
        if np.isfinite(cop).all():
            parts.append(f"<span style='color:{COP_COLOR}'>■</span> COP ML "
                         f"{cop[0] * 1000:+.0f} / AP {cop[1] * 1000:+.0f} mm")
        cmc = f.com_minus_cop[row]
        if np.isfinite(cmc).all():
            parts.append(f"<span style='color:{COM_COLOR}'>■</span> COM−COP ML "
                         f"{cmc[0] * 1000:+.0f} / AP {cmc[1] * 1000:+.0f} mm")
        self.balance_info.setText("<small>" + "&nbsp;&nbsp;·&nbsp;&nbsp;".join(parts)
                                  + "</small>")
