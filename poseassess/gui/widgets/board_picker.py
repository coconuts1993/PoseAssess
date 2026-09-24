"""``BoardPointPicker``: click the Wii Balance Board landmarks TL, TR, BR, BL, C on a camera
frame. Subclass of the calibration ``CornerPicker`` (zoom/pan/drag behaviour unchanged) with
fixed labels, per-label colours, at most 5 points and a front/back swap.

Behaviour (``CornerPicker`` itself is NOT modified):
* left-click on an empty spot adds the next landmark (``LABELS[n]``); clicks after the 5th are
  ignored; left-drag on a point moves it (``points_changed`` is emitted on release), left-drag
  elsewhere pans, wheel zooms;
* right-click removes the LAST point (``undo_last``): removing a middle point would silently
  relabel the others;
* the image follows the view size (fit) until the user zooms; key F / Home fits it again;
* once the 4 corners exist the outline TL-TR-BR-BL is drawn, the front edge TL-TR thicker;
* ``snap_enabled`` stays False (board corners are rounded); never use ``reverse_order`` (it
  would put C first): use ``swap_front_back``.
"""

from __future__ import annotations

from PySide6.QtCore import QLineF, QPointF, Qt
from PySide6.QtGui import QBrush, QColor, QPen
from PySide6.QtWidgets import (
    QGraphicsEllipseItem,
    QGraphicsLineItem,
    QGraphicsSimpleTextItem,
)

from .corner_picker import _DOT_R, _IGNORE_TF, CornerPicker

LABELS = ("TL", "TR", "BR", "BL", "C")
HINTS = {
    "TL": "front-left corner (front = long edge OPPOSITE the power button; left/right = the "
          "subject's, standing on the board facing the front)",
    "TR": "front-right corner",
    "BR": "back-right corner (power-button edge)",
    "BL": "back-left corner (power-button edge)",
    "C": "centre of the top surface",
}
# Same colours as the QC overlay (core.balance.overlay.CLICK_COLORS, BGR there)
COLORS = {"TL": QColor(255, 40, 40), "TR": QColor(255, 200, 0), "BR": QColor(0, 220, 0),
          "BL": QColor(0, 128, 255), "C": QColor(255, 0, 255)}
OUTLINE = QColor(0, 230, 255)
FRONT = QColor(255, 140, 0)
FRONT_BACK_SWAP = (2, 3, 0, 1, 4)  # new[i] = old[FRONT_BACK_SWAP[i]]: TL<->BR, TR<->BL, C


class BoardPointPicker(CornerPicker):
    """CornerPicker limited to the 5 board landmarks (see module docstring)."""

    LABELS = LABELS

    def __init__(self):
        super().__init__()
        self.snap_enabled = False
        self._outline_items: list[QGraphicsLineItem] = []
        self._user_zoom = False  # the image follows the view size until the user zooms

    def fit(self) -> None:
        """Show the whole image (also: key F / Home)."""
        if self._pix_item is not None:
            self.resetTransform()
            self.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio)
        self._user_zoom = False

    def next_label(self) -> str | None:
        """Label of the next click, or None when all 5 are set."""
        n = len(self.points())
        return LABELS[n] if n < len(LABELS) else None

    def has_image(self) -> bool:
        return self._pix_item is not None

    def image_size(self) -> tuple[int, int] | None:
        """(w, h) of the shown image, or None."""
        return tuple(self._img_size) if self._pix_item is not None else None

    def clear_image(self) -> None:
        """Remove the image and the points."""
        self.clear_points()
        self._scene.clear()
        self._pix_item = None
        self._img_size = (0, 0)
        self._gray = None
        self.points_changed.emit(0)

    def undo_last(self) -> None:
        """Remove the last point."""
        pts = self.points()
        if pts:
            self.set_points(pts[:-1])

    def swap_front_back(self) -> bool:
        """[TL, TR, BR, BL, C] -> [BR, BL, TL, TR, C] (only with 5 points); emits. Returns
        whether the points were swapped."""
        pts = self.points()
        if len(pts) != len(LABELS):
            return False
        self.set_points([pts[j] for j in FRONT_BACK_SWAP])
        return True

    # ------------------------------------------------------------ overrides
    def _set_pixmap(self, pix) -> None:
        super()._set_pixmap(pix)
        self._user_zoom = False

    def wheelEvent(self, ev):
        self._user_zoom = True
        super().wheelEvent(ev)

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        if not self._user_zoom:
            self.fit()

    def showEvent(self, ev):
        super().showEvent(ev)
        if not self._user_zoom:
            self.fit()

    def keyPressEvent(self, ev):
        if ev.key() in (Qt.Key_F, Qt.Key_Home):
            self.fit()
            return
        super().keyPressEvent(ev)

    def clear_points(self) -> None:
        self._clear_outline()
        super().clear_points()

    def _add_point(self, scene_pt: QPointF) -> None:
        idx = len(self._points)
        if idx >= len(LABELS):
            return  # all 5 landmarks set: ignore further clicks
        name = LABELS[idx]
        color = COLORS[name]
        self._points.append(scene_pt)
        r = _DOT_R * 1.5
        dot = QGraphicsEllipseItem(-r, -r, 2 * r, 2 * r)
        dot.setBrush(QBrush(Qt.NoBrush))
        dot.setPen(QPen(color, 2.0))
        dot.setPos(scene_pt)
        dot.setFlag(_IGNORE_TF, True)
        dot.setZValue(12)
        self._scene.addItem(dot)
        self._dot_items.append(dot)

        label = QGraphicsSimpleTextItem(f"{idx + 1}:{name}")
        f = label.font()
        f.setBold(True)
        label.setFont(f)
        label.setBrush(QBrush(color))
        label.setPen(QPen(QColor(0, 0, 0), 0.6))
        label.setPos(scene_pt)
        label.setFlag(_IGNORE_TF, True)
        label.setTransform(label.transform().translate(r + 1, -2 * r - 10))
        label.setZValue(13)
        self._scene.addItem(label)
        self._label_items.append(label)
        self._update_outline()

    def mousePressEvent(self, ev):
        if ev.button() == Qt.RightButton:
            self.undo_last()
            return
        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev):
        super().mouseMoveEvent(ev)
        if self._dragging is not None:
            self._update_outline()

    def mouseReleaseEvent(self, ev):
        was_dragging = self._dragging is not None
        super().mouseReleaseEvent(ev)
        if was_dragging:
            self._update_outline()
            self.points_changed.emit(len(self._points))

    # -------------------------------------------------------------- outline
    def _clear_outline(self) -> None:
        for it in self._outline_items:
            try:
                if it.scene() is not None:
                    self._scene.removeItem(it)
            except RuntimeError:  # already deleted with the scene
                pass
        self._outline_items = []

    def _update_outline(self) -> None:
        self._clear_outline()
        if len(self._points) < 4:
            return
        corners = self._points[:4]
        for i in range(4):
            a, b = corners[i], corners[(i + 1) % 4]
            front = i == 0  # TL -> TR
            pen = QPen(FRONT if front else OUTLINE, 3.0 if front else 1.5)
            pen.setCosmetic(True)
            if not front:
                pen.setStyle(Qt.DashLine)
            item = QGraphicsLineItem(QLineF(a, b))
            item.setPen(pen)
            item.setZValue(8)
            self._scene.addItem(item)
            self._outline_items.append(item)
