"""Interactive checkerboard corner picker.

A zoomable/pannable image view where the user clicks the board's inner corners
(or auto-detects them). Points are numbered in click order; drag to fine-tune,
right-click to remove the nearest. Coordinates are kept in image pixels so they
feed straight into solvePnP for extrinsic calibration.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from PySide6.QtCore import Qt, Signal, QPointF, QRectF
from PySide6.QtGui import QBrush, QColor, QImage, QPen, QPixmap
from PySide6.QtWidgets import (
    QGraphicsEllipseItem, QGraphicsItem, QGraphicsPixmapItem, QGraphicsScene,
    QGraphicsSimpleTextItem, QGraphicsView,
)

# Dot radius in SCREEN pixels (constant regardless of zoom, like LabelMe), so
# points stay small and don't merge when you zoom into the image.
_DOT_R = 4.0
_IGNORE_TF = QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations


class CornerPicker(QGraphicsView):
    """Click checkerboard corners on an image. Emits `points_changed(count)`."""

    points_changed = Signal(int)

    def __init__(self):
        super().__init__()
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setDragMode(QGraphicsView.ScrollHandDrag)  # left-drag pans
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setRenderHints(self.renderHints())

        self._pix_item: Optional[QGraphicsPixmapItem] = None
        self._img_size = (0, 0)  # (w, h)
        self._points: list[QPointF] = []
        self._dot_items: list[QGraphicsEllipseItem] = []
        self._label_items: list[QGraphicsSimpleTextItem] = []
        self._dragging: Optional[int] = None
        self._maybe_add: bool = False
        self._press_view = None
        self._gray: Optional[np.ndarray] = None   # for sub-pixel corner snapping
        # OFF by default: on densely-packed boards snapping can jump to the wrong
        # corner. Primary workflow is zoom + precise manual click + drag-adjust.
        self.snap_enabled: bool = False

    # ---- image loading ---------------------------------------------------- #
    def load_image(self, path: Path) -> bool:
        import cv2
        pix = QPixmap(str(path))
        if pix.isNull():
            return False
        self._set_pixmap(pix)
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        self._gray = img
        return True

    def load_bgr(self, img: np.ndarray) -> None:
        import cv2
        h, w = img.shape[:2]
        rgb = img[:, :, ::-1].copy()
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
        self._set_pixmap(QPixmap.fromImage(qimg))
        self._gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def _snap_corner(self, scene_pt: QPointF, search: int = 6) -> QPointF:
        """Snap a rough/blurry click to the nearest real checkerboard corner.

        Finds corner candidates in a small window around the click (Harris /
        goodFeaturesToTrack, robust even when the full board isn't detectable),
        picks the one nearest the click, and refines it to sub-pixel accuracy.
        Guarded: if nothing suitable is near, keep the original click.
        """
        if not self.snap_enabled or self._gray is None:
            return scene_pt
        try:
            import cv2
            x, y = float(scene_pt.x()), float(scene_pt.y())
            h, w = self._gray.shape[:2]
            x0, y0 = int(round(x)) - search, int(round(y)) - search
            x1, y1 = x0 + 2 * search, y0 + 2 * search
            if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
                return scene_pt
            win = self._gray[y0:y1, x0:x1]
            cand = cv2.goodFeaturesToTrack(win, maxCorners=8, qualityLevel=0.01,
                                           minDistance=4, useHarrisDetector=True, k=0.04)
            if cand is None:
                return scene_pt
            cand = cand.reshape(-1, 2) + np.array([x0, y0], dtype=np.float32)
            d = np.hypot(cand[:, 0] - x, cand[:, 1] - y)
            best = cand[int(np.argmin(d))]
            if float(d.min()) > search:      # nearest corner too far -> trust the click
                return scene_pt
            pt = best.reshape(1, 1, 2).astype(np.float32)
            crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)
            cv2.cornerSubPix(self._gray, pt, (5, 5), (-1, -1), crit)
            return QPointF(float(pt[0, 0, 0]), float(pt[0, 0, 1]))
        except Exception:  # noqa: BLE001 - snapping is best-effort
            pass
        return scene_pt

    def _set_pixmap(self, pix: QPixmap) -> None:
        self.clear_points()
        self._scene.clear()
        self._pix_item = self._scene.addPixmap(pix)
        self._img_size = (pix.width(), pix.height())
        self._scene.setSceneRect(QRectF(pix.rect()))
        self.resetTransform()
        self.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio)

    # ---- points ----------------------------------------------------------- #
    def points(self) -> list[tuple[float, float]]:
        return [(p.x(), p.y()) for p in self._points]

    def set_points(self, pts) -> None:
        self.clear_points()
        for x, y in pts:
            self._add_point(QPointF(float(x), float(y)))
        self.points_changed.emit(len(self._points))

    def reverse_order(self) -> None:
        """Flip the corner order (origin <-> last). Fixes the 180-degree board
        ambiguity so every camera's origin corner is the same physical corner."""
        if self._points:
            self.set_points([(p.x(), p.y()) for p in reversed(self._points)])
            self.points_changed.emit(len(self._points))

    def set_grid_shape(self, corners_nb) -> None:
        """Tell the picker the board's (rows, cols) so `rotate_90` can re-index
        the grid. Call after loading/auto-detecting a full board."""
        self._grid_rows = int(corners_nb[0])
        self._grid_cols = int(corners_nb[1])

    def rotate_90(self, clockwise: bool = True) -> bool:
        """Rotate the corner ORDER by a quarter turn (re-number as if the board
        were physically turned 90 degrees). For a non-square board this swaps
        rows<->cols; the 2D points themselves don't move — only their order and
        which one is the green origin. Needs the full grid (rows*cols points).

        Handles the case auto-detect can't: a board seen rotated ~90 degrees in
        one camera, so its detected grid is transposed relative to the others.
        """
        import numpy as np
        r = getattr(self, "_grid_rows", 0)
        c = getattr(self, "_grid_cols", 0)
        pts = self.points()
        if r * c == 0 or len(pts) != r * c:
            return False
        grid = np.asarray(pts, float).reshape(r, c, 2)
        rot = np.rot90(grid, k=(-1 if clockwise else 1))  # k>0 = CCW in numpy
        self._grid_rows, self._grid_cols = int(rot.shape[0]), int(rot.shape[1])
        self.set_points([(float(x), float(y)) for x, y in rot.reshape(-1, 2)])
        self.points_changed.emit(len(self._points))
        return True

    def clear_points(self) -> None:
        for it in self._dot_items + self._label_items:
            if it.scene():
                self._scene.removeItem(it)
        self._points.clear()
        self._dot_items.clear()
        self._label_items.clear()

    def _add_point(self, scene_pt: QPointF) -> None:
        idx = len(self._points)
        self._points.append(scene_pt)
        is_origin = idx == 0
        r = _DOT_R * (1.5 if is_origin else 1.0)
        # Hollow ring (no fill) so the exact corner shows through the centre and
        # nearby points don't blob together; ItemIgnoresTransformations keeps it a
        # constant SCREEN size at any zoom (like LabelMe).
        dot = QGraphicsEllipseItem(-r, -r, 2 * r, 2 * r)
        dot.setBrush(QBrush(Qt.NoBrush))
        dot.setPen(QPen(QColor(40, 220, 90) if is_origin else QColor(255, 70, 70),
                        2.0 if is_origin else 1.5))
        dot.setPos(scene_pt)
        dot.setFlag(_IGNORE_TF, True)
        dot.setZValue(12 if is_origin else 10)
        self._scene.addItem(dot)
        self._dot_items.append(dot)

        label = QGraphicsSimpleTextItem("0" if is_origin else str(idx))
        label.setBrush(QBrush(QColor(120, 255, 120) if is_origin else QColor(255, 235, 60)))
        label.setPos(scene_pt)
        label.setFlag(_IGNORE_TF, True)
        # nudge the number a few screen-px up-right of the ring so it never covers
        # the corner itself
        label.setTransform(label.transform().translate(r + 1, -2 * r - 6))
        label.setZValue(13)
        self._scene.addItem(label)
        self._label_items.append(label)

    def _nearest_point(self, scene_pt: QPointF, screen_px: float = 10.0) -> Optional[int]:
        # threshold in SCREEN pixels -> scene units via current zoom, so grabbing
        # a point feels the same at any zoom and you can place close points when
        # zoomed in without snapping onto a neighbour.
        scale = abs(self.transform().m11()) or 1.0
        max_dist = screen_px / scale
        best, best_d = None, max_dist
        for i, p in enumerate(self._points):
            d = (p - scene_pt).manhattanLength()
            if d < best_d:
                best, best_d = i, d
        return best

    # ---- auto-detect ------------------------------------------------------ #
    def auto_detect(self, corners_nb) -> bool:
        """Fill corners via OpenCV checkerboard detection on the current image."""
        import cv2

        if self._pix_item is None:
            return False
        qimg = self._pix_item.pixmap().toImage().convertToFormat(QImage.Format_RGB888)
        w, h = qimg.width(), qimg.height()
        ptr = qimg.constBits()
        arr = np.frombuffer(ptr, np.uint8).reshape(h, qimg.bytesPerLine())[:, : w * 3].reshape(h, w, 3)
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        rows, cols = int(corners_nb[0]), int(corners_nb[1])
        found, corners = cv2.findChessboardCorners(
            gray, (cols, rows),
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
        if not found:
            return False
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), crit)
        pts = np.asarray(corners, dtype=float).reshape(-1, 2)
        self.set_points([(float(x), float(y)) for x, y in pts])
        return True

    # ---- mouse ------------------------------------------------------------ #
    def wheelEvent(self, ev):
        factor = 1.25 if ev.angleDelta().y() > 0 else 0.8
        self.scale(factor, factor)

    def mousePressEvent(self, ev):
        view_pt = ev.position().toPoint()
        scene_pt = self.mapToScene(view_pt)
        if ev.button() == Qt.RightButton:
            i = self._nearest_point(scene_pt)
            if i is not None:
                self._rebuild_without(i)
            return
        if ev.button() == Qt.LeftButton:
            self._press_view = view_pt
            i = self._nearest_point(scene_pt)
            if i is not None:
                # grab an existing point to fine-tune it
                self._dragging = i
                self.setDragMode(QGraphicsView.NoDrag)
                return
            # empty area: let ScrollHandDrag begin a pan; on release with no
            # movement we treat it as a click and ADD a point there.
            self._maybe_add = True
        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev):
        if self._dragging is not None:
            scene_pt = self.mapToScene(ev.position().toPoint())
            self._points[self._dragging] = scene_pt
            self._dot_items[self._dragging].setPos(scene_pt)
            self._label_items[self._dragging].setPos(scene_pt)
            return
        super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev):
        if self._dragging is not None:
            # snap the fine-tuned point to the precise corner
            snapped = self._snap_corner(self._points[self._dragging])
            self._points[self._dragging] = snapped
            self._dot_items[self._dragging].setPos(snapped)
            self._label_items[self._dragging].setPos(snapped)
            self._dragging = None
            self.setDragMode(QGraphicsView.ScrollHandDrag)
            return
        if ev.button() == Qt.LeftButton and getattr(self, "_maybe_add", False):
            self._maybe_add = False
            super().mouseReleaseEvent(ev)  # end any pan grab
            moved = (ev.position().toPoint() - self._press_view).manhattanLength()
            if moved <= 4:  # a click, not a pan -> add a corner here
                scene_pt = self.mapToScene(self._press_view)
                if self._pix_item and self._scene.sceneRect().contains(scene_pt):
                    self._add_point(self._snap_corner(scene_pt))  # sub-pixel snap
                    self.points_changed.emit(len(self._points))
            return
        super().mouseReleaseEvent(ev)

    def _rebuild_without(self, idx: int) -> None:
        pts = [(p.x(), p.y()) for j, p in enumerate(self._points) if j != idx]
        self.set_points(pts)
