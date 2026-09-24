"""``CameraGrid``: live preview thumbnails of the capture cameras (Owner: GUI-CAPTURE).

A grid of cells updated by the page's QTimer from ``CaptureSession.latest(cam)``; each cell
shows the camera name, measured fps and a red "● REC" while recording.

Cells paint the (downscaled) frame themselves instead of using a QLabel pixmap, so a large
camera image never grows the page (the window's minimum width must not change).
"""

from __future__ import annotations

import math

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QFont, QImage, QPainter
from PySide6.QtWidgets import QGridLayout, QLabel, QSizePolicy, QWidget

REC_COLOR = QColor("#e05c5c")


class _Cell(QWidget):
    """One camera: the frame (aspect kept, letterboxed) + overlay texts."""

    def __init__(self, name: str, parent=None):
        super().__init__(parent)
        self.name = name
        self.image: QImage | None = None
        self.text = ""
        self.message = "not open"
        self.recording = False
        self.setMinimumSize(120, 80)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)

    def set_image(self, image: QImage | None, text: str) -> None:
        self.image, self.text = image, text
        self.update()

    def set_message(self, message: str) -> None:
        self.image, self.message, self.text = None, message, ""
        self.update()

    def paintEvent(self, _event):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(24, 24, 24))
        w, h = self.width(), self.height()
        if self.image is not None and not self.image.isNull():
            iw, ih = self.image.width(), self.image.height()
            s = min(w / iw, h / ih)
            dw, dh = iw * s, ih * s
            p.setRenderHint(QPainter.SmoothPixmapTransform)
            p.drawImage(QRectF((w - dw) / 2, (h - dh) / 2, dw, dh), self.image)
        else:
            p.setPen(QColor(170, 170, 170))
            p.drawText(self.rect(), Qt.AlignCenter, f"{self.name}\n{self.message}")
        font = QFont(self.font())
        font.setBold(True)
        p.setFont(font)
        label = self.name + (f"  ·  {self.text}" if self.text else "")
        if self.image is not None:
            p.setPen(QColor(0, 0, 0, 200))
            p.drawText(9, 19, label)
            p.setPen(QColor(255, 255, 255))
            p.drawText(8, 18, label)
        if self.recording:
            p.setPen(REC_COLOR)
            p.drawText(QRectF(0, 4, w - 8, 20), Qt.AlignRight | Qt.AlignTop, "● REC")
        p.end()


class CameraGrid(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._grid = QGridLayout(self)
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._grid.setSpacing(4)
        self._cells: dict[str, _Cell] = {}
        self._recording = False
        self._empty = QLabel("No cameras: open a project with cameras.")
        self._empty.setAlignment(Qt.AlignCenter)
        self._grid.addWidget(self._empty, 0, 0)
        self.setMinimumSize(240, 160)

    # ------------------------------------------------------------------ cells
    def names(self) -> list[str]:
        return list(self._cells)

    def cell(self, name: str) -> _Cell | None:
        return self._cells.get(name)

    def set_cameras(self, names: list[str]) -> None:
        """(Re)build one cell per camera name."""
        names = list(names)
        if names == list(self._cells):
            return
        for c in self._cells.values():
            self._grid.removeWidget(c)
            c.deleteLater()
        self._cells = {}
        self._empty.setVisible(not names)
        cols = max(1, math.ceil(math.sqrt(len(names))))
        for i, n in enumerate(names):
            cell = _Cell(n, self)
            cell.recording = self._recording
            self._cells[n] = cell
            self._grid.addWidget(cell, 1 + i // cols, i % cols)
        for r in range(self._grid.rowCount()):
            self._grid.setRowStretch(r, 1 if r >= 1 else 0)
        for c in range(self._grid.columnCount()):
            self._grid.setColumnStretch(c, 1)

    def set_frame(self, name: str, bgr, text: str = "") -> None:
        """Show a BGR frame (numpy) in the cell of ``name`` with an overlay text."""
        cell = self._cells.get(name)
        if cell is None:
            return
        if bgr is None:
            cell.set_message(text or "no frames yet")
            return
        cell.set_image(_to_qimage(bgr, cell.width(), cell.height()), text)

    def set_message(self, name: str, message: str) -> None:
        """Show a text instead of a frame (camera closed, no frames, error)."""
        cell = self._cells.get(name)
        if cell is not None:
            cell.set_message(message)

    def set_recording(self, on: bool) -> None:
        """Show the red "● REC" in every cell."""
        self._recording = bool(on)
        for c in self._cells.values():
            c.recording = self._recording
            c.update()


def _to_qimage(bgr, max_w: int, max_h: int) -> QImage:
    """BGR (or gray) numpy image -> RGB QImage, downscaled to about the cell size (the cell
    scales it again when painting; shrinking with OpenCV first keeps previews cheap)."""
    import cv2
    import numpy as np

    img = np.asarray(bgr)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    h, w = img.shape[:2]
    s = min(1.0, max(max_w, 1) / w, max(max_h, 1) / h)
    if s < 1.0:
        img = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))),
                         interpolation=cv2.INTER_AREA)
    rgb = np.ascontiguousarray(img[:, :, ::-1])
    h, w = rgb.shape[:2]
    return QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()
