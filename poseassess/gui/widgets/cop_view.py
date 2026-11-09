"""Top-down Wii Balance Board view with COP / COM trails (ported from PoseBoard 3ea6d8a,
``poseboard/gui/widgets.py`` CopView; QPainter only).

Board frame: +x = subject's right (TR/BR side), +y = front (the TL/TR long edge OPPOSITE the
power button), metres. The front edge is drawn at the top.

PoseAssess changes (same API): the "power button" label and the legend line get their own space
(no overlap in short views), the trails are drawn as polylines, and the legend lists the COM
only when a COM trail is given.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QWidget


class CopView(QWidget):
    """Top-down board view: outline, sensors, COP trail (blue) and projected COM trail (orange)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(300, 200)
        self.board_mm = (511.0, 316.0)
        self.sensor_mm = (433.0, 238.0)
        self.cop_trail = np.zeros((0, 2))
        self.com_trail = np.zeros((0, 2))
        self.total_kg = 0.0
        self.text = ""

    def set_data(self, cop_trail, com_trail, total_kg, text=""):
        self.cop_trail = np.asarray(cop_trail, float).reshape(-1, 2)
        self.com_trail = np.asarray(com_trail, float).reshape(-1, 2)
        self.total_kg = total_kg
        self.text = text
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor(250, 250, 250))
        bw, bh = self.board_mm[0] / 1000, self.board_mm[1] / 1000
        margin = 20
        top, bottom = 16, 30  # header line / "power button" label + legend line
        avail_h = max(1.0, self.height() - 2 * margin - top - bottom)
        s = max(1e-6, min((self.width() - 2 * margin) / bw, avail_h / bh))
        cx, cy = self.width() / 2, margin + top + avail_h / 2

        def pt(x, y):  # board coords (meters, y forward) -> screen (front facing up)
            return QPointF(cx + x * s, cy - y * s)

        p.setPen(QPen(QColor(90, 90, 90), 2))
        p.setBrush(QColor(235, 235, 235))
        p.drawRoundedRect(QRectF(pt(-bw / 2, bh / 2), pt(bw / 2, -bh / 2)), 12, 12)
        p.setPen(QPen(QColor(200, 200, 200), 1, Qt.DashLine))
        p.drawLine(pt(-bw / 2, 0), pt(bw / 2, 0))
        p.drawLine(pt(0, -bh / 2), pt(0, bh / 2))
        sx, sy = self.sensor_mm[0] / 2000, self.sensor_mm[1] / 2000
        p.setPen(QColor(120, 120, 120))
        for name, (x, y) in {"TL": (-sx, sy), "TR": (sx, sy), "BL": (-sx, -sy), "BR": (sx, -sy)}.items():
            p.setBrush(QColor(150, 150, 150))
            p.drawEllipse(pt(x, y), 5, 5)
            p.drawText(pt(x, y) + QPointF(-8, -9 if y > 0 else 20), name)
        p.drawText(QPointF(8, 14), "Front ↑ = TL/TR edge (opposite the power button)")
        p.drawText(pt(0, -bh / 2) + QPointF(-40, 14), "power button")

        for trail, color in ((self.com_trail, QColor(255, 140, 0)), (self.cop_trail, QColor(30, 100, 220))):
            ok = trail[np.all(np.isfinite(trail), axis=1)]
            if len(ok) > 1:
                c = QColor(color)
                c.setAlpha(140)
                p.setPen(QPen(c, 1.5))
                p.setBrush(Qt.NoBrush)
                p.drawPolyline(QPolygonF([pt(*a) for a in ok]))
            if len(ok):
                p.setPen(Qt.NoPen)
                p.setBrush(color)
                p.drawEllipse(pt(*ok[-1]), 6, 6)
        p.setPen(QColor(30, 30, 30))
        legend = "● COP (blue)" + ("  ● COM projection (orange)" if len(self.com_trail) else "")
        p.drawText(QPointF(8, self.height() - 8),
                   f"{self.total_kg:6.1f} kg".replace("-0.0 ", " 0.0 ")
                   + f"   {self.text}   {legend}")
