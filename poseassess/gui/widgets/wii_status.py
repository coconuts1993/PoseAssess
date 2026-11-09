"""Permanent status-bar widget showing the Wii Balance Board state (``WiiController``).

A coloured dot (green connected, amber searching/connecting, red error / hidapi missing, grey
off) and ``controller.status_text()``, elided so a long error never widens the main window.
Clicking it opens the "2b. Wii Board" page (found through the main window's ``pages``/``nav``,
so the shared shell needs no hook).
"""

from __future__ import annotations

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtWidgets import QLabel

GOOD, WARN, BAD, OFF = "#3ad35a", "#e0a94c", "#e05c5c", "#8a8f98"
MAX_TEXT_PX = 380  # the status text is elided beyond this (full text in the tooltip)

HELP = ("Wii Balance Board (optional). Pair it once via Bluetooth; it connects automatically "
        "when switched on. Click to open the \"2b. Wii Board\" page.")


def state_color(controller) -> str:
    """Dot colour for the controller's current state."""
    if controller.source is None:
        return OFF
    if getattr(controller, "fatal_error", None):
        return BAD
    st = controller.state
    if st == "connected":
        return GOOD
    if st in ("searching", "connecting"):
        return WARN
    if st.startswith("error"):
        return BAD
    return OFF


class WiiStatusWidget(QLabel):
    """Shows ``controller.status_text()``; updates on ``status_changed`` / ``state_changed``.
    Emits ``clicked`` and jumps to the Wii Board page when clicked."""

    clicked = Signal()

    def __init__(self, controller, parent=None):
        super().__init__(parent)
        self._controller = controller
        self._text = ""
        self.setObjectName("wiiStatus")
        self.setTextFormat(Qt.RichText)
        self.setCursor(Qt.PointingHandCursor)
        self.setContentsMargins(6, 0, 6, 0)
        controller.status_changed.connect(self._on_status)
        self._on_status(controller.status_text())

    # ------------------------------------------------------------------ text
    def full_text(self) -> str:
        """The unelided status line."""
        return self._text

    def _on_status(self, text: str) -> None:
        self._text = str(text)
        fm = self.fontMetrics()
        shown = fm.elidedText(self._text, Qt.ElideRight, MAX_TEXT_PX)
        color = state_color(self._controller)
        self.setText(f"<span style='color:{color}; font-size:14px'>●</span>&nbsp;"
                     f"{_html(shown)}")
        self.setToolTip(f"{self._text}\n\n{HELP}")

    def minimumSizeHint(self) -> QSize:  # never let a long status widen the window
        h = super().minimumSizeHint()
        return QSize(min(h.width(), 120), h.height())

    def sizeHint(self) -> QSize:
        h = super().sizeHint()
        return QSize(min(h.width(), MAX_TEXT_PX + 40), h.height())

    # ----------------------------------------------------------------- click
    def mousePressEvent(self, ev):
        if ev.button() == Qt.LeftButton:
            self.clicked.emit()
            self.open_wii_page()
        super().mousePressEvent(ev)

    def open_wii_page(self) -> bool:
        """Select the "2b. Wii Board" page in the main window; True when found."""
        win = self.window()
        pages, nav = getattr(win, "pages", None), getattr(win, "nav", None)
        if not pages or nav is None:
            return False
        for i, page in enumerate(pages):
            if type(page).__name__ == "WiiBoardPage":
                nav.setCurrentRow(i)
                return True
        return False


def _html(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
