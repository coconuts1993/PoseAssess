"""Main window: left step-navigator + stacked pages."""
from __future__ import annotations

import logging

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QHBoxLayout, QListWidget, QListWidgetItem, QMainWindow, QStackedWidget,
    QStatusBar, QWidget,
)

from .state import AppState
from .pages import ALL_PAGES

log = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PoseAssess — 3D Rehab Pose Assessment")
        self.resize(1000, 680)

        self.state = AppState()

        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.nav = QListWidget()
        self.nav.setFixedWidth(190)
        self.nav.setObjectName("nav")
        layout.addWidget(self.nav)

        self.stack = QStackedWidget()
        layout.addWidget(self.stack, 1)

        self.pages = []
        for PageCls in ALL_PAGES:
            page = PageCls(self.state)
            self.pages.append(page)
            self.stack.addWidget(page)
            QListWidgetItem(page.nav_title, self.nav)

        self.nav.currentRowChanged.connect(self.stack.setCurrentIndex)
        self.nav.setCurrentRow(0)

        self.setStatusBar(QStatusBar())
        self.state.project_changed.connect(self._on_project_changed)
        self._on_project_changed(None)
        # The capture export (3b. Capture) writes videos/camNN.mp4: keep the table of
        # "3. Videos" in sync (that page itself only refreshes on project change / Refresh).
        self.state.balance_changed.connect(self._on_balance_changed)

        # Wii Balance Board (optional): permanent status-bar item + plug-and-play start once
        # the event loop runs. Failures here must never prevent the app from starting.
        self._wii_status = None
        try:
            from .widgets.wii_status import WiiStatusWidget
            self._wii_status = WiiStatusWidget(self.state.wii)
            self.statusBar().addPermanentWidget(self._wii_status)
            QTimer.singleShot(0, self._start_wii)
        except Exception:  # noqa: BLE001
            log.exception("Wii Balance Board support unavailable")

        self.setStyleSheet(_STYLE)

    def _start_wii(self):
        try:
            self.state.wii.start_default()
        except Exception:  # noqa: BLE001
            log.exception("Wii Balance Board auto-connect failed to start")

    def closeEvent(self, event):
        for page in self.pages:
            try:
                if not page.can_close():
                    event.ignore()
                    return
            except Exception:  # noqa: BLE001
                log.exception("%s.can_close failed", type(page).__name__)
        for page in self.pages:
            try:
                page.shutdown()
            except Exception:  # noqa: BLE001
                log.exception("%s.shutdown failed", type(page).__name__)
        try:
            self.state.shutdown()
        except Exception:  # noqa: BLE001
            log.exception("shutdown of background services failed")
        super().closeEvent(event)

    def _on_balance_changed(self, what: str) -> None:
        if what != "trial":
            return
        from .pages.videos_page import VideosPage

        for page in self.pages:
            if isinstance(page, VideosPage):
                try:
                    page._refresh()
                except Exception:  # noqa: BLE001
                    log.exception("refreshing the Videos page failed")

    def _on_project_changed(self, project):
        if project is None:
            self.statusBar().showMessage("No project open")
        else:
            self.statusBar().showMessage(
                f"{project.config.name}  |  {project.config.num_cameras} cams  |  {project.root}")


_STYLE = """
QListWidget#nav {
    background: #2b2f36;
    color: #cfd3da;
    border: none;
    outline: none;
    font-size: 14px;
}
QListWidget#nav::item { padding: 14px 16px; }
QListWidget#nav::item:selected { background: #3d8bfd; color: white; }
QGroupBox { font-weight: 600; margin-top: 8px; }
"""
