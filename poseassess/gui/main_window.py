"""Main window: left step-navigator + stacked pages."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout, QListWidget, QListWidgetItem, QMainWindow, QStackedWidget,
    QStatusBar, QWidget,
)

from .state import AppState
from .pages import ALL_PAGES


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

        self.setStyleSheet(_STYLE)

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
