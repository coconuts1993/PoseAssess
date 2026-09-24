"""Base class for wizard-style pages."""
from __future__ import annotations

from PySide6.QtWidgets import QWidget

from ..state import AppState


class BasePage(QWidget):
    #: Title shown in the left navigation.
    nav_title = "Page"
    #: Whether this page needs an open project to be usable.
    needs_project = True

    def __init__(self, state: AppState):
        super().__init__()
        self.state = state
        self.state.project_changed.connect(self._on_project_changed)

    def _on_project_changed(self, project) -> None:
        """Refresh page contents when the active project changes."""
        self.setEnabled(project is not None or not self.needs_project)
        self.on_project_changed(project)

    def on_project_changed(self, project) -> None:  # override
        pass

    def can_close(self) -> bool:  # override
        """Asked by MainWindow before the app closes; return False to cancel (e.g. after asking
        the user whether to stop a running recording)."""
        return True

    def shutdown(self) -> None:  # override
        """Called once when the app closes: stop threads, timers, cameras."""
