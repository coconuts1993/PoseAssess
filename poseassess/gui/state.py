"""Shared application state for the GUI.

A single `AppState` holds the currently-open Project and notifies pages when it
changes, so each page can enable/disable itself and refresh its view.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtCore import QObject, Signal

from poseassess.core.project import Project


class AppState(QObject):
    project_changed = Signal(object)  # emits Project | None

    def __init__(self):
        super().__init__()
        self._project: Optional[Project] = None

    @property
    def project(self) -> Optional[Project]:
        return self._project

    def set_project(self, project: Optional[Project]) -> None:
        self._project = project
        self.project_changed.emit(project)

    def open_project(self, root: str | Path) -> Project:
        proj = Project.load(root)
        self.set_project(proj)
        return proj

    def has_project(self) -> bool:
        return self._project is not None
