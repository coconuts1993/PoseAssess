"""Shared application state for the GUI.

A single `AppState` holds the currently-open Project and notifies pages when it
changes, so each page can enable/disable itself and refresh its view.

It also owns the shared Wii Balance Board connection (`wii`, created on first use),
`balance_changed`, emitted when a page changed Wii data of the project on disk (board
registration, trial recording, alignment) so other pages can reload, and app-wide "busy"
flags (`busy` / `set_busy`): a capture export replacing videos/ ("export") and a pipeline run
reading them ("pipeline") must not overlap.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtCore import QObject, Signal

from poseassess.core.project import Project


class AppState(QObject):
    project_changed = Signal(object)  # emits Project | None
    # "board" | "trial" | "recording" | "alignment": Wii data of the project changed on disk
    balance_changed = Signal(str)
    busy_changed = Signal()  # a `set_busy` flag was set or cleared

    def __init__(self):
        super().__init__()
        self._project: Optional[Project] = None
        self._wii = None
        self._busy: dict[str, str] = {}

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

    # ---- Wii Balance Board (optional feature) ----------------------------- #
    @property
    def wii(self):
        """The shared `WiiController` (created on first access; never imports hidapi)."""
        if self._wii is None:
            from .wii_controller import WiiController
            self._wii = WiiController(self)
        return self._wii

    def busy(self, key: str) -> str | None:
        """What holds ``key`` right now, as text for a message (e.g. "the export of the take
        … to videos/"), or None. Keys: "export" (3b. Capture replaces videos/ and Config.toml),
        "pipeline" (4. Run reads them)."""
        return self._busy.get(key)

    def set_busy(self, key: str, text: str | None) -> None:
        """Set (``text``) or clear (None) the busy flag ``key``; emits ``busy_changed``."""
        if text:
            self._busy[key] = text
        elif self._busy.pop(key, None) is None:
            return
        self.busy_changed.emit()

    def notify_balance_changed(self, what: str) -> None:
        """Tell every page that Wii data of the project changed ("board" | "trial" |
        "recording" | "alignment")."""
        self.balance_changed.emit(what)

    def shutdown(self) -> None:
        """Stop background services (called by MainWindow on close)."""
        if self._wii is not None:
            self._wii.shutdown()
