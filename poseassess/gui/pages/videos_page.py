"""Videos page: assign one trial video per camera (cam01..camNN)."""
from __future__ import annotations

import re
import shutil
from pathlib import Path

from PySide6.QtWidgets import (
    QFileDialog, QHBoxLayout, QLabel, QMessageBox, QPushButton, QTableWidget,
    QTableWidgetItem, QVBoxLayout,
)

from .base_page import BasePage

_VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".MP4", ".AVI", ".MOV", ".MKV")


def _cam_number(name: str) -> int | None:
    """Pull a camera index from a filename like cam01 / cam_1 / camera3 / 01."""
    m = re.search(r"cam(?:era)?[_\-]?(\d+)", name, re.IGNORECASE)
    if not m:
        m = re.fullmatch(r"0*(\d+)", Path(name).stem)  # bare '01', '3'
    return int(m.group(1)) if m else None


class VideosPage(BasePage):
    nav_title = "3. Videos"

    def __init__(self, state):
        super().__init__(state)
        root = QVBoxLayout(self)
        root.addWidget(QLabel(
            "Assign one trial video per camera. Files are copied into the "
            "project's videos/ folder as cam01.mp4 … camNN.mp4."))

        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Camera", "Assigned file", "Status"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setColumnWidth(1, 380)
        root.addWidget(self.table, 1)

        row = QHBoxLayout()
        assign = QPushButton("Assign video to selected camera…")
        assign.clicked.connect(self._assign)
        bulk = QPushButton("Assign all from folder…")
        bulk.setToolTip("Pick a folder; files named cam01…camNN (or 01…NN) are "
                        "matched to each camera and copied in. Frame rate is kept "
                        "as-is (no downsampling).")
        bulk.clicked.connect(self._assign_folder)
        refresh = QPushButton("Refresh")
        refresh.clicked.connect(self._refresh)
        row.addWidget(assign)
        row.addWidget(bulk)
        row.addStretch(1)
        row.addWidget(refresh)
        root.addLayout(row)

    def on_project_changed(self, project):
        self._refresh()

    def _refresh(self):
        proj = self.state.project
        self.table.setRowCount(0)
        if not proj:
            return
        n = proj.config.num_cameras
        self.table.setRowCount(n)
        for i in range(1, n + 1):
            vid = proj.video_file(i)
            self.table.setItem(i - 1, 0, QTableWidgetItem(proj.cam_name(i)))
            self.table.setItem(i - 1, 1, QTableWidgetItem(vid.name if vid.exists() else "—"))
            self.table.setItem(i - 1, 2, QTableWidgetItem("ready" if vid.exists() else "missing"))

    def _assign(self):
        proj = self.state.project
        if not proj:
            return
        row = self.table.currentRow()
        if row < 0:
            return
        cam = row + 1
        src, _ = QFileDialog.getOpenFileName(
            self, f"Video for {proj.cam_name(cam)}", "",
            "Videos (*.mp4 *.avi *.mov *.mkv)")
        if not src:
            return
        dst = proj.video_file(cam, ext=Path(src).suffix.lstrip("."))
        # normalize everyone to camNN.<ext>; keep original extension
        proj.videos_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, dst)
        self._refresh()

    def _assign_folder(self):
        proj = self.state.project
        if not proj:
            return
        folder = QFileDialog.getExistingDirectory(
            self, "Folder with the trial videos (cam01…camNN)")
        if not folder:
            return
        n = proj.config.num_cameras
        # map camera index -> source file, from filenames in the folder
        found: dict[int, Path] = {}
        for p in sorted(Path(folder).iterdir()):
            if p.suffix not in _VIDEO_EXTS:
                continue
            k = _cam_number(p.name)
            if k is not None and 1 <= k <= n and k not in found:
                found[k] = p
        if not found:
            QMessageBox.warning(
                self, "Assign all",
                "No files named cam01…camNN (or 01…NN) found in that folder.")
            return
        proj.videos_dir.mkdir(parents=True, exist_ok=True)
        copied = 0
        for cam, src in sorted(found.items()):
            dst = proj.video_file(cam, ext=src.suffix.lstrip("."))
            shutil.copy(src, dst)
            copied += 1
        self._refresh()
        missing = [f"cam{i:02d}" for i in range(1, n + 1) if i not in found]
        msg = f"Assigned {copied}/{n} cameras."
        if missing:
            msg += " Not matched: " + ", ".join(missing)
        QMessageBox.information(self, "Assign all", msg)
