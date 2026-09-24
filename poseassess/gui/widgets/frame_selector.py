"""Interactive video frame selector.

Scrub through a video, optionally overlay live checkerboard detection to judge
whether a frame is usable, and capture selected frames to disk. Used by the
Calibration page for both intrinsic (many frames) and extrinsic (one reference
frame) selection.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QCheckBox, QHBoxLayout, QLabel, QPushButton, QSizePolicy, QSlider,
    QSpinBox, QVBoxLayout, QWidget,
)


def _bgr_to_qpixmap(img: np.ndarray) -> QPixmap:
    """Convert an OpenCV BGR frame to a QPixmap (RGB)."""
    h, w = img.shape[:2]
    rgb = img[:, :, ::-1].copy()  # BGR -> RGB, contiguous
    qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)


class FrameSelector(QWidget):
    """Video scrubber with capture. Emits `frame_captured(index, path)`."""

    frame_captured = Signal(int, str)

    def __init__(self):
        super().__init__()
        self._cap = None
        self._video_path: Optional[Path] = None
        self._n_frames = 0
        self._cur_index = 0
        self._cur_frame: Optional[np.ndarray] = None
        # checkerboard overlay params (rows, cols) inner corners
        self._board: Optional[tuple[int, int]] = None

        root = QVBoxLayout(self)

        # ---- image display ----
        self.view = QLabel("Load a video to begin.")
        self.view.setAlignment(Qt.AlignCenter)
        self.view.setMinimumHeight(360)
        self.view.setStyleSheet("background:#111; color:#888;")
        self.view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        root.addWidget(self.view, 1)

        # ---- scrub controls ----
        scrub = QHBoxLayout()
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setEnabled(False)
        self.slider.valueChanged.connect(self._on_slider)
        self.frame_spin = QSpinBox()
        self.frame_spin.setEnabled(False)
        self.frame_spin.valueChanged.connect(self._on_spin)
        self.total_lbl = QLabel("/ 0")
        scrub.addWidget(self.slider, 1)
        scrub.addWidget(self.frame_spin)
        scrub.addWidget(self.total_lbl)
        root.addLayout(scrub)

        # ---- jump + capture buttons ----
        btns = QHBoxLayout()
        for text, delta in [("⏮ -30", -30), ("◀ -1", -1), ("+1 ▶", 1), ("+30 ⏭", 30)]:
            b = QPushButton(text)
            b.clicked.connect(lambda _=False, d=delta: self._jump(d))
            btns.addWidget(b)
        self.detect_cb = QCheckBox("Detect board")
        self.detect_cb.toggled.connect(lambda _: self._show_current())
        btns.addWidget(self.detect_cb)
        btns.addStretch(1)
        self.detect_status = QLabel("")
        btns.addWidget(self.detect_status)
        self.capture_btn = QPushButton("📸 Capture frame")
        self.capture_btn.setEnabled(False)
        self.capture_btn.clicked.connect(self._capture)
        btns.addWidget(self.capture_btn)
        root.addLayout(btns)

        # where captured frames are written; set by the page
        self._out_dir: Optional[Path] = None

    # ---- public API ------------------------------------------------------- #
    def set_board(self, corners_nb: Optional[Sequence[int]]) -> None:
        """Set inner-corner grid [rows, cols] for the detection overlay."""
        if corners_nb:
            self._board = (int(corners_nb[0]), int(corners_nb[1]))
        else:
            self._board = None

    def set_output_dir(self, out_dir: Path) -> None:
        self._out_dir = Path(out_dir)

    def load_video(self, path: Path) -> bool:
        import cv2

        self.close_video()
        path = Path(path)
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            self.view.setText(f"Cannot open:\n{path}")
            return False
        self._cap = cap
        self._video_path = path
        self._n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._cur_index = 0

        self.slider.setEnabled(True)
        self.slider.setRange(0, max(0, self._n_frames - 1))
        self.frame_spin.setEnabled(True)
        self.frame_spin.setRange(0, max(0, self._n_frames - 1))
        self.total_lbl.setText(f"/ {self._n_frames - 1}")
        self.capture_btn.setEnabled(True)
        self._seek(0)
        return True

    def current_video(self) -> Optional[Path]:
        """Path of the video currently loaded, or None."""
        return self._video_path

    def close_video(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._cur_frame = None

    # ---- internals -------------------------------------------------------- #
    def _seek(self, index: int) -> None:
        import cv2

        if self._cap is None:
            return
        index = max(0, min(index, self._n_frames - 1))
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = self._cap.read()
        if not ok:
            return
        self._cur_index = index
        self._cur_frame = frame
        # keep the two navigators in sync without feedback loops
        for w in (self.slider, self.frame_spin):
            w.blockSignals(True)
            w.setValue(index)
            w.blockSignals(False)
        self._show_current()

    def _show_current(self) -> None:
        if self._cur_frame is None:
            return
        disp = self._cur_frame
        if self.detect_cb.isChecked() and self._board is not None:
            disp = self._overlay_board(self._cur_frame.copy())
        pix = _bgr_to_qpixmap(disp)
        self.view.setPixmap(pix.scaled(
            self.view.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def _overlay_board(self, frame: np.ndarray) -> np.ndarray:
        import cv2

        rows, cols = self._board
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(
            gray, (cols, rows),
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
        if found:
            cv2.drawChessboardCorners(frame, (cols, rows), corners, found)
            self.detect_status.setText("✅ board detected")
            self.detect_status.setStyleSheet("color:#3ad35a;")
        else:
            self.detect_status.setText("❌ not detected")
            self.detect_status.setStyleSheet("color:#e06c6c;")
        return frame

    def _on_slider(self, v: int) -> None:
        self._seek(v)

    def _on_spin(self, v: int) -> None:
        self._seek(v)

    def _jump(self, delta: int) -> None:
        self._seek(self._cur_index + delta)

    def _capture(self) -> None:
        import cv2

        if self._cur_frame is None or self._out_dir is None:
            return
        self._out_dir.mkdir(parents=True, exist_ok=True)
        out = self._out_dir / f"{self._cur_index:06d}.jpg"
        cv2.imwrite(str(out), self._cur_frame)
        self.frame_captured.emit(self._cur_index, str(out))

    def resizeEvent(self, ev):  # rescale the shown frame to the new size
        super().resizeEvent(ev)
        self._show_current()
