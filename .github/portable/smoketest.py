"""Headless smoke test of the portable bundle (run with the bundle's venv python by CI).

Checks that the heavy dependencies import (Pose2Sim, OpenSim, rtmlib, OpenCV, hidapi, ...),
that the bundled rtmlib models are present, and that the GUI opens every page offscreen.
Exits with 0 on success.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
BUNDLE = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path.cwd()


def main() -> int:
    import cv2
    import hid  # noqa: F401  Wii Balance Board
    import numpy
    import onnxruntime
    import opensim
    import Pose2Sim  # noqa: F401
    import pyqtgraph
    import rtmlib  # noqa: F401

    print("Python", sys.version.split()[0], "| numpy", numpy.__version__, "| OpenCV",
          cv2.__version__, "| onnxruntime", onnxruntime.__version__, "| OpenSim",
          getattr(opensim, "__version__", "?"), "| pyqtgraph", pyqtgraph.__version__)

    ckpt = BUNDLE / "models" / "rtmlib" / "hub" / "checkpoints"
    onnx = sorted(p.name for p in ckpt.glob("*.onnx"))
    print("bundled models:", onnx)
    assert len(onnx) >= 4, f"rtmlib models missing in {ckpt}"

    from PySide6.QtWidgets import QApplication

    from poseassess.gui.main_window import MainWindow

    app = QApplication([])
    w = MainWindow()
    w.show()
    for i, page in enumerate(w.pages):
        w.nav.setCurrentRow(i)
        end = time.perf_counter() + 0.3
        while time.perf_counter() < end:
            app.processEvents()
            time.sleep(0.01)
        print("page OK:", page.nav_title)
    w.close()
    app.processEvents()
    print("PoseAssess portable smoke test OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
