"""GUI application entry point."""
from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from .main_window import MainWindow


def main(argv=None) -> int:
    app = QApplication(argv if argv is not None else sys.argv)
    app.setApplicationName("PoseAssess")
    win = MainWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
