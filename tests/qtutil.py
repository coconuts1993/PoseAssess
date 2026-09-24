"""Qt helpers for GUI tests: process events for a while / until a condition holds."""

import time


def pump(seconds: float = 0.05) -> None:
    """Process Qt events for ``seconds``."""
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    end = time.perf_counter() + seconds
    while True:
        app.processEvents()
        if time.perf_counter() >= end:
            return
        time.sleep(0.005)


def pump_until(cond, timeout: float = 3.0) -> bool:
    """Process Qt events until ``cond()`` is true (returns True) or ``timeout`` s pass (False)."""
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    end = time.perf_counter() + timeout
    while time.perf_counter() < end:
        app.processEvents()
        if cond():
            return True
        time.sleep(0.005)
    app.processEvents()
    return bool(cond())
