"""App-level smoke tests: all pages (old and new) build, the old pages keep their titles and
order, the app never imports hidapi at startup, and closing calls the shutdown hooks.
Owner: GUI-WII (the owner of the shared GUI shell files)."""

import sys

from tests.qtutil import pump

EXPECTED = ["1. Project", "2. Calibration", "2b. Wii Board", "3. Videos", "3b. Capture",
            "4. Run", "5. 3D View", "6. Results", "7. Benchmark"]


def test_pages_and_titles(qapp):
    from poseassess.gui.main_window import MainWindow

    w = MainWindow()
    try:
        assert [p.nav_title for p in w.pages] == EXPECTED
        assert [w.nav.item(i).text() for i in range(w.nav.count())] == EXPECTED
        pump(0.05)
        assert "hid" not in sys.modules
    finally:
        w.close()


def test_open_project_and_visit_pages(qapp, demo_trial):
    from poseassess.gui.main_window import MainWindow

    w = MainWindow()
    try:
        w.state.open_project(demo_trial["project"].root)
        for i in range(len(w.pages)):
            w.nav.setCurrentRow(i)
            pump(0.02)
        w.state.notify_balance_changed("trial")
        pump(0.02)
    finally:
        w.close()


def test_close_calls_page_hooks(qapp, monkeypatch):
    from poseassess.gui.main_window import MainWindow

    w = MainWindow()
    calls = []
    page = w.pages[2]
    monkeypatch.setattr(page, "can_close", lambda: calls.append("can_close") or False)
    w.close()
    assert calls == ["can_close"] and w.isVisible() is False  # close refused (never shown)
    monkeypatch.setattr(page, "can_close", lambda: True)
    monkeypatch.setattr(page, "shutdown", lambda: calls.append("shutdown"))
    w.close()
    assert calls[-1] == "shutdown"
