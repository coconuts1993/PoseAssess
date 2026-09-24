"""Integration glue between the pages (fixes made while wiring the agents' work together):

* 2b. Wii Board lists frames saved elsewhere while it was hidden (3b. Capture "Board snapshots");
* 3. Videos shows the videos written by the capture export (``balance_changed("trial")``);
* Results > Balance (Wii): the default "whole trial" window covers every .trc frame although the
  time spin boxes round to 0.01 s;
* ``CameraStream.measured_fps`` is not inflated right after a file source starts;
* ``fuse_trial`` warns when the .trc is older than the videos (e.g. a new take was exported to
  videos/ but the pipeline was not run again).
"""

import os
import shutil
import time

import cv2
import numpy as np
import pytest

from tests.qtutil import pump, pump_until

pytestmark = pytest.mark.filterwarnings("ignore:.*AA_UseHighDpiPixmaps.*:DeprecationWarning")


@pytest.fixture
def quiet_dialogs(monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    shown = []
    for kind in ("information", "warning", "critical"):
        monkeypatch.setattr(QMessageBox, kind,
                            staticmethod(lambda *a, _k=kind, **k: shown.append((_k, a[1]))
                                         or QMessageBox.Ok))
    monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.Yes))
    return shown


def test_board_page_lists_frames_added_while_hidden(qapp, demo_trial, quiet_dialogs):
    from poseassess.core.balance.paths import WiiPaths
    from poseassess.gui.pages.wii_board_page import WiiBoardPage
    from poseassess.gui.state import AppState

    proj = demo_trial["project"]
    state = AppState()
    page = WiiBoardPage(state)
    page.resize(1300, 800)
    page.show()
    state.open_project(proj.root)
    page.tabs.setCurrentWidget(page.board_tab)
    pump(0.1)
    assert page.frame_list.count() == 1 and len(page.picker.points()) == 5
    shown_img = page._image
    try:
        page.hide()
        d = WiiPaths(proj).board_cam_dir(1)
        snap = d / "snapshot_20260101_120500.jpg"
        shutil.copy(d / "board.jpg", snap)  # e.g. "📸 Board snapshots" on 3b. Capture
        page.show()
        pump(0.1)
        rows = [page.frame_list.item(i).data(0x0100) for i in range(page.frame_list.count())]
        assert str(snap) in rows
        assert page._image == shown_img and len(page.picker.points()) == 5  # kept
        page.hide()
        snap.unlink()
        page.show()
        pump(0.1)
        assert page.frame_list.count() == 1
    finally:
        page.shutdown()
        page.close()
        state.shutdown()


def test_videos_page_follows_the_capture_export(qapp, demo_trial, quiet_dialogs):
    from poseassess.gui.main_window import MainWindow
    from poseassess.gui.pages.videos_page import VideosPage
    from tests.synth import write_test_video

    proj = demo_trial["project"]
    w = MainWindow()
    try:
        w.state.open_project(proj.root)
        vp = next(p for p in w.pages if isinstance(p, VideosPage))
        assert [vp.table.item(r, 2).text() for r in range(3)] == ["missing"] * 3
        for i in (1, 2, 3):  # what the capture export writes ...
            write_test_video(proj.video_file(i), n_frames=5)
        w.state.notify_balance_changed("trial")  # ... and announces
        assert [vp.table.item(r, 2).text() for r in range(3)] == ["ready"] * 3
        w.state.notify_balance_changed("alignment")  # other values: no refresh needed
    finally:
        w.close()


def test_balance_window_covers_every_trc_frame(qapp, tmp_path, quiet_dialogs):
    from poseassess.gui.pages.results_page import ResultsPage
    from poseassess.gui.state import AppState
    from tests.synth import make_demo_trial

    # 179 frames at 30 fps: the last .trc time 5.9333 s shows as 5.93 s in the spin box
    demo = make_demo_trial(tmp_path / "demo", duration_s=179 / 30, jump_at=3.0, stomp_at=None)
    state = AppState()
    page = ResultsPage(state)
    page.resize(1300, 800)
    page.show()
    try:
        state.open_project(demo["project"].root)
        page.tabs.setCurrentIndex(1)
        panel = page.balance
        assert pump_until(lambda: panel.summary is not None, 8.0)
        assert panel.to_spin.value() == pytest.approx(5.93)
        a, b = panel.window()
        assert a == 0.0 and b == pytest.approx(178 / 30)
        assert panel.summary["frames"] == 179
        panel.set_window(1.0, 2.0)  # a real sub-window is used as typed
        assert panel.window() == (1.0, 2.0) and panel.summary["frames"] == 31
    finally:
        page.shutdown()
        page.close()


def test_file_camera_rate_is_not_inflated_at_start(tmp_path):
    from poseassess.core.capture.camera import CameraStream
    from tests.synth import write_test_video

    src = write_test_video(tmp_path / "src.avi", n_frames=30, fps=30.0)
    cap = cv2.VideoCapture(str(src))
    assert cap.get(cv2.CAP_PROP_FPS) == pytest.approx(30.0)
    cap.release()
    cam = CameraStream("cam01", str(src))
    cam.start()
    try:
        end = time.perf_counter() + 3.0
        while cam.latest() is None or cam.latest().index < 5:
            assert time.perf_counter() < end
            time.sleep(0.01)
        rates = []
        for _ in range(10):
            rates.append(cam.measured_fps)
            time.sleep(0.03)
        assert np.max(rates) < 45.0, rates
    finally:
        cam.stop()


def test_fusion_warns_when_the_trc_is_older_than_the_videos(demo_trial):
    from poseassess.core.balance.fusion import fuse_trial
    from tests.synth import write_test_video

    proj = demo_trial["project"]
    trc = demo_trial["trc"]
    assert not any("older than the videos" in w for w in fuse_trial(proj).warnings)
    t = trc.stat().st_mtime
    for i in (1, 2, 3):
        v = write_test_video(proj.video_file(i), n_frames=3)
        os.utime(v, (t + 60, t + 60))  # videos exported after the pipeline ran
    assert any("older than the videos" in w for w in fuse_trial(proj).warnings)
    os.utime(trc, (t + 120, t + 120))  # pipeline run again
    assert not any("older than the videos" in w for w in fuse_trial(proj).warnings)
