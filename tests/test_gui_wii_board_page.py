"""GUI-WII: 2b. Wii Board page (device tab without project, live COP with simulator, board tab:
load demo_trial frames, BoardPointPicker labels/limit/undo/swap, save clicks, compute board, QC
grid, corner check TL=ok / BR=swap+recompute / TR=redo with a ManualBoard ForceSource)."""

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from tests.qtutil import pump, pump_until
from tests.wii_fakes import PATH, FakeBus


def manual_board():
    """Force source whose four sensor loads (TR, BR, TL, BL) the test sets (PoseBoard
    test_gui_fixes.py pattern)."""
    from poseassess.wii.device import ForceSource

    class _Manual(ForceSource):
        name = "manual"

        def __init__(self):
            super().__init__()
            self.kg = np.zeros(4)

        @property
        def device_key(self):
            return "manual"

        def _run(self):
            self._set_state("connected")
            while not self._stop.is_set():
                self._emit_kg(time.perf_counter(), self.kg.copy())
                time.sleep(0.01)

    return _Manual()


@pytest.fixture(autouse=True)
def _no_modal_dialogs(monkeypatch):
    """Record message boxes instead of opening modal dialogs."""
    from PySide6.QtWidgets import QMessageBox

    calls = []
    for name in ("warning", "information", "critical"):
        monkeypatch.setattr(QMessageBox, name,
                            staticmethod(lambda *a, _n=name, **k: calls.append((_n, a[1:3]))))
    monkeypatch.setattr(QMessageBox, "question",
                        staticmethod(lambda *a, **k: QMessageBox.Yes))
    had = "hid" in sys.modules
    yield calls
    if not had:
        sys.modules.pop("hid", None)


@pytest.fixture
def no_boards(monkeypatch):
    from poseassess.wii.device import BalanceBoardHID

    monkeypatch.setattr(BalanceBoardHID, "list_devices", staticmethod(lambda: []))


@pytest.fixture
def env(qapp):
    """(state, page) with the page shown; everything stopped afterwards."""
    from poseassess.gui.pages.wii_board_page import WiiBoardPage
    from poseassess.gui.state import AppState

    state = AppState()
    state.wii.poll_interval_s = 0.05
    page = WiiBoardPage(state)
    page.messages = []
    page._warn = lambda title, text: page.messages.append(("warn", text))
    page._info = lambda title, text: page.messages.append(("info", text))
    page.resize(1300, 800)
    page.show()
    pump(0.05)
    yield state, page
    page.shutdown()
    page.close()
    state.shutdown()


def open_demo(state, page, demo):
    state.open_project(demo["project"].root)
    page.tabs.setCurrentWidget(page.board_tab)
    pump(0.1)


def emitted(state):
    got = []
    state.balance_changed.connect(got.append)
    return got


def read_json(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


# ------------------------------------------------------------------ device tab
def test_device_tab_without_project_and_live_cop(env, no_boards):
    state, page = env
    assert page.isEnabled() and not page.board_tab.isEnabled()
    assert page.device_tab.isEnabled()
    assert "Not connected" in page.status_lbl.text()
    assert not page.btn_tare.isEnabled() and not page.btn_disconnect.isEnabled()

    page.btn_sim.click()
    c = state.wii
    assert c.is_simulator and not page.chk_auto.isChecked()
    assert pump_until(lambda: c.connected and len(c.recent(1.0)) > 30)
    assert pump_until(lambda: len(page.cop_view.cop_trail) > 20)
    assert page.cop_view.total_kg > 50
    assert "Simulator: connected" in page.status_lbl.text()
    assert "COP x=" in page.status_lbl.text()
    assert page.btn_tare.isEnabled() and page.btn_disconnect.isEnabled()

    page.btn_tare.click()
    assert c.tare_values.sum() > 50
    c.lock_tare("recording")
    pump(0.05)
    assert not page.btn_tare.isEnabled() and "recording" in page.btn_tare.toolTip()
    c.lock_tare(None)
    pump(0.05)
    assert page.btn_tare.isEnabled()

    page.min_kg.setValue(20.0)
    assert c.source.min_total_kg == 20.0

    page.hide()  # the live view only runs while the page is visible
    assert not page._live.isActive()
    page.show()
    assert page._live.isActive()
    page.btn_disconnect.click()
    assert c.source is None and not page.chk_auto.isChecked()
    pump(0.1)
    assert page.cop_view.total_kg == 0.0


def test_auto_connect_toggle_scan_and_connect(env, monkeypatch):
    from poseassess.wii.device import BalanceBoardHID, WiiAutoConnect

    state, page = env
    c = state.wii
    bus = FakeBus(monkeypatch)
    t = time.perf_counter()
    page.chk_auto.setChecked(True)
    assert time.perf_counter() - t < 0.5
    assert isinstance(c.source, WiiAutoConnect)
    assert pump_until(lambda: "searching" in page.status_lbl.text())
    bus.present = True
    assert pump_until(lambda: c.connected)
    assert pump_until(lambda: "Auto-connect: connected" in page.status_lbl.text())
    assert page.chk_auto.isChecked()

    t = time.perf_counter()
    page.chk_auto.setChecked(False)
    assert time.perf_counter() - t < 1.0 and c.source is None

    page._scan()
    assert page.dev_combo.count() == 1 and "1 board" in page.scan_lbl.text()
    page.btn_connect.click()  # Auto-connect off: one-shot connection
    assert type(c.source) is BalanceBoardHID
    assert pump_until(lambda: c.connected)
    page.chk_auto.setChecked(True)  # replaces it with plug and play
    assert isinstance(c.source, WiiAutoConnect)
    page.btn_connect.click()  # Auto-connect on: plug and play limited to the selected board
    assert isinstance(c.source, WiiAutoConnect) and c.source.target_path == PATH
    page.btn_disconnect.click()
    assert c.source is None and not page.chk_auto.isChecked()
    bus.present = False
    page._scan()
    assert page.dev_combo.count() == 0 and "No Balance Board" in page.scan_lbl.text()


def test_hidapi_missing_shows_the_hint(env, monkeypatch):
    from poseassess.wii import device

    def no_hid():
        raise device.HidapiUnavailable(device.HIDAPI_MISSING)

    monkeypatch.setattr(device, "_import_hid", no_hid)
    state, page = env
    page._scan()
    assert "hidapi" in page.scan_lbl.text()
    page.chk_auto.setChecked(True)
    assert pump_until(lambda: not state.wii.source.running)
    assert pump_until(lambda: "hidapi is not available" in page.hint_lbl.text())
    assert "pip install hidapi" in page.status_lbl.text()
    assert page.btn_sim.isEnabled()
    page.btn_sim.click()
    assert pump_until(lambda: state.wii.connected)
    assert "hidapi is not available" not in page.hint_lbl.text()


# ------------------------------------------------------------------ picker
def test_board_picker_labels_limit_undo_swap_and_drag(qapp):
    from PySide6.QtCore import QPoint, QPointF, Qt
    from PySide6.QtTest import QTest

    from poseassess.gui.widgets.board_picker import LABELS, BoardPointPicker

    p = BoardPointPicker()
    p.resize(640, 400)
    p.show()
    p.load_bgr(np.full((360, 640, 3), 90, np.uint8))
    pump(0.05)
    p.fit()
    counts = []
    p.points_changed.connect(counts.append)
    vp = p.viewport()
    targets = [(100, 100), (400, 100), (400, 300), (100, 300), (250, 200), (300, 250)]
    for x, y in targets:
        pos = p.mapFromScene(QPointF(x, y))
        QTest.mouseClick(vp, Qt.LeftButton, Qt.NoModifier, pos)
    pts = p.points()
    assert len(pts) == 5  # the 6th click is ignored
    np.testing.assert_allclose(pts, targets[:5], atol=2.0)
    assert [it.text() for it in p._label_items] == [f"{i + 1}:{n}" for i, n in enumerate(LABELS)]
    assert p.next_label() is None and len(p._outline_items) == 4

    QTest.mouseClick(vp, Qt.RightButton, Qt.NoModifier, QPoint(5, 5))  # right-click: undo last
    assert len(p.points()) == 4 and p.next_label() == "C"
    assert p.swap_front_back() is False  # needs all 5
    p.undo_last()
    assert len(p.points()) == 3 and not p._outline_items
    p.set_points(targets[:5])
    assert p.swap_front_back() is True
    np.testing.assert_allclose(p.points(), [targets[i] for i in (2, 3, 0, 1, 4)])

    # drag a point: moved and points_changed emitted on release
    n = len(counts)
    start = p.mapFromScene(QPointF(*p.points()[0]))
    QTest.mousePress(vp, Qt.LeftButton, Qt.NoModifier, start)
    QTest.mouseMove(vp, start + QPoint(20, 10))
    QTest.mouseRelease(vp, Qt.LeftButton, Qt.NoModifier, start + QPoint(20, 10))
    assert len(counts) > n and counts[-1] == 5
    assert p.points()[0][0] > targets[2][0] + 5
    p.clear_image()
    assert not p.has_image() and p.points() == [] and p.image_size() is None
    p.close()


# ------------------------------------------------------------------ board tab
def test_board_tab_loads_frames_and_saves_clicks(env, demo_trial):
    from poseassess.core.balance.paths import WiiPaths

    state, page = env
    open_demo(state, page, demo_trial)
    proj = demo_trial["project"]
    paths = WiiPaths(proj)
    assert page.board_tab.isEnabled() and page.cam_combo.count() == 3
    assert page.frame_list.count() == 1 and page.frame_list.item(0).text().startswith("board.jpg")
    saved = read_json(paths.board_points_file(1))
    np.testing.assert_allclose(page.picker.points(), saved["points"])
    assert "All 5 points set" in page.click_hint.text() and not page._dirty
    assert "cam01" in page.clicks_status.text() and "matches the calibration" in \
        page.frame_info.text()

    pts = page.picker.points()
    page.picker.set_points([(x + 3.0, y) for x, y in pts])
    assert page._dirty and "unsaved" in page.click_hint.text()
    page.btn_save.click()
    d = read_json(paths.board_points_file(1))
    assert not page._dirty and d["image"] == "board.jpg" and d["image_size"] == [1280, 720]
    assert d["source"] == "synthetic" and d["labels"] == ["TL", "TR", "BR", "BL", "C"]
    np.testing.assert_allclose(d["points"], [(x + 3.0, y) for x, y in pts])

    # unsaved clicks are saved when switching camera, and camera 2 shows its own clicks
    page.picker.undo_last()
    page.cam_combo.setCurrentIndex(1)
    pump(0.05)
    assert len(read_json(paths.board_points_file(1))["points"]) == 4
    np.testing.assert_allclose(page.picker.points(), read_json(paths.board_points_file(2))["points"])
    # ... and when leaving the page
    page.picker.undo_last()
    page.hide()
    assert len(read_json(paths.board_points_file(2))["points"]) == 4


def test_compute_board_qc_and_errors(env, demo_trial, monkeypatch):
    from poseassess.core.balance import board as B
    from poseassess.core.balance.paths import WiiPaths

    state, page = env
    paths = WiiPaths(demo_trial["project"])
    paths.board_json.unlink()
    open_demo(state, page, demo_trial)
    assert page._board is None and "No board position yet" in page.result_txt.toPlainText()
    assert not page.btn_corner.isEnabled()
    got = emitted(state)

    page.btn_compute.click()
    assert paths.board_json.is_file() and got == ["board"]
    assert page._board is not None and page._board.cameras_used == ["cam01", "cam02", "cam03"]
    text = page.result_txt.toPlainText()
    assert "multi-camera triangulation" in text and "cam02:" in text and "px" in text
    np.testing.assert_allclose(page._board.pose.board_to_world.t, demo_trial["board_t"],
                               atol=1e-3)
    assert page.view_tabs.currentWidget() is page.qc_tab
    pump(0.05)
    assert page._qc_grid.shape[2] == 3 and page._qc_grid.size > 0
    grid_zoom = page._qc_grid.shape
    page.chk_qc_zoom.setChecked(False)
    assert page._qc_grid.shape == grid_zoom  # same layout, whole frames

    def stub(*a, **k):
        raise NotImplementedError("CORE-BOARD")

    monkeypatch.setattr(B, "compute_board", stub)
    page.btn_compute.click()
    assert page.messages[-1][0] == "warn" and "not available yet" in page.messages[-1][1]

    def bad(*a, **k):
        raise ValueError("No camera has all 5 board points")

    monkeypatch.setattr(B, "compute_board", bad)
    page.btn_compute.click()
    assert "No camera has all 5" in page.messages[-1][1]
    assert got == ["board"]  # failures do not notify


def test_single_camera_pnp_and_resolution_mismatch(env, demo_trial):
    from poseassess.core.balance.paths import WiiPaths

    state, page = env
    paths = WiiPaths(demo_trial["project"])
    for i in (2, 3):
        paths.board_points_file(i).unlink()
    open_demo(state, page, demo_trial)
    page.btn_compute.click()
    assert page._board.pose.method == "pnp" and "single-camera PnP" in \
        page.result_txt.toPlainText()
    # a frame at another resolution is flagged
    small = paths.board_cam_dir(2) / "small.jpg"
    small.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(small), np.zeros((240, 320, 3), np.uint8))
    page.cam_combo.setCurrentIndex(1)
    pump(0.05)
    assert page._image.name == "board.jpg"  # no clicks saved: the first frame
    assert "matches the calibration" in page.frame_info.text()
    page.frame_list.setCurrentRow(1)
    pump(0.05)
    assert page._image == small
    assert "320x240" in page.frame_info.text() and "cannot be used" in page.frame_info.text()


def test_stale_banner_and_recompute(env, demo_trial):
    state, page = env
    open_demo(state, page, demo_trial)
    assert not page.stale_lbl.isVisibleTo(page)
    calib = demo_trial["project"].calib_toml
    calib.write_text(calib.read_text(encoding="utf-8") + "\n# recalibrated\n", encoding="utf-8")
    page.hide()
    page.show()  # showing the page again re-reads board.json + Calib.toml
    pump(0.05)
    assert page.stale_lbl.isVisibleTo(page) and page.btn_recompute.isVisibleTo(page)
    assert "OUTDATED" in page.stale_lbl.text()
    page.btn_recompute.click()
    assert page._board.stale is None and not page.stale_lbl.isVisibleTo(page)


def test_corner_check_ok_swapped_redo(env, demo_trial, monkeypatch):
    import poseassess.gui.pages.wii_board_page as mod
    from poseassess.core.balance.board import load_all_clicks, load_board

    state, page = env
    open_demo(state, page, demo_trial)
    proj = demo_trial["project"]
    got = emitted(state)
    assert not page.btn_corner.isEnabled()  # board registered but not connected
    board = manual_board()
    state.wii.use_source(board)
    assert pump_until(lambda: state.wii.connected and len(state.wii.recent(0.5)) > 20)
    assert pump_until(lambda: page.btn_corner.isEnabled())
    R0 = load_board(proj).pose.board_to_world.R.copy()
    clicks0 = {k: np.asarray(v.points) for k, v in load_all_clicks(proj).items()}

    def press(sensor_index, expect):
        board.kg[:] = 0.0
        pump(0.6)
        assert page.start_corner_check()
        # The target is the corner the CLICKS call TL, shown in the camera images: going by
        # the power button would always press the TL sensor and hide a front/back swap.
        txt = page.corner_lbl.text()
        assert "marked “PRESS”" in txt and "corner you clicked first (TL)" in txt
        assert "OPPOSITE the power button" not in txt and "not</b> go by the power" in txt
        assert page.view_tabs.currentWidget() is page.qc_tab
        marked = page._qc_grid.copy()  # with the "PRESS" rings
        board.kg[sensor_index] = 15.0
        assert pump_until(lambda: expect in page.corner_lbl.text(), 3.0), page.corner_lbl.text()
        assert page._corner is None and not page._corner_timer.isActive()
        assert marked.shape != page._qc_grid.shape or np.any(marked != page._qc_grid)

    assert "clicked as TL" in page.btn_corner.toolTip()
    assert "power button" in page.btn_corner.toolTip()  # ...namely: not from the power button

    press(2, "OK")  # TL responded
    assert load_board(proj).corner_check["result"] == "ok"
    assert got == ["board"]
    press(1, "Front and back were swapped; fixed")  # BR responded
    reg = load_board(proj)
    assert reg.corner_check["result"] == "swapped" and reg.corner_check["pressed"] == "BR"
    np.testing.assert_allclose(reg.pose.board_to_world.R[:, :2], -R0[:, :2], atol=1e-6)
    np.testing.assert_allclose(reg.pose.board_to_world.R[:, 2], R0[:, 2], atol=1e-6)
    for k, v in load_all_clicks(proj).items():
        np.testing.assert_allclose(v.points, clicks0[k][[2, 3, 0, 1, 4]])
    np.testing.assert_allclose(page.picker.points(), clicks0["cam01"][[2, 3, 0, 1, 4]])
    assert "front/back swapped" in page.result_txt.toPlainText()
    press(2, "OK")
    press(0, "Redo the clicks")  # TR: mirrored / turned 90°
    cc = load_board(proj).corner_check
    assert cc["result"] == "redo" and cc["pressed"] == "TR"
    assert got == ["board"] * 4

    # timeout and disconnect
    monkeypatch.setattr(mod, "CORNER_CHECK_TIMEOUT_S", 0.4)
    board.kg[:] = 0.0
    pump(0.6)
    assert page.start_corner_check()
    assert pump_until(lambda: "no clear press" in page.corner_lbl.text(), 2.0)
    assert page.start_corner_check()
    state.wii.disconnect()
    assert pump_until(lambda: "cancelled" in page.corner_lbl.text(), 2.0)
    assert not page.start_corner_check()  # needs the board
    assert "connect it first" in page.messages[-1][1]


def test_geometry_apply_recomputes_and_keeps_corner_check(env, demo_trial):
    from poseassess.core.balance.board import load_board, set_corner_check

    state, page = env
    proj = demo_trial["project"]
    set_corner_check(proj, "ok", "TL")
    open_demo(state, page, demo_trial)
    got = emitted(state)
    assert page.geo_spins["height_mm"].value() == 53.0
    assert not page.geo_spins["length_mm"].isVisibleTo(page)
    page.chk_geo_adv.setChecked(True)
    assert page.geo_spins["length_mm"].isVisibleTo(page)
    page.geo_spins["height_mm"].setValue(60.0)
    page.geo_spins["sensor_dx_mm"].setValue(430.0)
    page.btn_geo.click()
    reg = load_board(proj)
    assert reg.geometry.height_mm == 60.0 and reg.geometry.sensor_dx_mm == 430.0
    assert reg.corner_check["result"] == "ok" and got == ["board"]
    assert state.wii._sensor_m == pytest.approx((0.430, 0.238))
    assert page.cop_view.sensor_mm == (430.0, 238.0)


def test_frame_sources_calibration_image_and_video(env, demo_trial, monkeypatch):
    from PySide6.QtWidgets import QFileDialog

    from poseassess.core.balance.paths import WiiPaths
    from tests.synth import write_test_video

    state, page = env
    proj = demo_trial["project"]
    paths = WiiPaths(proj)
    open_demo(state, page, demo_trial)

    # "Use calibration frame": no frame -> information; then a copy of the extrinsic frame
    page.btn_calib_frame.click()
    assert page.messages[-1][0] == "info" and "No calibration" in page.messages[-1][1]
    ext = proj.extrinsic_cam_dir(1)
    ext.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(ext / "ext0.jpg"), np.full((720, 1280, 3), 80, np.uint8))
    page.btn_calib_frame.click()
    dst = paths.board_cam_dir(1) / "calib_ext0.jpg"
    assert dst.is_file() and page._image == dst and page.picker.points() == []
    page.picker.set_points([(100, 100), (300, 100), (300, 200), (100, 200), (200, 150)])
    page.btn_save.click()
    d = read_json(paths.board_points_file(1))
    assert d["image"] == "calib_ext0.jpg" and d["source"] == "calibration/extrinsics/cam01/ext0.jpg"

    # "Load image…": copied into wii/board/cam01 (unique name)
    src = demo_trial["project"].root / "outside.jpg"
    cv2.imwrite(str(src), np.full((720, 1280, 3), 60, np.uint8))
    monkeypatch.setattr(QFileDialog, "getOpenFileName", staticmethod(lambda *a, **k: (str(src), "")))
    page.btn_load_img.click()
    assert page._image == paths.board_cam_dir(1) / "outside.jpg"
    page.btn_load_img.click()
    assert page._image == paths.board_cam_dir(1) / "outside_2.jpg"
    assert page.frame_list.count() == 4

    # "Grab frame from video…" on videos/cam01.mp4
    write_test_video(proj.videos_dir / "cam01.mp4", n_frames=20, size=(320, 240))
    page.btn_grab.click()
    dlg = page._grab_dlg
    assert dlg is not None and dlg.isVisible() and dlg.video.name == "cam01.mp4"
    dlg.selector._seek(7)
    dlg.selector._capture()
    pump(0.05)
    grabbed = paths.board_cam_dir(1) / "000007.jpg"
    assert grabbed.is_file() and page._image == grabbed and not dlg.isVisible()
    assert "320x240" in page.frame_info.text() and "cannot be used" in page.frame_info.text()
    page.picker.set_points([(10, 10), (30, 10), (30, 20), (10, 20), (20, 15)])
    page.btn_save.click()
    d = read_json(paths.board_points_file(1))
    assert d["image"] == "000007.jpg" and d["source"] == "videos/cam01.mp4#frame=7"
    assert d["image_size"] == [320, 240]
    # another frame shows no clicks; back on the clicked frame its saved points come back
    names = [page.frame_list.item(i).data(0x0100) for i in range(page.frame_list.count())]
    page.frame_list.setCurrentRow(names.index(str(dst)))
    pump(0.05)
    assert page._image == dst and page.picker.points() == []
    page.frame_list.setCurrentRow(names.index(str(grabbed)))
    pump(0.05)
    assert page._image == grabbed and len(page.picker.points()) == 5 and not page._dirty


def test_body_mass_percent_and_project_switch(env, demo_trial, project):
    from poseassess.core.balance.trial import load_trial, save_trial

    state, page = env
    open_demo(state, page, demo_trial)
    info = load_trial(demo_trial["project"])
    info.body_mass_kg = 70.0
    save_trial(demo_trial["project"], info)
    state.notify_balance_changed("trial")
    page.tabs.setCurrentWidget(page.device_tab)
    state.wii.start_simulator()
    assert pump_until(lambda: "% of body mass" in page.cop_view.text)

    page.tabs.setCurrentWidget(page.board_tab)
    page.picker.undo_last()  # unsaved -> saved into the demo project on switch
    state.set_project(project)  # an empty project: no calibration, no frames
    from poseassess.core.balance.board import load_clicks

    assert len(load_clicks(demo_trial["project"], 1).points) == 4
    assert page.cam_combo.count() == 3 and page.frame_list.count() == 0
    assert not page.picker.has_image() and page._board is None
    page.btn_compute.click()
    assert "calibrat" in page.messages[-1][1]
    state.set_project(None)
    assert not page.board_tab.isEnabled() and page.device_tab.isEnabled()


def test_page_in_main_window_close_hooks(qapp, demo_trial):
    from poseassess.gui.main_window import MainWindow

    w = MainWindow()
    try:
        w.state.open_project(demo_trial["project"].root)
        w.show()
        w.nav.setCurrentRow(2)
        page = w.pages[2]
        page.tabs.setCurrentWidget(page.board_tab)
        pump(0.1)
        project_msg = w.statusBar().currentMessage()
        assert project_msg.startswith("demo")
        page._flash("Board position computed", 50)  # temporary, then the project line again
        assert w.statusBar().currentMessage() == "Board position computed"
        assert pump_until(lambda: w.statusBar().currentMessage() == project_msg, 2.0)
        page.picker.undo_last()
        assert page._dirty
    finally:
        w.close()
    from poseassess.core.balance.board import load_clicks

    assert len(load_clicks(demo_trial["project"], 1).points) == 4  # saved on close
    from PySide6.QtWidgets import QApplication

    # Qt's "offscreen" platform on Windows has no real fonts (falls back to a much wider one), so
    # widths there are meaningless: 2128 px vs 1068 px with the native "windows" platform on the
    # same CI machine. CI checks the native width with .github/portable/gui_sizes.py instead.
    if not (sys.platform.startswith("win") and QApplication.platformName() == "offscreen"):
        assert w.minimumSizeHint().width() < 1400


def test_exploring_another_frame_keeps_the_complete_clicks(env, demo_trial):
    from poseassess.core.balance.board import load_clicks

    state, page = env
    proj = demo_trial["project"]
    ext = proj.extrinsic_cam_dir(1)
    ext.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(ext / "ext0.jpg"), np.full((720, 1280, 3), 80, np.uint8))
    open_demo(state, page, demo_trial)
    before = load_clicks(proj, 1)
    page.btn_calib_frame.click()
    page.picker.set_points([(10, 10), (20, 10)])  # 2 exploratory clicks on another frame
    assert page._dirty
    rows = [page.frame_list.item(i).text() for i in range(page.frame_list.count())]
    page.frame_list.setCurrentRow([i for i, t in enumerate(rows) if t.startswith("board.jpg")][0])
    pump(0.05)
    after = load_clicks(proj, 1)
    assert after.image == "board.jpg" and after.points == before.points
    assert len(page.picker.points()) == 5 and not page._dirty


def test_the_source_cannot_change_during_a_take(env, project, no_boards, monkeypatch):
    """While 3b. Capture records a take, 2b cannot swap in the simulator (or disconnect, switch
    auto-connect, change the COP threshold): the take's wii.csv holds the real board only.
    Connect asks first (for a board of the take that dropped out)."""
    from PySide6.QtWidgets import QMessageBox

    from poseassess.gui.pages.capture_page import CapturePage
    from poseassess.wii import io as wio

    state, page = env
    cap = CapturePage(state)
    try:
        state.set_project(project)
        board = manual_board()
        board.kg[:] = 12.5  # 50 kg
        state.wii.use_source(board)
        assert pump_until(lambda: state.wii.connected and len(state.wii.recent(0.3)) > 10)
        pump(0.1)
        assert page.btn_sim.isEnabled() and page.min_kg.isEnabled()
        cap.chk_record_cams.setChecked(False)
        cap._start_recording()  # Wii-only take
        s = cap._session
        assert s.recording and state.wii.tare_locked == "recording"
        pump(0.1)
        for w in (page.btn_sim, page.chk_auto, page.btn_disconnect, page.min_kg, page.btn_tare):
            assert not w.isEnabled()
        assert "recording" in page.btn_sim.toolTip() and page.btn_connect.isEnabled()
        for call in (state.wii.start_simulator, state.wii.start_auto, state.wii.disconnect,
                     lambda: state.wii.set_min_load(20.0),
                     lambda: state.wii.use_source(manual_board()),
                     lambda: state.wii.connect_device(PATH)):
            with pytest.raises(RuntimeError, match="recording"):
                call()
        page._use_simulator()  # even when called directly: refused with a message
        assert "recording" in page.messages[-1][1]
        monkeypatch.setattr(QMessageBox, "question",
                            staticmethod(lambda *a, **k: QMessageBox.No))
        page.btn_connect.click()  # "record another board into the take?" -> No
        assert state.wii.source is board
        pump(0.3)
        folder = s.folder
        cap._stop_recording()  # "use as the trial's Wii recording?" -> No
        tot = wio.read_wii_csv(folder)["total_kg"]
        assert len(tot) > 20 and np.allclose(tot[np.isfinite(tot)], 50.0)
        hist = wio.read_session_json(folder)["force_source_history"]
        assert not any("Simulated" in json.dumps(h) for h in hist)
        pump(0.1)
        assert state.wii.tare_locked is None
        assert page.btn_sim.isEnabled() and page.min_kg.isEnabled() and page.chk_auto.isEnabled()
        assert "Synthetic data" in page.btn_sim.toolTip()
        assert "start of every session" in page.btn_tare.toolTip()  # tare kept in memory only
    finally:
        cap.shutdown()
        cap.close()


def test_qc_frames_are_read_without_cv2_imread(env, demo_trial, tmp_path, monkeypatch):
    """On Windows cv2.imread cannot open non-ASCII paths (e.g. Chinese folder names): the check
    view decodes the file bytes instead."""
    import poseassess.gui.pages.wii_board_page as mod

    state, page = env
    open_demo(state, page, demo_trial)
    monkeypatch.setattr(cv2, "imread", lambda *a, **k: None)  # what Windows does
    img, pts = page._qc_image(1)
    assert img is not None and img.shape[2] == 3 and pts is not None
    d = tmp_path / "\u5b9e\u9a8c" / "\u53d7\u8bd5\u800501"  # Chinese names
    d.mkdir(parents=True)
    _, buf = cv2.imencode(".png", np.full((20, 30, 3), 90, np.uint8))
    (d / "\u5e27.png").write_bytes(buf.tobytes())
    assert mod._imread(d / "\u5e27.png").shape == (20, 30, 3)
    assert mod._imread(d / "missing.png") is None
    (d / "broken.jpg").write_bytes(b"not an image")
    assert mod._imread(d / "broken.jpg") is None
