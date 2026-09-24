"""GUI-CAPTURE: 3b. Capture page (camera table from project, preview with file sources, record/stop with the shared WiiController simulator, mark event, export in worker thread, can_close while recording, shutdown stops cameras)."""

import json
import sys

import numpy as np
import pytest

from poseassess.core.balance.paths import WiiPaths
from poseassess.core.balance.trial import load_trial
from poseassess.wii import io as wio
from poseassess.wii.device import SimulatedBoard
from tests.qtutil import pump, pump_until
from tests.synth import write_test_video
from tests.test_capture_session_export import regular, write_take


class Answers:
    """Non-blocking stand-ins for the QMessageBox dialogs; records what was shown."""

    def __init__(self, monkeypatch, question="yes"):
        from PySide6.QtWidgets import QMessageBox

        self.shown: list[tuple[str, str, str]] = []
        self.question = question

        def ask(_parent, title, text, *a, **k):
            self.shown.append(("question", title, text))
            return QMessageBox.Yes if self.question == "yes" else QMessageBox.No

        def note(kind):
            def f(_parent, title, text, *a, **k):
                self.shown.append((kind, title, text))
                return QMessageBox.Ok
            return f

        monkeypatch.setattr(QMessageBox, "question", staticmethod(ask))
        for kind in ("information", "warning", "critical"):
            monkeypatch.setattr(QMessageBox, kind, staticmethod(note(kind)))

    def kinds(self):
        return [k for k, _, _ in self.shown]


@pytest.fixture
def page(qapp, project, tmp_path, monkeypatch):
    from poseassess.gui.pages.capture_page import CapturePage
    from poseassess.gui.state import AppState

    project.config.num_cameras = 2
    project.save()
    state = AppState()
    p = CapturePage(state)
    state.set_project(project)
    p.resize(1200, 800)
    p.show()
    pump(0.05)
    events = []
    state.balance_changed.connect(events.append)
    p.balance_events = events
    p.sources = [write_test_video(tmp_path / f"src{i}.avi", n_frames=60) for i in (1, 2)]
    yield p
    p.shutdown()
    state.shutdown()
    p.close()
    pump(0.02)


def start_simulator(state, monkeypatch):
    """The shared controller's simulator (or, while GUI-WII's controller is a stub, a
    SimulatedBoard handed to it)."""
    try:
        state.wii.start_simulator()
    except NotImplementedError:
        sim = SimulatedBoard()
        sim.start()
        monkeypatch.setattr(state.wii, "_source", sim, raising=False)
        state.wii.source_changed.emit(sim)
        state.wii.connection_event.emit("wii_connected")
    assert pump_until(lambda: state.wii.connected, 3.0)
    pump(0.1)  # deliver the controller's queued signals


def set_file_sources(page):
    for r, src in enumerate(page.sources):
        page.cam_table.cellWidget(r, 2).setEditText(str(src))
    pump(0.01)


def open_cameras(page):
    set_file_sources(page)
    page.b_open.click()
    assert pump_until(lambda: all(page._session.latest(c) is not None
                                  for c in ("cam01", "cam02")), 5.0)


def test_camera_table_follows_the_project_and_saves_capture_json(page):
    proj = page.state.project
    assert page.cam_table.rowCount() == 2
    assert [page.cam_table.item(r, 0).text() for r in range(2)] == ["cam01", "cam02"]
    assert page.cam_table.cellWidget(1, 2).currentText() == "1"  # default: device index 1
    assert page.grid.names() == ["cam01", "cam02"]
    set_file_sources(page)
    page.cam_table.cellWidget(0, 4).setEditText("1280x720")
    page.cam_table.cellWidget(0, 5).setEditText("30")
    page.fps_spin.setValue(25)
    page.chk_keep_raw.setChecked(False)
    saved = json.loads(WiiPaths(proj).capture_json.read_text(encoding="utf-8"))
    assert saved["cameras"]["cam01"] == {"source": str(page.sources[0]), "width": 1280,
                                         "height": 720, "fps": 30.0, "enabled": True}
    assert saved["cameras"]["cam02"]["source"] == str(page.sources[1])
    assert saved["output_fps"] == 25 and saved["keep_raw"] is False
    assert "hid" not in sys.modules


def test_preview_record_mark_stop_and_export(page, monkeypatch):
    answers = Answers(monkeypatch)
    state, proj = page.state, page.state.project
    start_simulator(state, monkeypatch)
    open_cameras(page)
    assert pump_until(lambda: page.grid.cell("cam01").image is not None, 2.0)
    assert "fps" in page.grid.cell("cam01").text

    page.b_record.click()
    s = page._session
    assert s.recording and page.b_record.text() == "■ Stop"
    assert state.wii.tare_locked == "recording" and not page.b_tare.isEnabled()
    assert not page.b_open.isEnabled() and page.b_mark.isEnabled()
    pump(0.4)
    page.event_edit.setText("sync")
    page.b_mark.click()
    page.event_edit.setText("")
    page.b_mark.click()
    state.wii.connection_event.emit("wii_connected")  # the state the take started with
    state.wii.connection_event.emit("wii_disconnected")  # a change: logged as an event
    state.wii.connection_event.emit("wii_disconnected")  # repeated: not logged again
    pump(0.8)
    assert "● REC" in page.rec_label.text()
    folder = s.folder
    page.b_record.click()  # stop -> videos/ is empty -> exported automatically
    assert not s.recording and state.wii.tare_locked is None
    assert pump_until(lambda: page._thread is None, 20.0)
    assert page.balance_events[:2] == ["recording", "trial"]

    labels = [e["label"] for e in wio.read_events_csv(folder)]
    assert labels == ["sync", "mark_2", "wii_disconnected"]
    meta = wio.read_session_json(folder)
    assert meta["camera_names"] == ["cam01", "cam02"] and meta["has_wii"]
    assert meta["samples"]["wii"] > 50 and meta["export"]["n_frames"] > 20
    for c in ("cam01", "cam02"):
        assert (proj.videos_dir / f"{c}.mp4").is_file()
    info = load_trial(proj)
    assert info.recording == folder.name and info.alignment.method == "recorded"
    assert answers.kinds()[-1] == "information"  # export summary
    assert "Next: run the pipeline on 4. Run" in answers.shown[-1][2]
    assert page.takes.rowCount() == 1
    assert page.takes.item(0, 0).text() == folder.name
    assert page.takes.item(0, 6).text() == "✓ recorded"
    assert page.takes.item(0, 5).text().endswith("fps")


def test_export_fps_change_updates_the_project(page, monkeypatch):
    answers = Answers(monkeypatch)
    proj = page.state.project
    rec = WiiPaths(proj).recording_dir("20260924_100000")
    write_take(rec, {"cam01": regular(0.0, 50), "cam02": regular(0.01, 50)})
    page._refresh_takes()
    changed = []
    page.state.project_changed.connect(changed.append)
    page.fps_spin.setValue(15)
    page._select_take(rec.name)
    page.b_export.click()
    assert page._export_running()
    assert pump_until(lambda: page._thread is None, 20.0)
    assert proj.config.frame_rate == 15 and changed == [proj]  # other pages reloaded
    assert load_trial(proj).alignment.details["fps"] == 15
    assert "project frame rate is now 15 fps" in answers.shown[-1][2]
    assert page._session is not None  # same project: the page kept its session


def test_cancel_export_leaves_videos_untouched(page, monkeypatch):
    answers = Answers(monkeypatch)
    proj = page.state.project
    old = proj.videos_dir / "cam01.mp4"
    old.write_bytes(b"previous")
    rec = WiiPaths(proj).recording_dir("20260924_110000")
    write_take(rec, {"cam01": regular(0.0, 400), "cam02": regular(0.0, 400)})
    assert page._start_export(rec.name)
    page._cancel_export()
    assert pump_until(lambda: page._thread is None, 20.0)
    assert "cancelled" in page.export_label.text()
    assert old.read_bytes() == b"previous" and sorted(p.name for p in proj.videos_dir.iterdir()) \
        == ["cam01.mp4"]
    assert "critical" not in answers.kinds()


def test_record_refuses_without_all_cameras_and_wii_only_take(page, monkeypatch):
    answers = Answers(monkeypatch)
    state, proj = page.state, page.state.project
    page.b_record.click()  # nothing open, no board
    assert not page._session.recording
    assert answers.shown[-1][0] == "warning" and "cam01 is not open" in answers.shown[-1][2]

    page.chk_record_cams.setChecked(False)
    page.b_record.click()  # Wii only, but no board
    assert not page._session.recording and "Nothing to record" in answers.shown[-1][2]

    start_simulator(state, monkeypatch)
    page.b_record.click()
    assert page._session.recording
    pump(0.5)
    folder = page._session.folder
    page.b_record.click()  # stop -> "Use as the trial's Wii recording?" -> yes
    assert sorted(p.name for p in folder.iterdir()) == ["session.json", "wii.csv"]
    info = load_trial(proj)
    assert info.recording == folder.name and info.source == "external"
    assert info.alignment.method == "none"
    assert page.balance_events[-1] == "trial"
    assert page.takes.item(0, 3).text() == "Wii only"


def test_video_take_without_board_asks_first(page, monkeypatch):
    answers = Answers(monkeypatch, question="no")
    open_cameras(page)
    page.b_record.click()
    assert not page._session.recording
    assert answers.shown[-1][0] == "question" and "No Wii Balance Board" in answers.shown[-1][2]
    answers.question = "yes"
    page.b_record.click()
    assert page._session.recording
    pump(0.3)
    answers.question = "no"  # do not replace videos/... (videos/ is empty: exports anyway)
    (page.state.project.videos_dir / "cam01.mp4").write_bytes(b"x")
    page.b_record.click()
    assert not page._session.recording and page._thread is None  # declined the export
    meta = wio.read_session_json(page._session.folder)
    assert meta["has_wii"] is False and meta["has_video"] is True


def test_can_close_and_shutdown(page, monkeypatch):
    answers = Answers(monkeypatch, question="no")
    start_simulator(page.state, monkeypatch)
    open_cameras(page)
    streams = list(page._session.streams.values())
    page.b_record.click()
    assert page._session.recording
    assert page.can_close() is False  # "Stop the recording and quit?" -> No
    answers.question = "yes"
    assert page.can_close() is True
    folder = page._session.folder
    page.shutdown()
    assert page._session is None and all(not s.running for s in streams)
    assert page.state.wii.tare_locked is None
    meta = wio.read_session_json(folder)
    assert "t_stop" in meta  # the take was finalized


def test_project_change_stops_the_take(page, monkeypatch, tmp_path):
    from poseassess.core.project import Project

    answers = Answers(monkeypatch)
    start_simulator(page.state, monkeypatch)
    page.chk_record_cams.setChecked(False)
    page.b_record.click()
    folder = page._session.folder
    other = Project(tmp_path / "other")
    other.config.num_cameras = 3
    other.create()
    page.state.set_project(other)
    assert answers.shown[-1][0] == "information" and "another project" in answers.shown[-1][2]
    assert "t_stop" in wio.read_session_json(folder)
    assert page.cam_table.rowCount() == 3 and not page._session.recording
    assert page.state.wii.tare_locked is None


def test_board_snapshots_and_scan(page, monkeypatch):
    answers = Answers(monkeypatch)
    page.b_snap.click() if page.b_snap.isEnabled() else page._board_snapshots()
    assert answers.shown[-1][0] == "warning"  # no camera open
    open_cameras(page)
    page.b_snap.click()
    assert answers.shown[-1][0] == "information"
    files = sorted((WiiPaths(page.state.project).board_dir).glob("cam*/snapshot_*.jpg"))
    assert [f.parent.name for f in files] == ["cam01", "cam02"]

    from poseassess.core.capture import session as session_mod

    seen = {}

    def fake_probe(max_index=8, timeout_s=3.0, skip=None):
        seen["skip"] = skip
        return [{"index": 0, "width": 1280, "height": 720, "fps": 30.0}]

    monkeypatch.setattr(session_mod, "probe_cameras", fake_probe)
    page.b_scan.click()
    assert pump_until(lambda: page._probe_thread is None, 5.0)
    assert seen["skip"] == set()  # file sources are not device indices
    assert "Found: 0 (1280x720, 30 fps)" in page.scan_label.text()
    combo = page.cam_table.cellWidget(0, 2)
    assert combo.currentText() == str(page.sources[0])  # the chosen source is kept
    assert combo.itemText(0).startswith("0 · found 1280x720")


def test_live_cop_view_follows_the_board(page, monkeypatch):
    start_simulator(page.state, monkeypatch)
    try:
        has_data = bool(page.state.wii.recent(1.0))
    except NotImplementedError:
        has_data = False
    pump(0.4)
    assert "●" in page.wii_label.text()
    if has_data or page.state.wii.recent(1.0):
        assert pump_until(lambda: len(page.cop.cop_trail) > 5, 2.0)
        assert np.isfinite(page.cop.cop_trail).all() and page.cop.total_kg > 50


def test_board_connecting_during_a_take_is_added(page, monkeypatch):
    answers = Answers(monkeypatch)  # "record the videos without force data?" -> yes
    open_cameras(page)
    page.b_record.click()
    s = page._session
    assert s.recording and answers.shown[-1][0] == "question"
    folder = s.folder
    # the board connects during the take (the confirmed "Connect" of 2b. Wii Board: the only
    # way to change the source while recording)
    page.state.wii.use_source(SimulatedBoard(), allow_during_lock=True)
    assert pump_until(lambda: page.state.wii.connected, 3.0)
    assert pump_until(lambda: s.recorder.counts["wii"] > 20, 3.0)
    (page.state.project.videos_dir / "cam01.mp4").write_bytes(b"x")
    answers.question = "no"  # do not replace videos/
    page.b_record.click()
    assert [e["label"] for e in wio.read_events_csv(folder)] == ["wii_connected"]
    meta = wio.read_session_json(folder)
    assert meta["has_wii"] is True and meta["samples"]["wii"] > 20
    assert [h["reason"] for h in meta["force_source_history"]][:2] == ["attached", "connected"]


def test_open_cameras_does_not_block_the_window(page, monkeypatch):
    """Opening a camera driver can take seconds: the cameras open on worker threads, the
    window keeps running; a driver that never answers is reported, not waited for."""
    import threading
    import time

    from poseassess.core.capture import camera as camera_mod
    from poseassess.core.capture import session as session_mod

    answers = Answers(monkeypatch)
    real_start = camera_mod.CameraStream.start
    release = threading.Event()
    late, opened_late = [], threading.Event()

    def slow_start(self):
        if self.name == "cam02" and not release.is_set():
            late.append(self)
            release.wait(10)  # a driver that hangs
            real_start(self)
            opened_late.set()
            return
        time.sleep(0.4)  # a slow but working driver
        real_start(self)

    monkeypatch.setattr(camera_mod.CameraStream, "start", slow_start)
    monkeypatch.setattr(session_mod, "OPEN_TIMEOUT_S", 1.0)
    set_file_sources(page)
    t0 = time.perf_counter()
    page.b_open.click()
    assert time.perf_counter() - t0 < 0.3  # returned at once
    assert not page.b_open.isEnabled() and page.b_open.text() == "Opening cameras…"
    assert not page.b_record.isEnabled() and not page.b_snap.isEnabled()
    pump(0.2)
    assert page.grid.cell("cam01").message == "opening…"
    assert pump_until(lambda: page._opening is None, 5.0)
    s = page._session
    assert list(s.streams) == ["cam01"] and page.b_open.isEnabled()
    assert answers.shown[-1][0] == "warning" and "cam02" in answers.shown[-1][2]
    assert "did not respond" in answers.shown[-1][2]
    assert pump_until(lambda: s.latest("cam01") is not None, 3.0)
    release.set()  # the hanging driver returns late: its own thread closes it again
    assert pump_until(lambda: opened_late.is_set() and not late[0].running
                      and late[0]._cap is None, 5.0)
    assert "cam02" not in s.streams
    page.b_close.click()
    assert not s.streams


def test_export_and_pipeline_run_exclude_each_other(page, monkeypatch):
    """The export replaces videos/ and Config.toml: no pipeline run and no "Assign video" while
    it runs, and no export while a pipeline run reads them. The project frame rate is updated
    on the GUI thread, not by the export worker."""
    import threading

    from poseassess.core.capture import export as export_mod
    from poseassess.gui.pages import run_page as run_mod
    from poseassess.gui.pages.videos_page import VideosPage

    answers = Answers(monkeypatch)
    state, proj = page.state, page.state.project
    rec = WiiPaths(proj).recording_dir("20260924_130000")
    write_take(rec, {"cam01": regular(0.0, 300), "cam02": regular(0.0, 300)})
    page._refresh_takes()
    page._select_take(rec.name)

    state.set_busy("pipeline", "the pipeline run (4. Run)")
    pump(0.01)
    assert not page.b_export.isEnabled() and "pipeline" in page.b_export.toolTip()
    assert page._start_export(rec.name) is False and page._thread is None
    assert answers.shown[-1][0] == "warning" and "pipeline run" in answers.shown[-1][2]
    state.set_busy("pipeline", None)
    pump(0.01)
    assert page.b_export.isEnabled()

    threads = []
    real_apply = export_mod.apply_export_fps
    monkeypatch.setattr(export_mod, "apply_export_fps", lambda p, fps: threads.append(
        threading.current_thread() is threading.main_thread()) or real_apply(p, fps))
    started = []
    monkeypatch.setattr(run_mod, "PipelineWorker", lambda *a, **k: started.append(1))
    run = run_mod.RunPage(state)
    videos = VideosPage(state)
    from PySide6.QtWidgets import QFileDialog
    monkeypatch.setattr(QFileDialog, "getExistingDirectory",
                        staticmethod(lambda *a, **k: pytest.fail("asked for a folder")))
    page.fps_spin.setValue(15)
    assert page._start_export(rec.name)
    assert "export of the take" in state.busy("export")
    run._run()
    assert started == [] and "Wait until the export" in answers.shown[-1][2]
    videos._assign_folder()
    assert "Wait until the export" in answers.shown[-1][2]
    assert pump_until(lambda: page._thread is None, 20.0)
    assert state.busy("export") is None
    assert threads == [True] and proj.config.frame_rate == 15  # applied on the GUI thread
    assert "project frame rate is now 15 fps" in answers.shown[-1][2]
    # the run page holds the "pipeline" flag while it runs
    monkeypatch.setattr(run_mod, "run_in_thread", lambda w: None)
    monkeypatch.setattr(run_mod, "PipelineWorker", _FakeWorker)
    run._run()
    assert state.busy("pipeline")
    run._on_finished([])
    assert state.busy("pipeline") is None
    run.close()
    videos.close()


class _FakeWorker:
    """Stands in for ``PipelineWorker`` (no subprocess)."""

    def __init__(self, *a, **k):
        from types import SimpleNamespace
        sig = SimpleNamespace(connect=lambda *a: None)
        self.log = self.stage_done = self.finished = self.failed = sig


def test_camera_latency_is_saved_and_subtracted_by_the_export(page, monkeypatch):
    answers = Answers(monkeypatch)
    proj = page.state.project
    page.latency_spin.setValue(40)
    saved = json.loads(WiiPaths(proj).capture_json.read_text(encoding="utf-8"))
    assert saved["latency_ms"] == 40.0
    assert "exposure" in page.latency_spin.toolTip()
    rec = WiiPaths(proj).recording_dir("20260924_140000")
    write_take(rec, {"cam01": regular(0.0, 60), "cam02": regular(0.0, 60)})
    page._refresh_takes()
    page._select_take(rec.name)
    page.b_export.click()
    assert pump_until(lambda: page._thread is None, 20.0)
    assert answers.kinds()[-1] == "information"
    fr = wio.read_frames_csv(rec)
    from poseassess.core.capture.export import plan_export
    ref = plan_export(rec, fps=30)
    np.testing.assert_allclose(fr["t_rel"], ref.t_rel - 0.040, atol=1e-6)
    assert wio.read_session_json(rec)["export"]["latency_ms"] == {"cam01": 40.0, "cam02": 40.0}
