"""CORE-WII: WiiRecorder (Wii-only / video-only / both with a fake VideoSink, t/t_rel/t_unix
columns, events BOM + flush, attach_force mid-recording, force_source_history, same-second
folders, stop twice, crash safety) and capture_clock_offset."""

import csv
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from poseassess.core.balance.geometry import BoardGeometry, BoardPose, RigidTransform
from poseassess.wii import io as wio
from poseassess.wii import recorder as recorder_mod
from poseassess.wii.device import ForceSource, SimulatedBoard
from poseassess.wii.recorder import (
    WiiRecorder,
    capture_clock_offset,
    iso_time,
    new_recording_folder,
    on_console_close,
    remove_console_close,
)
from tests.wii_fakes import wait_until

ROOT = Path(__file__).resolve().parents[1]


class FakeSink:
    """A camera for ``WiiRecorder`` (the ``VideoSink`` protocol) that writes placeholder files."""

    def __init__(self, name, fps=30.0, fail=False, ext=".mkv"):
        self.name, self.source, self.fps = name, f"fake:{name}", fps
        self.fail, self.ext = fail, ext
        self.calls: list = []
        self.stopped = 0

    def start_recording(self, video_path, clock_offset_unix=None, t0=None):
        if self.fail:
            raise RuntimeError(f"Camera {self.name} has no frames yet")
        video_path = Path(video_path).with_suffix(self.ext)  # ".avi": the writer's fallback
        video_path.write_bytes(b"video")
        video_path.with_name(video_path.stem + wio.TIMESTAMPS_SUFFIX).write_text(
            ",".join(wio.TIMESTAMPS_HEADER) + "\n")
        self.calls.append((video_path, clock_offset_unix, t0))
        return video_path

    def stop_recording(self):
        self.stopped += 1


class ManualSource(ForceSource):
    """A ForceSource without a thread: samples are emitted by the test."""

    def __init__(self, key="manual", **kw):
        super().__init__(**kw)
        self.key = key

    @property
    def device_key(self):
        return self.key

    def emit(self, kg=(20.0, 20.0, 20.0, 20.0), t=None):
        return self._emit_kg(time.perf_counter() if t is None else t, np.asarray(kg, float))


def rows(path, encoding="utf-8"):
    with open(path, newline="", encoding=encoding) as f:
        return list(csv.reader(f))


# ------------------------------------------------------------------ clock
def test_capture_clock_offset_and_iso_time():
    t, u = capture_clock_offset()
    assert abs(t - time.perf_counter()) < 0.5 and abs(u - time.time()) < 0.5
    offsets = []
    for _ in range(5):
        t, u = capture_clock_offset()
        offsets.append(u - t)
    assert max(offsets) - min(offsets) < 0.02  # same offset to within a few ms
    s = iso_time(1.7e9 + 0.1234)
    assert s[10] == "T" and s.count(":") >= 2 and ("+" in s[19:] or "-" in s[19:] or "Z" in s)
    assert ".123" in s


# ------------------------------------------------------------------ folders
def test_new_recording_folder_same_second_and_subject(tmp_path):
    t = 1.7e9
    f1 = new_recording_folder(tmp_path / "rec", "S 01/ä", t)
    f2 = new_recording_folder(tmp_path / "rec", "S 01/ä", t)
    f3 = new_recording_folder(tmp_path / "rec", "S 01/ä", t)
    assert f1.name.endswith("_S_01") and len(f1.name) == len("YYYYmmdd_HHMMSS_S_01")
    assert f2.name == f1.name + "_2" and f3.name == f1.name + "_3"
    assert new_recording_folder(tmp_path / "rec", "  ", t).name == f1.name[:15]
    assert all(f.is_dir() and not any(f.iterdir()) for f in (f1, f2, f3))


def test_recordings_in_the_same_second_get_separate_folders(tmp_path, monkeypatch):
    monkeypatch.setattr(recorder_mod, "capture_clock_offset",
                        lambda: (time.perf_counter(), 1.7e9))
    rec = WiiRecorder()
    f1 = rec.start(root=tmp_path, subject="S")
    rec.add_event("sync")
    rec.stop()
    f2 = rec.start(root=tmp_path, subject="S")
    rec.stop()
    assert f2.name == f1.name + "_2"
    assert (f1 / wio.EVENTS_CSV).exists() and not (f2 / wio.EVENTS_CSV).exists()
    meta = wio.read_session_json(f2)
    assert meta["samples"] == {"wii": 0, "events": 0}
    assert not meta["has_wii"] and not meta["has_video"]


def test_start_arguments(tmp_path):
    rec = WiiRecorder()
    with pytest.raises(ValueError):
        rec.start()
    folder = rec.start(folder=tmp_path / "given" / "take")
    try:
        assert folder == tmp_path / "given" / "take" and rec.recording
        with pytest.raises(RuntimeError, match="Already recording"):
            rec.start(root=tmp_path)
    finally:
        rec.stop()
    assert not rec.recording and rec.folder is None and rec.last_folder == folder


# ------------------------------------------------------------------ Wii only
def test_wii_only_recording(tmp_path):
    sim = SimulatedBoard()
    sim.start()
    rec = WiiRecorder()
    try:
        assert wait_until(lambda: sim.connected)
        before = time.time()
        folder = rec.start(root=tmp_path / "rec", force=sim, geometry=BoardGeometry(),
                           subject="S01", notes="eyes open",
                           extra_meta={"project": {"name": "p", "root": str(tmp_path)}})
        assert folder.parent == tmp_path / "rec" and folder.name.endswith("_S01")
        started = wio.read_session_json(folder)  # written at start (crash safety)
        assert started["schema"] == wio.SESSION_SCHEMA and "t_stop" not in started
        assert started["force_source_history"][0]["reason"] == "start"
        time.sleep(0.4)
        ev = rec.add_event("  yeux   fermés ")  # whitespace-normalized, non-ASCII
        assert ev["label"] == "yeux fermés" and ev["t_rel"] == pytest.approx(ev["t"] - rec.t0)
        assert rec.add_event("")["label"] == "event"
        # events are flushed immediately (a crash must not lose them)
        assert len(wio.read_events_csv(folder)) == 2
        time.sleep(0.2)
        assert rec.stop() == folder
        after = time.time()
    finally:
        sim.stop()
    assert rec.stop() is None  # stop twice
    assert rec.add_event("late") is None

    meta = json.loads((folder / wio.SESSION_JSON).read_text(encoding="utf-8"))
    for key in ("schema", "app", "created", "subject", "notes", "clock", "t0", "t0_unix",
                "clock_offset_unix", "start_time_iso", "has_wii", "has_video", "camera_names",
                "streams", "world_frame", "board_geometry", "board_pose", "force_source",
                "force_source_history", "t_stop", "t_stop_unix", "stop_time_iso", "duration_s",
                "samples", "project"):
        assert key in meta, key
    assert meta["app"].startswith("PoseAssess") and meta["subject"] == "S01"
    assert meta["notes"] == "eyes open" and meta["project"]["name"] == "p"
    assert meta["has_wii"] and not meta["has_video"] and meta["camera_names"] == []
    assert meta["board_pose"] is None and meta["board_geometry"]["sensor_dx_mm"] == 433.0
    assert meta["clock_offset_unix"] == pytest.approx(meta["t0_unix"] - meta["t0"])
    assert before - 0.1 < meta["t0_unix"] < after
    assert 0.55 < meta["duration_s"] < 5 and meta["samples"]["events"] == 2
    (h,) = meta["force_source_history"]
    assert h["reason"] == "start" and h["device"] == "simulator" and h["min_total_kg"] == 5.0
    assert meta["force_source"]["type"] == "SimulatedBoard" and "errors" not in meta

    head, *body = rows(folder / wio.WII_CSV)
    assert head == wio.WII_HEADER and len(body) == meta["samples"]["wii"] > 40
    wii = wio.read_wii_csv(folder)
    np.testing.assert_allclose(wii["t_rel"], wii["t"] - meta["t0"], atol=2e-6)
    np.testing.assert_allclose(wii["t_unix"], wii["t"] + meta["clock_offset_unix"], atol=2e-5)
    assert wii["t_rel"][0] >= -0.05 and wii["t_rel"][-1] <= meta["duration_s"] + 0.05
    assert np.all(np.diff(wii["t"]) > 0)
    assert 60 < np.nanmean(wii["total_kg"]) < 80
    np.testing.assert_allclose(wii["total_kg"], wii["TR_kg"] + wii["BR_kg"] + wii["TL_kg"]
                               + wii["BL_kg"], atol=5e-6)
    assert np.all(np.isfinite(wii["cop_x_board"])) and np.all(np.isnan(wii["cop_x_world"]))

    raw = (folder / wio.EVENTS_CSV).read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf")  # BOM: Excel detects UTF-8
    evs = wio.read_events_csv(folder)
    assert [e["label"] for e in evs] == ["yeux fermés", "event"]
    assert evs[0]["t_unix"] == pytest.approx(evs[0]["t"] + meta["clock_offset_unix"], abs=2e-5)
    assert not (folder / (wio.SESSION_JSON + ".tmp")).exists()


def test_board_pose_fills_cop_world(tmp_path):
    a = np.deg2rad(30)
    R = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    pose = BoardPose(RigidTransform(R, np.array([0.5, 0.2, 0.053])), "triangulation",
                     {"cam01": 0.4})
    src = ManualSource()
    rec = WiiRecorder()
    folder = rec.start(root=tmp_path, force=src, board=pose, geometry=BoardGeometry())
    src.emit([30.0, 10.0, 20.0, 10.0])
    src.emit([0.5, 0.5, 0.5, 0.5])  # below min_total_kg: COP (board and world) empty
    rec.stop()
    wii = wio.read_wii_csv(folder)
    cop_b = np.array([wii["cop_x_board"][0], wii["cop_y_board"][0], 0.0])
    np.testing.assert_allclose([wii["cop_x_world"][0], wii["cop_y_world"][0],
                                wii["cop_z_world"][0]], R @ cop_b + [0.5, 0.2, 0.053], atol=2e-6)
    assert np.isnan(wii["cop_x_board"][1]) and np.isnan(wii["cop_z_world"][1])
    assert wii["total_kg"][1] == pytest.approx(2.0)
    meta = wio.read_session_json(folder)
    assert meta["board_pose"]["method"] == "triangulation"
    np.testing.assert_allclose(meta["board_pose"]["board_to_world"]["R"], R)


# ------------------------------------------------------------------ video
def test_video_only_with_fake_sinks(tmp_path):
    cams = [FakeSink("cam01"), FakeSink("cam02", fps=25.0, ext=".avi")]
    rec = WiiRecorder()
    folder = rec.start(root=tmp_path, cams=cams)
    try:
        for c in cams:
            (path, offset, t0), = c.calls
            assert path.parent == folder and offset == rec.clock_offset_unix and t0 == rec.t0
        assert not (folder / wio.WII_CSV).exists()
    finally:
        rec.stop()
    assert [c.stopped for c in cams] == [1, 1]
    meta = wio.read_session_json(folder)
    assert meta["has_video"] and not meta["has_wii"] and meta["force_source"] is None
    assert meta["camera_names"] == ["cam01", "cam02"]
    assert meta["streams"] == [
        {"name": "cam01", "source": "fake:cam01", "fps": 30.0, "video": "cam01.mkv",
         "timestamps": "cam01_timestamps.csv"},
        {"name": "cam02", "source": "fake:cam02", "fps": 25.0, "video": "cam02.avi",
         "timestamps": "cam02_timestamps.csv"}]
    for st in meta["streams"]:
        assert (folder / st["video"]).is_file() and (folder / st["timestamps"]).is_file()


def test_wii_and_video_together(tmp_path):
    src = ManualSource()
    cam = FakeSink("cam01")
    rec = WiiRecorder()
    folder = rec.start(root=tmp_path, force=src, cams=[cam])
    for _ in range(5):
        src.emit()
    rec.stop()
    meta = wio.read_session_json(folder)
    assert meta["has_wii"] and meta["has_video"] and meta["samples"]["wii"] == 5
    assert cam.stopped == 1 and len(wio.read_wii_csv(folder)["t"]) == 5


def test_camera_start_failure_rolls_back(tmp_path):
    ok, bad = FakeSink("cam01"), FakeSink("cam02", fail=True)
    src = ManualSource()
    rec = WiiRecorder()
    with pytest.raises(RuntimeError, match="no frames"):
        rec.start(root=tmp_path / "rec", force=src, cams=[ok, bad])
    assert ok.stopped == 1 and not rec.recording and rec.folder is None
    assert not src._listeners  # nothing listens to the board
    assert list((tmp_path / "rec").iterdir()) == []  # the folder it created is removed
    # a folder given by the caller is left in place
    given = tmp_path / "given"
    with pytest.raises(RuntimeError):
        rec.start(folder=given, cams=[FakeSink("cam01"), FakeSink("cam02", fail=True)])
    assert given.is_dir()
    # the recorder is usable afterwards
    folder = rec.start(root=tmp_path / "rec", force=src)
    src.emit()
    rec.stop()
    assert wio.read_session_json(folder)["samples"]["wii"] == 1


def test_stop_survives_a_failing_camera(tmp_path):
    cam = FakeSink("cam01")

    def boom():
        raise OSError("device gone")

    cam.stop_recording = boom
    rec = WiiRecorder()
    folder = rec.start(root=tmp_path, cams=[cam])
    assert rec.stop() == folder
    meta = wio.read_session_json(folder)
    assert "t_stop" in meta and any("device gone" in e for e in meta["errors"])
    assert "device gone" in rec.error


# ------------------------------------------------------------------ force source changes
def test_attach_force_mid_recording(tmp_path):
    a, b = ManualSource("board-a"), ManualSource("board-b")
    rec = WiiRecorder()
    folder = rec.start(root=tmp_path)  # started before the board was connected
    assert not (folder / wio.WII_CSV).exists()
    rec.attach_force(a)
    assert (folder / wio.WII_CSV).exists()
    a.emit([10, 10, 10, 10])
    rec.attach_force(a)  # same source: no-op
    rec.attach_force(b)  # the board reconnected as a new object
    a.emit([99, 99, 99, 99])  # the replaced source is ignored
    b.emit([20, 20, 20, 20])
    rec.stop()
    b.emit([30, 30, 30, 30])  # after stop: ignored
    assert not a._listeners and not b._listeners
    wii = wio.read_wii_csv(folder)
    np.testing.assert_allclose(wii["total_kg"], [40, 80])
    meta = wio.read_session_json(folder)
    assert meta["has_wii"] and meta["samples"]["wii"] == 2
    assert [(h["reason"], h["device"]) for h in meta["force_source_history"]] == [
        ("attached", "board-a"), ("attached", "board-b")]
    rec.attach_force(a)  # not recording: no-op
    assert not a._listeners


def test_attach_none_detaches(tmp_path):
    a = ManualSource()
    rec = WiiRecorder()
    folder = rec.start(root=tmp_path, force=a)
    a.emit()
    rec.attach_force(None)
    a.emit()
    assert not a._listeners
    rec.attach_force(a)
    a.emit()
    rec.stop()
    assert wio.read_session_json(folder)["samples"]["wii"] == 2


def test_same_source_in_consecutive_recordings_is_not_duplicated(tmp_path):
    src = ManualSource()
    rec = WiiRecorder()
    f1 = rec.start(root=tmp_path, force=src)
    src.emit()
    rec.stop()
    f2 = rec.start(root=tmp_path, force=src)
    src.emit()
    src.emit()
    rec.stop()
    assert len(src._listeners) == 0
    assert wio.read_session_json(f1)["samples"]["wii"] == 1
    assert len(wio.read_wii_csv(f2)["t"]) == 2


def test_log_force_source_and_update_meta(tmp_path):
    src = ManualSource()
    rec = WiiRecorder()
    folder = rec.start(root=tmp_path, force=src)
    src.emit([5, 5, 5, 5])
    src.do_tare(1.0)
    rec.log_force_source(src, "tare")
    rec.update_meta(capture={"settings": {"output_fps": 30}})
    mid = wio.read_session_json(folder)
    assert mid["capture"]["settings"]["output_fps"] == 30
    assert [h["reason"] for h in mid["force_source_history"]] == ["start", "tare"]
    assert mid["force_source_history"][1]["tare_kg"] == [5, 5, 5, 5]
    rec.stop()
    rec.update_meta(export={"fps": 30})  # after stop: the last recording's session.json
    meta = wio.read_session_json(folder)
    assert meta["export"] == {"fps": 30} and "t_stop" in meta and meta["capture"]
    rec.log_force_source(src, "late")  # not recording: ignored
    assert len(wio.read_session_json(folder)["force_source_history"]) == 2


def test_to_unix(tmp_path):
    rec = WiiRecorder()
    assert abs(rec.to_unix(time.perf_counter()) - time.time()) < 0.1
    rec.start(root=tmp_path)
    rec.stop()
    assert rec.to_unix(rec.t0) == pytest.approx(rec.t0_unix)


# ------------------------------------------------------------------ errors and threads
class BrokenWriter:
    def writerow(self, row):
        raise OSError(28, "No space left on device")


def test_wii_write_error_is_reported_not_raised(tmp_path):
    src = ManualSource()
    rec = WiiRecorder()
    folder = rec.start(root=tmp_path, force=src)
    src.emit()
    rec._wii_csv = BrokenWriter()  # the disk is full from now on
    src.emit()
    src.emit()
    assert "No space left" in rec.error and rec.recording
    assert rec.add_event("still works")["label"] == "still works"
    rec.stop()
    meta = wio.read_session_json(folder)
    assert meta["samples"]["wii"] == 1 and meta["has_wii"]
    assert len(meta["errors"]) == 1 and "wii.csv" in meta["errors"][0]


def test_concurrent_events_samples_and_stop(tmp_path):
    sim = SimulatedBoard(rate_hz=500)
    sim.start()
    rec = WiiRecorder()
    try:
        folder = rec.start(root=tmp_path, force=sim)
        errors = []

        def mark(n):
            try:
                for i in range(50):
                    rec.add_event(f"t{n}_{i}")
                    rec.update_meta(**{f"k{n}": i})
            except Exception as e:  # noqa: BLE001
                errors.append(e)

        threads = [threading.Thread(target=mark, args=(n,)) for n in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        time.sleep(0.2)
        rec.stop()
    finally:
        sim.stop()
    assert not errors
    meta = wio.read_session_json(folder)
    assert meta["samples"]["events"] == 200 and all(meta[f"k{n}"] == 49 for n in range(4))
    assert len(wio.read_events_csv(folder)) == 200
    assert len(wio.read_wii_csv(folder)["t"]) == meta["samples"]["wii"] > 50


def test_console_close_helpers_are_harmless():
    if not sys.platform.startswith("win"):
        assert on_console_close(lambda: None) is None
    remove_console_close(None)


def test_write_session_json_is_atomic_and_numpy_safe(tmp_path):
    meta = {"fps": np.float32(29.97), "n": np.int64(3), "arr": np.arange(3), "path": Path("a/b"),
            "dev": b"\\\\?\\hid#vid_057e", "text": "\u4e2d\u6587"}
    p = wio.write_session_json(tmp_path, meta)
    back = wio.read_session_json(tmp_path)
    assert back["n"] == 3 and back["arr"] == [0, 1, 2] and back["path"] == "a/b"
    assert back["fps"] == pytest.approx(29.97, abs=1e-4) and back["text"] == "\u4e2d\u6587"
    assert p == tmp_path / wio.SESSION_JSON and not list(tmp_path.glob("*.tmp"))
    with pytest.raises(TypeError):
        wio.write_session_json(tmp_path, {"bad": object()})
    assert wio.read_session_json(tmp_path)["n"] == 3  # the old file is intact


# ------------------------------------------------------------------ crash safety
CHILD = r'''
import os, sys, time
sys.path.insert(0, sys.argv[1])
from pathlib import Path
import cv2
import numpy as np
from poseassess.core.capture.camera import CameraStream
from poseassess.wii.device import SimulatedBoard
from poseassess.wii.recorder import WiiRecorder

out = Path(sys.argv[2])
# Noisy frames like a real camera sensor (flat synthetic frames compress to almost nothing and
# would just sit in the writer's buffer)
vw = cv2.VideoWriter(str(out / "in.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30, (640, 480))
rng = np.random.default_rng(0)
for _ in range(30):
    vw.write(rng.integers(0, 255, (480, 640, 3), dtype=np.uint8))
vw.release()
cam = CameraStream("cam01", str(out / "in.mp4"))
cam.start()
while cam.latest() is None:
    time.sleep(0.01)
sim = SimulatedBoard()
sim.start()
rec = WiiRecorder()
rec.start(root=out / "rec", force=sim, cams=[cam])
time.sleep(1.2)
rec.add_event("sync")
time.sleep(1.3)
os._exit(1)  # crash / killed / power loss: nothing is closed
'''


def test_recording_survives_an_abrupt_exit(tmp_path):
    import cv2

    script = tmp_path / "child.py"
    script.write_text(CHILD, encoding="utf-8")
    r = subprocess.run([sys.executable, str(script), str(ROOT), str(tmp_path)], timeout=120,
                       capture_output=True, text=True)
    assert r.returncode == 1, r.stderr
    (folder,) = list((tmp_path / "rec").iterdir())
    v = cv2.VideoCapture(str(folder / "cam01.mkv"))
    n = 0
    while v.read()[0]:
        n += 1
    v.release()
    assert n > 10  # an MP4 would be unreadable without its index
    ts = wio.read_timestamps_csv(folder / "cam01_timestamps.csv")
    wii = wio.read_wii_csv(folder)
    assert len(ts["t"]) > 10 and len(wii["t"]) > 50  # flushed about once per second
    meta = wio.read_session_json(folder)  # valid JSON, written at start
    assert meta["schema"] == wio.SESSION_SCHEMA and meta["has_video"] and "t_stop" not in meta
    assert meta["streams"][0]["video"] == "cam01.mkv"
    assert [e["label"] for e in wio.read_events_csv(folder)] == ["sync"]


def test_readers_drop_a_row_cut_off_by_a_crash(tmp_path):
    src = ManualSource()
    rec = WiiRecorder()
    folder = rec.start(root=tmp_path, force=src)
    for _ in range(3):
        src.emit([17.5, 17.5, 17.5, 17.5])
    rec.stop()
    p = folder / wio.WII_CSV
    data = p.read_bytes()
    p.write_bytes(data + b"99999.1,12.3,17")  # killed while writing a row
    wii = wio.read_wii_csv(folder)
    assert len(wii["t"]) == 3 and np.all(wii["total_kg"] == 70.0)
    p.write_bytes(data.rstrip(b"\r\n"))  # a complete file without the final line break
    assert len(wio.read_wii_csv(folder)["t"]) == 2  # the last row cannot be trusted
    ts = tmp_path / "cam01_timestamps.csv"
    ts.write_text("frame,t,t_rel,t_unix\r\n0,1.0,0.0,1.7e9\r\n1,1.03", encoding="utf-8")
    assert list(wio.read_timestamps_csv(ts)["frame"]) == [0]
    # files written in one go keep every row
    ev = tmp_path / "events.csv"
    ev.write_text("t,t_rel,t_unix,label\n1,0,2,sync", encoding="utf-8")
    assert [e["label"] for e in wio.read_events_csv(ev)] == ["sync"]
    fr = tmp_path / "frames.csv"
    fr.write_text("frame,t,t_rel,t_unix\n0,1,0,2\n1,1.1,0.1,2.1", encoding="utf-8")
    assert list(wio.read_frames_csv(fr)["frame"]) == [0, 1]
    assert wio.read_csv_columns(ev)["t"].size == 1
    assert wio.read_csv_columns(ev, complete_rows_only=True)["t"].size == 0
