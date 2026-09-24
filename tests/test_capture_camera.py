"""GUI-CAPTURE: CameraStream port (DirectShow MJPG order, stop never releases during a blocking read, writer failure, frames of another size not timestamped, write error keeps capturing, abrupt-exit MKV readable) using tests/synth.write_test_video sources."""

import csv
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from poseassess.core.capture import camera as camera_mod
from poseassess.core.capture.camera import (
    VIDEO_EXT,
    CameraStream,
    configure_capture,
    open_video_writer,
)
from poseassess.wii import io as wio
from tests.synth import write_test_video
from tests.wii_fakes import wait_until

ROOT = Path(__file__).resolve().parents[1]


class FakeCap:
    """Stand-in for cv2.VideoCapture (port of PoseBoard tests/test_runtime.py)."""

    def __init__(self, sizes=((240, 320),), block_after=None, block_s=0.0):
        self.calls, self.sizes, self.n = [], list(sizes), 0
        self.block_after, self.block_s = block_after, block_s
        self.reading = False
        self.released = False
        self.released_during_read = False
        self.fourcc = 0

    def set(self, prop, value):
        self.calls.append((prop, value))
        if prop == cv2.CAP_PROP_FOURCC:
            self.fourcc = int(value)
        return True

    def get(self, prop):
        return self.fourcc if prop == cv2.CAP_PROP_FOURCC else 30.0

    def isOpened(self):
        return True

    def read(self):
        self.reading = True
        try:
            if self.block_after is not None and self.n >= self.block_after:
                time.sleep(self.block_s)  # e.g. a stalled network stream
            else:
                time.sleep(0.005)
            h, w = self.sizes[min(self.n // 10, len(self.sizes) - 1)]
            self.n += 1
            return True, np.full((h, w, 3), self.n % 255, np.uint8)
        finally:
            self.reading = False

    def release(self):
        self.released_during_read |= self.reading
        self.released = True


def count_frames(path) -> int:
    v = cv2.VideoCapture(str(path))
    n = 0
    while v.read()[0]:
        n += 1
    v.release()
    return n


def test_directshow_mjpg_is_requested_last():
    cap = FakeCap()
    configure_capture(cap, 1280, 720, 30, mjpg=True)
    props = [p for p, _ in cap.calls]
    assert props == [cv2.CAP_PROP_FPS, cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT,
                     cv2.CAP_PROP_FOURCC]
    assert cap.calls[-1][1] == cv2.VideoWriter_fourcc(*"MJPG")


def test_open_capture_only_uses_directshow_on_windows(monkeypatch):
    seen = []

    class Cap(FakeCap):
        def __init__(self, *args):
            super().__init__()
            seen.append(args)

    monkeypatch.setattr(camera_mod.cv2, "VideoCapture", Cap)
    monkeypatch.setattr(camera_mod.sys, "platform", "win32")
    cap = camera_mod.open_capture("1", 640, 480, 30)
    assert seen[-1] == (1, cv2.CAP_DSHOW)  # digit string -> device index, DirectShow
    assert cap.calls[-1][0] == cv2.CAP_PROP_FOURCC
    monkeypatch.setattr(camera_mod.sys, "platform", "linux")
    cap = camera_mod.open_capture("rtsp://example/stream")
    assert seen[-1] == ("rtsp://example/stream",)
    assert all(p != cv2.CAP_PROP_FOURCC for p, _ in cap.calls)  # no MJPG outside DirectShow


def test_camera_stop_never_releases_during_a_blocking_read(monkeypatch):
    cap = FakeCap(block_after=3, block_s=0.6)
    monkeypatch.setattr(camera_mod, "open_capture", lambda *a, **k: cap)
    s = CameraStream("net", "rtsp://example/stream")
    s.stop_timeout_s = 0.1
    s.start()
    assert wait_until(lambda: cap.n >= 3 and cap.reading)
    t = time.perf_counter()
    s.stop()
    assert time.perf_counter() - t < 0.5 and not cap.released  # left to the capture thread
    assert wait_until(lambda: cap.released, 2.0)
    assert not cap.released_during_read


def test_video_writer_failure_is_an_error(tmp_path):
    with pytest.raises(RuntimeError, match="Cannot write video"):
        open_video_writer(tmp_path / "missing_dir" / "cam01.mkv", 30, (320, 240))


def test_start_recording_needs_a_frame(tmp_path, monkeypatch):
    cap = FakeCap(block_after=0, block_s=0.5)
    monkeypatch.setattr(camera_mod, "open_capture", lambda *a, **k: cap)
    s = CameraStream("cam01", "0")
    s.stop_timeout_s = 0.05
    s.start()
    try:
        with pytest.raises(RuntimeError, match="no frames yet"):
            s.start_recording(tmp_path / "cam01.mkv")
    finally:
        s.stop()


def test_frames_of_another_size_are_not_timestamped(tmp_path, monkeypatch):
    cap = FakeCap(sizes=((240, 320), (120, 160), (240, 320)))
    monkeypatch.setattr(camera_mod, "open_capture", lambda *a, **k: cap)
    s = CameraStream("cam01", "0")
    s.start()
    try:
        assert wait_until(lambda: s.latest() is not None)
        path = s.start_recording(tmp_path / "cam01.mkv")
        assert wait_until(lambda: cap.n > 35)
        s.stop_recording()
    finally:
        s.stop()
    assert s.dropped_frames > 0
    ts = tmp_path / wio.timestamps_csv_name("cam01")
    rows = list(csv.DictReader(open(ts, newline="")))
    assert count_frames(path) == len(rows) > 5
    assert list(rows[0]) == wio.TIMESTAMPS_HEADER


def test_recording_write_error_does_not_kill_the_camera(tmp_path, monkeypatch):
    cap = FakeCap()
    monkeypatch.setattr(camera_mod, "open_capture", lambda *a, **k: cap)
    s = CameraStream("cam01", "0")
    s.start()
    try:
        assert wait_until(lambda: s.latest() is not None)
        s.start_recording(tmp_path / "cam01.mkv")

        def boom(*a):
            raise OSError(28, "No space left on device")

        monkeypatch.setattr(s, "_write_frame", boom)
        assert wait_until(lambda: s.error is not None and "recording stopped" in s.error)
        n = cap.n
        assert wait_until(lambda: cap.n > n + 5) and s.running  # still capturing
        assert s._writer is None
    finally:
        s.stop()


def test_file_source_loops_and_records_with_shared_clock(tmp_path):
    src = write_test_video(tmp_path / "in.avi", n_frames=10, fps=30.0)
    s = CameraStream("cam01", str(src))
    s.start()
    try:
        assert wait_until(lambda: s.latest() is not None)
        assert s.fps == pytest.approx(30.0, abs=0.5)
        t0 = time.perf_counter()
        off = 1.7e9
        path = s.start_recording(tmp_path / f"cam01{VIDEO_EXT}", clock_offset_unix=off, t0=t0)
        assert wait_until(lambda: s.latest().index > 25, 5.0)  # looped past the 10 frames
        s.stop_recording()
    finally:
        s.stop()
    assert path.name == "cam01.mkv" and s.video_path == path
    ts = wio.read_timestamps_csv(tmp_path / "cam01_timestamps.csv")
    n = len(ts["frame"])
    assert n > 10 and count_frames(path) == n
    np.testing.assert_array_equal(ts["frame"], np.arange(n))
    np.testing.assert_allclose(ts["t_rel"], ts["t"] - t0, atol=2e-6)
    np.testing.assert_allclose(ts["t_unix"], ts["t"] + off, atol=2e-6)
    assert np.all(np.diff(ts["t"]) > 0)
    assert not s.running


CHILD = r'''
import os, sys, time
sys.path.insert(0, sys.argv[1])
from pathlib import Path
import cv2
import numpy as np
from poseassess.core.capture.camera import CameraStream

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
cam.start_recording(out / "cam01.mkv")
time.sleep(2.5)
os._exit(1)  # crash / killed / power loss: nothing is closed
'''


def test_recording_survives_an_abrupt_exit(tmp_path):
    script = tmp_path / "child.py"
    script.write_text(CHILD, encoding="utf-8")
    r = subprocess.run([sys.executable, str(script), str(ROOT), str(tmp_path)], timeout=60,
                       capture_output=True, text=True)
    assert r.returncode == 1, r.stderr
    n = count_frames(tmp_path / "cam01.mkv")
    assert n > 10  # an MP4 would be unreadable without its index
    ts = wio.read_timestamps_csv(tmp_path / "cam01_timestamps.csv")
    assert len(ts["frame"]) > 10  # flushed about once per second
