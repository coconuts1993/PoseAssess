"""GUI-CAPTURE: CaptureSession with file sources + SimulatedBoard (take folder layout, timestamps, events), probe_cameras smoke, plan_export grid (latest start, jitter, dropped frames, fps choice), export_recording_to_project (equal frame counts, frames.csv, other-extension camNN.* removed, project fps + Config.toml updated, trial.json recorded alignment, keep_raw, cancel leaves videos/ untouched)."""

import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from poseassess.core.balance.paths import WiiPaths
from poseassess.core.balance.trial import load_trial, videos_changed, videos_fingerprint
from poseassess.core.capture import camera as camera_mod
from poseassess.core.capture import export as export_mod
from poseassess.core.capture.export import (
    ExportCancelled,
    export_recording_to_project,
    plan_export,
    take_cameras,
)
from poseassess.core.capture.session import CaptureSession, probe_cameras
from poseassess.core.capture.settings import (
    CameraSetting,
    CaptureSettings,
    default_settings,
    load_capture_settings,
    save_capture_settings,
)
from poseassess.wii import io as wio
from poseassess.wii.device import SimulatedBoard
from tests.synth import ring_cameras, write_calib_toml, write_test_video
from tests.wii_fakes import wait_until

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib

SIZE = (320, 240)
T0, OFFSET = 1000.0, 1.7e9


# ------------------------------------------------------------------ synthetic takes
def pattern(idx: int, size=SIZE) -> np.ndarray:
    """A frame whose index is written as 8 big black/white blocks (survives mp4v coding)."""
    img = np.full((size[1], size[0], 3), 60, np.uint8)
    for b in range(8):
        if (idx >> b) & 1:
            img[100:140, b * 40:b * 40 + 36] = 255
    return img


def decode(img: np.ndarray) -> int:
    return sum(1 << b for b in range(8) if img[105:135, b * 40 + 4:b * 40 + 32].mean() > 160)


def read_indices(path) -> list[int]:
    cap = cv2.VideoCapture(str(path))
    out = []
    while True:
        ok, img = cap.read()
        if not ok:
            break
        out.append(decode(img))
    cap.release()
    return out


def write_take(folder: Path, cams: dict, fps=30.0, size=SIZE, raw=True, video_frames=None):
    """A take recorded by the capture: ``cams = {name: t_rel array of the frames that were
    written}``; the raw video's frame k shows the pattern of k. session.json as WiiRecorder
    writes it. ``video_frames`` = {name: n} writes fewer frames than timestamps (crash)."""
    folder.mkdir(parents=True, exist_ok=True)
    streams = []
    for name, tr in cams.items():
        tr = np.asarray(tr, float)
        with open(folder / wio.timestamps_csv_name(name), "w", newline="") as f:
            f.write(",".join(wio.TIMESTAMPS_HEADER) + "\n")
            for k, t in enumerate(tr):
                f.write(f"{k},{T0 + t:.6f},{t:.6f},{T0 + OFFSET + t:.6f}\n")
        if raw:
            vw = cv2.VideoWriter(str(folder / f"{name}.mkv"), cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps, size)
            n = (video_frames or {}).get(name, len(tr))
            for k in range(n):
                vw.write(pattern(k, size))
            vw.release()
        streams.append({"name": name, "source": "0", "fps": fps, "video": f"{name}.mkv",
                        "timestamps": wio.timestamps_csv_name(name)})
    meta = {"schema": wio.SESSION_SCHEMA, "t0": T0, "t0_unix": T0 + OFFSET,
            "clock_offset_unix": OFFSET, "has_wii": False, "has_video": True,
            "camera_names": list(cams), "streams": streams, "duration_s": 5.0}
    wio.write_session_json(folder, meta)
    return folder


def regular(start, n, fps=30.0, jitter_ms=0.0, seed=0, drop=()):
    rng = np.random.default_rng(seed)
    t = start + np.arange(n) / fps + rng.uniform(-jitter_ms, jitter_ms, n) / 1000.0
    keep = np.ones(n, bool)
    keep[list(drop)] = False
    return np.round(np.sort(t[keep]), 6)  # as stored in the CSV (%.6f)


def nearest(tr, t):
    return int(np.argmin(np.abs(np.asarray(tr) - t)))


@pytest.fixture
def take3(project):
    """A 3-camera take: different start times, jitter, dropped frames (cam03)."""
    cams = {"cam01": regular(0.10, 90, jitter_ms=4, seed=1),
            "cam02": regular(0.25, 90, jitter_ms=4, seed=2),
            "cam03": regular(0.05, 95, jitter_ms=4, seed=3, drop=range(40, 43))}
    folder = write_take(WiiPaths(project).recording_dir("20260924_120000"), cams)
    return project, folder, cams


# ------------------------------------------------------------------------- settings
def test_capture_settings_defaults_and_roundtrip(project):
    s = load_capture_settings(project)
    assert list(s.cameras) == ["cam01", "cam02", "cam03"]
    assert [c.source for c in s.cameras.values()] == [0, 1, 2]
    s.cameras["cam02"] = CameraSetting("C:/videos/x.mp4", 1280, 720, 30.0, False)
    s.output_fps, s.keep_raw, s.subject = 25, False, "S01"
    p = save_capture_settings(project, s)
    assert p == WiiPaths(project).capture_json
    back = load_capture_settings(project)
    assert back.to_dict() == s.to_dict()
    project.config.num_cameras = 2  # cameras beyond the project are dropped
    assert list(load_capture_settings(project).cameras) == ["cam01", "cam02"]
    assert default_settings(1).cameras["cam01"].source == 0


def test_capture_latency_setting(project):
    s = load_capture_settings(project)
    assert s.latency_ms == 0.0 and s.latency_s(["cam01"]) == {"cam01": 0.0}
    s.latency_ms = 60.0
    s.cameras["cam02"].latency_ms = 95.0  # per-camera override
    save_capture_settings(project, s)
    back = load_capture_settings(project)
    assert back.latency_ms == 60.0 and back.cameras["cam02"].latency_ms == 95.0
    assert "latency_ms" not in back.cameras["cam01"].to_dict()  # only written when set
    assert back.latency_s(["cam01", "cam02"]) == pytest.approx({"cam01": 0.060, "cam02": 0.095})
    assert CaptureSettings.from_dict({"latency_ms": "junk"}).latency_ms == 0.0


def test_export_subtracts_the_camera_latency(take3):
    """Frames are stamped when read() returns: the export moves them back by the capture
    latency, so frames.csv (and the recorded alignment) hold exposure times."""
    project, folder, cams = take3
    ref = plan_export(folder, fps=30)
    lat = {"cam01": 0.040, "cam02": 0.040, "cam03": 0.040}
    plan = plan_export(folder, fps=30, latency_s=lat)
    np.testing.assert_allclose(plan.t_rel, ref.t_rel - 0.040, atol=1e-9)
    for c in cams:
        np.testing.assert_array_equal(plan.src[c], ref.src[c])  # same frames, earlier times
    assert plan_export(folder, fps=30, latency_s=0.040).t_rel[0] == pytest.approx(plan.t_rel[0])
    export_recording_to_project(project, folder.name, fps=30, latency_s=lat)
    fr = wio.read_frames_csv(folder)
    np.testing.assert_allclose(fr["t_rel"], ref.t_rel - 0.040, atol=1e-6)
    np.testing.assert_allclose(fr["t"], ref.t_rel - 0.040 + T0, atol=1e-6)
    assert wio.read_session_json(folder)["export"]["latency_ms"] == \
        {"cam01": 40.0, "cam02": 40.0, "cam03": 40.0}
    info = load_trial(project)
    assert info.alignment.offset_s == pytest.approx(ref.t_rel[0] - 0.040, abs=1e-6)
    assert info.alignment.details["latency_ms"]["cam01"] == 40.0


# ------------------------------------------------------------------------------ plan
def test_plan_grid_latest_start_jitter_and_dropped_frames(take3):
    project, folder, cams = take3
    plan = plan_export(folder, fps=30)
    start = max(v[0] for v in cams.values())
    end = min(v[-1] for v in cams.values())
    assert plan.t_rel[0] == pytest.approx(cams["cam02"][0]) == pytest.approx(start)
    assert plan.t_rel[-1] <= end + 1e-9 and plan.t_rel[-1] + 1 / 30 > end
    assert plan.n_frames == int(np.floor((end - start) * 30 + 1e-6)) + 1
    np.testing.assert_allclose(np.diff(plan.t_rel), 1 / 30)
    np.testing.assert_allclose(plan.t_grid, plan.t_rel + T0)
    np.testing.assert_allclose(plan.t_unix, plan.t_rel + T0 + OFFSET)
    for c, tr in cams.items():
        want = [nearest(tr, t) for t in plan.t_rel]
        np.testing.assert_array_equal(plan.src[c], want)
        np.testing.assert_allclose(plan.dt_ms[c], (tr[want] - plan.t_rel) * 1000, atol=1e-6)
        if c != "cam03":
            assert plan.max_dt_ms(c) <= 1000 / 30  # within one frame period
        assert plan.sizes[c] == SIZE
    assert plan.max_dt_ms("cam03") > 1000 / 30  # 3 dropped frames: a gap of 4 periods
    assert any(w.startswith("cam03: frames are missing") for w in plan.warnings)
    assert plan.src["cam02"][0] == 0  # the latest-starting camera begins with its first frame
    # the grid is in phase with cam02 (jitter < half a period): every frame used once
    np.testing.assert_array_equal(plan.src["cam02"], np.arange(plan.n_frames))
    assert plan.duplicated("cam02") == plan.skipped("cam02") == 0
    for c in cams:
        s = plan.src[c]
        assert plan.duplicated(c) == int(np.sum(s[1:] == s[:-1]))
        assert plan.skipped(c) == int(s[-1] - s[0] + 1 - len(set(s.tolist())))
    assert plan.duplicated("cam03") >= 2  # the dropped frames are filled with duplicates
    assert set(plan.cameras) == set(cams) and plan.measured_fps["cam01"] == pytest.approx(30, 0.05)


def test_plan_fps_choice_and_warnings(project):
    rec = WiiPaths(project).recording_dir("t")
    cams = {"cam01": regular(0.0, 100, fps=30.0),
            "cam02": regular(0.0, 83, fps=25.0, drop=range(30, 45))}
    write_take(rec, cams, raw=False)
    plan = plan_export(rec)  # default: the slowest camera's measured rate
    assert plan.fps == 25
    plan = plan_export(rec, fps=30)
    text = "\n".join(plan.warnings)
    assert "cam02 delivered only 25.0 fps" in text
    assert "cam02: frames are missing for up to" in text
    assert "raw video of cam01 is missing" in text
    assert plan.videos["cam01"] is None and plan.sizes["cam01"] == (0, 0)
    assert "cam01" not in text.split("raw video")[0]  # cam01 itself is fine
    only = plan_export(rec, cams=["cam01"])
    assert only.cameras == ["cam01"] and only.fps == 30
    with pytest.raises(ValueError, match="cam07"):
        plan_export(rec, cams=["cam07"])


def test_plan_errors(project):
    paths = WiiPaths(project)
    wii_only = paths.recording_dir("wii_only")
    wii_only.mkdir(parents=True)
    wio.write_session_json(wii_only, {"schema": wio.SESSION_SCHEMA, "t0": T0, "streams": []})
    with pytest.raises(ValueError, match="no camera video"):
        plan_export(wii_only)
    apart = write_take(paths.recording_dir("apart"),
                       {"cam01": regular(0.0, 30), "cam02": regular(5.0, 30)}, raw=False)
    with pytest.raises(ValueError, match="do not overlap"):
        plan_export(apart)
    one = write_take(paths.recording_dir("one"), {"cam01": [0.5]}, raw=False)
    with pytest.raises(ValueError, match="fewer than 2 frames"):
        plan_export(one)


def test_take_cameras_without_session_json(tmp_path):
    folder = write_take(tmp_path / "rec", {"cam01": regular(0, 10), "cam02": regular(0, 10)})
    (folder / wio.SESSION_JSON).unlink()
    tc = take_cameras(folder)
    assert sorted(tc) == ["cam01", "cam02"]
    assert tc["cam01"][0] == folder / "cam01.mkv"
    plan = plan_export(folder, fps=30)  # t0 recovered from t - t_rel
    np.testing.assert_allclose(plan.t_grid, plan.t_rel + T0, atol=1e-6)


# ---------------------------------------------------------------------------- export
def test_export_writes_videos_frames_csv_project_fps_and_trial(take3):
    project, folder, cams = take3
    assert project.config.frame_rate == 30
    project.config.frame_rate = 60
    project.save()
    vdir = project.videos_dir
    (vdir / "cam01.avi").write_bytes(b"old avi")      # other extension: must go
    (vdir / "cam02.mp4").write_bytes(b"old mp4")      # replaced
    (vdir / "cam03.MOV").write_bytes(b"old mov")      # other extension, upper case
    (vdir / "notes.txt").write_text("not a video")    # not ours: kept
    calls = []
    res = export_recording_to_project(project, folder.name, fps=30,
                                      progress=lambda d, t, m: calls.append((d, t, m)))
    plan = plan_export(folder, fps=30)
    m = plan.n_frames
    assert res.n_frames == m and res.fps == 30 and res.fps_changed
    assert sorted(p.name for p in vdir.iterdir()) == ["cam01.mp4", "cam02.mp4", "cam03.mp4",
                                                       "notes.txt"]
    for c in cams:
        idx = read_indices(vdir / f"{c}.mp4")
        assert len(idx) == m  # every camera has the same frame count
        np.testing.assert_array_equal(idx, plan.src[c])  # nearest source frame, in order
    assert calls and calls[-1][0] == calls[-1][1] and all(d <= t for d, t, _ in calls)
    assert not (folder / export_mod.EXPORT_TMP).exists()

    # frames.csv: one row per exported frame, the grid times and per-camera choices
    fr = wio.read_frames_csv(folder)
    assert list(fr)[:4] == wio.FRAMES_HEADER_BASE
    np.testing.assert_array_equal(fr["frame"], np.arange(m))
    np.testing.assert_allclose(fr["t_rel"], plan.t_rel, atol=1e-6)
    np.testing.assert_allclose(fr["t"], plan.t_rel + T0, atol=1e-6)
    np.testing.assert_allclose(fr["t_unix"], plan.t_rel + T0 + OFFSET, atol=1e-5)
    for c in cams:
        np.testing.assert_array_equal(fr[f"{c}_src"], plan.src[c])
        np.testing.assert_allclose(fr[f"{c}_dt_ms"], plan.dt_ms[c], atol=1e-5)

    # project fps + Config.toml follow the exported videos
    assert project.config.frame_rate == 30
    toml = tomllib.loads(project.project_file.read_text(encoding="utf-8"))
    assert toml["frame_rate"] == 30
    cfg = tomllib.loads(project.config_file.read_text(encoding="utf-8"))
    assert cfg["project"]["frame_rate"] == 30

    # trial.json: this take, recorded alignment, fingerprint of the new videos
    info = load_trial(project)
    assert info.recording == folder.name and info.source == "capture"
    assert info.alignment.method == "recorded" and info.alignment.frames_csv == wio.FRAMES_CSV
    assert info.alignment.offset_s == pytest.approx(plan.t_rel[0], abs=1e-6)
    assert info.videos_fingerprint == videos_fingerprint(project)
    assert not videos_changed(project, info)

    # session.json "export"; raw videos kept (keep_raw default)
    meta = wio.read_session_json(folder)
    ex = meta["export"]
    assert ex["fps"] == 30 and ex["n_frames"] == m and ex["frames_csv"] == wio.FRAMES_CSV
    assert ex["videos"] == ["videos/cam01.mp4", "videos/cam02.mp4", "videos/cam03.mp4"]
    assert set(ex["max_dt_ms"]) == set(cams) and ex["keep_raw"] is True
    assert ex["duplicated"] == res.duplicated and ex["skipped"] == res.skipped
    assert all((folder / f"{c}.mkv").is_file() for c in cams)


def test_exported_frames_map_to_wii_time(take3):
    """.trc Frame# f -> Wii t_rel through frames.csv (CORE-BOARD's alignment reader)."""
    from poseassess.core.balance.alignment import alignment_status, trc_rows_to_wii_time

    project, folder, _ = take3
    export_recording_to_project(project, folder.name, fps=30)
    plan = plan_export(folder, fps=30)
    frames = np.array([0, 5, 17, plan.n_frames - 1])
    t = trc_rows_to_wii_time(project, frames, frames / 30.0)
    np.testing.assert_allclose(t, plan.t_rel[frames], atol=1e-6)
    level, _text = alignment_status(project)
    assert level != "none"


def test_export_keep_raw_false_deletes_raw_videos_only(take3):
    project, folder, cams = take3
    export_recording_to_project(project, folder.name, fps=30, keep_raw=False)
    for c in cams:
        assert not (folder / f"{c}.mkv").exists()
        assert (folder / wio.timestamps_csv_name(c)).is_file()  # timestamps always kept
    assert wio.read_session_json(folder)["export"]["keep_raw"] is False
    with pytest.raises(ValueError, match="raw videos .* are missing"):
        export_recording_to_project(project, folder.name, fps=30)


def test_export_without_project_update_keeps_fps(take3):
    project, folder, _ = take3
    project.config.frame_rate = 60
    res = export_recording_to_project(project, folder.name, fps=30, update_project=False)
    assert not res.fps_changed and project.config.frame_rate == 60


def test_cancel_leaves_videos_untouched(take3):
    project, folder, _ = take3
    old = project.videos_dir / "cam01.mp4"
    old.write_bytes(b"previous trial video")
    n = [0]

    def cancel():
        n[0] += 1
        return n[0] > 40  # in the middle of cam02

    with pytest.raises(ExportCancelled):
        export_recording_to_project(project, folder.name, fps=30, cancel=cancel)
    assert sorted(p.name for p in project.videos_dir.iterdir()) == ["cam01.mp4"]
    assert old.read_bytes() == b"previous trial video"
    assert not (folder / export_mod.EXPORT_TMP).exists()
    assert not (folder / wio.FRAMES_CSV).exists()
    assert not WiiPaths(project).trial_json.exists()
    assert "export" not in wio.read_session_json(folder)


def test_failed_replace_restores_the_old_videos(take3, monkeypatch):
    project, folder, _ = take3
    vdir = project.videos_dir
    for c, ext in (("cam01", "mp4"), ("cam02", "avi"), ("cam03", "mp4")):
        (vdir / f"{c}.{ext}").write_bytes(f"old {c}".encode())
    before = {p.name: p.read_bytes() for p in vdir.iterdir()}
    real = export_mod.os.replace

    def replace(a, b):
        if Path(b).parent == vdir and Path(b).name == "cam02.mp4":
            raise PermissionError(13, "The process cannot access the file")
        return real(a, b)

    monkeypatch.setattr(export_mod.os, "replace", replace)
    with pytest.raises(RuntimeError, match="Cannot replace the videos"):
        export_recording_to_project(project, folder.name, fps=30)
    assert {p.name: p.read_bytes() for p in vdir.iterdir()} == before
    assert not (folder / export_mod.EXPORT_TMP).exists()
    assert not WiiPaths(project).trial_json.exists() and project.config.frame_rate == 30


def test_export_refuses_incomplete_takes(project):
    paths = WiiPaths(project)
    two = write_take(paths.recording_dir("two"), {"cam01": regular(0, 30),
                                                  "cam02": regular(0, 30)})
    with pytest.raises(ValueError, match="no video of cam03"):
        export_recording_to_project(project, two.name)
    with pytest.raises(ValueError, match="no take"):
        export_recording_to_project(project, "does_not_exist")
    assert not any(project.videos_dir.iterdir())


def test_export_warnings_calib_size_and_short_raw_video(project):
    write_calib_toml(project.calib_toml, ring_cameras(3))  # 1280x720 cameras
    cams = {c: regular(0.0, 40) for c in ("cam01", "cam02", "cam03")}
    folder = write_take(WiiPaths(project).recording_dir("short"), cams,
                        video_frames={"cam02": 30})
    res = export_recording_to_project(project, folder.name, fps=30)
    text = "\n".join(res.warnings)
    assert "cam01 was recorded at 320x240 but Calib.toml is for 1280x720" in text
    assert "cam02: the raw video ended" in text
    idx = read_indices(project.videos_dir / "cam02.mp4")
    assert len(idx) == res.n_frames and idx[-1] == 29  # last frame repeated


# --------------------------------------------------------------------------- session
def file_settings(tmp_path, n=2, fps=30.0, frames=60):
    return CaptureSettings({f"cam{i:02d}": CameraSetting(str(write_test_video(
        tmp_path / f"src{i}.avi", n_frames=frames, fps=fps))) for i in range(1, n + 1)})


def wait_frames(session, cams, timeout=5.0):
    return wait_until(lambda: all(session.latest(c) is not None for c in cams), timeout)


@pytest.fixture
def project2(project):
    project.config.num_cameras = 2
    project.save()
    return project


def test_session_records_a_take_and_exports_it(project2, tmp_path):
    project = project2
    s = CaptureSession(project, file_settings(tmp_path))
    sim = SimulatedBoard()
    sim.start()
    try:
        assert s.open() == {}
        assert wait_frames(s, ["cam01", "cam02"])
        assert s.video_take_problems() == [] and not s.recording and s.elapsed_s == 0.0
        folder = s.start_recording(force=sim, subject="S 01", notes="quiet stance")
        assert s.recording and s.folder == folder
        assert folder.parent == WiiPaths(project).recordings_dir and folder.name.endswith("S_01")
        with pytest.raises(RuntimeError, match="Already recording"):
            s.start_recording(force=sim)
        with pytest.raises(RuntimeError, match="while recording"):
            s.open()
        st = s.camera_status()
        assert st["cam01"]["recording"] and st["cam01"]["running"]
        assert st["cam01"]["size"] == (320, 240)
        time.sleep(0.4)
        ev = s.add_event("sync")
        assert ev is not None and ev["label"] == "sync"
        time.sleep(1.2)
        assert s.elapsed_s > 1.5
        assert s.stop_recording() == folder and not s.recording
        assert s.stop_recording() is None and s.folder == folder
        assert all(s.latest(c) is not None for c in ("cam01", "cam02"))  # still previewing
    finally:
        sim.stop()
        s.close()
    assert not any(st.running for st in s.streams.values()) and s.streams == {}

    names = sorted(p.name for p in folder.iterdir())
    assert names == ["cam01.mkv", "cam01_timestamps.csv", "cam02.mkv", "cam02_timestamps.csv",
                     "events.csv", "session.json", "wii.csv"]
    meta = wio.read_session_json(folder)
    assert meta["camera_names"] == ["cam01", "cam02"] and meta["has_video"] and meta["has_wii"]
    assert meta["subject"] == "S 01" and meta["notes"] == "quiet stance"
    assert meta["project"]["root"] == str(project.root)
    assert meta["capture"]["settings"]["cameras"]["cam01"]["source"].endswith("src1.avi")
    assert meta["capture"]["calib_sizes"] == {}  # no Calib.toml in this project
    assert meta["capture"]["frame_sizes"] == {"cam01": [320, 240], "cam02": [320, 240]}
    assert meta["samples"]["wii"] > 50 and meta["samples"]["events"] == 1
    for c in ("cam01", "cam02"):
        ts = wio.read_timestamps_csv(folder / wio.timestamps_csv_name(c))
        cap = cv2.VideoCapture(str(folder / f"{c}.mkv"))
        n = 0
        while cap.read()[0]:
            n += 1
        cap.release()
        assert n == len(ts["frame"]) > 20
        np.testing.assert_allclose(ts["t_rel"], ts["t"] - meta["t0"], atol=2e-6)
    evs = wio.read_events_csv(folder)
    assert [e["label"] for e in evs] == ["sync"] and 0.3 < evs[0]["t_rel"] < 1.0
    wii = wio.read_wii_csv(folder)
    assert np.nanmean(wii["total_kg"]) == pytest.approx(70, abs=3)

    res = export_recording_to_project(project, folder.name, fps=30)
    assert res.n_frames > 30
    for c in ("cam01", "cam02"):
        assert (project.videos_dir / f"{c}.mp4").is_file()
    assert load_trial(project).alignment.method == "recorded"


def test_session_open_is_idempotent_and_reports_errors(project2, tmp_path):
    settings = file_settings(tmp_path)
    settings.cameras["cam02"].source = str(tmp_path / "missing.avi")
    s = CaptureSession(project2, settings)
    try:
        errors = s.open()
        assert list(errors) == ["cam02"] and "Cannot open camera" in errors["cam02"]
        first = s.streams["cam01"]
        assert s.open().keys() == {"cam02"} and s.streams["cam01"] is first  # kept
        other = write_test_video(tmp_path / "other.avi", n_frames=20)
        settings.cameras["cam02"].source = str(other)
        settings.cameras["cam01"].enabled = False
        assert s.open() == {}
        assert "cam01" not in s.streams and not first.running  # disabled -> stopped
        assert s.streams["cam02"].source == str(other)
        assert s.latest("cam01") is None
        assert wait_frames(s, ["cam02"])
        problems = s.video_take_problems()
        assert problems == ["cam01 is not enabled"]
        with pytest.raises(RuntimeError, match="cam01 is not enabled"):
            s.start_recording()
    finally:
        s.close()


def test_wii_only_take_while_cameras_preview(project2, tmp_path):
    s = CaptureSession(project2, file_settings(tmp_path))
    sim = SimulatedBoard()
    sim.start()
    try:
        s.open()
        assert wait_frames(s, ["cam01", "cam02"])
        with pytest.raises(RuntimeError, match="Nothing to record"):
            s.start_recording(force=None, record_cameras=False)
        folder = s.start_recording(force=sim, record_cameras=False)
        time.sleep(0.5)
        s.stop_recording()
    finally:
        sim.stop()
        s.close()
    names = sorted(p.name for p in folder.iterdir())
    assert names == ["session.json", "wii.csv"]
    meta = wio.read_session_json(folder)
    assert meta["has_video"] is False and meta["capture"]["record_cameras"] is False
    with pytest.raises(ValueError, match="no camera video"):
        plan_export(folder)


def test_attach_force_and_camera_lost_events(project2, tmp_path, monkeypatch):
    s = CaptureSession(project2, file_settings(tmp_path))
    try:
        s.open()
        assert wait_frames(s, ["cam01", "cam02"])
        folder = s.start_recording(force=None)
        sim = SimulatedBoard()
        sim.start()
        s.attach_force(sim)  # the board connected during the take
        s.log_force_source(sim, "connected")
        assert wait_until(lambda: s.recorder.counts["wii"] > 20)
        cam = s.streams["cam02"]
        monkeypatch.setattr(cam, "frame_age", lambda now=None: 10.0)
        assert s.check_cameras() == {"cam02"}
        assert s.check_cameras() == {"cam02"}  # logged once
        monkeypatch.setattr(cam, "frame_age", lambda now=None: 0.01)
        assert s.check_cameras() == set()
        s.stop_recording()
        sim.stop()
    finally:
        s.close()
    labels = [e["label"] for e in wio.read_events_csv(folder)]
    assert labels == ["camera_lost cam02", "camera_recovered cam02"]
    meta = wio.read_session_json(folder)
    assert meta["has_wii"] is True
    assert [h["reason"] for h in meta["force_source_history"]] == ["attached", "connected"]


def test_board_snapshots_and_size_warnings(project2, tmp_path):
    write_calib_toml(project2.calib_toml, ring_cameras(2))  # 1280x720 cameras
    s = CaptureSession(project2, file_settings(tmp_path))
    try:
        s.open()
        assert wait_frames(s, ["cam01", "cam02"])
        files = s.save_board_snapshots()
        warn = s.size_warnings()
        assert s.calib_sizes() == {"cam01": [1280, 720], "cam02": [1280, 720]}
    finally:
        s.close()
    paths = WiiPaths(project2)
    assert [f.parent for f in files] == [paths.board_dir / "cam01", paths.board_dir / "cam02"]
    for f in files:
        assert f.name.startswith("snapshot_") and f.suffix == ".jpg"
        assert cv2.imread(str(f)).shape == (240, 320, 3)
    assert len(warn) == 2 and "cam01 delivers 320x240 but Calib.toml is for 1280x720" in warn[0]


def test_probe_cameras_with_fake_devices(monkeypatch):
    class Cap:
        def __init__(self, index):
            self.index = index
            self.released = False

        def isOpened(self):
            return self.index in (0, 2, 3)

        def read(self):
            if self.index == 2:
                time.sleep(1.0)  # a stalled device
            return True, np.zeros((480, 640, 3), np.uint8)

        def get(self, prop):
            return 30.0 if prop == cv2.CAP_PROP_FPS else 0.0

        def release(self):
            self.released = True

    opened = []

    def open_capture(index, *a, **k):
        opened.append(index)
        return Cap(index)

    monkeypatch.setattr(camera_mod, "open_capture", open_capture)
    t = time.perf_counter()
    found = probe_cameras(max_index=5, timeout_s=0.3, skip={3})
    assert time.perf_counter() - t < 1.5
    assert found == [{"index": 0, "width": 640, "height": 480, "fps": 30.0}]
    assert 3 not in opened and opened == [0, 1, 2, 4]


def test_probe_cameras_without_devices_is_empty():
    assert isinstance(probe_cameras(max_index=1, timeout_s=1.0), list)
