"""CORE-WII: python -m poseassess.wii (simulate, --project writes into wii/recordings, auto-connect
with tare via FakeBus, wait timeout + --list, exit codes, device by number, Enter before recording
ignored, events from stdin, connection events, close handlers restored). Port of the CLI tests of
PoseBoard 3ea6d8a tests/test_wii_auto.py and tests/test_runtime.py."""

import io
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from poseassess.wii import device as device_mod
from poseassess.wii import io as wio
from poseassess.wii import record_cli
from poseassess.wii.device import HIDAPI_MISSING, HidapiUnavailable, SimulatedBoard
from poseassess.wii.record_cli import main as record_main
from tests.wii_fakes import FakeBus, FakeHid, wait_until

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _no_real_hid_leak():
    """Never leave the real hidapi module imported (the app smoke test checks it is not)."""
    had = "hid" in sys.modules
    yield
    if not had:
        sys.modules.pop("hid", None)


def only_session(root):
    folders = [p for p in Path(root).iterdir() if p.is_dir()]
    assert len(folders) == 1, folders
    return folders[0]


def test_exit_codes_and_parser():
    assert (record_cli.EXIT_OK, record_cli.EXIT_NO_BOARD, record_cli.EXIT_ERROR,
            record_cli.EXIT_NO_SAMPLES, record_cli.EXIT_INTERRUPTED) == (0, 2, 3, 4, 130)
    ap = record_cli.build_parser()
    a = ap.parse_args(["--project", "p", "--seconds", "2", "--device", "0", "--min-kg", "3"])
    assert a.project == "p" and a.out is None and a.seconds == 2 and a.min_kg == 3.0
    assert ap.parse_args([]).min_kg == 5.0 and ap.parse_args([]).tare == 0.0
    with pytest.raises(SystemExit):
        ap.parse_args(["--project", "p", "--out", "o"])  # mutually exclusive


def test_cli_simulate(tmp_path, capsys):
    assert record_main(["--simulate", "--seconds", "1", "--out", str(tmp_path), "--subject", "cli",
                        "--notes", "wii only"]) == 0
    folder = only_session(tmp_path)
    assert folder.name.endswith("_cli")
    assert str(folder) in capsys.readouterr().out
    wii = wio.read_wii_csv(folder)
    assert len(wii["t"]) > 50
    assert 60 < np.nanmean(wii["total_kg"]) < 80
    meta = json.loads((folder / "session.json").read_text(encoding="utf-8"))
    assert meta["schema"] == wio.SESSION_SCHEMA
    assert meta["force_source"]["type"] == "SimulatedBoard" and meta["notes"] == "wii only"
    assert meta["board_geometry"]["sensor_dx_mm"] == 433.0 and meta["board_pose"] is None
    assert meta["recorded_with"] == "python -m poseassess.wii" and "project" not in meta
    assert not list(folder.glob("*.mp4")) and not list(folder.glob("*.mkv"))
    assert not (folder / "events.csv").exists()


def test_cli_project_records_into_wii_recordings(project, capsys):
    from poseassess.core.balance.paths import WiiPaths

    assert record_main(["--simulate", "--seconds", "0.5", "--project", str(project.root),
                        "--subject", "S01"]) == 0
    paths = WiiPaths(project)
    (rec_id,) = paths.list_recordings()
    assert rec_id.endswith("_S01")
    folder = paths.recording_dir(rec_id)
    meta = wio.read_session_json(folder)
    assert meta["project"] == {"name": "test", "root": str(Path(project.root).resolve())}
    assert meta["samples"]["wii"] > 20
    assert not paths.trial_json.exists()  # the active trial is not changed
    out = capsys.readouterr().out
    assert "Balance (Wii)" in out and rec_id in out
    # nothing else is written into the project (Pose2Sim folders untouched)
    assert sorted(p.name for p in (project.root / "wii").iterdir()) == ["recordings"]
    assert not any((project.root / "videos").iterdir())


def test_cli_project_uses_the_board_registration(demo_trial):
    from poseassess.core.balance.board import load_board
    from poseassess.core.balance.trial import load_trial

    proj = demo_trial["project"]
    trial_before = load_trial(proj).to_dict()
    reg = load_board(proj)
    assert reg is not None and not reg.stale
    assert record_main(["--simulate", "--seconds", "0.4", "--project", str(proj.root)]) == 0
    from poseassess.core.balance.paths import WiiPaths

    new = [r for r in WiiPaths(proj).list_recordings() if r != demo_trial["rec_id"]]
    (rec_id,) = new
    folder = WiiPaths(proj).recording_dir(rec_id)
    meta = wio.read_session_json(folder)
    assert meta["board_pose"]["method"] == reg.pose.method
    wii = wio.read_wii_csv(folder)
    ok = np.isfinite(wii["cop_x_board"])
    assert ok.any()
    cop_w = reg.pose.board_to_world.apply(
        np.c_[wii["cop_x_board"][ok], wii["cop_y_board"][ok], np.zeros(ok.sum())])
    np.testing.assert_allclose(np.c_[wii["cop_x_world"][ok], wii["cop_y_world"][ok],
                                     wii["cop_z_world"][ok]], cop_w, atol=1e-5)
    assert load_trial(proj).to_dict() == trial_before


def test_cli_project_with_stale_board(demo_trial, capsys):
    proj = demo_trial["project"]
    calib = next(iter(sorted((proj.root / "calibration").glob("Calib*.toml"))))
    calib.write_text(calib.read_text() + "\n# recalibrated\n")
    assert record_main(["--simulate", "--seconds", "0.3", "--project", str(proj.root)]) == 0
    out = capsys.readouterr().out
    assert "calibration changed" in out and "COP world columns stay empty" in out
    from poseassess.core.balance.paths import WiiPaths

    rec_id = [r for r in WiiPaths(proj).list_recordings() if r != demo_trial["rec_id"]][0]
    folder = WiiPaths(proj).recording_dir(rec_id)
    assert wio.read_session_json(folder)["board_pose"] is None
    assert np.all(np.isnan(wio.read_wii_csv(folder)["cop_x_world"]))


def test_cli_project_must_be_a_project(tmp_path, capsys):
    assert record_main(["--simulate", "--seconds", "0.2", "--project", str(tmp_path)]) == 3
    assert "Not a PoseAssess project" in capsys.readouterr().out
    assert not (tmp_path / "wii").exists()


def test_cli_auto_connect_with_tare(tmp_path, monkeypatch):
    bus = FakeBus(monkeypatch)
    bus.present = True
    assert record_main(["--seconds", "0.5", "--tare", "0.2", "--out", str(tmp_path)]) == 0
    folder = only_session(tmp_path)
    wii = wio.read_wii_csv(folder)
    assert len(wii["t"]) > 20
    np.testing.assert_allclose(wii["total_kg"], 0.0, atol=1e-6)  # tared before recording
    meta = json.loads((folder / "session.json").read_text(encoding="utf-8"))
    assert meta["force_source"]["type"] == "WiiAutoConnect"
    np.testing.assert_allclose(meta["force_source"]["tare_kg"], 8.5)
    assert bus.opened[-1].closed  # the board is released at the end


def test_cli_wait_timeout_and_list(tmp_path, monkeypatch, capsys):
    FakeBus(monkeypatch)  # no board present
    t = time.monotonic()
    assert record_main(["--wait", "0.3", "--out", str(tmp_path / "rec")]) == 2
    assert time.monotonic() - t < 3.0
    assert not (tmp_path / "rec").exists()
    assert "PIN empty" in capsys.readouterr().out  # pairing hint
    assert record_main(["--list"]) == 0
    assert "No Wii Balance Board found" in capsys.readouterr().out


def test_cli_list_shows_boards(monkeypatch, capsys):
    bus = FakeBus(monkeypatch)
    bus.present = True
    assert record_main(["--list"]) == 0
    out = capsys.readouterr().out
    assert "[0] Nintendo RVL-WBC-01" in out and "path=fake-board-path" in out


def test_cli_without_hidapi(tmp_path, monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "hid", None)  # "import hid" raises ImportError
    assert record_main(["--list"]) == 2
    assert HIDAPI_MISSING in capsys.readouterr().out
    t = time.monotonic()
    assert record_main(["--out", str(tmp_path / "x")]) == 2  # fails at once, no waiting forever
    assert time.monotonic() - t < 5.0
    assert "pip install hidapi" in capsys.readouterr().out
    assert not (tmp_path / "x").exists()
    # the simulator does not need hidapi
    assert record_main(["--simulate", "--seconds", "0.2", "--out", str(tmp_path / "s")]) == 0


def test_cli_ignores_enter_before_recording(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "stdin", io.StringIO("\nlabel typed too early\n"))
    assert record_main(["--simulate", "--seconds", "0.8", "--tare", "0.3", "--out", str(tmp_path)]) == 0
    folder = only_session(tmp_path)
    meta = json.loads((folder / "session.json").read_text(encoding="utf-8"))
    assert meta["duration_s"] > 0.7 and meta["samples"]["wii"] > 30
    assert meta["samples"]["events"] == 0
    assert meta["force_source"]["min_total_kg"] == 5.0


def test_cli_events_from_stdin_and_enter_stops(tmp_path, monkeypatch, capsys):
    r, w = os.pipe()
    monkeypatch.setattr(sys, "stdin", os.fdopen(r, "r", encoding="utf-8"))
    out = tmp_path / "rec"

    def typist():
        assert wait_until(lambda: out.exists() and any(out.iterdir()), timeout=10)
        time.sleep(0.3)
        os.write(w, "sync\n".encode())
        time.sleep(0.2)
        os.write(w, "  eyes   closed \u95ed\u773c \n".encode())
        time.sleep(0.2)
        os.write(w, b"\n")  # empty line: stop

    th = threading.Thread(target=typist, daemon=True)
    th.start()
    t = time.monotonic()
    try:
        assert record_main(["--simulate", "--seconds", "20", "--out", str(out)]) == 0
    finally:
        os.close(w)
    assert time.monotonic() - t < 10
    folder = only_session(out)
    evs = wio.read_events_csv(folder)
    assert [e["label"] for e in evs] == ["sync", "eyes closed \u95ed\u773c"]
    assert 0.2 < evs[0]["t_rel"] < evs[1]["t_rel"]
    assert "Event 'sync' at t_rel=" in capsys.readouterr().out
    meta = wio.read_session_json(folder)
    assert meta["samples"]["events"] == 2 and meta["duration_s"] < 10


def test_cli_logs_disconnect_and_reconnect(tmp_path, monkeypatch):
    bus = FakeBus(monkeypatch)
    bus.present = True
    monkeypatch.setattr(device_mod.WiiAutoConnect, "__init__",
                        _with_defaults(device_mod.WiiAutoConnect.__init__, poll_interval_s=0.05))
    out = tmp_path / "rec"
    done = {}

    def drop_and_return():
        assert wait_until(lambda: out.exists() and any(out.iterdir()) and bus.opened, timeout=10)
        time.sleep(0.3)
        bus.present = False
        bus.opened[-1].broken = True  # Bluetooth drop
        time.sleep(0.4)
        bus.next_device = lambda: FakeHid([2700, 2800, 2900, 3000])  # 17 kg per sensor
        bus.present = True
        done["ok"] = True

    th = threading.Thread(target=drop_and_return, daemon=True)
    th.start()
    assert record_main(["--seconds", "2.0", "--out", str(out)]) == 0
    th.join(5)
    assert done.get("ok")
    folder = only_session(out)
    labels = [e["label"] for e in wio.read_events_csv(folder)]
    assert labels == ["wii_disconnected", "wii_connected"]
    meta = wio.read_session_json(folder)
    assert [h["reason"] for h in meta["force_source_history"]] == ["start", "connected"]
    wii = wio.read_wii_csv(folder)
    assert np.nanmax(np.diff(wii["t_rel"])) > 0.3  # the gap is visible in wii.csv
    np.testing.assert_allclose(wii["total_kg"][-1], 68.0)
    assert meta["force_source"]["disconnects"] == 1 and meta["force_source"]["connections"] == 2


def _with_defaults(init, **defaults):
    def wrapped(self, *a, **kw):
        for k, v in defaults.items():
            kw.setdefault(k, v)
        init(self, *a, **kw)

    return wrapped


def test_cli_exit_codes(tmp_path, monkeypatch, capsys):
    # Taring fails: defined exit code, nothing recorded
    def no_data(self, seconds=1.0):
        raise RuntimeError("No data available for taring")

    with monkeypatch.context() as m:
        m.setattr(SimulatedBoard, "do_tare", no_data)
        assert record_main(["--simulate", "--tare", "0.1", "--out", str(tmp_path / "a")]) == 3
    assert not (tmp_path / "a").exists()
    assert "Taring failed" in capsys.readouterr().out

    # hidapi unusable: fails at once instead of waiting forever
    def missing():
        raise HidapiUnavailable(HIDAPI_MISSING)

    with monkeypatch.context() as m:
        m.setattr(device_mod.BalanceBoardHID, "list_devices", staticmethod(missing))
        t = time.monotonic()
        assert record_main(["--out", str(tmp_path / "b")]) == 2
        assert time.monotonic() - t < 5.0
    assert "hidapi is not installed" in capsys.readouterr().out

    # A board that stops sending before the recording: no samples -> exit code 4
    def one_sample(self):
        self._set_state("connected")
        self._emit_kg(time.perf_counter(), np.array([20.0, 20, 20, 20]))
        while not self._stop.is_set():
            time.sleep(0.01)

    with monkeypatch.context() as m:
        m.setattr(SimulatedBoard, "_run", one_sample)
        assert record_main(["--simulate", "--seconds", "0.3", "--min-kg", "2",
                            "--out", str(tmp_path / "c")]) == 4
    meta = json.loads((only_session(tmp_path / "c") / "session.json").read_text(encoding="utf-8"))
    assert meta["force_source"]["min_total_kg"] == 2.0

    # The output folder cannot be created: error before recording
    blocker = tmp_path / "file"
    blocker.write_text("not a folder")
    assert record_main(["--simulate", "--seconds", "0.2", "--out", str(blocker / "sub")]) == 3
    assert "Cannot create the recording" in capsys.readouterr().out


def test_cli_keyboard_interrupt_keeps_the_recording(tmp_path, monkeypatch):
    def interrupted(rec, src, stop, seconds):
        time.sleep(0.3)
        raise KeyboardInterrupt

    monkeypatch.setattr(record_cli, "_record", interrupted)
    assert record_main(["--simulate", "--out", str(tmp_path)]) == 0
    meta = wio.read_session_json(only_session(tmp_path))
    assert "t_stop" in meta and meta["samples"]["wii"] > 5


def test_cli_device_by_number(tmp_path, monkeypatch):
    bus = FakeBus(monkeypatch)
    bus.present = True
    assert record_main(["--device", "3", "--out", str(tmp_path / "x")]) == 2
    assert record_main(["--device", "0", "--seconds", "0.3", "--out", str(tmp_path / "y")]) == 0
    assert len(bus.opened) == 1
    meta = wio.read_session_json(only_session(tmp_path / "y"))
    assert meta["force_source"]["device_path"] == "fake-board-path"


def test_cli_device_by_unknown_path_waits_for_it(tmp_path, monkeypatch, capsys):
    FakeBus(monkeypatch)
    assert record_main(["--device", "some-other-path", "--wait", "0.2",
                        "--out", str(tmp_path / "x")]) == 2
    assert "no paired board has this path" in capsys.readouterr().out


def test_cli_close_handler_restores_signals(tmp_path):
    before = signal.getsignal(signal.SIGTERM)
    assert record_main(["--simulate", "--seconds", "0.2", "--out", str(tmp_path)]) == 0
    assert signal.getsignal(signal.SIGTERM) is before


@pytest.mark.skipif(sys.platform.startswith("win"), reason="POSIX signal")
def test_cli_sigterm_stops_cleanly(tmp_path):
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    p = subprocess.Popen([sys.executable, "-m", "poseassess.wii", "--simulate",
                          "--out", str(tmp_path)], cwd=tmp_path, env=env,
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         text=True)
    try:
        assert wait_until(lambda: any(tmp_path.iterdir()) and
                          (only_session(tmp_path) / "wii.csv").exists(), timeout=60)
        time.sleep(0.5)
        p.send_signal(signal.SIGTERM)
        out, err = p.communicate(timeout=30)
    finally:
        if p.poll() is None:
            p.kill()
    assert p.returncode == 0, err
    meta = wio.read_session_json(only_session(tmp_path))
    assert "t_stop" in meta and meta["samples"]["wii"] > 10


def test_module_entry_point(tmp_path, project):
    """``python -m poseassess.wii --simulate --seconds 2 --project <p>`` writes a valid
    recording (the acceptance check)."""
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    r = subprocess.run([sys.executable, "-m", "poseassess.wii", "--simulate", "--seconds", "2",
                        "--project", str(project.root)], cwd=tmp_path, env=env,
                       stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr
    from poseassess.core.balance.paths import WiiPaths

    (rec_id,) = WiiPaths(project).list_recordings()
    folder = WiiPaths(project).recording_dir(rec_id)
    meta = wio.read_session_json(folder)
    assert 1.9 < meta["duration_s"] < 5 and meta["samples"]["wii"] > 150
    wii = wio.read_wii_csv(folder)
    assert len(wii["t"]) == meta["samples"]["wii"] and np.all(np.diff(wii["t_rel"]) > 0)
    h = subprocess.run([sys.executable, "-m", "poseassess.wii", "--help"], cwd=tmp_path, env=env,
                       capture_output=True, text=True, timeout=120)
    assert h.returncode == 0 and "--project" in h.stdout and "python -m poseassess.wii" in h.stdout
