"""CORE-WII: protocol (calibration bytes, sensor reports, output reports, COP, pressed_sensor) and
device (sort_boards, WiiAutoConnect reconnect + tare memory, paired-but-off, stalled link, hidapi
missing / shadowed, 0x7 retry, Wii Remote skipped, simulator). Port of PoseBoard 3ea6d8a
tests/test_wii_auto.py + the Wii part of tests/test_core.py; fakes in tests/wii_fakes.py."""

import json
import math
import subprocess
import sys
import time
import types
from pathlib import Path

import numpy as np
import pytest

from poseassess.wii import device as device_mod
from poseassess.wii import protocol as P
from poseassess.wii.device import (
    HID_SHADOWED,
    HIDAPI_MISSING,
    BalanceBoardHID,
    ForceSource,
    HidapiUnavailable,
    SimulatedBoard,
    WiiAutoConnect,
    hidapi_status,
    sort_boards,
)
from tests.wii_fakes import BOARD, CAL, KG0, KG17, PATH, FakeBus, FakeHid, wait_until

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _no_real_hid_leak():
    """Never leave the real hidapi module imported (the app smoke test checks it is not)."""
    had = "hid" in sys.modules
    yield
    if not had:
        sys.modules.pop("hid", None)


# ------------------------------------------------------------------ protocol
def test_calibration_parse_and_kg():
    kg0, kg17, kg34 = [1000, 1100, 1200, 1300], [2700, 2800, 2900, 3000], [4400, 4500, 4600, 4700]
    data = bytes(4) + b"".join(v.to_bytes(2, "big") for v in kg0 + kg17 + kg34) + bytes(4)
    cal = P.Calibration.from_bytes(data)
    np.testing.assert_allclose(cal.to_kg(np.array(kg0)), 0)
    np.testing.assert_allclose(cal.to_kg(np.array(kg17)), 17)
    np.testing.assert_allclose(cal.to_kg(np.array(kg34)), 34)
    np.testing.assert_allclose(cal.to_kg(np.array([1850, 1950, 2050, 2150])), 8.5)
    np.testing.assert_allclose(cal.to_kg(np.array([3550, 3650, 3750, 3850])), 25.5)
    assert cal.to_dict() == {"kg0": kg0, "kg17": kg17, "kg34": kg34}
    json.dumps(cal.to_dict())
    with pytest.raises(ValueError):
        P.Calibration.from_bytes(bytes(10))


def test_parse_sensor_report():
    vals = [0x1234, 0x0102, 0xABCD, 0x0FF0]
    rep = bytes([0x32, 0, 0]) + b"".join(v.to_bytes(2, "big") for v in vals) + bytes(11)
    np.testing.assert_array_equal(P.parse_sensor_raw(rep), vals)
    assert P.parse_sensor_raw(bytes([0x20] + [0] * 21)) is None
    assert P.parse_sensor_raw(b"") is None
    assert P.parse_sensor_raw(bytes([0x32, 0, 0, 1])) is None  # truncated


def test_output_reports():
    r = P.write_memory_report(0xA400F0, b"\x55")
    assert len(r) == P.OUTPUT_REPORT_LEN
    assert list(r[:7]) == [0x16, 0x04, 0xA4, 0x00, 0xF0, 0x01, 0x55]
    assert list(P.read_memory_report(0xA40020, 32)[:7]) == [0x17, 0x04, 0xA4, 0x00, 0x20, 0x00, 0x20]
    assert list(P.set_mode_report()[:3]) == [0x12, 0x04, 0x32]
    assert list(P.set_mode_report(continuous=False)[:3]) == [0x12, 0x00, 0x32]
    assert list(P.led_report(True)[:2]) == [0x11, 0x10] and list(P.led_report(False)[:2]) == [0x11, 0]
    assert list(P.status_request_report()[:2]) == [0x15, 0x00]
    off, data, err = P.parse_read_data(bytes([0x21, 0, 0, 0xF0, 0x00, 0x20]) + bytes(range(16)))
    assert (off, len(data), err) == (0x20, 16, 0)
    off, data, err = P.parse_read_data(bytes([0x21, 0, 0, 0x07, 0x00, 0xFA]) + bytes(16))
    assert (off, len(data), err) == (0xFA, 1, 7)
    with pytest.raises(ValueError):
        P.write_memory_report(0xA40000, bytes(17))


def test_extension_type():
    assert P.is_balance_board(P.BALANCE_BOARD_EXT_ID)
    assert not P.is_balance_board(bytes([0, 0, 0xA4, 0x20, 0, 0]))  # Nunchuk
    assert not P.is_balance_board(bytes([0, 0, 0xA4, 0x20, 1, 1]))  # Classic Controller
    assert not P.is_balance_board(b"\x04\x02")  # too short


def test_cop():
    dx, dy = 0.433, 0.238
    assert P.center_of_pressure([10, 10, 10, 10], dx, dy) == (0, 0)
    x, y = P.center_of_pressure([20, 0, 0, 0], dx, dy)  # all load on TR
    assert x == pytest.approx(dx / 2) and y == pytest.approx(dy / 2)
    x, y = P.center_of_pressure([0, 0, 0, 20], dx, dy)  # BL
    assert x == pytest.approx(-dx / 2) and y == pytest.approx(-dy / 2)
    x, y = P.center_of_pressure([0, 0, 20, 0], dx, dy)  # TL: left (-x), front (+y)
    assert x == pytest.approx(-dx / 2) and y == pytest.approx(dy / 2)
    assert np.isnan(P.center_of_pressure([0.1, 0, 0, 0], dx, dy)[0])
    assert np.isnan(P.center_of_pressure([1, 1, 1, 1], dx, dy, min_total_kg=5.0)[1])


def test_pressed_sensor():
    base = [10.0, 10.0, 10.0, 10.0]
    for i, name in enumerate(P.SENSOR_ORDER):
        pressed = list(base)
        pressed[i] += 8.0
        assert P.pressed_sensor(base, pressed) == name
    assert P.pressed_sensor(base, [11, 10.5, 12.9, 10]) is None  # below 3 kg
    assert P.pressed_sensor(base, [11, 10.5, 12.9, 10], min_rise_kg=2.0) == "TL"
    assert P.SENSOR_ORDER == ("TR", "BR", "TL", "BL")
    assert P.KG_TO_N == pytest.approx(9.80665)


# ------------------------------------------------------------------ simulator
def test_simulator_cop_matches_model():
    sim = SimulatedBoard()
    kg = sim.kg_at(1.234)
    x, y = P.center_of_pressure(kg, sim.sensor_dx_m, sim.sensor_dy_m)
    assert x == pytest.approx(0.06 * math.sin(2 * math.pi * 0.23 * 1.234), abs=1e-9)
    assert y == pytest.approx(0.04 * math.sin(2 * math.pi * 0.31 * 1.234 + 0.7), abs=1e-9)


def test_simulator_thread_and_tare():
    sim = SimulatedBoard(mass_kg=0.0)
    got = []
    sim.add_listener(got.append)
    sim.start()
    time.sleep(0.3)
    sim.do_tare(0.2)
    sim.stop()
    assert len(got) > 10
    assert np.all(np.abs(sim.tare) < 0.2)
    assert all(b.t > a.t for a, b in zip(got, got[1:]))  # perf_counter stamps, increasing


def test_simulator_status_and_info():
    sim = SimulatedBoard()
    assert sim.status == "stopped" and not sim.connected and sim.latest() is None
    states = []
    sim.add_status_listener(states.append)
    sim.start()
    assert wait_until(lambda: sim.connected and sim.latest() is not None)
    s = sim.latest()
    assert s.kg.shape == (4,) and 60 < s.total_kg < 80 and np.all(np.isfinite(s.cop_board))
    assert sim.device_key == "simulator"
    info = sim.info()
    assert info["type"] == "SimulatedBoard" and info["min_total_kg"] == 5.0
    json.dumps(info)
    sim.stop()
    assert sim.status == "stopped" and states[-1] == "stopped" and "connected" in states


def test_simulator_does_not_burst_after_a_stall(monkeypatch):
    """A stalled simulator thread (machine suspended) must not emit a burst of samples with
    identical timestamps afterwards."""
    sim = SimulatedBoard(rate_hz=100)
    got = []
    sim.add_listener(got.append)
    sim.start()
    assert wait_until(lambda: len(got) > 5)
    real = sim._emit_kg
    stalled = []

    def slow_emit(t, kg, raw=None):
        if not stalled:
            stalled.append(t)
            time.sleep(1.0)  # the reader thread hangs for 1 s
        return real(t, kg, raw)

    monkeypatch.setattr(sim, "_emit_kg", slow_emit)
    time.sleep(1.5)
    sim.stop()
    after = [s.t for s in got if s.t > stalled[0] + 0.5]
    assert after and len(after) < 0.9 * 100  # ~50 samples in the last 0.5 s, not a 150-burst
    assert np.median(np.diff(after)) > 0.005  # regular 100 Hz spacing, not back-to-back


def test_listener_errors_do_not_stop_the_source():
    sim = SimulatedBoard()
    ok = []

    def bad(_):
        raise RuntimeError("listener bug")

    sim.add_listener(bad)
    sim.add_listener(ok.append)
    sim.add_status_listener(lambda s: 1 / 0)
    sim.start()
    try:
        assert wait_until(lambda: len(ok) > 10)
        assert sim.running and sim.error is None
    finally:
        sim.stop()
    sim.remove_listener(bad)
    sim.remove_listener(bad)  # not registered any more: no error


def test_recent_and_tare_without_data():
    src = ForceSource()
    assert src.recent(1.0) == [] and src.latest() is None
    with pytest.raises(RuntimeError, match="No data"):
        src.do_tare(0.5)
    src._emit_kg(time.perf_counter(), np.array([1.0, 2.0, 3.0, 4.0]))
    assert len(src.recent(1.0)) == 1 and src.latest().total_kg == pytest.approx(10.0)
    tare = src.do_tare(1.0)
    np.testing.assert_allclose(tare, [1, 2, 3, 4])
    s = src._emit_kg(time.perf_counter(), np.array([1.0, 2.0, 3.0, 24.0]))
    np.testing.assert_allclose(s.kg, [0, 0, 0, 20])
    assert s.cop_board == pytest.approx((-0.433 / 2, -0.238 / 2))


# ------------------------------------------------------------------ device list
def test_sort_boards_prefers_balance_board_and_drops_remotes():
    devs = [{"path": b"a", "product_string": "Nintendo RVL-CNT-01"},
            {"path": b"b", "product_string": ""},
            {"path": b"c", "product_string": "Nintendo RVL-WBC-01"},
            {"path": b"d", "product_string": None}]
    assert [d["path"] for d in sort_boards(devs)] == [b"c", b"b", b"d"]


def test_list_devices_filters_vendor_and_product(monkeypatch):
    fake = types.ModuleType("hid")
    fake.device = object

    def enumerate_(vid, pid):
        assert vid == P.VENDOR_ID and pid == 0
        return [{"path": b"remote", "product_id": 0x0330, "product_string": "RVL-CNT-01-TR"},
                {"path": b"old-remote", "product_id": 0x0306, "product_string": "Nintendo RVL-CNT-01"},
                dict(BOARD)]

    fake.enumerate = enumerate_
    monkeypatch.setitem(sys.modules, "hid", fake)
    assert [d["path"] for d in BalanceBoardHID.list_devices()] == [PATH]


# ------------------------------------------------------------------ hidapi availability
def test_hidapi_status(monkeypatch):
    fake = types.ModuleType("hid")
    fake.device, fake.enumerate = object, lambda *a: []
    monkeypatch.setitem(sys.modules, "hid", fake)
    assert hidapi_status() == (True, "")

    monkeypatch.setitem(sys.modules, "hid", None)  # "import hid" raises ImportError
    ok, why = hidapi_status()
    assert not ok and why == HIDAPI_MISSING and "pip install hidapi" in why

    monkeypatch.setitem(sys.modules, "hid", types.ModuleType("hid"))  # the other 'hid' package
    ok, why = hidapi_status()
    assert not ok and why == HID_SHADOWED


def test_hidapi_dll_load_failure_is_reported(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def failing_import(name, *a, **kw):
        if name == "hid":
            raise OSError("[WinError 126] The specified module could not be found")
        return real_import(name, *a, **kw)

    monkeypatch.delitem(sys.modules, "hid", raising=False)
    monkeypatch.setattr(builtins, "__import__", failing_import)
    ok, why = hidapi_status()
    assert not ok and "hidapi cannot be loaded" in why and "WinError 126" in why
    with pytest.raises(HidapiUnavailable):
        device_mod._import_hid()


def test_auto_connect_without_hidapi_stops_with_the_reason(monkeypatch):
    monkeypatch.setitem(sys.modules, "hid", None)  # "import hid" raises ImportError
    src = WiiAutoConnect(poll_interval_s=0.02)
    states = []
    src.add_status_listener(states.append)
    src.start()
    try:
        assert wait_until(lambda: not src.running)
        assert src.fatal_error == HIDAPI_MISSING
        assert src.status == f"error: {HIDAPI_MISSING}" and "pip install hidapi" in src.status
        assert src.connections == 0 and states[-1] == "stopped"
    finally:
        src.stop()
    # hidapi becomes available (installed meanwhile): start() tries again
    bus = FakeBus(monkeypatch)
    bus.present = True
    src.start()
    try:
        assert wait_until(lambda: src.connected and src.latest() is not None)
        assert src.fatal_error is None and src.error is None
    finally:
        src.stop()


def test_auto_connect_with_shadowed_hid_module(monkeypatch):
    monkeypatch.setitem(sys.modules, "hid", types.ModuleType("hid"))
    src = WiiAutoConnect(poll_interval_s=0.02)
    src.start()
    try:
        assert wait_until(lambda: not src.running)
        assert src.fatal_error == HID_SHADOWED and "pip uninstall hid" in src.status
    finally:
        src.stop()


def test_importing_never_imports_hid():
    code = ("import sys\n"
            "import poseassess.wii, poseassess.wii.protocol, poseassess.wii.device, "
            "poseassess.wii.io, poseassess.wii.recorder, poseassess.wii.record_cli\n"
            "from poseassess.wii.device import WiiAutoConnect, SimulatedBoard, BalanceBoardHID\n"
            "WiiAutoConnect(); SimulatedBoard(); BalanceBoardHID()\n"
            "poseassess.wii.record_cli.build_parser()\n"
            "assert 'hid' not in sys.modules, 'hid imported'\n"
            "print('ok')\n")
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True,
                       timeout=120)
    assert r.returncode == 0 and r.stdout.strip() == "ok", r.stderr


# ------------------------------------------------------------------ plug and play
def test_auto_connect_reconnect_and_tare(monkeypatch):
    bus = FakeBus(monkeypatch)
    src = WiiAutoConnect(poll_interval_s=0.05)
    states: list[str] = []
    src.add_status_listener(states.append)
    src.start()
    try:
        assert wait_until(lambda: src.status == "searching")
        time.sleep(0.15)
        assert not bus.opened and src.latest() is None

        # Board switched on -> found and connected without any user action
        bus.present = True
        assert wait_until(lambda: src.connected and len(src.recent(1.0)) > 20)
        assert src.path == PATH and src.connections == 1
        np.testing.assert_allclose(src.calibration.kg17, KG17)
        s = src.latest()
        np.testing.assert_allclose(s.kg, CAL.to_kg(np.array([1850, 1950, 2050, 2150])))
        np.testing.assert_allclose(s.kg, 8.5)
        assert s.total_kg == pytest.approx(34.0)
        assert s.cop_board == pytest.approx(P.center_of_pressure(s.kg, src.sensor_dx_m, src.sensor_dy_m))
        dev1 = bus.opened[0]
        assert P.set_mode_report() in dev1.writes and P.led_report(True) in dev1.writes
        info = src.info()
        assert info["device_path"] == PATH.decode() and info["product_string"] == BOARD["product_string"]
        assert info["auto_connect"] and info["connections"] == 1
        json.dumps(info)  # goes into session.json

        # Tare with the board "empty"
        time.sleep(0.1)
        tare = src.do_tare(0.1).copy()
        np.testing.assert_allclose(tare, 8.5, atol=1e-9)

        # Bluetooth drop: reads fail and the device disappears from the list
        bus.present = False
        dev1.broken = True
        assert wait_until(lambda: src.status == "searching")
        assert dev1.closed and src.path is None and src.disconnects == 1
        assert "disconnected" in src.last_error
        assert src.error is None  # not fatal: still running and waiting
        assert src.running

        # Board comes back with a different load -> reconnects, tare kept
        bus.next_device = lambda: FakeHid(KG17)  # 17 kg on every sensor
        n_before = len(src.buffer)
        bus.present = True
        assert wait_until(lambda: src.connected and len(src.buffer) > n_before + 20)
        assert src.connections == 2 and len(bus.opened) == 2
        np.testing.assert_allclose(src.tare, tare)
        np.testing.assert_allclose(src.latest().kg, 17.0 - 8.5)
    finally:
        t = time.perf_counter()
        src.stop()
        assert time.perf_counter() - t < 1.0
    assert src.status == "stopped"
    assert bus.opened[-1].closed and bus.opened[-1].writes[-1] == P.led_report(False)
    expected = ["searching", "connecting", "connected", "searching", "connecting", "connected", "stopped"]
    it = iter(states)
    assert all(e in it for e in expected), states  # ordered subsequence


def test_auto_connect_board_paired_but_off(monkeypatch):
    bus = FakeBus(monkeypatch)
    bus.present = True
    bus.next_device = lambda: FakeHid(KG0, write_ok=False)  # listed, but writes fail
    src = WiiAutoConnect(poll_interval_s=0.02)
    src.start()
    try:
        assert wait_until(lambda: len(bus.opened) >= 3)
        assert src.status in ("searching", "connecting")
        assert "not responding" in src.last_error and src.connections == 0
        assert all(d.closed for d in bus.opened[:-1])
        assert src.fatal_error is None and src.running
    finally:
        src.stop()


def test_auto_connect_detects_stalled_link(monkeypatch):
    bus = FakeBus(monkeypatch)
    bus.present = True
    first = FakeHid(KG0, silent=True)
    devices = iter([first])
    bus.next_device = lambda: next(devices, None) or FakeHid(KG0)
    src = WiiAutoConnect(poll_interval_s=0.02, silence_timeout_s=0.3)
    src.start()
    try:
        assert wait_until(lambda: src.disconnects == 1)
        assert first.closed and "no data" in src.last_error
        assert P.status_request_report() in first.writes  # tried to wake the stream first
        assert wait_until(lambda: src.connected and src.latest() is not None)
    finally:
        src.stop()


def test_auto_connect_open_failure_is_retried_later(monkeypatch):
    bus = FakeBus(monkeypatch)
    bus.present = True
    calls = []

    def open_fails(path):
        calls.append(path)
        if len(calls) == 1:
            raise OSError("open failed")
        return FakeHid([1850, 1950, 2050, 2150])

    monkeypatch.setattr(BalanceBoardHID, "open_device", staticmethod(open_fails))
    src = WiiAutoConnect(poll_interval_s=0.02, retry_failed_s=0.2)
    src.start()
    try:
        assert wait_until(lambda: src.status.startswith("error:"))
        assert "open failed" in src.last_error and src.running and src.fatal_error is None
        assert wait_until(lambda: src.connected and src.latest() is not None)
        assert len(calls) == 2
    finally:
        src.stop()


def test_auto_connect_target_path(monkeypatch):
    other = {**BOARD, "path": b"other-board"}
    monkeypatch.setattr(BalanceBoardHID, "list_devices", staticmethod(lambda: [other, dict(BOARD)]))
    opened = []
    monkeypatch.setattr(BalanceBoardHID, "open_device",
                        staticmethod(lambda p: opened.append(p) or FakeHid(KG17)))
    src = WiiAutoConnect(path=PATH, poll_interval_s=0.02)
    src.start()
    try:
        assert wait_until(lambda: src.connected and src.latest() is not None)
        assert opened == [PATH] and src.path == PATH
    finally:
        src.stop()


def test_balance_board_hid_surfaces_disconnect(monkeypatch):
    bus = FakeBus(monkeypatch)
    src = BalanceBoardHID(PATH)
    src.start()
    try:
        assert wait_until(lambda: src.connected and src.latest() is not None)
        bus.opened[0].broken = True
        assert wait_until(lambda: not src.running)
        assert "disconnected" in src.error.lower() and src.status.startswith("error:")
        assert bus.opened[0].closed
    finally:
        src.stop()


def test_balance_board_hid_without_board(monkeypatch):
    FakeBus(monkeypatch)  # nothing paired
    src = BalanceBoardHID()
    src.start()
    try:
        assert wait_until(lambda: not src.running)
        assert "not found" in src.error and "pair" in src.error
    finally:
        src.stop()


# ------------------------------------------------------------ device identity and tare
NUNCHUK_ID = bytes([0x00, 0x00, 0xA4, 0x20, 0x00, 0x00])


def test_wii_remote_with_accessory_is_not_used_as_a_board(monkeypatch):
    bus = FakeBus(monkeypatch)
    bus.present = True
    bus.next_device = lambda: FakeHid(KG0, ext_id=NUNCHUK_ID)
    src = WiiAutoConnect(poll_interval_s=0.02)
    src.start()
    try:
        assert wait_until(lambda: "not a Balance Board" in (src.last_error or ""))
        time.sleep(0.2)
        assert len(bus.opened) == 1  # skipped for good, not retried
        assert src.connections == 0 and src.latest() is None and src.status == "searching"
        assert bus.opened[0].closed
    finally:
        src.stop()
    one_shot = BalanceBoardHID(PATH)
    one_shot.start()
    try:
        assert wait_until(lambda: not one_shot.running)
        assert "not a Balance Board" in one_shot.error
    finally:
        one_shot.stop()


def test_extension_read_error_0x7_is_retried_once(monkeypatch):
    bus = FakeBus(monkeypatch)
    bus.present = True
    dev = FakeHid(KG17, ext_id=None)  # the first read of the extension type fails with 0x7
    real_write = dev.write

    def write(data):
        n = real_write(data)
        if bytes(data)[0] == P.REPORT_READ_MEMORY and dev.ext_id is None:
            dev.ext_id = P.BALANCE_BOARD_EXT_ID  # answers the retry
        return n

    dev.write = write
    bus.next_device = lambda: dev
    src = BalanceBoardHID(PATH)
    src.start()
    try:
        assert wait_until(lambda: src.connected and src.latest() is not None)
    finally:
        src.stop()


def test_extension_read_error_0x7_twice_is_not_a_board(monkeypatch):
    bus = FakeBus(monkeypatch)
    bus.present = True
    bus.next_device = lambda: FakeHid(KG17, ext_id=None)  # never answers
    src = BalanceBoardHID(PATH)
    src.start()
    try:
        assert wait_until(lambda: not src.running)
        assert "error 0x7" in src.error
    finally:
        src.stop()


def test_tare_is_kept_per_board(monkeypatch):
    bus = FakeBus(monkeypatch)
    bus.present = True
    src = WiiAutoConnect(poll_interval_s=0.02)
    src.start()
    try:
        assert wait_until(lambda: src.connected and len(src.recent(0.2)) > 5)
        tare = src.do_tare(0.1).copy()
        np.testing.assert_allclose(tare, 8.5)
    finally:
        src.stop()
    # Replaced by a new source object (Connect / Auto-connect toggled): same board, same tare
    again = WiiAutoConnect(poll_interval_s=0.02)
    again.adopt_tares(src)
    again.start()
    try:
        assert wait_until(lambda: again.connected and again.latest() is not None)
        np.testing.assert_allclose(again.tare, tare)
        np.testing.assert_allclose(again.latest().kg, 0.0, atol=1e-9)
    finally:
        again.stop()
    # A different board does not inherit it
    other = {**BOARD, "path": b"other-board"}
    monkeypatch.setattr(BalanceBoardHID, "list_devices", staticmethod(lambda: [other]))
    monkeypatch.setattr(BalanceBoardHID, "open_device", staticmethod(lambda p: FakeHid(KG17)))
    third = WiiAutoConnect(poll_interval_s=0.02)
    third.adopt_tares(again)
    third.start()
    try:
        assert wait_until(lambda: third.connected and third.latest() is not None)
        np.testing.assert_allclose(third.tare, 0.0)
        np.testing.assert_allclose(third.latest().kg, 17.0)
        assert third.info()["min_total_kg"] == 5.0
    finally:
        third.stop()
    third.adopt_tares(None)  # no-op


def test_simulator_tare_survives_source_replacement():
    sim = SimulatedBoard(mass_kg=0.0)
    sim.start()
    try:
        assert wait_until(lambda: len(sim.recent(0.3)) > 10)
        tare = sim.do_tare(0.2).copy()
    finally:
        sim.stop()
    again = SimulatedBoard()
    again.adopt_tares(sim)
    np.testing.assert_allclose(again.tare, tare)
