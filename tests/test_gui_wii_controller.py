"""GUI-WII: WiiController (start_default honours POSEASSESS_WII, FakeBus board appears/drops ->
signals on the GUI thread, connection_event, tare memory kept across sources, tare locked while
recording, simulator, hidapi missing message, shutdown) and WiiStatusWidget."""

import sys
import time

import numpy as np
import pytest

from tests.qtutil import pump, pump_until
from tests.wii_fakes import PATH, FakeBus


@pytest.fixture
def ctl(qapp):
    from poseassess.gui.wii_controller import WiiController

    c = WiiController()
    c.poll_interval_s = 0.05
    yield c
    c.shutdown()


@pytest.fixture
def no_boards(monkeypatch):
    from poseassess.wii.device import BalanceBoardHID

    monkeypatch.setattr(BalanceBoardHID, "list_devices", staticmethod(lambda: []))


@pytest.fixture(autouse=True)
def _no_hid_leak():
    """Tests here may import hidapi (hidapi_status); keep the smoke test's check meaningful."""
    had = "hid" in sys.modules
    yield
    if not had:
        sys.modules.pop("hid", None)


class Recorder:
    """Collects every public controller signal with the thread it arrived on."""

    def __init__(self, c):
        from PySide6.QtCore import QThread

        self.events, self.states, self.sources, self.tares, self.locks = [], [], [], [], []
        self.statuses = []
        self.threads = set()
        self._main = QThread.currentThread()

        def rec(lst):
            def slot(v):
                self.threads.add(QThread.currentThread() is self._main)
                lst.append(v)
            return slot

        c.connection_event.connect(rec(self.events))
        c.state_changed.connect(rec(self.states))
        c.source_changed.connect(rec(self.sources))
        c.tare_changed.connect(rec(self.tares))
        c.tare_lock_changed.connect(rec(self.locks))
        c.status_changed.connect(rec(self.statuses))


def test_wii_start_mode_and_battery(monkeypatch):
    from poseassess.gui.wii_controller import battery_percent, wii_start_mode

    for v, mode in (("0", "off"), ("off", "off"), ("sim", "sim"), ("", "auto"), ("1", "auto"),
                    ("auto", "auto")):
        monkeypatch.setenv("POSEASSESS_WII", v)
        assert wii_start_mode() == mode
    monkeypatch.delenv("POSEASSESS_WII")
    assert wii_start_mode() == "auto"
    assert battery_percent(0xC0) == 100 and battery_percent(96) == 50
    assert battery_percent(250) == 100 and battery_percent(None) is None


def test_start_default_modes(ctl, monkeypatch, no_boards):
    from poseassess.wii.device import SimulatedBoard, WiiAutoConnect

    monkeypatch.setenv("POSEASSESS_WII", "0")
    ctl.start_default()
    assert ctl.source is None and ctl.status_text() == "Wii: off" and not ctl.connected

    monkeypatch.setenv("POSEASSESS_WII", "sim")
    ctl.start_default()
    assert isinstance(ctl.source, SimulatedBoard) and ctl.is_simulator
    assert pump_until(lambda: ctl.connected and ctl.latest() is not None)
    assert ctl.status_text().startswith("Wii: simulator · ")
    sim = ctl.source
    monkeypatch.setenv("POSEASSESS_WII", "")
    ctl.start_default()  # a source was already chosen: kept
    assert ctl.source is sim

    ctl.disconnect()
    ctl.start_default()  # plug and play
    assert isinstance(ctl.source, WiiAutoConnect) and ctl.auto_connect
    assert pump_until(lambda: ctl.state == "searching")
    pump(0.05)
    assert ctl.status_text() == "Wii: searching for a paired board…"
    assert not sim.running


def test_auto_connect_board_appears_drops_and_reconnects(ctl, monkeypatch):
    bus = FakeBus(monkeypatch)
    rec = Recorder(ctl)
    t = time.perf_counter()
    ctl.start_auto()
    assert time.perf_counter() - t < 0.5  # never blocks the GUI
    src = ctl.source
    assert rec.sources == [src]
    assert pump_until(lambda: src.status == "searching")
    pump(0.2)
    assert rec.events == []

    bus.present = True  # board switched on
    assert pump_until(lambda: rec.events == ["wii_connected"])
    assert ctl.connected and pump_until(lambda: ctl.latest() is not None)
    assert "connected" in rec.states
    assert ctl.status_text().startswith("Wii: connected")
    assert "Auto-connect: connected" in ctl.detail_text()

    bus.opened[-1].broken = True  # Bluetooth drop -> reconnects by itself
    assert pump_until(lambda: src.disconnects == 1 and src.connections == 2
                      and rec.events == ["wii_connected", "wii_disconnected", "wii_connected"])
    assert rec.threads == {True}  # every signal arrived on the GUI thread
    assert ctl.source is src  # the same plug-and-play source all along
    assert "connections 2, drops 1" in ctl.detail_text()


def test_late_messages_of_a_replaced_source_are_ignored(ctl, no_boards):
    rec = Recorder(ctl)
    ctl.start_simulator()
    old = ctl.source
    assert pump_until(lambda: rec.events == ["wii_connected"])
    ctl.start_auto()
    assert rec.events == ["wii_connected", "wii_disconnected"]  # replacing logs the drop
    ctl._forward.emit(old, "connected")  # late message from the old reader thread
    pump(0.1)
    assert rec.events == ["wii_connected", "wii_disconnected"]
    assert not old.running


def test_tare_memory_is_kept_per_device_across_sources(ctl, monkeypatch):
    bus = FakeBus(monkeypatch)
    rec = Recorder(ctl)
    ctl.start_simulator()
    assert pump_until(lambda: len(ctl.recent(1.0)) > 50)
    t_sim = ctl.tare(1.0)
    assert t_sim.sum() > 50 and len(rec.tares) == 1
    np.testing.assert_allclose(rec.tares[0], t_sim)
    np.testing.assert_allclose(ctl.tare_values, t_sim)
    assert "tare" in ctl.detail_text()

    ctl.start_simulator()  # a new source object for the same (simulated) board
    np.testing.assert_allclose(ctl.tare_values, t_sim)
    ctl.disconnect()
    assert ctl.source is None and rec.sources[-1] is None
    ctl.start_simulator()  # memory survives a disconnect
    np.testing.assert_allclose(ctl.tare_values, t_sim)

    bus.present = True
    ctl.start_auto()  # a different board starts untared
    assert pump_until(lambda: ctl.connected and len(ctl.recent(0.6)) > 20)
    np.testing.assert_allclose(ctl.tare_values, np.zeros(4))
    t_board = ctl.tare(0.5)
    assert t_board.sum() > 10 and not np.allclose(t_board, t_sim)
    ctl.start_simulator()
    np.testing.assert_allclose(ctl.tare_values, t_sim)
    ctl.connect_device(PATH)  # the same board again: its own tare comes back on connect
    assert pump_until(lambda: ctl.connected)
    np.testing.assert_allclose(ctl.tare_values, t_board)


def test_tare_is_locked_while_recording(ctl, no_boards):
    rec = Recorder(ctl)
    with pytest.raises(RuntimeError, match="connect"):
        ctl.tare()
    ctl.start_simulator()
    assert pump_until(lambda: len(ctl.recent(1.0)) > 50)
    ctl.lock_tare("recording")
    ctl.lock_tare("recording")  # no duplicate signal
    assert ctl.tare_locked == "recording" and rec.locks == ["recording"]
    with pytest.raises(RuntimeError, match="recording"):
        ctl.tare()
    np.testing.assert_allclose(ctl.tare_values, np.zeros(4))
    assert "tare locked (recording)" in ctl.detail_text()
    ctl.lock_tare(None)
    assert rec.locks == ["recording", None]
    ctl.tare(0.5)
    assert ctl.tare_values.sum() > 30


def test_hidapi_missing(ctl, monkeypatch):
    from poseassess.wii import device

    def no_hid():
        raise device.HidapiUnavailable(device.HIDAPI_MISSING)

    monkeypatch.setattr(device, "_import_hid", no_hid)
    ctl.start_auto()
    src = ctl.source
    assert pump_until(lambda: not src.running)
    pump(0.05)
    assert ctl.fatal_error == device.HIDAPI_MISSING
    assert ctl.status_text() == "Wii: off (hidapi not installed — pip install hidapi)"
    assert "pip install hidapi" in ctl.detail_text()
    assert ctl.list_devices() == [] and "hidapi" in ctl.last_scan_error
    assert ctl.hidapi_status() == (False, device.HIDAPI_MISSING)
    ctl.start_simulator()  # the simulator still works
    assert pump_until(lambda: ctl.connected)
    assert ctl.fatal_error is None


def test_connect_device_one_shot_or_reconnecting(ctl, monkeypatch):
    from poseassess.wii.device import BalanceBoardHID, WiiAutoConnect

    bus = FakeBus(monkeypatch)
    bus.present = True
    devs = ctl.list_devices()
    assert [d["path"] for d in devs] == [PATH] and ctl.last_scan_error is None
    ctl.connect_device(PATH, reconnect=False)
    assert type(ctl.source) is BalanceBoardHID and not ctl.auto_connect
    assert pump_until(lambda: ctl.connected)
    ctl.connect_device(PATH)
    assert isinstance(ctl.source, WiiAutoConnect) and ctl.source.target_path == PATH
    assert ctl.auto_connect and pump_until(lambda: ctl.connected)
    assert len(bus.opened) == 2 and bus.opened[0].closed


def test_latest_is_never_stale(ctl, no_boards):
    ctl.start_simulator()
    assert pump_until(lambda: ctl.latest() is not None)
    assert ctl.recent(0.5)
    ctl.source.stop()  # the reader stops delivering (e.g. a frozen link)
    pump(0.3)
    assert ctl.latest(max_age_s=0.2) is None
    assert ctl.source.latest() is not None  # the raw buffer still has the old sample


def test_min_load_and_sensor_spacing_apply_to_new_sources(ctl, no_boards):
    ctl.set_min_load(12.0)
    ctl.set_sensor_spacing(0.5, 0.3)
    ctl.start_simulator()
    s = ctl.source
    assert s.min_total_kg == 12.0 and (s.sensor_dx_m, s.sensor_dy_m) == (0.5, 0.3)
    ctl.set_min_load(3.0)
    assert s.min_total_kg == 3.0 and ctl.min_load == 3.0
    ctl.start_simulator()
    assert ctl.source.min_total_kg == 3.0 and ctl.source.sensor_dx_m == 0.5


def test_use_source_disconnect_and_shutdown(ctl, no_boards):
    from poseassess.wii.device import SimulatedBoard

    rec = Recorder(ctl)
    src = SimulatedBoard()
    ctl.use_source(src)
    assert ctl.source is src and src.running
    assert pump_until(lambda: rec.events == ["wii_connected"])
    ctl.disconnect()
    assert rec.events == ["wii_connected", "wii_disconnected"]
    assert rec.sources == [src, None] and not src.running
    assert rec.statuses[-1] == "Wii: off"
    ctl.disconnect()  # nothing to do
    ctl.start_simulator()
    src2 = ctl.source
    assert pump_until(lambda: ctl.connected)
    n = len(rec.events)
    ctl.shutdown()
    ctl.shutdown()  # idempotent
    assert ctl.source is None and not src2.running
    assert len(rec.events) == n  # app exit: no event


def test_status_widget(qapp, ctl, no_boards):
    from poseassess.gui.widgets.wii_status import (
        GOOD,
        MAX_TEXT_PX,
        OFF,
        WiiStatusWidget,
    )

    w = WiiStatusWidget(ctl)
    assert w.full_text() == "Wii: off" and OFF in w.text()
    ctl.start_simulator()
    assert pump_until(lambda: "simulator" in w.full_text())
    assert GOOD in w.text() and "simulator" in w.toolTip()
    ctl.status_changed.emit("Wii: error (" + "x" * 500 + ")")
    assert w.minimumSizeHint().width() <= 120
    assert w.sizeHint().width() <= MAX_TEXT_PX + 40
    assert w.full_text().endswith(")") and "…" in w.text()


def test_status_widget_click_opens_the_wii_page(qapp):
    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtTest import QTest

    from poseassess.gui.main_window import MainWindow

    w = MainWindow()
    try:
        w.show()
        pump(0.05)
        sw = w._wii_status
        assert sw is not None and w.nav.currentRow() == 0
        QTest.mouseClick(sw, Qt.LeftButton, Qt.NoModifier, QPoint(5, 5))
        assert w.nav.currentRow() == 2 and type(w.pages[2]).__name__ == "WiiBoardPage"
        assert sw.full_text() == "Wii: off"  # POSEASSESS_WII=0 in the tests
    finally:
        w.close()


def test_source_is_locked_while_recording(ctl, monkeypatch):
    """A take is running (lock_tare): no other source may write into it, except the
    explicitly confirmed Connect of a board that dropped out."""
    from poseassess.wii.device import SimulatedBoard, WiiAutoConnect

    bus = FakeBus(monkeypatch)
    bus.present = True
    ctl.set_min_load(7.0)
    ctl.start_simulator()
    sim = ctl.source
    ctl.lock_tare("recording")
    for call in (ctl.start_simulator, ctl.start_auto, ctl.disconnect,
                 lambda: ctl.connect_device(PATH), lambda: ctl.set_min_load(9.0),
                 lambda: ctl.use_source(SimulatedBoard())):
        with pytest.raises(RuntimeError, match="while recording is running"):
            call()
    assert ctl.source is sim and ctl.min_load == 7.0
    ctl.set_min_load(7.0)  # unchanged: fine
    ctl.connect_device(PATH, allow_during_lock=True)  # confirmed by the user
    assert isinstance(ctl.source, WiiAutoConnect) and pump_until(lambda: ctl.connected)
    ctl.lock_tare(None)
    ctl.set_min_load(9.0)
    ctl.disconnect()
    assert ctl.source is None
