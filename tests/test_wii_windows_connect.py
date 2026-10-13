"""A board that Windows has paired must be found and connected even when (a) its HID product
string looks like a Wii Remote ("RVL-CNT"), or (b) the Bluetooth stack rejects / ignores
hidapi's WriteFile - then the output reports go through HidD_SetOutputReport (WiimoteLib's
"alternate write method"). A listed board that never answers is reported as such, not as
"searching". ``--diagnose`` shows what happens step by step."""

import sys
import time
import types

import pytest

from poseassess.wii import device as device_mod
from poseassess.wii import protocol as P
from poseassess.wii.device import BalanceBoardHID, WiiAutoConnect, diagnose
from tests.wii_fakes import BOARD, FakeHid, wait_until


@pytest.fixture(autouse=True)
def _no_real_hid_leak():
    had = "hid" in sys.modules
    yield
    if not had:
        sys.modules.pop("hid", None)


class IgnoringHid(FakeHid):
    """WriteFile 'succeeds' but the board never sees the reports (some Bluetooth stacks)."""

    def write(self, data):
        if self.closed:
            raise ValueError("not open")
        self.writes.append(bytes(data))
        return len(data)


class FakeAlt:
    """Stand-in for the Windows second-handle writers (WinWriteFile / WinOutputReport):
    delivers the reports to the fake board; methods in ``FakeAlt.broken`` fail."""

    instances: list = []
    broken: set = set()

    def __init__(self, dev, method="set_output_report"):
        self.dev, self.method, self.writes, self.closed = dev, method, [], False
        self.last_error = ""
        FakeAlt.instances.append(self)

    def write(self, report):
        report = bytes(report)
        if self.method == "writefile_exact":
            report = report[:P.real_length(report)]
        self.writes.append(report)
        if self.method in FakeAlt.broken:
            self.last_error = f"{self.method} failed: The parameter is incorrect."
            return -1
        return _deliver(self.dev, report)

    def close(self):
        self.closed = True


def _deliver(dev, report):
    ok = dev.write_ok
    dev.write_ok = True
    try:
        return FakeHid.write(dev, list(report))
    finally:
        dev.write_ok = ok


def _fake_hid_module(monkeypatch, devs):
    fake = types.ModuleType("hid")
    fake.device = object
    fake.enumerate = lambda vid, pid: [dict(d) for d in devs]
    monkeypatch.setitem(sys.modules, "hid", fake)


def _setup(monkeypatch, make_dev, methods=("write", "set_output_report"), devs=(BOARD,)):
    """Every open_device gives a fresh fake handle (like the real one); the alternate writer
    talks to the handle opened last."""
    _fake_hid_module(monkeypatch, devs)
    opened = []

    def open_device(path):
        opened.append(make_dev())
        return opened[-1]

    monkeypatch.setattr(BalanceBoardHID, "open_device", staticmethod(open_device))
    monkeypatch.setattr(BalanceBoardHID, "WRITE_METHODS", methods)
    FakeAlt.instances, FakeAlt.broken = [], set()
    monkeypatch.setattr(device_mod, "_alt_writer",
                        lambda method, path: FakeAlt(opened[-1], method))
    return opened


def test_board_named_like_a_wii_remote_connects(monkeypatch):
    _setup(monkeypatch, lambda: FakeHid([1850, 1950, 2050, 2150]), methods=("write",),
           devs=(dict(BOARD, product_string="Nintendo RVL-CNT-01"),))
    src = WiiAutoConnect(poll_interval_s=0.02)
    src.start()
    try:
        assert wait_until(lambda: src.connected and src.latest() is not None)
        assert src.write_method == "write" and src.info()["write_method"] == "write"
    finally:
        src.stop()


def test_real_report_lengths():
    assert P.real_length(P.read_memory_report(0xA40020, 32)) == 7
    assert P.real_length(P.write_memory_report(0xA400F0, b"\x55")) == 22
    assert P.real_length(P.set_mode_report()) == 3
    assert P.real_length(P.status_request_report()) == 2 and P.real_length(P.led_report(True)) == 2
    assert P.real_length(b"") == 0


def test_writefile_exact_is_tried_before_set_output_report(monkeypatch):
    _setup(monkeypatch, lambda: FakeHid([1850, 1950, 2050, 2150], write_ok=False),
           methods=("write", "writefile_exact", "set_output_report"))
    src = WiiAutoConnect(poll_interval_s=0.02)
    src.start()
    try:
        assert wait_until(lambda: src.connected and src.latest() is not None, timeout=8)
        assert src.write_method == "writefile_exact"
        alt = FakeAlt.instances[-1]
        assert alt.method == "writefile_exact"
        assert any(w[0] == P.REPORT_READ_MEMORY and len(w) == 7 for w in alt.writes)  # unpadded
    finally:
        src.stop()


@pytest.mark.parametrize("cls", [FakeHid, IgnoringHid], ids=["write-fails", "write-ignored"])
def test_falls_back_to_set_output_report(monkeypatch, cls):
    kw = {"write_ok": False} if cls is FakeHid else {}
    _setup(monkeypatch, lambda: cls([1850, 1950, 2050, 2150], **kw),
           methods=("write", "writefile_exact", "set_output_report"))
    FakeAlt.broken = {"writefile_exact"}
    src = WiiAutoConnect(poll_interval_s=0.02)
    src.start()
    try:
        assert wait_until(lambda: src.connected and src.latest() is not None, timeout=8)
        assert src.write_method == "set_output_report" and src.unresponsive is None
        assert src.calibration is not None
        alt = FakeAlt.instances[-1]
        assert [w[0] for w in alt.writes][-2:] == [P.REPORT_LEDS, P.REPORT_MODE]  # init via alt
        n = len(alt.writes)
        src._write(P.status_request_report())  # later writes too
        assert len(alt.writes) == n + 1
    finally:
        src.stop()
    assert FakeAlt.instances[-1].closed  # second handle closed on stop


def test_listed_board_that_never_answers_is_reported(monkeypatch, qapp):
    from poseassess.gui.wii_controller import WiiController

    _setup(monkeypatch, lambda: FakeHid([1850, 1950, 2050, 2150], write_ok=False),
           methods=("write",))
    c = WiiController()
    c.poll_interval_s = 0.02
    try:
        c.start_auto()
        src = c.source
        assert wait_until(lambda: src.unresponsive is not None, timeout=5)
        assert "HID write failed" in src.unresponsive
        assert src.status == "searching"
        assert c.status_text().startswith("Wii: board found but not responding")
    finally:
        c.shutdown()


def test_diagnose_reports_each_step(monkeypatch):
    other = dict(BOARD, path=b"keyboard", product_id=0x1234, product_string="Something")
    _setup(monkeypatch, lambda: FakeHid([1850, 1950, 2050, 2150], write_ok=False),
           devs=(BOARD, other))
    lines = []
    assert diagnose(out=lines.append, listen_s=0.2) == 1
    text = "\n".join(lines)
    assert "Nintendo HID devices (vendor 0x057E): 2" in text and "RVL-WBC-01" in text
    assert "[0] write: FAILED" in text and "HID write failed" in text
    assert "[0] set_output_report: OK" in text and "[1] skipped" in text
    assert "writefile_exact" not in text  # not in this test's WRITE_METHODS
    assert "Result: 1 board(s) delivering data." in text


def test_diagnose_flag(monkeypatch, capsys):
    from poseassess.wii import record_cli

    monkeypatch.setattr(device_mod, "diagnose", lambda: 1)
    assert record_cli.main(["--diagnose"]) == record_cli.EXIT_OK
    monkeypatch.setattr(device_mod, "diagnose", lambda: 0)
    assert record_cli.main(["--diagnose"]) == record_cli.EXIT_NO_BOARD


def test_a_dead_entry_listed_first_does_not_starve_the_board(monkeypatch):
    """A paired Wii Remote that is off (or a stale duplicate) listed before the board."""
    dead = dict(BOARD, path=b"dead", product_string="Nintendo RVL-WBC-01")
    board = dict(BOARD, path=b"board", product_string="Nintendo RVL-CNT-01")
    _fake_hid_module(monkeypatch, (dead, board))
    tries = []

    def open_device(path):
        tries.append(path)
        return FakeHid([1850, 1950, 2050, 2150], write_ok=(path == b"board"))

    monkeypatch.setattr(BalanceBoardHID, "open_device", staticmethod(open_device))
    monkeypatch.setattr(BalanceBoardHID, "WRITE_METHODS", ("write",))
    src = WiiAutoConnect(poll_interval_s=0.02)
    src.start()
    try:
        assert wait_until(lambda: src.connected and src.latest() is not None, timeout=5)
        assert src.path == b"board" and tries[0] == b"dead"
    finally:
        src.stop()


def test_not_a_board_is_skipped_for_a_while_only(monkeypatch):
    _fake_hid_module(monkeypatch, (BOARD,))
    src = WiiAutoConnect()
    src._not_boards[BOARD["path"]] = time.perf_counter() + 60
    assert src._find_board() is None
    src._not_boards[BOARD["path"]] = time.perf_counter() - 1  # expired: tried again
    assert src._find_board()["path"] == BOARD["path"]


@pytest.mark.skipif(sys.platform != "win32", reason="Windows HID API")
@pytest.mark.parametrize("cls", ["WinOutputReport", "WinWriteFile"])
def test_win_writers_bad_path_raises(cls):
    """The ctypes signatures load on Windows; a missing device gives a clear OSError."""
    with pytest.raises(OSError, match="CreateFileW failed"):
        getattr(device_mod, cls)(b"\\\\?\\HID#VID_057E&PID_0306#no-such-device")
