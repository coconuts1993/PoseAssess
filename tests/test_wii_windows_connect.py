"""A board that Windows has paired must be found and connected even when (a) its HID product
string looks like a Wii Remote ("RVL-CNT"), or (b) the Bluetooth stack rejects / ignores
hidapi's WriteFile - then the output reports go through HidD_SetOutputReport (WiimoteLib's
"alternate write method"). A listed board that never answers is reported as such, not as
"searching". ``--diagnose`` shows what happens step by step."""

import sys
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
    """Stand-in for WinOutputReport: delivers the reports to the fake board."""

    instances: list = []

    def __init__(self, dev):
        self.dev, self.writes, self.closed = dev, [], False
        FakeAlt.instances.append(self)

    def write(self, report):
        self.writes.append(bytes(report))
        return FakeHid.write(self.dev, list(P.pad(report))) if self.dev.write_ok else \
            _deliver(self.dev, report)

    def close(self):
        self.closed = True


def _deliver(dev, report):
    ok = dev.write_ok
    dev.write_ok = True
    try:
        return FakeHid.write(dev, list(P.pad(report)))
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
    FakeAlt.instances = []
    monkeypatch.setattr(device_mod, "WinOutputReport", lambda path: FakeAlt(opened[-1]))
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


@pytest.mark.parametrize("cls", [FakeHid, IgnoringHid], ids=["write-fails", "write-ignored"])
def test_falls_back_to_set_output_report(monkeypatch, cls):
    kw = {"write_ok": False} if cls is FakeHid else {}
    _setup(monkeypatch, lambda: cls([1850, 1950, 2050, 2150], **kw))
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
    assert "Result: 1 board(s) delivering data." in text


def test_diagnose_flag(monkeypatch, capsys):
    from poseassess.wii import record_cli

    monkeypatch.setattr(device_mod, "diagnose", lambda: 1)
    assert record_cli.main(["--diagnose"]) == record_cli.EXIT_OK
    monkeypatch.setattr(device_mod, "diagnose", lambda: 0)
    assert record_cli.main(["--diagnose"]) == record_cli.EXIT_NO_BOARD


@pytest.mark.skipif(sys.platform != "win32", reason="Windows HID API")
def test_win_output_report_bad_path_raises():
    """The ctypes signatures load on Windows; a missing device gives a clear OSError."""
    with pytest.raises(OSError, match="CreateFileW failed"):
        device_mod.WinOutputReport(b"\\\\?\\HID#VID_057E&PID_0306#no-such-device")
