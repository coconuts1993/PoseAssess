"""Fake hidapi Balance Board devices for tests (ported from PoseBoard 3ea6d8a,
tests/test_wii_auto.py). Import from any test: ``from tests.wii_fakes import FakeBus``.

``FakeBus(monkeypatch)`` replaces ``BalanceBoardHID.list_devices`` / ``open_device`` so boards can
appear (``bus.present = True``), drop out (``bus.opened[-1].broken = True``) and come back.
"""

import time
from collections import deque

from poseassess.wii import protocol as P
from poseassess.wii.device import BalanceBoardHID

KG0 = [1000, 1100, 1200, 1300]
KG17 = [2700, 2800, 2900, 3000]
KG34 = [4400, 4500, 4600, 4700]
CAL_BYTES = bytes(4) + b"".join(v.to_bytes(2, "big") for v in KG0 + KG17 + KG34) + bytes(4)
CAL = P.Calibration.from_bytes(CAL_BYTES)
PATH = b"fake-board-path"
BOARD = {"path": PATH, "product_id": 0x0306, "product_string": "Nintendo RVL-WBC-01",
         "serial_number": "00191d000000"}


def sensor_report(raw) -> bytes:
    return bytes([P.INPUT_BUTTONS_EXT8, 0, 0]) + b"".join(int(v).to_bytes(2, "big") for v in raw) + bytes(11)


class FakeHid:
    """Stand-in for a hidapi ``hid.device`` connected to a Balance Board."""

    def __init__(self, raw, write_ok=True, silent=False, ext_id=P.BALANCE_BOARD_EXT_ID):
        self.raw = list(raw)
        self.write_ok = write_ok
        self.silent = silent  # never sends sensor data (link stalled)
        self.ext_id = ext_id  # extension type at 0xA400FA (None: no extension -> error 0x7)
        self.broken = False  # set to simulate a Bluetooth drop
        self.closed = False
        self.streaming = False
        self.writes: list[bytes] = []
        self.queue: deque[bytes] = deque()

    def write(self, data):
        if self.closed:
            raise ValueError("not open")
        r = bytes(data)
        self.writes.append(r)
        if not self.write_ok or self.broken:
            return -1
        if r[0] == P.REPORT_READ_MEMORY:
            addr = (r[2] << 16) | (r[3] << 8) | r[4]
            size = (r[5] << 8) | r[6]
            if addr == P.ADDR_EXT_TYPE and self.ext_id is None:
                a = addr & 0xFFFF
                self.queue.append(bytes([P.INPUT_READ_DATA, 0, 0, 0x07, a >> 8, a & 0xFF]) + bytes(16))
                return len(r)
            data = (bytes(self.ext_id) if addr == P.ADDR_EXT_TYPE
                    else CAL_BYTES[addr - P.ADDR_CALIBRATION:])[:size]
            for off in range(0, len(data), 16):
                chunk = data[off:off + 16]
                a = (addr + off) & 0xFFFF
                self.queue.append(bytes([P.INPUT_READ_DATA, 0, 0, (len(chunk) - 1) << 4, a >> 8, a & 0xFF])
                                  + chunk.ljust(16, b"\0"))
        elif r[0] == P.REPORT_MODE:
            self.streaming = True
        elif r[0] == P.REPORT_STATUS_REQUEST and not self.silent:
            self.queue.append(bytes([P.INPUT_STATUS, 0, 0, 0, 0, 0, 0xC0]) + bytes(15))
        return len(r)

    def read(self, n, timeout_ms):
        if self.closed:
            raise ValueError("not open")
        if self.broken:
            raise OSError("read error")
        if self.queue:
            return list(self.queue.popleft())
        time.sleep(0.002 if self.streaming and not self.silent else timeout_ms / 1000)
        if self.streaming and not self.silent:
            return list(sensor_report(self.raw))
        return []

    def close(self):
        self.closed = True


class FakeBus:
    """Monkeypatched device list + opener: boards can appear, drop out and come back."""

    def __init__(self, monkeypatch):
        self.present = False
        self.next_device = lambda: FakeHid([1850, 1950, 2050, 2150])
        self.opened: list[FakeHid] = []
        monkeypatch.setattr(BalanceBoardHID, "list_devices", staticmethod(self.list_devices))
        monkeypatch.setattr(BalanceBoardHID, "open_device", staticmethod(self.open_device))

    def list_devices(self):
        return [dict(BOARD)] if self.present else []

    def open_device(self, path):
        assert path == PATH
        dev = self.next_device()
        self.opened.append(dev)
        return dev


def wait_until(cond, timeout=3.0):
    """Poll ``cond()`` until it is true or ``timeout`` seconds pass; returns the last value."""
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.01)
    return bool(cond())
