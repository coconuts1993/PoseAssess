"""``WiiController``: the ONE shared Wii Balance Board connection of the app (``AppState.wii``).

Owns at most one ``ForceSource`` at a time (plug-and-play ``WiiAutoConnect`` by default, a
manually chosen ``BalanceBoardHID``/``WiiAutoConnect(path=...)``, or ``SimulatedBoard``), keeps the
per-device tare memory across source changes (``ForceSource.adopt_tares``) and re-emits the
reader thread's state changes as Qt signals on the GUI thread. Pages never create ForceSources
themselves: they call this controller and poll ``latest()`` / ``recent()`` from a QTimer.

Optional by design: nothing here imports ``hid``; when hidapi is missing the auto-connect stops
with ``fatal_error`` and ``status_text()`` explains it, the simulator still works.

Environment: ``POSEASSESS_WII`` = ``0``/``off`` (no auto-connect at start; tests and screenshots),
``sim`` (start the simulator), anything else / unset: plug-and-play auto-connect.

Recording lock: while ``lock_tare(reason)`` is set (a take is being recorded), the source cannot
be replaced, disconnected or re-configured (``start_*``, ``use_source``, ``connect_device``,
``disconnect``, ``set_min_load`` raise RuntimeError): simulated or another board's samples would
end up in the same wii.csv as the real data. Only ``connect_device(..., allow_during_lock=True)``
(the confirmed "Connect" of a board that dropped out) and ``use_source(...,
allow_during_lock=True)`` bypass it.

Threading (PoseBoard 3ea6d8a ``gui/app.py`` pattern): the ForceSource reader thread only calls
the status listener installed by ``_set_source``, which emits the private ``_forward`` signal;
the queued connection delivers it to ``_on_source_state`` in the GUI thread, which is the only
place that emits the public signals.
"""

from __future__ import annotations

import logging
import os
import time
from types import SimpleNamespace

import numpy as np
from PySide6.QtCore import QObject, Qt, QTimer, Signal

log = logging.getLogger(__name__)

STALE_FORCE_S = 1.0  # a sample older than this is never shown as live
STATUS_REFRESH_MS = 1000  # periodic status text refresh (weight, battery) while a source exists
PAIRING_HINT = ("Pair the board once via Bluetooth (Windows: Settings > Bluetooth > Add device, "
                "press the red SYNC button in the battery compartment, leave the PIN empty), "
                "then switch it on: it connects automatically.")


def wii_start_mode() -> str:
    """``"off"`` | ``"sim"`` | ``"auto"`` from the ``POSEASSESS_WII`` environment variable."""
    v = os.environ.get("POSEASSESS_WII", "").strip().lower()
    if v in ("0", "off", "false", "no", "none"):
        return "off"
    if v in ("sim", "simulator", "simulate"):
        return "sim"
    return "auto"


def battery_percent(raw) -> int | None:
    """Battery byte of the status report -> percent (WiimoteLib scale: 0xC0 = 100 %)."""
    if raw is None:
        return None
    try:
        return int(max(0, min(100, round(float(raw) * 100.0 / 192.0))))
    except (TypeError, ValueError):
        return None


class WiiController(QObject):
    """Shared Balance Board service. All signals are emitted on the GUI thread.

    Signals
    -------
    state_changed(str)     raw ForceSource state: 'stopped' | 'searching' | 'connecting' |
                           'connected' | 'error: <msg>'
    status_changed(str)    human-readable one-line status for labels / the status bar
    source_changed(object) the ForceSource now in use (or None); a running recording must
                           ``attach_force`` it
    connection_event(str)  "wii_connected" | "wii_disconnected" (log it as a recording event)
    tare_changed(object)   new tare (np.ndarray (4,) kg)
    tare_lock_changed(object)  the tare lock reason (str) or None (addition to the design)
    """

    state_changed = Signal(str)
    status_changed = Signal(str)
    source_changed = Signal(object)
    connection_event = Signal(str)
    tare_changed = Signal(object)
    tare_lock_changed = Signal(object)
    # (source, state): emitted from the reader thread, delivered queued in the GUI thread
    _forward = Signal(object, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._source = None
        self._listener = None
        self._state = "stopped"  # last state of the current source seen in the GUI thread
        self._tare_lock_reason: str | None = None
        self._tare_memory: dict[str, np.ndarray] = {}  # device_key -> tare (kept across sources)
        self._last_status = ""
        self._min_total_kg: float | None = None  # None: ForceSource default
        self._sensor_m: tuple[float, float] | None = None  # (dx, dy); None: ForceSource default
        self.poll_interval_s = 2.0  # auto-connect: how often to look for a paired board
        self.retry_failed_s = 10.0
        self.last_scan_error: str | None = None
        self._forward.connect(self._on_source_state, Qt.QueuedConnection)
        self._timer = QTimer(self)
        self._timer.setInterval(STATUS_REFRESH_MS)
        self._timer.timeout.connect(self._emit_status)

    # ------------------------------------------------------------- properties
    @property
    def source(self):
        """Current ``ForceSource`` or None."""
        return self._source

    @property
    def connected(self) -> bool:
        s = self._source
        return bool(s is not None and s.connected)

    @property
    def state(self) -> str:
        """Current state of the source ('stopped' without one); read live, thread-safe."""
        s = self._source
        return "stopped" if s is None else s.status

    @property
    def auto_connect(self) -> bool:
        """True while the plug-and-play auto-connect source is in use."""
        from poseassess.wii.device import WiiAutoConnect

        return isinstance(self._source, WiiAutoConnect)

    @property
    def is_simulator(self) -> bool:
        from poseassess.wii.device import SimulatedBoard

        return isinstance(self._source, SimulatedBoard)

    @property
    def fatal_error(self) -> str | None:
        """Why the Balance Board cannot be used at all (hidapi missing / unusable), else None."""
        return getattr(self._source, "fatal_error", None)

    @property
    def min_load(self) -> float:
        """COP threshold (kg) applied to the sources."""
        if self._min_total_kg is not None:
            return self._min_total_kg
        from poseassess.wii.device import DEFAULT_COP_MIN_KG

        return DEFAULT_COP_MIN_KG

    def hidapi_status(self) -> tuple[bool, str]:
        """``(usable, reason)`` (``poseassess.wii.device.hidapi_status``). Imports hid when
        available: call it on demand only."""
        from poseassess.wii.device import hidapi_status

        return hidapi_status()

    # ---------------------------------------------------------------- control
    def start_default(self) -> None:
        """Called once by MainWindow after startup: auto-connect / simulator / nothing according
        to ``wii_start_mode()``. Does nothing when a source was already chosen."""
        if self._source is not None:
            return
        mode = wii_start_mode()
        if mode == "sim":
            self.start_simulator()
        elif mode == "auto":
            self.start_auto()

    def start_auto(self, path: bytes | None = None) -> None:
        """Replace the current source by ``WiiAutoConnect(path)`` (plug-and-play; ``path`` None =
        first board found) and start it. RuntimeError while recording."""
        from poseassess.wii.device import WiiAutoConnect

        self._check_unlocked("switch to auto-connect")
        self._set_source(WiiAutoConnect(path, poll_interval_s=self.poll_interval_s,
                                        retry_failed_s=self.retry_failed_s))

    def connect_device(self, path: bytes | None, reconnect: bool = True,
                       allow_during_lock: bool = False) -> None:
        """Use one specific HID device (``path`` None: the first board found). ``reconnect``
        (default): keeps reconnecting to it after drops (``WiiAutoConnect(path)``); False: a
        one-shot ``BalanceBoardHID(path)`` connection. RuntimeError while recording unless
        ``allow_during_lock`` (the user confirmed that the take's board dropped out)."""
        from poseassess.wii.device import BalanceBoardHID, WiiAutoConnect

        if not allow_during_lock:
            self._check_unlocked("connect another board")
        if reconnect:
            src = WiiAutoConnect(path, poll_interval_s=self.poll_interval_s,
                                 retry_failed_s=self.retry_failed_s)
        else:
            src = BalanceBoardHID(path)
        self._set_source(src, allow_during_lock=allow_during_lock)

    def start_simulator(self, mass_kg: float = 70.0) -> None:
        """Replace the current source by a ``SimulatedBoard``. RuntimeError while recording
        (never mixed into a real take)."""
        from poseassess.wii.device import SimulatedBoard

        self._check_unlocked("start the simulator")
        self._set_source(SimulatedBoard(mass_kg=mass_kg))

    def use_source(self, src, allow_during_lock: bool = False) -> None:
        """Use any (not yet started) ``ForceSource`` instance, e.g. a test double or a plugin;
        it is started, and replaces the current source like the methods above. Addition to
        the design. RuntimeError while recording unless ``allow_during_lock``."""
        self._set_source(src, allow_during_lock=allow_during_lock)

    def disconnect(self, allow_during_lock: bool = False) -> None:
        """Stop and drop the current source (tare memory is kept). RuntimeError while
        recording unless ``allow_during_lock``."""
        if self._source is None:
            return
        if not allow_during_lock:
            self._check_unlocked("disconnect the board")
        self._drop_source()
        self._timer.stop()
        self.source_changed.emit(None)
        self.state_changed.emit("stopped")
        self._emit_status(force=True)

    def list_devices(self) -> list[dict]:
        """HID devices that may be Balance Boards (``BalanceBoardHID.list_devices``); [] when
        hidapi is unusable (the reason is kept in ``last_scan_error``)."""
        from poseassess.wii.device import BalanceBoardHID

        try:
            devs = BalanceBoardHID.list_devices()
        except Exception as e:  # noqa: BLE001  (hidapi missing / DLL fails / HID error)
            self.last_scan_error = str(e) or type(e).__name__
            return []
        self.last_scan_error = None
        return list(devs)

    def tare(self, seconds: float = 1.0):
        """Zero the board with the last ``seconds`` of data (board empty). RuntimeError when
        tare is locked (recording running) or no data. Emits ``tare_changed``."""
        if self._tare_lock_reason:
            raise RuntimeError(f"Tare is not possible while {self._tare_lock_reason} is running "
                               "(it would change the zero in the middle of the data). Tare "
                               "before starting.")
        src = self._source
        if src is None or not src.connected:
            raise RuntimeError("No Wii Balance Board connected: connect it first.")
        if not src.recent(seconds):
            raise RuntimeError("No data from the Wii Balance Board yet: wait a second and "
                               "try again.")
        tare = np.array(src.do_tare(seconds), float)
        self._tare_memory.update({k: np.array(v, float) for k, v in src._tares.items()})
        self.tare_changed.emit(tare.copy())
        self._emit_status(force=True)
        return tare

    def set_min_load(self, kg: float) -> None:
        """COP threshold (kg) of the current and future sources. RuntimeError while recording
        (the COP columns of one wii.csv would use two thresholds)."""
        if self._min_total_kg is None or float(kg) != self._min_total_kg:
            self._check_unlocked("change the COP threshold")
        self._min_total_kg = float(kg)
        if self._source is not None:
            self._source.min_total_kg = float(kg)

    def set_sensor_spacing(self, dx_m: float, dy_m: float) -> None:
        """Sensor centre spacing (m, left-right / front-back) used for the COP of the current
        and future sources (``BoardGeometry.sensor_dx_mm / 1000``...). Addition to the design."""
        self._sensor_m = (float(dx_m), float(dy_m))
        if self._source is not None:
            self._source.sensor_dx_m, self._source.sensor_dy_m = self._sensor_m

    def lock_tare(self, reason: str | None) -> None:
        """Disable taring while ``reason`` is set (e.g. "recording"); None unlocks."""
        reason = reason or None
        if reason == self._tare_lock_reason:
            return
        self._tare_lock_reason = reason
        self.tare_lock_changed.emit(reason)

    @property
    def tare_locked(self) -> str | None:
        return self._tare_lock_reason

    def _check_unlocked(self, what: str) -> None:
        reason = self._tare_lock_reason
        if reason:
            raise RuntimeError(f"Cannot {what} while {reason} is running: the data of the "
                               "take would come from two sources. Stop the recording first.")

    @property
    def tare_values(self) -> np.ndarray:
        """Tare (kg, TR, BR, TL, BL) of the current source (zeros without one)."""
        s = self._source
        return np.zeros(4) if s is None else np.array(s.tare, float)

    # ------------------------------------------------------------------- data
    def latest(self, max_age_s: float = STALE_FORCE_S):
        """Newest ``ForceSample`` if not older than ``max_age_s``, else None."""
        src = self._source
        if src is None:
            return None
        s = src.latest()
        if s is None or time.perf_counter() - s.t > max_age_s:
            return None
        return s

    def recent(self, seconds: float) -> list:
        """Samples of the last ``seconds`` (thread-safe copy); [] without a source."""
        src = self._source
        return [] if src is None else src.recent(seconds)

    def status_text(self) -> str:
        """One line, e.g. "Wii: connected · 71.3 kg · battery 80%", "Wii: searching for a
        paired board…", "Wii: off (hidapi not installed — pip install hidapi)"."""
        src = self._source
        if src is None:
            return "Wii: off"
        fatal = getattr(src, "fatal_error", None)
        if fatal:
            return f"Wii: off ({_short_hid_reason(fatal)})"
        st = src.status
        if st == "searching":
            return "Wii: searching for a paired board…"
        if st == "connecting":
            return "Wii: connecting…"
        if st == "connected":
            head = "Wii: simulator" if self.is_simulator else "Wii: connected"
            parts = [head]
            s = self.latest()
            if s is not None and np.isfinite(s.total_kg):
                parts.append(f"{s.total_kg + 0.0:.1f} kg".replace("-0.0 ", "0.0 "))
            bat = battery_percent(getattr(src, "battery", None))
            if bat is not None:
                parts.append(f"battery {bat}%")
            return " · ".join(parts)
        if st.startswith("error"):
            msg = st[len("error: "):] if st.startswith("error: ") else st
            if self.auto_connect:
                return f"Wii: error, retrying ({msg})"
            return f"Wii: error ({msg})"
        return f"Wii: {st}"

    def detail_text(self, sample=None) -> str:
        """Multi-line text for the device panel: source, state, sensor values, battery,
        connection counters, last error, hints (addition to the design)."""
        src = self._source
        if src is None:
            return ("Not connected (the Balance Board is optional).\n"
                    "Tick Auto-connect or click Use simulator.")
        from poseassess.wii.device import BalanceBoardHID, SimulatedBoard

        kind = ("Simulator" if isinstance(src, SimulatedBoard)
                else "Auto-connect" if self.auto_connect
                else "Board" if isinstance(src, BalanceBoardHID) else type(src).__name__)
        fatal = getattr(src, "fatal_error", None)
        if fatal:
            return (f"{kind}: stopped - {fatal}.\nThe Balance Board cannot be used without "
                    "hidapi; the simulator still works.\n"
                    "Install: venv\\Scripts\\python -m pip install hidapi")
        st = src.status
        lines = []
        if st == "searching":
            lines.append("Auto-connect: searching for a paired board.\n"
                         "Switch the board on (power button).")
            last = getattr(src, "last_error", None) or ""
            if last.startswith("board not responding"):
                lines.append("A paired board is listed but does not answer: switch it on, or "
                             "pair it again (often needed after it was switched off).")
            elif "not a Balance Board" in last:
                lines.append(last)
        elif st == "connecting":
            lines.append(f"{kind}: connecting...")
        elif st.startswith("error"):
            retry = " (retrying)" if self.auto_connect else ""
            lines.append(f"{kind} error{retry}: {st[len('error: '):]}")
        elif st != "connected":
            lines.append(f"{kind}: {st}")
        else:
            bat = battery_percent(getattr(src, "battery", None))
            lines.append(f"{kind}: connected" + (f", battery {bat}%" if bat is not None else ""))
            s = sample if sample is not None else self.latest()
            if s is None:
                lines.append("waiting for data...")
            else:
                k, cop = s.kg, s.cop_board
                lines.append(f"TL {k[2]:6.2f}   TR {k[0]:6.2f}")
                lines.append(f"BL {k[3]:6.2f}   BR {k[1]:6.2f} kg")
                lines.append(f"Total {s.total_kg:6.2f} kg")
                if np.all(np.isfinite(cop)):
                    lines.append(f"COP x={cop[0] * 1000:7.1f} y={cop[1] * 1000:7.1f} mm")
                else:
                    lines.append(f"COP -   (load < {src.min_total_kg:g} kg)")
        if hasattr(src, "connections"):
            lines.append(f"connections {src.connections}, drops {src.disconnects}")
            last = getattr(src, "last_error", None)
            if last and st == "connected":
                lines.append(f"last problem: {last}")
        tare = np.asarray(src.tare, float)
        if np.any(tare):
            lines.append("tare " + " ".join(f"{v:.2f}" for v in tare) + " kg")
        if self._tare_lock_reason:
            lines.append(f"tare locked ({self._tare_lock_reason})")
        return "\n".join(lines)

    def shutdown(self) -> None:
        """Stop the source (called on app exit). Safe to call more than once."""
        self._timer.stop()
        if self._source is not None:
            self._drop_source(log_event=False)

    # --------------------------------------------------------------- internal
    def _set_source(self, src, allow_during_lock: bool = False) -> None:
        """Make ``src`` the only running source (the previous one is stopped). The tare of
        each board is carried over, so reconnecting the same board keeps its zero.
        RuntimeError while recording unless ``allow_during_lock``."""
        if not allow_during_lock:
            self._check_unlocked("change the Balance Board source")
        old = self._source
        if old is not None:
            self._drop_source()
        src.adopt_tares(old)
        src.adopt_tares(SimpleNamespace(_tares=self._tare_memory))
        if self._min_total_kg is not None:
            src.min_total_kg = self._min_total_kg
        if self._sensor_m is not None:
            src.sensor_dx_m, src.sensor_dy_m = self._sensor_m

        def listener(state: str, src=src):  # reader thread -> GUI thread (queued)
            try:
                self._forward.emit(src, state)
            except RuntimeError:  # controller already deleted (app exit)
                pass

        src.add_status_listener(listener)
        self._listener = listener
        self._source = src
        self._state = "stopped"
        try:
            src.start()
        finally:
            self._timer.start()
            self.source_changed.emit(src)
            self._emit_status(force=True)

    def _drop_source(self, log_event: bool = True) -> None:
        """Stop the current source without emitting ``source_changed`` (callers do)."""
        src, self._source = self._source, None
        was_connected = self._state == "connected"
        self._state = "stopped"
        if src is None:
            return
        if self._listener is not None:
            src.remove_status_listener(self._listener)
            self._listener = None
        self._tare_memory.update({k: np.array(v, float) for k, v in src._tares.items()})
        try:
            src.stop()
        except Exception:  # noqa: BLE001
            log.exception("stopping the Wii source failed")
        if was_connected and log_event:
            self.connection_event.emit("wii_disconnected")

    def _on_source_state(self, src, state: str) -> None:
        """Connection state change of ``src`` (GUI thread)."""
        if src is not self._source:
            return  # late message from a source that has been replaced
        prev, self._state = self._state, state
        self.state_changed.emit(state)
        connected = state == "connected"
        if connected and prev != "connected":
            self.connection_event.emit("wii_connected")
        elif prev == "connected" and not connected:
            self.connection_event.emit("wii_disconnected")
        self._emit_status(force=True)

    def _emit_status(self, force: bool = False) -> None:
        try:
            text = self.status_text()
        except Exception as e:  # noqa: BLE001  (never break the GUI for a status line)
            text = f"Wii: {e}"
        if force or text != self._last_status:
            self._last_status = text
            self.status_changed.emit(text)


def _short_hid_reason(reason: str) -> str:
    from poseassess.wii.device import HIDAPI_MISSING

    if reason == HIDAPI_MISSING:
        return "hidapi not installed — pip install hidapi"
    return reason
