"""Wii-only command-line recorder: ``python -m poseassess.wii`` (or ``run_wii.bat``).

Port of PoseBoard 3ea6d8a ``poseboard/wii/record.py`` on top of ``WiiRecorder``. Use it when the
videos are recorded by another program: record the Wii data here, import the videos on the
Videos page, then align both on the Results page ("Balance (Wii)" tab).

Usage::

    python -m poseassess.wii [--project DIR | --out DIR] [--subject S] [--notes N]
                             [--seconds N] [--simulate] [--wait S] [--tare S] [--min-kg KG]
                             [--device PATH|N] [--list]

* ``--project DIR``: record into ``DIR/wii/recordings/<stamp>/`` of a PoseAssess project (it does
  NOT change the project's active trial; pick the recording on the Results page). The project's
  board model (``wii/board/board.json``) is used for the sensor spacing and, when it is up to
  date, to fill the (informative) COP world columns.
* ``--out DIR``: record into ``DIR/<stamp>/`` (default ``./recordings``).
* While recording: type a label + Enter to add an event (e.g. ``sync``); an empty line (or
  Ctrl+C / closing the console) stops. Lines typed before recording starts are ignored.

Works like a stand-alone Wii Balance Board recorder: start it, switch the board on (it must have
been paired via Bluetooth) and it connects by itself; after a Bluetooth drop it reconnects
automatically, and the drop / reconnection are logged as ``wii_disconnected`` /
``wii_connected`` events. Files use the ``poseassess.wii.io`` recording format (session.json,
wii.csv with t / t_rel / t_unix, events.csv).

Exit codes: 0 recorded, 2 no board / hidapi unusable (also ``--list`` when the devices cannot be
enumerated), 3 error before recording (e.g. tare failed, not a project, folder not writable),
4 recorded but no samples, 130 interrupted before recording.
"""

from __future__ import annotations

import argparse
import signal
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from poseassess.wii.device import (
    DEFAULT_COP_MIN_KG,
    BalanceBoardHID,
    ForceSource,
    SimulatedBoard,
    WiiAutoConnect,
)
from poseassess.wii.recorder import WiiRecorder, on_console_close, remove_console_close

EXIT_OK, EXIT_NO_BOARD, EXIT_ERROR, EXIT_NO_SAMPLES, EXIT_INTERRUPTED = 0, 2, 3, 4, 130

PAIRING_HINT = ("Pair the board via Bluetooth (Windows: Settings > Bluetooth > Add device, press "
                "the red SYNC button in the battery compartment, leave the PIN empty; with the "
                "Microsoft Bluetooth stack this usually has to be repeated after the board was "
                "switched off), then switch it on with its power button.")
DEFAULT_OUT = "recordings"
# Default board model (= poseassess.core.balance.geometry.BoardGeometry), used when that module
# cannot be imported (it needs OpenCV): the Wii-only recorder must work anyway.
DEFAULT_SENSOR_DX_MM, DEFAULT_SENSOR_DY_MM = 433.0, 238.0


def build_parser() -> argparse.ArgumentParser:
    """Argument parser (see the module docstring)."""
    ap = argparse.ArgumentParser(
        prog="python -m poseassess.wii",
        description="Record Wii Balance Board force / centre-of-pressure data without cameras "
                    "(e.g. while the videos are recorded by another program). The board is "
                    "detected and connected automatically. While recording, type a label + "
                    "Enter to mark an event (e.g. 'sync'); an empty line or Ctrl+C stops.")
    dest = ap.add_mutually_exclusive_group()
    dest.add_argument("--project", metavar="DIR", default=None,
                      help="PoseAssess project folder: record into DIR/wii/recordings/ (then "
                           "select the recording on '6. Results' > 'Balance (Wii)')")
    dest.add_argument("--out", metavar="DIR", default=None,
                      help=f"output root folder when not using --project (default: "
                           f"./{DEFAULT_OUT})")
    ap.add_argument("--subject", default="",
                    help="subject name, used in the recording folder name")
    ap.add_argument("--notes", default="", help="free-text notes stored in session.json")
    ap.add_argument("--seconds", type=float, default=None,
                    help="stop automatically after N seconds (default: until Enter / Ctrl+C)")
    ap.add_argument("--simulate", action="store_true", help="use the built-in board simulator")
    ap.add_argument("--wait", type=float, default=None,
                    help="max seconds to wait for a board (default: wait forever)")
    ap.add_argument("--tare", type=float, default=0.0, metavar="S",
                    help="zero the board over S seconds before recording (board must be empty)")
    ap.add_argument("--min-kg", type=float, default=DEFAULT_COP_MIN_KG, metavar="KG",
                    help=f"COP is left empty below this total load (default "
                         f"{DEFAULT_COP_MIN_KG:g} kg, as in the GUI)")
    ap.add_argument("--device", default=None, metavar="PATH|N",
                    help="only use this board: its number or HID path from --list (quote the "
                         "path); default: first board found")
    ap.add_argument("--list", action="store_true", help="list detected boards and exit")
    return ap


def _path_text(path) -> str:
    return path.decode("utf-8", "replace") if isinstance(path, (bytes, bytearray)) else str(path)


def list_boards() -> int:
    """Print the paired boards (``--list``)."""
    try:
        devs = BalanceBoardHID.list_devices()
    except Exception as e:  # noqa: BLE001
        print(f"Cannot enumerate HID devices: {e}")
        return EXIT_NO_BOARD
    if not devs:
        print("No Wii Balance Board found. " + PAIRING_HINT)
        return EXIT_OK
    for i, d in enumerate(devs):
        print(f"[{i}] {d.get('product_string') or 'Balance Board'}  "
              f"serial={d.get('serial_number') or '-'}  path={_path_text(d.get('path'))}")
    return EXIT_OK


def _live_line(src: ForceSource, elapsed: float | None = None) -> str:
    s = src.latest()
    head = f"{elapsed:7.1f} s  " if elapsed is not None else ""
    if s is None:
        return f"{head}[{src.status}] no data yet"
    x, y = s.cop_board
    # Nearest sensor: press a corner to check the orientation (TL/TR = edge opposite the
    # power button, TR/BR = right side for a subject facing that edge)
    near = ("T" if y >= 0 else "B") + ("R" if x >= 0 else "L")
    cop = ("COP (nobody on the board)" if not np.isfinite(x)
           else f"COP x={x * 1000:6.1f} y={y * 1000:6.1f} mm ({near})")
    bat = getattr(src, "battery", None)
    tail = f"  battery {bat}" if bat is not None else ""
    return f"{head}total {s.total_kg:7.2f} kg  {cop}  [{src.status}]{tail}"


def _sleep(stop: threading.Event, seconds: float) -> None:
    """Sleep in short steps (Ctrl+C stays responsive on Windows)."""
    end = time.monotonic() + seconds
    while not stop.is_set() and time.monotonic() < end:
        time.sleep(min(0.05, max(0.0, end - time.monotonic())))


def wait_for_board(src: ForceSource, wait_s: float | None,
                   stop: threading.Event | None = None) -> bool:
    """Block until ``src`` is connected and delivering samples; False on timeout, when ``stop``
    is set, or when the source can never connect (e.g. hidapi missing)."""
    deadline = None if wait_s is None else time.monotonic() + wait_s
    seen: set[str] = set()
    while True:
        if getattr(src, "fatal_error", None) or (stop is not None and stop.is_set()):
            return False
        status = src.status
        if status not in seen:  # each message once (a paired but switched-off board cycles)
            seen.add(status)
            if status == "searching":
                print("Waiting for the Wii Balance Board... switch it on. " + PAIRING_HINT)
            elif status != "connected":
                print(f"Board: {status}")
        if status == "connected" and src.latest() is not None:
            return True
        if src.error or (deadline is not None and time.monotonic() > deadline):
            return False
        time.sleep(0.05)


def _read_stdin(on_line) -> None:
    """Forward stdin lines to ``on_line`` until EOF (no usable stdin: do nothing)."""
    try:
        for line in sys.stdin:
            on_line(line.strip())
    except (OSError, ValueError, AttributeError, TypeError):
        pass  # no usable stdin (e.g. under a test runner or a service)


def _resolve_device(arg: str | None) -> bytes | None:
    """``--device``: a number from ``--list`` or an HID path. Warns if no listed board matches;
    ValueError for a board number that does not exist."""
    if not arg:
        return None
    try:
        devs = BalanceBoardHID.list_devices()
    except Exception as e:  # noqa: BLE001
        print(f"Cannot enumerate HID devices: {e}")
        devs = []
    if arg.strip().isdigit():
        i = int(arg)
        if i >= len(devs):
            raise ValueError(f"No board number {i} (see --list: {len(devs)} board(s) found)")
        return devs[i]["path"]
    path = arg.encode()
    if not any(d.get("path") == path for d in devs):
        print("Warning: no paired board has this path right now (see --list; quote the path, "
              "e.g. --device \"<path>\"). Waiting for it anyway...")
    return path


def _install_close_handlers(stop: threading.Event, on_close) -> list:
    """Ctrl+Break / SIGTERM end the recording normally (``stop``). Closing the console window
    (Windows) gives the process only a few seconds: ``on_close`` then closes the files at once.
    Returns what is needed to restore the previous handlers."""
    restore = []
    for name in ("SIGBREAK", "SIGTERM"):
        sig = getattr(signal, name, None)
        if sig is None:
            continue
        try:
            restore.append((sig, signal.signal(sig, lambda *_: stop.set())))
        except (ValueError, OSError):  # not the main thread
            pass

    def close():
        stop.set()
        on_close()

    handle = on_console_close(close)
    if handle is not None:
        restore.append(("console", handle))
    return restore


def _restore_handlers(restore: list) -> None:
    for sig, old in restore:
        try:
            if sig == "console":
                remove_console_close(old)
            else:
                signal.signal(sig, old)
        except Exception:  # noqa: BLE001
            pass


class _Destination:
    """Where and with which board model to record (``--project`` / ``--out``)."""

    def __init__(self, root: Path, project=None, geometry=None, board_pose=None):
        self.root = root
        self.project = project
        self.geometry = geometry
        self.board_pose = board_pose
        self.notes: list[str] = []

    @property
    def sensor_spacing_m(self) -> tuple[float, float]:
        g = self.geometry
        if g is None:
            return DEFAULT_SENSOR_DX_MM / 1000, DEFAULT_SENSOR_DY_MM / 1000
        return float(g.sensor_dx_mm) / 1000, float(g.sensor_dy_mm) / 1000

    def extra_meta(self) -> dict:
        extra: dict = {"recorded_with": "python -m poseassess.wii"}
        if self.project is not None:
            extra["project"] = {"name": self.project.config.name,
                                "root": str(Path(self.project.root).resolve())}
        return extra


def _default_geometry():
    try:
        from poseassess.core.balance.geometry import BoardGeometry
    except Exception:  # noqa: BLE001 - e.g. OpenCV unusable: record with the default spacing
        return None
    return BoardGeometry()


def _destination(args) -> _Destination:
    """Resolve ``--project`` / ``--out``; ValueError with a readable message."""
    if not args.project:
        return _Destination(Path(args.out or DEFAULT_OUT), geometry=_default_geometry())
    from poseassess.core.balance.paths import WiiPaths
    from poseassess.core.project import Project

    root = Path(args.project).expanduser()
    if not Project.is_project(root):
        raise ValueError(f"Not a PoseAssess project (no project.toml): {root}")
    try:
        project = Project.load(root)
    except Exception as e:  # noqa: BLE001
        raise ValueError(f"Cannot open the project {root}: {e}") from e
    dest = _Destination(WiiPaths(project).recordings_dir, project, _default_geometry())
    # The project's board model (sensor spacing) and, when current, its position: the COP world
    # columns are then filled (informative only; PoseAssess recomputes them from board.json).
    try:
        from poseassess.core.balance.board import load_board

        reg = load_board(project)
    except Exception as e:  # noqa: BLE001
        reg = None
        dest.notes.append(f"Note: the board position could not be loaded ({e}).")
    if reg is not None:
        dest.geometry = reg.geometry
        if reg.stale:
            dest.notes.append(f"Note: {reg.stale} The COP world columns stay empty.")
        else:
            dest.board_pose = reg.pose
    return dest


def main(argv: list[str] | None = None) -> int:
    """Run the CLI; returns an exit code (see the module docstring)."""
    args = build_parser().parse_args(argv)
    if args.list:
        return list_boards()
    try:
        dest = _destination(args)
    except ValueError as e:
        print(e)
        return EXIT_ERROR

    if args.simulate:
        src: ForceSource = SimulatedBoard()
    else:
        try:
            src = WiiAutoConnect(path=_resolve_device(args.device))
        except ValueError as e:
            print(e)
            return EXIT_NO_BOARD
    src.sensor_dx_m, src.sensor_dy_m = dest.sensor_spacing_m
    src.min_total_kg = args.min_kg

    rec = WiiRecorder()
    stop = threading.Event()
    recording = threading.Event()  # stdin lines before the recording starts are ignored
    folder: Path | None = None

    def on_line(line: str) -> None:
        if not recording.is_set():
            return
        if not line:
            stop.set()
            return
        try:
            ev = rec.add_event(line)
        except OSError as e:
            print(f"Cannot save the event: {e}")
            return
        if ev is not None:
            print(f"Event '{ev['label']}' at t_rel={ev['t_rel']:.3f} s")

    # Read stdin from the start, so Enter pressed while waiting or taring is consumed (and ignored)
    threading.Thread(target=_read_stdin, args=(on_line,), name="wii-cli-stdin",
                     daemon=True).start()
    restore = _install_close_handlers(stop, rec.stop)
    src.start()
    try:
        if dest.project is not None:
            print(f"Project '{dest.project.config.name}': recording into {dest.root}")
        for note in dest.notes:
            print(note)
        if not wait_for_board(src, args.wait, stop):
            if stop.is_set():
                return EXIT_INTERRUPTED
            fatal = getattr(src, "fatal_error", None)
            print(f"No board connected ({fatal or src.status}). "
                  + ("" if fatal else PAIRING_HINT))
            return EXIT_NO_BOARD
        info = src.info()
        print(f"Board connected: {info.get('product_string') or info['type']}"
              + (f"  ({info['device_path']})" if info.get("device_path") else ""))

        if args.tare > 0:
            print(f"Taring for {args.tare:g} s: keep the board EMPTY...")
            _sleep(stop, args.tare)
            if stop.is_set():
                return EXIT_INTERRUPTED
            try:
                tare = src.do_tare(args.tare)
            except RuntimeError as e:
                print(f"Taring failed ({e}): the board sent no data (connection lost?). "
                      "Nothing was recorded.")
                return EXIT_ERROR
            print(f"Tare (TR, BR, TL, BL): {np.round(tare, 2).tolist()} kg")
        else:
            s = src.latest()
            if s is not None and 1.0 < abs(s.total_kg) < 10.0:
                print(f"Warning: the board reads {s.total_kg:.1f} kg. If nobody is on it, this "
                      "is zero drift: use --tare 2 to zero it before recording.")

        try:
            folder = rec.start(root=dest.root, force=src, cams=[], board=dest.board_pose,
                               geometry=dest.geometry, subject=args.subject, notes=args.notes,
                               extra_meta=dest.extra_meta())
        except OSError as e:
            print(f"Cannot create the recording in {dest.root}: {e}")
            return EXIT_ERROR
        recording.set()
        _record(rec, src, stop, args.seconds)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        recording.clear()
        stop.set()
        if folder is not None:
            rec.stop()
        src.stop()
        _restore_handlers(restore)
    if folder is None:
        return EXIT_INTERRUPTED
    n = rec.counts.get("wii", 0)
    print(f"Saved {n} force samples to {folder}")
    if rec.error:
        print(f"Warning: {rec.error}")
    if not n:
        print("Warning: no force samples were recorded (the recording was stopped at once, or "
              "the board sent no data).")
        return EXIT_NO_SAMPLES
    if dest.project is not None:
        print(f"In PoseAssess: open the project, import the videos on '3. Videos', then select "
              f"the recording {folder.name} on '6. Results' > 'Balance (Wii)' and align it.")
    return EXIT_OK


def _add_event(rec: WiiRecorder, label: str) -> None:
    try:
        rec.add_event(label)
    except OSError as e:
        print(f"Cannot save the event '{label}': {e}")


def _record(rec: WiiRecorder, src: ForceSource, stop: threading.Event,
            seconds: float | None) -> None:
    """Print a live line once per second until ``seconds`` elapse, Enter, or a board error.
    Link drops / reconnects are logged in events.csv so gaps in wii.csv are explained."""
    labels = {"searching": "wii_disconnected", "connected": "wii_connected"}
    last: list[str | None] = [None]
    lock = threading.Lock()

    def on_status(state: str) -> None:  # the board's reader thread
        label = labels.get(state)
        if label is None:
            return
        with lock:
            if label == last[0]:
                return
            last[0] = label
        _add_event(rec, label)
        if label == "wii_connected":
            rec.log_force_source(src, "connected")
        print(f"Board {label[4:]}")

    src.add_status_listener(on_status)
    try:
        with lock:
            if last[0] is None:
                last[0] = "wii_connected" if src.status == "connected" else "wii_disconnected"
                lost = last[0] == "wii_disconnected"
            else:
                lost = False
        if lost:
            _add_event(rec, "wii_disconnected")  # already lost between connecting and the start
        started = rec.meta.get("start_time_iso") or datetime.now().isoformat(
            timespec="milliseconds")
        print(f"Recording to {rec.folder}  (started {started})")
        if seconds is None:
            print("Type a label + Enter to add an event marker; press Enter on an empty line "
                  "or Ctrl+C to stop.")
        else:
            print(f"Recording for {seconds:g} s (Enter or Ctrl+C stops early).")
        t_start = time.monotonic()
        next_print = t_start + 1.0
        while not stop.is_set():
            now = time.monotonic()
            if seconds is not None and now - t_start >= seconds:
                break
            if src.error:
                print(f"Board error: {src.error}")
                break
            if now >= next_print:
                print(_live_line(src, now - t_start))
                next_print += 1.0
            time.sleep(0.05)
    finally:
        src.remove_status_listener(on_status)


if __name__ == "__main__":
    raise SystemExit(main())
