"""Synchronized recording of Wii Balance Board data, event markers and (optionally) camera
video into one recording folder, all on the same ``time.perf_counter`` clock.

Port of PoseBoard 3ea6d8a ``poseboard/session.py`` (``SessionRecorder``) WITHOUT the live 3D
pose part (PoseAssess computes pose offline with Pose2Sim) and WITHOUT post-processing
(fusion happens later in ``poseassess.core.balance``). Every stream is optional: Wii-only
(``cams=[]``), video-only (``force=None``) or both.

Files written (names/columns from ``poseassess.wii.io``): session.json, wii.csv, events.csv,
and whatever each camera writes (``CameraStream.start_recording`` -> camNN.mkv +
camNN_timestamps.csv). Rows carry ``t`` / ``t_rel`` / ``t_unix`` (see ``poseassess.wii.io``).

Cameras are duck-typed (``VideoSink``) so this module does not depend on OpenCV capture code:
``poseassess.core.capture.camera.CameraStream`` satisfies it.

Thread safety: ``_on_force`` runs on the ForceSource reader thread; ``add_event``,
``attach_force``, ``update_meta`` and ``stop`` may be called from any thread; CSV writes are
serialized by one lock, session.json writes by another (and replace the file atomically).
wii.csv is flushed about once per second, events immediately, so a crash (or a killed process)
loses at most ~1 s of force data; session.json then lacks only the stop fields.
"""

from __future__ import annotations

import copy
import csv
import logging
import re
import shutil
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from poseassess.wii import io as wio

log = logging.getLogger(__name__)

FLUSH_INTERVAL_S = 1.0


class VideoSink(Protocol):
    """What ``WiiRecorder`` needs from a camera (``CameraStream`` has all of it)."""

    name: str      # "cam01" ... (becomes the video file stem)
    source: Any    # device index / URL / file (stored in session.json)
    fps: float | None

    def start_recording(self, video_path: Path, clock_offset_unix: float | None = None,
                        t0: float | None = None) -> Path: ...

    def stop_recording(self) -> None: ...


def _f(v) -> str:
    return "" if v is None or not np.isfinite(v) else f"{v:.6f}"


def capture_clock_offset(max_wait_s: float = 0.05) -> tuple[float, float]:
    """Read ``time.perf_counter()`` and ``time.time()`` back to back right after the wall clock
    ticks; returns ``(t_perf, t_unix)`` accurate to ~1 ms (port of PoseBoard's function), even
    with a coarse (15.6 ms) ``time.time()``."""
    prev = time.time()
    deadline = time.perf_counter() + max_wait_s
    best: tuple[float, float, float] | None = None
    while True:
        a = time.perf_counter()
        u = time.time()
        b = time.perf_counter()
        if best is None or b - a < best[0]:
            best = (b - a, (a + b) / 2, u)
        if u != prev:  # the wall clock just ticked: u is fresh
            return (a + b) / 2, u
        if b > deadline:
            return best[1], best[2]
        prev = u


def iso_time(t_unix: float) -> str:
    """Local ISO 8601 time with timezone, e.g. ``2026-09-24T15:30:00.123+08:00``."""
    return datetime.fromtimestamp(t_unix).astimezone().isoformat(timespec="milliseconds")


def on_console_close(callback):
    """Windows only: call ``callback`` (in a system thread) when the console window is closed,
    or the user logs off / shuts down (``SetConsoleCtrlHandler``). The process is then ended
    within a few seconds without cleanup, so ``callback`` must close files quickly (e.g.
    ``recorder.stop()``). Returns a handle for ``remove_console_close`` (None if unavailable)."""
    if not sys.platform.startswith("win"):
        return None
    try:
        import ctypes

        @ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_uint)
        def handler(ctrl_type):
            if ctrl_type in (2, 5, 6):  # CTRL_CLOSE_EVENT, CTRL_LOGOFF_EVENT, CTRL_SHUTDOWN_EVENT
                try:
                    callback()
                except Exception:  # noqa: BLE001
                    log.exception("closing the recording failed")
                return 1
            return 0  # Ctrl+C / Ctrl+Break: Python's own handling

        if not ctypes.windll.kernel32.SetConsoleCtrlHandler(handler, True):
            return None
        return handler  # keep a reference: the callback must stay alive
    except Exception:  # noqa: BLE001
        log.debug("console close handler not installed", exc_info=True)
        return None


def remove_console_close(handle) -> None:
    """Undo ``on_console_close`` (None is ignored)."""
    if handle is None:
        return
    try:
        import ctypes

        ctypes.windll.kernel32.SetConsoleCtrlHandler(handle, False)
    except Exception:  # noqa: BLE001
        log.debug("console close handler not removed", exc_info=True)


def _safe_subject(subject: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", subject.strip()).strip("_")


def new_recording_folder(root: str | Path, subject: str = "", t_unix: float | None = None) -> Path:
    """Create and return an empty folder ``root/YYYYmmdd_HHMMSS[_subject]`` (local time of
    ``t_unix``, default now); a second recording in the same second gets ``_2``, ``_3``...
    ``subject`` is sanitized to letters, digits, ``-`` and ``_`` (other characters -> ``_``)."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.fromtimestamp(time.time() if t_unix is None else t_unix).strftime(
        "%Y%m%d_%H%M%S")
    subj = _safe_subject(subject)
    name = f"{stamp}_{subj}" if subj else stamp
    folder, n = root / name, 2
    while True:
        try:
            folder.mkdir()
            return folder
        except FileExistsError:
            folder, n = root / f"{name}_{n}", n + 1


class WiiRecorder:
    """One recording at a time. Typical use::

        rec = WiiRecorder()
        folder = rec.start(folder=None, root=paths.recordings_dir, force=src, cams=cams,
                           board=board_pose, geometry=BoardGeometry(), subject="S01")
        rec.add_event("sync")              # any thread
        rec.attach_force(new_src)          # board (re)connected as a new object mid-recording
        rec.stop()                         # safe to call twice

    Attributes (valid after ``start``): ``folder`` (None when not recording), ``last_folder``
    (the running or last recording), ``t0`` (perf_counter), ``t0_unix``, ``clock_offset_unix``
    (= t0_unix - t0), ``meta`` (the session.json dict), ``counts`` (``{"wii": n, "events": n}``),
    ``recording`` (bool), ``error`` (first file write error of the recording, else None).

    Robustness: a failed ``start`` leaves nothing running (cameras stopped, a folder it created
    removed); a write error on wii.csv (disk full, drive removed) closes wii.csv, sets ``error``
    and is listed in session.json ``errors`` instead of failing in the board's reader thread;
    samples from a source that was replaced (``attach_force``) or from an earlier recording are
    never written.
    """

    def __init__(self) -> None:
        self.folder: Path | None = None
        self.last_folder: Path | None = None
        self.t0: float | None = None
        self.t0_unix: float | None = None
        self.clock_offset_unix: float | None = None
        self.meta: dict = {}
        self.counts: dict[str, int] = {"wii": 0, "events": 0}
        self.error: str | None = None
        self._force = None
        self._listener: tuple | None = None  # (source, callback) registered for this recording
        self._gen = 0  # bumped by start / attach_force / stop: callbacks of older ones are ignored
        self._cams: list = []
        self._board = None
        self._wii_file = self._wii_csv = None
        self._wii_created = False
        self._events_file = self._events_csv = None
        self._flushed = 0.0
        self._lock = threading.Lock()  # recorder state + CSV writes
        self._meta_lock = threading.Lock()  # serializes session.json writes (taken BEFORE _lock)

    @property
    def recording(self) -> bool:
        return self.folder is not None

    def to_unix(self, t: float) -> float:
        """perf_counter time -> Unix time with this recording's fixed offset."""
        off = self.clock_offset_unix
        return t + (off if off is not None else time.time() - time.perf_counter())

    # ------------------------------------------------------------------ start
    def start(self, *, folder: str | Path | None = None, root: str | Path | None = None,
              force=None, cams: list[VideoSink] | None = None, board=None, geometry=None,
              subject: str = "", notes: str = "", extra_meta: dict | None = None) -> Path:
        """Start recording and return the recording folder.

        * ``folder``: use this (empty or new) folder; else ``new_recording_folder(root, subject)``.
        * ``force``: a ``ForceSource`` (may be None: video-only; attach later with
          ``attach_force``). wii.csv is created only when a force source is present.
        * ``cams``: cameras to record (``VideoSink``); each writes ``folder/<name>.mkv`` via
          ``start_recording(path, clock_offset_unix, t0)``. If one fails, the ones already
          started are stopped and the exception propagates (no half-started recording; a folder
          created by this call is removed again).
        * ``board``: a ``BoardPose`` (duck-typed: ``board_to_world.apply`` + ``to_dict``) or
          None. When given, wii.csv COP world columns are filled; otherwise they stay empty
          (PoseAssess recomputes COP world later from the current board registration anyway).
        * ``geometry``: a ``BoardGeometry`` (duck-typed ``to_dict``) or None.
        * ``extra_meta``: merged into session.json (e.g. ``{"project": ..., "capture": ...}``).

        session.json keys written at start (see docs/WII_INTEGRATION.md, "session.json"):
        schema, app, created, subject, notes, clock, t0, t0_unix, clock_offset_unix,
        start_time_iso, has_wii, has_video, camera_names, streams[{name, source, fps, video,
        timestamps}], world_frame, board_geometry, board_pose, force_source,
        force_source_history. Raises RuntimeError when already recording.
        """
        if self.recording:
            raise RuntimeError("Already recording")
        cams = list(cams or [])
        t0, t0_unix = capture_clock_offset()
        created = folder is None
        if folder is None:
            if root is None:
                raise ValueError("start() needs folder or root")
            folder = new_recording_folder(root, subject, t0_unix)
        folder = Path(folder)
        offset = t0_unix - t0
        started: list = []
        wii_file = wii_csv = None
        try:
            folder.mkdir(parents=True, exist_ok=True)
            meta = self._initial_meta(t0, t0_unix, offset, force, cams, board, geometry,
                                      subject, notes)
            meta.update(extra_meta or {})
            for c, st in zip(cams, meta["streams"]):
                path = c.start_recording(folder / f"{c.name}.mkv", clock_offset_unix=offset,
                                         t0=t0)
                started.append(c)
                st["video"] = Path(path).name
                st["timestamps"] = wio.timestamps_csv_name(Path(path).stem)
            wio.write_session_json(folder, meta)
            if force is not None:
                wii_file, wii_csv = self._create_wii_csv(folder)
        except BaseException:
            for c in started:
                try:
                    c.stop_recording()
                except Exception:  # noqa: BLE001
                    log.exception("stopping the recording of %s failed", getattr(c, "name", c))
            if wii_file is not None:
                wii_file.close()
            if created:
                shutil.rmtree(folder, ignore_errors=True)
            raise
        with self._lock:
            self.folder = self.last_folder = folder
            self.t0, self.t0_unix, self.clock_offset_unix = t0, t0_unix, offset
            self.meta = meta
            self.counts = {"wii": 0, "events": 0}
            self.error = None
            self._board, self._force, self._cams = board, force, cams
            self._wii_file, self._wii_csv = wii_file, wii_csv
            self._wii_created = wii_file is not None
            self._events_file = self._events_csv = None
            self._flushed = 0.0
            self._gen += 1
            gen = self._gen
            self._listener = None
        if force is not None:
            self.log_force_source(force, "start")
            self._subscribe(force, gen)
            self._write_meta()  # with the "start" entry of force_source_history
        return folder

    @staticmethod
    def _initial_meta(t0, t0_unix, offset, force, cams, board, geometry, subject, notes) -> dict:
        try:
            from poseassess import __version__ as app_version
        except Exception:  # noqa: BLE001
            app_version = ""
        return {
            "schema": wio.SESSION_SCHEMA,
            "app": f"PoseAssess {app_version}".strip(),
            "created": datetime.fromtimestamp(t0_unix).isoformat(timespec="seconds"),
            "subject": subject,
            "notes": notes,
            "clock": "time.perf_counter (seconds); t_rel = t - t0; "
                     "t_unix = t + clock_offset_unix (Unix seconds, offset fixed at start)",
            "t0": t0,
            "t0_unix": t0_unix,
            "clock_offset_unix": offset,
            "start_time_iso": iso_time(t0_unix),
            "has_wii": force is not None,
            "has_video": bool(cams),
            "camera_names": [c.name for c in cams],
            "streams": [{"name": c.name, "source": str(c.source), "fps": c.fps,
                         "video": None, "timestamps": wio.timestamps_csv_name(c.name)}
                        for c in cams],
            "world_frame": "Calib.toml world (checkerboard), metres",
            "board_geometry": None if geometry is None else geometry.to_dict(),
            "board_pose": None if board is None else board.to_dict(),
            "force_source": None if force is None else force.info(),
            "force_source_history": [],
        }

    @staticmethod
    def _create_wii_csv(folder: Path):
        f = open(folder / wio.WII_CSV, "w", newline="", encoding="utf-8")  # noqa: SIM115
        try:
            w = csv.writer(f)
            w.writerow(wio.WII_HEADER)
            f.flush()
        except BaseException:
            f.close()
            raise
        return f, w

    # ------------------------------------------------------------ force source
    def _subscribe(self, force, gen: int) -> None:
        """Listen to ``force`` for recording generation ``gen`` (ignored once it is stale)."""
        def on_sample(s, _gen=gen):
            self._on_force(s, _gen)

        force.add_listener(on_sample)
        with self._lock:
            current = self._gen == gen and self.folder is not None
            if current:
                self._listener = (force, on_sample)
        if not current:  # stopped / replaced meanwhile
            force.remove_listener(on_sample)

    @staticmethod
    def _unsubscribe(listener) -> None:
        if listener is None:
            return
        src, fn = listener
        try:
            src.remove_listener(fn)
        except Exception:  # noqa: BLE001
            log.exception("removing the force listener failed")

    def attach_force(self, force) -> None:
        """Switch the force source during a recording (board connected / reconnected as a new
        object; None detaches). Samples continue in the same wii.csv (created now if needed);
        logged in ``force_source_history``. No-op when not recording or ``force`` is the
        current one."""
        if not self.recording or force is self._force:
            return
        info = None
        if force is not None:
            try:
                info = force.info()
            except Exception:  # noqa: BLE001
                log.exception("force source info failed")
        with self._lock:
            if self.folder is None or force is self._force:
                return
            self._force = force
            self._gen += 1
            gen = self._gen
            old_listener, self._listener = self._listener, None
            if force is not None:
                if self._wii_csv is None and not self._wii_created:
                    try:
                        self._wii_file, self._wii_csv = self._create_wii_csv(self.folder)
                        self._wii_created = True
                    except OSError as e:
                        self._write_failed_locked(wio.WII_CSV, e)
                self.meta["has_wii"] = True
                if info is not None:
                    self.meta["force_source"] = info
        self._unsubscribe(old_listener)
        if force is not None:
            self.log_force_source(force, "attached")
            self._subscribe(force, gen)
        self._write_meta()

    def log_force_source(self, force, reason: str) -> None:
        """Append ``{t_rel, t_unix, reason, type, device, tare_kg, min_total_kg}`` to
        ``meta["force_source_history"]`` (e.g. reason "start", "attached", "connected",
        "tare"); written to session.json at the next ``update_meta`` / ``stop``."""
        if force is None or not self.recording:
            return
        t = time.perf_counter()
        try:
            info = force.info()
            device = getattr(force, "device_key", None)
        except Exception:  # noqa: BLE001
            log.exception("force source info failed")
            return
        with self._lock:
            if self.folder is None:
                return
            entry = {"t_rel": t - self.t0, "t_unix": t + self.clock_offset_unix,
                     "reason": reason, "type": info.get("type"), "device": device,
                     "tare_kg": info.get("tare_kg"), "min_total_kg": info.get("min_total_kg")}
            self.meta.setdefault("force_source_history", []).append(entry)

    def _on_force(self, s, gen: int | None = None) -> None:
        # Runs on the ForceSource reader thread.
        with self._lock:
            if self._wii_csv is None or (gen is not None and gen != self._gen):
                return
            try:
                cw = [np.nan] * 3
                if self._board is not None and np.all(np.isfinite(s.cop_board)):
                    cw = self._board.board_to_world.apply(
                        np.array([s.cop_board[0], s.cop_board[1], 0.0]))
                t_unix = s.t + self.clock_offset_unix
                self._wii_csv.writerow([f"{s.t:.6f}", f"{s.t - self.t0:.6f}", f"{t_unix:.6f}",
                                        *[_f(v) for v in s.kg], _f(s.total_kg),
                                        _f(s.cop_board[0]), _f(s.cop_board[1]),
                                        *[_f(v) for v in cw], wio.local_time(t_unix)])
                self.counts["wii"] += 1
                if s.t - self._flushed >= FLUSH_INTERVAL_S:
                    self._wii_file.flush()
                    self._flushed = s.t
            except (OSError, ValueError) as e:  # disk full, drive removed, file closed
                self._write_failed_locked(wio.WII_CSV, e)

    def _write_failed_locked(self, what: str, e: Exception) -> None:
        # caller holds self._lock: stop writing wii.csv, keep the recording (events, video) going
        msg = f"{what}: {e}"
        log.error("recording: writing %s failed, no more force data is saved: %s", what, e)
        if self.error is None:
            self.error = msg
        self.meta.setdefault("errors", []).append(msg)
        f, self._wii_file, self._wii_csv = self._wii_file, None, None
        if f is not None:
            try:
                f.close()
            except (OSError, ValueError):
                pass

    # ------------------------------------------------------------------ events
    def add_event(self, label: str, t: float | None = None) -> dict | None:
        """Write an event row (``t`` = perf_counter, default now) to events.csv (created on the
        first event, UTF-8 BOM, flushed immediately). Labels are whitespace-normalized
        ("" -> "event"). Returns ``{t, t_rel, t_unix, label}`` or None when not recording.
        Raises OSError when the file cannot be written (the error is also kept in ``error``)."""
        if t is None:
            t = time.perf_counter()
        with self._lock:
            if self.folder is None:
                return None
            label = " ".join(str(label).split()) or "event"
            ev = {"t": t, "t_rel": t - self.t0, "t_unix": t + self.clock_offset_unix,
                  "label": label}
            try:
                if self._events_csv is None:
                    self._events_file = open(self.folder / wio.EVENTS_CSV, "w",  # noqa: SIM115
                                             newline="", encoding="utf-8-sig")
                    self._events_csv = csv.writer(self._events_file)
                    self._events_csv.writerow(wio.EVENTS_HEADER)
                self._events_csv.writerow([f"{ev['t']:.6f}", f"{ev['t_rel']:.6f}",
                                           f"{ev['t_unix']:.6f}", label,
                                           wio.local_time(ev["t_unix"])])
                self._events_file.flush()  # markers are rare and valuable
            except OSError as e:
                msg = f"{wio.EVENTS_CSV}: {e}"
                if self.error is None:
                    self.error = msg
                self.meta.setdefault("errors", []).append(msg)
                raise
            self.counts["events"] += 1
            return ev

    # -------------------------------------------------------------- metadata
    def _write_meta(self, folder: Path | None = None, raise_errors: bool = False) -> None:
        """Write the current ``meta`` to session.json of ``folder`` (default: the running or
        last recording). Writes are serialized, so an older snapshot never overwrites a newer
        one. A write error is logged and kept in ``error`` (raised with ``raise_errors``)."""
        with self._meta_lock:
            with self._lock:
                folder = folder or self.folder or self.last_folder
                if folder is None:
                    return
                snapshot = copy.deepcopy(self.meta)
            try:
                wio.write_session_json(folder, snapshot)
            except OSError as e:
                log.error("cannot write %s: %s", Path(folder) / wio.SESSION_JSON, e)
                with self._lock:
                    if self.error is None:
                        self.error = f"{wio.SESSION_JSON}: {e}"
                if raise_errors:
                    raise

    def update_meta(self, **items) -> None:
        """Merge ``items`` into session.json now (thread-safe; rewrites the file). After
        ``stop`` it updates the last recording's session.json. Raises OSError when the file
        cannot be written."""
        with self._lock:
            self.meta.update(items)
        self._write_meta(raise_errors=True)

    # ------------------------------------------------------------------- stop
    def stop(self) -> Path | None:
        """Stop: detach from the force source, stop camera recordings, close files and finalize
        session.json (t_stop, t_stop_unix, stop_time_iso, duration_s, samples, has_wii,
        force_source, and ``errors`` when a file could not be written). Returns the folder, or
        None when not recording (safe to call twice, from any thread)."""
        t_stop = time.perf_counter()
        with self._lock:
            folder, self.folder = self.folder, None
            if folder is None:
                return None
            self._gen += 1
            listener, self._listener = self._listener, None
            files = (self._wii_file, self._events_file)
            self._wii_file = self._wii_csv = None
            self._events_file = self._events_csv = None
            force, cams = self._force, self._cams
            self._force, self._cams, self._board = None, [], None
        self._unsubscribe(listener)
        errors = []
        for c in cams:
            try:
                c.stop_recording()
            except Exception as e:  # noqa: BLE001
                log.exception("stopping the recording of %s failed", getattr(c, "name", c))
                errors.append(f"{getattr(c, 'name', c)}: {e}")
        for f in files:
            if f is not None:
                try:
                    f.close()
                except OSError as e:
                    log.error("closing %s failed: %s", getattr(f, "name", f), e)
                    errors.append(f"{Path(getattr(f, 'name', '?')).name}: {e}")
        info = None
        if force is not None:
            try:
                info = force.info()
            except Exception:  # noqa: BLE001
                log.exception("force source info failed")
        with self._lock:
            m = self.meta
            m["t_stop"] = t_stop
            m["t_stop_unix"] = t_stop + self.clock_offset_unix
            m["stop_time_iso"] = iso_time(t_stop + self.clock_offset_unix)
            m["duration_s"] = t_stop - self.t0
            m["samples"] = dict(self.counts)
            m["has_wii"] = self._wii_created
            if info is not None:
                m["force_source"] = info
            if errors:
                m.setdefault("errors", []).extend(errors)
                if self.error is None:
                    self.error = errors[0]
        self._write_meta(folder)
        return folder
