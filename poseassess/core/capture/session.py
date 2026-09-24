"""``CaptureSession``: the project's cameras + synchronized recording of takes.

A take is recorded into ``wii/recordings/<YYYYmmdd_HHMMSS[_subject]>/`` by
``poseassess.wii.recorder.WiiRecorder`` (session.json, wii.csv, events.csv) with every enabled
camera writing ``camNN.mkv`` + ``camNN_timestamps.csv`` on the same perf_counter clock. The take
is then exported into ``videos/`` by ``export.export_recording_to_project``.

Threading: each ``CameraStream`` owns a capture thread; ``latest()`` is thread-safe and meant to
be polled by a GUI timer (~15-30 Hz). Starting/stopping must be called from one thread (the GUI
thread). Opening a camera driver can block for seconds, so a GUI opens the cameras in three
steps: ``begin_open`` (GUI thread), ``open_streams`` (a worker thread: every camera in parallel,
one shared deadline) and ``finish_open`` (GUI thread); ``open`` does all three at once. Nothing
here imports Qt.

Camera identity is positional (docs/WII_INTEGRATION.md §3): ``camNN`` of the capture settings is
camera NN of the project, i.e. the NN-th table of Calib.toml and ``videos/camNN.mp4``.
"""

from __future__ import annotations

import gc
import logging
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2

from poseassess.core.balance.paths import WiiPaths, cam_name
from poseassess.wii.recorder import WiiRecorder

from . import camera as camera_mod
from .camera import CameraStream, Frame  # noqa: F401
from .settings import CaptureSettings

log = logging.getLogger(__name__)

CAMERA_LOST_S = 2.0  # no new frame for this long (and 5 frame periods): the camera is "lost"
OPEN_TIMEOUT_S = 8.0  # opening all cameras (in parallel) may take this long before "no response"


def open_streams(streams: dict[str, CameraStream],
                 timeout_s: float = OPEN_TIMEOUT_S) -> dict[str, str | None]:
    """Start every stream on its own daemon thread and wait at most ``timeout_s`` in total
    (blocks: call it from a worker thread). Returns ``{cam: None (running) | error text}``.

    A camera whose driver has not opened by the deadline counts as failed ("does not
    respond"); when its open returns later, its own thread stops it again (the device is
    released, never registered anywhere)."""
    lock = threading.Lock()
    results: dict[str, str | None] = {}
    abandoned: set[str] = set()

    def work(cam: str, s: CameraStream) -> None:
        try:
            s.start()
            err = None
        except Exception as e:  # noqa: BLE001 - one bad camera must not block the others
            err = str(e) or type(e).__name__
        with lock:
            late = cam in abandoned
            if not late:
                results[cam] = err
        if late and err is None:
            s.stop()

    threads = [threading.Thread(target=work, args=(c, st), name=f"open-camera-{c}", daemon=True)
               for c, st in streams.items()]
    for th in threads:
        th.start()
    deadline = time.monotonic() + float(timeout_s)
    for th in threads:
        th.join(max(0.0, deadline - time.monotonic()))
    with lock:
        for c in streams:
            if c not in results:
                abandoned.add(c)
                results[c] = (f"the camera driver did not respond within {timeout_s:g} s "
                              "(unplug and reconnect the camera, or close other programs using "
                              "it)")
        return dict(results)


def probe_cameras(max_index: int = 8, timeout_s: float = 3.0,
                  skip: set[int] | None = None) -> list[dict]:
    """Try device indices 0..max_index-1; return ``[{index, width, height, fps}]`` for those
    that deliver a frame (DirectShow on Windows, like ``camera.open_capture``). Slow (opens
    every device): run it in a worker thread.

    ``skip``: device indices not to open (e.g. the ones this app is already streaming from; a
    webcam usually cannot be opened twice). ``fps`` is None when the driver does not report
    it. A device whose first ``read()`` blocks longer than ``timeout_s`` is left out (its probe
    thread releases it when the read returns)."""
    found = []
    restore = _quiet_opencv()
    try:
        for index in range(int(max_index)):
            if skip and index in skip:
                continue
            info = _probe_one(index, timeout_s)
            if info is not None:
                found.append(info)
    finally:
        restore()
    return found


def _quiet_opencv():
    """Silence OpenCV's warnings about missing device indices while probing; returns a
    function that restores the previous log level."""
    try:
        lg = cv2.utils.logging
        prev = lg.getLogLevel()
        lg.setLogLevel(lg.LOG_LEVEL_SILENT)
        return lambda: lg.setLogLevel(prev)
    except Exception:  # noqa: BLE001 - older OpenCV builds without cv2.utils.logging
        return lambda: None


def _probe_one(index: int, timeout_s: float) -> dict | None:
    result: dict = {}
    lock = threading.Lock()

    def work():
        try:
            cap = camera_mod.open_capture(index)
        except Exception:  # noqa: BLE001 - a broken driver must not stop the scan
            log.debug("probing camera %d failed", index, exc_info=True)
            return
        try:
            if not cap.isOpened():
                return
            ok, img = cap.read()
            if not ok or img is None:
                return
            h, w = img.shape[:2]
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            with lock:
                result.update(index=index, width=int(w), height=int(h),
                              fps=fps if 1 < fps < 1000 else None)
        except Exception:  # noqa: BLE001
            log.debug("probing camera %d failed", index, exc_info=True)
        finally:
            cap.release()  # this thread owns the capture: never released during its read

    th = threading.Thread(target=work, name=f"probe-camera-{index}", daemon=True)
    th.start()
    th.join(timeout_s)
    with lock:
        return dict(result) if result else None


class CaptureSession:
    """Cameras of one project (``settings.cameras`` with ``enabled``) and at most one running
    take. Wii-only takes are allowed (no camera enabled/opened, or ``record_cameras=False``)."""

    def __init__(self, project, settings: CaptureSettings):
        self.project = project
        self.settings = settings
        self.streams: dict[str, CameraStream] = {}
        self._recorder = WiiRecorder()
        self._stream_cfg: dict[str, tuple] = {}  # cam -> settings the stream was opened with
        self._lost: set[str] = set()

    # ---------------------------------------------------------------- cameras
    @property
    def project_cameras(self) -> list[str]:
        """``cam01..camNN`` of the project (``num_cameras``)."""
        n = int(getattr(getattr(self.project, "config", None), "num_cameras", 0) or 0)
        return [cam_name(i) for i in range(1, n + 1)]

    @staticmethod
    def _cfg_key(cs) -> tuple:
        return (cs.source, cs.width, cs.height, cs.fps)

    def open(self, timeout_s: float = OPEN_TIMEOUT_S) -> dict[str, str]:
        """Start a ``CameraStream`` for every enabled camera; returns ``{cam: error}`` for the
        ones that failed (others keep running). Blocks until every camera opened or
        ``timeout_s`` passed (GUI: ``begin_open`` / ``open_streams`` / ``finish_open``).

        Idempotent: running cameras whose settings did not change are kept, cameras that were
        disabled / removed are stopped and cameras whose source, size or fps changed are
        restarted. RuntimeError while recording (the take's cameras cannot change)."""
        new = self.begin_open()
        return self.finish_open(new, open_streams(new, timeout_s) if new else {})

    def begin_open(self) -> dict[str, CameraStream]:
        """First step of ``open`` (caller's thread): stop the cameras that were disabled /
        removed / changed (in parallel) and return the new, NOT started ``CameraStream``s
        ``{cam: stream}`` to start with ``open_streams``. RuntimeError while recording."""
        if self.recording:
            raise RuntimeError("Cannot change the cameras while recording")
        wanted = {c: cs for c, cs in sorted(self.settings.cameras.items()) if cs.enabled}
        self._stop_streams([c for c, s in self.streams.items()
                            if c not in wanted or not s.running
                            or self._stream_cfg.get(c) != self._cfg_key(wanted[c])])
        return {c: CameraStream(c, cs.source, cs.width, cs.height, cs.fps)
                for c, cs in wanted.items() if c not in self.streams}

    def finish_open(self, started: dict[str, CameraStream],
                    results: dict[str, str | None]) -> dict[str, str]:
        """Last step of ``open`` (caller's thread): register the streams of ``begin_open`` that
        ``open_streams`` started; returns ``{cam: error}`` of the others."""
        errors: dict[str, str] = {}
        for c, s in started.items():
            err = results.get(c, "not opened")
            if err is not None or not s.running:
                errors[c] = err or "stopped right after opening"
                continue
            cs = self.settings.cameras.get(c)
            if c in self.streams or cs is None or not cs.enabled or self.recording:
                s.stop()  # the settings changed meanwhile: not wanted any more
                continue
            self.streams[c] = s
            self._stream_cfg[c] = self._cfg_key(cs)
        self._lost.clear()
        return errors

    def _stop_stream(self, cam: str) -> None:
        self._stop_streams([cam])

    def _stop_streams(self, cams: list[str]) -> None:
        """Stop these cameras in parallel: every stop is requested first, then each capture
        thread gets the remainder of ONE shared ``stop_timeout_s``."""
        streams = []
        for c in cams:
            s = self.streams.pop(c, None)
            self._stream_cfg.pop(c, None)
            self._lost.discard(c)
            if s is not None:
                s.request_stop()
                streams.append((c, s))
        deadline = time.monotonic() + CameraStream.stop_timeout_s
        for c, s in streams:
            try:
                s.stop(timeout_s=deadline - time.monotonic())
            except Exception:  # noqa: BLE001
                log.exception("stopping camera %s failed", c)

    def stop_cameras(self) -> None:
        """Stop every camera (not while recording: RuntimeError)."""
        if self.recording:
            raise RuntimeError("Cannot close the cameras while recording")
        self._stop_streams(list(self.streams))

    def close(self) -> None:
        """Stop recording (if any) and all camera threads."""
        try:
            self.stop_recording()
        finally:
            self._stop_streams(list(self.streams))

    def latest(self, cam: str) -> Frame | None:
        """Latest frame of ``cam`` (None when not running / no frame yet)."""
        s = self.streams.get(cam)
        if s is None or not s.running:
            return None
        return s.latest()

    def camera_status(self) -> dict[str, dict]:
        """``{cam: {running, measured_fps, size, dropped_frames, error, frame_age_s}}`` for every
        camera of the settings (plus ``fps`` = nominal fps reported by the camera, ``lost`` and
        ``recording``). ``size`` is ``(w, h)`` of the latest frame or None."""
        out = {}
        now = time.perf_counter()
        rec = set(self._recorder.meta.get("camera_names", [])) if self.recording else set()
        for c in sorted(set(self.settings.cameras) | set(self.streams)):
            s = self.streams.get(c)
            f = s.latest() if s is not None else None
            out[c] = {
                "running": bool(s is not None and s.running),
                "measured_fps": float(s.measured_fps) if s is not None else 0.0,
                "fps": float(s.fps) if s is not None and s.fps else None,
                "size": None if f is None else (int(f.image.shape[1]), int(f.image.shape[0])),
                "dropped_frames": int(s.dropped_frames) if s is not None else 0,
                "error": s.error if s is not None else None,
                "frame_age_s": s.frame_age(now) if s is not None else None,
                "lost": c in self._lost,
                "recording": c in rec,
            }
        return out

    def check_cameras(self, now: float | None = None) -> set[str]:
        """Detect cameras that stopped delivering frames (unplugged, driver stalled); while
        recording, ``camera_lost camNN`` / ``camera_recovered camNN`` events are written.
        Call it from the GUI timer. Returns the cameras currently lost."""
        now = time.perf_counter() if now is None else now
        for c, s in list(self.streams.items()):
            age = s.frame_age(now)
            limit = max(CAMERA_LOST_S, 5.0 / max(float(s.fps or 1.0), 1.0))
            lost = (not s.running) or (age is not None and age > limit)
            if lost and c not in self._lost:
                self._lost.add(c)
                log.warning("camera %s: no new frames", c)
                self._event_quiet(f"camera_lost {c}")
            elif not lost and c in self._lost:
                self._lost.discard(c)
                self._event_quiet(f"camera_recovered {c}")
        return set(self._lost)

    def _event_quiet(self, label: str) -> None:
        if not self.recording:
            return
        try:
            self._recorder.add_event(label)
        except OSError:
            log.exception("cannot write the event %s", label)

    def video_take_problems(self) -> list[str]:
        """Why a video take cannot start now: every project camera (``cam01..camNN``) must be
        enabled, running and delivering frames. [] when ready."""
        out = []
        for c in self.project_cameras:
            cs = self.settings.cameras.get(c)
            s = self.streams.get(c)
            if cs is None or not cs.enabled:
                out.append(f"{c} is not enabled")
            elif s is None or not s.running:
                out.append(f"{c} is not open")
            elif s.latest() is None:
                out.append(f"{c} has not delivered a frame yet")
            elif c in self._lost:
                out.append(f"{c} delivers no new frames")
        return out

    def calib_sizes(self) -> dict[str, list[int]]:
        """``{camNN: [w, h]}`` image sizes of the project's Calib.toml ({} without one)."""
        try:
            from poseassess.core.balance.calib import project_cameras

            cams = project_cameras(self.project)
        except Exception:  # noqa: BLE001 - unreadable calibration: nothing to compare with
            log.debug("Calib.toml not readable", exc_info=True)
            return {}
        return {c: [int(v) for v in cam.image_size] for c, cam in cams.items()
                if all(cam.image_size)}

    def size_warnings(self) -> list[str]:
        """Running cameras whose frame size differs from their Calib.toml size (intrinsics are
        resolution-specific, so the 3D reconstruction would be wrong)."""
        cal = self.calib_sizes()
        out = []
        for c, s in sorted(self.streams.items()):
            f = s.latest()
            if f is None or c not in cal:
                continue
            w, h = int(f.image.shape[1]), int(f.image.shape[0])
            cw, ch = cal[c]
            if (w, h) != (cw, ch):
                out.append(f"{c} delivers {w}x{h} but Calib.toml is for {cw}x{ch}: set the "
                           f"camera to {cw}x{ch} or calibrate again at {w}x{h}.")
        return out

    def save_board_snapshots(self) -> list[Path]:
        """Save the latest frame of every running camera as
        ``wii/board/camNN/snapshot_<YYYYmmdd_HHMMSS>.jpg`` (reference frames for clicking the board
        on the "2b. Wii Board" page). Returns the written files."""
        paths = WiiPaths(self.project)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = []
        for c, s in sorted(self.streams.items()):
            f = s.latest() if s.running else None
            if f is None:
                continue
            try:
                d = paths.board_dir / c
                d.mkdir(parents=True, exist_ok=True)
                p, k = d / f"snapshot_{stamp}.jpg", 2
                while p.exists():
                    p, k = d / f"snapshot_{stamp}_{k}.jpg", k + 1
                ok, buf = cv2.imencode(".jpg", f.image, [cv2.IMWRITE_JPEG_QUALITY, 95])
                if not ok:
                    raise OSError("JPEG encoding failed")
                p.write_bytes(buf.tobytes())  # works with non-ASCII paths (cv2.imwrite does not)
                out.append(p)
            except OSError:
                log.exception("cannot save the board snapshot of %s", c)
        return out

    # --------------------------------------------------------------- recording
    @property
    def recorder(self) -> WiiRecorder:
        """The ``WiiRecorder`` of this session (``counts``, ``meta``, ``error``...)."""
        return self._recorder

    @property
    def recording(self) -> bool:
        return self._recorder.recording

    @property
    def folder(self) -> Path | None:
        """Folder of the running (or last) take."""
        return self._recorder.folder or self._recorder.last_folder

    @property
    def elapsed_s(self) -> float:
        """Seconds since the running take started (0 when not recording)."""
        t0 = self._recorder.t0
        if not self.recording or t0 is None:
            return 0.0
        return time.perf_counter() - t0

    def start_recording(self, force=None, board=None, geometry=None, subject: str = "",
                        notes: str = "", record_cameras: bool = True) -> Path:
        """Start a take with every running camera and ``force`` (a ForceSource or None) via
        ``WiiRecorder.start(root=WiiPaths(project).recordings_dir, ...)``; ``extra_meta`` records
        the project name/root, camera settings and the Calib.toml image sizes. Returns the
        recording folder. RuntimeError when already recording.

        ``record_cameras=False`` records a Wii-only take (cameras keep previewing; needs
        ``force``). With cameras, every project camera must be ready (``video_take_problems``),
        otherwise RuntimeError: a take without all cameras could not replace the trial
        videos."""
        if self.recording:
            raise RuntimeError("Already recording")
        cams: list[CameraStream] = []
        if record_cameras:
            problems = self.video_take_problems()
            if problems:
                raise RuntimeError("Cannot record the cameras: " + "; ".join(problems) + ".")
            cams = [self.streams[c] for c in self.project_cameras]
        elif force is None:
            raise RuntimeError("Nothing to record: no camera and no Wii Balance Board.")
        sizes = {}
        for s in cams:
            f = s.latest()
            if f is not None:
                sizes[s.name] = [int(f.image.shape[1]), int(f.image.shape[0])]
        root = getattr(self.project, "root", self.project)
        extra = {
            "project": {"name": getattr(getattr(self.project, "config", None), "name", ""),
                        "root": str(Path(root).resolve())},
            "capture": {"settings": self.settings.to_dict(),
                        "calib_sizes": self.calib_sizes(),
                        "frame_sizes": sizes,
                        "record_cameras": bool(record_cameras)},
        }
        paths = WiiPaths(self.project)
        # A full garbage collection pauses every thread (the Wii reader and the camera threads,
        # whose samples / frames are time-stamped when read) for about 0.1 s in a process as
        # large as the GUI. Collect now, before t0, so that one is unlikely during the take.
        gc.collect()
        folder = self._recorder.start(root=paths.recordings_dir, force=force, cams=cams,
                                      board=board, geometry=geometry, subject=subject,
                                      notes=notes, extra_meta=extra)
        for c in sorted(self._lost):  # already without frames at the start
            self._event_quiet(f"camera_lost {c}")
        return folder

    def add_event(self, label: str) -> dict | None:
        """Mark an event now (thread-safe). None when not recording; OSError when events.csv
        cannot be written."""
        return self._recorder.add_event(label)

    def attach_force(self, force) -> None:
        """Hand a new ForceSource (board (re)connected) to the running take."""
        self._recorder.attach_force(force)

    def log_force_source(self, force, reason: str) -> None:
        """Note the force source's state (device, tare) in session.json (e.g. "connected")."""
        self._recorder.log_force_source(force, reason)

    def stop_recording(self) -> Path | None:
        """Stop the take (cameras keep previewing). Returns its folder, None if not recording."""
        return self._recorder.stop()
