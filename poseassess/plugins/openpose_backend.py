"""OpenPose 2D backend (external executable, BODY_25B).

OpenPose is CMU software whose licence forbids redistribution, so PoseAssess
never bundles it: this backend runs the user's OWN install, found by
`openpose_setup.resolve_openpose_dir` from the ``openpose_dir`` option (the
project setting), the folder remembered on this computer, or a usual place such
as ``C:\\openpose`` (the folder with ``bin/OpenPoseDemo.exe`` and ``models/``).

``OpenPoseDemo.exe`` runs with the install root as working directory (it finds
``models/`` there) and writes one JSON per frame, ``<video stem>_<12-digit
frame>_keypoints.json``, which is already the format the rest of the pipeline
consumes. It writes them into a work folder inside the camera folder
(``camNN_json/.openpose_partial``, which Pose2Sim does not read); only after a
successful run are they moved up as ``frame_00000.json``, ... in frame order. So a
camera whose run failed or was cancelled has no JSON at all instead of a partial
or stale mix. OpenPose's output lines (at most `_LINES_PER_WINDOW` a second) and
the number of frames done go to `progress` while it runs. On Windows the process
is put in a Job Object that kills it when the pipeline process ends, so Cancel
(which terminates that process) also stops OpenPose instead of leaving it running
on the GPU.
"""
from __future__ import annotations

import ctypes
import locale
import os
import queue
import re
import shutil
import subprocess
import threading
import time
from collections import deque
from functools import lru_cache
from pathlib import Path
from typing import IO, Optional

from ..core import openpose_setup as ops
from .base import Pose2DBackend, Pose2DResult, ProgressFn, register_pose2d

MODEL_POSE = ops.MODEL_POSE              # "BODY_25B"
NET_RESOLUTION = ops.NET_RESOLUTION      # "-1x480", CMU's setting for BODY_25B
TAIL_LINES = 20        # output lines quoted when OpenPose fails
_POLL_S = 0.25         # how often the loop wakes up when OpenPose prints nothing
_REPORT_S = 1.0        # frame progress at most this often
_LINES_WINDOW_S = 1.0  # OpenPose output lines passed to progress: at most
_LINES_PER_WINDOW = 20  # this many per window, the rest are only counted
_WORK_DIR = ".openpose_partial"  # inside the camera folder; no *.json at its level
_KEYPOINTS_GLOB = "*_keypoints.json"
_KEYPOINTS_RE = re.compile(r"_(\d+)_keypoints\.json$")


@register_pose2d("openpose")
class OpenPoseBackend(Pose2DBackend):
    display_name = "OpenPose (GPU)"
    skeleton = "BODY_25B"
    requires_gpu = True
    progress_throttled = True

    def resolve(self) -> tuple[Optional[Path], str]:
        """(install root, where it came from); see `openpose_setup.resolve_openpose_dir`.

        Looks at several folders (C:\\openpose, D:\\openpose, which may be a slow or
        removable drive, ...) and reads the prototxt, so the GUI must call it (and
        `is_available`) in a worker thread.
        """
        return ops.resolve_openpose_dir(self.options.get("openpose_dir"))

    def _model_pose(self) -> str:
        return str(self.options.get("model_pose") or MODEL_POSE)

    def _net_resolution(self) -> str:
        return str(self.options.get("net_resolution") or NET_RESOLUTION)

    def is_available(self) -> tuple[bool, str]:
        """(usable, why not); slow like `resolve`, so not on the GUI thread."""
        root, how = self.resolve()
        if root is None:
            return False, _not_found_message(how)
        if self._model_pose() != MODEL_POSE:
            return True, ""  # check_install only knows the BODY_25B files
        st = ops.check_install(root)
        if not st.ok:
            return False, f"OpenPose at {root} cannot run {MODEL_POSE}: {'; '.join(st.errors)}"
        return True, ""

    def command(self, exe: Path, video: str, output_dir: str) -> list[str]:
        """OpenPoseDemo command line for one video."""
        extra = self.options.get("extra_args") or ()
        if isinstance(extra, str):
            extra = extra.split()
        return [
            str(exe),
            "--video", video,
            "--write_json", output_dir,
            "--model_pose", self._model_pose(),
            "--net_resolution", self._net_resolution(),
            "--display", "0",
            "--render_pose", "0",
            *(str(a) for a in extra),
        ]

    def process_video(
        self,
        video_path: Path,
        output_dir: Path,
        camera_index: int,
        progress: Optional[ProgressFn] = None,
    ) -> Pose2DResult:
        report: ProgressFn = progress or (lambda f, t, m: None)
        tag = f"cam{camera_index:02d}"
        root, how = self.resolve()
        exe = ops.demo_exe(root) if root is not None else None
        if root is None or exe is None:
            raise RuntimeError(_not_found_message(how))

        # absolute: OpenPose runs in its own folder
        video = Path(video_path).absolute()
        if not video.is_file():
            raise FileNotFoundError(f"video not found: {video}")
        out = Path(output_dir).absolute()
        out.mkdir(parents=True, exist_ok=True)
        _remove_old_results(out)
        work = out / _WORK_DIR
        work.mkdir()
        try:
            written = self._run(exe, root, video, out, work, report, tag)
        except BaseException:
            # a failed camera is left empty, never half-filled (Pose2Sim reads every *.json)
            _remove_old_results(out, strict=False)
            raise
        finally:
            shutil.rmtree(work, ignore_errors=True)
        return Pose2DResult(camera_index=camera_index, frames_written=written, output_dir=out)

    def _run(self, exe: Path, root: Path, video: Path, out: Path, work: Path,
             report: ProgressFn, tag: str) -> int:
        """Run OpenPose on `video` into `work`, then move its JSON to `out`; frames written."""
        video_arg, work_arg = _arg_path(video), _arg_path(work)
        for p in (video_arg, work_arg):
            if not p.isascii():
                report(0, None, f"{tag}: warning: OpenPose may fail to open paths with "
                                f"non-ASCII characters ({p})")
        total = _frame_count(video_arg)  # the short path also helps OpenCV
        report(0, total, f"{tag}: OpenPose {self._model_pose()} {self._net_resolution()} "
                         f"on {video.name}" + (f" ({total} frames)" if total else ""))
        code, tail = _run_openpose(self.command(exe, video_arg, work_arg), root, work,
                                   total, report, tag)
        output = "\n".join(tail) or "(no output)"
        if code != 0:
            raise RuntimeError(f"OpenPose failed ({_exit_text(code)}): {output}")

        written = _move_frames(work, out)
        if written == 0:
            raise RuntimeError(f"OpenPose finished but wrote no keypoint files for {video.name} "
                               f"(could it read the video?): {output}")
        report(written, total or written, f"{tag}: {written} frames")
        return written


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _not_found_message(how: str) -> str:
    """User-facing message when no install was found; `how` is the reason from
    `resolve_openpose_dir` ("OpenPoseDemo.exe not found (looked in: ...)")."""
    prefix = f"{ops.EXE_NAME} not found "
    detail = how[len(prefix):] if how.startswith(prefix) else f"({how})"
    return f"OpenPose not found: set the OpenPose folder on 1. Project {detail}"


def _remove_old_results(out: Path, *, strict: bool = True) -> None:
    """Delete the JSON of an earlier or failed run and OpenPose's work folder (left by
    a cancelled run), so nothing old is mixed in or in the way. `strict`: raise
    RuntimeError when something cannot be removed; else leave it (cleanup after an
    error, which must not hide that error)."""
    for pattern in (_KEYPOINTS_GLOB, "frame_*.json"):
        for f in out.glob(pattern):
            try:
                f.unlink()
            except FileNotFoundError:
                pass
            except OSError as e:
                if strict:
                    raise RuntimeError(f"cannot remove the earlier result {f}: {e}") from e
    work = out / _WORK_DIR
    try:
        shutil.rmtree(work)
    except FileNotFoundError:
        pass
    except OSError as e:
        if strict:
            raise RuntimeError(f"cannot remove the earlier result {work}: {e}") from e


def _frame_number(path: Path) -> Optional[int]:
    m = _KEYPOINTS_RE.search(path.name)
    return int(m.group(1)) if m else None


def _move_frames(src: Path, out: Path) -> int:
    """Move OpenPose's ``*_keypoints.json`` from `src` to `out` as ``frame_00000.json``,
    ... in frame order; returns how many."""
    def key(p: Path) -> tuple[bool, int, str]:
        n = _frame_number(p)
        return n is None, n or 0, p.name

    files = sorted(src.glob(_KEYPOINTS_GLOB), key=key)
    for i, f in enumerate(files):
        dest = out / f"frame_{i:05d}.json"
        try:
            f.replace(dest)  # replace, not rename: no FileExistsError on Windows
        except OSError as e:
            raise RuntimeError(f"cannot move {f} to {dest}: {e}") from e
    return len(files)


def _count_keypoint_files(out: Path) -> int:
    try:
        with os.scandir(out) as it:
            return sum(1 for e in it if e.name.endswith("_keypoints.json"))
    except OSError:
        return 0


def _frame_count(video: str | Path) -> Optional[int]:
    """Number of frames of `video` according to OpenCV, None when unknown."""
    try:
        import cv2
    except Exception:  # noqa: BLE001 - only used for progress
        return None
    cap = cv2.VideoCapture(str(video))
    try:
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    finally:
        cap.release()
    return n if n > 0 else None


def _frames_text(tag: str, done: int, total: Optional[int]) -> str:
    if total:
        return f"{tag}: {done}/{total} frames ({min(100, 100 * done // total)}%)"
    return f"{tag}: {done} frames"


# Windows status codes a crashed OpenPoseDemo.exe typically exits with
_NTSTATUS_TEXT = {
    0xC0000005: "access violation (crash)",
    0xC000001D: "illegal instruction (crash)",
    0xC000007B: "a DLL is not a 64-bit Windows DLL",
    0xC0000135: "a DLL was not found",
    0xC0000139: "a DLL lacks a needed function (wrong version)",
    0xC0000142: "a DLL failed to initialize",
    0xC0000409: "stack buffer overrun / abort (crash)",
}


def _exit_text(code: int) -> str:
    """'exit N' for OpenPose's exit code.

    Windows reports exit codes as unsigned 32-bit numbers: OpenPose's own error
    exit (``return -1`` after it caught an exception: missing model file, unreadable
    video, out of GPU memory) arrives as 4294967295 and is shown as 'exit -1 =
    0xFFFFFFFF'; a known crash code as e.g. 'exit 0xC0000135: a DLL was not found'.
    """
    if code < 0x80000000:
        return f"exit {code}"
    known = _NTSTATUS_TEXT.get(code)
    if known:
        return f"exit 0x{code:08X}: {known}"
    return f"exit {code - 0x100000000} = 0x{code:08X}"


@lru_cache(maxsize=1)
def _locale_encoding() -> str:
    """The ANSI code page OpenPose writes non-ASCII text in (also under PYTHONUTF8=1)."""
    get = getattr(locale, "getencoding", None)  # 3.11+: ignores Python's UTF-8 mode
    return get() if get is not None else locale.getpreferredencoding(False)


def _decode(raw: bytes) -> str:
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        pass
    try:
        return raw.decode(_locale_encoding(), errors="replace")
    except LookupError:
        return raw.decode("latin-1")


def _pump_lines(stream: IO[bytes], lines: queue.Queue) -> None:
    """Reader thread: put each non-empty output line into `lines`, then None at the end."""
    try:
        for raw in iter(stream.readline, b""):
            for line in _decode(raw).splitlines():
                if line.strip():
                    lines.put(line.rstrip())
    except (OSError, ValueError):  # pipe closed under us
        pass
    finally:
        lines.put(None)


def _is_windows() -> bool:
    return os.name == "nt"


def _popen_kwargs() -> dict:
    kwargs: dict = {"stdin": subprocess.DEVNULL, "stdout": subprocess.PIPE,
                    "stderr": subprocess.STDOUT}
    if _is_windows():
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    return kwargs


def _hidden_text(tag: str, hidden: int) -> str:
    return f"{tag} OpenPose: ({hidden} more output line{'s' * (hidden != 1)} not shown)"


def _run_openpose(cmd: list[str], cwd: Path, json_dir: Path, total: Optional[int],
                  report: ProgressFn, tag: str) -> tuple[int, list[str]]:
    """Run OpenPose to the end, forwarding its output (at most `_LINES_PER_WINDOW`
    lines per `_LINES_WINDOW_S`) and the number of JSON files in `json_dir` to
    `report`; (exit code, last output lines, all of them counted)."""
    try:
        proc = subprocess.Popen(cmd, cwd=str(cwd), **_popen_kwargs())
    except OSError as e:
        raise RuntimeError(f"cannot start {cmd[0]}: {e}") from e
    job = None
    tail: deque[str] = deque(maxlen=TAIL_LINES)
    try:
        if _is_windows():
            try:
                job = kill_with_parent(proc.pid)
            except OSError as e:
                # expected when OpenPose has already ended (e.g. a DLL missing): the
                # real error follows from its exit code
                if proc.poll() is None:
                    report(0, total, f"{tag}: warning: Cancel may not stop OpenPose ({e})")
        lines: queue.Queue = queue.Queue()
        reader = threading.Thread(target=_pump_lines, args=(proc.stdout, lines),
                                  name="openpose-output", daemon=True)
        reader.start()
        done = 0
        last_report = window = time.monotonic()
        shown = hidden = 0
        while True:
            try:
                line = lines.get(timeout=_POLL_S)
            except queue.Empty:
                line = ""
            if line is None:  # OpenPose closed its output: it has ended
                break
            now = time.monotonic()
            if now - window >= _LINES_WINDOW_S:
                if hidden:
                    report(done, total, _hidden_text(tag, hidden))
                window, shown, hidden = now, 0, 0
            if line:
                tail.append(line)
                if shown < _LINES_PER_WINDOW:
                    shown += 1
                    report(done, total, f"{tag} OpenPose: {line}")
                else:
                    hidden += 1
            if now - last_report >= _REPORT_S:
                last_report = now
                n = _count_keypoint_files(json_dir)
                if n > done:
                    done = n
                    report(done, total, _frames_text(tag, done, total))
        if hidden:
            report(done, total, _hidden_text(tag, hidden))
        code = proc.wait()
        reader.join(timeout=5)
    finally:
        if proc.poll() is None:  # an error / interrupt here: do not leave it running
            proc.kill()
            proc.wait()
        if job is not None:
            close_handle(job)
        if proc.stdout is not None:
            proc.stdout.close()
    return code, list(tail)


# --------------------------------------------------------------------------- #
# Windows: Job Object (OpenPose dies with the pipeline process) and short paths
# --------------------------------------------------------------------------- #
JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
_JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9   # JOBOBJECTINFOCLASS
_PROCESS_TERMINATE = 0x0001
_PROCESS_SET_QUOTA = 0x0100


class _BasicLimitInformation(ctypes.Structure):
    """JOBOBJECT_BASIC_LIMIT_INFORMATION."""
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_int64),
        ("PerJobUserTimeLimit", ctypes.c_int64),
        ("LimitFlags", ctypes.c_uint32),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", ctypes.c_uint32),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", ctypes.c_uint32),
        ("SchedulingClass", ctypes.c_uint32),
    ]


class _IoCounters(ctypes.Structure):
    """IO_COUNTERS."""
    _fields_ = [(name, ctypes.c_uint64) for name in (
        "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
        "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]


class _ExtendedLimitInformation(ctypes.Structure):
    """JOBOBJECT_EXTENDED_LIMIT_INFORMATION."""
    _fields_ = [
        ("BasicLimitInformation", _BasicLimitInformation),
        ("IoInfo", _IoCounters),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


@lru_cache(maxsize=1)
def _kernel32():
    """kernel32 with the prototypes used here (a private instance: the prototypes of
    other code's ``ctypes.windll.kernel32`` stay untouched)."""
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]
    handle, dword, bool_ = ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int
    k32.CreateJobObjectW.argtypes = (ctypes.c_void_p, ctypes.c_wchar_p)
    k32.CreateJobObjectW.restype = handle
    k32.SetInformationJobObject.argtypes = (handle, ctypes.c_int, ctypes.c_void_p, dword)
    k32.SetInformationJobObject.restype = bool_
    k32.OpenProcess.argtypes = (dword, bool_, dword)
    k32.OpenProcess.restype = handle
    k32.AssignProcessToJobObject.argtypes = (handle, handle)
    k32.AssignProcessToJobObject.restype = bool_
    k32.CloseHandle.argtypes = (handle,)
    k32.CloseHandle.restype = bool_
    k32.GetShortPathNameW.argtypes = (ctypes.c_wchar_p, ctypes.c_wchar_p, dword)
    k32.GetShortPathNameW.restype = dword
    return k32


def _win_error() -> OSError:
    return ctypes.WinError(ctypes.get_last_error())  # type: ignore[attr-defined]


def create_kill_on_close_job() -> int:
    """Windows: a new Job Object whose processes are killed when its last handle closes."""
    k32 = _kernel32()
    job = k32.CreateJobObjectW(None, None)
    if not job:
        raise _win_error()
    info = _ExtendedLimitInformation()
    info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    if not k32.SetInformationJobObject(job, _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
                                       ctypes.byref(info), ctypes.sizeof(info)):
        err = _win_error()
        k32.CloseHandle(job)
        raise err
    return job


def assign_to_job(job: int, pid: int) -> None:
    """Windows: put process `pid` into `job`."""
    k32 = _kernel32()
    proc = k32.OpenProcess(_PROCESS_SET_QUOTA | _PROCESS_TERMINATE, False, pid)
    if not proc:
        raise _win_error()
    try:
        if not k32.AssignProcessToJobObject(job, proc):
            raise _win_error()
    finally:
        k32.CloseHandle(proc)


def close_handle(handle: int) -> None:
    """Windows: close a handle (for a kill-on-close job: kills what still runs in it)."""
    _kernel32().CloseHandle(handle)


def kill_with_parent(pid: int) -> int:
    """Windows: tie process `pid` to this process; returns the job handle.

    Keep the handle open while the process runs and pass it to `close_handle`
    afterwards. When this process ends first (Cancel terminates it), Windows
    closes the handle and kills the process. OSError when that is not possible.
    """
    job = create_kill_on_close_job()
    try:
        assign_to_job(job, pid)
    except OSError:
        close_handle(job)
        raise
    return job


def _arg_path(path: Path) -> str:
    """`path` for OpenPose's command line. OpenPose 1.7 uses narrow (ANSI) strings, so
    on Windows a path with non-ASCII characters is replaced by its 8.3 short form
    when that is ASCII (short names may be disabled on a drive: then unchanged)."""
    s = str(path)
    if s.isascii() or not _is_windows():
        return s
    try:
        buf = ctypes.create_unicode_buffer(32768)
        n = _kernel32().GetShortPathNameW(s, buf, len(buf))
    except (OSError, AttributeError):
        return s
    short = buf.value if 0 < n < len(buf) else ""
    return short if short and short.isascii() else s
