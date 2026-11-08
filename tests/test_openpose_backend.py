"""Tests for the OpenPose 2D backend (poseassess.plugins.openpose_backend).

OpenPose itself is never run: ``subprocess.Popen`` inside the backend is replaced by
one that starts a fake OpenPoseDemo (a small Python script) with the same arguments.
The fake writes BODY_25B keypoint JSON with OpenPose's real file naming
(``<video stem>_<12-digit frame>_keypoints.json``) and prints like OpenPose; its
behaviour is set with ``FAKE_OP_*`` environment variables. Install trees are fakes
under tmp_path.
"""
from __future__ import annotations

import ctypes
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from poseassess.core import openpose_setup as ops
from poseassess.plugins import openpose_backend as ob

FAKE_DEMO = r'''
import json, os, sys, time

args = sys.argv[1:]


def opt(name):
    return args[args.index(name) + 1]


env = os.environ.get
print("Starting OpenPose demo...", flush=True)
print("cwd=" + os.getcwd(), flush=True)
for i in range(int(env("FAKE_OP_NOISE", "0"))):
    print(f"noise {i}", flush=True)
out, stem = opt("--write_json"), os.path.splitext(os.path.basename(opt("--video")))[0]
for i in range(int(env("FAKE_OP_FRAMES", "3"))):
    kp = [v for k in range(25) for v in (float(i), float(k), 0.9)]
    doc = {"version": 1.3, "run": env("FAKE_OP_RUN", ""),
           "people": [{"person_id": [-1], "pose_keypoints_2d": kp}]}
    with open(os.path.join(out, f"{stem}_{i:012d}_keypoints.json"), "w") as f:
        json.dump(doc, f)
    time.sleep(float(env("FAKE_OP_DELAY", "0")))
code = int(env("FAKE_OP_EXIT", "0"))
if code:  # after the frames: a failure / crash part way leaves partial output
    print("Error:", file=sys.stderr, flush=True)
    print(env("FAKE_OP_ERROR", "fake failure"), file=sys.stderr, flush=True)
    sys.exit(code)
print("OpenPose demo successfully finished. Total time: 0.1 seconds.", flush=True)
'''


# --------------------------------------------------------------------------- #
# fixtures / helpers
# --------------------------------------------------------------------------- #
@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    """Per-test settings folder, no search of the usual install places, fast polling."""
    monkeypatch.setenv("POSEASSESS_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.setattr(ops, "candidate_dirs", lambda: [])
    monkeypatch.setattr(ob, "_POLL_S", 0.02)
    monkeypatch.setattr(ob, "_REPORT_S", 0.05)
    for name in ("FAKE_OP_NOISE", "FAKE_OP_EXIT", "FAKE_OP_ERROR", "FAKE_OP_FRAMES",
                 "FAKE_OP_RUN", "FAKE_OP_DELAY"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def popen_calls(tmp_path, monkeypatch) -> list[dict]:
    """Make the backend start the fake OpenPoseDemo; returns the recorded Popen calls."""
    script_dir = tmp_path / "fake_bin"
    script_dir.mkdir()
    script = script_dir / "fake_openpose_demo.py"
    script.write_text(FAKE_DEMO, encoding="utf-8")
    calls: list[dict] = []
    real_popen = subprocess.Popen

    def popen(cmd, **kwargs):
        calls.append({"cmd": list(cmd), **kwargs})
        proc = real_popen([sys.executable, str(script), *cmd[1:]], **kwargs)
        calls[-1]["proc"] = proc
        return proc

    monkeypatch.setattr(ob.subprocess, "Popen", popen)
    return calls


def make_install(root: Path, *, gpu: bool = True,
                 caffemodel: str | None = ops.CAFFEMODEL_NAME) -> Path:
    """Fake OpenPose install tree (GPU build with BODY_25B by default); returns the root."""
    bin_dir = root / "bin"
    bin_dir.mkdir(parents=True)
    (bin_dir / ops.EXE_NAME).write_bytes(b"MZ fake")
    if gpu:
        (bin_dir / "cudart64_110.dll").write_bytes(b"MZ cudart")
    mdir = root / ops.MODEL_SUBDIR
    mdir.mkdir(parents=True)
    (mdir / ops.PROTOTXT_NAME).write_text('name: "fake"\n', encoding="utf-8")
    if caffemodel:
        (mdir / caffemodel).write_bytes(b"weights" * 100)
    return root


def make_video(path: Path, frames: int) -> Path:
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    path.parent.mkdir(parents=True, exist_ok=True)
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30, (64, 48))
    for i in range(frames):
        w.write(np.full((48, 64, 3), (i * 20) % 256, np.uint8))
    w.release()
    return path


def backend(root: Path | None, **options) -> ob.OpenPoseBackend:
    return ob.OpenPoseBackend(openpose_dir=str(root) if root else "", **options)


def run(root: Path, video: Path, out: Path, progress=None):
    return backend(root).process_video(video, out, camera_index=1, progress=progress)


def json_names(folder: Path) -> list[str]:
    return sorted(p.name for p in folder.glob("*.json"))


def all_names(folder: Path) -> list[str]:
    """Everything in `folder`, recursively (work folder included)."""
    return sorted(str(p.relative_to(folder)) for p in folder.rglob("*"))


def write_frames(folder: Path, n: int, run: str = "old") -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        (folder / f"frame_{i:05d}.json").write_text(json.dumps({"run": run}))


def arg(cmd: list[str], name: str) -> str:
    return cmd[cmd.index(name) + 1]


# --------------------------------------------------------------------------- #
# process_video
# --------------------------------------------------------------------------- #
def test_happy_path_renames_frames_reports_progress_and_runs_in_root(
        tmp_path, popen_calls, monkeypatch):
    root = make_install(tmp_path / "openpose")
    video = make_video(tmp_path / "proj" / "videos" / "cam01.avi", 12)
    out = tmp_path / "proj" / "pose" / "cam01_json"
    monkeypatch.setenv("FAKE_OP_FRAMES", "12")
    monkeypatch.setenv("FAKE_OP_DELAY", "0.03")
    seen: list[tuple] = []

    res = run(root, video, out, progress=lambda f, t, m: seen.append((f, t, m)))

    assert res.frames_written == 12 and res.camera_index == 1 and res.output_dir == out
    assert json_names(out) == [f"frame_{i:05d}.json" for i in range(12)]
    for i in (0, 5, 11):  # frame order kept
        doc = json.loads((out / f"frame_{i:05d}.json").read_text())
        kp = doc["people"][0]["pose_keypoints_2d"]
        assert len(kp) == 25 * 3 and kp[0] == float(i)

    (call,) = popen_calls
    cmd = call["cmd"]
    assert cmd[0] == str(root / "bin" / ops.EXE_NAME)
    assert arg(cmd, "--model_pose") == "BODY_25B"
    assert arg(cmd, "--net_resolution") == "-1x368"  # OpenPose's own default
    assert arg(cmd, "--display") == "0" and arg(cmd, "--render_pose") == "0"
    assert arg(cmd, "--video") == str(video)
    assert arg(cmd, "--write_json") == str(out / ob._WORK_DIR)  # moved to out on success
    assert call["cwd"] == str(root)
    assert call["stdout"] == subprocess.PIPE and call["stderr"] == subprocess.STDOUT
    assert "text" not in call and "encoding" not in call  # bytes, decoded line by line

    messages = [m for _, _, m in seen]
    assert "cam01 OpenPose: Starting OpenPose demo..." in messages
    (cwd,) = [m.split("cwd=", 1)[1] for m in messages if m.startswith("cam01 OpenPose: cwd=")]
    assert os.path.realpath(cwd) == os.path.realpath(root)  # the fake ran in the install root
    assert any(m.startswith("cam01 OpenPose: OpenPose demo successfully finished")
               for m in messages)
    frame_reports = [(f, t) for f, t, m in seen if "/12 frames" in m]
    assert frame_reports and all(t == 12 for _, t in frame_reports)
    assert seen[-1] == (12, 12, "cam01: 12 frames")
    assert all_names(out) == [f"frame_{i:05d}.json" for i in range(12)]  # no work folder


def test_second_run_replaces_earlier_results(tmp_path, popen_calls, monkeypatch):
    root = make_install(tmp_path / "openpose")
    video = make_video(tmp_path / "videos" / "cam01.avi", 3)
    out = tmp_path / "pose" / "cam01_json"

    monkeypatch.setenv("FAKE_OP_FRAMES", "5")
    monkeypatch.setenv("FAKE_OP_RUN", "first")
    assert run(root, video, out).frames_written == 5
    # stale files of an interrupted run must not be mixed in either: raw OpenPose
    # output, and the work folder of a cancelled run (OpenPose killed with the pipeline)
    (out / "cam01_000000000099_keypoints.json").write_text("{}")
    (out / ob._WORK_DIR).mkdir()
    (out / ob._WORK_DIR / "cam01_000000000007_keypoints.json").write_text("{}")

    monkeypatch.setenv("FAKE_OP_FRAMES", "3")
    monkeypatch.setenv("FAKE_OP_RUN", "second")
    assert run(root, video, out).frames_written == 3

    assert all_names(out) == ["frame_00000.json", "frame_00001.json", "frame_00002.json"]
    assert {json.loads(p.read_text())["run"] for p in out.glob("*.json")} == {"second"}


def test_move_frames_replaces_existing_targets(tmp_path, monkeypatch):
    """Path.replace, not Path.rename: on Windows rename onto an existing file fails."""
    work, out = tmp_path / "out" / ob._WORK_DIR, tmp_path / "out"
    work.mkdir(parents=True)
    write_frames(out, 1, run="old")
    for i in (10, 2, 1):  # frame numbers, not names, give the order
        (work / f"v_{i:012d}_keypoints.json").write_text(json.dumps({"run": f"new{i}"}))

    def windows_rename(self, target):
        if Path(target).exists():
            raise FileExistsError(f"[WinError 183] {target}")
        return original_rename(self, target)

    original_rename = Path.rename
    monkeypatch.setattr(Path, "rename", windows_rename)

    assert ob._move_frames(work, out) == 3
    assert [json.loads((out / f"frame_{i:05d}.json").read_text())["run"]
            for i in range(3)] == ["new1", "new2", "new10"]
    assert list(work.iterdir()) == []


def test_failure_raises_with_openposes_last_output_lines(tmp_path, popen_calls, monkeypatch):
    root = make_install(tmp_path / "openpose")
    video = make_video(tmp_path / "videos" / "cam01.avi", 3)
    monkeypatch.setenv("FAKE_OP_EXIT", "3")
    monkeypatch.setenv("FAKE_OP_NOISE", "30")
    monkeypatch.setenv("FAKE_OP_ERROR", "Prototxt file not found: models/pose/body_25b/"
                                        "pose_deploy.prototxt.")
    with pytest.raises(RuntimeError) as ei:
        run(root, video, tmp_path / "out")
    msg = str(ei.value)
    assert msg.startswith("OpenPose failed (exit 3): ")
    assert "Prototxt file not found: models/pose/body_25b/pose_deploy.prototxt." in msg
    assert "noise 29" in msg and "noise 5\n" not in msg  # only the last lines
    assert len(msg.splitlines()) == ob.TAIL_LINES


def test_failed_run_leaves_the_camera_folder_empty(tmp_path, popen_calls, monkeypatch):
    """Neither the earlier results nor OpenPose's partial output stay behind."""
    root = make_install(tmp_path / "openpose")
    video = make_video(tmp_path / "videos" / "cam01.avi", 3)
    out = tmp_path / "pose" / "cam01_json"
    monkeypatch.setenv("FAKE_OP_FRAMES", "4")
    run(root, video, out)

    monkeypatch.setenv("FAKE_OP_FRAMES", "2")  # then out of GPU memory, say
    monkeypatch.setenv("FAKE_OP_EXIT", "1")
    with pytest.raises(RuntimeError, match=r"OpenPose failed \(exit 1\)"):
        run(root, video, out)
    assert all_names(out) == []


def test_failure_while_moving_the_frames_leaves_the_camera_folder_empty(
        tmp_path, popen_calls, monkeypatch):
    root = make_install(tmp_path / "openpose")
    video = make_video(tmp_path / "videos" / "cam01.avi", 3)
    out = tmp_path / "out"
    moved: list[str] = []
    original_replace = Path.replace

    def replace(self, target):  # the 3rd file is locked (antivirus, Explorer preview)
        if len(moved) == 2:
            raise PermissionError(13, "The process cannot access the file", str(self))
        moved.append(self.name)
        return original_replace(self, target)

    monkeypatch.setattr(Path, "replace", replace)
    monkeypatch.setenv("FAKE_OP_FRAMES", "5")
    with pytest.raises(RuntimeError, match="cannot move .*cam01_000000000002_keypoints"):
        run(root, video, out)
    assert len(moved) == 2 and all_names(out) == []


def test_interrupted_run_kills_openpose_and_leaves_no_json(tmp_path, popen_calls, monkeypatch):
    root = make_install(tmp_path / "openpose")
    video = make_video(tmp_path / "videos" / "cam01.avi", 3)
    out = tmp_path / "out"
    write_frames(out, 3)
    monkeypatch.setenv("FAKE_OP_FRAMES", "200")
    monkeypatch.setenv("FAKE_OP_DELAY", "0.02")

    class Stop(Exception):
        pass

    def progress(f, t, m):
        if "frames (" in m and f >= 2:  # e.g. Ctrl+C while OpenPose is writing frames
            raise Stop

    with pytest.raises(Stop):
        run(root, video, out, progress=progress)
    assert popen_calls[0]["proc"].poll() is not None
    assert all_names(out) == []


def test_exit_0_without_json_is_a_clear_error(tmp_path, popen_calls, monkeypatch):
    root = make_install(tmp_path / "openpose")
    video = make_video(tmp_path / "videos" / "cam01.avi", 3)
    monkeypatch.setenv("FAKE_OP_FRAMES", "0")
    with pytest.raises(RuntimeError, match="wrote no keypoint files for cam01.avi") as ei:
        run(root, video, tmp_path / "out")
    assert "OpenPose demo successfully finished" in str(ei.value)
    assert all_names(tmp_path / "out") == []


@pytest.mark.parametrize("window", [60.0, 0.01])
def test_output_lines_are_rate_limited(tmp_path, popen_calls, monkeypatch, window):
    """A chatty OpenPose must not flood the log: at most _LINES_PER_WINDOW lines per
    window go to progress, the others are counted (and all are in the error tail)."""
    root = make_install(tmp_path / "openpose")
    video = make_video(tmp_path / "videos" / "cam01.avi", 3)
    monkeypatch.setattr(ob, "_LINES_WINDOW_S", window)
    monkeypatch.setenv("FAKE_OP_NOISE", "300")
    seen: list[str] = []
    run(root, video, tmp_path / "out", progress=lambda f, t, m: seen.append(m))

    note = "cam01 OpenPose: ("
    shown = [m for m in seen if m.startswith("cam01 OpenPose: ") and not m.startswith(note)]
    hidden = [int(m[len(note):].split()[0]) for m in seen if m.startswith(note)]
    assert len(shown) + sum(hidden) == 303  # start, cwd, 300 noise, finished
    assert hidden and all(n > 0 for n in hidden)
    if window == 60.0:
        assert len(shown) == ob._LINES_PER_WINDOW
        assert seen[-2] == "cam01 OpenPose: (283 more output lines not shown)"


def test_missing_video_and_missing_install(tmp_path, popen_calls):
    root = make_install(tmp_path / "openpose")
    with pytest.raises(FileNotFoundError, match="video not found"):
        run(root, tmp_path / "nope.avi", tmp_path / "out")
    with pytest.raises(RuntimeError, match="OpenPose not found"):
        run(tmp_path / "nothing", tmp_path / "nope.avi", tmp_path / "out")
    assert popen_calls == []


def test_extra_args_and_options(tmp_path):
    root = make_install(tmp_path / "openpose")
    be = backend(root, net_resolution="-1x368", extra_args=["--number_people_max", "1"])
    cmd = be.command(root / "bin" / ops.EXE_NAME, "v.avi", "out")
    assert arg(cmd, "--net_resolution") == "-1x368" and arg(cmd, "--model_pose") == "BODY_25B"
    assert cmd[-2:] == ["--number_people_max", "1"]


def test_decode_utf8_first_then_the_ansi_code_page(monkeypatch):
    assert ob._decode("Fehler: Datei fehlt".encode()) == "Fehler: Datei fehlt"
    assert ob._decode("C:\\データ".encode()) == "C:\\データ"

    monkeypatch.setattr(ob, "_locale_encoding", lambda: "cp932")  # Japanese Windows
    assert ob._decode("エラー: C:\\データ".encode("cp932")) == "エラー: C:\\データ"
    assert ob._decode("C:\\データ".encode()) == "C:\\データ"  # UTF-8 still wins
    monkeypatch.setattr(ob, "_locale_encoding", lambda: "cp1252")  # Western Windows
    assert ob._decode(b"C:\\caf\xe9") == "C:\\café"
    assert ob._decode(b"caf\xe9\x81").startswith("café")  # undefined byte: no error
    monkeypatch.setattr(ob, "_locale_encoding", lambda: "no-such-codec")
    assert ob._decode(b"caf\xe9") == "café"  # latin-1 as the last resort


def test_exit_text_shows_signed_codes_and_windows_crashes():
    assert ob._exit_text(1) == "exit 1"
    assert ob._exit_text(-9) == "exit -9"  # POSIX: killed by a signal
    # OpenPoseDemo's own error exit (return -1), as Windows reports it
    assert ob._exit_text(4294967295) == "exit -1 = 0xFFFFFFFF"
    assert ob._exit_text(0xC0000135) == "exit 0xC0000135: a DLL was not found"
    assert ob._exit_text(0xC0000374) == "exit -1073740940 = 0xC0000374"


class _FakeKernel32:
    """GetShortPathNameW of a drive with (or without) 8.3 names."""

    def __init__(self, short: str | None):
        self.short, self.calls = short, []

    def GetShortPathNameW(self, long, buf, size):  # noqa: N802 - Windows API name
        self.calls.append(long)
        if self.short is None:
            return 0  # failed (e.g. path gone)
        buf.value = self.short
        return len(self.short)


def test_short_paths_go_to_openpose_and_the_frame_count(tmp_path, popen_calls, monkeypatch):
    """OpenCV, like OpenPose, may not open a non-ASCII path: both get `_arg_path`."""
    root = make_install(tmp_path / "openpose")
    video = make_video(tmp_path / "videos" / "cam01.avi", 3)
    short = "C:\\VIDEOS~1\\cam01.avi"
    monkeypatch.setattr(ob, "_arg_path", lambda p: short if p == video else str(p))
    counted: list[str] = []
    monkeypatch.setattr(ob, "_frame_count", lambda v: counted.append(v) or 3)
    run(root, video, tmp_path / "out")
    assert counted == [short] and arg(popen_calls[0]["cmd"], "--video") == short


def test_arg_path_uses_the_ascii_short_form_on_windows(monkeypatch):
    long = Path("/Users/山田/動画/cam01.mp4")
    monkeypatch.setattr(ob, "_is_windows", lambda: True)

    k32 = _FakeKernel32("C:\\Users\\8B71~1\\8B95~1\\cam01.mp4")
    monkeypatch.setattr(ob, "_kernel32", lambda: k32)
    assert ob._arg_path(long) == "C:\\Users\\8B71~1\\8B95~1\\cam01.mp4"
    assert ob._arg_path(Path("/videos/cam01.mp4")) == str(Path("/videos/cam01.mp4"))
    assert k32.calls == [str(long)]  # ASCII paths are passed as they are

    for short in (str(long), None):  # 8.3 names disabled on the drive / call failed
        monkeypatch.setattr(ob, "_kernel32", lambda k=_FakeKernel32(short): k)
        assert ob._arg_path(long) == str(long)

    monkeypatch.setattr(ob, "_is_windows", lambda: False)
    monkeypatch.setattr(ob, "_kernel32", lambda: pytest.fail("not on Windows"))
    assert ob._arg_path(long) == str(long)


# --------------------------------------------------------------------------- #
# is_available
# --------------------------------------------------------------------------- #
def test_is_available_ok(tmp_path):
    root = make_install(tmp_path / "openpose")
    assert backend(root).is_available() == (True, "")
    assert backend(root / "bin").is_available() == (True, "")


def test_is_available_missing_install(tmp_path):
    ok, why = backend(tmp_path / "nothing").is_available()
    assert not ok
    assert why.startswith("OpenPose not found: set the OpenPose folder on 1. Project (looked in: ")
    assert str(tmp_path / "nothing") in why
    ok, why = backend(None).is_available()
    assert not ok and why.startswith("OpenPose not found: set the OpenPose folder on 1. Project")


def test_is_available_cpu_build(tmp_path):
    ok, why = backend(make_install(tmp_path / "openpose", gpu=False)).is_available()
    assert not ok and "CPU-only build" in why and str(tmp_path / "openpose") in why


def test_is_available_missing_or_misnamed_caffemodel(tmp_path):
    ok, why = backend(make_install(tmp_path / "a", caffemodel=None)).is_available()
    assert not ok and "missing BODY_25B weights" in why and ops.CAFFEMODEL_NAME in why
    ok, why = backend(make_install(tmp_path / "b",
                                   caffemodel="pose_iter_464000.caffemodel")).is_available()
    assert not ok and f"must be named {ops.CAFFEMODEL_NAME}" in why
    assert "pose_iter_464000.caffemodel" in why


def test_is_available_uses_the_remembered_folder(tmp_path):
    root = make_install(tmp_path / "openpose")
    ops.remember_dir(root)
    assert backend(None).is_available() == (True, "")
    assert backend(None).resolve() == (root, "remembered on this computer")


# --------------------------------------------------------------------------- #
# Windows Job Object
# --------------------------------------------------------------------------- #
def test_job_limit_structure_has_the_windows_layout():
    expected = {8: 144, 4: 112}[ctypes.sizeof(ctypes.c_void_p)]
    assert ctypes.sizeof(ob._ExtendedLimitInformation) == expected


def test_job_object_failure_warns_only_while_openpose_runs(tmp_path, popen_calls, monkeypatch):
    """When OpenPose ended right away (e.g. 0xC0000135) AssignProcessToJobObject fails
    too; that must not add a misleading 'Cancel may not stop OpenPose' warning."""
    root = make_install(tmp_path / "openpose")
    video = make_video(tmp_path / "videos" / "cam01.avi", 3)
    monkeypatch.setattr(ob, "_is_windows", lambda: True)
    monkeypatch.setattr(ob, "_popen_kwargs",
                        lambda: {"stdin": subprocess.DEVNULL, "stdout": subprocess.PIPE,
                                 "stderr": subprocess.STDOUT})
    monkeypatch.setattr(ob, "close_handle", lambda h: pytest.fail("no job to close"))
    warning = "cam01: warning: Cancel may not stop OpenPose ([Errno 5] Access is denied)"

    def deny(pid):
        raise OSError(5, "Access is denied")

    monkeypatch.setattr(ob, "kill_with_parent", deny)  # OpenPose still starting up
    monkeypatch.setenv("FAKE_OP_DELAY", "0.1")
    seen: list[str] = []
    run(root, video, tmp_path / "a", progress=lambda f, t, m: seen.append(m))
    assert warning in seen

    if not hasattr(os, "waitid"):
        return

    def deny_after_exit(pid):
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:  # wait until it has ended, without reaping it
            if os.waitid(os.P_PID, pid, os.WEXITED | os.WNOHANG | os.WNOWAIT) is not None:
                break
            time.sleep(0.01)
        raise OSError(5, "Access is denied")

    monkeypatch.setattr(ob, "kill_with_parent", deny_after_exit)
    monkeypatch.setenv("FAKE_OP_DELAY", "0")
    monkeypatch.setenv("FAKE_OP_EXIT", "1")
    seen.clear()
    with pytest.raises(RuntimeError, match=r"OpenPose failed \(exit 1\)"):
        run(root, video, tmp_path / "b", progress=lambda f, t, m: seen.append(m))
    assert not any("Cancel may not stop" in m for m in seen)


@pytest.mark.skipif(os.name != "nt", reason="Windows Job Objects")
def test_job_object_kills_the_process_when_closed():
    job = ob.create_kill_on_close_job()
    ob.close_handle(job)

    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"],
                            creationflags=subprocess.CREATE_NO_WINDOW)
    try:
        job = ob.kill_with_parent(proc.pid)
        time.sleep(0.2)
        assert proc.poll() is None
        ob.close_handle(job)  # what Windows does when the pipeline process is terminated
        proc.wait(timeout=10)
    finally:
        if proc.poll() is None:
            proc.kill()


# --------------------------------------------------------------------------- #
# Pipeline and Config.toml
# --------------------------------------------------------------------------- #
def _openpose_project(project, root: Path | None, frames: int = 4):
    project.config.pose2d_backend = "openpose"
    project.config.openpose_dir = str(root) if root else ""
    project.save()
    make_video(project.videos_dir / "cam01.avi", frames)
    return project


def test_pipeline_runs_the_resolved_install_and_logs_openpose_output(
        tmp_path, project, popen_calls, monkeypatch):
    from poseassess.core.pipeline import Pipeline

    root = make_install(tmp_path / "openpose")
    _openpose_project(project, root / "bin")  # any path inside the install is fine
    monkeypatch.setenv("FAKE_OP_FRAMES", "4")
    logs: list[str] = []
    Pipeline(project, progress=logs.append)._run_plugin_pose("openpose")

    assert f"[pose:openpose] OpenPose at {root} (project setting)" in logs
    assert any("is not one of the two official" in m for m in logs)  # the fake prototxt
    assert "  cam01 OpenPose: Starting OpenPose demo..." in logs  # not thinned out
    assert popen_calls[0]["cwd"] == str(root)
    assert json_names(project.pose2d_cam_dir(1)) == [f"frame_{i:05d}.json" for i in range(4)]


def test_pipeline_logs_a_usual_place_and_an_unusable_project_setting(
        tmp_path, project, popen_calls, monkeypatch):
    from poseassess.core.pipeline import Pipeline

    root = make_install(tmp_path / "openpose")
    monkeypatch.setattr(ops, "candidate_dirs", lambda: [root])  # e.g. C:\\openpose
    _openpose_project(project, None)
    logs: list[str] = []
    Pipeline(project, progress=logs.append)._run_plugin_pose("openpose")
    assert f"[pose:openpose] OpenPose at {root} (found in a usual place)" in logs
    assert not any("is not usable" in m for m in logs)

    broken = make_install(tmp_path / "broken", caffemodel=None)
    project.config.openpose_dir = str(broken)
    logs.clear()
    Pipeline(project, progress=logs.append)._run_plugin_pose("openpose")
    (warning,) = [m for m in logs if "is not usable" in m]
    assert warning.startswith(f"[pose:openpose] warning: the OpenPose folder set on 1. "
                              f"Project ({broken}) is not usable: missing BODY_25B weights")
    assert f"[pose:openpose] OpenPose at {root} (found in a usual place)" in logs
    assert popen_calls[-1]["cwd"] == str(root)


def test_pipeline_failure_leaves_no_older_results_in_any_camera(
        tmp_path, project, popen_calls, monkeypatch):
    """Pose2Sim reads every camNN_json: after cam01 fails, cam02 must not keep the
    results of an earlier run (it was never re-run)."""
    from poseassess.core.pipeline import Pipeline

    _openpose_project(project, make_install(tmp_path / "openpose"))
    make_video(project.videos_dir / "cam02.avi", 4)
    for i in (1, 2):
        write_frames(project.pose2d_cam_dir(i), 4)
    (project.pose2d_cam_dir(2) / "cam02_000000000000.json").write_text("{}")  # rtmlib's
    monkeypatch.setenv("FAKE_OP_EXIT", "1")
    with pytest.raises(RuntimeError, match="OpenPose failed"):
        Pipeline(project, progress=lambda m: None)._run_plugin_pose("openpose")
    assert len(popen_calls) == 1
    assert all_names(project.pose2d_cam_dir(1)) == []
    assert all_names(project.pose2d_cam_dir(2)) == []


def test_pipeline_without_install_says_where_to_set_it(tmp_path, project, popen_calls):
    from poseassess.core.pipeline import Pipeline

    _openpose_project(project, None)
    with pytest.raises(RuntimeError, match="OpenPose not found: set the OpenPose folder on "
                                           "1. Project"):
        Pipeline(project, progress=lambda m: None)._run_plugin_pose("openpose")
    assert popen_calls == []


def test_cli_stage_result_is_one_line(project, monkeypatch, capsys):
    """OpenPose's quoted output must reach the GUI in the single @@STAGE_RESULT line."""
    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt

    from poseassess.core import pipeline, pipeline_cli

    monkeypatch.setattr(plt, "show", plt.show)    # main() replaces them: restore afterwards
    monkeypatch.setattr(plt, "pause", plt.pause)

    def fail(self):
        raise RuntimeError("OpenPose failed (exit 1): Error:\nPrototxt file not found")

    monkeypatch.setattr(pipeline.Pipeline, "pose", fail)
    assert pipeline_cli.main([str(project.root), "pose"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert ("@@STAGE_RESULT pose FAIL RuntimeError: OpenPose failed (exit 1): Error: | "
            "Prototxt file not found") in lines


def test_config_toml_uses_body_25b_for_openpose(project):
    pytest.importorskip("Pose2Sim")
    import tomllib

    from poseassess.core.config_gen import generate_config

    project.config.pose2d_backend = "openpose"
    cfg = generate_config(project)
    with open(cfg, "rb") as f:
        data = tomllib.load(f)
    assert data["pose"]["pose_model"] == "BODY_25B"
