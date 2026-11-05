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
code = int(env("FAKE_OP_EXIT", "0"))
if code:
    print("Error:", file=sys.stderr, flush=True)
    print(env("FAKE_OP_ERROR", "fake failure"), file=sys.stderr, flush=True)
    sys.exit(code)
out, stem = opt("--write_json"), os.path.splitext(os.path.basename(opt("--video")))[0]
for i in range(int(env("FAKE_OP_FRAMES", "3"))):
    kp = [v for k in range(25) for v in (float(i), float(k), 0.9)]
    doc = {"version": 1.3, "run": env("FAKE_OP_RUN", ""),
           "people": [{"person_id": [-1], "pose_keypoints_2d": kp}]}
    with open(os.path.join(out, f"{stem}_{i:012d}_keypoints.json"), "w") as f:
        json.dump(doc, f)
    time.sleep(float(env("FAKE_OP_DELAY", "0")))
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
        return real_popen([sys.executable, str(script), *cmd[1:]], **kwargs)

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
    assert arg(cmd, "--net_resolution") == "-1x480"
    assert arg(cmd, "--display") == "0" and arg(cmd, "--render_pose") == "0"
    assert arg(cmd, "--video") == str(video) and arg(cmd, "--write_json") == str(out)
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


def test_second_run_replaces_earlier_results(tmp_path, popen_calls, monkeypatch):
    root = make_install(tmp_path / "openpose")
    video = make_video(tmp_path / "videos" / "cam01.avi", 3)
    out = tmp_path / "pose" / "cam01_json"

    def no_rename(self, target):  # Windows: rename onto an existing file fails
        if Path(target).exists():
            raise FileExistsError(f"[WinError 183] {target}")
        return original_rename(self, target)

    original_rename = Path.rename
    monkeypatch.setattr(Path, "rename", no_rename)

    monkeypatch.setenv("FAKE_OP_FRAMES", "5")
    monkeypatch.setenv("FAKE_OP_RUN", "first")
    assert run(root, video, out).frames_written == 5
    # a stale file of an interrupted run must not be mixed in either
    (out / "cam01_000000000099_keypoints.json").write_text("{}")

    monkeypatch.setenv("FAKE_OP_FRAMES", "3")
    monkeypatch.setenv("FAKE_OP_RUN", "second")
    assert run(root, video, out).frames_written == 3

    assert json_names(out) == ["frame_00000.json", "frame_00001.json", "frame_00002.json"]
    assert {json.loads(p.read_text())["run"] for p in out.glob("*.json")} == {"second"}


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


def test_exit_0_without_json_is_a_clear_error(tmp_path, popen_calls, monkeypatch):
    root = make_install(tmp_path / "openpose")
    video = make_video(tmp_path / "videos" / "cam01.avi", 3)
    monkeypatch.setenv("FAKE_OP_FRAMES", "0")
    with pytest.raises(RuntimeError, match="wrote no keypoint files for cam01.avi") as ei:
        run(root, video, tmp_path / "out")
    assert "OpenPose demo successfully finished" in str(ei.value)


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


def test_decode_falls_back_for_non_utf8_output():
    assert ob._decode("Fehler: Datei fehlt".encode()) == "Fehler: Datei fehlt"
    assert ob._decode("C:\\データ".encode()) == "C:\\データ"
    assert ob._decode(b"C:\\caf\xe9\xff").startswith("C:\\caf")  # never raises


def test_exit_text_shows_windows_ntstatus():
    assert ob._exit_text(1) == "exit 1"
    assert ob._exit_text(3221225781) == "exit 3221225781 = 0xC0000135"


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
