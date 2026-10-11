"""The pipeline subprocess must work on Windows with a non-UTF-8 code page.

Reported on a Japanese PC (cp932): "=== pose ===  pose: FAIL UnicodeEncodeError: 'cp932'
codec can't encode character '\\xb7'": the child printed "[pose] Pose2Sim rtmlib: RTMPose-m
· HALPE_26 (balanced)" to a pipe encoded with the ANSI code page. Simulated here with
PYTHONIOENCODING=cp932 (what a Windows pipe gets by default)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parent.parent)
SAMPLE = "[pose] Pose2Sim rtmlib: RTMPose-m · HALPE_26 (balanced) — 30 → 60 Hz"


def _child(code: str, env_extra: dict) -> subprocess.CompletedProcess:
    env = dict(os.environ, PYTHONPATH=ROOT, **env_extra)
    env.pop("PYTHONUTF8", None)
    return subprocess.run([sys.executable, "-c", code], capture_output=True, env=env,
                          cwd=ROOT, timeout=60)


def test_cp932_pipe_reproduces_the_report():
    """Sanity check of the simulation: a plain print fails like on the Japanese PC."""
    r = _child(f"print({SAMPLE!r})", {"PYTHONIOENCODING": "cp932"})
    assert r.returncode != 0 and b"UnicodeEncodeError" in r.stderr and b"cp932" in r.stderr


def test_pipeline_cli_writes_utf8_whatever_the_code_page():
    code = ("from poseassess.core import pipeline_cli; pipeline_cli._utf8_streams(); "
            f"print({SAMPLE!r}, flush=True)")
    r = _child(code, {"PYTHONIOENCODING": "cp932"})
    assert r.returncode == 0, r.stderr.decode("utf-8", "replace")
    assert r.stdout.decode("utf-8").strip() == SAMPLE


def test_worker_runs_the_child_in_utf8(monkeypatch):
    """PipelineWorker forces UTF-8 in the child and decodes its output as UTF-8."""
    from poseassess.gui import workers

    seen = {}

    class FakePopen:
        def __init__(self, args, **kw):
            seen.update(kw, args=args)
            self.stdout = iter([SAMPLE + "\n", "@@STAGE_START pose\n",
                                "@@STAGE_RESULT pose OK done\n", "@@DONE\n"])

        def wait(self):
            return 0

        def poll(self):
            return 0

    monkeypatch.setattr(subprocess, "Popen", FakePopen)
    monkeypatch.setenv("PYTHONIOENCODING", "cp932")
    proj = type("P", (), {"root": Path(ROOT)})()
    w = workers.PipelineWorker(proj, ["pose"])
    logs, done = [], []
    w.log.connect(logs.append)
    w.finished.connect(done.append)
    w.run()
    assert seen["encoding"] == "utf-8" and seen["errors"] == "replace" and seen["text"]
    assert seen["env"]["PYTHONUTF8"] == "1" and seen["env"]["PYTHONIOENCODING"] == "utf-8"
    assert SAMPLE in logs and done and done[0][0].ok


def test_pose_stage_end_to_end_under_cp932(project, qapp):
    """The real child process, as started by the GUI, on a project without videos: the pose
    stage must fail because there are no videos — not with a UnicodeEncodeError."""
    from poseassess.core.config_gen import generate_config
    from poseassess.gui import workers

    project.config.pose2d_backend = "rtmpose"
    project.save()
    generate_config(project)
    os.environ["PYTHONIOENCODING"] = "cp932"  # what the child inherits on a Japanese PC
    try:
        w = workers.PipelineWorker(project, ["pose"])
        logs, done = [], []
        w.log.connect(logs.append)
        w.finished.connect(done.append)
        w.run()
    finally:
        os.environ.pop("PYTHONIOENCODING", None)
    text = "\n".join(logs)
    assert done, text
    res = done[0]
    assert res and res[0].name == "pose"
    assert "UnicodeEncodeError" not in res[0].message, res[0].message
    assert "RTMPose-m · HALPE_26" in text
