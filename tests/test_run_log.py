"""Run page log: progress lines are throttled, messages (warnings, errors) never dropped, and
the last progress line is shown before each stage marker."""

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _run(monkeypatch, lines):
    from poseassess.gui import workers

    class FakePopen:
        def __init__(self, args, **kw):
            self.stdout = iter(line + "\n" for line in lines)

        def wait(self):
            return 0

        def poll(self):
            return 0

    monkeypatch.setattr(subprocess, "Popen", FakePopen)
    proj = type("P", (), {"root": ROOT})()
    w = workers.PipelineWorker(proj, ["pose"])
    logs, done = [], []
    w.log.connect(logs.append)
    w.finished.connect(done.append)
    w.run()
    return logs, done


def test_messages_are_never_dropped(monkeypatch):
    warnings = [f"WARNING: camera {i} is out of sync" for i in range(6)]
    logs, done = _run(monkeypatch, ["@@STAGE_START pose", *warnings,
                                    "  cam01 OpenPose: Error: prototxt not found",
                                    "@@STAGE_RESULT pose FAIL OpenPose failed (exit -1) | Error",
                                    "@@DONE"])
    assert all(w in logs for w in warnings)
    assert "  cam01 OpenPose: Error: prototxt not found" in logs
    assert done and not done[0][0].ok


def test_progress_is_throttled_but_the_last_one_is_shown(monkeypatch):
    progress = [f"  cam01: {i}/600 frames ({i * 100 // 600}%)" for i in range(1, 601)]
    ik = [f"Frame {i} (t = {i / 60:.3f}):\t total squared error = 0.001" for i in range(300)]
    logs, _ = _run(monkeypatch, ["@@STAGE_START pose", *progress, "@@STAGE_RESULT pose OK done",
                                 "@@STAGE_START kinematics", *ik,
                                 "@@STAGE_RESULT kinematics OK done", "@@DONE"])
    shown_progress = [line for line in logs if "/600 frames" in line]
    assert 1 <= len(shown_progress) < 20  # throttled
    assert shown_progress[-1] == progress[-1]  # the final state before the stage ended
    shown_ik = [line for line in logs if line.startswith("Frame ")]
    assert shown_ik and shown_ik[-1] == ik[-1] and len(shown_ik) < 20
    assert logs.index(progress[-1]) < logs.index("=== kinematics ===")
