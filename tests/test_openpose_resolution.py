"""OpenPose network resolution: default -1x368, the project setting and its way to the
OpenPoseDemo command line, and the speed / time-left text of the progress lines.

-1x480 (CMU's suggestion for BODY_25B) needs more than 11 GB of GPU memory: a user measured
0.9 fps on a GTX 1080 Ti (11 GB) against 16 fps with OpenPose's default -1x368.
"""
from __future__ import annotations

import pytest

from poseassess.core import openpose_setup as ops
from poseassess.core.project import Project, ProjectConfig
from poseassess.plugins import openpose_backend as ob


# --------------------------------------------------------------------------- #
# values
# --------------------------------------------------------------------------- #
def test_default_is_openposes_own_and_first_choice():
    assert ops.NET_RESOLUTION == "-1x368"
    assert ops.NET_RESOLUTIONS[0][0] == ops.NET_RESOLUTION
    values = [v for v, _ in ops.NET_RESOLUTIONS]
    assert "-1x480" in values and len(set(values)) == len(values)
    assert all(ops.normalize_net_resolution(v) == v for v in values)
    assert "11 GB" in dict(ops.NET_RESOLUTIONS)["-1x480"]


@pytest.mark.parametrize("raw, expected", [
    ("-1x368", "-1x368"), (" -1X480 ", "-1x480"), ("-1 x 256", "-1x256"),
    ("-1x128", "-1x128"), ("-1x1024", "-1x1024"),
    ("-1x370", None),      # not a multiple of 16
    ("-1x112", None), ("-1x1040", None),  # out of range
    ("368", None), ("656x368", None), ("-1x", None), ("", None), (None, None), (480, None),
])
def test_normalize(raw, expected):
    assert ops.normalize_net_resolution(raw) == expected
    assert ops.net_resolution_or_default(raw) == (expected or ops.NET_RESOLUTION)


# --------------------------------------------------------------------------- #
# project setting
# --------------------------------------------------------------------------- #
def test_project_setting_round_trip_and_old_projects(project):
    assert ProjectConfig().openpose_net_resolution == ""
    project.config.openpose_net_resolution = "-1x480"
    project.save()
    assert Project.load(project.root).config.openpose_net_resolution == "-1x480"

    # a project.toml written before the setting existed: default
    d = project.config.to_dict()
    d.pop("openpose_net_resolution")
    assert ProjectConfig.from_dict(d).openpose_net_resolution == ""


@pytest.mark.parametrize("setting, expected, warned", [
    ("", "-1x368", False),
    ("-1x480", "-1x480", False),
    ("-1x256", "-1x256", False),
    ("-1x370", "-1x368", True),
    ("big", "-1x368", True),
])
def test_pipeline_passes_the_setting_to_the_backend(project, setting, expected, warned):
    from poseassess.core.pipeline import Pipeline

    logged: list[str] = []
    project.config.openpose_net_resolution = setting
    assert Pipeline(project, logged.append)._openpose_net_resolution() == expected
    assert any("not usable" in m and repr(setting) in m for m in logged) == warned


def test_pipeline_options_reach_the_command(project, tmp_path, monkeypatch):
    """_run_plugin_pose gives the backend net_resolution (OpenPose only)."""
    from poseassess.core.pipeline import Pipeline
    from poseassess.plugins import base

    made: list[tuple[str, dict]] = []

    class Stop(Exception):
        pass

    def fake_get(name, **options):
        made.append((name, options))
        raise Stop

    monkeypatch.setattr(base, "get_pose2d_backend", fake_get)
    monkeypatch.setattr(Pipeline, "_openpose_dir", lambda self: "")
    project.config.openpose_net_resolution = "-1x480"
    for backend in ("openpose", "mediapipe"):
        with pytest.raises(Stop):
            Pipeline(project)._run_plugin_pose(backend)
    assert made[0] == ("openpose", {"openpose_dir": "", "net_resolution": "-1x480"})
    assert made[1][0] == "mediapipe" and "net_resolution" not in made[1][1]

    be = ob.OpenPoseBackend(openpose_dir="", net_resolution="-1x480")
    cmd = be.command(tmp_path / "OpenPoseDemo.exe", "v.mp4", "out")
    assert cmd[cmd.index("--net_resolution") + 1] == "-1x480"
    for given in (None, "", "-1x370"):           # missing / unusable: the default
        cmd = ob.OpenPoseBackend(net_resolution=given).command(tmp_path / "x.exe", "v", "o")
        assert cmd[cmd.index("--net_resolution") + 1] == "-1x368"


# --------------------------------------------------------------------------- #
# progress text
# --------------------------------------------------------------------------- #
def test_frames_text_with_speed_and_time_left():
    assert ob._frames_text("cam01", 5, 100) == "cam01: 5/100 frames (5%)"
    assert ob._frames_text("cam01", 5, None) == "cam01: 5 frames"
    assert (ob._frames_text("cam01", 1234, 8068, 15.83)
            == "cam01: 1234/8068 frames (15%), 15.8 fps, about 7 min left")
    assert (ob._frames_text("cam01", 10, 8068, 0.9)
            == "cam01: 10/8068 frames (0%), 0.9 fps, about 2 h 29 min left")
    assert ob._frames_text("cam01", 100, 100, 9.0) == "cam01: 100/100 frames (100%), 9.0 fps"
    assert ob._frames_text("cam01", 7, None, 3.0) == "cam01: 7 frames, 3.0 fps"
    assert ob._frames_text("cam01", 7, 10, 0.0) == "cam01: 7/10 frames (70%)"
    # still a progress line for the Run page's throttling
    from poseassess.gui.workers import _PROGRESS_RE
    assert _PROGRESS_RE.search(ob._frames_text("cam01", 1234, 8068, 15.8))


@pytest.mark.parametrize("seconds, text", [
    (0, "0 s"), (59.4, "59 s"), (60, "1 min"), (89, "1 min"), (90, "2 min"),
    (3569, "59 min"), (3600, "1 h 00 min"), (8939, "2 h 29 min")])
def test_duration_text(seconds, text):
    assert ob._duration_text(seconds) == text


def test_speed_over_a_sliding_window():
    s = ob._Speed(window_s=10)
    assert s.add(0.0, 0) is None                 # one sample: no speed yet
    assert s.add(1.0, 16) == pytest.approx(16.0)
    assert s.add(5.0, 80) == pytest.approx(16.0)
    # slows down: after the window the old fast part no longer counts
    for t in range(6, 31):
        s.add(float(t), 80 + 2 * (t - 5))
    assert s.add(31.0, 80 + 2 * 26) == pytest.approx(2.0)
    assert s.add(31.0, 80 + 2 * 26) == pytest.approx(2.0)  # same time again: no division by 0
