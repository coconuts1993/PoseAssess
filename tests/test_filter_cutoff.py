"""Butterworth low-pass cut-off of Pose2Sim's filtering stage as a project setting:
project.toml (default 6 Hz, old projects), Config.toml (clamped below the Nyquist frequency),
the filtering stage's log line and the "Filter cut-off" row of 1. Project (offscreen)."""
from __future__ import annotations

import math
import sys

import pytest

from poseassess.core.config_gen import (
    effective_filter_cutoff,
    max_filter_cutoff,
    parse_filter_cutoff,
)
from poseassess.core.project import (
    DEFAULT_FILTER_CUTOFF_HZ,
    PROJECT_FILE,
    Project,
    ProjectConfig,
)
from tests.qtutil import pump

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib

import tomli_w


def read_toml(path) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


def butterworth(config_file) -> dict:
    """[filtering] of a Config.toml, with the [filtering.butterworth] table."""
    return read_toml(config_file)["filtering"]


def drop_key_from_project_toml(project: Project, key: str) -> None:
    """Make `project`'s project.toml look like one written before `key` existed."""
    data = read_toml(project.project_file)
    data.pop(key)
    with open(project.project_file, "wb") as f:
        tomli_w.dump(data, f)


# --------------------------------------------------------------------------- #
# project.toml
# --------------------------------------------------------------------------- #
def test_default_is_pose2sim_default():
    assert DEFAULT_FILTER_CUTOFF_HZ == 6
    assert ProjectConfig().filter_cutoff_hz == 6
    assert ProjectConfig.from_dict({}).filter_cutoff_hz == 6


def test_default_matches_the_pose2sim_template():
    pytest.importorskip("Pose2Sim")
    from poseassess.core.config_gen import pose2sim_template_path

    filt = butterworth(pose2sim_template_path())
    assert filt["type"] == "butterworth"
    assert filt["butterworth"]["cut_off_frequency"] == DEFAULT_FILTER_CUTOFF_HZ


def test_saved_and_loaded(project):
    project.config.filter_cutoff_hz = 13
    project.save()
    assert read_toml(project.project_file)["filter_cutoff_hz"] == 13
    assert Project.load(project.root).config.filter_cutoff_hz == 13


def test_old_project_loads_with_the_default(project):
    project.config.frame_rate = 120
    project.save()
    drop_key_from_project_toml(project, "filter_cutoff_hz")
    assert "filter_cutoff_hz" not in (project.root / PROJECT_FILE).read_text("utf-8")
    loaded = Project.load(project.root)
    assert loaded.config.filter_cutoff_hz == 6 and loaded.config.frame_rate == 120


# --------------------------------------------------------------------------- #
# clamping
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("frame_rate, top", [
    (60, 29), (61, 30), (30, 14), (25, 12), (120, 59), (240, 119), (1000, 499),
    (5, 2), (4, 1), (3, 1), (2, 1), (1, 1),
])
def test_max_filter_cutoff(frame_rate, top):
    assert max_filter_cutoff(frame_rate) == top
    if frame_rate >= 3:               # no whole Hz is below Nyquist at 1-2 fps
        assert top < frame_rate / 2


@pytest.mark.parametrize("cutoff, frame_rate, expected", [
    (6, 60, 6), (13, 60, 13), (29, 60, 29), (30, 60, 29), (500, 60, 29),
    (0, 60, 1), (-4, 60, 1), (20, 30, 14), (15, 30, 14), (14, 30, 14),
    (13.4, 60, 13), (13.9, 60, 13), ("17", 60, 17), ("13.4", 60, 13),
    ("abc", 60, 6), (None, 60, 6), (float("nan"), 60, 6), (float("inf"), 60, 6),
    (6, 10, 4), (6, 2, 1),
])
def test_effective_filter_cutoff(cutoff, frame_rate, expected):
    assert effective_filter_cutoff(cutoff, frame_rate) == expected


def test_parse_filter_cutoff():
    assert parse_filter_cutoff(13.7) == 13          # int() like Pose2Sim
    assert parse_filter_cutoff(0) == 1
    assert parse_filter_cutoff("x") == DEFAULT_FILTER_CUTOFF_HZ
    assert parse_filter_cutoff(400) == 400          # no frame rate: not clamped


def test_scipy_accepts_every_effective_cutoff():
    """What Pose2Sim does with it: signal.butter(order / 2, cutoff / (frame_rate / 2))."""
    signal = pytest.importorskip("scipy.signal")
    for frame_rate in range(3, 241):
        for wanted in (1, 6, 13, frame_rate // 2, frame_rate, 10_000):
            cutoff = effective_filter_cutoff(wanted, frame_rate)
            assert 1 <= cutoff < frame_rate / 2
            b, a = signal.butter(4 / 2, cutoff / (frame_rate / 2), "low", analog=False)
            assert all(math.isfinite(x) for x in [*b, *a])


# --------------------------------------------------------------------------- #
# Config.toml
# --------------------------------------------------------------------------- #
@pytest.fixture
def pose2sim():
    return pytest.importorskip("Pose2Sim")


def test_config_toml_default(project, pose2sim):
    from poseassess.core.config_gen import generate_config

    filt = butterworth(generate_config(project))
    assert filt["type"] == "butterworth"
    assert filt["butterworth"]["cut_off_frequency"] == 6
    assert filt["butterworth"]["order"] == 4      # the template's, unchanged
    assert filt["display_figures"] is False


def test_config_toml_writes_the_project_value(project, pose2sim):
    from poseassess.core.config_gen import generate_config

    project.config.frame_rate = 60
    project.config.filter_cutoff_hz = 13
    data = read_toml(generate_config(project))
    assert data["filtering"]["butterworth"]["cut_off_frequency"] == 13
    assert isinstance(data["filtering"]["butterworth"]["cut_off_frequency"], int)
    assert data["filtering"]["type"] == "butterworth"
    assert data["project"]["frame_rate"] == 60


def test_config_toml_clamps_below_nyquist(project, pose2sim):
    from poseassess.core.config_gen import generate_config

    project.config.frame_rate = 30
    project.config.filter_cutoff_hz = 20
    assert butterworth(generate_config(project))["butterworth"]["cut_off_frequency"] == 14
    project.config.frame_rate = 60
    project.config.filter_cutoff_hz = 30
    assert butterworth(generate_config(project))["butterworth"]["cut_off_frequency"] == 29
    project.config.filter_cutoff_hz = 0
    assert butterworth(generate_config(project))["butterworth"]["cut_off_frequency"] == 1


def test_config_toml_of_an_old_project(project, pose2sim):
    from poseassess.core.config_gen import generate_config

    drop_key_from_project_toml(project, "filter_cutoff_hz")
    old = Project.load(project.root)
    assert butterworth(generate_config(old))["butterworth"]["cut_off_frequency"] == 6


# --------------------------------------------------------------------------- #
# filtering stage log
# --------------------------------------------------------------------------- #
def run_filtering(project, monkeypatch) -> list[str]:
    """Pipeline.filtering() with Pose2Sim's stage replaced; returns the log lines."""
    from poseassess.core import pipeline

    ran = []
    monkeypatch.setattr(pipeline.Pipeline, "_pose2sim_stage",
                        lambda self, name: ran.append(name) or pipeline.StageResult(name, True))
    logs: list[str] = []
    res = pipeline.Pipeline(project, progress=logs.append).filtering()
    assert res.ok and ran == ["filtering"]
    return logs


def test_filtering_logs_the_cutoff(project, pose2sim, monkeypatch):
    from poseassess.core.config_gen import generate_config

    project.config.frame_rate = 60
    project.config.filter_cutoff_hz = 13
    generate_config(project)
    assert run_filtering(project, monkeypatch) == [
        "[filtering] Butterworth low-pass 13 Hz (Config.toml)"]


def test_filtering_log_follows_config_toml(project, monkeypatch):
    def write(filtering: dict) -> None:
        with open(project.config_file, "wb") as f:
            tomli_w.dump({"filtering": filtering}, f)

    write({"type": "butterworth", "butterworth": {"cut_off_frequency": 8, "order": 4}})
    assert run_filtering(project, monkeypatch) == [
        "[filtering] Butterworth low-pass 8 Hz (Config.toml)"]
    write({"type": "kalman"})
    assert run_filtering(project, monkeypatch) == [
        "[filtering] filter type 'kalman' (Config.toml)"]
    write({"filter": False, "type": "butterworth"})
    assert run_filtering(project, monkeypatch) == [
        "[filtering] no low-pass filter (filter = false in Config.toml)"]
    project.config_file.write_text("not = [toml", encoding="utf-8")
    assert run_filtering(project, monkeypatch) == []
    project.config_file.unlink()
    assert run_filtering(project, monkeypatch) == []    # Pose2Sim reports it itself


# --------------------------------------------------------------------------- #
# 1. Project: "Filter cut-off" row
# --------------------------------------------------------------------------- #
@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    """Per-test settings folder, no auto-detected OpenPose install, no nvidia-smi."""
    from poseassess.core import openpose_setup as ops

    monkeypatch.setenv("POSEASSESS_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.setattr(ops, "candidate_dirs", lambda: [])
    monkeypatch.setattr(ops, "gpu_info", lambda **kw: {"gpus": []})


@pytest.fixture
def page(qapp):
    from poseassess.gui.pages.project_page import ProjectPage
    from poseassess.gui.state import AppState

    p = ProjectPage(AppState())
    yield p
    p.shutdown()
    p.deleteLater()


def test_row_under_frame_rate(page):
    from PySide6.QtWidgets import QFormLayout

    spin = page.cutoff_spin
    form = page._form
    row, _role = form.getWidgetPosition(spin)
    fps_row, _role = form.getWidgetPosition(page.fps_spin)
    assert row == fps_row + 1
    assert form.itemAt(row, QFormLayout.LabelRole).widget().text() == "Filter cut-off"
    assert spin.suffix() == " Hz" and spin.value() == 6 and spin.minimum() == 1
    assert page.fps_spin.value() == 60 and spin.maximum() == 29
    tip = spin.toolTip()
    for text in ("Butterworth", "4th order", "zero-lag", "before inverse kinematics",
                 "3-6 Hz", "6-15 Hz", "12-20 Hz", "same value", "Filter 3D"):
        assert text in tip, text


def test_maximum_follows_the_frame_rate(page):
    spin = page.cutoff_spin
    for fps, top in ((30, 14), (61, 30), (120, 59), (1000, 499), (2, 1), (1, 1), (60, 29)):
        page.fps_spin.setValue(fps)
        assert spin.maximum() == top, fps
    spin.setValue(29)
    assert spin.value() == 29
    spin.setValue(40)
    assert spin.value() == 29                 # never at or above Nyquist

    spin.setValue(13)
    page.fps_spin.setValue(20)
    assert spin.maximum() == 9 and spin.value() == 9
    page.fps_spin.setValue(60)
    assert spin.value() == 13                 # the value asked for comes back

    for fps in (1, 12, 120):                  # typing "120" into Frame rate
        page.fps_spin.setValue(fps)
    assert spin.maximum() == 59 and spin.value() == 13

    page.fps_spin.setValue(20)
    spin.setValue(5)                          # chosen while the maximum is low
    page.fps_spin.setValue(60)
    assert spin.value() == 5


def test_typing_a_frame_rate_keeps_the_cutoff(page):
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest

    page.settings_box.setEnabled(True)        # as with a project open
    page.cutoff_spin.setValue(18)
    fps = page.fps_spin
    fps.selectAll()
    QTest.keyClicks(fps, "120")               # 1, 12, 120: maximum 1, 5, 59 on the way
    QTest.keyClick(fps, Qt.Key_Enter)
    assert fps.value() == 120
    assert page.cutoff_spin.value() == 18 and page.cutoff_spin.maximum() == 59


def test_create_stores_the_cutoff(page, tmp_path, pose2sim):
    root = tmp_path / "new_project"
    page.path_edit.setText(str(root))
    page.fps_spin.setValue(120)
    page.cutoff_spin.setValue(15)
    page._create()
    proj = page.state.project
    assert proj is not None and proj.config.filter_cutoff_hz == 15
    assert Project.load(root).config.filter_cutoff_hz == 15
    assert butterworth(proj.config_file)["butterworth"]["cut_off_frequency"] == 15


def test_save_stores_the_cutoff(page, project, pose2sim):
    page.state.set_project(project)
    assert page.cutoff_spin.value() == 6 and page.cutoff_spin.maximum() == 14   # 30 fps
    page.fps_spin.setValue(60)
    page.cutoff_spin.setValue(18)
    page._save()
    assert project.config.filter_cutoff_hz == 18 and project.config.frame_rate == 60
    assert Project.load(project.root).config.filter_cutoff_hz == 18
    assert butterworth(project.config_file)["butterworth"]["cut_off_frequency"] == 18
    assert "Saved." in page.status.text()


def test_load_shows_the_project_value(page, project, tmp_path):
    project.config.frame_rate = 60
    project.config.filter_cutoff_hz = 13
    project.save()
    page.state.set_project(Project.load(project.root))
    assert page.fps_spin.value() == 60 and page.cutoff_spin.value() == 13

    # another project: the frame rate is set first, so its cut-off is not clipped
    other = Project(tmp_path / "other")
    other.config.frame_rate = 240
    other.config.filter_cutoff_hz = 40
    other.create()
    page.state.set_project(Project.load(other.root))
    assert page.cutoff_spin.maximum() == 119 and page.cutoff_spin.value() == 40

    # back to the 60 fps one, then a value typed into project.toml above Nyquist
    project.config.filter_cutoff_hz = 45
    page.state.set_project(project)
    assert page.cutoff_spin.value() == 29        # what Config.toml gets

    project.config.filter_cutoff_hz = "fast"     # not a number
    page.state.set_project(project)
    assert page.cutoff_spin.value() == 6


def test_load_old_project_shows_the_default(page, project):
    page.cutoff_spin.setValue(11)
    drop_key_from_project_toml(project, "filter_cutoff_hz")
    page.state.set_project(Project.load(project.root))
    assert page.cutoff_spin.value() == 6


def test_window_min_width_unchanged_by_the_row(qapp):
    """Like .github/portable/gui_sizes.py (offscreen): the "Filter cut-off" row must not make
    the 1. Project page or the main window wider than the form without it."""
    from poseassess.gui.main_window import MainWindow

    w = MainWindow()
    try:
        page = w.pages[0]
        w.show()
        pump(0.05)
        with_row = page.minimumSizeHint().width(), w.minimumSizeHint().width()

        form = page._form
        index, _role = form.getWidgetPosition(page.cutoff_spin)
        taken = form.takeRow(page.cutoff_spin)
        label, field = taken.labelItem.widget(), taken.fieldItem.widget()
        label.hide()
        field.hide()
        pump(0.05)
        without_row = page.minimumSizeHint().width(), w.minimumSizeHint().width()
        assert with_row[0] <= without_row[0] and with_row[1] <= without_row[1]

        form.insertRow(index, label, field)
        label.show()
        field.show()
        page.fps_spin.setValue(1000)               # the widest value: "499 Hz"
        page.cutoff_spin.setValue(499)
        pump(0.05)
        assert page.minimumSizeHint().width() <= without_row[0]
        assert w.minimumSizeHint().width() <= without_row[1]
    finally:
        w.close()
        w.deleteLater()


@pytest.mark.parametrize("frame_rate, filtered", [(1, False), (2, False), (3, True), (60, True)])
def test_no_low_pass_where_no_whole_hz_is_below_nyquist(project, pose2sim, frame_rate, filtered):
    """At 1-2 fps every whole-Hz cut-off reaches Nyquist (scipy would fail): filter = false,
    so Pose2Sim's filtering stage only copies the data instead of stopping with an error."""
    from poseassess.core.config_gen import generate_config

    project.config.frame_rate = frame_rate
    filt = read_toml(generate_config(project))["filtering"]
    assert filt.get("filter", True) is filtered
    assert filt["butterworth"]["cut_off_frequency"] >= 1
