"""GUI: Results page "Angles in -180..180 deg" box (checked by default, passed to export_all) and
the throwing angles: <base>_throwing.csv next to "Export all angles (CSV)…" when the .trc of
the loaded .mot exists, and the "Throwing angles from .trc…" button (also without a project)."""

import csv

import numpy as np
import pytest
from PySide6.QtWidgets import QFileDialog, QMessageBox

from tests.synth import throwing_pose, write_trc

pytestmark = pytest.mark.filterwarnings("ignore:.*AA_UseHighDpiPixmaps.*:DeprecationWarning")


@pytest.fixture
def boxes(monkeypatch):
    """Record information / warning / critical boxes instead of showing them."""
    shown = []

    def box(kind):
        def f(parent, title, text, *a, **k):
            shown.append((kind, title, text))
            return QMessageBox.StandardButton.Ok
        return f

    for kind in ("information", "warning", "critical"):
        monkeypatch.setattr(QMessageBox, kind, box(kind))
    return shown


@pytest.fixture
def dialogs(monkeypatch):
    """``dialogs["save"]`` / ``dialogs["open"]``: the paths the file dialogs return ("" =
    cancelled); ``dialogs["calls"]``: (kind, start directory) of each call."""
    d = {"save": "", "open": "", "calls": []}

    def save(parent, caption, start="", *a, **k):
        d["calls"].append(("save", start))
        return d["save"], "CSV (*.csv)"

    def open_(parent, caption, start="", *a, **k):
        d["calls"].append(("open", start))
        return d["open"], ""

    monkeypatch.setattr(QFileDialog, "getSaveFileName", save)
    monkeypatch.setattr(QFileDialog, "getOpenFileName", open_)
    return d


def _page(project=None):
    from poseassess.gui.pages.results_page import ResultsPage
    from poseassess.gui.state import AppState

    state = AppState()
    page = ResultsPage(state)
    if project is not None:
        state.set_project(project)
    return page


def _write_mot(path, n=50):
    """A Pose2Sim-like .mot whose arm_rot_r drifted by a whole turn (400 deg = 40 deg)."""
    t = np.arange(n) / 50.0
    cols = {"time": t, "pelvis_tx": np.linspace(0, 0.5, n),
            "hip_flexion_r": 20 * np.sin(2 * np.pi * t), "hip_flexion_l": 18 * np.sin(2 * np.pi * t),
            "arm_rot_r": np.full(n, 400.0), "arm_flex_r": np.full(n, -200.0)}
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [path.name, "version=1", f"nRows={n}", f"nColumns={len(cols)}", "inDegrees=yes",
             "endheader", "\t".join(cols)]
    lines += ["\t".join(f"{c[i]:.6f}" for c in cols.values()) for i in range(n)]
    path.write_text("\n".join(lines) + "\n")
    return path


def _write_trc(path, ers=(30.0, 90.0, 150.0), drop=()):
    """Pose2Sim-style .trc (Y-up) of ``throwing_pose`` frames with right/left ER ``ers``."""
    frames = [throwing_pose(e, e - 10.0) for e in ers]
    names = [n for n in frames[0] if n not in drop]
    pts = np.array([[f[n] for n in names] for f in frames])[..., [1, 2, 0]]  # zup -> yup
    return write_trc(path, names, pts, 100.0)


def _csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    return {c: [r[i] for r in rows[2:]] for i, c in enumerate(rows[0])}, rows[1]


# ------------------------------------------------------------------ -180..180 box
def test_wrap_box_default_tooltip_and_layout(qapp):
    page = _page()
    assert page.wrap_check.isChecked()
    assert page.wrap_check.text() == "Angles in −180..180°"
    tip = page.wrap_check.toolTip()
    assert "whole turns" in tip and "400° = 40°" in tip and "jumps by 360°" in tip
    assert page.throwing_btn.isEnabled()  # works without a project / motion
    # the new row sits under the buttons and is narrower: the page does not get wider
    assert page._export_opts.indexOf(page.wrap_check) >= 0
    assert page._export_opts.indexOf(page.throwing_btn) >= 0
    assert page._top_bar.indexOf(page.wrap_check) < 0
    page.show()
    assert page._export_opts.minimumSize().width() < page._top_bar.minimumSize().width()
    page.close()


def test_export_all_uses_the_wrap_box(qapp, project, tmp_path, boxes, dialogs):
    page = _page(project)
    page._load(_write_mot(project.kinematics_dir / "trial.mot"))
    dialogs["save"] = str(tmp_path / "out" / "wrapped.csv")
    page.export_csv_btn.click()
    cols, units = _csv(tmp_path / "out" / "wrapped_angles.csv")
    assert set(cols["arm_rot_r"]) == {"40.00000"} and set(cols["arm_flex_r"]) == {"160.00000"}
    assert cols["pelvis_tx"][-1] == "0.50000"
    summary = (tmp_path / "out" / "wrapped_summary.csv").read_text().splitlines()
    assert next(r for r in summary if r.startswith("arm_rot_r,")).split(",")[2] == "40.000"
    assert boxes[-1][0] == "information" and "−180..180°" in boxes[-1][2]
    assert "_throwing" not in boxes[-1][2]  # no .trc for this .mot
    assert not (tmp_path / "out" / "wrapped_throwing.csv").exists()
    # unchecked: the values exactly as OpenSim wrote them
    page.wrap_check.setChecked(False)
    dialogs["save"] = str(tmp_path / "out" / "raw.csv")
    page.export_csv_btn.click()
    cols, _ = _csv(tmp_path / "out" / "raw_angles.csv")
    assert set(cols["arm_rot_r"]) == {"400.00000"}
    assert "−180..180°" not in boxes[-1][2]


def test_export_all_passes_the_box_to_export_all(qapp, project, tmp_path, boxes, dialogs,
                                                 monkeypatch):
    from poseassess.core import assessment

    got = []
    real = assessment.export_all
    monkeypatch.setattr(assessment, "export_all",
                        lambda *a, **k: got.append(k.get("wrap_degrees")) or real(*a, **k))
    page = _page(project)
    page._load(_write_mot(project.kinematics_dir / "trial.mot"))
    dialogs["save"] = str(tmp_path / "x.csv")
    page.export_csv_btn.click()
    page.wrap_check.setChecked(False)
    page.export_csv_btn.click()
    assert got == [True, False]


# ------------------------------------------------------------------ throwing CSV with "Export all"
def test_export_all_writes_throwing_csv_for_the_matching_trc(qapp, project, tmp_path, boxes,
                                                             dialogs):
    _write_trc(project.pose3d_dir / "trial_filt_butterworth.trc")
    _write_trc(project.pose3d_dir / "other.trc", ers=(0.0, 0.0, 0.0))
    page = _page(project)
    page._load(_write_mot(project.kinematics_dir / "trial_filt_butterworth.mot"))
    assert page._matching_trc() == project.pose3d_dir / "trial_filt_butterworth.trc"
    dialogs["save"] = str(tmp_path / "out" / "trial")
    page.export_csv_btn.click()
    out = tmp_path / "out" / "trial_throwing.csv"
    assert out.is_file()
    cols, units = _csv(out)
    assert units == ["s"] + ["deg"] * 6
    assert [float(v) for v in cols["shoulder_er_r"]] == pytest.approx([30, 90, 150], abs=1e-3)
    assert [float(v) for v in cols["shoulder_er_l"]] == pytest.approx([20, 80, 140], abs=1e-3)
    kind, _title, text = boxes[-1]
    assert kind == "information"
    assert "trial_throwing.csv" in text and "trial_filt_butterworth.trc" in text
    assert "trial_angles.csv" in text and "trial_summary.csv" in text


def test_trc_next_to_a_mot_without_project(qapp, tmp_path, boxes, dialogs):
    folder = tmp_path / "saved"
    _write_trc(folder / "pitch.trc")
    page = _page()
    page._load(_write_mot(folder / "pitch.mot"))
    assert page._matching_trc() == folder / "pitch.trc"
    dialogs["save"] = str(tmp_path / "res" / "pitch.csv")
    page.export_csv_btn.click()
    assert (tmp_path / "res" / "pitch_throwing.csv").is_file()


def test_throwing_failure_keeps_the_angle_export(qapp, project, tmp_path, boxes, dialogs):
    _write_trc(project.pose3d_dir / "trial.trc", drop=("RHip",))
    page = _page(project)
    page._load(_write_mot(project.kinematics_dir / "trial.mot"))
    dialogs["save"] = str(tmp_path / "trial.csv")
    page.export_csv_btn.click()
    assert (tmp_path / "trial_angles.csv").is_file()
    assert not (tmp_path / "trial_throwing.csv").exists()
    kind, _title, text = boxes[-1]
    assert kind == "information" and "Throwing angles not written" in text and "RHip" in text


# ------------------------------------------------------------------ "Throwing angles from .trc…"
def test_throwing_button_without_project(qapp, tmp_path, boxes, dialogs):
    trc = _write_trc(tmp_path / "anywhere" / "throw.trc")
    page = _page()
    dialogs["open"] = str(trc)
    dialogs["save"] = str(tmp_path / "results" / "throw_angles")  # no suffix: .csv added
    page.throwing_btn.click()
    out = tmp_path / "results" / "throw_angles.csv"
    assert out.is_file()
    cols, _ = _csv(out)
    assert [float(v) for v in cols["shoulder_er_r"]] == pytest.approx([30, 90, 150], abs=1e-3)
    assert dialogs["calls"][0] == ("open", "")
    assert dialogs["calls"][1][1] == str(trc.parent / "throw_throwing.csv")
    kind, _title, text = boxes[-1]
    assert kind == "information" and "throw_angles.csv" in text and "3 frames" in text
    assert "R 150.0°" in text and "L 140.0°" in text


def test_throwing_button_starts_in_pose3d_and_reports_errors(qapp, project, tmp_path, boxes,
                                                             dialogs):
    trc = _write_trc(project.pose3d_dir / "bad.trc", drop=("LShoulder",))
    page = _page(project)
    dialogs["open"] = str(trc)
    dialogs["save"] = str(tmp_path / "bad_throwing.csv")
    page.throwing_btn.click()
    assert dialogs["calls"][0] == ("open", str(project.pose3d_dir))
    assert dialogs["calls"][1][1] == str(project.root / "bad_throwing.csv")
    kind, title, text = boxes[-1]
    assert kind == "critical" and "LShoulder" in text
    assert not (tmp_path / "bad_throwing.csv").exists()
    # cancelled dialogs: nothing happens
    n = len(boxes)
    dialogs["open"] = ""
    page.throwing_btn.click()
    dialogs["open"], dialogs["save"] = str(trc), ""
    page.throwing_btn.click()
    assert len(boxes) == n
