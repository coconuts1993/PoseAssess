"""GUI-VIEW: Results page keeps the joint-angle tab unchanged; Balance (Wii) tab metrics/plots on demo_trial, AlignmentPanel (import recording, detect sync event, manual offset saved to trial.json), export buttons."""

import json

import numpy as np
import pytest
from PySide6.QtWidgets import QFileDialog, QMessageBox

from tests.qtutil import pump, pump_until

# matplotlib's toolbar icons query a Qt attribute that PySide6 marks as deprecated
pytestmark = pytest.mark.filterwarnings("ignore:.*AA_UseHighDpiPixmaps.*:DeprecationWarning")


@pytest.fixture
def no_dialogs(monkeypatch):
    """Answer every question with Yes and swallow information / warning boxes."""
    shown = []

    def box(kind):
        def f(parent, title, text, *a, **k):
            shown.append((kind, title, text))
            return QMessageBox.StandardButton.Ok
        return f

    monkeypatch.setattr(QMessageBox, "information", box("information"))
    monkeypatch.setattr(QMessageBox, "warning", box("warning"))
    monkeypatch.setattr(QMessageBox, "critical", box("critical"))
    monkeypatch.setattr(QMessageBox, "question",
                        lambda *a, **k: shown.append(("question", a[1], a[2]))
                        or QMessageBox.StandardButton.Yes)
    return shown


def _results(state):
    from poseassess.gui.pages.results_page import ResultsPage
    return ResultsPage(state)


def _shown_panel(demo_project):
    """A ResultsPage on ``demo_project`` with the balance tab visible and refreshed."""
    from poseassess.gui.state import AppState

    state = AppState()
    page = _results(state)
    page.resize(1300, 800)
    page.show()
    state.open_project(demo_project.root)
    page.tabs.setCurrentIndex(1)
    pump(0.02)
    return state, page, page.balance


def _table(panel):
    """``{"section/label": [COP column, COM column]}`` of the metrics table."""
    t = panel.table
    out, section = {}, ""
    for r in range(t.rowCount()):
        label = t.item(r, 0).text()
        if t.columnSpan(r, 0) == 3:
            section = label
            continue
        out[f"{section}/{label}"] = [t.item(r, c).text() if t.item(r, c) else "" for c in (1, 2)]
    return out


def _events(state):
    got = []
    state.balance_changed.connect(got.append)
    return got


def _trial_json(project):
    return json.loads((project.root / "wii" / "trial.json").read_text(encoding="utf-8"))


def _write_mot(path):
    t = np.arange(0, 2, 0.01)
    cols = {"time": t, "hip_flexion_r": 20 * np.sin(2 * np.pi * t),
            "hip_flexion_l": 18 * np.sin(2 * np.pi * t), "knee_angle_r": 40 * np.sin(t),
            "knee_angle_l": 35 * np.sin(t), "pelvis_tilt": 2 * np.cos(t)}
    lines = [path.name, "version=1", f"nRows={len(t)}", f"nColumns={len(cols)}",
             "inDegrees=yes", "endheader", "\t".join(cols)]
    for i in range(len(t)):
        lines.append("\t".join(f"{c[i]:.6f}" for c in cols.values()))
    path.write_text("\n".join(lines) + "\n")
    return path


# ------------------------------------------------------------------ joint angles
def test_joint_angles_tab_is_unchanged(qapp, project):
    from poseassess.gui.state import AppState
    from poseassess.gui.widgets.plot_canvas import PlotCanvas

    state = AppState()
    page = _results(state)
    assert [page.tabs.tabText(i) for i in range(page.tabs.count())] == \
        ["Joint angles", "Balance (Wii)"]
    assert page.tabs.currentIndex() == 0
    split = page.tabs.widget(0)
    assert split.indexOf(page.table.parentWidget()) == 0
    assert isinstance(page.canvas, PlotCanvas) and split.indexOf(page.canvas.parentWidget()) == 1
    assert [page.table.horizontalHeaderItem(i).text() for i in range(5)] == \
        ["Joint", "R ROM°", "L ROM°", "Sym %", "Flag"]
    assert "Symmetry index" in page.note.text()
    state.set_project(project)
    assert "Load latest result" in page.file_label.text()
    mot = _write_mot(project.kinematics_dir / "trial.mot")
    page._load_latest()
    assert page._mot_path == mot and page.table.rowCount() > 0
    assert page.joint_combo.count() > 0 and page.export_btn.isEnabled()


# ------------------------------------------------------------------ balance tab
def test_balance_tab_fills_on_the_demo(qapp, demo_trial):
    from poseassess.core.balance.fusion import balance_summary, fuse_trial

    state, page, panel = _shown_panel(demo_trial["project"])
    assert panel.stack.currentIndex() == 1
    assert panel.fused is not None and panel.fused.recording == demo_trial["rec_id"]
    expected = balance_summary(fuse_trial(demo_trial["project"]))
    rows = _table(panel)
    sway, cc = "Sway (board frame)/", "COM − COP (board frame)/"
    assert rows[sway + "RMS ML (mm)"][0] == f"{expected['cop']['rms_ml_mm']:.1f}"
    assert rows[sway + "RMS AP (mm)"][1] == f"{expected['com']['rms_ap_mm']:.1f}"
    assert rows[sway + "Path length (mm)"][0] == f"{expected['cop']['path_length_mm']:.0f}"
    assert rows[cc + "RMS distance (mm)"][0] == f"{expected['com_cop']['rms_distance_mm']:.1f}"
    assert rows[cc + "RMS ML (mm)"][0] == f"{expected['com_cop']['rms_ml_mm']:.1f}"
    assert rows["Load/Body mass (kg)"][0].startswith("70.0")
    assert "measured" in rows["Load/Body mass (kg)"][0]
    assert set(panel.plot.panels) == {"load", "ml", "ap"}
    assert set(panel.path_plot.panels) == {"path"}
    assert demo_trial["rec_id"] in panel.info.text()
    # the recording combo lists the recording and the status is the manual offset
    al = panel.alignment
    assert al.rec_combo.currentData() == demo_trial["rec_id"]
    assert "+2.500" in al.status_lbl.text()
    assert al.offset_spin.value() == pytest.approx(2.5)
    assert al.align_box.isVisible() and not al.recorded_box.isVisible()
    page.close()


def test_balance_tab_uses_fuse_trial(qapp, demo_trial, monkeypatch):
    from poseassess.core.balance import fusion
    from tests.synth import fused_from_demo

    truth = fused_from_demo(demo_trial)
    monkeypatch.setattr(fusion, "fuse_trial", lambda project, trc_path=None: truth)
    state, page, panel = _shown_panel(demo_trial["project"])
    assert panel.fused is truth
    s = fusion.balance_summary(truth, *panel.window())
    assert panel.summary["cop"] == s["cop"]
    page.close()


def test_window_and_events_menu(qapp, demo_trial):
    state, page, panel = _shown_panel(demo_trial["project"])
    tt = panel.fused.trc_time
    assert panel.window() == pytest.approx((tt.min(), tt.max()), abs=0.01)
    panel.set_window(3.0, 5.5)
    assert panel.summary["window"] == pytest.approx([3.0, 5.5])
    assert panel.summary["frames"] == int(((tt >= 3.0 - 1e-9) & (tt <= 5.5 + 1e-9)).sum())
    assert "3.00–5.50" in panel.info.text()
    assert "load" in panel.plot.panels
    # events menu: start at the "sync" mark (t = 6.0 s on the .trc axis)
    starts = panel.events_menu.actions()[0].menu().actions()
    sync = next(a for a in starts if a.text().startswith("sync"))
    sync.trigger()
    assert panel.window()[0] == pytest.approx(6.0, abs=0.01)
    panel.whole_btn.click()
    assert panel.window() == pytest.approx((tt.min(), tt.max()), abs=0.01)
    # the window survives a refresh of the same .trc
    panel.set_window(1.0, 2.0)
    panel.refresh()
    assert panel.window() == pytest.approx((1.0, 2.0))
    page.close()


def test_manual_offset_is_saved(qapp, demo_trial, no_dialogs):
    state, page, panel = _shown_panel(demo_trial["project"])
    got = _events(state)
    al = panel.alignment
    al.offset_spin.setValue(3.25)
    al.save_btn.click()
    d = _trial_json(demo_trial["project"])
    assert d["alignment"]["method"] == "manual"
    assert d["alignment"]["offset_s"] == pytest.approx(3.25)
    assert d["alignment"]["trc"] == "pose-3d/demo_filt_butterworth.trc"
    assert "alignment" in got
    assert panel.fused.alignment.offset_s == pytest.approx(3.25)  # refreshed
    assert "+3.250" in al.status_lbl.text()
    page.close()


def test_detect_sync_event_then_save(qapp, demo_trial, no_dialogs):
    state, page, panel = _shown_panel(demo_trial["project"])
    al = panel.alignment
    al.offset_spin.setValue(0.0)
    al.detect_btn.click()
    assert al.busy and not al.detect_btn.isEnabled()
    assert pump_until(lambda: not al.busy, timeout=30)
    assert al._result is not None and al._result.ok
    assert al.offset_spin.value() == pytest.approx(demo_trial["offset_s"], abs=1 / 30)
    assert panel.plot_tabs.currentIndex() == 2  # "Alignment check" shown
    pump(0.3)
    assert "force" in al.plot.panels and "height" in al.plot.panels
    assert "Save alignment" in al.result_lbl.text()
    al.save_btn.click()
    d = _trial_json(demo_trial["project"])
    assert d["alignment"]["method"] in ("sync_event", "xcorr")
    assert d["alignment"]["offset_s"] == pytest.approx(demo_trial["offset_s"], abs=1 / 30)
    assert d["alignment"]["details"]["candidates"]
    # a changed spin value after the detection is saved as a manual offset
    al.detect_btn.click()
    assert pump_until(lambda: not al.busy, timeout=30)
    al.offset_spin.setValue(al.offset_spin.value() + 0.1)
    al.save_btn.click()
    assert _trial_json(demo_trial["project"])["alignment"]["method"] == "manual"
    page.close()


def test_detection_result_is_dropped_after_a_project_change(qapp, demo_trial, project,
                                                           no_dialogs):
    state, page, panel = _shown_panel(demo_trial["project"])
    al = panel.alignment
    al.detect_btn.click()
    state.set_project(project)  # while the worker runs
    assert pump_until(lambda: not al.busy, timeout=30)
    assert al._result is None and al._pending is None
    assert panel.stack.currentIndex() == 0
    page.close()


def test_alignment_plot_previews_the_offset(qapp, demo_trial):
    state, page, panel = _shown_panel(demo_trial["project"])
    al = panel.alignment
    panel.plot_tabs.setCurrentIndex(2)
    pump(0.3)
    ax = al.plot.panels["force"]
    x0 = ax.lines[0].get_xdata().copy()
    al.offset_spin.setValue(3.0)
    pump(0.4)
    x1 = al.plot.panels["force"].lines[0].get_xdata()
    assert np.nanmin(x0) - np.nanmin(x1) == pytest.approx(0.5, abs=1e-6)
    page.close()


def test_import_recording_and_choose(qapp, demo_trial, tmp_path, no_dialogs):
    from tests.synth import kg_from_cop, write_wii_recording

    state, page, panel = _shown_panel(demo_trial["project"])
    got = _events(state)
    t = np.arange(0, 5, 0.01)
    src = write_wii_recording(tmp_path / "outside" / "20260202_101010", t,
                              kg_from_cop(np.full(len(t), 65.0), np.zeros((len(t), 2))))
    al = panel.alignment
    rec = al.import_path(src)
    assert rec == "20260202_101010"
    d = _trial_json(demo_trial["project"])
    assert d["recording"] == rec and d["alignment"]["method"] == "none"
    assert "trial" in got
    assert al.rec_combo.currentData() == rec
    assert al.rec_combo.findData(demo_trial["rec_id"]) >= 0
    assert "Not aligned" in al.status_lbl.text()
    assert panel.stack.currentIndex() == 1  # fused with NaN force + a warning
    assert "not aligned" in panel.info.text()
    assert any(k == "question" for k, *_ in no_dialogs)  # the old alignment was confirmed
    # choose the demo recording again (by the combo) and then none
    i = al.rec_combo.findData(demo_trial["rec_id"])
    al.rec_combo.setCurrentIndex(i)
    al._on_recording_chosen(i)
    assert _trial_json(demo_trial["project"])["recording"] == demo_trial["rec_id"]
    al.rec_combo.setCurrentIndex(0)
    al._on_recording_chosen(0)
    assert _trial_json(demo_trial["project"])["recording"] is None
    assert panel.stack.currentIndex() == 0
    assert "No Wii recording" in panel.help.text()
    page.close()


def test_import_invalid_recording_warns(qapp, demo_trial, tmp_path, no_dialogs):
    state, page, panel = _shown_panel(demo_trial["project"])
    bad = tmp_path / "nothing"
    bad.mkdir()
    assert panel.alignment.import_path(bad) is None
    assert any(k == "warning" for k, *_ in no_dialogs)
    assert _trial_json(demo_trial["project"])["recording"] == demo_trial["rec_id"]
    page.close()


def test_recorded_take_is_read_only_until_override(qapp, demo_trial, no_dialogs):
    from poseassess.core.balance.alignment import recorded_alignment
    from poseassess.core.balance.paths import WiiPaths
    from poseassess.core.balance.trial import set_recording

    proj = demo_trial["project"]
    rec = WiiPaths(proj).recording_dir(demo_trial["rec_id"])
    n = 360
    lines = ["frame,t,t_rel,t_unix"] + [f"{i},{1000 + 2.5 + i / 30:.6f},{2.5 + i / 30:.6f},"
                                        f"{1.7e9 + 1000 + 2.5 + i / 30:.6f}" for i in range(n)]
    (rec / "frames.csv").write_text("\n".join(lines) + "\n")
    set_recording(proj, demo_trial["rec_id"], "capture",
                  recorded_alignment(proj, demo_trial["rec_id"]))
    state, page, panel = _shown_panel(proj)
    al = panel.alignment
    assert al.recorded_box.isVisible() and not al.align_box.isVisible()
    assert "Recorded with the videos" in al.status_lbl.text()
    assert panel.fused.alignment.method == "recorded"
    np.testing.assert_allclose(panel.fused.t_rel, panel.fused.trc_time + 2.5, atol=1e-6)
    al.realign_btn.click()
    assert al.align_box.isVisible() and not al.recorded_box.isVisible()
    al.offset_spin.setValue(2.6)
    al.save_btn.click()
    assert _trial_json(proj)["alignment"]["method"] == "manual"
    assert not al._override
    page.close()


def test_body_mass_override(qapp, demo_trial):
    state, page, panel = _shown_panel(demo_trial["project"])
    got = _events(state)
    panel.mass_spin.setValue(80.0)
    panel.mass_spin.editingFinished.emit()
    assert _trial_json(demo_trial["project"])["body_mass_kg"] == pytest.approx(80.0)
    assert "trial" in got
    assert _table(panel)["Load/Body mass (kg)"][0].startswith("80.0")
    assert "entered" in _table(panel)["Load/Body mass (kg)"][0]
    panel.mass_spin.setValue(0.0)
    panel.mass_spin.editingFinished.emit()
    assert _trial_json(demo_trial["project"])["body_mass_kg"] is None
    page.close()


def test_export_writes_the_files(qapp, demo_trial, no_dialogs, monkeypatch):
    state, page, panel = _shown_panel(demo_trial["project"])
    panel.set_window(1.0, 5.0)
    exports = demo_trial["project"].root / "wii" / "exports"
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *a, **k: str(exports))
    panel.export_btn.click()
    stem = "demo_filt_butterworth"
    for suffix in ("_fused.csv", "_balance_summary.json", "_grf.mot"):
        assert (exports / f"{stem}{suffix}").is_file(), suffix
    summary = json.loads((exports / f"{stem}_balance_summary.json").read_text())
    assert summary["window"] == pytest.approx([1.0, 5.0])
    header = (exports / f"{stem}_fused.csv").read_text().splitlines()[0].split(",")
    from poseassess.core.balance.fusion import FUSED_COLUMNS
    assert header == FUSED_COLUMNS
    assert no_dialogs[-1][0] == "information"
    # cancelled dialog: nothing happens
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *a, **k: "")
    n = len(no_dialogs)
    panel.export_btn.click()
    assert len(no_dialogs) == n
    page.close()


def test_help_without_project_or_wii_data(qapp, project):
    from poseassess.gui.state import AppState
    from tests.synth import STANDING, write_trc

    state = AppState()
    page = _results(state)
    page.show()
    page.tabs.setCurrentIndex(1)
    panel = page.balance
    assert panel.stack.currentIndex() == 0 and "Open a project" in panel.help.text()
    assert not panel.export_btn.isEnabled()
    state.set_project(project)
    assert panel.stack.currentIndex() == 0
    assert "No .trc" in panel.help.text() and "2 small" in panel.help.text()
    names = list(STANDING)
    pts = np.array([STANDING[n] for n in names], float)[None].repeat(20, axis=0)
    write_trc(project.pose3d_dir / "x_filt.trc", names, pts[..., [1, 2, 0]], 30.0)
    state.notify_balance_changed("trial")
    assert "No Wii recording" in panel.help.text()
    assert not panel.alignment.detect_btn.isVisible()
    assert not (project.root / "wii").exists()  # looking never writes into the project
    page.close()


def test_hidden_panel_refreshes_when_shown(qapp, demo_trial, monkeypatch):
    from poseassess.core.balance import fusion
    from poseassess.gui.state import AppState

    calls = []
    real = fusion.fuse_trial
    monkeypatch.setattr(fusion, "fuse_trial",
                        lambda p, trc_path=None: calls.append(1) or real(p, trc_path=trc_path))
    state = AppState()
    page = _results(state)
    page.show()  # the joint-angles tab is current: the balance panel is hidden
    state.open_project(demo_trial["project"].root)
    state.notify_balance_changed("alignment")
    assert calls == [] and page.balance._dirty
    page.tabs.setCurrentIndex(1)
    assert calls == [1] and not page.balance._dirty
    state.notify_balance_changed("board")  # visible: refreshed at once
    assert calls == [1, 1]
    page.close()


def test_shutdown_waits_for_detection(qapp, demo_trial, no_dialogs):
    state, page, panel = _shown_panel(demo_trial["project"])
    panel.alignment.detect_btn.click()
    page.shutdown()
    assert panel.alignment._thread is None or panel.alignment._thread.isFinished()
    pump(0.1)
    page.close()


def test_main_window_results_page(qapp, demo_trial):
    from poseassess.gui.main_window import MainWindow

    w = MainWindow()
    try:
        w.state.open_project(demo_trial["project"].root)
        page = next(p for p in w.pages if p.nav_title == "6. Results")
        w.show()
        w.nav.setCurrentRow(w.pages.index(page))
        page.tabs.setCurrentIndex(1)
        pump(0.05)
        assert page.balance.fused is not None
    finally:
        w.close()


# ------------------------------------------------------------------ stale data on disk
def _shift_trc(path, dy=0.10):
    """Rewrite a Pose2Sim .trc (Y-up) with every marker ``dy`` m higher (a pipeline re-run)."""
    lines = path.read_text().splitlines()
    out = lines[:5]
    for ln in lines[5:]:
        parts = ln.split("\t")
        vals = parts[:2] + [f"{float(v) + dy:.6f}" if (k % 3 == 1 and v.strip()) else v
                            for k, v in enumerate(parts[2:])]
        out.append("\t".join(vals))
    path.write_text("\n".join(out) + "\n")
    t = path.stat().st_mtime + 5  # a clearly newer file, whatever the file system's clock
    import os
    os.utime(path, (t, t))


def test_balance_tab_reloads_what_changed_on_disk(qapp, demo_trial):
    """A pipeline run (4. Run) or run_wii.bat change files without any notification: the tab
    re-reads them when it is shown again (and on "Reload")."""
    import shutil

    from poseassess.core.balance.fusion import fuse_trial
    from tests.synth import write_wii_recording

    proj = demo_trial["project"]
    trc = demo_trial["trc"]
    hold = trc.with_suffix(".hold")
    shutil.move(trc, hold)  # the pipeline has not run yet
    state, page, panel = _shown_panel(proj)
    assert panel.fused is None and panel.trc_combo.count() == 0
    page.hide()
    shutil.move(hold, trc)  # ... now it has
    page.show()
    pump(0.02)
    assert panel.fused is not None and panel.trc_combo.count() == 1
    f1 = panel.fused
    z1 = np.nanmean(f1.com_world[:, 2])
    page.hide()
    _shift_trc(trc)  # the pipeline ran again: the .trc was overwritten
    page.show()
    pump(0.02)
    assert panel.fused is not f1
    assert np.nanmean(panel.fused.com_world[:, 2]) == pytest.approx(z1 + 0.10, abs=1e-3)
    assert np.nanmean(fuse_trial(proj, trc).com_world[:, 2]) == \
        pytest.approx(np.nanmean(panel.fused.com_world[:, 2]))
    # a recording added from outside (run_wii.bat --project) shows up in the combo
    page.hide()
    tt = np.arange(0, 3, 0.01)
    write_wii_recording(proj.root / "wii" / "recordings" / "20260102_090000", tt,
                        np.full((len(tt), 4), 17.5))
    page.show()
    pump(0.02)
    al = panel.alignment
    assert al.rec_combo.findData("20260102_090000") >= 0
    # unchanged on disk: showing again does not recompute
    f2 = panel.fused
    page.hide()
    page.show()
    pump(0.02)
    assert panel.fused is f2
    # "Reload" re-reads everything
    assert panel.reload_btn.isEnabled()
    panel.reload_btn.click()
    assert panel.fused is not f2 and panel.fused is not None
    page.close()


def test_check_timing_of_a_recorded_take(qapp, demo_trial, tmp_path, no_dialogs):
    """"Check timing with a sync event": frames stamped 80 ms late (camera latency) are
    measured from the jump / stomp of the take."""
    from poseassess.core.balance.alignment import recorded_alignment
    from poseassess.core.balance.paths import WiiPaths
    from poseassess.core.balance.trial import set_recording
    from tests import synth

    demo = synth.make_demo_trial(tmp_path / "p", physical=True, duration_s=14.0, jump_at=5.0,
                                 stomp_at=9.0, with_images=False)
    proj = demo["project"]
    rec = WiiPaths(proj).recording_dir(demo["rec_id"])
    t_rel = demo["trial"]["t_trc"] + demo["offset_s"] + 0.080
    lines = ["frame,t,t_rel,t_unix"] + [f"{i},{1000 + t:.6f},{t:.6f},{1.7e9 + 1000 + t:.6f}"
                                        for i, t in enumerate(t_rel)]
    (rec / "frames.csv").write_text("\n".join(lines) + "\n")
    set_recording(proj, demo["rec_id"], "capture", recorded_alignment(proj, demo["rec_id"]))
    state, page, panel = _shown_panel(proj)
    al = panel.alignment
    assert al.recorded_box.isVisible() and al.check_btn.isEnabled()
    assert "check it once with a jump" in al.recorded_box.findChild(type(al.status_lbl)).text()
    al.check_btn.click()
    assert pump_until(lambda: al._thread is None and "Timing check" in al.check_lbl.text(), 20)
    assert "older than their timestamps" in al.check_lbl.text()
    import re
    ms = float(re.search(r"latency to about (\d+) ms", al.check_lbl.text()).group(1))
    assert ms == pytest.approx(80, abs=5)
    assert panel.plot_tabs.currentIndex() == 2  # the alignment plot shows the events
    assert _trial_json(proj)["alignment"]["method"] == "recorded"  # nothing was changed
    page.close()
