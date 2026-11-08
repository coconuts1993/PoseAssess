"""GUI tests for the local OpenPose BODY_25B setup: the "OpenPose folder" row of 1. Project and
the "Get OpenPose…" dialog (offscreen; fake install trees, no network, no nvidia-smi)."""
from __future__ import annotations

import dataclasses
import functools
import hashlib
import io
import os
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest

from poseassess.core import openpose_setup as ops
from tests.qtutil import pump, pump_until

VARIANTS = ("1_25BBkg", "1_25BSuperModel11FullVGG")
GPU_OK = {"name": "NVIDIA GeForce RTX 4090", "driver_version": "560.94", "compute_cap": "8.9"}
GPU_RTX5090 = {"name": "NVIDIA GeForce RTX 5090", "driver_version": "576.02",
               "compute_cap": "12.0"}
LONG = ("C:\\Users\\someone\\Documents\\Lab\\Motion capture\\openpose-1.7.0-binaries-win64-"
        "gpu-python3.7-flir-3d_recommended\\openpose")


# --------------------------------------------------------------------------- #
# fixtures / helpers
# --------------------------------------------------------------------------- #
@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    """Per-test settings folder, no real auto-detected install, a fine fake GPU."""
    monkeypatch.setenv("POSEASSESS_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.setattr(ops, "candidate_dirs", lambda: [])
    monkeypatch.setattr(ops, "gpu_info", lambda **kw: {**GPU_OK, "gpus": [GPU_OK]})


@pytest.fixture
def prototxts(monkeypatch) -> dict[str, bytes]:
    """{variant: prototxt bytes}; BODY25B_VARIANTS' SHA-256s patched to match them."""
    out = {}
    for v in VARIANTS:
        data = f'name: "{v}"\nlayer {{ name: "image" }}\n'.encode()
        out[v] = data
        monkeypatch.setitem(ops.BODY25B_VARIANTS, v, dataclasses.replace(
            ops.BODY25B_VARIANTS[v], prototxt_sha256=hashlib.sha256(data).hexdigest()))
    return out


def make_install(root: Path, *, prototxt: bytes | None = None,
                 caffemodel: str | None = ops.CAFFEMODEL_NAME) -> Path:
    """Fake OpenPose GPU install tree; returns the root."""
    (root / "bin").mkdir(parents=True, exist_ok=True)
    (root / "bin" / ops.EXE_NAME).write_bytes(b"MZ fake")
    (root / "bin" / "cudart64_110.dll").write_bytes(b"MZ cudart")
    mdir = root / ops.MODEL_SUBDIR
    mdir.mkdir(parents=True, exist_ok=True)
    if prototxt is not None:
        (mdir / ops.PROTOTXT_NAME).write_bytes(prototxt)
    if caffemodel:
        (mdir / caffemodel).write_bytes(b"weights" * 100)
    return root


def plain(label) -> str:
    """A status label's text without the zero-width spaces that let paths wrap."""
    return label.text().replace("\u200b", "")


def highlighted(page) -> bool:
    """The folder field has the amber "no usable OpenPose" frame."""
    return bool(page.openpose_edit.property("openposeMissing"))


@pytest.fixture
def page(qapp):
    from poseassess.gui.pages.project_page import ProjectPage
    from poseassess.gui.state import AppState

    p = ProjectPage(AppState())
    yield p
    p.shutdown()
    p.deleteLater()


def select_backend(page, name: str) -> None:
    """Programmatic change (e.g. loading a project)."""
    page.backend_combo.setCurrentIndex(page.backend_combo.findData(name))


def user_selects(page, name: str) -> None:
    """A choice in the combo box's popup list (opened, item made current, Enter): Qt's
    own path, currentIndexChanged then activated."""
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest

    combo = page.backend_combo
    combo.showPopup()
    view = combo.view()
    view.setCurrentIndex(combo.model().index(combo.findData(name), 0))
    QTest.keyClick(view, Qt.Key_Return)
    assert combo.currentData() == name and not view.isVisible()


def wait_report(page, text: str, timeout: float = 5.0) -> str:
    """Pump until the status line of the current folder text is shown; returns it after
    checking that it contains `text`."""
    assert pump_until(lambda: not page.openpose_check_pending(), timeout)
    line = plain(page.openpose_status)
    assert text in line, line
    return line


# --------------------------------------------------------------------------- #
# 1. Project: OpenPose folder row
# --------------------------------------------------------------------------- #
def test_row_follows_the_backend(page):
    assert page.backend_combo.currentData() != "openpose"
    assert page.openpose_row.isHidden()
    select_backend(page, "openpose")
    assert not page.openpose_row.isHidden()
    select_backend(page, "rtmpose")
    assert page.openpose_row.isHidden()
    assert "auto-detect" in page.openpose_edit.placeholderText()
    assert "C:\\openpose" in page.openpose_edit.placeholderText()


def test_no_checks_while_another_backend_is_selected(page, monkeypatch):
    calls = []
    monkeypatch.setattr(ops, "gpu_info", lambda **kw: calls.append(1))
    page.openpose_edit.setText("somewhere")
    pump(0.5)
    assert calls == [] and page.openpose_ok is None


def test_status_ok_tree_vs_missing_caffemodel(page, tmp_path, prototxts):
    from poseassess.gui.widgets.openpose_dialog import GOOD, WARN

    ok_root = make_install(tmp_path / "ok" / "openpose", prototxt=prototxts["1_25BBkg"])
    select_backend(page, "openpose")
    page.openpose_edit.setText(str(ok_root))
    line = wait_report(page, "OpenPose ready")
    assert str(ok_root) in line and "Option 2" in line
    assert page.openpose_ok is True
    assert GOOD in page.openpose_status.text()
    assert "\u200b" in page.openpose_status.text()   # long paths can wrap

    bad_root = make_install(tmp_path / "bad" / "openpose", prototxt=prototxts["1_25BBkg"],
                            caffemodel=None)
    page.openpose_edit.setText(str(bad_root))
    line = wait_report(page, "OpenPose not ready")
    assert "missing BODY_25B weights" in line and ops.CAFFEMODEL_NAME in line
    assert page.openpose_ok is False
    assert WARN in page.openpose_status.text() and GOOD not in page.openpose_status.text()

    # a misnamed weights file gets the rename hint
    (bad_root / ops.MODEL_SUBDIR / "pose_iter_464000.caffemodel").write_bytes(b"w" * 10)
    page.openpose_edit.setText(f'"{bad_root}"')   # Explorer's "Copy as path" quotes
    assert "rename" in wait_report(page, "pose_iter_464000.caffemodel")


def test_status_for_a_folder_without_openpose(page, tmp_path):
    from poseassess.gui.widgets.openpose_dialog import ENTER_FOLDER

    (tmp_path / "empty").mkdir()
    select_backend(page, "openpose")
    page.openpose_edit.setText(str(tmp_path / "empty"))
    line = wait_report(page, ENTER_FOLDER)
    assert f"{ops.EXE_NAME} not found in {tmp_path / 'empty'}" in line
    assert page.openpose_ok is False and page.openpose_missing and highlighted(page)


def test_autodetect_when_empty(page, tmp_path, monkeypatch, prototxts):
    found = make_install(tmp_path / "auto" / "openpose", prototxt=prototxts["1_25BBkg"])
    monkeypatch.setattr(ops, "candidate_dirs", lambda: [tmp_path / "nothing", found])
    select_backend(page, "openpose")
    assert page.openpose_edit.text() == ""
    line = wait_report(page, "auto-detected")
    assert f"OpenPose ready: {found}" in line and page.openpose_ok is True

    # a wrong folder falls back to the auto-detected install (that is what the run uses)
    page.openpose_edit.setText(str(tmp_path / "nothing"))
    line = wait_report(page, "instead")
    assert f"using {found} instead (auto-detected)" in line

    monkeypatch.setattr(ops, "candidate_dirs", lambda: [tmp_path / "nothing"])
    page.openpose_edit.setText("")
    line = wait_report(page, "Enter the OpenPose folder")
    assert str(tmp_path / "nothing") in line   # where it looked


def test_gpu_warnings_checked_once_off_the_gui_thread(page, monkeypatch):
    calls = []

    def fake_gpu_info(**kw):
        calls.append(threading.current_thread())
        return {**GPU_RTX5090, "gpus": [GPU_RTX5090]}

    monkeypatch.setattr(ops, "gpu_info", fake_gpu_info)
    select_backend(page, "openpose")
    # the GPU and the folder checks run in two threads: either may end first
    assert pump_until(lambda: "RTX 50 series" in plain(page.openpose_status))
    line = wait_report(page, "RTX 50 series")
    assert line.count("GPU: ") == 1
    select_backend(page, "rtmpose")
    select_backend(page, "openpose")
    page.openpose_edit.setText("x")
    wait_report(page, "not found in x")
    assert len(calls) == 1 and calls[0] is not threading.main_thread()
    assert "RTX 50 series" in plain(page.openpose_status)


def _record_file_checks(monkeypatch) -> list:
    """Records the thread of every find_root / check_install call."""
    seen = []
    for name in ("find_root", "check_install"):
        real = getattr(ops, name)

        def spy(*a, _real=real, _name=name, **k):
            seen.append((_name, threading.current_thread()))
            return _real(*a, **k)
        monkeypatch.setattr(ops, name, spy)
    return seen


def test_save_stores_and_remembers_the_folder(page, project, tmp_path, prototxts, monkeypatch):
    from poseassess.core.project import Project

    root = make_install(tmp_path / "op" / "openpose", prototxt=prototxts["1_25BBkg"])
    page.state.set_project(project)
    select_backend(page, "openpose")
    seen = _record_file_checks(monkeypatch)
    page.openpose_edit.setText(f"  {root}  ")
    page._save()                                  # while the folder check still runs
    cfg = Project.load(project.root).config
    assert cfg.openpose_dir == str(root) and cfg.pose2d_backend == "openpose"
    assert pump_until(lambda: ops.remembered_dir() == root)   # once the check said OK
    assert ops.user_settings_path().parent == tmp_path / "cfg"

    # checked already: remembered at once from that result
    ops.user_settings_path().unlink()
    assert ops.remembered_dir() is None and not page.openpose_check_pending()
    page._save()
    assert ops.remembered_dir() == root
    assert seen and all(t is not threading.main_thread() for _, t in seen)   # no file access

    # a folder that does not check OK is saved, but not remembered
    bad = make_install(tmp_path / "bad", caffemodel=None)
    page.openpose_edit.setText(str(bad))
    page._save()
    assert Project.load(project.root).config.openpose_dir == str(bad)
    line = wait_report(page, f"{bad} is not ready")   # the run uses the remembered one
    assert f"using {root} instead (remembered on this computer)" in line
    pump(0.05)
    assert ops.remembered_dir() == root

    # empty = auto-detect
    page.openpose_edit.setText("")
    page._save()
    assert Project.load(project.root).config.openpose_dir == ""
    assert all(t is not threading.main_thread() for _, t in seen)


def test_save_with_another_backend_remembers_after_a_check(page, project, tmp_path,
                                                           prototxts):
    """The row is hidden (no checks while typing): Save checks the folder in the background."""
    root = make_install(tmp_path / "op" / "openpose", prototxt=prototxts["1_25BBkg"])
    page.state.set_project(project)
    select_backend(page, "rtmpose")
    page.openpose_edit.setText(str(root))
    pump(0.4)
    assert page.openpose_ok is None              # nothing checked while hidden
    page._save()
    assert pump_until(lambda: ops.remembered_dir() == root)
    assert not highlighted(page)


def test_create_stores_the_folder(page, tmp_path, prototxts):
    from poseassess.core.project import Project

    root = make_install(tmp_path / "op" / "openpose", prototxt=prototxts["1_25BBkg"])
    page.path_edit.setText(str(tmp_path / "newproj"))
    select_backend(page, "openpose")
    page.openpose_edit.setText(str(root))
    page._create()
    assert Project.load(tmp_path / "newproj").config.openpose_dir == str(root)
    assert pump_until(lambda: ops.remembered_dir() == root)


def test_loading_a_project_fills_the_field(page, project):
    project.config.pose2d_backend = "openpose"
    project.config.openpose_dir = "D:\\tools\\openpose"
    project.save()
    page.state.set_project(project)
    assert page.openpose_edit.text() == "D:\\tools\\openpose"
    assert not page.openpose_row.isHidden()
    project.config.pose2d_backend = "rtmpose"
    project.config.openpose_dir = ""
    page.state.set_project(project)
    assert page.openpose_edit.text() == "" and page.openpose_row.isHidden()


def test_get_openpose_puts_the_chosen_root_into_the_field(page, tmp_path, monkeypatch):
    from poseassess.gui.widgets import openpose_dialog as dlg_mod

    root = make_install(tmp_path / "chosen" / "openpose")
    seen = {}

    def fake_exec(self):
        seen["start"] = self._root
        self._set_root(root, picked=True)
        return 0

    monkeypatch.setattr(dlg_mod.OpenPoseDialog, "exec", fake_exec)
    select_backend(page, "openpose")
    page._get_openpose()
    assert page.openpose_edit.text() == str(root) and seen["start"] is None


def test_get_openpose_starts_from_the_checked_folder(page, tmp_path, monkeypatch, prototxts):
    """The dialog gets the page's background check: opening it touches no drive on the GUI
    thread; a check still running when it opens is passed on when it ends."""
    from poseassess.gui.widgets import openpose_dialog as dlg_mod

    bad = make_install(tmp_path / "bad" / "openpose",
                       prototxt=prototxts["1_25BSuperModel11FullVGG"], caffemodel=None)
    seen = {}

    def fake_exec(self):
        seen["root"], seen["status"] = self._root, plain(self.status)
        seen["variant"] = self.variant_combo.currentData()
        pump_until(lambda: self._root is not None, 5.0)   # a check still running ends
        seen["later"] = self._root
        return 0

    monkeypatch.setattr(dlg_mod.OpenPoseDialog, "exec", fake_exec)
    select_backend(page, "openpose")
    page.openpose_edit.setText(str(bad))
    wait_report(page, "OpenPose not ready")
    checks = _record_file_checks(monkeypatch)
    page._get_openpose()
    assert seen["root"] == bad and "missing BODY_25B weights" in seen["status"]
    assert seen["variant"] == "1_25BSuperModel11FullVGG"
    # nothing checked on the GUI thread (the page re-checks in the background afterwards)
    assert all(t is not threading.main_thread() for _, t in checks)
    assert page._dialog is None

    # typed and clicked at once: the dialog opens empty and gets the result when it comes
    page.openpose_edit.setText(str(bad.parent))
    assert page.openpose_check_pending()
    page._get_openpose()
    assert seen["root"] is None and seen["later"] == bad
    assert all(t is not threading.main_thread() for _, t in checks)


def test_report_fix_hints(tmp_path, prototxts):
    from poseassess.gui.widgets.openpose_dialog import ENTER_FOLDER, openpose_report

    root = make_install(tmp_path / "op" / "openpose", prototxt=prototxts["1_25BBkg"],
                        caffemodel=None)
    text = " ".join(t for t, _ in openpose_report(str(root)).lines)
    assert "OpenPose not ready: missing BODY_25B weights" in text
    assert "Import BODY_25B model file" in text and text.endswith(ENTER_FOLDER)

    (root / ops.MODEL_SUBDIR / "pose_iter_464000.caffemodel").write_bytes(b"w" * 10)
    text = " ".join(t for t, _ in openpose_report(f'"{root}"').lines)
    assert "rename models" in text and "Fix: rename the file" in text

    (root / ops.MODEL_SUBDIR / "pose_iter_464000.caffemodel").rename(
        root / ops.MODEL_SUBDIR / ops.CAFFEMODEL_NAME)
    (root / "bin" / "cudart64_110.dll").unlink()
    report = openpose_report(str(root))
    text = " ".join(t for t, _ in report.lines)
    assert not report.ok and "CPU-only build" in text and "GPU build (Get OpenPose" in text

    (root / "bin" / "cudart64_110.dll").write_bytes(b"MZ")
    report = openpose_report(str(root))
    assert report.ok and report.root == root and ENTER_FOLDER not in str(report.lines)
    assert report.typed_root == root and report.typed_status.ok and report.setting == str(root)


def test_report_says_when_the_entered_folder_is_not_used(tmp_path, monkeypatch, prototxts):
    """The folder entered has OpenPoseDemo.exe but is incomplete, another install is ready:
    the run uses the other one, and the status line says so (in amber)."""
    from poseassess.gui.widgets.openpose_dialog import GOOD, WARN, openpose_report

    good = make_install(tmp_path / "C" / "openpose", prototxt=prototxts["1_25BBkg"])
    typed = make_install(tmp_path / "D" / "openpose", prototxt=prototxts["1_25BBkg"],
                         caffemodel=None)
    monkeypatch.setattr(ops, "candidate_dirs", lambda: [good])
    report = openpose_report(str(typed))
    assert report.ok and report.root == good
    assert report.typed_root == typed and not report.typed_status.ok
    (note, colour), (summary, colour2) = report.lines[:2]
    assert colour == WARN and note.startswith(f"{typed} is not ready (missing BODY_25B weights")
    assert note.endswith(f"; using {good} instead (auto-detected)")
    assert summary.startswith(f"OpenPose ready: {good}") and colour2 == GOOD

    # the parent folder of an incomplete install: same
    report = openpose_report(str(typed.parent))
    assert report.lines[0][0].startswith(f"{typed} is not ready")


def test_missing_install_highlights_the_field(page, tmp_path, prototxts, monkeypatch):
    from poseassess.gui.pages import project_page as pp
    from poseassess.gui.widgets.openpose_dialog import ENTER_FOLDER

    select_backend(page, "openpose")
    line = wait_report(page, ENTER_FOLDER)          # empty field, nothing auto-detected
    assert highlighted(page) and line.index(ENTER_FOLDER) < line.index("not found")
    assert page.openpose_edit.placeholderText() == "e.g. C:\\openpose (none found automatically)"

    # an incomplete install is no usable install either
    bad = make_install(tmp_path / "bad" / "openpose", prototxt=prototxts["1_25BBkg"],
                       caffemodel=None)
    page.openpose_edit.setText(str(bad))
    line = wait_report(page, "OpenPose not ready")
    assert ENTER_FOLDER in line and "Import BODY_25B model file" in line
    assert highlighted(page)

    # a valid folder: the highlight goes as soon as it is checked
    good = make_install(tmp_path / "good" / "openpose", prototxt=prototxts["1_25BBkg"])
    page.openpose_edit.setText(str(good))
    assert highlighted(page)                         # still the old result while typing
    line = wait_report(page, "OpenPose ready")
    assert not highlighted(page) and ENTER_FOLDER not in line
    assert "auto-detect" in page.openpose_edit.placeholderText()

    # back to a missing one, then another backend: no highlight on a hidden row
    page.openpose_edit.setText(str(tmp_path / "nowhere"))
    wait_report(page, ENTER_FOLDER)
    assert highlighted(page)
    select_backend(page, "rtmpose")
    assert not highlighted(page) and not page.openpose_missing

    # Browse…: checked at once (no typing delay); the parent folder is accepted too
    select_backend(page, "openpose")
    wait_report(page, ENTER_FOLDER)
    monkeypatch.setattr(pp.QFileDialog, "getExistingDirectory",
                        staticmethod(lambda *a, **k: str(tmp_path / "good")))
    page._browse_openpose()
    assert page.openpose_edit.text() == str(good)
    assert not page._check_timer.isActive()
    wait_report(page, "OpenPose ready")
    assert not highlighted(page)


def test_highlight_is_drawn(page, tmp_path):
    """The dynamic property really switches the amber frame of the field."""
    from PySide6.QtGui import QColor

    from poseassess.gui.widgets.openpose_dialog import WARN

    def edge_colour() -> QColor:
        page.openpose_edit.resize(200, 26)
        img = page.openpose_edit.grab().toImage()
        return img.pixelColor(0, img.height() // 2)

    def near(a: QColor, b: QColor) -> bool:
        return max(abs(a.red() - b.red()), abs(a.green() - b.green()),
                   abs(a.blue() - b.blue())) <= 24

    page._set_missing_highlight(False)
    assert not near(edge_colour(), QColor(WARN))
    page._set_missing_highlight(True)
    assert near(edge_colour(), QColor(WARN))
    page._set_missing_highlight(False)
    assert not near(edge_colour(), QColor(WARN))


def test_editing_finished_checks_at_once(page, tmp_path, prototxts):
    good = make_install(tmp_path / "op" / "openpose", prototxt=prototxts["1_25BBkg"])
    select_backend(page, "openpose")
    wait_report(page, "Enter the OpenPose folder")
    page.openpose_edit.setText(str(good))
    assert page._check_timer.isActive()              # debounced while typing
    page.openpose_edit.editingFinished.emit()        # Enter / focus leaves the field
    assert not page._check_timer.isActive()
    wait_report(page, "OpenPose ready", timeout=0.25)   # < CHECK_DELAY_MS + margin
    assert not highlighted(page)


def test_check_runs_off_the_gui_thread(page, monkeypatch):
    """A slow drive (e.g. a disconnected D:) must not freeze the window."""
    seen = []
    real = ops.resolve_openpose_dir

    def slow(setting):
        seen.append(threading.current_thread())
        time.sleep(0.6)
        return real(setting)

    monkeypatch.setattr(ops, "resolve_openpose_dir", slow)
    t0 = time.perf_counter()
    select_backend(page, "openpose")
    page.openpose_edit.setText("C:\\somewhere")
    page.openpose_edit.editingFinished.emit()
    assert time.perf_counter() - t0 < 0.3
    assert page.openpose_check_pending()
    wait_report(page, "Enter the OpenPose folder")
    assert seen and all(t is not threading.main_thread() for t in seen)


@pytest.fixture
def shown_page(page, project):
    """The page with a project open (the settings form is enabled), shown and active."""
    page.state.set_project(project)
    page.show()
    page.activateWindow()
    pump(0.05)
    page.backend_combo.setFocus()
    pump(0.02)
    assert page.window().focusWidget() is page.backend_combo
    return page


def test_choosing_openpose_without_install_focuses_the_field(shown_page):
    page = shown_page
    user_selects(page, "openpose")
    wait_report(page, "Enter the OpenPose folder")
    assert pump_until(lambda: page.window().focusWidget() is page.openpose_edit, 2.0)
    assert highlighted(page)

    # choosing it again (already selected) focuses at once
    page.backend_combo.setFocus()
    user_selects(page, "openpose")
    assert page.window().focusWidget() is page.openpose_edit


def test_arrow_keys_and_wheel_on_the_closed_box_keep_the_focus(shown_page):
    """Qt emits `activated` for arrow keys and wheel turns on the closed combo box too: the
    focus must stay in the box (Up must still go back), only a choice in the list moves it."""
    from PySide6.QtCore import QPoint, QPointF, Qt
    from PySide6.QtGui import QWheelEvent
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication

    page = shown_page
    combo = page.backend_combo
    before = combo.count() - 2
    assert combo.itemData(combo.count() - 1) == "openpose"    # the last item
    combo.setCurrentIndex(before)
    wait_report(page, "")
    QTest.keyClick(combo, Qt.Key_Down)
    assert combo.currentData() == "openpose"
    wait_report(page, "Enter the OpenPose folder")
    pump(0.1)
    assert page.window().focusWidget() is combo and highlighted(page)
    QTest.keyClick(page.window().focusWidget(), Qt.Key_Up)    # back again
    assert combo.currentIndex() == before
    QTest.keyClick(combo, Qt.Key_Down)
    wait_report(page, "Enter the OpenPose folder")

    combo.setCurrentIndex(before)
    pump(0.05)
    wheel = QWheelEvent(QPointF(5, 5), QPointF(combo.mapToGlobal(QPoint(5, 5))), QPoint(0, 0),
                        QPoint(0, -120), Qt.NoButton, Qt.NoModifier, Qt.NoScrollPhase, False)
    QApplication.sendEvent(combo, wheel)
    assert combo.currentData() == "openpose"
    wait_report(page, "Enter the OpenPose folder")
    pump(0.1)
    assert page.window().focusWidget() is combo

    # the popup closed with Esc, then an arrow key: still no focus change
    combo.setCurrentIndex(before)
    combo.showPopup()
    QTest.keyClick(combo.view(), Qt.Key_Escape)
    QTest.keyClick(combo, Qt.Key_Down)
    wait_report(page, "Enter the OpenPose folder")
    pump(0.1)
    assert combo.currentData() == "openpose" and page.window().focusWidget() is combo


def test_choosing_openpose_with_an_install_keeps_the_focus(shown_page, tmp_path, monkeypatch,
                                                           prototxts):
    found = make_install(tmp_path / "auto" / "openpose", prototxt=prototxts["1_25BBkg"])
    monkeypatch.setattr(ops, "candidate_dirs", lambda: [found])
    page = shown_page
    user_selects(page, "openpose")
    wait_report(page, "OpenPose ready")
    pump(0.05)
    assert page.window().focusWidget() is page.backend_combo and not highlighted(page)


def test_focus_only_for_a_user_choice_and_never_stolen(shown_page, project):
    page = shown_page
    # loading a project with OpenPose (programmatic): highlighted, focus stays put
    project.config.pose2d_backend = "openpose"
    project.save()
    page.state.set_project(project)
    wait_report(page, "Enter the OpenPose folder")
    pump(0.05)
    assert highlighted(page) and page.window().focusWidget() is page.backend_combo

    # the user chose OpenPose but moved on before the check ended: no focus steal
    user_selects(page, "rtmpose")
    user_selects(page, "openpose")
    page.name_edit.setFocus()
    wait_report(page, "Enter the OpenPose folder")
    pump(0.05)
    assert page.window().focusWidget() is page.name_edit

    # chose OpenPose, then another backend before the result: nothing happens
    page.backend_combo.setFocus()
    user_selects(page, "rtmpose")
    user_selects(page, "openpose")
    user_selects(page, "rtmpose")
    pump(0.3)
    assert page.window().focusWidget() is page.backend_combo


def test_window_min_width_unchanged(qapp, tmp_path, monkeypatch, prototxts):
    """Like .github/portable/gui_sizes.py (offscreen): the OpenPose row (highlighted, a long
    path, the fix hint and a GPU warning in its status) must not make the 1. Project page or
    the main window wider than before the change."""
    from poseassess.gui.main_window import MainWindow

    monkeypatch.setattr(ops, "gpu_info", lambda **kw: {**GPU_RTX5090, "gpus": [GPU_RTX5090]})
    w = MainWindow()
    try:
        page = w.pages[0]
        w.show()
        pump(0.05)
        assert page.openpose_row.isHidden()          # default backend
        hidden_row = page.minimumSizeHint().width(), w.minimumSizeHint().width()

        # the page before the change: the same form without the OpenPose row
        form = page._form
        index, _role = form.getWidgetPosition(page.openpose_row)
        taken = form.takeRow(page.openpose_row)
        label, field = taken.labelItem.widget(), taken.fieldItem.widget()
        label.hide()
        field.hide()
        pump(0.05)
        page_before, win_before = page.minimumSizeHint().width(), w.minimumSizeHint().width()
        assert hidden_row == (page_before, win_before)
        form.insertRow(index, label, field)
        assert form.getWidgetPosition(page.openpose_row)[0] == index

        long_root = tmp_path / ("very long folder name " * 6).strip() / "openpose"
        make_install(long_root, prototxt=prototxts["1_25BBkg"], caffemodel=None)
        select_backend(page, "openpose")
        page.openpose_edit.setText(str(long_root))
        line = wait_report(page, "OpenPose not ready")
        assert "RTX 50 series" in line and "Import BODY_25B" in line and highlighted(page)
        pump(0.05)
        assert page.minimumSizeHint().width() <= page_before
        assert w.minimumSizeHint().width() <= win_before

        page.openpose_edit.setText(LONG)              # not found at all
        wait_report(page, "Enter the OpenPose folder")
        pump(0.05)
        assert page.minimumSizeHint().width() <= page_before
        assert w.minimumSizeHint().width() <= win_before
    finally:
        w.close()                                     # shuts the pages down
        w.deleteLater()


# --------------------------------------------------------------------------- #
# Get OpenPose… dialog
# --------------------------------------------------------------------------- #
@pytest.fixture
def dialog_factory(qapp):
    from poseassess.gui.widgets.openpose_dialog import OpenPoseDialog

    made = []

    def make(root=None):
        d = OpenPoseDialog(None, root)
        made.append(d)
        return d

    yield make
    for d in made:
        if d.busy():
            d._job.cancel()
            assert pump_until(lambda d=d: not d.busy(), 5.0)
        d.deleteLater()


@pytest.fixture
def no_message_boxes(monkeypatch):
    """Records QMessageBox calls of the dialog instead of showing them."""
    from poseassess.gui.widgets import openpose_dialog as dlg_mod

    calls = []

    class Box:
        Yes, No = dlg_mod.QMessageBox.Yes, dlg_mod.QMessageBox.No
        answer = dlg_mod.QMessageBox.Yes

        @staticmethod
        def warning(parent, title, text, *a):
            calls.append(("warning", title, text))

        @staticmethod
        def information(parent, title, text, *a):
            calls.append(("information", title, text))

        @classmethod
        def question(cls, parent, title, text, *a):
            calls.append(("question", title, text))
            return cls.answer

    monkeypatch.setattr(dlg_mod, "QMessageBox", Box)
    return calls


def test_download_needs_the_license_checkbox(dialog_factory, tmp_path):
    from poseassess.gui.widgets.openpose_dialog import DOWNLOAD_MB

    d = dialog_factory()
    assert DOWNLOAD_MB == 418 and "418 MB" in d.b_download.text()
    assert not d.b_download.isEnabled()
    assert not d.b_import.isEnabled()           # no install known yet
    assert d.b_existing.isEnabled()
    d.accept_box.setChecked(True)
    assert d.b_download.isEnabled()
    d.accept_box.setChecked(False)
    assert not d.b_download.isEnabled()
    d._download()                               # ignored without the checkbox
    assert not d.busy()
    assert d.result_root() is None
    assert d.variant_combo.currentData() == ops.DEFAULT_VARIANT   # Option 2


def test_use_existing_folder(dialog_factory, tmp_path, monkeypatch, no_message_boxes,
                             prototxts):
    from poseassess.gui.widgets import openpose_dialog as dlg_mod

    root = make_install(tmp_path / "op" / "openpose",
                        prototxt=prototxts["1_25BSuperModel11FullVGG"])
    picks = iter([str(tmp_path / "op"), str(tmp_path)])  # the parent folder is accepted too
    monkeypatch.setattr(dlg_mod.QFileDialog, "getExistingDirectory",
                        staticmethod(lambda *a, **k: next(picks)))
    d = dialog_factory()
    d.b_existing.click()
    assert d.result_root() == root
    assert "OpenPose ready" in plain(d.status)
    assert d.variant_combo.currentData() == "1_25BSuperModel11FullVGG"  # the installed one
    assert d.b_import.isEnabled()
    d.b_existing.click()                        # tmp_path: no OpenPose there
    assert no_message_boxes and no_message_boxes[-1][0] == "warning"
    assert d.result_root() == root


def _fake_opener(prototxts):
    urls = {ops.BODY25B_VARIANTS[v].prototxt_url: data for v, data in prototxts.items()}

    def opener(url, timeout=None):
        return io.BytesIO(urls[url])
    return opener


def test_import_model_runs_in_a_worker(dialog_factory, tmp_path, monkeypatch, prototxts,
                                       no_message_boxes):
    from poseassess.gui.widgets import openpose_dialog as dlg_mod

    root = make_install(tmp_path / "openpose", caffemodel=None)
    src = tmp_path / "downloads" / "pose_iter_464000.caffemodel"
    src.parent.mkdir()
    src.write_bytes(b"real weights" * 1000)
    monkeypatch.setattr(dlg_mod.QFileDialog, "getOpenFileName",
                        staticmethod(lambda *a, **k: (str(src), "")))
    calls = []
    real = functools.partial(ops.install_caffemodel, opener=_fake_opener(prototxts))
    started = threading.Event()

    def install(root_, src_, *, variant=None):
        calls.append((threading.current_thread(), Path(root_), Path(src_), variant))
        started.wait(5)
        return real(root_, src_, variant=variant)

    monkeypatch.setattr(ops, "install_caffemodel", install)
    d = dialog_factory(root)
    assert "OpenPose not ready" in plain(d.status) and "Next: import" in plain(d.status)
    d.variant_combo.setCurrentIndex(d.variant_combo.findData("1_25BSuperModel11FullVGG"))
    d.b_import.click()
    assert d.busy()
    assert not d.b_import.isEnabled() and not d.b_existing.isEnabled()
    assert not d.b_cancel.isVisible()
    d.reject()                                   # cannot close while copying
    assert no_message_boxes[-1][0] == "information"
    started.set()
    assert pump_until(lambda: not d.busy(), 5.0)
    th, got_root, got_src, variant = calls[0]
    assert th is not threading.main_thread()
    assert (got_root, got_src, variant) == (root, src, "1_25BSuperModel11FullVGG")
    status = plain(d.status)
    assert "BODY_25B model file installed" in status
    assert "OpenPose ready" in status and "Option 1" in status
    assert (root / ops.MODEL_SUBDIR / ops.CAFFEMODEL_NAME).read_bytes() == src.read_bytes()
    assert d.result_root() == root and d.b_import.isEnabled()
    assert [c for c in no_message_boxes if c[0] == "warning"] == []


def test_import_model_error_is_shown(dialog_factory, tmp_path, monkeypatch, no_message_boxes):
    from poseassess.gui.widgets import openpose_dialog as dlg_mod

    root = make_install(tmp_path / "openpose", caffemodel=None)
    monkeypatch.setattr(dlg_mod.QFileDialog, "getOpenFileName",
                        staticmethod(lambda *a, **k: (str(tmp_path / "w.caffemodel"), "")))

    def install(root_, src_, *, variant=None):
        raise RuntimeError("could not download pose_deploy.prototxt: offline")

    monkeypatch.setattr(ops, "install_caffemodel", install)
    d = dialog_factory(root)
    d.b_import.click()
    assert pump_until(lambda: not d.busy(), 5.0)
    assert no_message_boxes == [("warning", "Import BODY_25B model",
                                 "could not download pose_deploy.prototxt: offline")]
    assert "OpenPose not ready" in plain(d.status)
    assert d.result_root() is None


def _fake_download(tmp_path, *, block: bool, record: dict):
    """download_openpose stand-in: reports progress; `block` waits for cancel."""
    def download(dest_parent, *, progress=None, cancelled=None, **kw):
        record["thread"] = threading.current_thread()
        record["dest_parent"] = Path(dest_parent)
        total = ops.OFFICIAL_ZIP_SIZE
        for i in range(1, 6):
            progress(i * total // 10, total)
            time.sleep(0.12)                     # > the 0.1 s progress throttle
        end = time.monotonic() + 10
        while block and time.monotonic() < end:
            if cancelled():
                record["cancelled"] = True
                raise RuntimeError("OpenPose download cancelled")
            time.sleep(0.01)
        progress(total, total)
        return make_install(Path(dest_parent) / "openpose")
    return download


def test_download_with_progress(dialog_factory, tmp_path, monkeypatch, no_message_boxes):
    from poseassess.gui.widgets import openpose_dialog as dlg_mod

    record: dict = {}
    monkeypatch.setattr(ops, "download_openpose", _fake_download(tmp_path, block=False,
                                                                 record=record))
    monkeypatch.setattr(dlg_mod.QFileDialog, "getExistingDirectory",
                        staticmethod(lambda *a, **k: str(tmp_path / "dl")))
    d = dialog_factory()
    values = []
    d.progress.valueChanged.connect(values.append)
    d.accept_box.setChecked(True)
    d.b_download.click()
    assert d.busy() and not d.b_download.isEnabled() and not d.accept_box.isEnabled()
    assert not d.b_cancel.isHidden()
    assert pump_until(lambda: not d.busy(), 10.0)
    assert record["thread"] is not threading.main_thread()
    assert record["dest_parent"] == tmp_path / "dl"
    assert len([v for v in values if v > 0]) >= 3 and max(values) <= 1000
    root = tmp_path / "dl" / "openpose"
    assert d.result_root() == root
    status = plain(d.status)
    assert f"downloaded to {root}" in status and "Next: import" in status
    assert d.b_import.isEnabled() and d.progress.isHidden() and d.b_cancel.isHidden()
    assert no_message_boxes == []


def test_download_existing_install_is_offered(dialog_factory, tmp_path, monkeypatch,
                                              no_message_boxes):
    from poseassess.gui.widgets import openpose_dialog as dlg_mod

    root = make_install(tmp_path / "openpose")
    monkeypatch.setattr(ops, "download_openpose",
                        lambda *a, **k: pytest.fail("must not download"))
    monkeypatch.setattr(dlg_mod.QFileDialog, "getExistingDirectory",
                        staticmethod(lambda *a, **k: str(tmp_path)))
    d = dialog_factory()
    d.accept_box.setChecked(True)
    d.b_download.click()
    assert no_message_boxes[-1][0] == "question"
    assert not d.busy() and d.result_root() == root


def test_download_cancel(dialog_factory, tmp_path, monkeypatch, no_message_boxes):
    from poseassess.gui.widgets import openpose_dialog as dlg_mod

    record: dict = {}
    monkeypatch.setattr(ops, "download_openpose", _fake_download(tmp_path, block=True,
                                                                 record=record))
    monkeypatch.setattr(dlg_mod.QFileDialog, "getExistingDirectory",
                        staticmethod(lambda *a, **k: str(tmp_path / "dl")))
    d = dialog_factory()
    d.accept_box.setChecked(True)
    d.b_download.click()
    assert pump_until(lambda: d.progress.value() > 0, 5.0)
    d.b_cancel.click()
    assert not d.b_cancel.isEnabled()
    assert pump_until(lambda: not d.busy(), 5.0)
    assert record.get("cancelled")
    assert "Download cancelled" in plain(d.status)
    assert no_message_boxes == []               # a cancel is no error
    assert d.result_root() is None and d.b_download.isEnabled()


def test_closing_the_dialog_cancels_the_download(dialog_factory, tmp_path, monkeypatch,
                                                 no_message_boxes):
    from poseassess.gui.widgets import openpose_dialog as dlg_mod

    record: dict = {}
    monkeypatch.setattr(ops, "download_openpose", _fake_download(tmp_path, block=True,
                                                                 record=record))
    monkeypatch.setattr(dlg_mod.QFileDialog, "getExistingDirectory",
                        staticmethod(lambda *a, **k: str(tmp_path / "dl")))
    d = dialog_factory()
    d.show()
    d.accept_box.setChecked(True)
    d.b_download.click()
    assert pump_until(lambda: d.progress.value() > 0, 5.0)

    no_message_boxes.clear()
    box = dlg_mod.QMessageBox                    # the recording stand-in
    box.answer = box.No
    d.reject()                                   # "Stop the download?" No: keeps running
    assert d.isVisible() and d.busy() and not record.get("cancelled")
    box.answer = box.Yes
    d.reject()
    assert not d.isVisible() and not d.busy()
    thread, _job = dlg_mod._ORPHANS[-1]
    assert pump_until(lambda: record.get("cancelled") and thread.isFinished(), 5.0)
    pump(0.1)
    assert [c[0] for c in no_message_boxes] == ["question", "question"]  # no error box
    dlg_mod.stop_jobs(1000)                      # app shutdown: nothing left to wait for


def test_download_shows_unpacking(dialog_factory, tmp_path, monkeypatch, no_message_boxes):
    from poseassess.gui.widgets import openpose_dialog as dlg_mod

    release = threading.Event()

    def download(dest_parent, *, progress=None, cancelled=None, **kw):
        total = ops.OFFICIAL_ZIP_SIZE
        progress(total // 2, total)
        time.sleep(0.12)
        progress(total, total)                   # all bytes in: unpacking starts
        release.wait(5)
        if cancelled():
            raise RuntimeError("OpenPose download cancelled")
        return make_install(Path(dest_parent) / "openpose")

    monkeypatch.setattr(ops, "download_openpose", download)
    monkeypatch.setattr(dlg_mod.QFileDialog, "getExistingDirectory",
                        staticmethod(lambda *a, **k: str(tmp_path / "dl")))
    d = dialog_factory()
    d.accept_box.setChecked(True)
    d.b_download.click()
    assert pump_until(lambda: "Unpacking…" in plain(d.status), 5.0)
    assert d.progress.maximum() == 0 and not d.progress.isHidden()   # busy bar (no text)
    assert d.b_cancel.isEnabled() and not d.b_cancel.isHidden()   # still cancellable
    assert not d.b_download.isEnabled() and not d.b_existing.isEnabled()
    release.set()
    assert pump_until(lambda: not d.busy(), 5.0)
    assert d.result_root() == tmp_path / "dl" / "openpose"
    assert "Unpacking" not in plain(d.status)


@pytest.mark.parametrize("variant", VARIANTS)
def test_dialog_preselects_the_installed_variant(dialog_factory, tmp_path, prototxts, variant):
    root = make_install(tmp_path / "openpose", prototxt=prototxts[variant])
    d = dialog_factory(root / "bin")             # any path find_root accepts
    assert d.variant_combo.currentData() == variant
    assert d.b_import.isEnabled() and d.result_root() is None   # nothing chosen yet
    assert "OpenPose ready" in plain(d.status)
    short = ops.BODY25B_VARIANTS[variant].short
    assert short in plain(d.status) and short in d.variant_combo.currentText()


def test_dialog_license_summary_and_link(dialog_factory):
    from PySide6.QtWidgets import QLabel

    d = dialog_factory()
    intro = d.findChildren(QLabel)[0]
    assert "noncommercial research only" in intro.text() and ops.LICENSE_URL in intro.text()
    assert intro.openExternalLinks()
    assert "license" in d.accept_box.text() and not d.accept_box.isChecked()
    assert "No OpenPose folder chosen yet" in plain(d.status)


def test_closing_while_the_download_ends_in_the_question(dialog_factory, tmp_path, monkeypatch,
                                                         no_message_boxes):
    """'Stop the OpenPose download?' runs an event loop: the download (here: the unpacking)
    may end before the user answers Yes. The dialog then just closes, with the result."""
    from poseassess.gui.widgets import openpose_dialog as dlg_mod

    release = threading.Event()

    def download(dest_parent, *, progress=None, cancelled=None, **kw):
        progress(ops.OFFICIAL_ZIP_SIZE, ops.OFFICIAL_ZIP_SIZE)   # "Unpacking…"
        release.wait(5)
        return make_install(Path(dest_parent) / "openpose")

    monkeypatch.setattr(ops, "download_openpose", download)
    monkeypatch.setattr(dlg_mod.QFileDialog, "getExistingDirectory",
                        staticmethod(lambda *a, **k: str(tmp_path / "dl")))
    box = dlg_mod.QMessageBox

    def question(parent, title, text, *a):
        no_message_boxes.append(("question", title, text))
        release.set()                            # unpacking ends while the box is open
        assert pump_until(lambda: not d.busy(), 5.0)
        return box.Yes

    monkeypatch.setattr(box, "question", staticmethod(question))
    orphans = len(dlg_mod._ORPHANS)
    d = dialog_factory()
    d.show()
    d.accept_box.setChecked(True)
    d.b_download.click()
    assert pump_until(lambda: "Unpacking…" in plain(d.status), 5.0)
    d.reject()
    assert not d.isVisible() and not d.busy()
    assert d.result_root() == tmp_path / "dl" / "openpose"
    assert len(dlg_mod._ORPHANS) == orphans      # nothing left running
    assert [c[0] for c in no_message_boxes] == ["question"]


def test_shutdown_does_not_wait_for_a_stopping_download(dialog_factory, tmp_path, monkeypatch,
                                                         no_message_boxes):
    """App close while a cancelled download still cleans up: the window does not freeze."""
    from poseassess.gui.widgets import openpose_dialog as dlg_mod

    release = threading.Event()

    def download(dest_parent, *, progress=None, cancelled=None, **kw):
        progress(1, 10)
        release.wait(5)                          # e.g. a stalled read
        raise RuntimeError("OpenPose download cancelled")

    monkeypatch.setattr(ops, "download_openpose", download)
    monkeypatch.setattr(dlg_mod.QFileDialog, "getExistingDirectory",
                        staticmethod(lambda *a, **k: str(tmp_path / "dl")))
    d = dialog_factory()
    d.accept_box.setChecked(True)
    d.b_download.click()
    pump(0.05)
    d.reject()
    thread, job = dlg_mod._ORPHANS[-1]
    assert dlg_mod.jobs_running() and job.cancelled()
    t0 = time.perf_counter()
    dlg_mod.stop_jobs()
    assert time.perf_counter() - t0 < 0.5 and thread.isRunning()
    release.set()
    assert pump_until(lambda: not dlg_mod.jobs_running(), 5.0)
    assert thread not in [t for t, _ in dlg_mod._ORPHANS]


EXIT_SCRIPT = """
import os, sys, threading, time
from pathlib import Path
os.environ["QT_QPA_PLATFORM"] = "offscreen"
from PySide6.QtWidgets import QApplication
from poseassess.core import openpose_setup as ops
from poseassess.gui.widgets import openpose_dialog as dm

marker = Path(sys.argv[1])

def slow_download(parent, *, progress=None, cancelled=None, **kw):
    progress(1, 10)
    try:
        while not cancelled():
            time.sleep(0.01)
        time.sleep(1.5)          # e.g. one big file of the archive still being unpacked
        raise RuntimeError("OpenPose download cancelled")
    finally:
        marker.write_text("partial files removed")

class Box:
    Yes, No = dm.QMessageBox.Yes, dm.QMessageBox.No
    @staticmethod
    def question(*a, **k):
        return Box.Yes

def main():
    app = QApplication([])
    ops.download_openpose = slow_download
    dm.QMessageBox = Box
    dm.QFileDialog.getExistingDirectory = staticmethod(lambda *a, **k: str(marker.parent))
    d = dm.OpenPoseDialog(None)
    d.accept_box.setChecked(True)
    d.b_download.click()
    end = time.time() + 0.3
    while time.time() < end:
        app.processEvents()
        time.sleep(0.01)
    d.reject()                   # cancels; the download is still stopping
    dm.stop_jobs()               # ProjectPage.shutdown()
    print("running at exit:", dm._ORPHANS[-1][0].isRunning(), flush=True)
    return 0

sys.exit(main())
"""


def test_exit_waits_for_a_stopping_download(tmp_path):
    """Python exit with a cancelled download still stopping: it may finish (its cleanup
    runs) and the process ends normally; Qt aborts when a running QThread is deleted."""
    marker = tmp_path / "cleanup.txt"
    root = Path(__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONPATH": str(root),
           "POSEASSESS_CONFIG_DIR": str(tmp_path / "cfg"), "QT_QPA_PLATFORM": "offscreen"}
    proc = subprocess.run([sys.executable, "-c", textwrap.dedent(EXIT_SCRIPT), str(marker)],
                          cwd=tmp_path, env=env, capture_output=True, text=True, timeout=60)
    assert "running at exit: True" in proc.stdout, proc.stdout + proc.stderr
    assert proc.returncode == 0, proc.stderr
    assert "Destroyed while thread" not in proc.stderr
    assert "Stopping the cancelled OpenPose download" in proc.stderr
    assert marker.read_text() == "partial files removed"


# --------------------------------------------------------------------------- #
# network resolution (OpenPose row)
# --------------------------------------------------------------------------- #
def test_resolution_choice_default_save_load_and_create(page, project, tmp_path):
    from poseassess.core.project import Project

    combo = page.openpose_res_combo
    assert combo.currentData() == ops.NET_RESOLUTION == "-1x368"
    assert [combo.itemData(i) for i in range(combo.count())] == [v for v, _ in ops.NET_RESOLUTIONS]
    assert combo.parent() is not None and page.openpose_row.isAncestorOf(combo)  # hidden with it

    page.state.set_project(project)
    select_backend(page, "openpose")
    combo.setCurrentIndex(combo.findData("-1x480"))
    page._save()
    assert Project.load(project.root).config.openpose_net_resolution == "-1x480"

    # loading: the project's value; "" (older project) -> the default
    project.config.openpose_net_resolution = ""
    page.state.set_project(project)
    assert combo.currentData() == "-1x368"
    project.config.openpose_net_resolution = "-1x480"
    page.state.set_project(project)
    assert combo.currentData() == "-1x480"

    # a usable value that is not a preset is kept as its own item; an unusable one -> default
    n = combo.count()
    project.config.openpose_net_resolution = "-1x320"
    page.state.set_project(project)
    assert combo.currentData() == "-1x320" and combo.count() == n + 1
    assert "320" in combo.currentText()
    page._save()
    assert Project.load(project.root).config.openpose_net_resolution == "-1x320"
    project.config.openpose_net_resolution = "-1x333"
    page.state.set_project(project)
    assert combo.currentData() == "-1x368"

    # a new project gets the choice too
    combo.setCurrentIndex(combo.findData("-1x256"))
    page.path_edit.setText(str(tmp_path / "newproj"))
    page._create()
    assert Project.load(tmp_path / "newproj").config.openpose_net_resolution == "-1x256"
