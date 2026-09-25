"""GUI-VIEW: BalanceSkeleton3DViewer overlays on demo_trial (board corners at viewer._tf(zup_to_yup(corners)), COP/force/COM follow the frame via index_for_time, toggles, clear), Viz3DPage wiring and reload on balance_changed. Rendering screenshots are @pytest.mark.gl."""

import shutil

import numpy as np
import pytest

from tests.qtutil import pump


# ------------------------------------------------------------------ helpers
def _viewer_with_demo(demo, fused=None):
    from poseassess.core.balance.board import load_board
    from poseassess.gui.widgets.balance_3d import BalanceSkeleton3DViewer
    from tests.synth import fused_from_demo

    v = BalanceSkeleton3DViewer()
    assert v.load_trc(demo["trc"])
    fused = fused if fused is not None else fused_from_demo(demo)
    board = load_board(demo["project"])
    v.set_balance(fused, board)
    return v, fused, board


def _disp(v, p_world):
    from poseassess.core.balance.trc import zup_to_yup
    return v._tf(zup_to_yup(np.asarray(p_world, float).reshape(-1, 3)))


def _disp_dir(v, d_world):
    from poseassess.core.balance.trc import zup_to_yup
    return zup_to_yup(np.asarray(d_world, float)) @ v._M.T


def _row_for(v, fused, frame):
    return fused.index_for_time(float(v._trc.times[frame]))


def _frame_where(v, fused, cond):
    for i in range(v._trc.n_frames):
        r = _row_for(v, fused, i)
        if r >= 0 and cond(r):
            return i, r
    raise AssertionError("no such frame")


def _mesh_vertexes(item):
    return np.asarray(item.opts["meshdata"].vertexes())


# ------------------------------------------------------------------ viewer
def test_viewer_without_data_is_the_original(qapp, demo_trial):
    from poseassess.gui.widgets.balance_3d import BalanceSkeleton3DViewer
    from poseassess.gui.widgets.skeleton_3d import Skeleton3DViewer

    v = BalanceSkeleton3DViewer()
    base = Skeleton3DViewer()
    assert v.load_trc(demo_trial["trc"]) and base.load_trc(demo_trial["trc"])
    # no balance GL item was added (only the world axes), the toggles and the read-out hidden
    assert not v._bal_items
    axes = [i for i in v._axes_items.values() if i.visible()]
    assert len(v.view.items) == len(base.view.items) + len(v._axes_items)
    assert axes and all(v._axes_items[k] in axes for k in ("world_axes",))
    assert not v._axes_items["board_axes"].visible()
    assert not any(c.isVisibleTo(v) for c in v._toggles)
    assert not v.balance_info.isVisibleTo(v)
    # the toggles live in the transport row, after "Cameras"
    row = v._transport_row()
    assert row.indexOf(v.cam_check) >= 0
    assert all(row.indexOf(c) > row.indexOf(v.cam_check) for c in v._toggles)
    np.testing.assert_allclose(v._M, base._M)
    np.testing.assert_allclose(v._center, base._center)
    # set_balance(None, None) keeps it that way
    v.set_balance(None, None)
    assert not v._bal_items and not v.has_balance()


def test_board_items_in_the_skeleton_frame(qapp, demo_trial):
    v, fused, board = _viewer_with_demo(demo_trial)
    it = v._bal_items
    corners = _disp(v, board.corners_world)
    np.testing.assert_allclose(it["outline"].pos, np.vstack([corners, corners[:1]]), atol=1e-9)
    np.testing.assert_allclose(it["front"].pos, corners[:2], atol=1e-9)  # TL-TR = front edge
    np.testing.assert_allclose(it["sensors"].pos, _disp(v, board.sensors_world), atol=1e-9)
    surf = _mesh_vertexes(it["surface"])
    up_d = _disp_dir(v, board.up_world)
    np.testing.assert_allclose(surf, corners - up_d * 0.002, atol=1e-9)
    for n in ("outline", "front", "sensors", "surface"):
        assert it[n].visible()
    for chk in (v.board_check, v.cop_check, v.force_check, v.com_check, v.balance_info):
        assert chk.isVisibleTo(v)
    # the board lies under the feet of the skeleton (same display frame)
    feet = np.array([v._pt(n) for n in ("RHeel", "LHeel", "RBigToe", "LBigToe")
                     if n in v._trc.markers])
    assert np.linalg.norm(feet.mean(axis=0) - corners.mean(axis=0)) < 0.25


def test_frame_overlays_follow_the_trc_time(qapp, demo_trial):
    from poseassess.core.balance.fusion import plumb_point
    from poseassess.gui.widgets.balance_3d import ARROW_M_PER_KG, LIFT_M

    v, fused, board = _viewer_with_demo(demo_trial)
    it = v._bal_items
    up_d = _disp_dir(v, board.up_world)
    checked = 0
    for frame in (0, 57, 200, 300):
        v._show_frame(frame)
        row = _row_for(v, fused, frame)
        assert v.balance_row == row >= 0
        if not np.isfinite(fused.cop_world[row]).all():
            continue
        checked += 1
        cop = _disp(v, fused.cop_world[row])[0] + up_d * LIFT_M
        np.testing.assert_allclose(it["cop"].pos.reshape(-1, 3)[0], cop, atol=1e-9)
        # force arrow: from the COP along up, total_kg * 5 mm long (tip = cone apex)
        start = it["force"].pos[0]
        tip = _mesh_vertexes(it["force_head"])[0]
        np.testing.assert_allclose(start, cop, atol=1e-9)
        np.testing.assert_allclose(tip - start, up_d * fused.total_kg[row] * ARROW_M_PER_KG,
                                   atol=1e-9)
        # COM + plumb line down to the board plane
        com = _disp(v, fused.com_world[row])[0]
        np.testing.assert_allclose(it["com"].pos.reshape(-1, 3)[0], com, atol=1e-9)
        foot = _disp(v, plumb_point(board, fused.com_world[row]))[0]
        np.testing.assert_allclose(it["plumb"].pos, np.vstack([com, foot]), atol=1e-9)
        for n in ("cop", "force", "force_head", "com", "plumb", "plumb_foot"):
            assert it[n].visible(), n
        assert "Wii t" in v.balance_info.text() and "COP ML" in v.balance_info.text()
    assert checked >= 3


def test_cop_trail_covers_two_seconds_and_breaks_at_gaps(qapp, demo_trial):
    v, fused, _ = _viewer_with_demo(demo_trial)
    frame, row = _frame_where(v, fused, lambda r: r >= 90 and np.isfinite(
        fused.cop_world[r - 60:r + 1]).all())
    v._show_frame(frame)
    seg = v._bal_items["cop_trail"].pos
    assert len(seg) == 2 * 60  # 2 s at 30 fps = 60 segments
    # during the jump flight (no load) the COP and the arrow disappear, the COM stays
    frame, row = _frame_where(v, fused, lambda r: not np.isfinite(fused.cop_world[r]).all())
    v._show_frame(frame)
    it = v._bal_items
    assert not it["cop"].visible() and not it["force"].visible()
    assert it["com"].visible()
    # a trail across the flight has fewer segments than rows (broken at the gap)
    frame, row = _frame_where(v, fused, lambda r: r >= 60 and np.isfinite(
        fused.cop_world[r]).all() and not np.isfinite(fused.cop_world[r - 60:r]).all())
    v._show_frame(frame)
    ok = np.isfinite(fused.cop_world[row - 60:row + 1]).all(axis=1)
    assert len(v._bal_items["cop_trail"].pos) == 2 * int((ok[:-1] & ok[1:]).sum())


def test_toggles_hide_and_show(qapp, demo_trial):
    v, fused, _ = _viewer_with_demo(demo_trial)
    it = v._bal_items
    v._show_frame(10)
    v.cop_check.setChecked(False)
    assert not it["cop"].visible() and not it["cop_trail"].visible()
    assert it["force"].visible()
    v.force_check.setChecked(False)
    assert not it["force"].visible() and not it["force_head"].visible()
    v.com_check.setChecked(False)
    assert not it["com"].visible() and not it["plumb"].visible()
    v.board_check.setChecked(False)
    assert not it["outline"].visible() and not it["surface"].visible()
    v.board_check.setChecked(True)
    v.cop_check.setChecked(True)
    assert it["outline"].visible() and it["cop"].visible()


def test_clear_and_clear_balance(qapp, demo_trial):
    v, fused, _ = _viewer_with_demo(demo_trial)
    v.clear_balance()
    assert not any(i.visible() for i in v._bal_items.values())
    assert not any(c.isVisibleTo(v) for c in v._toggles)
    assert v.balance_row == -1 and not v.has_balance()
    v._show_frame(5)  # no overlay comes back
    assert not any(i.visible() for i in v._bal_items.values())
    v.set_balance(fused, None)  # board taken from fused.board
    assert v._bal_items["outline"].visible()
    v.clear()
    assert v._trc is None and not v.has_balance()
    assert not any(i.visible() for i in v._bal_items.values())


def test_board_only_and_no_board(qapp, demo_trial):
    from poseassess.core.balance.board import load_board
    from poseassess.gui.widgets.balance_3d import BalanceSkeleton3DViewer
    from tests.synth import fused_from_demo

    v = BalanceSkeleton3DViewer()
    v.load_trc(demo_trial["trc"])
    v.set_balance(None, load_board(demo_trial["project"]))
    assert v._bal_items["outline"].visible()
    assert v.board_check.isVisibleTo(v) and not v.cop_check.isVisibleTo(v)
    assert not v.balance_info.isVisibleTo(v)
    # fused without a board: only the COM (no plumb line)
    fused = fused_from_demo(demo_trial)
    fused.board = None
    fused.cop_world[:] = np.nan
    fused.force_world[:] = np.nan
    v.set_balance(fused, None)
    v._show_frame(3)
    it = v._bal_items
    assert it["com"].visible() and not it["plumb"].visible()
    assert not it["outline"].visible() and not it["cop"].visible()
    assert v.com_check.isVisibleTo(v) and not v.board_check.isVisibleTo(v)


def test_load_trc_retransforms_the_board(qapp, demo_trial, tmp_path):
    from poseassess.core.balance.trc import read_trc_full
    from tests.synth import write_trc

    v, fused, board = _viewer_with_demo(demo_trial)
    before = v._bal_items["outline"].pos.copy()
    trc = read_trc_full(demo_trial["trc"])
    other = write_trc(tmp_path / "shifted.trc", trc.names, trc.coords + [0.5, 0.0, 0.3],
                      trc.data_rate)
    assert v.load_trc(other)
    corners = _disp(v, board.corners_world)
    np.testing.assert_allclose(v._bal_items["outline"].pos[:4], corners, atol=1e-9)
    assert not np.allclose(before, v._bal_items["outline"].pos)


def test_unaligned_rows_say_so(qapp, demo_trial):
    from tests.synth import fused_from_demo

    fused = fused_from_demo(demo_trial)
    fused.t_rel[:] = np.nan
    fused.total_kg[:] = np.nan
    fused.cop_world[:] = np.nan
    fused.force_world[:] = np.nan
    v, _, _ = _viewer_with_demo(demo_trial, fused)
    v._show_frame(1)
    assert "not aligned" in v.balance_info.text()
    assert not v.force_check.isVisibleTo(v)


# ------------------------------------------------------------------ page
def _page(state):
    from poseassess.gui.pages.viz3d_page import Viz3DPage
    return Viz3DPage(state)


def test_page_draws_overlays_for_the_demo(qapp, demo_trial):
    from poseassess.gui.state import AppState

    state = AppState()
    page = _page(state)
    page.resize(900, 700)
    page.show()
    state.open_project(demo_trial["project"].root)
    pump(0.05)
    assert page.viewer.has_balance()
    f = page.viewer._fused
    assert f is not None and f.recording == demo_trial["rec_id"]
    assert page.wii_status.isVisible()
    txt = page.wii_status.text()
    assert demo_trial["rec_id"] in txt and "+2.500" in txt
    assert "cameras" in page.status.text()  # the original status line is unchanged
    page.viewer._show_frame(30)
    assert page.viewer.balance_row == f.index_for_time(float(page.viewer._trc.times[30]))
    page.close()


def test_page_is_unchanged_without_wii_data(qapp, project):
    from poseassess.gui.state import AppState
    from tests.synth import STANDING, write_trc

    names = list(STANDING)
    pts = np.array([STANDING[n] for n in names], float)[None].repeat(20, axis=0)
    write_trc(project.pose3d_dir / "x_filt.trc", names, pts[..., [1, 2, 0]], 30.0)
    state = AppState()
    page = _page(state)
    page.show()
    state.open_project(project.root)
    pump(0.05)
    assert page.viewer._trc is not None
    assert not page.viewer.has_balance() and not page.viewer._bal_items
    assert not page.wii_status.isVisible()
    assert not any(c.isVisible() for c in page.viewer._toggles)
    assert not (project.root / "wii").exists()  # nothing written into the project
    page.close()


def test_page_reloads_on_balance_changed_and_when_shown(qapp, demo_trial):
    from poseassess.core.balance.alignment import set_manual_offset
    from poseassess.gui.state import AppState

    state = AppState()
    page = _page(state)
    page.show()
    state.open_project(demo_trial["project"].root)
    first = page.viewer._fused
    set_manual_offset(demo_trial["project"], 3.0)
    state.notify_balance_changed("alignment")
    second = page.viewer._fused
    assert second is not first and second.alignment.offset_s == pytest.approx(3.0)
    assert "+3.000" in page.wii_status.text()
    # hidden: only marked dirty, reloaded when shown again
    page.hide()
    set_manual_offset(demo_trial["project"], 2.5)
    state.notify_balance_changed("alignment")
    assert page.viewer._fused is second and page._balance_dirty
    page.show()
    assert page.viewer._fused is not second
    assert page.viewer._fused.alignment.offset_s == pytest.approx(2.5)
    page.close()


def test_page_uses_fuse_trial_and_survives_stubs(qapp, demo_trial, monkeypatch):
    from poseassess.core.balance import fusion
    from poseassess.gui.state import AppState
    from tests.synth import fused_from_demo

    truth = fused_from_demo(demo_trial)
    calls = []

    def fake(project, trc_path=None):
        calls.append(str(trc_path))
        return truth

    monkeypatch.setattr(fusion, "fuse_trial", fake)
    state = AppState()
    page = _page(state)
    page.show()
    state.open_project(demo_trial["project"].root)
    assert calls and calls[-1] == str(demo_trial["trc"])
    assert page.viewer._fused is truth

    def stub(project, trc_path=None):
        raise NotImplementedError("CORE-BOARD")

    monkeypatch.setattr(fusion, "fuse_trial", stub)
    state.notify_balance_changed("trial")
    assert page.viewer._fused is None and page.viewer._board is not None  # board only
    assert "not available yet" in page.wii_status.text()

    def boom(project, trc_path=None):
        raise RuntimeError("unexpected")

    monkeypatch.setattr(fusion, "fuse_trial", boom)
    state.notify_balance_changed("trial")  # never raises into Qt
    assert "unexpected" in page.wii_status.text()
    page.close()


def test_page_warnings_stale_board_and_no_recording(qapp, demo_trial):
    from poseassess.core.balance.trial import set_recording
    from poseassess.gui.state import AppState

    proj = demo_trial["project"]
    calib = proj.calibration_dir / "Calib.toml"
    calib.write_text(calib.read_text() + "\n# changed\n")  # recalibrated -> stale board
    state = AppState()
    page = _page(state)
    page.show()
    state.open_project(proj.root)
    assert "calibration changed" in page.wii_status.text()
    set_recording(proj, None)
    state.notify_balance_changed("trial")
    txt = page.wii_status.text()
    assert "board only" in txt and page.viewer._fused is None
    assert page.viewer._bal_items["outline"].visible()
    page.close()


def test_browsed_trc_outside_the_project_has_no_overlays(qapp, demo_trial, tmp_path,
                                                        monkeypatch):
    from PySide6.QtWidgets import QFileDialog

    from poseassess.gui.state import AppState

    outside = tmp_path / "elsewhere" / "copy.trc"
    outside.parent.mkdir()
    shutil.copy(demo_trial["trc"], outside)
    state = AppState()
    page = _page(state)
    page.show()
    state.open_project(demo_trial["project"].root)
    assert page.viewer.has_balance()
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a, **k: (str(outside), ""))
    page._browse()
    assert not page.viewer.has_balance()
    assert "pose-3d" in page.wii_status.text()
    monkeypatch.setattr(QFileDialog, "getOpenFileName",
                        lambda *a, **k: (str(demo_trial["trc"]), ""))
    page._browse()
    assert page.viewer.has_balance()
    page.close()


def test_main_window_3d_view_page(qapp, demo_trial):
    from poseassess.gui.main_window import MainWindow

    w = MainWindow()
    try:
        w.state.open_project(demo_trial["project"].root)
        page = next(p for p in w.pages if p.nav_title == "5. 3D View")
        w.show()
        w.nav.setCurrentRow(w.pages.index(page))
        pump(0.05)
        assert page.viewer.has_balance()
    finally:
        w.close()


# ------------------------------------------------------------------ rendering (xvfb)
def _grab(viewer):
    from PySide6.QtGui import QImage

    img = viewer.view.grabFramebuffer().convertToFormat(QImage.Format.Format_RGB888)
    w, h = img.width(), img.height()
    arr = np.frombuffer(img.constBits(), np.uint8, count=img.bytesPerLine() * h)
    return arr.reshape(h, img.bytesPerLine())[:, :3 * w].reshape(h, w, 3).copy()


def _count_color(img, hex_color, tol=60):
    c = np.array([int(hex_color[i:i + 2], 16) for i in (1, 3, 5)])
    return int((np.abs(img.astype(int) - c).sum(axis=2) < tol).sum())


def _fresh_gl_programs(monkeypatch):
    """pyqtgraph caches compiled shader programs per class / globally, i.e. per GL context:
    force recompiling them in this test's context (another GL test may have run before)."""
    from pyqtgraph.opengl import shaders
    from pyqtgraph.opengl.items.GLLinePlotItem import GLLinePlotItem
    from pyqtgraph.opengl.items.GLScatterPlotItem import GLScatterPlotItem

    monkeypatch.setattr(GLLinePlotItem, "_shaderProgram", None)
    monkeypatch.setattr(GLScatterPlotItem, "_shaderProgram", None)
    for prog in getattr(shaders, "Shaders", []):
        monkeypatch.setattr(prog, "prog", None)


@pytest.mark.gl
def test_render_overlays(qapp, demo_trial, monkeypatch):
    """One GL view only (pyqtgraph programs are bound to one context): without data, with the
    overlays, and after ``clear_balance`` (must be pixel-identical to the first image)."""
    from poseassess.core.balance.board import load_board
    from poseassess.core.calib_read import read_camera_poses
    from poseassess.gui.widgets.balance_3d import (
        BOARD_COLOR,
        COM_COLOR,
        FORCE_COLOR,
        BalanceSkeleton3DViewer,
    )
    from tests.synth import fused_from_demo

    _fresh_gl_programs(monkeypatch)
    poses = read_camera_poses(demo_trial["project"].calibration_dir / "Calib.toml")
    v = BalanceSkeleton3DViewer()
    v.resize(1000, 600)  # the transport row with every toggle fits
    v.show()
    assert v.load_trc(demo_trial["trc"])
    v.set_cameras(poses)
    v.view.setCameraPosition(distance=3.0, elevation=20, azimuth=-60)
    v._show_frame(40)
    pump(0.3)
    from poseassess.gui.widgets.balance_3d import AXIS_COLORS
    with_axes = _grab(v)
    v.axes_check.setChecked(False)  # the overlay checks below are made without the axes
    pump(0.3)
    plain = _grab(v)
    for col in AXIS_COLORS:
        assert _count_color(with_axes, col, tol=40) > _count_color(plain, col, tol=40) + 20
    assert _count_color(plain, FORCE_COLOR) == 0 and _count_color(plain, COM_COLOR) == 0
    v.set_balance(fused_from_demo(demo_trial), load_board(demo_trial["project"]))
    v._show_frame(40)
    pump(0.3)
    img = _grab(v)
    assert _count_color(img, FORCE_COLOR) > 20 and _count_color(img, COM_COLOR) > 20
    assert _count_color(img, BOARD_COLOR) > _count_color(plain, BOARD_COLOR) + 50
    v.clear_balance()
    pump(0.3)
    again = _grab(v)
    assert again.shape == plain.shape and np.array_equal(again, plain)
    v.close()


def test_page_does_not_fuse_a_trc_it_is_not_showing(qapp, demo_trial):
    """After a pipeline re-run the viewer still plays the old .trc from memory: a
    balance_changed must not fuse the new file under it; Reload shows both again."""
    import os

    from poseassess.core.balance.alignment import set_manual_offset
    from poseassess.gui.state import AppState

    state = AppState()
    page = _page(state)
    page.show()
    state.open_project(demo_trial["project"].root)
    pump(0.05)
    assert page.viewer.has_balance()
    trc = demo_trial["trc"]
    trc.write_text(trc.read_text())  # the pipeline wrote the .trc again
    t = trc.stat().st_mtime + 5
    os.utime(trc, (t, t))
    set_manual_offset(demo_trial["project"], 3.0)
    state.notify_balance_changed("alignment")
    assert not page.viewer.has_balance()
    assert "changed on disk" in page.wii_status.text() and "Reload" in page.wii_status.text()
    page._refresh_list()  # "Reload"
    pump(0.05)
    assert page.viewer.has_balance()
    assert page.viewer._fused.alignment.offset_s == pytest.approx(3.0)
    assert "changed on disk" not in page.wii_status.text()
    page.close()


# ------------------------------------------------------------------ axes, force label, Wii menu
def test_world_and_board_axes(qapp, demo_trial):
    from poseassess.gui.widgets.balance_3d import BOARD_AXIS_M, WORLD_AXIS_M

    v, fused, board = _viewer_with_demo(demo_trial)
    it = v._axes_items
    assert v.axes_check.isVisibleTo(v) and v.axes_check.isChecked()
    # world: origin of the calibration, X / Y / Z of the Calib world (Z up)
    w = it["world_axes"].pos
    np.testing.assert_allclose(w[0::2], np.repeat(_disp(v, [0, 0, 0]), 3, axis=0), atol=1e-9)
    np.testing.assert_allclose(w[1::2], _disp(v, np.eye(3) * WORLD_AXIS_M), atol=1e-9)
    # board: centre of the top surface, x right (TL->TR), y front (BL->TL), z = board up
    b = it["board_axes"].pos
    o, d = b[0], (b[1::2] - b[0::2]) / BOARD_AXIS_M
    c = board.corners_world  # TL, TR, BR, BL
    np.testing.assert_allclose(d[0], _unit(_disp_dir(v, c[1] - c[0])), atol=1e-6)
    np.testing.assert_allclose(d[1], _unit(_disp_dir(v, c[0] - c[3])), atol=1e-6)
    np.testing.assert_allclose(d[2], _unit(_disp_dir(v, board.up_world)), atol=1e-6)
    assert np.linalg.norm(o - _disp(v, board.center_world)[0]) < 0.01
    assert it["board_axes"].visible()
    # toggle off / on; without a board only the world axes stay
    v.axes_check.setChecked(False)
    assert not any(i.visible() for i in it.values())
    v.axes_check.setChecked(True)
    v.clear_balance()
    assert it["world_axes"].visible() and not it["board_axes"].visible()
    v.clear()
    assert not any(i.visible() for i in it.values()) and not v.axes_check.isVisibleTo(v)


def _unit(x):
    x = np.asarray(x, float)
    return x / np.linalg.norm(x)


def test_force_arrow_starts_at_the_cop_and_scales_with_the_load(qapp, demo_trial):
    from poseassess.gui.widgets.balance_3d import ARROW_M_PER_KG, LIFT_M

    v, fused, board = _viewer_with_demo(demo_trial)
    lengths = []
    for i in range(0, v._trc.n_frames, 7):
        v._show_frame(i)
        r = v.balance_row
        if r < 0 or not np.isfinite(fused.total_kg[r]) or not np.isfinite(fused.cop_world[r]).all():
            continue
        start = v._bal_items["force"].pos[0]
        cop = _disp(v, fused.cop_world[r])[0] + _unit(_disp_dir(v, board.up_world)) * LIFT_M
        np.testing.assert_allclose(start, cop, atol=1e-9)
        tip = _mesh_vertexes(v._bal_items["force_head"])[0]
        lengths.append((fused.total_kg[r], np.linalg.norm(tip - start)))
        if "force_label" in v._bal_items:
            assert f"{fused.total_kg[r]:.1f} kg" in v._bal_items["force_label"].text
    assert len(lengths) > 3
    kg, ln = np.array(lengths).T
    np.testing.assert_allclose(ln, kg * ARROW_M_PER_KG, rtol=1e-6)


def test_page_loads_an_original_wii_file_and_sets_the_offset(qapp, demo_trial, monkeypatch):
    from poseassess.core.balance.trial import load_trial
    from poseassess.gui.pages.viz3d_page import Viz3DPage
    from poseassess.gui.state import AppState
    from tests.test_wii_legacy import write_legacy

    proj = demo_trial["project"]
    state = AppState()
    page = Viz3DPage(state)
    page.resize(1000, 700)
    page.show()
    state.set_project(proj)
    pump(0.2)
    assert page.wii_btn.isEnabled()
    f = proj.root.parent / "S01-002.csv"
    write_legacy(f, n=600)
    # the automatic alignment fails (no jumps): quietly, offset 0
    from poseassess.core.balance import alignment
    from poseassess.core.balance.alignment import Alignment, AlignmentResult

    monkeypatch.setattr(alignment, "auto_align", lambda *a, **k: AlignmentResult(
        False, Alignment(), "no jumps", [], [], {}))
    rec = page.load_wii_file(f)
    pump(0.3)
    trial = load_trial(proj)
    assert trial.recording == rec and trial.alignment.method == "manual"
    assert trial.alignment.offset_s == 0.0
    assert page.wii_offset.isEnabled() and page.wii_offset.value() == 0.0
    assert page.viewer.has_balance()
    # the offset box is saved (debounced) as a manual alignment and redraws
    page.wii_offset.setValue(1.25)
    page._offset_timer.stop()
    page._wii_offset_committed()
    pump(0.2)
    trial = load_trial(proj)
    assert trial.alignment.method == "manual" and trial.alignment.offset_s == 1.25
    assert page.viewer._fused is not None
    np.testing.assert_allclose(page.viewer._fused.t_rel, page.viewer._fused.trc_time + 1.25)
    page.close()
