"""CORE-BOARD: overlay drawing smoke tests (draw_board / draw_cop / board_check_image on synth
images; projected corners land on the rendered board)."""

import cv2
import numpy as np

from poseassess.core.balance import overlay
from poseassess.core.balance.board import load_board
from poseassess.core.balance.calib import CameraCalibration, project_cameras
from tests import synth


def demo_scene(demo, i=1):
    proj = demo["project"]
    cam = project_cameras(proj)[f"cam{i:02d}"]
    img = cv2.imread(str(proj.root / "wii" / "board" / f"cam{i:02d}" / "board.jpg"))
    return cam, img, load_board(proj)


def test_project_world(demo_trial):
    cam, _, reg = demo_scene(demo_trial)
    corners = reg.corners_world
    px = overlay.project_world(cam, corners)
    np.testing.assert_allclose(px, reg.clicks["cam01"].as_array()[:4], atol=1e-6)
    behind = cam.center_world - 1.0 * (cam.R.T @ [0, 0, 1.0])  # 1 m behind the camera
    out = overlay.project_world(cam, np.vstack([corners[0], behind, [np.nan] * 3]))
    assert np.isfinite(out[0]).all() and np.isnan(out[1:]).all()
    # without extrinsics the world is the camera frame
    nc = CameraCalibration(cam.name, cam.image_size, cam.K, cam.dist)
    np.testing.assert_allclose(overlay.project_world(nc, cam.world_to_camera(corners)), px,
                               atol=1e-6)


def test_projected_corners_land_on_the_rendered_board(demo_trial):
    for i in (1, 2, 3):
        cam, img, reg = demo_scene(demo_trial, i)
        # the board centre and points just inside the corners are on the light board surface
        inner = reg.pose.board_to_world.apply(reg.geometry.landmarks() * [0.85, 0.85, 1])
        for p in overlay.project_world(cam, inner):
            x, y = np.round(p).astype(int)
            assert img[y, x].mean() > 200, (i, p)


def test_draw_board_clicks_and_cop(demo_trial):
    cam, img, reg = demo_scene(demo_trial)
    a = img.copy()
    overlay.draw_board(a, cam, reg)
    assert (a != img).any()
    front = overlay.project_world(cam, reg.corners_world[:2]).mean(axis=0)
    x, y = np.round(front).astype(int)
    patch = a[y - 2:y + 3, x - 2:x + 3].reshape(-1, 3)
    assert any(tuple(c) == overlay.FRONT_COLOR for c in patch)  # the front edge is highlighted
    b = img.copy()
    clicks = reg.clicks["cam01"].points
    overlay.draw_clicks(b, clicks[:3], next_hint="Click BL")
    overlay.draw_clicks(b, clicks)
    assert (b != img).any()
    c = img.copy()
    overlay.draw_cop(c, cam, reg, np.array([np.nan, 0.0]), 70.0)
    assert (c == img).all()  # nobody on the board: nothing drawn
    overlay.draw_cop(c, cam, reg, np.array([0.05, -0.02]), 70.0)
    p = overlay.project_world(cam, reg.pose.board_to_world.apply([[0.05, -0.02, 0]]))[0]
    x, y = np.round(p).astype(int)
    assert tuple(c[y, x]) == overlay.COP_COLOR


def test_board_check_image(demo_trial):
    cam, img, reg = demo_scene(demo_trial)
    before = img.copy()
    out = overlay.board_check_image(img, cam, reg, reg.clicks["cam01"].as_array())
    assert out.shape == img.shape and (img == before).all() and (out != img).any()
    gray = overlay.board_check_image(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cam, reg)
    assert gray.ndim == 3
    reg.stale = "changed"
    reg.pose.reproj_error_px.pop("cam01")
    assert (overlay.board_check_image(img, cam, reg) != out).any()


def test_overlay_z_down_world(tmp_path):
    demo = synth.make_demo_trial(tmp_path / "zd", up_sign=-1)
    cam, img, reg = demo_scene(demo, 2)
    np.testing.assert_allclose(overlay.project_world(cam, reg.corners_world),
                               reg.clicks["cam02"].as_array()[:4], atol=1e-6)
    a = img.copy()
    overlay.draw_cop(a, cam, reg, np.zeros(2), 70.0)
    # the force line points physically up: its tip is above the COP in the image
    base, tip = overlay.project_world(cam, [reg.center_world,
                                            reg.center_world + reg.up_world * 0.35])
    assert tip[1] < base[1]


def test_board_check_image_marks_the_corner_to_press(demo_trial):
    """Corner check: rings + "PRESS" at the registered TL corner (the corner that was clicked
    as TL), the label towards the board centre; nothing else changes."""
    cam, img, reg = demo_scene(demo_trial)
    plain = overlay.board_check_image(img, cam, reg, reg.clicks["cam01"].as_array())
    marked = overlay.board_check_image(img, cam, reg, reg.clicks["cam01"].as_array(),
                                       highlight="TL")
    diff = np.any(marked != plain, axis=2)
    assert diff.any()
    ys, xs = np.nonzero(diff)
    lm = overlay.project_world(cam, reg.pose.board_to_world.apply(reg.geometry.landmarks()))
    tl, centre = lm[0], lm[4]
    k = overlay._scale(img)
    ring = np.hypot(xs - tl[0], ys - tl[1])
    assert np.any(np.abs(ring - 22 * k) < 3) and np.any(np.abs(ring - 14 * k) < 3)
    far = ring > 28 * k  # the label (beyond the rings): on the board side of the corner
    assert far.any()
    to_c = (centre - tl) / np.linalg.norm(centre - tl)
    along = (xs[far] - tl[0]) * to_c[0] + (ys[far] - tl[1]) * to_c[1]
    assert np.all(along > 0)
    assert np.all(ring < 140 * k)  # a compact mark
