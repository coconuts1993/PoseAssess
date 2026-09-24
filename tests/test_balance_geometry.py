"""CORE-BOARD: board geometry and registration (``poseassess.core.balance.geometry``).

Ported from PoseBoard 3ea6d8a ``tests/test_geometry_io.py`` (IPPE branch, floor fit, mirrored
clicks, mixed extrinsics) and the geometry part of ``tests/test_core.py`` (kabsch, PnP,
triangulation, 180° rotation), plus Calib.toml worlds whose +Z points DOWN (PoseAssess
``invert_z`` / click origin): the same pixels in the conjugated world must give exactly the
conjugated pose, with the board normal physically up.
"""

import dataclasses

import cv2
import numpy as np
import pytest

from poseassess.core.balance.calib import CameraCalibration
from poseassess.core.balance.geometry import (
    FLIP_Z,
    MIRRORED_CLICKS_MSG,
    BoardGeometry,
    BoardPose,
    RigidTransform,
    backproject_to_plane,
    kabsch,
    procrustes_2d,
    project_to_image,
    register_board,
    reprojection_error,
    rotate_board_frame,
    solve_board_pnp,
    triangulate_point,
)
from tests import synth

GEO = BoardGeometry()
MODEL = GEO.landmarks()
MIRROR = [1, 0, 3, 2, 4]  # TR, TL, BL, BR, C: left and right swapped


def make_cam(name, center, target=(0, 0, 0), size=(1280, 720), f=1000.0):
    """Upright camera in a Z-up world (PoseBoard's ``make_cam``)."""
    return synth.to_calibration(synth.look_at_camera(center, target, size, f, name=name))


def no_ext(cam):
    return CameraCalibration(cam.name, cam.image_size, cam.K, cam.dist)


def flip_world(cam):
    """The same physical camera described in the world conjugated by FLIP_Z (+Z down)."""
    if not cam.has_extrinsics:
        return cam
    return dataclasses.replace(cam, rvec=cv2.Rodrigues(cam.R @ FLIP_Z)[0].ravel(),
                               tvec=cam.t.copy())


def flip_T(T):
    return RigidTransform(FLIP_Z @ T.R, FLIP_Z @ T.t)


def yaw_transform(deg=20.0, xy=(0.3, 0.2)):
    a = np.deg2rad(deg)
    R = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    return RigidTransform(R, np.array([xy[0], xy[1], GEO.height_mm / 1000]))


def board_transform():
    a = np.deg2rad(20)
    R = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    return RigidTransform(R, np.array([0.8, 0.5, 0.053]))


# ------------------------------------------------------------------ utilities
def test_kabsch_and_procrustes():
    rng = np.random.default_rng(1)
    src = rng.normal(size=(6, 3))
    T = board_transform()
    est = kabsch(src, T.apply(src))
    np.testing.assert_allclose(est.R, T.R, atol=1e-9)
    np.testing.assert_allclose(est.t, T.t, atol=1e-9)
    assert np.linalg.det(kabsch(src, src * [1, 1, -1]).R) == pytest.approx(1.0)  # never a mirror
    a = np.deg2rad(-35)
    R2 = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    pts = rng.normal(size=(5, 2))
    R, t = procrustes_2d(pts, pts @ R2.T + [0.3, -0.1])
    np.testing.assert_allclose(R, R2, atol=1e-12)
    np.testing.assert_allclose(t, [0.3, -0.1], atol=1e-12)


def test_triangulate_backproject_project():
    X = np.array([0.4, 0.3, 0.9])
    cams = [make_cam("a", [0.8, -2.5, 1.3], [0.8, 0.5, 0.9]), make_cam("b", [3.0, -1.0, 1.3])]
    np.testing.assert_allclose(triangulate_point(cams, [c.project(X)[0] for c in cams]), X,
                               atol=1e-9)
    with pytest.raises(ValueError):
        triangulate_point(cams[:1], [cams[0].project(X)[0]])
    floor = np.array([[0.2, 0.1, 0.053], [-0.3, 0.4, 0.053]])
    np.testing.assert_allclose(backproject_to_plane(cams[0], cams[0].project(floor), 0.053), floor,
                               atol=1e-9)
    with pytest.raises(ValueError, match="2. Calibration"):  # a plane above the camera
        backproject_to_plane(cams[0], cams[0].project(floor), 5.0)
    # without extrinsics the world is the camera frame
    nc = no_ext(cams[0])
    pc = cams[0].world_to_camera(floor)
    np.testing.assert_allclose(project_to_image(nc, pc), cams[0].project(floor), atol=1e-9)
    assert reprojection_error(nc, pc, cams[0].project(floor)) < 1e-9


# ------------------------------------------------------------------ board registration
def test_register_board_single_camera_pnp():
    T = board_transform()
    cam = make_cam("c0", [0.8, -1.5, 1.6], [0.8, 0.5, 0.0])
    clicks = cam.project(T.apply(MODEL)) + np.random.default_rng(0).normal(0, 0.3, (5, 2))
    pose = register_board(GEO, [cam], [clicks])
    assert pose.method == "pnp"
    np.testing.assert_allclose(pose.board_to_world.t, T.t, atol=0.01)
    np.testing.assert_allclose(pose.board_to_world.R, T.R, atol=0.02)
    assert pose.reproj_error_px["c0"] < 1.0


def test_register_board_triangulation():
    T = board_transform()
    cams = [make_cam("a", [-0.5, -1.5, 1.8], [0.8, 0.5, 0]),
            make_cam("b", [2.2, -1.2, 1.7], [0.8, 0.5, 0])]
    pose = register_board(GEO, cams, [c.project(T.apply(MODEL)) for c in cams])
    assert pose.method == "triangulation"
    np.testing.assert_allclose(pose.board_to_world.t, T.t, atol=1e-6)
    np.testing.assert_allclose(pose.board_to_world.R, T.R, atol=1e-6)


def test_pnp_never_returns_the_flipped_ippe_solution():
    """Low, oblique camera + noisy clicks: plain IPPE picked a board tilted ~150° in ~14% of
    the cases; the branch closest to world UP (or pointing up in the image) is always right,
    in a Z-up and in a Z-down world."""
    T = yaw_transform()
    cam = make_cam("c", (2.5, -2.5, 1.0), (0.3, 0.2, 0))
    rng = np.random.default_rng(0)
    cos10 = np.cos(np.deg2rad(10))
    n_amb = 0
    for _ in range(150):
        k = cam.project(T.apply(MODEL)) + rng.normal(0, 2.0, (5, 2))
        free = register_board(GEO, [cam], [k], floor=False)
        assert free.up_world[2] > cos10 and free.tilt_deg < 10
        down = register_board(GEO, [flip_world(cam)], [k], floor=False)  # up_sign detected
        assert down.up_world[2] < -cos10 and down.tilt_deg < 10
        T_bc, amb = solve_board_pnp(no_ext(cam), MODEL, k, return_ambiguity=True)
        n_amb += amb
        assert T_bc.R[:, 2] @ (cam.R @ [0, 0, 1.0]) > cos10  # upright-camera prior
    assert n_amb > 0  # close IPPE errors happen and are reported


def test_solve_board_pnp_up_world():
    T = yaw_transform()
    cam = make_cam("c", (1.5, -2.0, 1.6), (0.3, 0.2, 0))
    k = cam.project(T.apply(MODEL))
    fc = flip_world(cam)
    T_bc = solve_board_pnp(fc, MODEL, k, up_world=[0, 0, -1])
    # board normal in the flipped world points to -Z (physically up)
    assert (fc.R.T @ T_bc.R[:, 2])[2] < -0.99
    with pytest.raises(ValueError, match="mirrored"):
        solve_board_pnp(cam, MODEL, k[MIRROR])


@pytest.mark.parametrize("up_sign", [1, -1])
def test_floor_fit_removes_tilt_bias_single_camera(up_sign):
    T = yaw_transform()
    com = np.array([0.3, 0.2, 1.0])
    cam = make_cam("c", (1.5, -2.0, 1.6), (0.3, 0.2, 0))
    if up_sign < 0:
        T, com, cam = flip_T(T), FLIP_Z @ com, flip_world(cam)
    rng = np.random.default_rng(1)
    err_floor, err_free = [], []
    truth_b = T.inverse().apply(com)
    for _ in range(100):
        k = cam.project(T.apply(MODEL)) + rng.normal(0, 1.0, (5, 2))
        b = register_board(GEO, [cam], [k], up_sign=up_sign)
        assert b.floor_constrained and b.method == "pnp" and b.world_camera is None
        np.testing.assert_allclose(b.up_world, [0, 0, up_sign], atol=1e-12)
        assert b.board_to_world.t[2] == pytest.approx(up_sign * GEO.height_mm / 1000)
        assert np.linalg.det(b.board_to_world.R) == pytest.approx(1.0)
        assert b.tilt_deg is not None and b.reproj_error_px["c"] < 5
        err_floor.append(np.hypot(*(b.world_to_board.apply(com) - truth_b)[:2]))
        free = register_board(GEO, [cam], [k], floor=False, up_sign=up_sign)
        err_free.append(np.hypot(*(free.world_to_board.apply(com) - truth_b)[:2]))
    assert np.median(err_floor) < 0.006  # a few mm instead of 1-1.5 cm
    assert np.median(err_floor) < 0.5 * np.median(err_free)


@pytest.mark.parametrize("up_sign", [1, -1])
def test_floor_fit_two_cameras_and_large_tilt_warning(up_sign):
    T = yaw_transform(-35)
    cams = [make_cam("a", (-0.5, -1.5, 1.8), (0.3, 0.2, 0)),
            make_cam("b", (2.2, -1.2, 1.7), (0.3, 0.2, 0))]
    rng = np.random.default_rng(2)
    clicks = [c.project(T.apply(MODEL)) + rng.normal(0, 1.0, (5, 2)) for c in cams]
    a = np.deg2rad(20)
    tilt = RigidTransform(T.R @ np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)],
                                          [0, np.sin(a), np.cos(a)]]), T.t)
    tilt_clicks = [c.project(tilt.apply(MODEL)) for c in cams]
    if up_sign < 0:
        T, cams = flip_T(T), [flip_world(c) for c in cams]
    b = register_board(GEO, cams, clicks)  # up_sign detected from the cameras
    assert b.method == "triangulation" and b.floor_constrained
    np.testing.assert_allclose(b.up_world, [0, 0, up_sign], atol=1e-12)
    np.testing.assert_allclose(b.board_to_world.t, T.t, atol=0.01)
    assert not b.warnings
    # A board that is really tilted (not on the checkerboard floor): kept 6-DoF, with a warning
    b = register_board(GEO, cams, tilt_clicks)
    assert not b.floor_constrained and b.tilt_deg == pytest.approx(20, abs=0.5)
    assert b.warnings and "tilted" in b.warnings[0]
    assert b.up_world[2] * up_sign > 0.9


@pytest.mark.parametrize("up_sign", [1, -1])
@pytest.mark.parametrize("mirror", [MIRROR, [3, 2, 1, 0, 4]])  # left/right and front/back
def test_mirrored_clicks_are_rejected(mirror, up_sign):
    T = yaw_transform()
    cam = make_cam("c", (1.5, -2.0, 1.6), (0.3, 0.2, 0))
    cams = [cam, make_cam("d", (-0.8, -1.4, 1.7), (0.3, 0.2, 0))]
    clicks = [c.project(T.apply(MODEL)) for c in cams]
    if up_sign < 0:
        cams = [flip_world(c) for c in cams]
    k = clicks[0][mirror]
    with pytest.raises(ValueError, match="mirrored"):
        register_board(GEO, cams[:1], [k])
    with pytest.raises(ValueError, match="mirrored"):
        register_board(GEO, [no_ext(cams[0])], [k])
    with pytest.raises(ValueError, match="mirrored"):
        register_board(GEO, cams, [c[mirror] for c in clicks])
    # The correct order still works everywhere
    assert register_board(GEO, [no_ext(cams[0])], [clicks[0]]).method == "pnp"
    assert register_board(GEO, cams, clicks).method == "triangulation"
    assert "TL, TR, BR, BL, C" in MIRRORED_CLICKS_MSG


def test_cameras_without_extrinsics_are_not_mixed_in():
    T = yaw_transform()
    cam0 = make_cam("cam0", (1.5, -2.0, 1.6), (0.3, 0.2, 0))
    cam1 = make_cam("cam1", (-1.0, -1.5, 1.5), (0.3, 0.2, 0))
    k0, k1 = cam0.project(T.apply(MODEL)), cam1.project(T.apply(MODEL))
    b = register_board(GEO, [cam0, no_ext(cam1)], [k0, k1])
    assert set(b.reproj_error_px) == {"cam0"} and b.reproj_error_px["cam0"] < 0.5
    assert any("cam1" in n and "2. Calibration" in n for n in b.notes) and b.world_camera is None
    np.testing.assert_allclose(b.board_to_world.t, T.t, atol=1e-6)
    # No extrinsics at all: the first camera defines the world, errors only for it; an
    # explicit up_sign=-1 does not flip a camera-frame world
    for up in (None, -1):
        b = register_board(GEO, [no_ext(cam1), no_ext(cam0)], [k1, k0], up_sign=up)
        assert b.world_camera == "cam1" and set(b.reproj_error_px) == {"cam1"}
        assert b.reproj_error_px["cam1"] < 0.5 and not b.floor_constrained
        np.testing.assert_allclose(b.board_to_world.t, cam1.world_to_camera(T.t)[0], atol=1e-6)
    # Incomplete clicks are ignored; none at all is an error
    b = register_board(GEO, [cam0, cam1], [k0, k1[:4]])
    assert set(b.reproj_error_px) == {"cam0"} and b.method == "pnp"
    with pytest.raises(ValueError, match="5 board points"):
        register_board(GEO, [cam0], [None])


def test_z_down_world_is_the_conjugated_pose():
    """Same pixels, cameras described in the flipped world: the pose is exactly F @ pose."""
    cams_up = synth.ring_cameras(3, up_sign=1)
    cams = [synth.to_calibration(c) for c in cams_up]
    R, t = synth.board_pose()
    clicks = synth.board_clicks(cams_up, R, t, noise_px=0.7, seed=3)
    for sub in ([0], [0, 1], [0, 1, 2]):
        up = register_board(GEO, [cams[i] for i in sub], [clicks[i] for i in sub])
        down = register_board(GEO, [flip_world(cams[i]) for i in sub], [clicks[i] for i in sub])
        np.testing.assert_allclose(down.board_to_world.R, FLIP_Z @ up.board_to_world.R, atol=1e-7)
        np.testing.assert_allclose(down.board_to_world.t, FLIP_Z @ up.board_to_world.t, atol=1e-7)
        np.testing.assert_allclose(down.up_world, [0, 0, -1], atol=1e-9)
        assert down.method == up.method and down.floor_constrained == up.floor_constrained
        assert down.tilt_deg == pytest.approx(up.tilt_deg, abs=1e-6)
        for k, v in up.reproj_error_px.items():
            assert down.reproj_error_px[k] == pytest.approx(v, abs=1e-6)
        # the board frame stays right-handed: +x right, +y front, +z up
        np.testing.assert_allclose(np.linalg.det(down.board_to_world.R), 1.0)
    # the synth Z-down scene gives the synth truth
    cams_d = synth.ring_cameras(3, up_sign=-1)
    Rd, td = synth.board_pose(up_sign=-1)
    pose = register_board(GEO, [synth.to_calibration(c) for c in cams_d],
                          synth.board_clicks(cams_d, Rd, td))
    np.testing.assert_allclose(pose.board_to_world.R, Rd, atol=1e-6)
    np.testing.assert_allclose(pose.board_to_world.t, td, atol=1e-6)


def test_rotate_board_frame_only_180_and_pose_roundtrip():
    cam = make_cam("c", (1.5, -2.0, 1.6), (0.3, 0.2, 0))
    b = register_board(GEO, [cam], [cam.project(yaw_transform().apply(MODEL))])
    with pytest.raises(ValueError):
        rotate_board_frame(b, 90)
    r = rotate_board_frame(b, 180)
    np.testing.assert_allclose(r.board_to_world.apply([0.1, 0.05, 0]),
                               b.board_to_world.apply([-0.1, -0.05, 0]), atol=1e-9)
    assert r.floor_constrained and r.tilt_deg == b.tilt_deg
    back = BoardPose.from_dict(r.to_dict())
    np.testing.assert_allclose(back.board_to_world.R, r.board_to_world.R)
    assert back.floor_constrained and back.notes == r.notes
    old = {"board_to_world": b.board_to_world.to_dict(), "method": "pnp", "reproj_error_px": {}}
    assert BoardPose.from_dict(old).world_camera is None  # files of older versions
