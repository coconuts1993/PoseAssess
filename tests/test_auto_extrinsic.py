"""Tests for the automatic extrinsic pipeline (version 1, plain checkerboard).

Synthetic floor-board views (tests/synth.py) stand in for a 1080p GoPro about
3 m from the board. Run: python -m pytest tests -q
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from synth import make_K, render, ring_cameras  # noqa: E402
from poseassess.core.calibration.board_detect import (  # noqa: E402
    align_to_reference, corners_from_outer4, detect_checkerboard, detect_in_frames,
    reorder_grid,
)
from poseassess.core.calibration.origin import canonicalize, reorder_to_corner, z_up  # noqa: E402

NB = (5, 8)
K = make_K()


def _view(k, square=0.06, radius=3.0, height=2.0, blur=0.8, seed=None):
    rv, tv = ring_cameras(6, radius=radius, height=height)[k]
    img, gt = render(NB, square, rv, tv, K, blur=blur, seed=k if seed is None else seed)
    return img, gt, rv, tv


def test_detects_board_with_subpixel_accuracy():
    img, gt, *_ = _view(0)
    det = detect_checkerboard(img, NB)
    assert det.ok
    _, err = align_to_reference(det.points, gt, NB)
    assert err < 0.3


def test_small_blurry_board_found_by_locator():
    # 40 mm squares at ~3.5 m: ~10 px per square, blurred. The classic
    # detector the GUI used before misses this view.
    img, gt, *_ = _view(1, square=0.04, radius=3.5, blur=1.6)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    old_ok = cv2.findChessboardCorners(
        gray, (NB[1], NB[0]),
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK)[0]
    det = detect_checkerboard(img, NB)
    assert not old_ok
    assert det.ok
    assert align_to_reference(det.points, gt, NB)[1] < 1.5


def test_multi_frame_fusion_handles_flipped_numbering():
    img, gt, rv, tv = _view(2)
    frames = [render(NB, 0.06, rv, tv, K, seed=50 + j)[0] for j in range(4)]
    det = detect_in_frames(frames, NB)
    assert det.ok and det.frames_used == 4
    assert align_to_reference(det.points, gt, NB)[1] < 0.3


def test_origin_from_colour_and_handedness():
    for k in range(6):
        img, gt, *_ = _view(k)
        det = detect_checkerboard(img, NB)
        # scramble the numbering the way a detector might
        pts = reorder_grid(det.points, NB, ("id", "flipR", "flipC", "rot180")[k % 4])
        g = canonicalize(img, pts, NB, origin_square="dark")
        assert g.confident
        assert np.linalg.norm(g.points - gt, axis=1).mean() < 0.5, k
        # without the colour the guess is flagged for confirmation
        assert not canonicalize(img, pts, NB).confident


def test_handedness_matches_world_convention():
    _, gt, *_ = _view(3)
    assert z_up(gt, NB)
    assert not z_up(reorder_grid(gt, NB, "flipC"), NB)
    assert not z_up(gt, NB, invert_z=True)


def test_one_click_origin_correction():
    img, gt, *_ = _view(4)
    wrong = reorder_grid(gt, NB, "rot180")
    fixed, sym = reorder_to_corner(wrong, NB, gt[0] + np.array([3.0, -2.0]))
    assert sym == "rot180"
    assert np.allclose(fixed, gt)


def test_four_click_fallback():
    img, gt, *_ = _view(5)
    g = gt.reshape(NB[0], NB[1], 2)
    clicks = np.array([g[-1, 0], g[0, 0], g[-1, -1], g[0, -1]])      # any order
    clicks = clicks + np.random.default_rng(0).normal(0, 1.0, clicks.shape)
    pts = corners_from_outer4(img, NB, clicks)
    assert align_to_reference(pts, gt, NB)[1] < 0.3


def test_quality_report_flags_a_bad_camera():
    from poseassess.core.calibration.extrinsic import align_extrinsics
    from poseassess.core.calibration.quality import assess

    square = 0.06
    intr = {}
    corners = {}
    for k in range(4):
        img, gt, *_ = _view(k, square=square)
        det = detect_checkerboard(img, NB)
        g = canonicalize(img, det.points, NB, origin_square="dark")
        intr[k + 1] = SimpleNamespace(K=K, dist=np.zeros((1, 5)))
        corners[k + 1] = g.points
    extr, aligned = align_extrinsics(intr, corners, NB, square)
    rep = assess(intr, aligned, extr, NB, square)
    assert rep.level == "green", rep.summary_lines()

    # Nudge three corners of camera 3 by several pixels: it must stop being green.
    bad = dict(corners)
    bad[3] = corners[3].copy()
    bad[3][[5, 17, 30]] += np.array([6.0, -5.0])
    extr, aligned = align_extrinsics(intr, bad, NB, square)
    rep = assess(intr, aligned, extr, NB, square)
    levels = {c.camera_index: c.level for c in rep.cameras}
    assert levels[3] in ("yellow", "red"), rep.summary_lines()
    assert rep.level != "green"


def test_project_auto_detect_and_assess(tmp_path):
    """End to end on a project folder: auto-detect every camera, then rate."""
    pytest.importorskip("scipy")
    from poseassess.core.project import Project, ProjectConfig
    from poseassess.core.calibration.intrinsic import IntrinsicResult
    from poseassess.core.calibration.extrinsic import load_points_meta
    from poseassess.core.calibration.orchestrator import (
        assess_project_calibration, auto_detect_extrinsic_corners,
    )

    cfg = ProjectConfig(name="t", num_cameras=4)
    cfg.checkerboard.corners_nb = list(NB)
    cfg.checkerboard.square_size = 60.0
    cfg.checkerboard.origin_square = "dark"
    proj = Project(tmp_path, cfg).create()
    for i in range(1, 5):
        img, *_ = _view(i - 1)
        cv2.imwrite(str(proj.extrinsic_cam_dir(i) / "frame.jpg"), img)
        r = auto_detect_extrinsic_corners(proj, i)
        assert r.ok and r.origin_confident
        meta = load_points_meta(proj.extrinsic_points_file(i))
        assert meta["source"] == "auto" and meta["reviewed"] is False
    outcome = SimpleNamespace(intrinsics=[
        IntrinsicResult(i, K, np.zeros((1, 5)), (1920, 1080), 0.2, 10, 10) for i in range(1, 5)])
    rep = assess_project_calibration(proj, outcome)
    assert rep.level == "green", rep.summary_lines()
