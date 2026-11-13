"""CORE: throwing angles from the 3D keypoints (``core.assessment.throwing``): shoulder external
rotation (0 forearm forward / 90 up / 180 back with the arm abducted 90 deg, both sides
positive), abduction, elbow flexion; invariant under rotation + translation and Pose2Sim's
zup->yup permutation; NaN when undefined; continuity over time; CSV layout."""

import csv
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from poseassess.core.assessment.throwing import (
    THROWING_COLUMNS,
    continuous_degrees,
    export_throwing_csv,
    throwing_angles,
    trunk_axes,
)
from poseassess.core.trc_io import TrcData, read_trc
from tests.synth import FOREARM_M as FORE, THROWING_TRUNK as TRUNK, UPPER_ARM_M as UPPER
from tests.synth import throwing_pose as _frame  # Z-up, facing +Y: the subject's right is +X


def _trc(frames, rate=100.0, transform=None, drop=()):
    names = [n for n in frames[0] if n not in drop]
    markers = {n: np.array([f[n] for f in frames], float) for n in names}
    if transform is not None:
        markers = {n: transform(v) for n, v in markers.items()}
    return TrcData(times=np.arange(len(frames)) / rate, markers=markers,
                   source=Path("synthetic.trc"))


def _circ(a, b):
    """Signed difference a - b modulo 360, in [-180, 180)."""
    return (np.asarray(a) - np.asarray(b) + 180.0) % 360.0 - 180.0


def _random_rigid(seed=3):
    rng = np.random.default_rng(seed)
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q = q * np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    t = rng.normal(size=3) * 5
    return lambda p: p @ q.T + t


# ------------------------------------------------------------------ definitions
def test_trunk_frame_is_right_handed():
    one = {k: np.array([v]) for k, v in TRUNK.items()}
    up, lat, fwd = trunk_axes(one["RShoulder"], one["LShoulder"], one["RHip"], one["LHip"])
    np.testing.assert_allclose(up[0], [0, 0, 1], atol=1e-12)
    np.testing.assert_allclose(lat[0], [1, 0, 0], atol=1e-12)    # subject's right
    np.testing.assert_allclose(fwd[0], [0, 1, 0], atol=1e-12)    # up x right = forward


@pytest.mark.parametrize("er", [0.0, 45.0, 90.0, 135.0, -60.0])
def test_external_rotation_values_both_sides(er):
    a = throwing_angles(_trc([_frame(er, er)]))
    assert list(a.columns) == list(THROWING_COLUMNS)
    assert a["shoulder_er_r"].iloc[0] == pytest.approx(er, abs=1e-6)
    assert a["shoulder_er_l"].iloc[0] == pytest.approx(er, abs=1e-6)  # mirrored: same sign


def test_forearm_forward_up_back():
    ang = {}
    for label, er in (("forward", 0.0), ("up", 90.0), ("back", 180.0)):
        a = throwing_angles(_trc([_frame(er, er)]))
        ang[label] = (a["shoulder_er_r"].iloc[0], a["shoulder_er_l"].iloc[0])
    assert ang["forward"] == pytest.approx((0.0, 0.0), abs=1e-6)
    assert ang["up"] == pytest.approx((90.0, 90.0), abs=1e-6)
    assert _circ(ang["back"], 180.0) == pytest.approx([0.0, 0.0], abs=1e-6)  # +-180


def test_abduction_and_elbow_flexion():
    for abd in (30.0, 90.0, 150.0):
        for flex in (30.0, 90.0, 135.0):
            a = throwing_angles(_trc([_frame(40.0, 40.0, abd_deg=abd, flex_deg=flex)]))
            for s in "rl":
                assert a[f"shoulder_abduction_{s}"].iloc[0] == pytest.approx(abd, abs=1e-6)
                assert a[f"elbow_flexion_{s}"].iloc[0] == pytest.approx(flex, abs=1e-6)
                assert a[f"shoulder_er_{s}"].iloc[0] == pytest.approx(40.0, abs=1e-6)
    # arm down along the trunk = 0, overhead = 180
    down = throwing_angles(_trc([_frame(abd_deg=0.0)]))
    over = throwing_angles(_trc([_frame(abd_deg=180.0)]))
    assert down["shoulder_abduction_r"].iloc[0] == pytest.approx(0.0, abs=1e-6)
    assert over["shoulder_abduction_l"].iloc[0] == pytest.approx(180.0, abs=1e-6)


def test_arm_at_side_external_rotation_points_outward():
    """Arm down, elbow 90: forearm forward = 0, forearm turned outward (laterally) = +90."""
    d = _frame(abd_deg=0.0)
    d["RWrist"] = d["RElbow"] + [FORE, 0, 0]       # right forearm to the right
    d["LWrist"] = d["LElbow"] + [-FORE, 0, 0]      # left forearm to the left
    a = throwing_angles(_trc([d]))
    assert a["shoulder_er_r"].iloc[0] == pytest.approx(90.0, abs=1e-6)
    assert a["shoulder_er_l"].iloc[0] == pytest.approx(90.0, abs=1e-6)


def test_invariant_under_rotation_translation_and_zup_to_yup():
    ers = [0.0, 30.0, 80.0, 120.0, 150.0, 175.0]
    frames = [_frame(e, e - 20.0, abd_deg=80.0 + i, flex_deg=95.0 - 3 * i)
              for i, e in enumerate(ers)]
    ref = throwing_angles(_trc(frames))
    rot = throwing_angles(_trc(frames, transform=_random_rigid()))
    yup = throwing_angles(_trc(frames, transform=lambda p: p[:, [1, 2, 0]]))  # Pose2Sim zup2yup
    pd.testing.assert_frame_equal(ref, rot, atol=1e-8, rtol=0)
    pd.testing.assert_frame_equal(ref, yup, atol=1e-8, rtol=0)
    np.testing.assert_allclose(ref["shoulder_er_r"], ers, atol=1e-6)
    np.testing.assert_allclose(ref["shoulder_er_l"], np.array(ers) - 20.0, atol=1e-6)


def test_through_a_pose2sim_trc_file(tmp_path):
    """Written in the .trc frame (Y-up) with a blank cell, read with ``read_trc``."""
    from tests.synth import write_trc

    frames = [_frame(e, e) for e in (10.0, 60.0, 110.0)]
    names = list(frames[0])
    pts = np.array([[f[n] for n in names] for f in frames])[..., [1, 2, 0]]
    pts[1, names.index("LWrist"), 0] = np.nan
    p = write_trc(tmp_path / "t.trc", names, pts, 60.0)
    a = throwing_angles(read_trc(p))
    np.testing.assert_allclose(a.index, np.arange(3) / 60.0, atol=1e-6)
    np.testing.assert_allclose(a["shoulder_er_r"], [10, 60, 110], atol=1e-4)
    assert np.isnan(a["shoulder_er_l"].iloc[1]) and np.isnan(a["elbow_flexion_l"].iloc[1])
    assert a["shoulder_er_l"].iloc[2] == pytest.approx(110.0, abs=1e-4)
    assert a["elbow_flexion_r"].tolist() == pytest.approx([90, 90, 90], abs=1e-4)
    assert throwing_angles(p).equals(a)  # a path works too


# ------------------------------------------------------------------ undefined -> NaN
def test_straight_elbow_is_undefined():
    a = throwing_angles(_trc([_frame(50.0, 50.0, flex_deg=0.0),
                              _frame(50.0, 50.0, flex_deg=15.0),
                              _frame(50.0, 50.0, flex_deg=25.0)]))
    assert a[["shoulder_er_r", "shoulder_er_l"]].iloc[:2].isna().all().all()
    assert a["shoulder_er_r"].iloc[2] == pytest.approx(50.0, abs=1e-6)
    assert a["elbow_flexion_r"].tolist() == pytest.approx([0.0, 15.0, 25.0], abs=1e-6)
    assert a["shoulder_abduction_r"].tolist() == pytest.approx([90.0] * 3, abs=1e-6)


def test_upper_arm_pointing_forward_is_undefined():
    d = _frame()
    d["RElbow"] = np.array(TRUNK["RShoulder"]) + [0, UPPER, 0]   # straight forward
    d["RWrist"] = d["RElbow"] + [0, 0, FORE]                    # elbow 90, forearm up
    a = throwing_angles(_trc([d]))
    assert np.isnan(a["shoulder_er_r"].iloc[0])
    assert a["shoulder_abduction_r"].iloc[0] == pytest.approx(90.0, abs=1e-6)
    assert a["elbow_flexion_r"].iloc[0] == pytest.approx(90.0, abs=1e-6)
    assert a["shoulder_er_l"].iloc[0] == pytest.approx(0.0, abs=1e-6)


def test_nan_markers_give_nan():
    frames = [_frame(30.0, 30.0) for _ in range(3)]
    frames[1]["RHip"] = np.full(3, np.nan)        # trunk missing: shoulder angles undefined
    frames[2]["RWrist"] = np.full(3, np.nan)
    a = throwing_angles(_trc(frames))
    assert np.isnan(a[["shoulder_er_r", "shoulder_er_l", "shoulder_abduction_r"]].iloc[1]).all()
    assert a["elbow_flexion_r"].iloc[1] == pytest.approx(90.0)  # needs no trunk
    assert np.isnan(a["shoulder_er_r"].iloc[2]) and np.isnan(a["elbow_flexion_r"].iloc[2])
    assert a["shoulder_er_l"].iloc[2] == pytest.approx(30.0, abs=1e-6)


# ------------------------------------------------------------------ continuity over time
def test_continuity_past_180():
    ers = [150.0, 160.0, 170.0, 180.0, 190.0, 185.0]
    a = throwing_angles(_trc([_frame(e, e) for e in ers]))
    np.testing.assert_allclose(a["shoulder_er_r"], ers, atol=1e-6)   # 190, not -170
    np.testing.assert_allclose(a["shoulder_er_l"], ers, atol=1e-6)


def test_continuity_across_a_gap():
    """Unwrapped across a NaN gap (elbow straight for 2 frames): 150 -> 185, not -175."""
    frames = [_frame(130.0, 130.0), _frame(150.0, 150.0),
              _frame(0.0, 0.0, flex_deg=5.0), _frame(0.0, 0.0, flex_deg=5.0),
              _frame(185.0, 185.0), _frame(190.0, 190.0)]
    er = throwing_angles(_trc(frames))["shoulder_er_r"].to_numpy()
    assert np.isnan(er[2:4]).all()
    np.testing.assert_allclose(er[[0, 1, 4, 5]], [130, 150, 185, 190], atol=1e-6)


def test_median_shift():
    ers = np.linspace(-170.0, -260.0, 10)   # median -215: shifted by +360
    a = throwing_angles(_trc([_frame(e, e) for e in ers]))
    np.testing.assert_allclose(a["shoulder_er_r"], ers + 360.0, atol=1e-6)
    assert -180.0 <= np.median(a["shoulder_er_r"]) < 180.0


def test_continuous_degrees_helper():
    v = continuous_degrees([np.nan, 170.0, -175.0, np.nan, -160.0, 160.0])
    np.testing.assert_allclose(v, [np.nan, 170.0, 185.0, np.nan, 200.0, 160.0])
    assert np.isnan(continuous_degrees([np.nan, np.nan])).all()
    np.testing.assert_allclose(continuous_degrees([400.0, 410.0]), [40.0, 50.0])


# ------------------------------------------------------------------ missing markers
def test_missing_trunk_markers_raise():
    with pytest.raises(ValueError, match="RHip"):
        throwing_angles(_trc([_frame()], drop=("RHip",)))
    with pytest.raises(ValueError, match="LShoulder, LHip"):
        throwing_angles(_trc([_frame()], drop=("LShoulder", "LHip")))


def test_missing_arm_gives_nan_side_or_raises():
    a = throwing_angles(_trc([_frame(30.0, 45.0)], drop=("LWrist",)))
    assert a[["shoulder_er_l", "shoulder_abduction_l", "elbow_flexion_l"]].isna().all().all()
    assert a["shoulder_er_r"].iloc[0] == pytest.approx(30.0, abs=1e-6)
    with pytest.raises(ValueError, match="RElbow.*LWrist"):
        throwing_angles(_trc([_frame()], drop=("RElbow", "LWrist")))


# ------------------------------------------------------------------ CSV export
def test_export_csv_layout(tmp_path):
    frames = [_frame(20.0, 30.0), _frame(25.0, 35.0, flex_deg=5.0)]
    trc = _trc(frames, drop=("LElbow",))
    out = export_throwing_csv(trc, tmp_path / "sub" / "trial_throwing.csv")
    assert out == tmp_path / "sub" / "trial_throwing.csv" and out.is_file()
    with open(out, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    assert rows[0] == ["time"] + list(THROWING_COLUMNS)
    assert rows[1] == ["s"] + ["deg"] * 6
    assert len(rows) == 4
    r0 = dict(zip(rows[0], rows[2]))
    assert r0["time"] == "0.00000" and r0["shoulder_er_r"] == "20.00000"
    assert r0["shoulder_er_l"] == "" and r0["elbow_flexion_l"] == ""   # no LElbow: blank
    assert r0["elbow_flexion_r"] == "90.00000"
    r1 = dict(zip(rows[0], rows[3]))
    assert r1["time"] == "0.01000" and r1["shoulder_er_r"] == ""      # straight elbow: blank
    assert r1["elbow_flexion_r"] == "5.00000"
