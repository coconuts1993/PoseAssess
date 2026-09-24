"""CORE-BOARD: read_trc_full (missing markers keep columns, units mm/cm, Frame# kept, Pose2Sim
file written by tests/synth.write_trc and by Pose2Sim itself), COM on HALPE_26 / H36M / BVH /
LSTM marker sets, world conversion. COM tests ported from PoseBoard 3ea6d8a
``tests/test_geometry_io.py`` / ``test_core.py``."""

import logging

import numpy as np
import pytest

from poseassess.core.balance.com import center_of_mass
from poseassess.core.balance.trc import (
    find_trc_files,
    read_trc_full,
    trc_sort_key,
    yup_to_zup,
    zup_to_yup,
)
from tests import synth


def test_read_trc_keeps_columns_with_missing_markers(tmp_path):
    p = tmp_path / "x.trc"
    p.write_text("PathFileType\t4\t(X/Y/Z)\tx.trc\nDataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\n"
                 "30\t30\t2\t4\tm\nFrame#\tTime\tA\t\t\tB\t\t\tC\t\t\tD\n"
                 "\t\tX1\tY1\tZ1\tX2\tY2\tZ2\tX3\tY3\tZ3\tX4\tY4\tZ4\n"
                 "1\t0.000\t1\t2\t3\t4\t5\t6\t7\t8\t9\t10\t11\t12\n"
                 "2\t0.033\t1\t2\t3\t\t\t\t7\t8\t9\t\t\t\n", encoding="utf-8")
    tr = read_trc_full(p)
    assert tr.names == ["A", "B", "C", "D"] and tr.n_frames == 2
    np.testing.assert_array_equal(tr.frames, [1, 2])
    np.testing.assert_allclose(tr.times, [0.0, 0.033])
    np.testing.assert_allclose(tr.coords[1, 0], [1, 2, 3])
    assert np.isnan(tr.coords[1, 1]).all() and np.isnan(tr.coords[1, 3]).all()
    np.testing.assert_allclose(tr.coords[1, 2], [7, 8, 9])  # not shifted into B's slot
    assert tr.data_rate == 30.0 and tr.header["Units"] == "m"
    np.testing.assert_allclose(tr.marker("C")[0], [7, 8, 9])
    assert tr.marker("nope") is None


def test_read_trc_units_short_rows_and_missing_frame(tmp_path):
    p = tmp_path / "mm.trc"
    p.write_text("PathFileType\t4\t(X/Y/Z)\tmm.trc\n"
                 "DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\n"
                 "100\t100\t3\t2\tmm\t100\n"
                 "Frame#\tTime\tA\t\t\tB\t\t\n"
                 "\t\tX1\tY1\tZ1\tX2\tY2\tZ2\n"
                 "\n"
                 "\t0.00\t1000\t2000\t3000\t4000\t5000\t6000\n"
                 "7\tnan\t1\t1\t1\t1\t1\t1\n"         # no numeric Time: skipped
                 "8\t0.02\t10\t20\n", encoding="utf-8")  # short row: NaN-padded
    tr = read_trc_full(p)
    np.testing.assert_array_equal(tr.frames, [0, 8])  # missing Frame# -> row index
    np.testing.assert_allclose(tr.coords[0], [[1, 2, 3], [4, 5, 6]])
    np.testing.assert_allclose(tr.coords[1, 0, :2], [0.01, 0.02])
    assert np.isnan(tr.coords[1, 0, 2]) and np.isnan(tr.coords[1, 1]).all()
    q = tmp_path / "cm.trc"
    q.write_text(p.read_text().replace("\tmm\t", "\tcm\t"))
    np.testing.assert_allclose(read_trc_full(q).coords[0, 0], [10, 20, 30])


def test_read_trc_rejects_non_trc(tmp_path):
    p = tmp_path / "a.trc"
    p.write_text("hello\nworld\n")
    with pytest.raises(ValueError, match="not a TRC"):
        read_trc_full(p)
    p.write_text("PathFileType\t4\t(X/Y/Z)\ta.trc\nDataRate\tUnits\n30\tm\nFrame#\tTime\tA\t\t\n"
                 "\t\tX1\tY1\tZ1\n")
    with pytest.raises(ValueError, match="no data rows"):
        read_trc_full(p)
    with pytest.raises(ValueError):
        read_trc_full(tmp_path / "missing.trc")


def test_synth_trc_roundtrip_to_world(tmp_path):
    tr = synth.standing_trial(duration_s=2.0, fps=30, jump_at=None)
    R, t = synth.board_pose()
    world = tr["markers_board"] @ R.T + t
    world[5, 3] = np.nan  # a missing marker
    p = synth.write_trc(tmp_path / "pose-3d" / "s.trc", tr["names"], zup_to_yup(world), 30,
                        frames=np.arange(len(world)) + 12)
    got = read_trc_full(p)
    assert got.names == tr["names"] and got.n_frames == len(world)
    np.testing.assert_array_equal(got.frames, np.arange(len(world)) + 12)
    np.testing.assert_allclose(got.times, (np.arange(len(world)) + 12) / 30, atol=1e-6)
    np.testing.assert_allclose(got.world(), world, atol=1e-6, equal_nan=True)
    assert np.isnan(got.world()[5, 3]).all() and np.isfinite(got.world()[5, 4]).all()
    np.testing.assert_allclose(yup_to_zup(got.coords), world, atol=1e-6, equal_nan=True)


def test_pose2sim_written_trc(tmp_path):
    """A .trc written by Pose2Sim's own ``make_trc`` (Frame# starting after trimming, NaN)."""
    pd = pytest.importorskip("pandas")
    tri = pytest.importorskip("Pose2Sim.triangulation")
    names = ["Hip", "RHip", "LHip"]
    zup = np.arange(5 * 9, dtype=float).reshape(5, 9) / 10
    Q = pd.DataFrame(zup.copy(), index=range(7, 12))
    Q.iloc[1, 3:6] = np.nan
    Q.iloc[2, 6:9] = np.nan
    p = tri.make_trc({"project": {"project_dir": str(tmp_path), "frame_rate": 30}}, Q, names)
    got = read_trc_full(p)
    np.testing.assert_array_equal(got.frames, range(7, 12))
    np.testing.assert_allclose(got.times, np.arange(7, 12) / 30)
    assert got.names == names and got.data_rate == 30
    expect = zup.reshape(5, 3, 3)
    expect[1, 1] = np.nan
    expect[2, 2] = np.nan
    # Pose2Sim writes Y-up; world() gives the Z-up world back
    np.testing.assert_allclose(got.world(), expect, equal_nan=True)


def test_find_trc_files_order(tmp_path):
    d = tmp_path / "pose-3d"
    d.mkdir()
    for n in ("b_0-9.trc", "a_0-9_filt_butterworth.trc", "a_0-9_filt_butterworth_LSTM.trc"):
        (d / n).write_text("")
    assert [p.name for p in find_trc_files(tmp_path)] == [
        "a_0-9_filt_butterworth.trc", "b_0-9.trc", "a_0-9_filt_butterworth_LSTM.trc"]
    assert trc_sort_key(d / "x_filt.trc") < trc_sort_key(d / "x.trc")
    assert find_trc_files(tmp_path / "none") == []


# ------------------------------------------------------------------ centre of mass
def test_com_halpe26_standing():
    names = synth.HALPE26_TRC_MARKERS
    kp = np.array([synth.STANDING[n] for n in names]) + [0.3, 0.2, 0.05]
    com = center_of_mass(kp, names)
    assert com[0] == pytest.approx(0.3, abs=1e-6)  # left-right symmetric
    assert 0.85 < com[2] < 1.15
    _, segs = center_of_mass(kp, names, return_segments=True)
    assert {"head", "trunk", "l_foot", "r_foot", "l_shank", "r_forearm_hand"} <= set(segs)
    # rigid motion: the COM moves with the body
    a = np.deg2rad(30)
    R = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    np.testing.assert_allclose(center_of_mass(kp @ R.T + [1, 2, 3], names), com @ R.T + [1, 2, 3],
                               atol=1e-12)


H36M = {"Hip": (0, 0, .95), "RHip": (.1, 0, .95), "RKnee": (.1, .05, .5), "RFoot": (.1, 0, .08),
        "LHip": (-.1, 0, .95), "LKnee": (-.1, .05, .5), "LFoot": (-.1, 0, .08),
        "Spine": (0, 0, 1.2), "Thorax": (0, 0, 1.4), "Neck": (0, 0, 1.5), "Head": (0, 0, 1.65),
        "LShoulder": (-.18, 0, 1.45), "LElbow": (-.2, 0, 1.15), "LWrist": (-.2, 0, .9),
        "RShoulder": (.18, 0, 1.45), "RElbow": (.2, 0, 1.15), "RWrist": (.2, 0, .9)}


def test_com_h36m_and_bvh_foot_is_the_ankle():
    names, kp = list(H36M), np.array(list(H36M.values()))
    com, segs = center_of_mass(kp, names, return_segments=True)
    ankle = center_of_mass(kp, [n.replace("Foot", "Ankle") for n in names])
    np.testing.assert_allclose(com, ankle)
    assert {"l_shank", "r_shank"} <= set(segs)
    bvh = ["LeftShoulder", "RightShoulder", "LeftUpLeg", "LeftHip", "RightHip", "LeftKnee",
           "RightKnee", "LeftFoot", "RightFoot", "LeftToeBase", "RightToeBase", "Head"]
    pts = {"LeftShoulder": (-.18, 0, 1.45), "RightShoulder": (.18, 0, 1.45),
           "LeftUpLeg": (-.1, 0, .95), "LeftHip": (-.1, 0, .95), "RightHip": (.1, 0, .95),
           "LeftKnee": (-.1, 0, .5), "RightKnee": (.1, 0, .5), "LeftFoot": (-.1, 0, .08),
           "RightFoot": (.1, 0, .08), "LeftToeBase": (-.1, .15, .02),
           "RightToeBase": (.1, .15, .02), "Head": (0, 0, 1.65)}
    _, segs = center_of_mass(np.array([pts[n] for n in bvh]), bvh, return_segments=True)
    assert {"l_shank", "r_shank", "l_foot", "r_foot"} <= set(segs)


def test_com_minimum_keypoints_and_log(caplog):
    names = ["LShoulder", "RShoulder", "LHip", "RHip", "LKnee", "RKnee", "LEar", "REar"]
    kp = np.array([[-.18, 0, 1.45], [.18, 0, 1.45], [-.1, 0, .95], [.1, 0, .95],
                   [-.1, 0, .5], [.1, 0, .5], [-.07, 0, 1.62], [.07, 0, 1.62]])
    with caplog.at_level(logging.WARNING, logger="poseassess.core.balance.com"):
        assert center_of_mass(kp[:4], names[:4]) is None  # shoulders + hips: only ~50 %
        assert center_of_mass(kp[[0, 1, 2, 3, 6, 7]], names[:4] + names[6:]) is None  # + head
        assert center_of_mass(kp[:6], names[:6]) is not None  # + both knees
        assert center_of_mass(kp[[0, 1, 2, 3, 4, 6, 7]], names[:5] + names[6:]) is not None
    assert "No COM" in caplog.text and "60%" in caplog.text
    kp2 = kp.copy()
    kp2[0] = np.nan  # a missing shoulder: no trunk
    assert center_of_mass(kp2, names) is None


def test_com_pose2sim_lstm_markers():
    """Pose2Sim markerAugmentation (LSTM) marker names still give a COM (via the anatomical
    markers it keeps: RHip, RKnee, RAnkle, RShoulder...)."""
    lstm = ["RHip", "RKnee", "RAnkle", "LHip", "LKnee", "LAnkle", "Neck", "RShoulder",
            "LShoulder", "RElbow", "LElbow", "RWrist", "LWrist", "Nose", "r_calc_study",
            "r_toe_study", "L_calc_study", "L_toe_study", "r_knee_study", "L_knee_study"]
    pos = {**{k: v for k, v in synth.STANDING.items()},
           "r_calc_study": (0.1, -0.05, 0.03), "r_toe_study": (0.1, 0.15, 0.02),
           "L_calc_study": (-0.1, -0.05, 0.03), "L_toe_study": (-0.1, 0.15, 0.02),
           "r_knee_study": (0.12, 0.02, 0.5), "L_knee_study": (-0.12, 0.02, 0.5)}
    kp = np.array([pos[n] for n in lstm])
    com = center_of_mass(kp, lstm)
    assert com is not None and abs(com[0]) < 1e-6 and 0.85 < com[2] < 1.15
