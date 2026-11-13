"""'Export all angles' unit row: translations in metres (pelvis_tx/ty/tz and Abs_t1/t2), the
coupled knee *_beta angle in radians when OpenSim left it there (decided from the data).
``wrap_degrees``: angles in degrees wrapped to [-180, 180) (400 -> 40, -200 -> 160, 180 -> -180),
metres and radian betas untouched, the summary of the wrapped values; off by default."""

import csv
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from poseassess.core.assessment.export import coordinate_unit, export_all
from poseassess.core.assessment.mot_io import MotionData


def _motion(beta_in_radians=True, in_degrees=True):
    t = np.linspace(0, 1, 11)
    knee = np.linspace(5, 60, 11)
    df = pd.DataFrame({
        "pelvis_tilt": np.linspace(-5, 5, 11),
        "pelvis_tx": np.linspace(0, 1, 11),
        "knee_angle_r": knee,
        "knee_angle_r_beta": np.radians(knee) if beta_in_radians else knee,
        "Abs_t1": np.zeros(11),
        "Abs_t2": np.zeros(11),
        "L1_T12_Flex_Ext": np.linspace(-2, 2, 11),
    }, index=t)
    return MotionData(df=df, in_degrees=in_degrees, source=Path("x.mot"))


def _rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.reader(f))


def test_units_row_and_summary(tmp_path):
    m = _motion()
    paths = export_all(m, tmp_path, "trial")
    header, units = _rows(paths["angles"])[:2]
    got = dict(zip(header, units))
    assert got == {"time": "s", "pelvis_tilt": "deg", "pelvis_tx": "m", "knee_angle_r": "deg",
                   "knee_angle_r_beta": "rad", "Abs_t1": "m", "Abs_t2": "m",
                   "L1_T12_Flex_Ext": "deg"}
    summary = {r[0]: r[1] for r in _rows(paths["summary"])[1:]}
    assert summary["knee_angle_r_beta"] == "rad" and summary["Abs_t1"] == "m"
    assert summary["L1_T12_Flex_Ext"] == "deg"  # "_T12" is not a translation


def test_beta_follows_the_data():
    assert coordinate_unit(_motion(beta_in_radians=False), "knee_angle_r_beta") == "deg"
    m = _motion()
    m.df.drop(columns="knee_angle_r", inplace=True)  # no base column: OpenSim's behaviour
    assert coordinate_unit(m, "knee_angle_r_beta") == "rad"


def test_radian_files():
    m = _motion(in_degrees=False)
    assert coordinate_unit(m, "pelvis_tilt") == "rad"
    assert coordinate_unit(m, "knee_angle_r_beta") == "rad"
    assert coordinate_unit(m, "pelvis_ty") == "m" and coordinate_unit(m, "Abs_t2") == "m"


# ------------------------------------------------------------------ wrap_degrees
def _turns_motion():
    """Angles that drifted by whole turns (an OpenSim arm_rot_r of 400 = 40 deg)."""
    t = np.linspace(0, 1, 4)
    knee = np.array([10.0, 20.0, 30.0, 40.0])
    df = pd.DataFrame({
        "arm_rot_r": [400.0, -200.0, 180.0, np.nan],
        "arm_flex_r": [-180.0, 179.5, 540.0, -540.0],
        "pelvis_tx": [400.0, -200.0, 180.0, 1.0],      # metres: never wrapped
        "knee_angle_r": knee,
        "knee_angle_r_beta": np.radians(knee) + 2 * np.pi,  # rad: never wrapped
    }, index=t)
    return MotionData(df=df, in_degrees=True, source=Path("x.mot"))


def _angles(path):
    rows = _rows(path)
    return {c: [r[i] for r in rows[2:]] for i, c in enumerate(rows[0])}


def test_wrap_180():
    from poseassess.core.assessment.export import wrap_180

    got = wrap_180([400.0, -200.0, 180.0, -180.0, 179.999, 0.0, np.nan, 720.0 + 10.0])
    np.testing.assert_allclose(got[:6], [40.0, 160.0, -180.0, -180.0, 179.999, 0.0])
    assert np.isnan(got[6]) and got[7] == pytest.approx(10.0)


def test_wrap_degrees_time_series_and_summary(tmp_path):
    m = _turns_motion()
    before = m.df.copy()
    paths = export_all(m, tmp_path, "w", wrap_degrees=True)
    a = _angles(paths["angles"])
    assert a["arm_rot_r"] == ["40.00000", "160.00000", "-180.00000", "nan"]
    assert a["arm_flex_r"] == ["-180.00000", "179.50000", "-180.00000", "-180.00000"]
    assert a["pelvis_tx"] == ["400.00000", "-200.00000", "180.00000", "1.00000"]
    assert [float(v) for v in a["knee_angle_r_beta"]] == \
        pytest.approx(np.radians([10, 20, 30, 40]) + 2 * np.pi, abs=1e-5)
    s = {r[0]: r[1:] for r in _rows(paths["summary"])[1:]}
    # coordinate, unit, min, max, ROM, mean, peak_abs -- of the wrapped values
    assert s["arm_rot_r"] == ["deg", "-180.000", "160.000", "340.000", "6.667", "180.000"]
    assert s["pelvis_tx"][:3] == ["m", "-200.000", "400.000"]
    assert s["knee_angle_r_beta"][0] == "rad" and float(s["knee_angle_r_beta"][1]) > 6.0
    pd.testing.assert_frame_equal(m.df, before)  # the loaded motion is not changed


def test_wrap_degrees_default_is_off(tmp_path):
    m = _turns_motion()
    paths = export_all(m, tmp_path, "raw")
    a = _angles(paths["angles"])
    assert a["arm_rot_r"] == ["400.00000", "-200.00000", "180.00000", "nan"]
    s = {r[0]: r[1:] for r in _rows(paths["summary"])[1:]}
    assert s["arm_rot_r"][1:3] == ["-200.000", "400.000"]
    from poseassess.core.assessment.export import (
        all_coordinate_stats, export_angles_csv, export_summary_csv)
    export_angles_csv(m, tmp_path / "a.csv")
    export_summary_csv(m, tmp_path / "s.csv")
    assert _rows(tmp_path / "a.csv") == _rows(paths["angles"])
    assert _rows(tmp_path / "s.csv") == _rows(paths["summary"])
    stats = {c.coord: c for c in all_coordinate_stats(m, wrap_degrees=True)}
    assert stats["arm_rot_r"].max == pytest.approx(160.0)
    assert {c.coord: c for c in all_coordinate_stats(m)}["arm_rot_r"].max == 400.0


def test_wrap_degrees_radian_file_untouched(tmp_path):
    m = _motion(in_degrees=False)
    m.df["pelvis_tilt"] = np.linspace(4, 7, 11)  # rad > pi: not wrapped (no 'deg' coordinate)
    paths = export_all(m, tmp_path, "r", wrap_degrees=True)
    assert _rows(paths["angles"]) == _rows(export_all(m, tmp_path / "b", "r")["angles"])
