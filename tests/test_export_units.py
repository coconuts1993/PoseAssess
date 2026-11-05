"""'Export all angles' unit row: translations in metres (pelvis_tx/ty/tz and Abs_t1/t2), the
coupled knee *_beta angle in radians when OpenSim left it there (decided from the data)."""

import csv
from pathlib import Path

import numpy as np
import pandas as pd

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
