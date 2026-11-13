"""CORE: ``trc_io.read_trc`` keeps empty fields (Pose2Sim writes NaN as an empty cell between
tabs): a missing marker never shifts the markers after it; Time kept, short / text rows skipped,
all-NaN markers left out."""

import numpy as np
import pytest

from poseassess.core.trc_io import read_trc

HEADER = ("PathFileType\t4\t(X/Y/Z)\tx.trc\n"
          "DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\t"
          "OrigDataStartFrame\tOrigNumFrames\n"
          "30\t30\t3\t3\tm\t30\t0\t3\n"
          "Frame#\tTime\tA\t\t\tB\t\t\tC\t\t\t\n"
          "\t\tX1\tY1\tZ1\tX2\tY2\tZ2\tX3\tY3\tZ3\n")


def _write(tmp_path, rows, name="x.trc"):
    p = tmp_path / name
    p.write_text(HEADER + "\n".join(rows) + "\n", encoding="utf-8")
    return p


def test_blank_cells_keep_the_columns(tmp_path):
    # frame 1: marker A missing (3 empty cells); frame 2: only B.y missing
    p = _write(tmp_path, ["0\t0.0\t1\t2\t3\t4\t5\t6\t7\t8\t9",
                          "1\t0.033333\t\t\t\t14\t15\t16\t17\t18\t19",
                          "2\t0.066667\t21\t22\t23\t24\t\t26\t27\t28\t29"])
    t = read_trc(p)
    assert t.marker_names == ["A", "B", "C"]
    np.testing.assert_allclose(t.times, [0.0, 0.033333, 0.066667])
    np.testing.assert_allclose(t.markers["A"][0], [1, 2, 3])
    assert np.isnan(t.markers["A"][1]).all()
    np.testing.assert_allclose(t.markers["B"][1], [14, 15, 16])
    np.testing.assert_allclose(t.markers["C"][1], [17, 18, 19])  # not shifted by A's blanks
    b2 = t.markers["B"][2]
    assert b2[0] == 24 and np.isnan(b2[1]) and b2[2] == 26
    np.testing.assert_allclose(t.markers["C"][2], [27, 28, 29])


def test_trailing_blanks_short_rows_and_text_are_handled(tmp_path):
    p = _write(tmp_path, ["",                                          # blank line
                          "0\t0.0\t1\t2\t3\t4\t5\t6\t\t\t",           # C missing at the end
                          "1\t0.1\t1\t2\t3\t4\t5\t6",                  # short row: C padded
                          "2\t0.2\t1\tabc\t3\t4\t5\t6\t7\t8\t9",      # text: skipped
                          "3\t\t1\t2\t3\t4\t5\t6\t7\t8\t9",           # no Time: skipped
                          "4\t0.4\t7\t8\t9\r"])                       # Windows line end, short
    t = read_trc(p)
    np.testing.assert_allclose(t.times, [0.0, 0.1, 0.4])
    assert set(t.markers) == {"A", "B"}  # C is all-NaN: left out
    np.testing.assert_allclose(t.markers["B"][:2], [[4, 5, 6], [4, 5, 6]])
    np.testing.assert_allclose(t.markers["A"][2], [7, 8, 9])
    assert np.isnan(t.markers["B"][2]).all()


def test_no_data_rows(tmp_path):
    with pytest.raises(ValueError):
        read_trc(_write(tmp_path, ["\t\tX1"]))


def test_pose2sim_style_file_with_missing_markers(tmp_path):
    """``tests.synth.write_trc`` writes NaN like Pose2Sim (empty field): round trip."""
    from tests.synth import write_trc

    rng = np.random.default_rng(1)
    pts = rng.normal(size=(5, 4, 3))
    pts[1, 0] = np.nan          # first marker missing in frame 1
    pts[3, 2, 1] = np.nan       # one coordinate missing
    pts[:, 3] = np.nan          # last marker never seen
    p = write_trc(tmp_path / "s.trc", ["M0", "M1", "M2", "M3"], pts, 30.0)
    t = read_trc(p)
    assert set(t.markers) == {"M0", "M1", "M2"}
    for k in range(3):
        np.testing.assert_allclose(t.markers[f"M{k}"], pts[:, k], atol=1e-6)
    np.testing.assert_allclose(t.times, np.arange(5) / 30.0, atol=1e-6)
