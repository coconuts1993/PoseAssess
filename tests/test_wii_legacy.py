"""Files of the lab's original Wii program: detection, conversion, clock, import; and the
readable computer time (time_local) in recordings."""

from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path

import numpy as np
import pytest

from poseassess.wii import io as wio
from poseassess.wii import legacy


def write_legacy(path: Path, n: int = 300, ms0: int = 202348, seed: int = 0) -> np.ndarray:
    """A file like the original program writes (ms TL TR BL BR copx_cm copy_cm total)."""
    rng = np.random.default_rng(seed)
    ms = ms0 + np.cumsum(rng.integers(0, 28, n))
    tl, tr, bl, br = (rng.uniform(15, 25, n) for _ in range(4))
    tl[:20] = tr[:20] = bl[:20] = br[:20] = 0.01  # empty board at the start
    tot = tl + tr + bl + br
    x = (tr + br - tl - bl) / tot * 25
    y = (bl + br - tl - tr) / tot * 15
    with open(path, "w") as f:
        for row in zip(ms, tl, tr, bl, br, x, y, tot):
            f.write(" ".join(f"{v:g}" if i else str(int(v)) for i, v in enumerate(row)) + "\n")
    return np.column_stack([ms, tl, tr, bl, br, x, y, tot])


def test_detection(tmp_path):
    f = tmp_path / "trial-003"
    write_legacy(f)
    assert legacy.is_legacy_wii_file(f)
    other = tmp_path / "wii.csv"
    other.write_text("t,t_rel,total_kg\n0,0,1\n")
    assert not legacy.is_legacy_wii_file(other)
    assert not legacy.is_legacy_wii_file(tmp_path / "missing")


def test_conversion_matches_the_programs_own_cop(tmp_path):
    f = tmp_path / "a.csv"
    raw = write_legacy(f)
    cols, meta = legacy.read_legacy_wii(f, clock="none")
    np.testing.assert_allclose(cols["t"], raw[:, 0] / 1000)
    assert cols["t_rel"][0] == 0
    for j, name in enumerate(("TL_kg", "TR_kg", "BL_kg", "BR_kg")):
        np.testing.assert_allclose(cols[name], raw[:, 1 + j], rtol=1e-5)
    ok = cols["total_kg"] > 10
    # PoseAssess COP (x right, y front, m) vs the program's (x right, y back, cm, half board size)
    np.testing.assert_allclose(cols["cop_x_board"][ok] / (0.433 / 2) * 25, raw[ok, 5], atol=1e-3)
    np.testing.assert_allclose(-cols["cop_y_board"][ok] / (0.238 / 2) * 15, raw[ok, 6], atol=1e-3)
    assert np.isnan(cols["cop_x_board"][~ok]).all()
    assert np.isnan(cols["t_unix"]).all() and meta["clock_source"] == "none"


def test_clock_from_file_mtime_and_counter_zero(tmp_path):
    f = tmp_path / "a.csv"
    raw = write_legacy(f)
    end = 1_790_000_000.25
    os.utime(f, (end, end))
    cols, meta = legacy.read_legacy_wii(f)
    assert meta["clock_source"] == "file_mtime" and meta["clock_estimated"]
    assert cols["t_unix"][-1] == pytest.approx(end)
    np.testing.assert_allclose(np.diff(cols["t_unix"]), np.diff(raw[:, 0]) / 1000, atol=1e-6)
    boot = 1_789_999_000.0
    cols, meta = legacy.read_legacy_wii(f, clock="counter_zero", counter_zero_unix=boot)
    assert cols["t_unix"][0] == pytest.approx(boot + raw[0, 0] / 1000)
    with pytest.raises(ValueError):
        legacy.read_legacy_wii(f, clock="counter_zero")


def test_import_into_project(tmp_path):
    from poseassess.core.balance.trial import import_recording, load_trial
    from poseassess.core.project import Project

    proj = Project(tmp_path / "proj")
    proj.create()
    f = tmp_path / "S01-002.csv"
    raw = write_legacy(f, n=200)
    rec_id = import_recording(proj, f)
    folder = proj.root / "wii" / "recordings" / rec_id
    assert (folder / "original_S01-002.csv").is_file()
    head = next(csv.reader(open(folder / wio.WII_CSV, encoding="utf-8")))
    assert head == wio.WII_HEADER and head[-1] == wio.TIME_LOCAL
    cols = wio.read_wii_csv(folder)
    assert len(cols["t_rel"]) == 200
    np.testing.assert_allclose(cols["total_kg"], raw[:, 7], rtol=1e-5)
    meta = json.loads((folder / wio.SESSION_JSON).read_text(encoding="utf-8"))
    assert meta["clock_source"] == "file_mtime" and meta["clock_estimated"] is True
    assert meta["samples"]["wii"] == 200
    assert load_trial(proj).recording == rec_id
    # a second import of the same name gets its own folder
    assert import_recording(proj, f) == f"{rec_id}_2"


def test_local_time_format():
    t = time.time()
    s = wio.local_time(t)
    assert len(s) >= 29 and s[10] == " " and s[19] == "."  # 2026-09-25 14:03:12.345+08:00
    assert wio.local_time(float("nan")) == "" and wio.local_time(None) == ""


def test_recordings_carry_the_computer_time(tmp_path):
    from poseassess.wii.device import SimulatedBoard
    from poseassess.wii.recorder import WiiRecorder

    sim = SimulatedBoard()
    sim.start()
    try:
        rec = WiiRecorder()
        folder = rec.start(root=tmp_path, force=sim, subject="t")
        time.sleep(0.4)
        rec.add_event("sync")
        time.sleep(0.2)
        rec.stop()
    finally:
        sim.stop()
    rows = list(csv.DictReader(open(folder / wio.WII_CSV, encoding="utf-8")))
    assert len(rows) > 10
    for r in rows[:5] + rows[-5:]:
        shown = r[wio.TIME_LOCAL]
        assert shown == wio.local_time(float(r["t_unix"]))
    ev = list(csv.DictReader(open(folder / wio.EVENTS_CSV, encoding="utf-8-sig")))
    assert ev[0]["label"] == "sync" and ev[0][wio.TIME_LOCAL] == wio.local_time(float(ev[0]["t_unix"]))
