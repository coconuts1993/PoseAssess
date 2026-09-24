"""CORE-BOARD: trc_rows_to_wii_time (recorded via frames.csv, manual, none), auto_align
recovers the synth offset (sync_event and xcorr), fuse_trial on demo_trial (COP world on the
board top, force along up_world, COM near COP, NaN outside the recording), balance_summary,
export_fused_csv / export_grf_mot / export_all, Z-down demo."""

import csv
import json

import numpy as np
import pytest

from poseassess.core.balance import alignment as al
from poseassess.core.balance import fusion
from poseassess.core.balance.board import load_board
from poseassess.core.balance.paths import WiiPaths
from poseassess.core.balance.trc import read_trc_full, zup_to_yup
from poseassess.core.balance.trial import (
    Alignment,
    load_trial,
    save_trial,
    set_recording,
    videos_fingerprint,
)
from poseassess.wii import io as wio
from tests import synth

FRAME = 1 / 30


def write_frames_csv(folder, t_rel, t0=1000.0, off_unix=1.7e9):
    lines = [",".join(wio.FRAMES_HEADER_BASE + ["cam01_src", "cam01_dt_ms"])]
    for i, tr in enumerate(t_rel):
        lines.append(f"{i},{t0 + tr:.6f},{tr:.6f},{t0 + tr + off_unix:.6f},{i},0.0")
    (folder / wio.FRAMES_CSV).write_text("\n".join(lines) + "\n")


def make_recorded(demo, jitter_s=0.004):
    """Turn the demo trial into a captured take: frames.csv (grid time of each exported frame,
    with jitter) + trial.json method "recorded" + the videos fingerprint."""
    proj = demo["project"]
    rec = WiiPaths(proj).recording_dir(demo["rec_id"])
    n = len(demo["trial"]["t_trc"])
    rng = np.random.default_rng(5)
    t_rel = demo["trial"]["t_trc"] + demo["offset_s"] + rng.uniform(-jitter_s, jitter_s, n)
    write_frames_csv(rec, t_rel)
    (proj.videos_dir / "cam01.mp4").write_bytes(b"video")
    a = al.recorded_alignment(proj, demo["rec_id"])
    info = set_recording(proj, demo["rec_id"], "capture", a)
    info.videos_fingerprint = videos_fingerprint(proj)
    save_trial(proj, info)
    return t_rel


# ------------------------------------------------------------------ time mapping
def test_trc_rows_to_wii_time_methods(demo_trial):
    proj = demo_trial["project"]
    frames, times = np.arange(5) + 3, (np.arange(5) + 3) / 30
    np.testing.assert_allclose(al.trc_rows_to_wii_time(proj, frames, times), times + 2.5)
    info = load_trial(proj)
    info.alignment = Alignment("none")
    assert np.isnan(al.trc_rows_to_wii_time(proj, frames, times, info)).all()
    info.alignment, info.recording = Alignment("manual", 1.0), None
    assert np.isnan(al.trc_rows_to_wii_time(proj, frames, times, info)).all()
    # recorded: frames.csv lookup by Frame#, NaN beyond the table
    t_rel = make_recorded(demo_trial)
    got = al.trc_rows_to_wii_time(proj, np.array([0, 10, len(t_rel) - 1, len(t_rel) + 5]),
                                  np.zeros(4))
    np.testing.assert_allclose(got[:3], t_rel[[0, 10, -1]], atol=1e-6)  # %.6f in frames.csv
    assert np.isnan(got[3])
    # inverse mapping (Time = Frame# / rate)
    back = al.wii_time_to_trc_time(proj, t_rel[[0, 10]], rate_hz=30)
    np.testing.assert_allclose(back, [0.0, 10 / 30], atol=1e-5)
    # frames.csv lost: falls back to times + offset_s (= t_rel of frame 0)
    (WiiPaths(proj).recording_dir(demo_trial["rec_id"]) / "frames.csv").unlink()
    np.testing.assert_allclose(al.trc_rows_to_wii_time(proj, frames, times), times + t_rel[0])
    with pytest.raises(ValueError, match="frames.csv"):
        al.recorded_alignment(proj, demo_trial["rec_id"])


def test_recorded_alignment_and_status(demo_trial):
    proj = demo_trial["project"]
    assert al.alignment_status(proj) == ("ok", "Manual offset: +2.500 s")
    t_rel = make_recorded(demo_trial)
    a = load_trial(proj).alignment
    assert a.method == "recorded" and a.offset_s == pytest.approx(t_rel[0])
    assert a.details["n_frames"] == len(t_rel) and a.details["fps"] == pytest.approx(30, rel=0.02)
    assert al.alignment_status(proj) == ("ok", "Recorded with the videos (frame timestamps)")
    (proj.videos_dir / "cam01.mp4").write_bytes(b"replaced video")
    level, text = al.alignment_status(proj)
    assert level == "warn" and "Videos changed" in text
    set_recording(proj, demo_trial["rec_id"])  # external, alignment none
    assert al.alignment_status(proj)[0] == "warn" and "Not aligned" in al.alignment_status(proj)[1]
    set_recording(proj, None)
    assert al.alignment_status(proj) == ("none", "No Wii recording for this trial")
    set_recording(proj, "gone")
    assert al.alignment_status(proj)[0] == "warn"


# ------------------------------------------------------------------ auto_align
@pytest.mark.parametrize("up_sign", [1, -1])
def test_auto_align_sync_event_and_xcorr(tmp_path, up_sign):
    demo = synth.make_demo_trial(tmp_path / "p", up_sign=up_sign, with_images=False)
    proj = demo["project"]
    set_recording(proj, demo["rec_id"])  # external, not aligned yet
    res = al.auto_align(proj)
    assert res.ok and res.alignment.method == "sync_event"
    assert res.alignment.offset_s == pytest.approx(demo["offset_s"], abs=FRAME)
    assert res.alignment.offset_s == pytest.approx(demo["offset_s"], abs=0.005)
    assert "Sync event" in res.message and res.alignment.trc == "pose-3d/demo_filt_butterworth.trc"
    cand = res.alignment.details["candidates"]
    assert cand["sync_event"]["n_matched"] == 3 and cand["xcorr"]["ok"]
    json.dumps(res.alignment.to_dict(), allow_nan=False)  # storable in trial.json
    assert [e["type"] for e in res.trc_events] == ["takeoff", "landing", "stomp"]
    assert {e["type"] for e in res.force_events} >= {"takeoff", "landing", "stomp"}
    sig = res.signals
    assert set(sig) == {"force_t_rel", "force_total_kg", "trc_t", "trc_foot_up_m", "trc_com_up_m"}
    assert len(sig["trc_t"]) == len(sig["trc_com_up_m"]) == len(demo["trial"]["t_trc"])
    assert np.nanmax(sig["trc_com_up_m"]) > 0.8  # heights along the physical up
    # cross-correlation alone (global search)
    xc = al.auto_align(proj, methods=("xcorr",))
    assert xc.ok and xc.alignment.method == "xcorr"
    assert xc.alignment.offset_s == pytest.approx(demo["offset_s"], abs=FRAME)
    # applying stores it; status reports it
    al.apply_alignment(proj, res.alignment)
    info = load_trial(proj)
    assert info.alignment.method == "sync_event" and info.alignment.updated
    level, text = al.alignment_status(proj)
    assert level == "ok" and text.startswith("Sync event: offset +2.50") and "3 events" in text
    al.apply_alignment(proj, xc.alignment)
    assert al.alignment_status(proj)[1].startswith("Cross-correlation: offset +2.5")


def test_auto_align_physical_jump_xcorr_exact(tmp_path):
    demo = synth.make_demo_trial(tmp_path / "p", with_images=False, physical=True,
                                 stomp_at=None, offset_s=-1.75)
    res = al.auto_align(demo["project"], methods=("xcorr",))
    assert res.ok and res.alignment.offset_s == pytest.approx(-1.75, abs=0.005)
    assert res.alignment.details["candidates"]["xcorr"]["score"] > 0.95


def test_auto_align_failures(tmp_path, project):
    assert not al.auto_align(project).ok and "4. Run" in al.auto_align(project).message
    demo = synth.make_demo_trial(tmp_path / "quiet", jump_at=None, stomp_at=None,
                                 with_images=False)
    res = al.auto_align(demo["project"])
    assert not res.ok and res.alignment.method == "none"
    assert "2 small jumps" in res.message
    assert len(res.signals["trc_t"]) > 0  # the plot data is there anyway
    set_recording(demo["project"], None)
    assert "No Wii recording" in al.auto_align(demo["project"]).message
    set_recording(demo["project"], "missing")
    assert not al.auto_align(demo["project"]).ok


def test_auto_align_ignores_the_up_of_a_stale_board(tmp_path):
    """A recalibration that turns the world (Flip Z) leaves board.json stale: its normal is in
    the old world and must not be used for the marker heights (takeoff and landing would be
    swapped: an offset wrong by the flight time, reported as OK)."""
    import cv2

    from poseassess.gui.widgets.wii_alignment import up_vector

    off = 2.501
    demo = synth.make_demo_trial(tmp_path / "p", offset_s=off, physical=True, duration_s=14.0,
                                 jump_at=5.0, stomp_at=9.0, with_images=False)
    proj = demo["project"]
    up, src, note = al.world_up(proj)
    assert src == "board" and not note and up @ [0, 0, 1] > 0.99
    flip = np.diag([-1.0, 1.0, -1.0])  # Flip Z: X' = -X, Z' = -Z
    cams = []
    for c in demo["cams"]:
        R = cv2.Rodrigues(np.asarray(c["rvec"], float))[0] @ flip
        cams.append({**c, "rvec": cv2.Rodrigues(R)[0].ravel()})
    synth.write_calib_toml(proj.calib_toml, cams)
    tr = demo["trial"]
    world = (tr["markers_board"] @ np.asarray(demo["board_R"]).T
             + np.asarray(demo["board_t"])) @ flip.T
    synth.write_trc(proj.pose3d_dir / "demo_filt_butterworth.trc", tr["names"],
                    world[..., [1, 2, 0]], 30)
    assert load_board(proj).stale
    up, src, note = al.world_up(proj)
    np.testing.assert_allclose(up, [0, 0, -1])
    assert src == "calib" and "outdated" in note
    np.testing.assert_allclose(up_vector(proj), up)  # the alignment plot uses the same rule
    res = al.auto_align(proj)
    assert res.ok and res.alignment.method == "sync_event"
    assert res.alignment.offset_s == pytest.approx(off, abs=0.01)
    assert res.alignment.details["up_source"] == "calib"
    assert "outdated" in res.alignment.details["up_note"] and "outdated" in res.message


@pytest.mark.parametrize("latency", [0.0, 0.08])
def test_check_recorded_alignment_measures_the_camera_latency(tmp_path, latency):
    """Frames stamped when read() returns are ``latency`` late: a jump / stomp in the take
    measures it (and suggests the capture latency to set before exporting again)."""
    demo = synth.make_demo_trial(tmp_path / "p", physical=True, duration_s=14.0, jump_at=5.0,
                                 stomp_at=9.0, with_images=False)
    proj = demo["project"]
    assert not al.check_recorded_alignment(proj)["ok"]  # a manual offset: nothing to check
    rec = WiiPaths(proj).recording_dir(demo["rec_id"])
    write_frames_csv(rec, demo["trial"]["t_trc"] + demo["offset_s"] + latency)
    (proj.videos_dir / "cam01.mp4").write_bytes(b"video")
    info = set_recording(proj, demo["rec_id"], "capture",
                         al.recorded_alignment(proj, demo["rec_id"]))
    info.videos_fingerprint = videos_fingerprint(proj)
    save_trial(proj, info)
    chk = al.check_recorded_alignment(proj)
    assert chk["ok"] and chk["result"].ok
    assert chk["latency_ms"] == pytest.approx(latency * 1000, abs=5)
    assert chk["frame_ms"] == pytest.approx(1000 / 30, rel=0.01)
    assert chk["agrees"] is (latency == 0.0)
    if latency:
        assert chk["suggested_latency_ms"] == pytest.approx(80, abs=5)
        assert "older than their" in chk["message"] and "camera latency" in chk["message"]
    else:
        assert chk["suggested_latency_ms"] is None and "agrees" in chk["message"]


def test_set_manual_offset(demo_trial):
    proj = demo_trial["project"]
    info = al.set_manual_offset(proj, -0.25, demo_trial["trc"])
    assert info.alignment.method == "manual" and info.alignment.offset_s == -0.25
    assert load_trial(proj).alignment.trc == "pose-3d/demo_filt_butterworth.trc"
    assert al.alignment_status(proj) == ("ok", "Manual offset: -0.250 s")


# ------------------------------------------------------------------ fusion
def test_cop_to_world_and_plumb(demo_trial):
    reg = load_board(demo_trial["project"])
    R, t = demo_trial["board_R"], demo_trial["board_t"]
    np.testing.assert_allclose(fusion.cop_to_world(reg, [0.1, -0.05]), R @ [0.1, -0.05, 0] + t)
    many = fusion.cop_to_world(reg, np.array([[0.0, 0.0], [np.nan, np.nan]]))
    np.testing.assert_allclose(many[0], t)
    assert np.isnan(many[1]).all()
    com = R @ [0.02, 0.03, 0.9] + t
    np.testing.assert_allclose(fusion.plumb_point(reg, com), R @ [0.02, 0.03, 0] + t, atol=1e-12)
    np.testing.assert_allclose(fusion.plumb_point(reg, np.vstack([com, com]))[1],
                               R @ [0.02, 0.03, 0] + t, atol=1e-12)


@pytest.mark.parametrize("up_sign", [1, -1])
def test_fuse_trial_matches_ground_truth(tmp_path, up_sign):
    demo = synth.make_demo_trial(tmp_path / "p", up_sign=up_sign, with_images=False)
    f = fusion.fuse_trial(demo["project"])
    ref = synth.fused_from_demo(demo)
    assert f.warnings == [] and f.recording == demo["rec_id"] and f.body_mass_kg == pytest.approx(70)
    assert f.n_frames == ref.n_frames and f.trc_path == demo["trc"]
    np.testing.assert_array_equal(f.frames, ref.frames)
    np.testing.assert_allclose(f.t_rel, ref.t_rel, atol=1e-5)
    np.testing.assert_allclose(f.t_unix, ref.t_unix, atol=1e-3)
    np.testing.assert_allclose(f.total_kg, ref.total_kg, atol=0.01)
    np.testing.assert_allclose(f.kg, ref.kg, atol=2e-3)  # %.6f CSV + interpolation
    for k in ("cop_board", "cop_world", "com_world", "com_board"):
        np.testing.assert_allclose(getattr(f, k), getattr(ref, k), atol=1e-4, equal_nan=True)
    np.testing.assert_allclose(f.force_world, ref.force_world, atol=0.05)
    ok = np.isfinite(f.cop_world).all(axis=1)
    # COP on the board top (along the physical up); force along up_world; COM above, near COP
    up = f.board.up_world
    np.testing.assert_allclose((f.cop_world[ok] - demo["board_t"]) @ up, 0, atol=1e-9)
    np.testing.assert_allclose(f.force_world @ up, f.total_kg * fusion.G, rtol=1e-9)
    assert np.all(f.com_board[:, 2] > 0.8)
    assert np.nanmedian(np.linalg.norm(f.com_minus_cop, axis=1)) < 0.05
    # flight: nobody on the board -> no COP, zero force
    fl = (f.trc_time > 6.02) & (f.trc_time < 6.28)
    assert fl.any() and np.isnan(f.cop_board[fl]).all() and np.all(f.total_kg[fl] < 1)
    # full-rate data and marked events on the .trc time base
    np.testing.assert_allclose(f.wii["trc_time"], f.wii["t_rel"] - 2.5, atol=1e-9)
    assert [e["label"] for e in f.events] == ["sync"]
    assert f.events[0]["trc_time"] == pytest.approx(6.0)
    assert f.index_for_time(f.trc_time[42]) == 42


def test_fuse_trial_warnings_and_errors(tmp_path, project):
    with pytest.raises(ValueError, match="4. Run"):
        fusion.fuse_trial(project)
    demo = synth.make_demo_trial(tmp_path / "p", with_images=False)
    proj = demo["project"]
    # the Wii recording starts 3 s after the video: the first 3 s have no force data
    al.set_manual_offset(proj, -3.0)
    f = fusion.fuse_trial(proj)
    early = f.trc_time < 2.99
    assert np.isnan(f.total_kg[early]).all() and np.isfinite(f.total_kg[~early]).all()
    assert np.isfinite(f.com_world[early]).all()  # the COM does not need the Wii
    assert any("covers only" in w for w in f.warnings)
    # not aligned: no Wii rows, COM kept, a warning (no exception)
    set_recording(proj, demo["rec_id"])
    f = fusion.fuse_trial(proj)
    assert np.isnan(f.total_kg).all() and np.isfinite(f.com_world).all()
    assert any("not aligned" in w for w in f.warnings)
    assert np.isnan(f.wii["trc_time"]).all() and np.isnan(f.events[0]["trc_time"])
    # no board: board-frame only
    al.set_manual_offset(proj, -3.0)
    WiiPaths(proj).board_json.unlink()
    f = fusion.fuse_trial(proj)
    assert f.board is None and np.isnan(f.cop_world).all() and np.isnan(f.com_board).all()
    assert np.isfinite(f.cop_board).any() and any("2b. Wii Board" in w for w in f.warnings)
    # no recording
    set_recording(proj, None)
    with pytest.raises(ValueError, match="No Wii recording"):
        fusion.fuse_trial(proj)
    set_recording(proj, "missing")
    with pytest.raises(ValueError, match="missing"):
        fusion.fuse_trial(proj)


def test_fuse_trial_stale_board_and_recorded(demo_trial):
    proj = demo_trial["project"]
    t_rel = make_recorded(demo_trial, jitter_s=0.006)
    f = fusion.fuse_trial(proj)
    np.testing.assert_allclose(f.t_rel, t_rel, atol=1e-6)  # exact frame timestamps
    assert f.alignment.method == "recorded" and f.warnings == []
    # recalibrated -> stale board warning; replaced videos -> warning
    cams = synth.ring_cameras(3)
    cams[1]["tvec"] = np.asarray(cams[1]["tvec"]) + [0.002, 0, 0]
    synth.write_calib_toml(proj.calib_toml, cams)
    (proj.videos_dir / "cam01.mp4").write_bytes(b"other")
    w = fusion.fuse_trial(proj).warnings
    assert any("calibration changed" in x for x in w) and any("videos changed" in x for x in w)


def test_fuse_trial_warns_when_pose2sim_uses_another_calibration(demo_trial):
    import time

    from poseassess.core.balance.calib import pose2sim_calib_file

    proj = demo_trial["project"]
    assert not any("Pose2Sim" in x for x in fusion.fuse_trial(proj).warnings)
    time.sleep(0.02)
    cams = [dict(c) for c in demo_trial["cams"]]
    cams[0]["tvec"] = np.asarray(cams[0]["tvec"]) + [0.1, 0, 0]
    other = proj.calibration_dir / "Calib_easymocap.toml"
    synth.write_calib_toml(other, cams)
    if pose2sim_calib_file(proj) != other:
        pytest.skip("cannot make Calib_easymocap.toml the newest file here")
    w = fusion.fuse_trial(proj).warnings
    assert any("Pose2Sim will triangulate with calibration/Calib_easymocap.toml" in x
               for x in w)


# ------------------------------------------------------------------ summary + export
def test_balance_summary(demo_trial):
    f = synth.fused_from_demo(demo_trial)
    s = fusion.balance_summary(f)
    json.dumps(s, allow_nan=False)
    assert s["frames"] == f.n_frames and s["window"] == [0.0, pytest.approx(f.trc_time[-1])]
    assert s["body_mass_kg"] == 70.0 and s["recording"] == demo_trial["rec_id"]
    assert s["cop"]["samples"] > 2 * s["com"]["samples"]  # full-rate COP (100 Hz vs 30 fps)
    assert 5 < s["cop"]["rms_ml_mm"] < 20 and 5 < s["com"]["rms_ml_mm"] < 20
    cc = s["com_cop"]
    assert cc["rms_distance_mm"] < 20 and cc["corr_ml"] > 0.9 and cc["corr_ap"] > 0.9
    kinds = [(e["type"], round(e["trc_time"], 2)) for e in s["force_events"]]
    assert ("takeoff", 6.0) in kinds and ("landing", 6.3) in kinds
    assert [e["label"] for e in s["events"]] == ["sync"]
    # a window: 1..5 s (quiet standing only)
    w = fusion.balance_summary(f, 1.0, 5.0)
    assert w["window"] == [1.0, 5.0] and w["duration_s"] == 4.0
    assert w["force_events"] == [] and w["events"] == []
    assert w["cop"]["duration_s"] == pytest.approx(4.0, abs=0.02)
    assert w["mean_total_kg"] == pytest.approx(70.0, abs=0.1)
    # without full-rate data: per-frame COP
    f.wii = {}
    assert fusion.balance_summary(f)["cop"]["samples"] <= f.n_frames


def test_exports(demo_trial, tmp_path):
    proj = demo_trial["project"]
    f = fusion.fuse_trial(proj)
    paths = fusion.export_all(proj, f, t_from=1.0, t_to=11.0)
    assert set(paths) == {"fused", "summary", "grf"}
    assert all(p.parent == WiiPaths(proj).exports_dir for p in paths.values())
    assert paths["fused"].name == "demo_filt_butterworth_fused.csv"
    with open(paths["fused"], newline="") as fh:
        rows = list(csv.reader(fh))
    assert rows[0] == fusion.FUSED_COLUMNS and len(rows) == f.n_frames + 1
    assert rows[1][0] == "0" and rows[1][1] == "0.000000"
    fl = [r for r in rows[1:] if 6.05 < float(r[1]) < 6.25]
    assert fl and all(r[fusion.FUSED_COLUMNS.index("cop_x_board")] == "" for r in fl)
    tab = wio.read_csv_columns(paths["fused"])
    np.testing.assert_allclose(tab["com_z_world"], f.com_world[:, 2], atol=1e-6)
    summary = json.loads(paths["summary"].read_text())
    assert summary["window"] == [1.0, 11.0] and summary["cop"]["samples"] > 500
    # .mot: OpenSim external loads in the Y-up .trc frame
    lines = paths["grf"].read_text().splitlines()
    assert lines[1] == "version=1" and lines[2] == f"nRows={f.n_frames}"
    assert "endheader" in lines and lines[6].split("\t")[0] == "time"
    mot = np.array([[float(v) for v in ln.split("\t")] for ln in lines[7:]])
    np.testing.assert_allclose(mot[:, 0], f.trc_time, atol=1e-6)
    ok = np.isfinite(f.cop_world).all(axis=1)
    np.testing.assert_allclose(mot[ok, 1:4], zup_to_yup(f.force_world[ok]), atol=1e-5)
    np.testing.assert_allclose(mot[ok, 4:7], zup_to_yup(f.cop_world[ok]), atol=1e-6)
    assert np.median(mot[ok, 2]) == pytest.approx(70 * fusion.G, rel=0.01)  # vertical = Y
    np.testing.assert_allclose(mot[~ok, 1:4], 0)  # nobody on the board: zero force...
    np.testing.assert_allclose(mot[~ok, 4:7] - zup_to_yup(f.board.center_world), 0, atol=1e-6)
    np.testing.assert_allclose(mot[:, 7:], 0)
    # the .trc and the .mot share the time column
    np.testing.assert_allclose(read_trc_full(demo_trial["trc"]).times, mot[:, 0], atol=1e-6)
    # no board: no .mot, the summary says why
    WiiPaths(proj).board_json.unlink()
    f2 = fusion.fuse_trial(proj)
    with pytest.raises(ValueError):
        fusion.export_grf_mot(f2, tmp_path / "x.mot")
    p2 = fusion.export_all(proj, f2, out_dir=tmp_path / "out")
    assert set(p2) == {"fused", "summary"} and p2["fused"].parent == tmp_path / "out"
    assert any("No .mot" in w for w in json.loads(p2["summary"].read_text())["warnings"])
