"""CORE-BOARD: board.py project I/O (save/load clicks, compute_board from synth clicks for
up_sign +1/-1 and style poseassess/opencap Calib.toml, image-size mismatch skipped, stale after
Calib.toml change, swap_front_back_clicks + recompute == rotate 180, set_corner_check), and
trial.import_recording."""

import json
import shutil

import numpy as np
import pytest

from poseassess.core.balance import board as bb
from poseassess.core.balance.board import (
    CameraClicks,
    compute_board,
    load_all_clicks,
    load_board,
    load_clicks,
    save_clicks,
    set_corner_check,
    swap_front_back_clicks,
)
from poseassess.core.balance.geometry import BoardGeometry, rotate_board_frame
from poseassess.core.balance.paths import WiiPaths
from poseassess.core.balance.trial import import_recording, load_trial
from poseassess.core.calibration.extrinsic import load_clicked_points
from poseassess.wii import io as wio
from tests import synth


def scene(project, up_sign=1, style="poseassess", n=3, noise_px=0.0, seed=0, save=True):
    """Calib.toml + clicks (points.json) of a synthetic scene in ``project``."""
    cams = synth.ring_cameras(n, up_sign=up_sign)
    synth.write_calib_toml(project.calib_toml, cams, style=style)
    R, t = synth.board_pose(up_sign=up_sign)
    clicks = synth.board_clicks(cams, R, t, noise_px=noise_px, seed=seed)
    if save:
        for i, (cam, px) in enumerate(zip(cams, clicks), start=1):
            save_clicks(project, i, CameraClicks(f"cam{i:02d}", [tuple(p) for p in px],
                                                 "board.jpg", tuple(cam["size"]), "synthetic"))
    return cams, R, t, clicks


def test_clicks_io(project):
    assert load_clicks(project, 1) is None and load_all_clicks(project) == {}
    c = CameraClicks("cam02", [(1.5, 2.5), (3, 4)], "000123.jpg", (1920, 1080),
                     "videos/cam02.mp4#frame=123")
    f = save_clicks(project, 2, c)
    assert f == WiiPaths(project).board_points_file(2) and f.is_file()
    assert load_clicks(project, 2) == c
    d = json.loads(f.read_text())
    assert d["labels"] == ["TL", "TR"] and d["image_size"] == [1920, 1080]
    # the existing calibration reader understands it (compatible "points" key)
    np.testing.assert_allclose(load_clicked_points(f), [[1.5, 2.5], [3, 4]])
    (WiiPaths(project).board_dir / "notacam").mkdir()
    (WiiPaths(project).board_cam_dir(3)).mkdir()  # a folder without points.json
    WiiPaths(project).board_points_file(1).parent.mkdir(parents=True, exist_ok=True)
    WiiPaths(project).board_points_file(1).write_text("{broken")
    assert list(load_all_clicks(project)) == ["cam02"]
    assert not c.complete and c.as_array() is None


@pytest.mark.parametrize("style", ["poseassess", "opencap"])
@pytest.mark.parametrize("up_sign", [1, -1])
def test_compute_board_matches_truth(project, up_sign, style):
    cams, R, t, _ = scene(project, up_sign, style)
    reg = compute_board(project)
    assert reg.pose.method == "triangulation" and reg.cameras_used == ["cam01", "cam02", "cam03"]
    np.testing.assert_allclose(reg.pose.board_to_world.R, R, atol=1e-6)
    np.testing.assert_allclose(reg.pose.board_to_world.t, t, atol=1e-6)
    np.testing.assert_allclose(reg.up_world, [0, 0, up_sign], atol=1e-9)
    assert reg.world_up_sign == up_sign and reg.stale is None
    assert max(reg.pose.reproj_error_px.values()) < 1e-6
    # saved + reloaded; the derived world arrays are written for other tools
    back = load_board(project)
    assert back is not None and back.stale is None
    np.testing.assert_allclose(back.corners_world, synth.board_landmarks()[:4] @ R.T + t, atol=1e-6)
    d = json.loads(WiiPaths(project).board_json.read_text())
    assert d["schema"] == bb.BOARD_SCHEMA and d["world_up_sign"] == up_sign
    assert d["calib"]["file"] == "Calib.toml" and len(d["calib"]["sha1"]) == 40
    assert set(d["clicks"]) == {"cam01", "cam02", "cam03"} and d["corner_check"] is None
    np.testing.assert_allclose(d["sensors_world"], back.sensors_world)
    # a single camera: PnP, fitted flat on the floor
    for i in (2, 3):
        WiiPaths(project).board_points_file(i).unlink()
    one = compute_board(project, save=False)
    assert one.pose.method == "pnp" and one.cameras_used == ["cam01"]
    np.testing.assert_allclose(one.pose.board_to_world.t, t, atol=1e-6)
    assert load_board(project).cameras_used == ["cam01", "cam02", "cam03"]  # save=False


@pytest.mark.parametrize("up_sign", [1, -1])
def test_noisy_clicks_centimetre_accuracy(project, up_sign):
    cams, R, t, _ = scene(project, up_sign, noise_px=1.5, seed=7)
    reg = compute_board(project)
    truth = synth.board_landmarks() @ R.T + t
    est = reg.pose.board_to_world.apply(BoardGeometry().landmarks())
    assert np.linalg.norm(est - truth, axis=1).max() < 0.01
    assert reg.pose.floor_constrained and np.allclose(reg.up_world, [0, 0, up_sign])
    assert all(e < 5 for e in reg.pose.reproj_error_px.values())


def test_compute_board_errors_and_skipped_cameras(project):
    with pytest.raises(ValueError, match="2. Calibration"):
        compute_board(project)
    cams, R, t, clicks = scene(project, save=False)
    with pytest.raises(ValueError, match="5 board points"):
        compute_board(project)
    # cam01 clicked on an image of another resolution: not used (note); cam02 incomplete
    save_clicks(project, 1, CameraClicks("cam01", [tuple(p) for p in clicks[0] / 2], "half.jpg",
                                         (640, 360)))
    save_clicks(project, 2, CameraClicks("cam02", [tuple(p) for p in clicks[1][:3]], "x.jpg",
                                         (1280, 720)))
    with pytest.raises(ValueError, match="640x360"):
        compute_board(project)
    save_clicks(project, 3, CameraClicks("cam03", [tuple(p) for p in clicks[2]], "b.jpg", None))
    reg = compute_board(project)
    assert reg.cameras_used == ["cam03"] and any("cam01" in n and "640x360" in n
                                                 for n in reg.pose.notes)
    np.testing.assert_allclose(reg.pose.board_to_world.t, t, atol=1e-6)
    # clicks of a camera that is not in Calib.toml
    save_clicks(project, 5, CameraClicks("cam05", [tuple(p) for p in clicks[2]], "b.jpg"))
    assert any("cam05" in n for n in compute_board(project).pose.notes)
    # mirrored click order in every camera
    for i, px in enumerate(clicks, start=1):
        save_clicks(project, i, CameraClicks(f"cam{i:02d}", [tuple(px[j]) for j in (1, 0, 3, 2, 4)],
                                             "b.jpg", (1280, 720)))
    with pytest.raises(ValueError, match="mirrored"):
        compute_board(project)


def test_approximate_calib_size_is_accepted(project):
    """Converters (e.g. Pose2Sim's easymocap one) write ``size = [2 cx, 2 cy]``: clicks on the
    real 1280x720 frames must still be used; a clearly different resolution is not."""
    import tomli_w

    from poseassess.core.balance.calib import load_calib_toml

    cams, R, t, _ = scene(project)
    for c in cams:
        c["K"] = np.asarray(c["K"], float).copy()
        c["K"][:2, 2] += (-0.3, 1.8)  # principal point slightly off-centre
    synth.write_calib_toml(project.calib_toml, cams)
    import tomllib
    data = tomllib.loads(project.calib_toml.read_text())
    for d in data.values():
        if isinstance(d, dict) and "matrix" in d:
            d["size"] = [2 * d["matrix"][0][2], 2 * d["matrix"][1][2]]  # 1279.4 x 723.6
    project.calib_toml.write_bytes(tomli_w.dumps(data).encode())
    assert load_calib_toml(project.calib_toml)[0].image_size == (1279, 724)
    for i, px in enumerate(synth.board_clicks(cams, R, t), start=1):
        save_clicks(project, i, CameraClicks(f"cam{i:02d}", [tuple(p) for p in px], "b.jpg",
                                             (1280, 720)))
    reg = compute_board(project, save=False)
    assert reg.cameras_used == ["cam01", "cam02", "cam03"]
    assert any("approximate" in n and "1279x724" in n for n in reg.pose.notes)
    np.testing.assert_allclose(reg.pose.board_to_world.t, t, atol=1e-3)
    assert bb._size_compatible((1920, 1080), (1919, 1084))
    assert not bb._size_compatible((640, 360), (1280, 720))
    assert not bb._size_compatible((1280, 960), (1280, 720))


def test_pose2sim_calibration_file_mismatch(project):
    """Pose2Sim triangulates with the newest calibration/*.toml; when that is not the file the
    board was located in and the cameras differ, compute_board, load_board and fuse_trial say
    so (board.json records both files)."""
    import time

    from poseassess.core.balance.calib import calib_mismatch, pose2sim_calib_file

    cams, *_ = scene(project)
    assert pose2sim_calib_file(project) == project.calib_toml and calib_mismatch(project) is None
    reg = compute_board(project)
    d = json.loads(WiiPaths(project).board_json.read_text())
    assert d["calib"]["pose2sim_file"] == "Calib.toml"
    assert d["calib"]["pose2sim_sha1"] == d["calib"]["sha1"]
    assert not any("Pose2Sim" in w for w in reg.pose.warnings)
    # the same cameras in a newer file (e.g. Pose2Sim's calibration stage): no mismatch
    time.sleep(0.02)
    other = project.calibration_dir / "Calib_easymocap.toml"
    synth.write_calib_toml(other, cams)
    if pose2sim_calib_file(project) != other:  # coarse ctime: cannot order the files here
        pytest.skip("cannot make Calib_easymocap.toml the newest file here")
    assert calib_mismatch(project) is None and load_board(project).stale is None
    # different camera positions (e.g. Flip Z applied to Calib.toml only)
    moved = [dict(c) for c in cams]
    moved[0] = {**moved[0], "tvec": np.asarray(cams[0]["tvec"]) + [0.05, 0, 0]}
    synth.write_calib_toml(other, moved)
    if pose2sim_calib_file(project) != other:
        pytest.skip("rewriting changed the file order")
    msg = calib_mismatch(project)
    assert msg and "Calib_easymocap.toml" in msg and "Calib.toml" in msg
    assert "Pose2Sim uses (Calib_easymocap.toml)" in load_board(project).stale
    reg = compute_board(project)
    assert msg in reg.pose.warnings and load_board(project).stale is None
    assert json.loads(WiiPaths(project).board_json.read_text())["calib"]["pose2sim_file"] \
        == "Calib_easymocap.toml"


def test_swap_front_back_equals_180_rotation(project):
    scene(project, noise_px=0.8, seed=2)
    reg = compute_board(project)
    assert swap_front_back_clicks(project) == ["cam01", "cam02", "cam03"]
    swapped = compute_board(project)
    rot = rotate_board_frame(reg.pose, 180)
    np.testing.assert_allclose(swapped.pose.board_to_world.R, rot.board_to_world.R, atol=1e-9)
    np.testing.assert_allclose(swapped.pose.board_to_world.t, rot.board_to_world.t, atol=1e-9)
    np.testing.assert_allclose(swapped.corners_world, reg.corners_world[[2, 3, 0, 1]], atol=1e-9)
    # swapping twice restores the clicks
    swap_front_back_clicks(project)
    np.testing.assert_allclose(compute_board(project).pose.board_to_world.R,
                               reg.pose.board_to_world.R, atol=1e-12)


def test_stale_corner_check_and_geometry(project):
    cams, R, t, _ = scene(project)
    with pytest.raises(ValueError):
        set_corner_check(project, "ok", "TL")  # nothing registered yet
    geo = BoardGeometry(height_mm=40.0, sensor_dx_mm=430.0)
    compute_board(project, geometry=geo)
    with pytest.raises(ValueError):
        set_corner_check(project, "maybe", None)
    reg = set_corner_check(project, "swapped", "BR")
    assert reg.corner_check["result"] == "swapped" and load_board(project).corner_check["pressed"] == "BR"
    # recomputing keeps the geometry of board.json; the corner check only on request
    reg = compute_board(project, keep_corner_check=True)
    assert reg.geometry == geo and reg.corner_check["result"] == "swapped"
    assert compute_board(project).corner_check is None
    # recalibration (e.g. Flip Z) -> stale; missing calibration -> stale
    cams[0]["tvec"] = np.asarray(cams[0]["tvec"]) + [0.001, 0, 0]
    synth.write_calib_toml(project.calib_toml, cams)
    assert "recompute" in load_board(project).stale
    project.calib_toml.unlink()
    assert "missing" in load_board(project).stale
    WiiPaths(project).board_json.write_text("not json")
    assert load_board(project) is None


# --------------------------------------------------------------------- import_recording
def test_import_recording_folder_and_file(project, demo_trial, tmp_path):
    src = WiiPaths(demo_trial["project"]).recording_dir(demo_trial["rec_id"])
    (src / "cam01.mkv").write_bytes(b"not copied")
    rid = import_recording(project, src)
    assert rid == demo_trial["rec_id"]
    dst = WiiPaths(project).recording_dir(rid)
    assert sorted(p.name for p in dst.iterdir()) == ["events.csv", "session.json", "wii.csv"]
    meta = wio.read_session_json(dst)
    assert meta["imported_from"] == str(src.resolve()) and meta["schema"] == wio.SESSION_SCHEMA
    info = load_trial(project)
    assert info.recording == rid and info.source == "external"
    assert info.alignment.method == "none"
    assert wio.read_events_csv(dst)[0]["label"] == "sync"
    # same name again -> _2; not made active
    rid2 = import_recording(project, src, make_active=False)
    assert rid2 == f"{rid}_2" and load_trial(project).recording == rid
    # a bare wii.csv with only t + sensors: rewritten with t_rel / total / COP
    wii = wio.read_wii_csv(src)
    bare = tmp_path / "ext" / "wii.csv"
    bare.parent.mkdir()
    lines = ["t,TR_kg,BR_kg,TL_kg,BL_kg"] + [
        ",".join(f"{v:.6f}" for v in row)
        for row in zip(wii["t"], wii["TR_kg"], wii["BR_kg"], wii["TL_kg"], wii["BL_kg"])]
    bare.write_text("\n".join(lines) + "\n")
    rid3 = import_recording(project, bare, name="my take/1")
    assert rid3 == "my_take_1"
    got = wio.read_wii_csv(WiiPaths(project).recording_dir(rid3))
    np.testing.assert_allclose(got["t_rel"], wii["t"] - wii["t"][0], atol=1e-6)
    np.testing.assert_allclose(got["total_kg"], wii["total_kg"], atol=1e-5)
    np.testing.assert_allclose(got["cop_x_board"], wii["cop_x_board"], atol=1e-5)
    meta3 = wio.read_session_json(WiiPaths(project).recording_dir(rid3))
    assert meta3["app"] == "imported" and meta3["samples"]["wii"] == len(wii["t"])
    assert WiiPaths(project).list_recordings() == sorted([rid, rid2, rid3])
    # another file name: its stem
    other = tmp_path / "ext" / "session_7.csv"
    shutil.copy(src / "wii.csv", other)
    assert import_recording(project, other) == "session_7"
    # an existing recording of this project is only made active
    assert import_recording(project, WiiPaths(project).recording_dir(rid2)) == rid2
    assert load_trial(project).recording == rid2
    assert len(WiiPaths(project).list_recordings()) == 4


def test_import_recording_rejects_bad_sources(project, tmp_path):
    with pytest.raises(ValueError, match="does not exist"):
        import_recording(project, tmp_path / "nope")
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError, match="wii.csv"):
        import_recording(project, tmp_path / "empty")
    bad = tmp_path / "bad.csv"
    bad.write_text("a,b\n1,2\n3,4\n")
    with pytest.raises(ValueError, match="time column"):
        import_recording(project, bad)
    bad.write_text("t_rel,foo\n1,2\n3,4\n")
    with pytest.raises(ValueError, match="force columns"):
        import_recording(project, bad)
    bad.write_text("t_rel,total_kg\n1,2\n")
    with pytest.raises(ValueError, match="fewer than 2"):
        import_recording(project, bad)
    assert WiiPaths(project).list_recordings() == []
    assert load_trial(project).recording is None
