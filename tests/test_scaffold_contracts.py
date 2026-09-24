"""Contract tests for the parts of the Wii integration that the scaffold already implements
(data formats shared by all agents): recording readers, trial.json, board.json, capture.json,
Calib.toml loading and the Z-up / Y-up conversion. Owner: CORE-BOARD (keep them passing)."""

import json

import numpy as np
import pytest

from poseassess.core.balance import paths as bp
from poseassess.core.balance.board import BoardRegistration, CameraClicks
from poseassess.core.balance.calib import (
    find_calib_toml,
    load_calib_toml,
    project_cameras,
    world_up_sign,
)
from poseassess.core.balance.trc import yup_to_zup, zup_to_yup
from poseassess.core.balance.trial import (
    Alignment,
    TrialInfo,
    load_trial,
    save_trial,
    set_recording,
    videos_changed,
    videos_fingerprint,
)
from poseassess.core.capture.settings import (
    CameraSetting,
    load_capture_settings,
    save_capture_settings,
)
from poseassess.wii import io as wio
from tests import synth


def test_demo_trial_files(demo_trial):
    proj = demo_trial["project"]
    paths = bp.WiiPaths(proj)
    assert paths.list_recordings() == [demo_trial["rec_id"]]
    rec = paths.recording_dir(demo_trial["rec_id"])
    wii = wio.read_wii_csv(rec)
    assert set(wio.WII_HEADER) <= set(wii)
    assert np.all(np.diff(wii["t_rel"]) > 0)
    assert np.nanmax(wii["total_kg"]) > 100  # landing peak
    ev = wio.read_events_csv(rec)
    assert ev and ev[0]["label"] == "sync"
    assert wio.read_session_json(rec)["schema"] == wio.SESSION_SCHEMA
    info = load_trial(proj)
    assert info.recording == demo_trial["rec_id"]
    assert info.alignment.method == "manual"
    assert info.alignment.offset_s == pytest.approx(demo_trial["offset_s"])
    reg = BoardRegistration.from_dict(json.loads(paths.board_json.read_text()))
    assert np.allclose(reg.pose.board_to_world.R, demo_trial["board_R"])
    assert np.allclose(reg.up_world, [0, 0, 1])
    assert reg.cameras_used == ["cam01", "cam02", "cam03"]
    assert reg.clicks["cam01"].complete


def test_trial_roundtrip_and_fingerprint(project):
    assert load_trial(project) == TrialInfo()
    info = TrialInfo("rec1", "capture", Alignment("recorded", 1.25, "frames.csv"),
                     {"cam01.mp4": [1, 2]}, 71.5, "n")
    save_trial(project, info)
    assert load_trial(project) == info
    assert videos_changed(project, info)  # videos/ is empty now
    info.videos_fingerprint = videos_fingerprint(project)
    assert not videos_changed(project, info)
    info2 = set_recording(project, "rec2")
    assert info2.recording == "rec2" and info2.alignment.method == "none"
    assert info2.videos_fingerprint == {}


def test_board_registration_roundtrip(demo_trial):
    reg = BoardRegistration.from_dict(
        json.loads(bp.WiiPaths(demo_trial["project"]).board_json.read_text()))
    d = reg.to_dict()
    reg2 = BoardRegistration.from_dict(json.loads(json.dumps(d)))
    assert np.allclose(reg2.corners_world, reg.corners_world)
    assert np.allclose(reg2.sensors_world, reg.sensors_world)
    # corners are the projected landmarks of the synthetic scene
    lm = synth.board_landmarks()[:4] @ demo_trial["board_R"].T + demo_trial["board_t"]
    assert np.allclose(reg.corners_world, lm)
    c = CameraClicks("cam01", [(0, 0), (1, 0), (1, 1), (0, 1), (0.5, 0.5)])
    s = c.swapped_front_back()
    assert s.points == [(1, 1), (0, 1), (0, 0), (1, 0), (0.5, 0.5)]
    assert CameraClicks.from_dict("cam01", c.to_dict()) == c


@pytest.mark.parametrize("up_sign", [1, -1])
def test_calib_loading_and_up_sign(tmp_path, up_sign):
    cams = synth.ring_cameras(3, up_sign=up_sign)
    p = synth.write_calib_toml(tmp_path / "calibration" / "Calib.toml", cams)
    loaded = load_calib_toml(p)
    assert [c.name for c in loaded] == ["1", "2", "3"]
    assert loaded[0].image_size == (1280, 720)
    assert np.allclose(loaded[1].center_world, cams[1]["center"])
    assert world_up_sign(loaded) == up_sign
    assert find_calib_toml(tmp_path) == p
    assert list(project_cameras(tmp_path)) == ["cam01", "cam02", "cam03"]


def test_capture_settings_roundtrip(project):
    s = load_capture_settings(project)
    assert list(s.cameras) == ["cam01", "cam02", "cam03"]
    assert s.cameras["cam02"].source == 1
    s.cameras["cam01"] = CameraSetting("video.avi", 640, 480, 30.0)
    s.output_fps = 25
    save_capture_settings(project, s)
    s2 = load_capture_settings(project)
    assert s2.cameras["cam01"].source == "video.avi" and s2.output_fps == 25


def test_frames_conversion():
    p = np.random.default_rng(0).normal(size=(5, 3))
    assert np.allclose(yup_to_zup(zup_to_yup(p)), p)
    assert np.allclose(zup_to_yup(np.array([1.0, 2.0, 3.0])), [2.0, 3.0, 1.0])


def test_cam_names():
    assert bp.cam_name(3) == "cam03" and bp.cam_index("cam12") == 12
    with pytest.raises(ValueError):
        bp.cam_index("1")


def test_fused_from_demo_contract(demo_trial):
    fused = synth.fused_from_demo(demo_trial)
    n = fused.n_frames
    assert n == len(demo_trial["trial"]["t_trc"])
    tab = fused.table()
    assert list(tab) == __import__("poseassess.core.balance.fusion", fromlist=["x"]).FUSED_COLUMNS
    assert all(len(v) == n for v in tab.values())
    i = fused.index_for_time(fused.trc_time[10] + 0.001)
    assert i == 10 and fused.index_for_time(1e6) == -1
    ok = np.isfinite(fused.cop_world).all(axis=1)
    # COP lies on the board top surface, COM is above it and close to the COP horizontally
    assert np.allclose(fused.cop_world[ok][:, 2], demo_trial["board_t"][2])
    d = np.linalg.norm(fused.com_minus_cop[ok], axis=1)
    assert np.nanmedian(d) < 0.05
    assert fused.board is not None and fused.board.stale is None


def test_demo_trial_z_down(tmp_path):
    from poseassess.core.balance.board import load_board

    demo = synth.make_demo_trial(tmp_path / "zdown", up_sign=-1, with_images=False)
    reg = load_board(demo["project"])
    assert np.allclose(reg.up_world, [0, 0, -1]) and reg.world_up_sign == -1
    assert np.linalg.det(reg.pose.board_to_world.R) == pytest.approx(1.0)
    fused = synth.fused_from_demo(demo)
    ok = np.isfinite(fused.com_world).all(axis=1)
    # the person stands "above" the board along the physical up (-Z)
    assert np.all(fused.com_world[ok][:, 2] < demo["board_t"][2] - 0.5)
