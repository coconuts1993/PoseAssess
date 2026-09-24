"""Integration: the whole Wii Balance Board workflow end to end through the real MainWindow
(``tests/e2e_scenario.py``): plug-and-play connection of a scripted HID board (tare, drop and
reconnect), board snapshots + clicks + registration + live corner check (front/back swap fixed),
a camera + Wii take exported to videos/camNN.mp4, the real Pose2Sim stages started on "4. Run"
(personAssociation, triangulation, filtering on 2D keypoints projected from the truth), 3D View
overlays, Results > Balance (Wii) metrics and export, sync-event alignment of the recorded take
and of an external Wii-only take with a Bluetooth drop (offsets within one video frame), clean
shutdown. Run in a Z-up Calib world and in a Z-down one (calibration with the Z axis flipped).

About 35 s per world (two real-time takes of 6 s and 5 s, the export and a Pose2Sim
subprocess)."""

import pytest

from tests.e2e_scenario import FPS, run_scenario


@pytest.mark.slow
@pytest.mark.filterwarnings("ignore:Enum value:DeprecationWarning")  # matplotlib Qt toolbar
@pytest.mark.parametrize("up_sign", [1, -1], ids=["z_up", "z_down"])
def test_end_to_end_capture_board_alignment_views(qapp, tmp_path, monkeypatch, up_sign):
    res = run_scenario(tmp_path / "e2e", monkeypatch, shots=None, up_sign=up_sign,
                       log=lambda *_: None)

    assert res["videos"] == ["cam01.mp4", "cam02.mp4"]
    assert res["n_frames"] > 5 * FPS
    assert res["reconnections"] >= 2
    assert max(res["board_corner_err_mm"]) < 10.0
    assert res["trc"].endswith("_filt_butterworth.trc") and res["pose2sim_marker_err_mm"] < 30
    assert abs(res["take1_sync_offset_err_ms"]) < 1000.0 / FPS
    assert abs(res["take2_sync_offset_err_ms"]) < 1000.0 / FPS
    assert res["take2_force_corr"] > 0.95
    assert [e for e, _ in res["take2_drop"]["events"]] == ["wii_disconnected", "wii_connected"]
    assert set(res["exports"]) == {"fused", "summary", "grf"}
    assert res["summary"]["events"] == ["sync"]
    assert [t for t, _ in res["summary"]["force_events"]] == ["takeoff", "landing"]
    assert res["dialog_problems"] == []
