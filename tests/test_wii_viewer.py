"""Wii recording viewer: loading (original program / wii.csv) and real-time replay."""

from __future__ import annotations

import time

import numpy as np
import pytest

from tests.test_wii_legacy import write_legacy


def test_load_trace_original_program_and_wii_csv(tmp_path):
    from poseassess.core.balance.trial import import_recording
    from poseassess.core.project import Project
    from poseassess.wii.viewer import MIN_KG, load_trace

    f = tmp_path / "S01-003"
    raw = write_legacy(f, n=300)
    tr = load_trace(f)
    assert tr.kind == "original Wii program" and len(tr.t) == 300
    assert tr.t[0] == 0 and tr.duration == pytest.approx((raw[-1, 0] - raw[0, 0]) / 1000)
    assert np.isnan(tr.cop_x[raw[:, 7] < MIN_KG]).all()
    assert np.isfinite(tr.cop_x[raw[:, 7] > MIN_KG]).all()
    assert tr.index_at(-1) == 0 and tr.index_at(1e9) == len(tr.t) - 1

    proj = Project(tmp_path / "proj")
    proj.create()
    rec = proj.root / "wii" / "recordings" / import_recording(proj, f)
    tr2 = load_trace(rec)  # recording folder (wii.csv)
    assert tr2.kind == "wii.csv"
    np.testing.assert_allclose(tr2.cop_x, tr.cop_x, atol=1e-6, equal_nan=True)
    with pytest.raises(ValueError):
        load_trace(proj.root / "project.toml")


def test_viewer_replays_in_real_time(qapp, tmp_path):
    from poseassess.wii.viewer import WiiViewer

    f = tmp_path / "trial.csv"
    write_legacy(f, n=600)
    win = WiiViewer(f)
    win.show()
    w = win.replay
    assert w.trace is not None
    first_on = w.trace.t[np.flatnonzero(np.isfinite(w.trace.cop_x))[0]]
    assert w.pos == pytest.approx(max(0.0, first_on - 1.0))  # jumps to just before step-on
    w.seek(0.0)
    w.speed.setCurrentText("2x")
    w.play()
    end = time.perf_counter() + 0.5
    while time.perf_counter() < end:
        qapp.processEvents()
        time.sleep(0.01)
    w.pause()
    assert 0.6 < w.pos < 1.3  # ~2x real time
    w.seek(w.trace.duration * 0.8)
    i = w.trace.index_at(w.pos)
    assert f"{w.trace.total_kg[i]:6.1f} kg" in w.readout.text()
    assert len(w.board.cop_trail) > 1
    w.seek(1e9)
    assert w.pos == pytest.approx(w.trace.duration)
    win.close()
    assert not w.playing


def test_replay_tab_in_wii_board_page(qapp, tmp_path):
    from poseassess.gui.pages.wii_board_page import WiiBoardPage
    from poseassess.gui.state import AppState

    f = tmp_path / "S02-002.csv"
    write_legacy(f, n=300)
    state = AppState()
    page = WiiBoardPage(state)
    try:
        page.show()
        page.tabs.setCurrentWidget(page.replay_tab)
        assert page.tabs.tabText(page.tabs.currentIndex()) == "Replay file"
        assert page.replay_tab.load(f)
        page.replay_tab.play()
        qapp.processEvents()
        page.tabs.setCurrentWidget(page.device_tab)  # switching away pauses the replay
        qapp.processEvents()
        assert not page.replay_tab.playing
    finally:
        page.shutdown()
        page.close()
        state.shutdown()
