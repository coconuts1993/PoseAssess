"""CORE-BOARD: interp, sway_metrics, body_mass_from_force, detect_force_events (PoseBoard
KNOTS/TRUTH profile, ported from 3ea6d8a ``tests/test_alignment.py``), estimate_time_offset,
write_events_csv, detect_trc_events (synth jump + stomp), match_events, xcorr_offset."""

import csv

import numpy as np
import pytest

from poseassess.core.balance import analysis as an
from poseassess.core.balance.analysis import (
    body_mass_from_force,
    detect_force_events,
    detect_trc_events,
    estimate_time_offset,
    foot_marker_indices,
    interp,
    match_events,
    sway_metrics,
    trc_heights,
    write_events_csv,
    xcorr_offset,
)
from tests import synth

W = 70.0
# Piecewise-linear total force: step on (50 % at 2.0 s), countermovement jump (total crosses
# 5 kg at 5.60 s going up and at 5.90 s coming down: 0.30 s flight, 3 BW landing peak), stomp
# (1.8 BW peak at 8.0 s), step off (50 % at 11.0 s).
KNOTS = [(0.0, 0), (1.9, 0), (2.1, W), (5.0, W), (5.15, 0.35 * W), (5.3, W), (5.45, 2.0 * W),
         (5.55, 145), (5.60, 5), (5.61, 0), (5.89, 0), (5.90, 5), (5.93, 3 * W), (6.0, 50),
         (6.3, W), (7.95, W), (8.0, 1.8 * W), (8.05, W), (10.9, W), (11.1, 0), (12.0, 0)]
TRUTH = [("step_on", 2.0), ("takeoff", 5.60), ("landing", 5.90), ("stomp", 8.0), ("step_off", 11.0)]


def synthetic_force(rate=100.0, phase=0.0037, noise=0.3, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(0, 12, 1 / rate) + phase
    kx, ky = np.array(KNOTS).T
    f = np.interp(t, kx, ky)
    f += np.where(f > 10, 0.01 * W * np.sin(2 * np.pi * 0.7 * t), 0) + rng.normal(0, noise, len(t))
    return t, np.maximum(f, -1.0)


# ------------------------------------------------------------------ basic signal tools
def test_interp_gaps_range_and_columns():
    t = np.array([0.0, 0.01, 0.02, 0.03, 0.30, 0.31])
    v = np.array([0.0, 1.0, 2.0, 3.0, 30.0, 31.0])
    out = interp(t, v, np.array([-0.01, 0.0, 0.015, 0.03, 0.1, 0.30, 0.305, 0.4]), max_gap=0.1)
    np.testing.assert_allclose(out, [np.nan, 0.0, 1.5, 3.0, np.nan, 30.0, 30.5, np.nan])
    v2 = v.copy()
    v2[1] = np.nan  # a NaN sample is skipped (gap of 0.02 s is fine)
    assert interp(t, v2, np.array([0.01]))[0] == pytest.approx(1.0)
    both = interp(t, np.column_stack([v, -v]), np.array([0.005, 0.2]))
    assert both.shape == (2, 2)
    np.testing.assert_allclose(both[0], [0.5, -0.5])
    assert np.isnan(both[1]).all()
    assert np.isnan(interp([0.0], [1.0], np.array([0.0]))).all()  # fewer than 2 samples
    order = np.array([3, 0, 2, 1, 5, 4])
    np.testing.assert_allclose(interp(t[order], v[order], np.array([0.015])), [1.5])


def test_sway_metrics_circle():
    t = np.linspace(0, 10, 1001)
    r = 0.01  # 10 mm radius, one turn per 10 s
    m = sway_metrics(t, 0.02 + r * np.cos(2 * np.pi * t / 10), r * np.sin(2 * np.pi * t / 10))
    assert m["samples"] == 1001 and m["duration_s"] == pytest.approx(10)
    assert m["mean_x_mm"] == pytest.approx(20, abs=0.1) and m["mean_y_mm"] == pytest.approx(0, abs=0.1)
    assert m["range_ml_mm"] == pytest.approx(20, abs=0.01)
    assert m["rms_ml_mm"] == pytest.approx(10 / np.sqrt(2), rel=1e-2)
    assert m["path_length_mm"] == pytest.approx(2 * np.pi * 10, rel=1e-3)
    assert m["mean_velocity_mm_s"] == pytest.approx(2 * np.pi, rel=1e-3)
    # 95 % ellipse of a circle with variance r²/2 per axis: pi * 5.991 * r²/2
    assert m["ellipse95_area_mm2"] == pytest.approx(np.pi * 5.991 * 50, rel=2e-2)
    assert sway_metrics(t[:9], t[:9], t[:9]) == {}
    x = np.full(20, np.nan)
    assert sway_metrics(np.arange(20.0), x, x) == {}


@pytest.mark.parametrize("rate", [100, 93, 60])
def test_cop_sway_metrics_filter_sensor_noise(rate):
    """Raw ~100 Hz Wii COP: sensor noise inflates the path length (and makes it depend on the
    Bluetooth rate); the resampled + 10 Hz low-pass COP gives the true path at any rate."""
    from poseassess.wii.protocol import center_of_pressure

    rng = np.random.default_rng(0)
    n = int(30 * rate)
    t = np.sort(np.arange(n) / rate + rng.uniform(-0.3, 0.3, n) / rate)  # irregular stamps
    cop = np.column_stack([0.010 * np.sin(2 * np.pi * 0.23 * t),
                           0.007 * np.sin(2 * np.pi * 0.31 * t + 0.7)])
    kg = synth.kg_from_cop(np.full(n, 70.0), cop) + rng.normal(0, 0.02, (n, 4))
    c = np.array([center_of_pressure(k, 0.433, 0.238, 5.0) for k in kg])
    true = sway_metrics(t, cop[:, 0], cop[:, 1])
    m = an.cop_sway_metrics(t, c[:, 0], c[:, 1])
    assert m["path_length_raw_mm"] > 1.2 * true["path_length_mm"]  # the raw path is inflated
    assert m["path_length_mm"] == pytest.approx(true["path_length_mm"], rel=0.03)
    assert m["mean_velocity_mm_s"] == pytest.approx(true["mean_velocity_mm_s"], rel=0.03)
    assert m["rms_ml_mm"] == pytest.approx(true["rms_ml_mm"], rel=0.01)
    assert m["samples"] == n and m["filter"] == {"resample_hz": 100.0, "lowpass_hz": 10.0,
                                                 "order": 4, "max_gap_s": 0.1}


def test_cop_sway_metrics_never_counts_the_path_across_a_gap():
    t = np.arange(0, 10, 0.01)
    x = np.where(t < 5, 0.0, 0.05)  # 50 mm jump across a gap (nobody on the board)
    x[(t >= 4.9) & (t < 5.1)] = np.nan
    m = an.cop_sway_metrics(t, x, np.zeros_like(t))
    assert m["path_length_mm"] == pytest.approx(0.0, abs=1e-6)
    assert m["path_length_raw_mm"] == pytest.approx(50.0)
    assert m["mean_velocity_mm_s"] == pytest.approx(0.0, abs=1e-6)
    assert an.cop_sway_metrics(t[:9], x[:9], x[:9]) == {}
    short = an.cop_sway_metrics(t[:12], np.zeros(12), np.zeros(12))  # too short to filter
    assert short == {}


def test_body_mass_from_force():
    t, f = synthetic_force()
    assert body_mass_from_force(t, f) == pytest.approx(W, abs=1.0)
    assert body_mass_from_force(t, np.zeros_like(t)) is None
    assert body_mass_from_force(t[:3], f[:3]) is None
    demo = synth.standing_trial(stomp_at=9.0)
    assert body_mass_from_force(demo["t_wii"], demo["total_kg"]) == pytest.approx(70.0, abs=0.01)


# ------------------------------------------------------------------ force events
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_detect_force_events(seed):
    t, f = synthetic_force(seed=seed)
    events = detect_force_events((t, f))
    assert [e["type"] for e in events] == [k for k, _ in TRUTH]
    for e, (_, tt) in zip(events, TRUTH):
        assert abs(e["t_rel"] - tt) < 0.02, e
        assert np.isnan(e["t"]) and np.isnan(e["t_unix"])  # a (t_rel, total) pair has no clock
    take, land, stomp = events[1], events[2], events[3]
    assert land["flight_s"] == pytest.approx(0.30, abs=0.02)
    assert take["flight_s"] == land["flight_s"]
    assert land["peak_kg"] > 1.5 * W  # the landing impact is attached, not reported as a stomp
    assert 1.6 * W < stomp["peak_kg"] < 1.85 * W  # sampled at 100 Hz: a little below 1.8 BW
    assert 7.95 <= stomp["t_rel_onset"] < stomp["t_rel"]  # rise onset before the peak


def test_detect_force_events_dict_folder_and_no_subject(tmp_path):
    t, f = synthetic_force()
    cols = {"t": t + 5.0, "t_rel": t, "t_unix": t + 5.0 + 1.6e9, "total_kg": f}
    events = detect_force_events(cols)
    assert len(events) == 5
    assert events[0]["t_rel"] == pytest.approx(2.0, abs=0.02)
    assert events[0]["t"] == pytest.approx(events[0]["t_rel"] + 5.0, abs=1e-6)
    assert events[0]["t_unix"] == pytest.approx(1.6e9 + 5.0 + events[0]["t_rel"], abs=1e-3)
    # only "t": t_rel from the first sample
    ev_t = detect_force_events({"t": t + 5.0, "total_kg": f})
    assert ev_t[0]["t_rel"] == pytest.approx(events[0]["t_rel"] - t[0], abs=1e-9)
    # a recording folder / its wii.csv (sensor columns, clock from the file)
    kg = synth.kg_from_cop(f, np.zeros((len(f), 2)))
    rec = synth.write_wii_recording(tmp_path / "rec", t, kg, t0=100.0, clock_offset_unix=1.7e9)
    for src in (rec, rec / "wii.csv"):
        ev = detect_force_events(src)
        assert [e["type"] for e in ev] == [k for k, _ in TRUTH]
        assert ev[0]["t"] == pytest.approx(100.0 + ev[0]["t_rel"], abs=1e-6)
        assert ev[0]["t_unix"] == pytest.approx(1.7e9 + 100.0 + ev[0]["t_rel"], abs=1e-3)
    # Nobody on the board: no events
    assert detect_force_events((t, np.random.default_rng(0).normal(0, 0.3, len(t)))) == []
    assert detect_force_events((t[:2], f[:2])) == []


@pytest.mark.parametrize("gap", [(5.50, 5.62), (5.85, 5.97)], ids=["takeoff", "landing"])
def test_detect_force_events_jump_next_to_a_data_gap(gap):
    """A sample gap (Bluetooth dropout / stalled reader) across the takeoff or the landing: the
    jump's timing is uncertain, so it is not reported, and its push-off and landing peaks must
    not turn into stomps (they would be matched against real stomps of the markers)."""
    t, f = synthetic_force()
    keep = (t < gap[0]) | (t > gap[1])
    events = detect_force_events((t[keep], f[keep]))
    assert [e["type"] for e in events] == ["step_on", "stomp", "step_off"]
    assert events[1]["t_rel"] == pytest.approx(8.0, abs=0.02)  # the real stomp is kept
    # a gap far away from the jump changes nothing
    far = (t < 3.0) | (t > 3.2)
    assert [e["type"] for e in detect_force_events((t[far], f[far]))] == [k for k, _ in TRUTH]


def test_estimate_time_offset():
    ref = np.array([2.0, 5.9, 8.0, 11.0, 14.2])
    rng = np.random.default_rng(1)
    other = ref - 123.456 + rng.normal(0, 0.004, len(ref))
    other = np.append(np.delete(other, 3), 40.0)  # one event missed, one spurious
    d, n = estimate_time_offset(ref, other)
    assert d == pytest.approx(123.456, abs=0.01) and n == 4
    d, n = estimate_time_offset([], [1.0])
    assert np.isnan(d) and n == 0


def test_write_events_csv(tmp_path):
    t, f = synthetic_force()
    events = detect_force_events((t, f))
    p = write_events_csv(tmp_path / "ev.csv", events)
    with open(p, newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    assert rows[0] == an.EVENT_COLS and len(rows) == 6
    assert rows[1][0] == "step_on" and rows[1][1] == "" and rows[1][3] == ""  # NaN -> empty
    assert float(rows[3][2]) == pytest.approx(5.9, abs=0.02) and rows[3][4] != ""


# ------------------------------------------------------------------ TRC events
def trc_world(tr, up_sign=1):
    R, t = synth.board_pose(up_sign=up_sign)
    return tr["markers_board"] @ R.T + t, np.array([0.0, 0.0, float(up_sign)])


def test_foot_marker_indices():
    halpe = foot_marker_indices(synth.HALPE26_TRC_MARKERS)
    names = synth.HALPE26_TRC_MARKERS
    assert sorted(names[i] for i in halpe["R"]) == ["RAnkle", "RBigToe", "RHeel", "RSmallToe"]
    assert sorted(names[i] for i in halpe["L"]) == ["LAnkle", "LBigToe", "LHeel", "LSmallToe"]
    mp = ["left_heel", "right_foot_index", "left_knee", "nose", "right_ankle"]
    assert foot_marker_indices(mp) == {"L": [0], "R": [1, 4]}
    lstm = ["r_calc_study", "L_toe_study", "r_5meta_study", "r_knee_study", "RHip"]
    assert foot_marker_indices(lstm) == {"L": [1], "R": [0, 2]}
    assert foot_marker_indices(["LeftFoot", "RightToeBase", "Head"]) == {"L": [0], "R": [1]}


@pytest.mark.parametrize("up_sign", [1, -1])
@pytest.mark.parametrize("fps,jump_at,stomp_at", [(30, 6.0, 9.0), (30, 6.013, 9.021),
                                                  (60, 5.004, 8.5), (25, 4.37, 8.11)])
def test_detect_trc_events_jump_and_stomp(fps, jump_at, stomp_at, up_sign):
    tr = synth.standing_trial(11.0, fps, jump_at=jump_at, stomp_at=stomp_at)
    world, up = trc_world(tr, up_sign)
    ev = detect_trc_events(tr["t_trc"], tr["names"], world, up)
    assert [e["type"] for e in ev] == ["takeoff", "landing", "stomp"]
    assert ev[0]["t"] == pytest.approx(jump_at, abs=0.003)       # ballistic fit: sub-frame
    assert ev[1]["t"] == pytest.approx(jump_at + 0.30, abs=0.003)
    assert ev[0]["flight_s"] == pytest.approx(0.30, abs=0.005)
    assert ev[2]["t"] == pytest.approx(stomp_at, abs=0.3 / fps) and ev[2]["side"] == "R"
    # the wrong up direction: nothing sensible is found (feet "below" the floor)
    assert not [e for e in detect_trc_events(tr["t_trc"], tr["names"], world, -up)
                if e["type"] == "takeoff"]


def test_detect_trc_events_needs_feet_and_body_motion():
    tr = synth.standing_trial(8.0, 30, jump_at=4.0)
    world, up = trc_world(tr)
    names = tr["names"]
    assert detect_trc_events(tr["t_trc"], [n.replace("Heel", "X").replace("Toe", "Y")
                                           .replace("Ankle", "Z") for n in names], world) == []
    # one side only: a jump cannot be told from a step
    keep = [i for i, n in enumerate(names) if not n.startswith("L")]
    assert detect_trc_events(tr["t_trc"], [names[i] for i in keep], world[:, keep]) == []
    # feet markers jump but the body does not (tracking glitch): no jump
    glitch = synth.standing_trial(8.0, 30, jump_at=None)
    gw, _ = trc_world(glitch)
    feet = [i for i, n in enumerate(names) if any(k in n for k in ("Heel", "Toe", "Ankle"))]
    gw[:, feet] = world[:, feet]
    assert detect_trc_events(tr["t_trc"], names, gw) == []
    assert detect_trc_events(tr["t_trc"][:3], names, world[:3]) == []
    # precomputed heights give the same result
    h = trc_heights(names, world, up)
    assert detect_trc_events(tr["t_trc"], names, world, up, heights=h) == \
        detect_trc_events(tr["t_trc"], names, world, up)


# ------------------------------------------------------------------ matching
def test_match_events_type_aware_with_spurious_events():
    d_true = 2.345
    trc = [{"type": "takeoff", "t": 3.0}, {"type": "landing", "t": 3.3},
           {"type": "stomp", "t": 7.0}, {"type": "takeoff", "t": 9.0},
           {"type": "landing", "t": 9.28}]
    rng = np.random.default_rng(0)
    force = [{"type": e["type"], "t_rel": e["t"] + d_true + rng.normal(0, 0.003)} for e in trc[:4]]
    force += [{"type": "step_on", "t_rel": 1.0}, {"type": "stomp", "t_rel": 30.0}]
    force[2]["t_rel_onset"] = force[2]["t_rel"]
    force[2]["t_rel"] += 0.05  # the stomp peak comes after the contact: the onset is used
    m = match_events(force, trc)
    assert m["offset_s"] == pytest.approx(d_true, abs=0.005) and m["n_matched"] == 4
    assert m["residual_ms"] < 6
    assert [p[0] for p in m["pairs"]] == ["takeoff", "landing", "stomp", "takeoff"]
    # a stomp never matches a landing
    m = match_events([{"type": "stomp", "t_rel": 5.0}], [{"type": "landing", "t": 1.0}])
    assert m["n_matched"] == 0 and np.isnan(m["offset_s"]) and m["pairs"] == []
    # offsets beyond max_abs_offset_s are not considered
    assert match_events(force, trc, max_abs_offset_s=1.0)["n_matched"] <= 1
    assert match_events([], trc)["n_matched"] == 0


def test_events_offset_on_the_demo_scene():
    tr = synth.standing_trial(12.0, 30, jump_at=6.0, stomp_at=9.0)
    world, up = trc_world(tr)
    off = 2.5
    fe = detect_force_events((tr["t_wii"] + off, tr["total_kg"]))
    te = detect_trc_events(tr["t_trc"], tr["names"], world, up)
    m = match_events(fe, te)
    assert m["n_matched"] == 3 and m["offset_s"] == pytest.approx(off, abs=0.005)
    d, n = estimate_time_offset([e["t_rel"] for e in fe if e["type"] != "stomp"],
                                [e["t"] for e in te if e["type"] != "stomp"])
    assert d == pytest.approx(off, abs=0.005) and n == 2


# ------------------------------------------------------------------ cross-correlation
@pytest.mark.parametrize("physical,tol", [(True, 0.005), (False, 1 / 30)])
def test_xcorr_offset(physical, tol):
    tr = synth.standing_trial(12.0, 30, jump_at=6.0, stomp_at=None, physical=physical)
    world, up = trc_world(tr)
    _, _, com = trc_heights(tr["names"], world, up)
    off = -7.25
    t_rel = tr["t_wii"] + off
    pre = np.arange(t_rel[0] - 5.0, t_rel[0], 0.01)  # the recording starts earlier
    tf = np.concatenate([pre, t_rel])
    ff = np.concatenate([np.full(len(pre), 70.0), tr["total_kg"]])
    d, score = xcorr_offset(tf, ff, tr["t_trc"], com, search_s=(-30, 30))
    assert d == pytest.approx(off, abs=tol) and score > 0.5
    d2, s2 = xcorr_offset(tf, ff, tr["t_trc"], com, around_s=off + 0.4)
    assert d2 == pytest.approx(d, abs=1e-6) and s2 == pytest.approx(score)
    # not computable: too short, flat, nobody on the board
    assert np.isnan(xcorr_offset(tf[:5], ff[:5], tr["t_trc"], com)[0])
    assert np.isnan(xcorr_offset(tf, np.zeros_like(ff), tr["t_trc"], com)[0])
    flat = xcorr_offset(tf, ff, tr["t_trc"], np.full_like(com, 1.0))
    assert np.isnan(flat[0])


@pytest.mark.parametrize("jump_at", [4.0, 4.013, 4.027])
def test_detect_trc_events_short_jump_low_fps(jump_at, monkeypatch):
    """0.16 s flight at 25 fps: only 1-2 frames above the 3 cm threshold."""
    tr = synth.standing_trial(8.0, 25, jump_at=jump_at, flight_s=0.16)
    world, up = trc_world(tr)
    ev = detect_trc_events(tr["t_trc"], tr["names"], world, up)
    assert [e["type"] for e in ev] == ["takeoff", "landing"]
    assert ev[0]["t"] == pytest.approx(jump_at, abs=0.003)
    assert ev[1]["t"] == pytest.approx(jump_at + 0.16, abs=0.003)
    # without the ballistic fit (e.g. tucked feet): extrapolated contacts, within one frame
    monkeypatch.setattr(an, "_ballistic_roots", lambda t, s: (float("nan"), float("nan")))
    ev = detect_trc_events(tr["t_trc"], tr["names"], world, up)
    assert ev[0]["t"] == pytest.approx(jump_at, abs=1 / 25)
    assert ev[1]["t"] == pytest.approx(jump_at + 0.16, abs=1 / 25)


def test_detect_trc_events_with_marker_noise():
    tr = synth.standing_trial(12.0, 30, jump_at=6.0, stomp_at=9.0)
    world, up = trc_world(tr)
    world = world + np.random.default_rng(4).normal(0, 0.004, world.shape)
    ev = detect_trc_events(tr["t_trc"], tr["names"], world, up)
    kinds = [e["type"] for e in ev]
    assert kinds == ["takeoff", "landing", "stomp"]
    for e, truth in zip(ev, (6.0, 6.3, 9.0)):
        assert e["t"] == pytest.approx(truth, abs=1 / 30)


# ------------------------------------------------------------ local foot baseline
def _jumping_markers(t, jumps, flight=0.30):
    names = synth.HALPE26_TRC_MARKERS
    mk = np.repeat(np.array([synth.STANDING[n] for n in names])[None], len(t), 0).copy()
    lift = np.zeros_like(t)
    for j in jumps:
        s = t - j
        fl = (s >= 0) & (s <= flight)
        lift[fl] = an.G * flight / 2 * s[fl] - 0.5 * an.G * s[fl] ** 2
    mk[:, :, 2] += lift[:, None]
    return names, mk


@pytest.mark.parametrize("t_on", [3.5, 5.0])
def test_detect_trc_events_subject_steps_onto_the_board(t_on):
    """The subject starts on the floor (feet 53 mm lower) for 25-36 % of the video, steps onto
    the board and jumps twice: the jumps are found, the step on is no event."""
    t = np.arange(14 * 30) / 30
    names, mk = _jumping_markers(t, (6.0, 9.0))
    off = t < t_on
    mk[off, :, 2] -= 0.053
    mk[off, :, 0] -= 0.4
    ev = detect_trc_events(t, names, mk, np.array([0, 0, 1.0]))
    assert [e["type"] for e in ev] == ["takeoff", "landing"] * 2
    np.testing.assert_allclose([e["t"] for e in ev], [6.0, 6.3, 9.0, 9.3], atol=0.01)
    # and stepping off at the end is no event either
    down = t > 12.0
    mk[down, :, 2] -= 0.053
    assert [e["type"] for e in detect_trc_events(t, names, mk, np.array([0, 0, 1.0]))] == \
        ["takeoff", "landing"] * 2


@pytest.mark.parametrize("lift_from", [5.0, 6.0])
def test_detect_trc_events_single_leg_stance(lift_from):
    """2 sync jumps, then the left foot lifted 15 cm for the rest of a 30 s trial (80-83 % of
    the frames): the jumps stay jumps (not stomps), the lifted leg is no event."""
    t = np.arange(30 * 30) / 30
    names, mk = _jumping_markers(t, (2.0, 4.0))
    left = [i for i, n in enumerate(names)
            if n.startswith("L") and n[1:] in ("Ankle", "BigToe", "SmallToe", "Heel", "Knee")]
    mk[:, left, 2] += (0.15 * np.clip((t - lift_from) / 0.5, 0, 1))[:, None]
    ev = detect_trc_events(t, names, mk, np.array([0, 0, 1.0]))
    assert [e["type"] for e in ev] == ["takeoff", "landing"] * 2
    np.testing.assert_allclose([e["t"] for e in ev], [2.0, 2.3, 4.0, 4.3], atol=0.01)
