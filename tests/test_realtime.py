"""Real-time 3D skeleton: triangulation, calibration loading, filter, angles and the engine
with fake cameras / a fake 2D estimator that projects a known skeleton."""
from __future__ import annotations

import time

import cv2
import numpy as np
import pytest

from poseassess.core.capture.camera import Frame
from poseassess.core.realtime.angles import joint_angles
from poseassess.core.realtime.calib import LiveCamera, load_calib
from poseassess.core.realtime.engine import RealtimeEngine
from poseassess.core.realtime.filters import OneEuroFilter
from poseassess.core.realtime.skeleton import HALPE26, IDX
from poseassess.core.realtime.triangulate import triangulate

W, H = 1920, 1080


def look_at(cam_pos, target, up=(0, 0, 1)):
    cam_pos = np.asarray(cam_pos, float)
    z = np.asarray(target, float) - cam_pos
    z /= np.linalg.norm(z)
    x = np.cross(z, up)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])
    return R, -R @ cam_pos


def make_cams():
    f = (W / 2) / np.tan(np.radians(70) / 2)
    K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.0]])
    dist = np.array([-0.05, 0.01, 0.0, 0.0])
    cams = []
    for i, pos in enumerate([(3.0, -1.5, 1.2), (3.0, 1.5, 1.2)]):
        R, T = look_at(pos, (0, 0, 1.0))
        cams.append(LiveCamera(f"cam{i + 1:02d}", (W, H), K, dist, R, T))
    return cams


def standing_skeleton(knee_deg=0.0):
    """HALPE_26 points of a person standing at the origin (Z up), right knee flexed."""
    X = np.zeros((26, 3))
    def put(n, xyz):
        X[IDX[n]] = xyz
    for s, y in (("R", -0.1), ("L", 0.1)):
        put(s + "Hip", (0, y, 0.95))
        put(s + "Shoulder", (0, 1.8 * y, 1.45))
        put(s + "Elbow", (0, 2.2 * y, 1.18))
        put(s + "Wrist", (0.15, 2.2 * y, 1.0))
        put(s + "Eye", (0.08, 0.3 * y, 1.68))
        put(s + "Ear", (0.0, 0.7 * y, 1.66))
    put("RKnee", (0, -0.1, 0.5))
    a = np.radians(knee_deg)
    put("RAnkle", (-0.45 * np.sin(a), -0.1, 0.5 - 0.45 * np.cos(a)))
    put("LKnee", (0, 0.1, 0.5))
    put("LAnkle", (0, 0.1, 0.05))
    for s, y in (("R", -0.1), ("L", 0.1)):
        ank = X[IDX[s + "Ankle"]]
        put(s + "Heel", ank + (-0.05, 0, -0.03))
        put(s + "BigToe", ank + (0.18, 0.02 * np.sign(y), -0.04))
        put(s + "SmallToe", ank + (0.16, 0.06 * np.sign(y), -0.04))
    put("Neck", (0, 0, 1.47))
    put("Hip", (0, 0, 0.95))
    put("Head", (0, 0, 1.75))
    put("Nose", (0.1, 0, 1.65))
    return X


def test_triangulate_exact():
    cams = make_cams()
    X = standing_skeleton(30)
    uv = np.stack([c.normalize(c.project(X)) for c in cams])
    P = np.stack([c.P_norm for c in cams])
    Xr = triangulate(P, uv, np.ones((2, 26)))
    assert np.allclose(Xr, X, atol=1e-6)


def test_triangulate_missing_view_is_nan():
    cams = make_cams()
    X = standing_skeleton()
    uv = np.stack([c.normalize(c.project(X)) for c in cams])
    w = np.ones((2, 26))
    w[1, 3] = 0
    Xr = triangulate(np.stack([c.P_norm for c in cams]), uv, w)
    assert np.isnan(Xr[3]).all() and np.isfinite(np.delete(Xr, 3, 0)).all()


def test_load_calib_and_scaled_frames(tmp_path):
    cams = make_cams()
    lines = []
    for c in cams:
        rv = cv2.Rodrigues(c.R)[0].ravel()
        lines += [f"[{c.name}]", f'name = "{c.name}"', f"size = [{W}.0, {H}.0]",
                  "matrix = " + str(c.K.tolist()), "distortions = " + str(c.dist.tolist()),
                  "rotation = " + str(rv.tolist()), "translation = " + str(c.T.tolist()),
                  "fisheye = false", ""]
    lines += ["[metadata]", "adjusted = false", "error = 0.0"]
    p = tmp_path / "Calib.toml"
    p.write_text("\n".join(lines))
    loaded = load_calib(p)
    assert [c.name for c in loaded] == ["cam01", "cam02"]
    # the cameras stream at 720p: pixel coords are scaled to the 1080p calibration
    X = standing_skeleton(45)
    live = (1280, 720)
    uv = np.stack([c.normalize(c.project(X, live), live) for c in loaded])
    Xr = triangulate(np.stack([c.P_norm for c in loaded]), uv, np.ones((2, 26)))
    assert np.allclose(Xr, X, atol=1e-5)
    assert loaded[0].check_size(live) is None
    assert loaded[0].check_size((640, 480)) is not None


def test_one_euro_reduces_jitter_and_follows():
    rng = np.random.default_rng(0)
    f = OneEuroFilter(min_cutoff=1.0, beta=0.0)
    raw, out = [], []
    for i in range(300):
        x = np.array([1.0]) + rng.normal(0, 0.01, 1)
        raw.append(x[0])
        out.append(f(x, i / 30)[0])
    assert np.std(out[50:]) < 0.5 * np.std(raw[50:])
    # NaN in -> NaN out; recovers afterwards
    assert np.isnan(f(np.array([np.nan]), 10.1)[0])
    assert np.isfinite(f(np.array([1.0]), 10.2)[0])


@pytest.mark.parametrize("knee", [0.0, 30.0, 90.0])
def test_knee_angle(knee):
    a = joint_angles(standing_skeleton(knee))
    assert a["RKnee"] == pytest.approx(knee, abs=1e-6)
    assert a["LKnee"] == pytest.approx(0.0, abs=1e-6)
    assert a["Trunk"] == pytest.approx(0.0, abs=1e-6)


class FakeStream:
    def __init__(self, fps=30.0):
        self.t0 = time.perf_counter()
        self.fps = fps
        self.img = np.zeros((H, W, 3), np.uint8)

    def latest(self):
        t = time.perf_counter()
        i = int((t - self.t0) * self.fps)
        return Frame(i, self.t0 + i / self.fps, self.img)


class FakeEstimator:
    """Projects the ground-truth skeleton with a little pixel noise."""

    def __init__(self, cam, X, seed):
        self.cam, self.X = cam, X
        self.rng = np.random.default_rng(seed)

    def __call__(self, image):
        time.sleep(0.005)
        k = self.cam.project(self.X) + self.rng.normal(0, 1.0, (26, 2))
        return k, np.full(26, 0.9)


def test_engine_end_to_end(tmp_path):
    cams = make_cams()
    X = standing_skeleton(60)
    srcs = {"cam01": FakeStream(), "cam02": FakeStream()}
    ests = [FakeEstimator(c, X, i) for i, c in enumerate(cams)]
    csv_path = tmp_path / "live.csv"
    eng = RealtimeEngine(srcs, ests, cams, csv_path=csv_path)
    eng.start()
    time.sleep(1.0)
    eng.stop()
    assert eng.error is None
    r = eng.latest()
    assert r is not None and r.seq > 10
    assert np.nanmax(np.abs(r.X - X)) < 0.02
    assert r.angles["RKnee"] == pytest.approx(60, abs=3)
    s = eng.stats.summary()
    assert 15 < s["fps"] < 40
    assert s["latency_ms"] < 100
    head = csv_path.read_text().splitlines()[0].split(",")
    assert "angle_RKnee" in head and f"{HALPE26[-1]}_z" in head


def test_engine_2d_only():
    cams = make_cams()
    X = standing_skeleton()
    srcs = {"cam01": FakeStream(), "cam02": FakeStream()}
    eng = RealtimeEngine(srcs, [FakeEstimator(c, X, i) for i, c in enumerate(cams)], None)
    eng.start()
    time.sleep(0.4)
    eng.stop()
    r = eng.latest()
    assert eng.error is None and r is not None and r.X is None
