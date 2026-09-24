"""Synthetic, hardware-free test data for the Wii Balance Board integration.

Everything is generated from known ground truth, so tests can check registration, alignment
and fusion numerically:

* ``look_at_camera`` / ``ring_cameras`` / ``write_calib_toml``: calibrated cameras in a Calib.toml
  exactly like PoseAssess writes it (``[cam_1] name = "1"``, 4 distortion coefficients,
  Rodrigues world->camera, metres). ``up_sign=-1`` builds a world whose +Z points DOWN.
* ``board_pose`` / ``project_points`` / ``board_clicks``: a board pose and its 5 landmarks
  (TL, TR, BR, BL, C) projected into each camera.
* ``standing_trial``: a HALPE_26 person standing on the board (swaying; optional jump + stomp),
  in the world frame, plus the matching Wii force (COP, total load) on the same time base.
* ``write_trc``: Pose2Sim-format .trc (Y-up, Frame# 0-based, Time = Frame#/rate, NaN = empty).
* ``write_wii_recording``: a recording folder in the ``poseassess.wii.io`` format.
* ``make_demo_trial``: a complete PoseAssess project (Calib.toml, board clicks + board.json,
  .trc, Wii recording, trial.json) - used by tests and the screenshot scripts.
* ``write_test_video``: a small video file (for CameraStream / FrameSelector tests).

Conventions: body/board frame x = subject's right, y = front, z = up (metres).
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

HALPE26_TRC_MARKERS = [
    "Hip", "RHip", "RKnee", "RAnkle", "RBigToe", "RSmallToe", "RHeel", "LHip", "LKnee",
    "LAnkle", "LBigToe", "LSmallToe", "LHeel", "Neck", "Head", "Nose", "RShoulder", "RElbow",
    "RWrist", "LShoulder", "LElbow", "LWrist"]

# Standing pose in the body frame (x right, y front, z up; feet soles at z = 0), metres.
STANDING = {
    "Hip": (0.0, 0.0, 0.95), "RHip": (0.10, 0.0, 0.95), "RKnee": (0.10, 0.02, 0.50),
    "RAnkle": (0.10, 0.0, 0.08), "RBigToe": (0.12, 0.15, 0.02), "RSmallToe": (0.07, 0.13, 0.02),
    "RHeel": (0.10, -0.05, 0.03), "LHip": (-0.10, 0.0, 0.95), "LKnee": (-0.10, 0.02, 0.50),
    "LAnkle": (-0.10, 0.0, 0.08), "LBigToe": (-0.12, 0.15, 0.02),
    "LSmallToe": (-0.07, 0.13, 0.02), "LHeel": (-0.10, -0.05, 0.03), "Neck": (0.0, 0.0, 1.50),
    "Head": (0.0, 0.02, 1.72), "Nose": (0.0, 0.10, 1.65), "RShoulder": (0.18, 0.0, 1.45),
    "RElbow": (0.22, 0.0, 1.15), "RWrist": (0.24, 0.02, 0.90), "LShoulder": (-0.18, 0.0, 1.45),
    "LElbow": (-0.22, 0.0, 1.15), "LWrist": (-0.24, 0.02, 0.90)}

G = 9.80665
FLIP = np.diag([1.0, -1.0, -1.0])  # proper rotation mapping a Z-up world to a Z-down one


# ------------------------------------------------------------------ cameras
def look_at_camera(center, target=(0.0, 0.0, 0.8), size=(1280, 720), f=1000.0, up_sign=1,
                   name="") -> dict:
    """Upright pinhole camera at ``center`` looking at ``target`` (world). ``up_sign`` = sign
    of the world Z axis that points physically up. Returns ``{name, K, dist(5), rvec(3),
    tvec(3), size, R, center}`` (world -> camera: x_cam = R x_world + t)."""
    c, tg = np.asarray(center, float), np.asarray(target, float)
    z = tg - c
    z /= np.linalg.norm(z)
    x = np.cross(z, [0.0, 0.0, float(up_sign)])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.vstack([x, y, z])
    t = -R @ c
    K = np.array([[f, 0, size[0] / 2], [0, f, size[1] / 2], [0, 0, 1.0]])
    return {"name": name, "K": K, "dist": np.zeros(5), "rvec": cv2.Rodrigues(R)[0].ravel(),
            "tvec": t, "size": tuple(size), "R": R, "center": c}


def ring_cameras(n=3, radius=3.0, height=1.6, target=(0.0, 0.0, 0.6), up_sign=1,
                 size=(1280, 720), f=1000.0, start_deg=-90.0) -> list[dict]:
    """``n`` cameras on a circle around ``target`` (heights along the physical up)."""
    cams = []
    for i in range(n):
        a = np.deg2rad(start_deg + 360.0 * i / n)
        c = np.array([radius * np.cos(a), radius * np.sin(a), up_sign * height])
        tg = np.array([target[0], target[1], up_sign * target[2]])
        cams.append(look_at_camera(c, tg, size, f, up_sign, name=f"cam{i + 1:02d}"))
    return cams


def to_calibration(cam: dict, extrinsics: bool = True, name: str | None = None):
    """``look_at_camera`` dict -> ``poseassess.core.balance.calib.CameraCalibration`` (without
    extrinsics when ``extrinsics`` is False)."""
    from poseassess.core.balance.calib import CameraCalibration

    return CameraCalibration(name or cam["name"] or "cam", tuple(cam["size"]),
                             np.asarray(cam["K"], float), np.asarray(cam["dist"], float),
                             np.asarray(cam["rvec"], float).copy() if extrinsics else None,
                             np.asarray(cam["tvec"], float).copy() if extrinsics else None)


def write_calib_toml(path, cams: list[dict], style: str = "poseassess") -> Path:
    """Write a Calib.toml. ``style="poseassess"``: tables ``[cam_1]`` with ``name = "1"`` (what
    PoseAssess's own calibration writes); ``"opencap"``: ``[cam01]`` with ``name = "cam01"``."""
    import tomli_w

    data = {}
    for i, c in enumerate(cams, start=1):
        key, name = (f"cam_{i}", str(i)) if style == "poseassess" else (f"cam{i:02d}", f"cam{i:02d}")
        data[key] = {"name": name, "size": [float(c["size"][0]), float(c["size"][1])],
                     "matrix": np.asarray(c["K"], float).tolist(),
                     "distortions": np.asarray(c["dist"], float).ravel()[:4].tolist(),
                     "rotation": np.asarray(c["rvec"], float).ravel().tolist(),
                     "translation": np.asarray(c["tvec"], float).ravel().tolist(),
                     "fisheye": False}
    data["metadata"] = {"adjusted": False, "error": 0.0}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(tomli_w.dumps(data).encode("utf-8"))
    return path


def project_points(cam: dict, pts_w) -> np.ndarray:
    """World points (N,3) -> pixels (N,2) with ``cam``'s distortion."""
    img, _ = cv2.projectPoints(np.asarray(pts_w, float).reshape(-1, 3),
                               np.asarray(cam["rvec"], float), np.asarray(cam["tvec"], float),
                               cam["K"], cam["dist"])
    return img.reshape(-1, 2)


# -------------------------------------------------------------------- board
def board_landmarks(length_mm=511.0, width_mm=316.0) -> np.ndarray:
    """TL, TR, BR, BL, C in the board frame (5,3), like ``BoardGeometry.landmarks``."""
    hx, hy = length_mm / 2000.0, width_mm / 2000.0
    return np.array([[-hx, hy, 0], [hx, hy, 0], [hx, -hy, 0], [-hx, -hy, 0], [0, 0, 0]], float)


def board_pose(yaw_deg=90.0, xy=(0.2, 0.1), height_m=0.053, up_sign=1):
    """Board -> world ``(R, t)`` for a board lying flat on the floor (top surface at
    ``height_m`` along the physical up). With ``up_sign=-1`` the world's +Z points down."""
    a = np.deg2rad(yaw_deg)
    Rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1.0]])
    R = Rz if up_sign > 0 else FLIP @ Rz
    t = np.array([xy[0], xy[1] if up_sign > 0 else -xy[1], up_sign * height_m])
    return R, t


def board_clicks(cams: list[dict], R, t, noise_px=0.0, seed=0) -> list[np.ndarray]:
    """The 5 landmarks projected into every camera (optionally with pixel noise)."""
    rng = np.random.default_rng(seed)
    pts_w = board_landmarks() @ np.asarray(R).T + np.asarray(t)
    return [project_points(c, pts_w) + rng.normal(0, noise_px, (5, 2)) if noise_px
            else project_points(c, pts_w) for c in cams]


def render_board_image(cam: dict, R, t, bg=110) -> np.ndarray:
    """Simple BGR image of the board (dark outline, light top, power-button edge marked)."""
    w, h = cam["size"]
    img = np.full((h, w, 3), bg, np.uint8)
    pts = project_points(cam, board_landmarks()[:4] @ np.asarray(R).T + np.asarray(t))
    poly = np.round(pts * 16).astype(np.int32)
    cv2.fillPoly(img, [poly], (235, 235, 235), cv2.LINE_AA, shift=4)
    cv2.polylines(img, [poly], True, (40, 40, 40), 2, cv2.LINE_AA, shift=4)
    back = project_points(cam, np.array([[0.0, -0.15, 0.0]]) @ np.asarray(R).T + np.asarray(t))[0]
    cv2.circle(img, (int(back[0]), int(back[1])), 5, (200, 120, 0), -1)  # power button (blue)
    return img


# ------------------------------------------------------------ person + force
def kg_from_cop(total_kg, cop_xy, sensor_dx_m=0.433, sensor_dy_m=0.238) -> np.ndarray:
    """Split a total load over the four sensors (TR, BR, TL, BL) so the Wii COP is ``cop_xy``
    (bilinear, like ``SimulatedBoard.kg_at``). Shapes (N,), (N,2) -> (N,4)."""
    total = np.asarray(total_kg, float).reshape(-1)
    cop = np.nan_to_num(np.asarray(cop_xy, float).reshape(-1, 2))
    u = np.clip(cop[:, 0] / (sensor_dx_m / 2), -1, 1) / 2 + 0.5
    v = np.clip(cop[:, 1] / (sensor_dy_m / 2), -1, 1) / 2 + 0.5
    w = np.column_stack([u * v, u * (1 - v), (1 - u) * v, (1 - u) * (1 - v)])
    return w * total[:, None]


def jump_profile(tt, jump_at, flight_s=0.30, t_push=0.4, t_land=0.4):
    """Vertical displacement ``u`` (m) and acceleration (m/s²) of a physically consistent jump:
    quintic countermovement + push (``t_push`` s, starting and ending smoothly: zero velocity
    and acceleration at the start, ``v0`` and ``-g`` at takeoff), ballistic flight of
    ``flight_s`` s from ``jump_at``, and the mirrored landing (``t_land`` s). ``u`` is 0 at
    takeoff and landing, negative during the push and landing dips."""
    tt = np.asarray(tt, float)
    v0 = G * flight_s / 2

    def quintic(T):
        A = np.array([[T ** 3, T ** 4, T ** 5], [3 * T ** 2, 4 * T ** 3, 5 * T ** 4],
                      [6 * T, 12 * T ** 2, 20 * T ** 3]])
        return np.linalg.solve(A, [0.0, v0, -G])

    u, acc = np.zeros_like(tt), np.zeros_like(tt)
    s = tt - jump_at
    push = (s >= -t_push) & (s < 0)
    a3, a4, a5 = quintic(t_push)
    x = s[push] + t_push
    u[push] = a3 * x ** 3 + a4 * x ** 4 + a5 * x ** 5
    acc[push] = 6 * a3 * x + 12 * a4 * x ** 2 + 20 * a5 * x ** 3
    fl = (s >= 0) & (s <= flight_s)
    u[fl] = v0 * s[fl] - 0.5 * G * s[fl] ** 2
    acc[fl] = -G
    land = (s > flight_s) & (s <= flight_s + t_land)
    b3, b4, b5 = quintic(t_land)
    y = flight_s + t_land - s[land]
    u[land] = b3 * y ** 3 + b4 * y ** 4 + b5 * y ** 5
    acc[land] = 6 * b3 * y + 12 * b4 * y ** 2 + 20 * b5 * y ** 3
    return u, acc


def standing_trial(duration_s=12.0, fps=30.0, wii_rate=100.0, mass_kg=70.0, jump_at=6.0,
                   flight_s=0.30, stomp_at=None, sway_mm=(15.0, 10.0), physical=False) -> dict:
    """A person standing on the board, in the BOARD frame, on one time base ``t`` (s).

    Returns ``{"names", "t_trc" (N,), "markers_board" (N,K,3), "t_wii" (M,),
    "total_kg" (M,), "cop_board" (M,2), "kg" (M,4), "events": [(type, t)]}``: inverted-pendulum
    sway (COM path ~ ``sway_mm``), an optional jump (whole body lifted on a parabola during
    ``flight_s``, force 0 in flight, landing peak) and an optional right-foot stomp.
    Force = m (g + a_com_z) / g with the COP following the COM projection.

    ``physical=True`` (added for the cross-correlation tests): the jump is physically
    consistent instead - the whole body (feet included) moves by a smooth countermovement /
    push / ballistic flight / landing profile (``jump_profile``) and the force is exactly
    ``m (1 + z''/g)`` of the COM, so force and markers obey Newton's law.
    """
    names = list(HALPE26_TRC_MARKERS)
    base = np.array([STANDING[n] for n in names], float)

    def sway(tt):
        return np.column_stack([sway_mm[0] / 1000 * np.sin(2 * np.pi * 0.23 * tt),
                                sway_mm[1] / 1000 * np.sin(2 * np.pi * 0.31 * tt + 0.7)])

    def lift(tt):
        z = np.zeros_like(tt)
        if jump_at is not None and physical:
            return jump_profile(tt, jump_at, flight_s)[0]
        if jump_at is not None:
            v0 = G * flight_s / 2
            s = tt - jump_at
            fl = (s >= 0) & (s <= flight_s)
            z[fl] = v0 * s[fl] - 0.5 * G * s[fl] ** 2
        return z

    t_trc = np.arange(int(round(duration_s * fps))) / fps
    sw = sway(t_trc)
    k = base[:, 2] / 1.0  # sway grows with height (pendulum about the ankles)
    mk = np.repeat(base[None], len(t_trc), axis=0)
    mk[:, :, 0] += sw[:, :1] * k[None]
    mk[:, :, 1] += sw[:, 1:] * k[None]
    mk[:, :, 2] += lift(t_trc)[:, None]
    events = []
    if jump_at is not None:
        events += [("takeoff", jump_at), ("landing", jump_at + flight_s)]
    if stomp_at is not None:  # right foot lifted 8 cm and slammed down at stomp_at
        right = [i for i, n in enumerate(names) if n.startswith("R") and n[1:] in
                 ("Ankle", "BigToe", "SmallToe", "Heel")]
        s = t_trc - (stomp_at - 0.4)
        up = np.clip(np.sin(np.pi * s / 0.4), 0, None) * ((s >= 0) & (s <= 0.4))
        mk[:, right, 2] += 0.08 * up[:, None]
        events.append(("stomp", stomp_at))

    t_wii = np.arange(int(round(duration_s * wii_rate))) / wii_rate
    com_xy = sway(t_wii) * 0.95  # COM at about hip height
    # COP = COM - (h/g) * COM acceleration (inverted pendulum), h ~ 0.95 m
    acc = -(2 * np.pi * np.array([0.23, 0.31])) ** 2 * sway(t_wii)
    cop = com_xy - 0.95 / G * acc * 0.95
    total = np.full_like(t_wii, mass_kg)
    if jump_at is not None and physical:
        total = mass_kg * (1.0 + jump_profile(t_wii, jump_at, flight_s)[1] / G)
    elif jump_at is not None:
        s = t_wii - jump_at
        total[(s >= 0) & (s <= flight_s)] = 0.0
        push = (s > -0.35) & (s < 0)
        total[push] += mass_kg * 1.0 * np.sin(np.pi * (s[push] + 0.35) / 0.35)
        land = (s > flight_s) & (s < flight_s + 0.25)
        total[land] += mass_kg * 2.0 * np.sin(np.pi * (s[land] - flight_s) / 0.25)
    if stomp_at is not None:
        s = t_wii - stomp_at
        st = (s >= 0) & (s < 0.1)
        total[st] += mass_kg * 0.8 * np.sin(np.pi * s[st] / 0.1)
    cop[total < 5.0] = np.nan
    return {"names": names, "t_trc": t_trc, "markers_board": mk, "t_wii": t_wii,
            "total_kg": total, "cop_board": cop, "kg": kg_from_cop(total, cop),
            "events": events}


# --------------------------------------------------------------------- files
def write_trc(path, names: list[str], coords_trc: np.ndarray, rate: float,
              frames: np.ndarray | None = None) -> Path:
    """Write a Pose2Sim-style .trc: ``coords_trc`` (N,K,3) already in the .trc frame (Y-up),
    metres; Frame# = ``frames`` (default 0..N-1), Time = Frame# / rate; NaN -> empty field."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = coords_trc.shape[0]
    frames = np.arange(n) if frames is None else np.asarray(frames, int)
    lines = [f"PathFileType\t4\t(X/Y/Z)\t{path.name}",
             "DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\t"
             "OrigDataStartFrame\tOrigNumFrames",
             f"{rate:g}\t{rate:g}\t{n}\t{len(names)}\tm\t{rate:g}\t{frames[0]}\t{n}",
             "Frame#\tTime\t" + "\t\t\t".join(names) + "\t\t\t",
             "\t\t" + "\t".join(f"X{i + 1}\tY{i + 1}\tZ{i + 1}" for i in range(len(names))) + "\t"]
    for f, row in zip(frames, coords_trc):
        vals = ["" if not np.isfinite(v) else f"{v:.6f}" for v in row.ravel()]
        lines.append(f"{f}\t{f / rate:.6f}\t" + "\t".join(vals))
    path.write_text("\n".join(lines) + "\n")
    return path


def write_wii_recording(folder, t_rel, kg, events=(), t0=1000.0, clock_offset_unix=1.7e9,
                        sensor_dx_m=0.433, sensor_dy_m=0.238, min_total_kg=5.0,
                        meta_extra: dict | None = None) -> Path:
    """Write session.json + wii.csv (+ events.csv) in the ``poseassess.wii.io`` format.
    ``events``: iterable of ``(label, t_rel)``. COP board is computed like the device does;
    COP world columns are left empty."""
    from poseassess.wii import io as wio
    from poseassess.wii.protocol import center_of_pressure

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    t_rel = np.asarray(t_rel, float)
    kg = np.asarray(kg, float).reshape(-1, 4)

    def f(v):
        return "" if not np.isfinite(v) else f"{v:.6f}"

    rows = [",".join(wio.WII_HEADER)]
    for tr, k in zip(t_rel, kg):
        cop = center_of_pressure(k, sensor_dx_m, sensor_dy_m, min_total_kg)
        t = t0 + tr
        rows.append(",".join([f(t), f(tr), f(t + clock_offset_unix), *[f(v) for v in k],
                              f(k.sum()), f(cop[0]), f(cop[1]), "", "", ""]))
    (folder / wio.WII_CSV).write_text("\n".join(rows) + "\n")
    evs = list(events)
    if evs:
        lines = [",".join(wio.EVENTS_HEADER)]
        for label, tr in evs:
            lines.append(f"{t0 + tr:.6f},{tr:.6f},{t0 + tr + clock_offset_unix:.6f},{label}")
        (folder / wio.EVENTS_CSV).write_text("\n".join(lines) + "\n", encoding="utf-8-sig")
    meta = {"schema": wio.SESSION_SCHEMA, "app": "synthetic", "created": "2026-01-01T12:00:00",
            "subject": "", "notes": "synthetic", "t0": t0, "t0_unix": t0 + clock_offset_unix,
            "clock_offset_unix": clock_offset_unix, "has_wii": True, "has_video": False,
            "camera_names": [], "streams": [],
            "force_source": {"type": "SimulatedBoard", "tare_kg": [0, 0, 0, 0],
                             "sensor_dx_m": sensor_dx_m, "sensor_dy_m": sensor_dy_m,
                             "min_total_kg": min_total_kg},
            "force_source_history": [], "duration_s": float(t_rel[-1]) if len(t_rel) else 0.0,
            "samples": {"wii": int(len(t_rel)), "events": len(evs)}}
    meta.update(meta_extra or {})
    wio.write_session_json(folder, meta)
    return folder


def write_test_video(path, n_frames=60, size=(320, 240), fps=30.0) -> Path:
    """Small MJPG .avi (or any extension OpenCV can write) with the frame index drawn."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*("MJPG" if path.suffix.lower() == ".avi" else "mp4v"))
    vw = cv2.VideoWriter(str(path), fourcc, fps, size)
    for i in range(n_frames):
        img = np.full((size[1], size[0], 3), (i * 4) % 255, np.uint8)
        cv2.putText(img, str(i), (10, size[1] // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                    (255, 255, 255), 3)
        vw.write(img)
    vw.release()
    return path


# ------------------------------------------------------------------ project
def make_demo_trial(root, n_cams=3, up_sign=1, fps=30, duration_s=12.0, jump_at=6.0,
                    stomp_at=9.0, offset_s=2.5, with_board=True, with_recording=True,
                    with_images=True, physical=False) -> dict:
    """A complete PoseAssess project with Wii data, from known truth.

    Writes: project.toml (``num_cameras=n_cams``, ``frame_rate=fps``; no Config.toml),
    calibration/Calib.toml, wii/board/camNN/{board.jpg, points.json} (exact clicks),
    wii/board/board.json (the TRUE pose, method "triangulation", 0 px errors - so GUI work does
    not depend on ``register_board``), pose-3d/demo_filt_butterworth.trc (Y-up), a Wii recording
    ``wii/recordings/20260101_120000`` whose ``t_rel = trc_time + offset_s``, and trial.json
    (source "external", alignment "manual" with the true offset). ``physical`` is passed to
    ``standing_trial`` (a jump whose force obeys Newton's law, for cross-correlation tests).

    Returns ``{"project", "cams", "board_R", "board_t", "trc", "rec_id", "offset_s", "trial"
    (standing_trial dict), "up_sign"}``.
    """
    from poseassess.core.balance.board import BoardRegistration, CameraClicks
    from poseassess.core.balance.calib import calib_fingerprint
    from poseassess.core.balance.geometry import (
        BoardGeometry,
        BoardPose,
        RigidTransform,
    )
    from poseassess.core.balance.paths import WiiPaths
    from poseassess.core.balance.trial import Alignment, TrialInfo, save_trial
    from poseassess.core.project import Project

    root = Path(root)
    proj = Project(root)
    proj.config.name = "demo"
    proj.config.num_cameras = n_cams
    proj.config.frame_rate = int(fps)
    proj.create()
    cams = ring_cameras(n_cams, up_sign=up_sign)
    calib = write_calib_toml(proj.calib_toml, cams)
    R, t = board_pose(up_sign=up_sign)
    paths = WiiPaths(proj).ensure()

    clicks = {}
    for i, (cam, px) in enumerate(zip(cams, board_clicks(cams, R, t)), start=1):
        d = paths.board_cam_dir(i)
        d.mkdir(parents=True, exist_ok=True)
        if with_images:
            cv2.imwrite(str(d / "board.jpg"), render_board_image(cam, R, t))
        cc = CameraClicks(f"cam{i:02d}", [tuple(map(float, p)) for p in px],
                          "board.jpg" if with_images else None, tuple(cam["size"]), "synthetic")
        (d / "points.json").write_text(json.dumps(cc.to_dict(), indent=2))
        clicks[cc.cam] = cc
    if with_board:
        pose = BoardPose(RigidTransform(np.asarray(R, float), np.asarray(t, float)),
                         "triangulation", {c: 0.0 for c in clicks}, None, True, 0.0, [],
                         ["synthetic ground truth"])
        reg = BoardRegistration(pose, BoardGeometry(), clicks, calib_fingerprint(calib),
                                up_sign, "2026-01-01T12:00:00+00:00", None)
        paths.board_json.write_text(json.dumps(reg.to_dict(), indent=2))

    tr = standing_trial(duration_s, fps, jump_at=jump_at, stomp_at=stomp_at, physical=physical)
    world = tr["markers_board"] @ np.asarray(R).T + np.asarray(t)  # body frame == board frame
    trc_coords = world[..., [1, 2, 0]]  # Z-up world -> Pose2Sim Y-up
    trc = write_trc(proj.pose3d_dir / "demo_filt_butterworth.trc", tr["names"], trc_coords, fps)

    rec_id = None
    if with_recording:
        rec_id = "20260101_120000"
        t_rel = tr["t_wii"] + offset_s
        pre = np.arange(0.0, offset_s, 0.01)  # the Wii recording starts before the video
        t_all = np.concatenate([pre, t_rel])
        kg_all = np.vstack([kg_from_cop(np.full(len(pre), 70.0), np.zeros((len(pre), 2))),
                            tr["kg"]])
        evs = [("sync", jump_at + offset_s)] if jump_at is not None else []
        write_wii_recording(paths.recording_dir(rec_id), t_all, kg_all, evs)
        save_trial(proj, TrialInfo(rec_id, "external",
                                   Alignment("manual", float(offset_s), None,
                                             "pose-3d/demo_filt_butterworth.trc",
                                             {"synthetic": True}, "2026-01-01T12:00:00+00:00")))
    return {"project": proj, "cams": cams, "board_R": R, "board_t": t, "trc": trc,
            "rec_id": rec_id, "offset_s": offset_s, "trial": tr, "up_sign": up_sign}


def fused_from_demo(demo: dict):
    """A ``FusedTrial`` for ``make_demo_trial`` output built from the ground truth, WITHOUT
    ``fusion.fuse_trial`` (lets GUI work/tests proceed independently of the core
    implementation). Same semantics as ``fuse_trial``: rows = .trc rows, world = Calib world."""
    from poseassess.core.balance.board import load_board
    from poseassess.core.balance.com import center_of_mass
    from poseassess.core.balance.fusion import FusedTrial
    from poseassess.core.balance.paths import WiiPaths
    from poseassess.core.balance.trial import load_trial
    from poseassess.wii import io as wio

    proj, tr, off = demo["project"], demo["trial"], float(demo["offset_s"])
    R, t = np.asarray(demo["board_R"], float), np.asarray(demo["board_t"], float)
    tt = tr["t_trc"]
    n = len(tt)

    def at(v):
        return np.interp(tt, tr["t_wii"], v)

    kg = np.column_stack([at(tr["kg"][:, j]) for j in range(4)])
    total = at(tr["total_kg"])
    cop = np.column_stack([at(np.nan_to_num(tr["cop_board"][:, j])) for j in range(2)])
    cop[total < 5.0] = np.nan
    up = R[:, 2]
    cop_w = np.column_stack([cop, np.zeros(n)]) @ R.T + t
    world = tr["markers_board"] @ R.T + t
    com_w = np.array([center_of_mass(world[i], tr["names"]) for i in range(n)], float)
    com_b = (com_w - t) @ R
    trial = load_trial(proj)
    rec = WiiPaths(proj).recording_dir(demo["rec_id"]) if demo.get("rec_id") else None
    wii = wio.read_wii_csv(rec) if rec is not None else {}
    if wii:
        wii["trc_time"] = wii["t_rel"] - off
    events = []
    for e in (wio.read_events_csv(rec) if rec is not None else []):
        e = dict(e)
        e["trc_time"] = e["t_rel"] - off
        events.append(e)
    return FusedTrial(
        trc_path=Path(demo["trc"]), recording=demo.get("rec_id"), alignment=trial.alignment,
        frames=np.arange(n), trc_time=tt.copy(), t_rel=tt + off,
        t_unix=tt + off + 1000.0 + 1.7e9, kg=kg, total_kg=total, cop_board=cop,
        cop_world=cop_w, force_world=up[None, :] * (total * G)[:, None], com_world=com_w,
        com_board=com_b, board=load_board(proj), body_mass_kg=70.0, events=events, wii=wii,
        warnings=[])
