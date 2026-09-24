"""End-to-end scenario of the Wii Balance Board integration, driven through the real MainWindow.

Everything a user does is done through the pages (buttons, pickers, tabs); only the hardware is
simulated:

* a fake hidapi bus with a **scripted Balance Board** that speaks the real HID protocol
  (calibration block, 0x32 sensor reports) and whose load follows a ground-truth ``Scene``: a
  70 kg subject swaying on the board with a countermovement jump at a known time, per-sensor
  zero offsets (removed by the tare), and hand presses for the corner check;
* two file "cameras" (looping videos of the synthetic scene) and a known Calib.toml (2 cameras
  looking at the floor checkerboard world, Z up);
* the 2D pose stage: OpenPose-format HALPE_26 json per exported video frame, projected from the
  same ground truth (as if RTMPose were perfect). The .trc itself comes from the real Pose2Sim
  stages (personAssociation, triangulation, filtering) started on "4. Run".

Steps (``run_scenario``): 1. Project page: create the project; Calib.toml. 2. Plug-and-play:
the board appears and is connected automatically; tare; Bluetooth drop and reconnect (tare
kept). 3. 3b. Capture: open the file cameras, board snapshots. 4. 2b. Wii Board: click the board
in both cameras (deliberately front/back swapped), compute, live corner check (BR responds ->
clicks swapped and recomputed), corner check again (OK). 5. 3b. Capture: take 1 = cameras + Wii
with a jump and a "sync" mark; exported automatically to videos/cam01.mp4, cam02.mp4 (recorded
alignment). 6. 2D pose json + 4. Run (Pose2Sim) -> pose-3d/*_filt_butterworth.trc. 7. 5. 3D View:
board / COP / force / COM overlays. 8. 6. Results > Balance (Wii): metrics, plots, fused export.
9. Re-align take 1 by sync event (must match the recorded alignment within one frame).
10. External workflow: a Wii-only take 2 (same movement replayed, Bluetooth drop and reconnect
during the take), used as the trial recording, aligned by sync event (known offset).
11. Take 1 selected again (recorded timestamps), screenshots of every page, close.

Used by ``tests/test_integration_e2e.py`` (headless) and runnable as a script for screenshots
(needs OpenGL, i.e. xvfb)::

    QT_QPA_PLATFORM=xcb LIBGL_ALWAYS_SOFTWARE=1 xvfb-run -a -s "-screen 0 1600x1000x24" \\
        python -m tests.e2e_scenario OUT_DIR [--prefix=pa_final_]
"""

from __future__ import annotations

import json
import re
import shutil
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from tests.synth import (
    FLIP,
    HALPE26_TRC_MARKERS,
    STANDING,
    G,
    board_landmarks,
    board_pose,
    jump_profile,
    kg_from_cop,
    look_at_camera,
    project_points,
    render_board_image,
    write_calib_toml,
    write_trc,
)
from tests.wii_fakes import CAL, FakeHid, sensor_report

FPS = 30
SIZE = (640, 360)
FOCAL = 520.0
TARGET = (0.2, 0.1, 0.8)  # Z-up world; FLIP maps it to the Z-down variant (up_sign=-1)
CAM_CENTERS = [(-2.4, 1.6, 1.5), (-2.2, -1.9, 1.3)]
FEET = ("RAnkle", "RBigToe", "RSmallToe", "RHeel", "LAnkle", "LBigToe", "LSmallToe", "LHeel")
SENSOR_INDEX = {"TR": 0, "BR": 1, "TL": 2, "BL": 3}
COM_H = 0.95
BONES = [("Hip", "Neck"), ("Neck", "Head"), ("Hip", "RHip"), ("RHip", "RKnee"),
         ("RKnee", "RAnkle"), ("RAnkle", "RBigToe"), ("RAnkle", "RHeel"), ("Neck", "RShoulder"),
         ("RShoulder", "RElbow"), ("RElbow", "RWrist"), ("Hip", "LHip"), ("LHip", "LKnee"),
         ("LKnee", "LAnkle"), ("LAnkle", "LBigToe"), ("LAnkle", "LHeel"),
         ("Neck", "LShoulder"), ("LShoulder", "LElbow"), ("LElbow", "LWrist")]


# ================================================================ ground truth + hardware
class Scene:
    """Ground truth as a function of the scene time ``s = perf_counter() - t_ref``: an empty
    board (only sensor zero offsets), or the subject standing on it (inverted-pendulum sway,
    optional physically consistent jumps at scene times ``jumps``), plus an optional hand press
    on one sensor."""

    SENSOR_OFFSET_KG = np.array([0.40, -0.25, 0.30, 0.55])  # untared zero (TR, BR, TL, BL)

    def __init__(self, mass_kg: float = 70.0, sway_mm=(12.0, 8.0), flight_s: float = 0.30):
        self.t_ref = time.perf_counter()
        self.mass_kg = mass_kg
        self.sway_mm = sway_mm
        self.flight_s = flight_s
        self.on_board = False
        self.press: tuple[str, float] | None = None
        self.jumps: list[float] = []
        self._rng = np.random.default_rng(7)

    def s(self, now: float) -> float:
        return now - self.t_ref

    def sway(self, s) -> np.ndarray:
        s = np.atleast_1d(np.asarray(s, float))
        return np.column_stack([self.sway_mm[0] / 1000 * np.sin(2 * np.pi * 0.23 * s),
                                self.sway_mm[1] / 1000 * np.sin(2 * np.pi * 0.31 * s + 0.7)])

    def sway_acc(self, s) -> np.ndarray:
        return -(2 * np.pi * np.array([0.23, 0.31])) ** 2 * self.sway(s)

    def lift(self, s) -> tuple[np.ndarray, np.ndarray]:
        s = np.atleast_1d(np.asarray(s, float))
        u, acc = np.zeros_like(s), np.zeros_like(s)
        for j in self.jumps:
            uu, aa = jump_profile(s, j, self.flight_s)
            u += uu
            acc += aa
        return u, acc

    def kg(self, now: float) -> np.ndarray:
        """Untared sensor loads (TR, BR, TL, BL) at ``now``."""
        s = self.s(now)
        kg = self.SENSOR_OFFSET_KG + self._rng.normal(0.0, 0.02, 4)
        if self.on_board:
            _u, acc = self.lift(s)
            total = max(0.0, self.mass_kg * (1.0 + float(acc[0]) / G))
            com = self.sway(s)[0] * COM_H
            cop = com - COM_H / G * self.sway_acc(s)[0] * COM_H
            kg = kg + kg_from_cop([total], [cop])[0]
        if self.press is not None:
            kg[SENSOR_INDEX[self.press[0]]] += self.press[1]
        return kg

    def markers_board(self, s) -> np.ndarray:
        """HALPE_26 markers (N, K, 3) in the board frame (= body frame; soles on the top)."""
        s = np.atleast_1d(np.asarray(s, float))
        names = HALPE26_TRC_MARKERS
        base = np.array([STANDING[n] for n in names], float)
        sw = self.sway(s)
        mk = np.repeat(base[None], len(s), axis=0)
        mk[:, :, 0] += sw[:, :1] * base[None, :, 2]
        mk[:, :, 1] += sw[:, 1:] * base[None, :, 2]
        u, _ = self.lift(s)
        feet = np.array([n in FEET for n in names])
        mk[:, ~feet, 2] += u[:, None]                 # the body squats, pushes and flies
        mk[:, feet, 2] += np.maximum(u, 0.0)[:, None]  # the feet stay on the board until takeoff
        return mk


def kg_to_raw(kg) -> np.ndarray:
    """Inverse of ``protocol.Calibration.to_kg`` for the fake board's calibration block."""
    kg = np.asarray(kg, float)
    raw = np.where(kg < 17.0, CAL.kg0 + kg / 17.0 * (CAL.kg17 - CAL.kg0),
                   CAL.kg17 + (kg - 17.0) / 17.0 * (CAL.kg34 - CAL.kg17))
    return np.clip(np.round(raw), 0, 65535).astype(int)


class ScriptedHid(FakeHid):
    """Fake HID Balance Board streaming the scene at about 100 Hz."""

    def __init__(self, scene: Scene):
        super().__init__([0, 0, 0, 0])
        self.scene = scene

    def read(self, n, timeout_ms):
        if self.closed:
            raise ValueError("not open")
        if self.broken:
            raise OSError("read error")
        if self.queue:
            return list(self.queue.popleft())
        if not self.streaming:
            time.sleep(timeout_ms / 1000)
            return []
        time.sleep(0.01)
        return list(sensor_report(kg_to_raw(self.scene.kg(time.perf_counter()))))


class ScriptedBus:
    """Fake hidapi bus (``tests.wii_fakes.FakeBus`` pattern) whose boards play ``scene``."""

    def __init__(self, monkeypatch, scene: Scene):
        from poseassess.wii.device import BalanceBoardHID
        from tests.wii_fakes import BOARD, PATH

        self.scene = scene
        self.present = False
        self.opened: list[ScriptedHid] = []
        self._board, self._path = BOARD, PATH
        monkeypatch.setattr(BalanceBoardHID, "list_devices", staticmethod(self.list_devices))
        monkeypatch.setattr(BalanceBoardHID, "open_device", staticmethod(self.open_device))

    def list_devices(self):
        return [dict(self._board)] if self.present else []

    def open_device(self, path):
        assert path == self._path
        dev = ScriptedHid(self.scene)
        self.opened.append(dev)
        return dev


# ======================================================================== synthetic files
def board_truth(up_sign: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Board -> world ``(R, t)``: flat on the floor, the subject facing world -X (Z-up world);
    ``up_sign=-1``: the same scene in a Calib world whose +Z points down."""
    return board_pose(yaw_deg=90.0, xy=(0.2, 0.1), up_sign=up_sign)


def make_cameras(up_sign: int = 1) -> list[dict]:
    F = np.eye(3) if up_sign > 0 else FLIP
    return [look_at_camera(F @ np.asarray(c, float), F @ np.asarray(TARGET, float), SIZE, FOCAL,
                           up_sign=up_sign, name=f"cam{i:02d}")
            for i, c in enumerate(CAM_CENTERS, start=1)]


def scene_image(cam: dict, index: int, up_sign: int = 1) -> np.ndarray:
    """One camera frame: floor, board (power button marked), standing subject, frame number."""
    BOARD_R, BOARD_T = board_truth(up_sign)
    img = render_board_image(cam, BOARD_R, BOARD_T, bg=95)
    names = HALPE26_TRC_MARKERS
    pts_b = np.array([STANDING[n] for n in names], float)
    px = project_points(cam, pts_b @ BOARD_R.T + BOARD_T)
    at = {n: tuple(int(round(v)) for v in p) for n, p in zip(names, px)}
    for a, b in BONES:
        col = (230, 150, 60) if a.startswith("R") or b.startswith("R") else \
            (60, 140, 250) if a.startswith("L") or b.startswith("L") else (200, 200, 200)
        cv2.line(img, at[a], at[b], col, 3, cv2.LINE_AA)
    cv2.circle(img, at["Head"], 12, (200, 200, 200), 2, cv2.LINE_AA)
    cv2.putText(img, f"{cam['name']}  #{index:03d}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2, cv2.LINE_AA)
    return img


def write_source_video(path: Path, cam: dict, n_frames: int = 90, up_sign: int = 1) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), float(FPS), SIZE)
    for i in range(n_frames):
        vw.write(scene_image(cam, i, up_sign))
    vw.release()
    return path


HALPE26_IDS = {"Nose": 0, "LShoulder": 5, "RShoulder": 6, "LElbow": 7, "RElbow": 8, "LWrist": 9,
               "RWrist": 10, "LHip": 11, "RHip": 12, "LKnee": 13, "RKnee": 14, "LAnkle": 15,
               "RAnkle": 16, "Head": 17, "Neck": 18, "Hip": 19, "LBigToe": 20, "RBigToe": 21,
               "LSmallToe": 22, "RSmallToe": 23, "LHeel": 24, "RHeel": 25}


def write_pose_json(project, cams: list[dict], markers_world: np.ndarray, names: list[str],
                    score: float = 0.9) -> None:
    """What the 2D pose stage (RTMPose, HALPE_26) would write for the exported videos: one
    OpenPose-format json per video frame in ``pose/camNN_json/`` (keypoints projected from the
    world markers ``(N, K, 3)``; eyes / ears left empty)."""
    n = markers_world.shape[0]
    for i, cam in enumerate(cams, start=1):
        d = project.pose2d_cam_dir(i)
        d.mkdir(parents=True, exist_ok=True)
        px = project_points(cam, markers_world.reshape(-1, 3)).reshape(n, len(names), 2)
        for k in range(n):
            kp = np.zeros((26, 3))
            for j, name in enumerate(names):
                kp[HALPE26_IDS[name]] = (px[k, j, 0], px[k, j, 1], score)
            person = {"person_id": [-1], "pose_keypoints_2d": kp.ravel().round(3).tolist(),
                      "face_keypoints_2d": [], "hand_left_keypoints_2d": [],
                      "hand_right_keypoints_2d": [], "pose_keypoints_3d": [],
                      "face_keypoints_3d": [], "hand_left_keypoints_3d": [],
                      "hand_right_keypoints_3d": []}
            (d / f"cam{i:02d}_{k:06d}.json").write_text(
                json.dumps({"version": 1.3, "people": [person]}))


def write_mot(path: Path, n: int = 150, rate: float = FPS) -> Path:
    """A small OpenSim .mot (joint angles in degrees) for the Results "Joint angles" tab."""
    t = np.arange(n) / rate
    cols = {"knee_angle_r": 10 + 5 * np.sin(t), "knee_angle_l": 11 + 4 * np.sin(t),
            "hip_flexion_r": 5 + 3 * np.sin(t), "hip_flexion_l": 5 + 2.5 * np.sin(t),
            "ankle_angle_r": 2 + 2 * np.sin(t), "ankle_angle_l": 2 + 1.5 * np.sin(t),
            "pelvis_tilt": 1 + 0.5 * np.sin(t)}
    lines = [path.stem, "version=1", f"nRows={n}", f"nColumns={len(cols) + 1}",
             "inDegrees=yes", "endheader", "time\t" + "\t".join(cols)]
    for i in range(n):
        lines.append(f"{t[i]:.4f}\t" + "\t".join(f"{v[i]:.4f}" for v in cols.values()))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return path


# ============================================================================ Qt helpers
def _deferred_deletes() -> None:
    """Delete the objects ``deleteLater`` scheduled (a running event loop would; ``processEvents``
    alone does not, so replaced widgets would stay visible in screenshots)."""
    from PySide6.QtCore import QCoreApplication, QEvent

    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def pump(seconds: float = 0.05) -> None:
    from tests.qtutil import pump as _pump

    _pump(seconds)
    _deferred_deletes()


def pump_until(cond, timeout: float = 5.0, what: str = "") -> None:
    from tests.qtutil import pump_until as _pu

    ok = _pu(cond, timeout)
    _deferred_deletes()
    if not ok:
        raise AssertionError(f"timeout ({timeout:g} s) waiting for: {what or cond}")


class Dialogs:
    """Non-blocking stand-ins for QMessageBox (answers Yes) and QFileDialog (returns the path
    set in ``next_dir`` / ``next_file``); records what was shown."""

    def __init__(self, monkeypatch):
        from PySide6.QtWidgets import QFileDialog, QMessageBox

        self.shown: list[tuple[str, str, str]] = []
        self.next_dir = ""
        self.next_file = ""

        def ask(_parent, title, text, *a, **k):
            self.shown.append(("question", str(title), str(text)))
            return QMessageBox.Yes

        def note(kind):
            def f(_parent, title, text, *a, **k):
                self.shown.append((kind, str(title), str(text)))
                return QMessageBox.Ok
            return f

        monkeypatch.setattr(QMessageBox, "question", staticmethod(ask))
        for kind in ("information", "warning", "critical"):
            monkeypatch.setattr(QMessageBox, kind, staticmethod(note(kind)))
        monkeypatch.setattr(QFileDialog, "getExistingDirectory",
                            staticmethod(lambda *a, **k: self.next_dir))
        monkeypatch.setattr(QFileDialog, "getOpenFileName",
                            staticmethod(lambda *a, **k: (self.next_file, "")))

    def problems(self) -> list[tuple[str, str, str]]:
        return [d for d in self.shown if d[0] in ("warning", "critical")]


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower()


# ============================================================================== scenario
def run_scenario(work: Path, monkeypatch, shots: Path | None = None, prefix: str = "e2e_",
                 take1_s: float = 6.0, jump1: float = 3.0, take2_s: float = 5.0,
                 jump2: float = 2.0, pose2sim: bool = True, up_sign: int = 1,
                 project_name: str = "E2E_trial", subject: str = "S01", log=print) -> dict:
    """Run the whole scenario in ``work`` (emptied first). ``shots``: folder for screenshots
    (``None``: none). ``pose2sim``: the .trc comes from the real Pose2Sim stages started on
    "4. Run" (personAssociation, triangulation, filtering) on 2D keypoints projected from the
    truth; False writes the .trc directly. ``up_sign=-1``: the Calib.toml world has +Z pointing
    down (as after a calibration with the Z axis flipped). ``project_name`` / ``subject`` may be
    non-ASCII (e.g. Chinese). Raises AssertionError on the first failed check;
    returns the measured numbers."""
    from PySide6.QtWidgets import QApplication

    from poseassess.core.balance import alignment as AL
    from poseassess.core.balance.board import load_all_clicks, load_board
    from poseassess.core.balance.fusion import fuse_trial
    from poseassess.core.balance.paths import WiiPaths
    from poseassess.core.balance.trial import load_trial
    from poseassess.core.project import Project
    from poseassess.gui.main_window import MainWindow
    from poseassess.wii import io as wio

    app = QApplication.instance() or QApplication([])
    work = Path(work)
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    root = work / project_name
    res: dict = {"root": str(root)}
    dialogs = Dialogs(monkeypatch)
    monkeypatch.setenv("POSEASSESS_WII", "auto")  # plug-and-play, as in the real app
    scene = Scene()
    bus = ScriptedBus(monkeypatch, scene)
    cams = make_cameras(up_sign)
    BOARD_R, BOARD_T = board_truth(up_sign)
    up_w = np.array([0.0, 0.0, float(up_sign)])
    sources = [write_source_video(work / "sources" / f"camera{i}.avi", c, up_sign=up_sign)
               for i, c in enumerate(cams, start=1)]

    def shot(name: str) -> None:
        if shots is None:
            return
        shots.mkdir(parents=True, exist_ok=True)
        pump(0.3)
        p = shots / f"{prefix}{name}.png"
        w.grab().save(str(p))
        res.setdefault("screenshots", []).append(str(p))
        log(f"  screenshot {p.name}")

    w = MainWindow()
    w.state.wii.poll_interval_s = 0.2      # quicker hot-plug polling for the test
    w.state.wii.retry_failed_s = 0.5
    w.resize(1400, 860)
    w.show()
    pages = {type(p).__name__: p for p in w.pages}

    def goto(name: str):
        page = pages[name]
        w.nav.setCurrentRow(w.pages.index(page))
        pump(0.15)
        return page

    try:
        # ---- 1. Project page: create the project, then "calibrate" ------------------ #
        log("1. create project + Calib.toml")
        pp = goto("ProjectPage")
        pp.path_edit.setText(str(root))
        pp.name_edit.setText(project_name.replace("_", " "))
        pp.cameras_spin.setValue(2)
        pp.fps_spin.setValue(FPS)
        pp._create()
        proj = w.state.project
        assert proj is not None and Project.is_project(root) and proj.config_file.is_file()
        write_calib_toml(proj.calib_toml, cams)
        w.state.open_project(root)  # like reopening after "2. Calibration"
        proj = w.state.project
        paths = WiiPaths(proj)
        write_mot(proj.kinematics_dir / f"{project_name}_0-149.mot")

        # ---- 2. plug-and-play: the paired board is switched on --------------------- #
        log("2. plug-and-play connect, tare, drop + reconnect")
        wii = w.state.wii
        pump(0.3)
        assert wii.source is not None and not wii.connected, "auto-connect must be searching"
        assert "searching" in wii.status_text()
        bus.present = True
        pump_until(lambda: wii.connected, 6.0, "automatic connection")
        pump_until(lambda: len(wii.recent(1.2)) > 60, 4.0, "force data")
        wb = goto("WiiBoardPage")
        wb.tabs.setCurrentWidget(wb.device_tab)
        shot("wii_device_empty")
        wb.btn_tare.click()
        tare = np.asarray(wii.tare_values, float)
        assert np.allclose(tare, Scene.SENSOR_OFFSET_KG, atol=0.05), tare
        res["tare_kg"] = tare.round(3).tolist()
        bus.opened[-1].broken = True  # Bluetooth drop
        pump_until(lambda: len(bus.opened) >= 2 and wii.connected
                   and len(wii.recent(0.5)) > 20, 8.0, "reconnection")
        pump(0.3)
        lat = wii.latest()
        assert lat is not None and abs(float(lat.total_kg)) < 0.3, "tare kept after reconnect"
        assert getattr(wii.source, "connections", 0) >= 2
        res["reconnections"] = int(wii.source.connections)

        # ---- 3. Capture page: open the cameras, board snapshots --------------------- #
        log("3. capture page: open cameras + board snapshots")
        cp = goto("CapturePage")
        for r, src in enumerate(sources):
            cp.cam_table.cellWidget(r, 2).setEditText(str(src))
            cp.cam_table.cellWidget(r, 4).setEditText(f"{SIZE[0]}x{SIZE[1]}")
            cp.cam_table.cellWidget(r, 5).setEditText(str(FPS))
        pump(0.05)
        cp.b_open.click()
        pump_until(lambda: all(cp._session.latest(c) is not None for c in ("cam01", "cam02")),
                   6.0, "camera frames")
        pump(0.5)
        cp.b_snap.click()
        snaps = {i: sorted(paths.board_cam_dir(i).glob("snapshot_*.jpg")) for i in (1, 2)}
        assert all(len(v) == 1 for v in snaps.values()), snaps
        for i, v in snaps.items():
            h, wd = cv2.imread(str(v[0])).shape[:2]
            assert (wd, h) == SIZE
        shot("capture_preview")

        # ---- 4. Wii Board page: clicks (front/back swapped), compute, corner check -- #
        log("4. board location: clicks, compute, corner check")
        wb = goto("WiiBoardPage")
        wb.tabs.setCurrentWidget(wb.board_tab)
        pump(0.1)
        land_w = board_landmarks() @ BOARD_R.T + BOARD_T
        rng = np.random.default_rng(3)
        for i, cam in enumerate(cams, start=1):
            wb.cam_combo.setCurrentIndex(i - 1)
            pump(0.05)
            rows = [wb.frame_list.item(k).data(0x0100) for k in range(wb.frame_list.count())]
            wb.frame_list.setCurrentRow(rows.index(str(snaps[i][0])))
            pump(0.05)
            px = project_points(cam, land_w) + rng.normal(0, 0.4, (5, 2))
            swapped = px[[2, 3, 0, 1, 4]]  # the user took the power-button edge for the front
            wb.picker.set_points([tuple(map(float, p)) for p in swapped])
            wb.btn_save.click()
            pump(0.05)
        clicks = load_all_clicks(proj)
        assert sorted(c for c, v in clicks.items() if v.complete) == ["cam01", "cam02"]
        wb.btn_compute.click()
        pump(0.2)
        reg = load_board(proj)
        assert reg is not None and reg.pose.method == "triangulation", reg
        swap_err = np.linalg.norm(reg.corners_world - land_w[[2, 3, 0, 1]], axis=1).max()
        assert swap_err < 0.01, f"swapped clicks must give the board turned by 180° ({swap_err})"
        shot("wii_board_swapped")

        wb.tabs.setCurrentWidget(wb.device_tab)
        pump(0.1)
        wb.tabs.setCurrentWidget(wb.board_tab)
        pump_until(lambda: wb.btn_corner.isEnabled(), 3.0, "corner check enabled")
        assert wb.start_corner_check()
        pump(0.7)
        scene.press = ("BR", 12.0)  # the hand is on the corner the user believes is TL
        pump_until(lambda: wb._corner is None, 6.0, "corner check result (BR)")
        scene.press = None
        reg = load_board(proj)
        assert reg.corner_check and reg.corner_check["result"] == "swapped", reg.corner_check
        corner_err = np.linalg.norm(reg.corners_world - land_w[:4], axis=1)
        assert corner_err.max() < 0.01, corner_err
        assert np.dot(reg.up_world, up_w) > 0.9995 and reg.pose.floor_constrained
        assert reg.pose.tilt_deg < 3.0, reg.pose.tilt_deg  # unconstrained fit, noisy clicks
        res["board_corner_err_mm"] = (corner_err * 1000).round(2).tolist()
        res["board_reproj_px"] = reg.pose.reproj_error_px
        shot("wii_board_fixed")
        pump(0.8)
        assert wb.start_corner_check()
        pump(0.7)
        scene.press = ("TL", 12.0)
        pump_until(lambda: wb._corner is None, 6.0, "corner check result (TL)")
        scene.press = None
        assert load_board(proj).corner_check["result"] == "ok"
        shot("wii_board_check_ok")

        # ---- 5. take 1: cameras + Wii, jump, sync mark; automatic export ----------- #
        log("5. take 1 (cameras + Wii) and export to videos/")
        scene.on_board = True
        pump(0.5)
        cp = goto("CapturePage")
        cp.subject_edit.setText(subject)
        cp.notes_edit.setText("e2e take 1")
        cp.chk_record_cams.setChecked(True)
        cp.b_record.click()
        s = cp._session
        assert s.recording and s.recorder.meta.get("has_video")
        rec1_t0 = s.recorder.t0
        d1 = rec1_t0 - scene.t_ref
        scene.jumps = [d1 + jump1]  # takeoff at t_rel = jump1 of this take
        pump_until(lambda: time.perf_counter() >= rec1_t0 + jump1 - 0.03, jump1 + 2, "jump")
        cp.event_edit.setText("sync")
        cp.b_mark.click()
        cp.event_edit.setText("")
        pump_until(lambda: time.perf_counter() >= rec1_t0 + take1_s * 0.6, take1_s, "take")
        shot("capture_recording")
        pump_until(lambda: time.perf_counter() >= rec1_t0 + take1_s, take1_s, "take end")
        rec1 = s.folder.name
        cp.b_record.click()  # stop -> videos/ empty -> exported automatically
        pump_until(lambda: not cp._export_running() and load_trial(proj).alignment.method
                   == "recorded", 60.0, "export of take 1")
        pump(0.3)
        shot("capture_exported")
        rec1_dir = paths.recording_dir(rec1)
        vids = sorted(p.name for p in proj.videos_dir.iterdir())
        assert vids == ["cam01.mp4", "cam02.mp4"], vids
        frames = wio.read_frames_csv(rec1_dir / wio.FRAMES_CSV)
        n_frames = len(frames["frame"])
        assert np.array_equal(frames["frame"], np.arange(n_frames))
        assert np.allclose(np.diff(frames["t_rel"]), 1.0 / FPS, atol=1e-5)
        for i in (1, 2):
            cap = cv2.VideoCapture(str(proj.video_file(i)))
            got = (int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), round(cap.get(cv2.CAP_PROP_FPS)),
                   int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            cap.release()
            assert got == (n_frames, FPS, SIZE[0], SIZE[1]), got
            assert (rec1_dir / wio.timestamps_csv_name(f"cam{i:02d}")).is_file()
        wii1 = wio.read_wii_csv(rec1_dir)
        rate = (len(wii1["t_rel"]) - 1) / (wii1["t_rel"][-1] - wii1["t_rel"][0])
        assert rate > 60, f"Wii sample rate {rate:.0f} Hz"
        quiet = (wii1["t_rel"] > 0.5) & (wii1["t_rel"] < jump1 - 0.6)
        assert abs(np.median(wii1["total_kg"][quiet]) - 70.0) < 0.5, "tared body weight"
        evs = [e["label"] for e in wio.read_events_csv(rec1_dir)]
        assert "sync" in evs, evs
        meta = wio.read_session_json(rec1_dir)
        assert meta["export"]["n_frames"] == n_frames and meta["subject"] == subject
        trial = load_trial(proj)
        assert trial.recording == rec1 and trial.source == "capture"
        vp = goto("VideosPage")
        assert [vp.table.item(r, 2).text() for r in range(2)] == ["ready", "ready"]
        shot("videos_ready")
        res.update(take1=rec1, n_frames=n_frames, t_rel0=float(frames["t_rel"][0]),
                   wii_rate_hz=round(rate, 1), videos=vids)

        # ---- 6. the pipeline: 2D pose -> 4. Run (Pose2Sim) -> .trc ---------------- #
        s_k = frames["t_rel"] + d1
        mk_w = scene.markers_board(s_k) @ BOARD_R.T + BOARD_T  # truth at the video frames
        if pose2sim:
            log("6. 2D pose json + 4. Run: Pose2Sim personAssociation, triangulation, "
                "filtering")
            write_pose_json(proj, cams, mk_w, list(HALPE26_TRC_MARKERS))
            run = goto("RunPage")
            for name, cb in run.stage_checks.items():
                cb.setChecked(name in ("personAssociation", "triangulation", "filtering"))
            run.run_btn.click()
            pump_until(lambda: run.summary.text().startswith(("Done", "Pipeline error")),
                       240.0, "the Pose2Sim run")
            assert run.summary.text() == "Done: 3/3 stages ok.", \
                run.summary.text() + "\n" + run.log_view.toPlainText()[-3000:]
            shot("run_done")
            trcs = sorted(proj.pose3d_dir.glob("*_filt_butterworth.trc"))
            assert len(trcs) == 1, list(proj.pose3d_dir.iterdir())
            trc_path = trcs[0]
            from poseassess.core.balance.trc import read_trc_full

            tt = read_trc_full(trc_path)
            # Frame# = index of the exported video frame (Pose2Sim 0.10's filtering drops the
            # last row of the triangulated .trc)
            fr = np.asarray(tt.frames, int)
            assert fr[0] == 0 and np.array_equal(fr, np.arange(len(fr))), fr[:5]
            assert n_frames - 1 <= len(fr) <= n_frames, (len(fr), n_frames)
            err = np.linalg.norm(tt.world() - mk_w[fr], axis=2)
            res["pose2sim_marker_err_mm"] = round(float(np.nanmax(err)) * 1000, 2)
            assert np.nanmedian(err) < 0.01, "Pose2Sim .trc must match the truth (Y-up)"
        else:
            log("6. write the .trc (Pose2Sim HALPE_26, Y-up)")
            trc_path = proj.pose3d_dir / f"{project_name}_0-{n_frames - 1}_filt_butterworth.trc"
            write_trc(trc_path, list(HALPE26_TRC_MARKERS), mk_w[..., [1, 2, 0]], FPS,
                      frames=frames["frame"].astype(int))
            fr = frames["frame"].astype(int)
        res["trc"] = trc_path.name
        trc_t_rel = frames["t_rel"][fr]  # Wii time of every .trc row (recorded alignment)
        n_trc = len(fr)

        # ---- 7. 3D View: overlays ----------------------------------------------- #
        log("7. 3D view overlays")
        vz = goto("Viz3DPage")
        vz._refresh_list()
        pump_until(lambda: vz.viewer.has_balance() and vz.viewer._fused is not None, 5.0,
                   "3D overlays")
        fused = vz.viewer._fused
        assert fused.alignment.method == "recorded"
        assert np.allclose(fused.t_rel, trc_t_rel, atol=1e-6)
        assert "Recorded with the videos" in vz.wii_status.text(), vz.wii_status.text()
        res["overlay_checks"] = check_overlays(vz.viewer, fused, reg, trc_t_rel, jump1)
        v = vz.viewer
        if shots is not None:
            v.view.setCameraPosition(distance=3.2, elevation=14, azimuth=-150)
            k_st = int(np.argmin(np.abs(trc_t_rel - 1.5)))
            v._show_frame(k_st)
            shot("viz3d_stance")
            k_fl = int(np.argmin(np.abs(trc_t_rel - (jump1 + 0.15))))
            v._show_frame(k_fl)
            shot("viz3d_flight")
            v._show_frame(k_st)

        # ---- 8. Results > Balance (Wii): metrics, plots, export --------------------- #
        log("8. results balance tab + export")
        rp = goto("ResultsPage")
        rp.tabs.setCurrentWidget(rp.balance)
        bp = rp.balance
        pump_until(lambda: bp.fused is not None and bp.summary is not None, 8.0, "balance")
        summ = bp.summary
        assert bp.table.rowCount() > 10
        assert summ["frames"] == n_trc and abs(summ["body_mass_kg"] - 70.0) < 1.0, summ
        assert summ["cop"] and summ["com"] and summ["com_cop"], summ
        assert summ["com_cop"]["corr_ml"] > 0.9 and summ["com_cop"]["corr_ap"] > 0.9, summ
        res["summary"] = {"body_mass_kg": round(summ["body_mass_kg"], 2),
                          "com_cop": {k: round(v, 3) for k, v in summ["com_cop"].items()
                                      if isinstance(v, float)},
                          "force_events": [(e.get("type"), round(e["trc_time"], 3))
                                           for e in summ["force_events"]],
                          "events": [e.get("label") for e in summ["events"]]}
        exp = bp.export_to(paths.exports_dir)
        assert exp and all(Path(p).is_file() for p in exp.values()), exp
        res["exports"] = {k: Path(p).name for k, p in exp.items()}
        with open(exp["fused"], encoding="utf-8") as f:
            header = f.readline().strip().split(",")
            n_rows = sum(1 for _ in f)
        assert n_rows == n_trc and "cop_x_world" in header and "com_z_world" in header, header
        shot("results_balance")

        # ---- 9. take 1 aligned by sync event (must equal the recorded alignment) ----- #
        log("9. sync-event alignment of take 1")
        ap = bp.alignment
        assert ap.recorded_box.isVisible()
        ap.realign_btn.click()
        ap.detect_btn.click()
        pump_until(lambda: not ap.busy and ap._result is not None, 30.0, "detection")
        assert ap._result.ok, ap._result.message
        off1 = float(ap.offset_spin.value())
        err1 = off1 - float(frames["t_rel"][0])
        assert abs(err1) < 1.0 / FPS, f"sync event offset error {err1 * 1000:.1f} ms"
        ap.save_btn.click()
        pump(0.2)
        trial = load_trial(proj)
        assert trial.alignment.method == "sync_event"
        res["take1_sync_offset_err_ms"] = round(err1 * 1000, 2)
        shot("results_alignment_take1")

        # ---- 10. external workflow: a Wii-only take, then alignment ----------------- #
        log("10. Wii-only take 2 + sync-event alignment")
        cp = goto("CapturePage")
        cp.chk_record_cams.setChecked(False)
        cp.notes_edit.setText("e2e take 2 (Wii only)")
        cp.b_record.click()
        s = cp._session
        assert s.recording and not s.recorder.meta.get("has_video")
        rec2_t0 = s.recorder.t0
        scene.t_ref = rec2_t0 + jump2 - scene.jumps[0]  # the same movement, jump at t_rel=jump2
        drop_at = jump2 + 1.0  # a Bluetooth drop after the landing: the take goes on
        pump_until(lambda: time.perf_counter() >= rec2_t0 + drop_at, take2_s, "drop time")
        n_dev = len(bus.opened)
        bus.opened[-1].broken = True
        pump_until(lambda: len(bus.opened) > n_dev and wii.connected, 5.0, "reconnection")
        t_back = time.perf_counter() - rec2_t0
        pump_until(lambda: time.perf_counter() >= rec2_t0 + max(take2_s, t_back + 0.8),
                   take2_s + 5, "take 2")
        rec2 = s.folder.name
        cp.b_record.click()  # -> "Use the take as the trial's Wii recording?" -> Yes
        pump(0.3)
        rec2_dir = paths.recording_dir(rec2)
        ev2 = [(e["label"], e["t_rel"]) for e in wio.read_events_csv(rec2_dir)]
        labels = [lb for lb, _ in ev2]
        assert "wii_disconnected" in labels and "wii_connected" in labels, ev2
        t_off = dict(ev2)["wii_disconnected"]
        assert drop_at - 0.2 < t_off < drop_at + 1.0, ev2
        wii2 = wio.read_wii_csv(rec2_dir)
        assert (wii2["t_rel"] > t_back + 0.3).sum() > 20, "samples after the reconnection"
        gap = np.diff(wii2["t_rel"]).max()
        res["take2_drop"] = {"events": ev2, "gap_s": round(float(gap), 3)}
        trial = load_trial(proj)
        assert trial.recording == rec2 and trial.source == "external", trial
        assert trial.alignment.method == "none"
        assert sorted(p.name for p in proj.videos_dir.iterdir()) == vids, "videos untouched"
        assert AL.alignment_status(proj)[0] == "warn"
        rp = goto("ResultsPage")
        rp.tabs.setCurrentWidget(rp.balance)
        pump(0.3)
        ap.detect_btn.click()
        pump_until(lambda: not ap.busy and ap._result is not None, 30.0, "detection 2")
        assert ap._result.ok, ap._result.message
        off2 = float(ap.offset_spin.value())
        true2 = float(frames["t_rel"][0]) + jump2 - jump1
        err2 = off2 - true2
        assert abs(err2) < 1.0 / FPS, f"take 2 offset error {err2 * 1000:.1f} ms"
        ap.save_btn.click()
        pump(0.2)
        assert load_trial(proj).alignment.method == "sync_event"
        f2 = fuse_trial(proj, trc_path)
        ok = np.isfinite(f2.total_kg) & np.isfinite(fused.total_kg)
        assert ok.sum() > 0.5 * n_trc
        r = np.corrcoef(f2.total_kg[ok], fused.total_kg[ok])[0, 1]
        assert r > 0.95, f"take 2 vs take 1 force correlation {r:.3f}"
        res.update(take2=rec2, take2_offset_true_s=round(true2, 4),
                   take2_sync_offset_err_ms=round(err2 * 1000, 2), take2_force_corr=round(r, 4))
        shot("results_alignment_take2")

        # ---- 11. back to take 1 (recorded timestamps); every page ------------------- #
        log("11. take 1 again (recorded), screenshots of every page")
        i = ap.rec_combo.findData(rec1)
        ap.rec_combo.setCurrentIndex(i)
        ap.rec_combo.activated.emit(i)  # as a user choice: "use its frame timestamps?" -> Yes
        pump(0.3)
        trial = load_trial(proj)
        assert trial.recording == rec1 and trial.alignment.method == "recorded"
        assert AL.alignment_status(proj) == ("ok", "Recorded with the videos (frame timestamps)")
        rp.tabs.setCurrentIndex(0)
        rp._load_latest()
        assert rp.table.rowCount() > 0
        if shots is not None:
            cal = pages["CalibrationPage"]
            for k, page in enumerate(w.pages):
                goto(type(page).__name__)
                if page is cal:
                    for t in range(cal.tabs.count()):
                        cal.tabs.setCurrentIndex(t)
                        shot(f"page_{k + 1}_{_slug(page.nav_title)}_tab{t + 1}")
                    cal.tabs.setCurrentIndex(0)
                    continue
                if page is pages["Viz3DPage"]:
                    page.viewer.view.setCameraPosition(distance=3.2, elevation=14,
                                                       azimuth=-150)
                    page.viewer._show_frame(int(np.argmin(np.abs(trc_t_rel - 1.5))))
                if page is pages["WiiBoardPage"]:
                    page.tabs.setCurrentWidget(page.board_tab)
                    shot(f"page_{k + 1}_{_slug(page.nav_title)}_board")
                    page.tabs.setCurrentWidget(page.device_tab)
                if page is pages["ResultsPage"]:
                    shot(f"page_{k + 1}_{_slug(page.nav_title)}_angles")
                    page.tabs.setCurrentWidget(page.balance)
                if page is pages["CapturePage"]:
                    page.tabs.setCurrentIndex(1)
                    shot(f"page_{k + 1}_{_slug(page.nav_title)}_takes")
                    page.tabs.setCurrentIndex(0)
                shot(f"page_{k + 1}_{_slug(page.nav_title)}")
        res["dialog_problems"] = dialogs.problems()
        res["dialogs"] = [(k, t) for k, t, _ in dialogs.shown]
    finally:
        pump(0.1)
        w.close()
        pump(0.2)
    assert not w.isVisible(), "the window must close"
    src = w.state.wii.source
    assert src is None or not src.running, "the Wii source must be stopped on exit"
    left = [t.name for t in threading.enumerate()
            if t is not threading.main_thread() and t.is_alive()
            and (t.name.startswith("cam-") or "wii" in t.name.lower())]
    assert not left, f"threads still running after close: {left}"
    app.processEvents()
    return res


def check_overlays(viewer, fused, reg, t_rel, jump_t: float) -> dict:
    """Numerical checks of the 3D overlays in display coordinates: the board lies flat under the
    feet, the COP is on the board between the feet, the force arrow points up with the right
    length, the COM is above the COP with its plumb line on the board; in flight there is no
    COP / force."""
    out = {}
    k = int(np.argmin(np.abs(np.asarray(t_rel) - 1.5)))
    viewer._show_frame(k)
    it = viewer._bal_items
    assert viewer.balance_row == k
    corners = viewer._disp(reg.corners_world)
    bz = corners[:, 2]
    assert np.ptp(bz) < 0.01, f"board not flat in the view: {bz}"
    feet = np.array([viewer._pt(n) for n in FEET])
    gap = float(feet[:, 2].min() - bz.mean())
    assert 0.0 < gap < 0.06, f"feet {gap:.3f} m above the board top"
    lo, hi = corners[:, :2].min(axis=0), corners[:, :2].max(axis=0)
    assert (feet[:, :2] > lo - 0.02).all() and (feet[:, :2] < hi + 0.02).all(), "feet on board"
    cop = np.asarray(it["cop"].pos, float).reshape(-1, 3)[0]
    assert it["cop"].visible() and (cop[:2] > lo).all() and (cop[:2] < hi).all()
    assert abs(cop[2] - bz.mean()) < 0.01
    ankles = np.array([viewer._pt("RAnkle"), viewer._pt("LAnkle")])
    between = np.dot(cop[:2] - ankles[1, :2], ankles[0, :2] - ankles[1, :2]) / \
        np.dot(ankles[0, :2] - ankles[1, :2], ankles[0, :2] - ankles[1, :2])
    assert 0.2 < between < 0.8, f"COP not between the feet ({between:.2f})"
    arrow = np.asarray(it["force"].pos, float)
    vec = arrow[1] - arrow[0]
    assert it["force"].visible() and vec[2] / np.linalg.norm(vec) > 0.999, vec
    tip_len = np.linalg.norm(vec) + 0.5 * min(0.06, 0.4 * fused.total_kg[k] * 0.005)
    assert abs(tip_len - fused.total_kg[k] * 0.005) < 0.01, tip_len
    com = np.asarray(it["com"].pos, float).reshape(-1, 3)[0]
    plumb = np.asarray(it["plumb"].pos, float)
    assert 0.8 < com[2] - bz.mean() < 1.1, f"COM height {com[2] - bz.mean():.3f}"
    assert abs(plumb[1, 2] - bz.mean()) < 0.01 and np.allclose(plumb[1, :2], com[:2], atol=1e-6)
    assert np.linalg.norm(com[:2] - cop[:2]) < 0.05, "COM above the COP in quiet stance"
    out.update(stance_frame=k, feet_above_board_m=round(gap, 4),
               arrow_len_m=round(float(tip_len), 3), com_height_m=round(float(com[2] - bz.mean()), 3),
               cop_between_feet=round(float(between), 3))
    kf = int(np.argmin(np.abs(np.asarray(t_rel) - (jump_t + 0.15))))
    viewer._show_frame(kf)
    assert not it["force"].visible() and not it["cop"].visible(), "no force / COP in flight"
    feet_f = np.array([viewer._pt(n) for n in FEET])
    assert feet_f[:, 2].min() - bz.mean() > 0.05, "feet above the board in flight"
    out["flight_frame"] = kf
    viewer._show_frame(k)
    return out


def main(argv=None) -> int:
    import argparse
    import os

    import pytest

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("out", help="output folder (work data + screenshots)")
    ap.add_argument("--prefix", default="e2e_", help="screenshot file name prefix")
    ap.add_argument("--shots", default=None, help="screenshot folder (default: OUT)")
    ap.add_argument("--no-pose2sim", action="store_true",
                    help="write the .trc directly instead of running Pose2Sim on 4. Run")
    ap.add_argument("--z-down", action="store_true",
                    help="a Calib.toml world whose +Z points down (flipped calibration)")
    ap.add_argument("--unicode", action="store_true",
                    help="Chinese project folder and subject names")
    a = ap.parse_args(argv)
    os.environ.setdefault("MPLBACKEND", "Agg")
    out = Path(a.out)
    mp = pytest.MonkeyPatch()
    try:
        res = run_scenario(out / "work", mp, shots=Path(a.shots) if a.shots else out,
                           prefix=a.prefix, pose2sim=not a.no_pose2sim,
                           up_sign=-1 if a.z_down else 1,
                           **({"project_name": "\u5e73\u8861\u6d4b\u8bd5_\u8bd5\u9a8c1",
                               "subject": "\u53d7\u8bd5\u800501"}  # Chinese names
                              if a.unicode else {}))
    finally:
        mp.undo()
    print(json.dumps({k: v for k, v in res.items() if k != "screenshots"}, indent=1,
                     default=str))
    print(f"{len(res.get('screenshots', []))} screenshots in {a.shots or out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
