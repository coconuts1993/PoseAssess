"""Project-level board registration: clicked landmarks per camera -> ``wii/board/board.json``.

Workflow (GUI: page "2b. Wii Board", tab "Board location"):
1. For each camera pick a frame where the board is visible (saved as ``wii/board/camNN/*.jpg``).
2. Click TL, TR, BR, BL, C (``LANDMARK_NAMES``) -> ``save_clicks`` -> ``wii/board/camNN/points.json``.
3. ``compute_board(project)``: every camera with 5 clicks + Calib.toml -> ``register_board`` ->
   ``BoardRegistration`` saved to ``wii/board/board.json``.
4. Optional corner check with the live board (``protocol.pressed_sensor``): if the subject
   pressed the corner that was clicked as BR while asked for TL, front and back were swapped:
   ``swap_front_back_clicks`` + ``compute_board`` again (equivalent to a 180° rotation of the
   board frame about its normal, but keeps the clicks and the result consistent).

points.json (per camera, compatible with ``calibration.extrinsic.load_clicked_points``)::

    {"points": [[x, y], ...up to 5],            # pixels of "image", order TL, TR, BR, BL, C
     "labels": ["TL", "TR", "BR", "BL", "C"],
     "image": "000123.jpg",                     # file in the same folder (or project-relative)
     "image_size": [w, h],
     "source": "videos/cam01.mp4#frame=123"}    # free text: where the frame came from

board.json (``BOARD_SCHEMA``)::

    {"schema": "poseassess.wii.board/1", "created": iso,
     "calib": {"file": "Calib.toml", "sha1": "...", "mtime_ns": 0,
               "pose2sim_file": "Calib.toml", "pose2sim_sha1": "..."},  # the file Pose2Sim
                                                # triangulates with (calib.pose2sim_calib_file)
     "world_frame": "Calib.toml world (checkerboard), metres",
     "world_up_sign": 1,                        # -1: Calib world +Z points down
     "geometry": {length_mm, width_mm, sensor_dx_mm, sensor_dy_mm, height_mm},
     "pose": BoardPose.to_dict(),               # board_to_world R/t, method, reproj_error_px,
                                                # world_camera, floor_constrained, tilt_deg,
                                                # warnings, notes
     "cameras_used": ["cam01", "cam02"],
     "clicks": {"cam01": <points.json content>, ...},   # snapshot used for this result
     "corners_world": [[x, y, z] x4],           # TL, TR, BR, BL outer surface corners
     "center_world": [x, y, z], "sensors_world": [[x, y, z] x4],  # sensors TR, BR, TL, BL
     "up_world": [x, y, z],
     "corner_check": null | {"result": "ok"|"swapped"|"redo", "pressed": "TL", "time": iso}}

The derived ``*_world`` arrays are for other tools (and humans); PoseAssess code rebuilds them
from ``pose`` + ``geometry``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .calib import (
    calib_fingerprint,
    calib_mismatch,
    find_calib_toml,
    pose2sim_calib_file,
    project_cameras,
    world_up_sign,
)
from .geometry import LANDMARK_NAMES, BoardGeometry, BoardPose, register_board
from .paths import WiiPaths, cam_index, cam_name
from .trial import now_iso

BOARD_SCHEMA = "poseassess.wii.board/1"
FRONT_BACK_SWAP = (2, 3, 0, 1, 4)  # new[i] = old[FRONT_BACK_SWAP[i]]: TL<->BR, TR<->BL, C


@dataclass
class CameraClicks:
    """Clicked landmarks of one camera (``points`` in ``LANDMARK_NAMES`` order, <= 5)."""

    cam: str
    points: list[tuple[float, float]] = field(default_factory=list)
    image: str | None = None
    image_size: tuple[int, int] | None = None
    source: str = ""

    @property
    def complete(self) -> bool:
        return len(self.points) == 5

    def as_array(self) -> np.ndarray | None:
        """(5,2) float array when complete, else None."""
        return np.asarray(self.points, np.float64).reshape(5, 2) if self.complete else None

    def swapped_front_back(self) -> CameraClicks:
        """Copy with TL<->BR and TR<->BL exchanged (only when complete; else unchanged copy)."""
        pts = ([self.points[j] for j in FRONT_BACK_SWAP] if self.complete else list(self.points))
        return CameraClicks(self.cam, pts, self.image, self.image_size, self.source)

    def to_dict(self) -> dict:
        return {"points": [[float(x), float(y)] for x, y in self.points],
                "labels": list(LANDMARK_NAMES[:len(self.points)]), "image": self.image,
                "image_size": None if self.image_size is None else list(self.image_size),
                "source": self.source}

    @classmethod
    def from_dict(cls, cam: str, d: dict) -> CameraClicks:
        size = d.get("image_size")
        return cls(cam, [(float(p[0]), float(p[1])) for p in d.get("points", [])][:5],
                   d.get("image"), None if not size else (int(size[0]), int(size[1])),
                   str(d.get("source") or ""))


@dataclass
class BoardRegistration:
    """The saved board registration (board.json). ``stale`` is runtime-only (not saved): a
    reason string when the calibration file changed since the board was computed."""

    pose: BoardPose
    geometry: BoardGeometry = field(default_factory=BoardGeometry)
    clicks: dict[str, CameraClicks] = field(default_factory=dict)
    calib: dict = field(default_factory=dict)
    world_up_sign: int = 1
    created: str = ""
    corner_check: dict | None = None
    stale: str | None = None

    @property
    def cameras_used(self) -> list[str]:
        return sorted(self.pose.reproj_error_px)

    @property
    def corners_world(self) -> np.ndarray:
        """(4,3) outer top-surface corners TL, TR, BR, BL in the Calib.toml world (m)."""
        return self.pose.board_to_world.apply(self.geometry.landmarks()[:4])

    @property
    def center_world(self) -> np.ndarray:
        """(3,) centre of the top surface (board origin) in the world (m)."""
        return self.pose.board_to_world.apply(self.geometry.landmarks()[4:5])[0]

    @property
    def sensors_world(self) -> np.ndarray:
        """(4,3) sensor positions TR, BR, TL, BL (same order as the Wii data) in the world."""
        return self.pose.board_to_world.apply(self.geometry.sensors())

    @property
    def up_world(self) -> np.ndarray:
        """(3,) unit board normal (physically up) in the world."""
        return self.pose.up_world

    def to_dict(self) -> dict:
        return {"schema": BOARD_SCHEMA, "created": self.created, "calib": dict(self.calib),
                "world_frame": "Calib.toml world (checkerboard), metres",
                "world_up_sign": int(self.world_up_sign), "geometry": self.geometry.to_dict(),
                "pose": self.pose.to_dict(), "cameras_used": self.cameras_used,
                "clicks": {k: v.to_dict() for k, v in sorted(self.clicks.items())},
                "corners_world": self.corners_world.tolist(),
                "center_world": self.center_world.tolist(),
                "sensors_world": self.sensors_world.tolist(),
                "up_world": self.up_world.tolist(), "corner_check": self.corner_check}

    @classmethod
    def from_dict(cls, d: dict) -> BoardRegistration:
        geo = BoardGeometry(**{k: float(v) for k, v in (d.get("geometry") or {}).items()
                               if k in BoardGeometry.__dataclass_fields__})
        return cls(BoardPose.from_dict(d["pose"]), geo,
                   {k: CameraClicks.from_dict(k, v) for k, v in (d.get("clicks") or {}).items()},
                   dict(d.get("calib") or {}), int(d.get("world_up_sign", 1)),
                   str(d.get("created") or ""), d.get("corner_check"))


def load_clicks(project, index: int) -> CameraClicks | None:
    """Clicks of camera ``index`` (1-based) from wii/board/camNN/points.json, or None."""
    f = WiiPaths(project).board_points_file(index)
    try:
        return CameraClicks.from_dict(cam_name(index), json.loads(f.read_text(encoding="utf-8")))
    except (OSError, ValueError, KeyError, TypeError, IndexError):
        return None


def save_clicks(project, index: int, clicks: CameraClicks) -> Path:
    """Write wii/board/camNN/points.json (creates the folder). Returns its path."""
    f = WiiPaths(project).board_points_file(index)
    f.parent.mkdir(parents=True, exist_ok=True)
    _write_json(f, clicks.to_dict())
    return f


def load_all_clicks(project) -> dict[str, CameraClicks]:
    """``{"camNN": CameraClicks}`` for every camera folder with a points.json."""
    d = WiiPaths(project).board_dir
    out = {}
    if d.is_dir():
        for sub in sorted(d.iterdir()):
            try:
                i = cam_index(sub.name)
            except ValueError:
                continue
            c = load_clicks(project, i)
            if c is not None:
                out[c.cam] = c
    return out


def swap_front_back_clicks(project) -> list[str]:
    """Swap TL<->BR and TR<->BL in every complete points.json; returns the cameras changed."""
    changed = []
    for cam, c in load_all_clicks(project).items():
        if c.complete:
            save_clicks(project, cam_index(cam), c.swapped_front_back())
            changed.append(cam)
    return changed


def compute_board(project, geometry: BoardGeometry | None = None, floor: bool = True,
                  save: bool = True, keep_corner_check: bool = False) -> BoardRegistration:
    """Register the board from all saved clicks and the project's calibration.

    Uses ``calib.find_calib_toml`` + ``calib.project_cameras`` (camera i = camNN), every camera
    with 5 clicks, ``geometry`` (default: the geometry of the existing board.json, else
    ``BoardGeometry()``) and ``geometry.register_board(..., up_sign=calib.world_up_sign(cams))``.
    Clicks made on an image whose ``image_size`` clearly differs from the Calib.toml ``size`` of
    that camera are not used (listed in ``pose.notes``); a size within max(8 px, 1 %) (or of
    2 x the principal point: an approximate ``size``) is accepted with a note. Saves board.json when ``save``.
    ``keep_corner_check`` keeps the corner check result of the existing board.json (default:
    cleared, since the result may change). Raises ValueError with a user-readable message (no
    calibration, no complete clicks, mirrored click order, ...).
    """
    calib_file = find_calib_toml(project)
    if calib_file is None:
        raise ValueError("No camera calibration (calibration/Calib.toml) in this project: "
                         "calibrate the cameras first (2. Calibration).")
    try:
        cams = project_cameras(project, calib_file)
    except Exception as e:  # noqa: BLE001  (malformed TOML, missing keys...)
        raise ValueError(f"Cannot read {calib_file.name}: {e}") from e
    if not cams:
        raise ValueError(f"{calib_file.name} contains no camera.")
    old = load_board(project)
    geo = geometry or (old.geometry if old is not None else BoardGeometry())
    all_clicks = load_all_clicks(project)

    notes: list[str] = []
    used_cams, used_pts, used_clicks = [], [], {}
    for name, cam in cams.items():
        c = all_clicks.get(name)
        if c is None or not c.complete:
            continue
        cal_size = tuple(int(v) for v in cam.image_size)
        if c.image_size and all(cal_size) and tuple(c.image_size) != cal_size:
            if not _size_compatible(c.image_size, cal_size, cam.K):
                notes.append(f"{name}: clicks not used - they were made on a "
                             f"{c.image_size[0]}x{c.image_size[1]} image, but this camera is "
                             f"calibrated at {cal_size[0]}x{cal_size[1]}. Use a frame of the "
                             "calibrated resolution.")
                continue
            notes.append(f"{name}: the {calib_file.name} size {cal_size[0]}x{cal_size[1]} is "
                         "approximate (some converters write 2 x the principal point); the "
                         f"clicks on the {c.image_size[0]}x{c.image_size[1]} image are used.")
        used_cams.append(cam)
        used_pts.append(c.as_array())
        used_clicks[name] = c
    extra = sorted(k for k, c in all_clicks.items() if k not in cams and c.complete)
    if extra:
        notes.append(f"Clicks of {', '.join(extra)} not used: {calib_file.name} has only "
                     f"{len(cams)} camera(s).")
    if not used_cams:
        incomplete = sorted(k for k, c in all_clicks.items() if not c.complete and c.points)
        msg = "No camera has all 5 board points (TL, TR, BR, BL, C) clicked"
        if incomplete:
            msg += f" (incomplete: {', '.join(incomplete)})"
        raise ValueError(msg + ("." if not notes else ". " + " ".join(notes)))

    up_sign = world_up_sign(list(cams.values()))
    try:
        pose = register_board(geo, used_cams, used_pts, floor=floor, up_sign=up_sign)
    except ValueError:
        raise
    except (RuntimeError, cv2.error, np.linalg.LinAlgError) as e:
        raise ValueError(f"The board position could not be computed ({e}). Check the clicks "
                         "(order TL, TR, BR, BL, C) and redo them if needed.") from e
    pose.notes = notes + list(pose.notes)
    mismatch = calib_mismatch(project)
    if mismatch:
        pose.warnings = list(pose.warnings) + [mismatch]
    used = set(pose.reproj_error_px)
    reg = BoardRegistration(
        pose=pose, geometry=geo,
        clicks={k: v for k, v in used_clicks.items() if k in used},
        calib=_calib_record(project, calib_file), world_up_sign=up_sign, created=now_iso(),
        corner_check=(old.corner_check if keep_corner_check and old is not None else None))
    if save:
        save_board(project, reg)
    return reg


def _size_compatible(image_size, cal_size, K=None) -> bool:
    """Whether clicks made on an ``image_size`` (w, h) image fit a camera whose Calib.toml
    ``size`` is ``cal_size``: equal within max(8 px, 1 %) per axis, or ``image_size`` within
    that tolerance of 2 x the principal point (Pose2Sim's easymocap converter and others write
    ``size = [2 cx, 2 cy]``, e.g. 1919x1084 for a 1920x1080 camera)."""
    w, h = (float(v) for v in image_size)

    def near(a, b):
        return all(abs(x - y) <= max(8.0, 0.01 * x) for x, y in zip((w, h), (a, b)))

    if near(*cal_size):
        return True
    if K is not None:
        K = np.asarray(K, np.float64)
        return bool(K.shape == (3, 3) and near(2 * K[0, 2], 2 * K[1, 2]))
    return False


def _calib_record(project, calib_file: Path) -> dict:
    """board.json "calib": fingerprint of the file used + name and sha1 of the file Pose2Sim
    triangulates with."""
    rec = calib_fingerprint(calib_file)
    p2s = pose2sim_calib_file(project)
    if p2s is not None:
        rec["pose2sim_file"] = p2s.name
        rec["pose2sim_sha1"] = (rec["sha1"] if p2s == Path(calib_file)
                                else calib_fingerprint(p2s)["sha1"])
    return rec


def load_board(project) -> BoardRegistration | None:
    """board.json as ``BoardRegistration`` (``stale`` set when the current calibration file's
    sha1 differs from ``calib["sha1"]`` or it is missing, or when the file Pose2Sim
    triangulates with changed since), or None when not registered / unreadable."""
    f = WiiPaths(project).board_json
    try:
        reg = BoardRegistration.from_dict(json.loads(f.read_text(encoding="utf-8")))
    except (OSError, ValueError, KeyError, TypeError):
        return None
    calib = find_calib_toml(project)
    if calib is None:
        reg.stale = "The camera calibration (Calib.toml) is missing."
    elif reg.calib.get("sha1") and calib_fingerprint(calib)["sha1"] != reg.calib["sha1"]:
        reg.stale = ("The camera calibration changed after the board was located: "
                     "recompute the board position (2b. Wii Board).")
    elif reg.calib.get("pose2sim_sha1"):
        p2s = pose2sim_calib_file(project)
        sha = None if p2s is None else (reg.calib["sha1"] if p2s == calib
                                        else calib_fingerprint(p2s)["sha1"])
        if sha != reg.calib["pose2sim_sha1"]:
            reg.stale = (f"The calibration file Pose2Sim uses "
                         f"({p2s.name if p2s is not None else 'none'}) changed after the board "
                         "was located: recompute the board position (2b. Wii Board).")
    return reg


def save_board(project, reg: BoardRegistration) -> Path:
    """Write board.json atomically. Returns its path."""
    f = WiiPaths(project).board_json
    f.parent.mkdir(parents=True, exist_ok=True)
    _write_json(f, reg.to_dict())
    return f


def set_corner_check(project, result: str, pressed: str | None) -> BoardRegistration:
    """Record the corner check (``result`` in "ok" | "swapped" | "redo") in board.json and
    return the updated registration. ValueError when no board is registered."""
    if result not in ("ok", "swapped", "redo"):
        raise ValueError(f"unknown corner check result {result!r}")
    reg = load_board(project)
    if reg is None:
        raise ValueError("No board position computed yet")
    reg.corner_check = {"result": result, "pressed": pressed, "time": now_iso()}
    save_board(project, reg)
    return reg


def _write_json(path: Path, data: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def board_paths(project) -> WiiPaths:
    """Shortcut for ``WiiPaths(project)`` (board_dir, board_cam_dir(i), board_json...)."""
    return WiiPaths(project)
