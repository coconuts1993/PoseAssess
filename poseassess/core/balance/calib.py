"""Camera calibration for board registration: reads PoseAssess / Pose2Sim ``Calib.toml``.

``CameraCalibration`` is ported from PoseBoard 3ea6d8a ``poseboard/calibration.py``.

Conventions (identical to PoseAssess's own calibration and to Pose2Sim):
* world = the extrinsic checkerboard frame, metres. Z = X x Y of the clicked checkerboard
  axes, so it is usually "up" but may point DOWN (``invert_z`` / click origin) — use
  ``world_up_sign`` instead of assuming +Z.
* extrinsics are world -> camera: ``X_cam = R @ X_world + t`` (``rotation`` = Rodrigues vector,
  ``translation`` in metres).
* camera identity is POSITIONAL: the i-th camera table of Calib.toml is camera i = ``camNN``
  (table names differ between producers: "1", "01", "cam01"...). Never match by name.
"""

from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib


@dataclass
class CameraCalibration:
    name: str
    image_size: tuple[int, int]  # (width, height)
    K: np.ndarray
    dist: np.ndarray
    rvec: np.ndarray | None = None  # world -> camera
    tvec: np.ndarray | None = None
    intrinsic_rms: float | None = None
    extrinsic_rms: float | None = None

    # ------------------------------------------------------------------ props
    @property
    def has_extrinsics(self) -> bool:
        return self.rvec is not None and self.tvec is not None

    @property
    def R(self) -> np.ndarray:
        return cv2.Rodrigues(np.asarray(self.rvec, dtype=np.float64))[0]

    @property
    def t(self) -> np.ndarray:
        return np.asarray(self.tvec, dtype=np.float64).reshape(3)

    @property
    def center_world(self) -> np.ndarray:
        """Camera optical center in world coordinates."""
        return -self.R.T @ self.t

    def projection_matrix(self, normalized: bool = False) -> np.ndarray:
        Rt = np.hstack([self.R, self.t.reshape(3, 1)])
        return Rt if normalized else self.K @ Rt

    # ------------------------------------------------------------ transforms
    def world_to_camera(self, pts_w: np.ndarray) -> np.ndarray:
        pts_w = np.asarray(pts_w, dtype=np.float64).reshape(-1, 3)
        return pts_w @ self.R.T + self.t

    def project(self, pts_w: np.ndarray) -> np.ndarray:
        """Project world points to image pixel coordinates (N,2)."""
        pts_w = np.asarray(pts_w, dtype=np.float64).reshape(-1, 3)
        if len(pts_w) == 0:
            return np.zeros((0, 2))
        img, _ = cv2.projectPoints(pts_w, np.asarray(self.rvec, np.float64),
                                   np.asarray(self.tvec, np.float64), self.K, self.dist)
        return img.reshape(-1, 2)

    def undistort_normalized(self, pts_px: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts_px, dtype=np.float64).reshape(-1, 1, 2)
        return cv2.undistortPoints(pts, self.K, self.dist).reshape(-1, 2)

    # ---------------------------------------------------------- (de)serialise
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "image_size": list(self.image_size),
            "K": np.asarray(self.K).tolist(),
            "dist": np.asarray(self.dist).ravel().tolist(),
            "rvec": None if self.rvec is None else np.asarray(self.rvec).ravel().tolist(),
            "tvec": None if self.tvec is None else np.asarray(self.tvec).ravel().tolist(),
            "intrinsic_rms": self.intrinsic_rms,
            "extrinsic_rms": self.extrinsic_rms,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CameraCalibration":
        return cls(
            name=d["name"],
            image_size=tuple(d["image_size"]),
            K=np.asarray(d["K"], dtype=np.float64),
            dist=np.asarray(d["dist"], dtype=np.float64),
            rvec=None if d.get("rvec") is None else np.asarray(d["rvec"], dtype=np.float64),
            tvec=None if d.get("tvec") is None else np.asarray(d["tvec"], dtype=np.float64),
            intrinsic_rms=d.get("intrinsic_rms"),
            extrinsic_rms=d.get("extrinsic_rms"),
        )


def load_calib_toml(path: str | Path) -> list[CameraCalibration]:
    """All camera tables of a Pose2Sim-format Calib.toml, in file order (camera i = index i-1).

    Distortion is padded to 5 coefficients; ``size`` gives ``image_size``; an all-zero
    rotation AND translation means "no extrinsics" (``rvec = tvec = None``). ``name`` is the
    table's ``name`` field (or the table key). The ``[metadata]`` table is skipped.
    """
    data = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    cams = []
    for key, d in data.items():
        if not isinstance(d, dict) or "matrix" not in d:
            continue
        dist = np.zeros(5)
        src = np.asarray(d.get("distortions", [0, 0, 0, 0]), np.float64).ravel()
        dist[: min(len(src), 5)] = src[:5]
        rvec = np.asarray(d["rotation"], np.float64).ravel() if "rotation" in d else None
        tvec = np.asarray(d["translation"], np.float64).ravel() if "translation" in d else None
        if rvec is not None and tvec is not None and not np.any(rvec) and not np.any(tvec):
            rvec = tvec = None  # all zeros: a placeholder, not a camera pose
        cams.append(CameraCalibration(
            name=str(d.get("name", key)),
            image_size=tuple(int(round(float(v))) for v in d.get("size", [0, 0])),
            K=np.asarray(d["matrix"], np.float64),
            dist=dist,
            rvec=rvec,
            tvec=tvec,
        ))
    return cams


def find_calib_toml(project) -> Path | None:
    """The calibration file PoseAssess uses: first of ``sorted(calibration/Calib*.toml)``
    (``Calib.toml`` before ``Calib_easymocap.toml``), same rule as the 3D View and the Verify
    tab. None when there is none."""
    d = (project.calibration_dir if hasattr(project, "calibration_dir")
         else Path(project) / "calibration")
    return next(iter(sorted(d.glob("Calib*.toml"))), None) if d.is_dir() else None


def pose2sim_calib_file(project) -> Path | None:
    """The calibration file Pose2Sim 0.10 triangulates with: the ``calibration/*.toml`` with the
    newest ``st_ctime`` (``triangulation.py`` / ``personAssociation.py``). Usually the same as
    ``find_calib_toml``; it differs e.g. after Pose2Sim's own calibration stage wrote
    ``Calib_easymocap.toml``. None when there is none."""
    d = (project.calibration_dir if hasattr(project, "calibration_dir")
         else Path(project) / "calibration")
    if not d.is_dir():
        return None
    files = []
    for f in d.glob("*.toml"):
        try:
            files.append((f.stat().st_ctime, f))
        except OSError:
            continue
    return max(files, key=lambda x: x[0])[1] if files else None


def extrinsics_differ(a: list[CameraCalibration], b: list[CameraCalibration],
                      rot_deg: float = 0.5, trans_m: float = 0.005) -> bool:
    """True when two calibrations place the cameras differently (camera count, extrinsics
    present or not, rotation > ``rot_deg`` or camera centre > ``trans_m`` apart)."""
    if len(a) != len(b):
        return True
    for ca, cb in zip(a, b):
        if ca.has_extrinsics != cb.has_extrinsics:
            return True
        if not ca.has_extrinsics:
            continue
        d = ca.R @ cb.R.T
        ang = np.degrees(np.arccos(np.clip((np.trace(d) - 1) / 2, -1.0, 1.0)))
        if ang > rot_deg or np.linalg.norm(ca.center_world - cb.center_world) > trans_m:
            return True
    return False


def calib_mismatch(project) -> str | None:
    """A warning when Pose2Sim will triangulate with another calibration file
    (``pose2sim_calib_file``) than the one PoseAssess uses for the board, the 3D View cameras
    and "Flip Z" (``find_calib_toml``), and their extrinsics differ; None otherwise (also
    when a file cannot be read)."""
    ours, theirs = find_calib_toml(project), pose2sim_calib_file(project)
    if ours is None or theirs is None or ours == theirs:
        return None
    try:
        if not extrinsics_differ(load_calib_toml(ours), load_calib_toml(theirs)):
            return None
    except Exception:  # noqa: BLE001  (malformed TOML: other checks report it)
        return None
    return (f"Pose2Sim will triangulate with calibration/{theirs.name} (the newest .toml), but "
            f"the board was located in calibration/{ours.name}, whose camera positions differ: "
            f"the Wii overlays would not match the skeleton. Delete or regenerate "
            f"{theirs.name}, then run the pipeline again.")


def project_cameras(project, calib_path: str | Path | None = None) -> dict[str, CameraCalibration]:
    """``{"cam01": CameraCalibration, ...}`` from the project's calibration file (positional:
    i-th table -> ``camNN``; ``.name`` is set to ``camNN``). {} when there is no calibration."""
    p = Path(calib_path) if calib_path else find_calib_toml(project)
    if p is None or not Path(p).is_file():
        return {}
    out = {}
    for i, cam in enumerate(load_calib_toml(p), start=1):
        cam.name = f"cam{i:02d}"
        out[cam.name] = cam
    return out


def calib_fingerprint(path: str | Path) -> dict:
    """``{"file": <name>, "sha1": <hex>, "mtime_ns": int}`` of a calibration file; used to
    notice that the board registration was computed with an older calibration."""
    p = Path(path)
    return {"file": p.name, "sha1": hashlib.sha1(p.read_bytes()).hexdigest(),
            "mtime_ns": p.stat().st_mtime_ns}


def world_up_sign(cams: list[CameraCalibration]) -> int:
    """+1 when world +Z points up (cameras above the checkerboard plane), -1 when it points down.

    Uses the median Z of the camera centres of cameras with extrinsics (cameras are mounted
    above the floor). Returns +1 when no camera has extrinsics.
    """
    zs = [float(c.center_world[2]) for c in cams if c.has_extrinsics]
    if not zs:
        return 1
    return 1 if float(np.median(zs)) >= 0 else -1
