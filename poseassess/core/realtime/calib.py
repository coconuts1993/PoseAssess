"""Load a Pose2Sim ``Calib.toml`` into per-camera projection data for live triangulation.

Camera identity is positional (docs/WII_INTEGRATION.md §3): the NN-th camera table is
``camNN``. Each table has ``size`` [w, h], ``matrix`` (K), ``distortions`` (k1 k2 p1 p2),
``rotation`` (Rodrigues) and ``translation`` (m) with x_cam = R x_world + T.

Live frames may come at another resolution than the calibration images (e.g. calibrated at
1080p, streamed at 720p). Pixel coordinates are scaled to the calibrated size before
undistortion; that is exact when the camera scales the full sensor image (same aspect), and
``check_size`` warns otherwise.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib


@dataclass
class LiveCamera:
    name: str
    size: tuple[int, int]      # calibrated (w, h)
    K: np.ndarray              # 3x3
    dist: np.ndarray           # (k,) OpenCV distortion
    R: np.ndarray              # 3x3 world -> camera
    T: np.ndarray              # (3,)

    @property
    def P_norm(self) -> np.ndarray:
        """3x4 projection for undistorted *normalised* image coordinates: [R | T]."""
        return np.hstack([self.R, self.T.reshape(3, 1)])

    @property
    def P(self) -> np.ndarray:
        """3x4 pixel projection K [R | T] (no distortion)."""
        return self.K @ self.P_norm

    def scale_to_calib(self, frame_size: tuple[int, int]) -> tuple[float, float]:
        w, h = frame_size
        return self.size[0] / float(w), self.size[1] / float(h)

    def check_size(self, frame_size: tuple[int, int]) -> str | None:
        """Warning text when the live frame cannot simply be scaled to the calibrated size."""
        w, h = frame_size
        if (w, h) == tuple(self.size):
            return None
        if abs(w / h - self.size[0] / self.size[1]) > 0.01:
            return (f"{self.name}: live {w}x{h} has another aspect ratio than the calibration "
                    f"{self.size[0]}x{self.size[1]}; 3D will be wrong - stream at the "
                    "calibrated aspect ratio")
        return None

    def normalize(self, pts_px: np.ndarray, frame_size: tuple[int, int] | None = None) -> np.ndarray:
        """(N, 2) live pixels -> (N, 2) undistorted normalised coordinates."""
        import cv2

        pts = np.asarray(pts_px, dtype=np.float64).reshape(-1, 2)
        if frame_size is not None:
            sx, sy = self.scale_to_calib(frame_size)
            pts = pts * (sx, sy)
        if len(pts) == 0:
            return pts
        out = cv2.undistortPoints(pts.reshape(-1, 1, 2), self.K, self.dist)
        return out.reshape(-1, 2)

    def project(self, X: np.ndarray, frame_size: tuple[int, int] | None = None) -> np.ndarray:
        """(N, 3) world points -> (N, 2) pixels (with distortion) in the live frame."""
        import cv2

        X = np.asarray(X, dtype=np.float64).reshape(-1, 3)
        rvec = cv2.Rodrigues(self.R)[0]
        px = cv2.projectPoints(X, rvec, self.T.reshape(3, 1), self.K, self.dist)[0].reshape(-1, 2)
        if frame_size is not None:
            sx, sy = self.scale_to_calib(frame_size)
            px = px / (sx, sy)
        return px


def load_calib(calib_toml: str | Path) -> list[LiveCamera]:
    import cv2

    with open(calib_toml, "rb") as f:
        data = tomllib.load(f)
    cams: list[LiveCamera] = []
    for key, cam in data.items():
        if key.lower() == "metadata" or not isinstance(cam, dict):
            continue
        if not all(k in cam for k in ("matrix", "rotation", "translation")):
            continue
        rvec = np.asarray(cam["rotation"], dtype=float).reshape(3)
        size = cam.get("size", [1920, 1080])
        cams.append(LiveCamera(
            name=str(cam.get("name", key)),
            size=(int(round(float(size[0]))), int(round(float(size[1])))),
            K=np.asarray(cam["matrix"], dtype=float).reshape(3, 3),
            dist=np.asarray(cam.get("distortions", [0, 0, 0, 0]), dtype=float).reshape(-1),
            R=cv2.Rodrigues(rvec)[0],
            T=np.asarray(cam["translation"], dtype=float).reshape(3),
        ))
    if len(cams) < 2:
        raise ValueError(f"{calib_toml}: need at least 2 calibrated cameras, found {len(cams)}")
    return cams
