"""Read camera poses from a Pose2Sim Calib.toml for 3D visualization.

Each camera table stores `rotation` (Rodrigues), `translation` (metres), plus
`matrix`/`size` (intrinsics). The world->camera transform is x_cam = R x_world +
T, so the camera centre in world coordinates is C = -R^T T, and its optical axis
(camera +Z) points along R^T [0,0,1] in the world frame.
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
class CameraPose:
    name: str
    center: np.ndarray     # (3,) world position
    R: np.ndarray          # 3x3 world->camera rotation
    size: tuple[float, float]   # (w, h) px
    fx: float
    fy: float

    def image_corner_rays(self, depth: float = 0.25) -> np.ndarray:
        """4 world-space points of a small image rectangle in front of the camera.

        Uses the real intrinsics so the frustum's aspect/FOV matches the camera.
        """
        w, h = self.size
        # half-extents on the image plane at `depth`, from focal lengths
        hx = depth * (w / 2.0) / max(self.fx, 1.0)
        hy = depth * (h / 2.0) / max(self.fy, 1.0)
        corners_cam = np.array([
            [-hx, -hy, depth], [hx, -hy, depth],
            [hx, hy, depth], [-hx, hy, depth],
        ])
        # world = R^T (cam - T); but cam points already relative to camera frame
        # origin, and center = -R^T T, so world = R^T cam + center.
        return corners_cam @ self.R + self.center


def read_camera_poses(calib_toml: str | Path) -> list[CameraPose]:
    import cv2

    with open(calib_toml, "rb") as f:
        data = tomllib.load(f)
    poses = []
    for key, cam in data.items():
        if key.lower() == "metadata" or not isinstance(cam, dict):
            continue
        if "rotation" not in cam or "translation" not in cam:
            continue
        rvec = np.asarray(cam["rotation"], dtype=float).reshape(3)
        T = np.asarray(cam["translation"], dtype=float).reshape(3)
        R = cv2.Rodrigues(rvec)[0]
        center = -R.T @ T
        mat = np.asarray(cam.get("matrix", [[1, 0, 0], [0, 1, 0], [0, 0, 1]]), dtype=float)
        size = cam.get("size", [1920.0, 1080.0])
        poses.append(CameraPose(
            name=str(cam.get("name", key)),
            center=center, R=R,
            size=(float(size[0]), float(size[1])),
            fx=float(mat[0][0]), fy=float(mat[1][1]),
        ))
    return poses
