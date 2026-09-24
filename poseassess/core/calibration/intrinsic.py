"""Intrinsic camera calibration from checkerboard frames.

For each camera we detect the checkerboard in a set of still frames (extracted
from the intrinsic calibration clip at various board angles) and run
`cv2.calibrateCamera` to recover the intrinsic matrix K and distortion coeffs.

Board convention matches Pose2Sim's Config: `corners_nb = [rows, cols]` counts
the *inner* corners. OpenCV's patternSize is (cols, rows), handled here.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np


ProgressFn = Callable[[str], None]
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


@dataclass
class IntrinsicResult:
    camera_index: int
    K: np.ndarray            # 3x3
    dist: np.ndarray         # 1x5
    image_size: tuple[int, int]   # (width, height)
    rms_error: float
    frames_used: int
    frames_total: int


def _object_points(corners_nb: Sequence[int], square_size: float) -> np.ndarray:
    """3D coordinates of the board corners in board frame (z=0), millimetres."""
    rows, cols = int(corners_nb[0]), int(corners_nb[1])
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= float(square_size)
    return objp


def calibrate_intrinsic_camera(
    frames_dir: Path,
    corners_nb: Sequence[int],
    square_size: float,
    camera_index: int = 1,
    progress: Optional[ProgressFn] = None,
) -> IntrinsicResult:
    """Calibrate one camera from all checkerboard images in `frames_dir`."""
    import cv2

    frames_dir = Path(frames_dir)
    images = sorted(
        p for p in frames_dir.iterdir()
        if p.suffix.lower() in IMG_EXTS
    )
    if not images:
        raise FileNotFoundError(f"no calibration images in {frames_dir}")

    rows, cols = int(corners_nb[0]), int(corners_nb[1])
    pattern = (cols, rows)  # OpenCV wants (points_per_row, points_per_col)
    objp = _object_points(corners_nb, square_size)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    obj_points: list[np.ndarray] = []
    img_points: list[np.ndarray] = []
    image_size: Optional[tuple[int, int]] = None

    for p in images:
        img = cv2.imread(str(p))
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if image_size is None:
            image_size = (gray.shape[1], gray.shape[0])  # (w, h)
        found, corners = cv2.findChessboardCorners(
            gray, pattern,
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
        )
        if not found:
            if progress:
                progress(f"cam{camera_index:02d}: no board in {p.name}")
            continue
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        obj_points.append(objp)
        img_points.append(corners)
        if progress:
            progress(f"cam{camera_index:02d}: board found in {p.name} "
                     f"({len(obj_points)} usable)")

    if len(obj_points) < 3:
        raise RuntimeError(
            f"cam{camera_index:02d}: only {len(obj_points)} usable frames "
            f"(need >=3). Check board size / image quality."
        )

    rms, K, dist, _rvecs, _tvecs = cv2.calibrateCamera(
        obj_points, img_points, image_size, None, None
    )
    return IntrinsicResult(
        camera_index=camera_index,
        K=np.asarray(K, dtype=np.float64),
        dist=np.asarray(dist, dtype=np.float64).reshape(1, -1),
        image_size=image_size,
        rms_error=float(rms),
        frames_used=len(obj_points),
        frames_total=len(images),
    )
