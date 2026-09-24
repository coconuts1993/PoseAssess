"""Import OpenCap camera calibration into a Pose2Sim Calib.toml.

OpenCap stores per-camera calibration as a pickled dict
(`cameraIntrinsicsExtrinsics.pickle`) with keys:
    intrinsicMat (3x3), distortion (1x5), imageSize (2x1),
    rotation (3x3), translation (3x1, millimetres), rotation_EulerAngles (3x1).

Pose2Sim has converters for qualisys/easymocap/… but not OpenCap, so we write
the Calib.toml directly. Notes:
  * The stored `imageSize` is [h, w]; the true frame size (and the one the
    intrinsic principal point matches) is taken from the actual video.
  * Pose2Sim's `rotation` is a Rodrigues vector -> cv2.Rodrigues(R).
  * Translation is converted millimetres -> metres.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

try:
    import tomli_w
except ImportError:  # pragma: no cover
    tomli_w = None


def load_opencap_calib(pickle_path: Path) -> dict:
    with open(pickle_path, "rb") as f:
        return pickle.load(f)


def _cam_toml_entry(name: str, calib: dict, size_wh: tuple[int, int]) -> dict:
    import cv2

    K = np.asarray(calib["intrinsicMat"], dtype=float)
    dist = np.asarray(calib["distortion"], dtype=float).ravel()
    R = np.asarray(calib["rotation"], dtype=float)
    T = np.asarray(calib["translation"], dtype=float).ravel()

    rvec = cv2.Rodrigues(R)[0].ravel()
    # OpenCap translation is in millimetres; Pose2Sim expects metres.
    tvec_m = T / 1000.0
    # Pose2Sim uses [k1, k2, p1, p2] (4 coeffs); OpenCap has [k1,k2,p1,p2,k3].
    dist4 = dist[:4].tolist()

    return {
        "name": name,
        "size": [float(size_wh[0]), float(size_wh[1])],
        "matrix": [[float(K[i, j]) for j in range(3)] for i in range(3)],
        "distortions": [float(x) for x in dist4],
        "rotation": [float(x) for x in rvec],
        "translation": [float(x) for x in tvec_m],
        "fisheye": False,
    }


def opencap_to_calib_toml(
    cam_pickles: Sequence[Path],
    cam_sizes_wh: Sequence[tuple[int, int]],
    out_toml: Path,
    cam_names: Optional[Sequence[str]] = None,
) -> Path:
    """Write a Pose2Sim Calib.toml from a list of OpenCap calibration pickles.

    Args:
        cam_pickles: per-camera pickle paths, in camera order.
        cam_sizes_wh: per-camera true (width, height) from the actual videos.
        out_toml: output Calib.toml path.
        cam_names: optional camera names; defaults to cam01..camNN.
    """
    if tomli_w is None:
        raise RuntimeError("tomli-w required to write Calib.toml")
    if cam_names is None:
        cam_names = [f"cam{i:02d}" for i in range(1, len(cam_pickles) + 1)]

    doc: dict = {}
    for name, pk, size in zip(cam_names, cam_pickles, cam_sizes_wh):
        doc[name] = _cam_toml_entry(name, load_opencap_calib(pk), size)
    doc["metadata"] = {"adjusted": False, "error": 0.0}

    out_toml = Path(out_toml)
    out_toml.parent.mkdir(parents=True, exist_ok=True)
    with open(out_toml, "wb") as f:
        tomli_w.dump(doc, f)
    return out_toml
