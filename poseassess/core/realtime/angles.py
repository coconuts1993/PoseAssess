"""Simple 3D joint angles from HALPE_26 keypoints (degrees).

These are geometric angles between segments, good for live feedback. Clinical joint angles
(with segment coordinate systems) remain the job of the offline OpenSim IK.

* knee / elbow flexion: 180 - angle(hip-knee-ankle) / (shoulder-elbow-wrist); 0 = straight.
* hip flexion: angle between the trunk (hip -> neck) and the thigh (hip -> knee), as 180 - x.
* shoulder: angle between the trunk (shoulder -> hip direction) and the upper arm.
* trunk lean: angle of hip -> neck from the vertical (``up`` axis of the calibration, +Z).
"""
from __future__ import annotations

import numpy as np

from .skeleton import IDX


def _angle(a: np.ndarray, b: np.ndarray) -> float:
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        return float("nan")
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return float("nan")
    c = float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


def joint_angles(X: np.ndarray, up=(0.0, 0.0, 1.0)) -> dict[str, float]:
    """X: (26, 3) HALPE_26 points (NaN = missing). Returns {name: degrees (NaN if unknown)}."""
    p = {n: X[i] for n, i in IDX.items()}
    out: dict[str, float] = {}
    for s in ("R", "L"):
        hip, knee, ank = p[s + "Hip"], p[s + "Knee"], p[s + "Ankle"]
        sho, elb, wri = p[s + "Shoulder"], p[s + "Elbow"], p[s + "Wrist"]
        out[s + "Knee"] = _angle(knee - hip, ank - knee)
        out[s + "Elbow"] = _angle(elb - sho, wri - elb)
        out[s + "Hip"] = _angle(p["Neck"] - p["Hip"], knee - hip)
        out[s + "Hip"] = 180.0 - out[s + "Hip"] if np.isfinite(out[s + "Hip"]) else out[s + "Hip"]
        out[s + "Shoulder"] = _angle(p["Hip"] - p["Neck"], elb - sho)
    out["Trunk"] = _angle(p["Neck"] - p["Hip"], np.asarray(up, dtype=float))
    return out


ANGLE_ORDER = ["RKnee", "LKnee", "RHip", "LHip", "RElbow", "LElbow", "RShoulder", "LShoulder",
               "Trunk"]
