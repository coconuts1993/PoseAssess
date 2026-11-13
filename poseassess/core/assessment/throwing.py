"""Throwing (pitching) angles from the 3D keypoints of a .trc, independent of OpenSim.

Why
---
Pose2Sim's OpenSim model (Model_Pose2Sim_muscles_flex.osim) has the shoulder as a Z-X-Y
rotation sequence (arm_flex, arm_add, arm_rot) without limits. Near 90 deg abduction, the arm
position of a pitch, that sequence is in gimbal lock: arm_flex and arm_rot swap, and the IK
coordinates can drift by whole turns (arm_rot_r = 400 deg). The angles below are computed
directly from the keypoints with the definitions of pitching biomechanics, so they have neither
problem.

Markers
-------
``RShoulder, LShoulder, RHip, LHip`` (trunk, required: ``ValueError`` naming the missing ones)
and ``RElbow, RWrist`` / ``LElbow, LWrist`` (arms). These are the names of Pose2Sim's BODY_25B /
HALPE_26 / COCO .trc files (the ``_LSTM.trc`` contains them too). A side whose elbow or wrist is
missing from the file gives all-NaN columns for that side; with neither arm there is nothing to
compute (``ValueError``).

Definitions
-----------
Only vectors between markers are used, so the angles are the same in Pose2Sim's Y-up .trc or in
any rotated / translated right-handed frame (all Pose2Sim .trc files are right-handed), and in
any length unit (only directions matter).

* Trunk: ``up = unit(mid_shoulders - mid_hips)``; ``lateral = RShoulder - LShoulder`` made
  orthogonal to ``up``, unit (points to the subject's right); ``forward = up x lateral``
  (right-handed: up x right = forward).
* Upper arm ``u = unit(Elbow - Shoulder)``, forearm ``f = Wrist - Elbow``.
* ``elbow_flexion = angle(u, f)``: 0 deg = straight arm.
* ``shoulder_abduction = angle(u, -up)``: 0 deg = arm down along the trunk, 90 deg =
  horizontal, 180 deg = overhead (in any plane: abduction and flexion are not separated).
* ``shoulder_er`` (external rotation, as in pitching biomechanics): in the plane perpendicular
  to ``u``, ``e1 = unit(forward - (forward.u) u)``, ``e2 = s * (u x e1)`` with ``s = +1`` for the
  right arm and ``-1`` for the left arm (both sides positive for external rotation),
  ``fp = f - (f.u) u``, ``er = atan2(fp.e2, fp.e1)``. With the arm abducted 90 deg: 0 deg =
  forearm pointing forward, 90 deg = forearm up, about 180 deg = forearm pointing backward
  (maximum layback); negative = internal rotation (forearm down).

Undefined values are NaN: a needed marker is NaN; for ``shoulder_er`` also an elbow flexion
below 20 deg (forearm almost in line with the upper arm: the rotation about the upper arm is
undefined) and ``|forward - (forward.u) u| < 0.2`` (upper arm within about 12 deg of pointing
straight forward or backward: no reference direction).

Continuity: ``shoulder_er`` is unwrapped over time (``np.unwrap``, period 360) over the valid
samples, ACROSS NaN gaps: after a gap the value closest (modulo 360) to the last valid one is
kept, so 170 -> 190 stays 190 (not -170) even when a few frames with a straight elbow lie in
between. The whole series is then shifted by a multiple of 360 so that its median lies in
[-180, 180). This assumes less than 180 deg of rotation between two consecutive valid samples:
the arm's internal rotation after maximum layback can exceed 5000 deg/s, so the acceleration
phase needs a high frame rate (about 120 Hz or more) to be followed.

Limits
------
* ``shoulder_er`` needs a flexed elbow (>= 20 deg); around ball release and in the
  follow-through the elbow is nearly straight and the value is often missing.
* The trunk frame comes from the shoulders and hips only (no sternum / spine markers): trunk
  lateral bending or a shrugged shoulder tilt it.
* The keypoints are joint centres estimated from 2D pose models (triangulated), not anatomical
  markers: expect several degrees of error, more when a joint is hidden from most cameras.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd

from ..trc_io import TrcData, read_trc

TRUNK_MARKERS = ("RShoulder", "LShoulder", "RHip", "LHip")
ARM_MARKERS = {"r": ("RShoulder", "RElbow", "RWrist"), "l": ("LShoulder", "LElbow", "LWrist")}
THROWING_COLUMNS = ("shoulder_er_r", "shoulder_er_l",
                    "shoulder_abduction_r", "shoulder_abduction_l",
                    "elbow_flexion_r", "elbow_flexion_l")
MIN_ELBOW_FLEXION_DEG = 20.0   # below: shoulder_er undefined (forearm in line with the upper arm)
MIN_FORWARD_PERP = 0.2         # |forward - (forward.u) u| below: shoulder_er undefined
_EPS = 1e-12


def _as_trc(trc: TrcData | str | Path) -> TrcData:
    return trc if isinstance(trc, TrcData) else read_trc(trc)


def _marker(trc: TrcData, name: str) -> np.ndarray | None:
    """(N,3) trajectory of ``name`` (exact name, else case-insensitive), or None."""
    m = trc.markers.get(name)
    if m is None:
        m = next((v for k, v in trc.markers.items() if k.lower() == name.lower()), None)
    return None if m is None else np.asarray(m, dtype=float)


def _dot(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.einsum("ij,ij->i", a, b)


def _norm(v: np.ndarray) -> np.ndarray:
    return np.linalg.norm(v, axis=1)


def _unit(v: np.ndarray) -> np.ndarray:
    """Row-wise unit vectors; NaN rows where the vector is NaN or of zero length."""
    n = _norm(v)
    ok = n > _EPS
    out = np.full_like(v, np.nan, dtype=float)
    out[ok] = v[ok] / n[ok, None]
    return out


def _reject(v: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Component of ``v`` perpendicular to the unit vectors ``u`` (row-wise)."""
    return v - _dot(v, u)[:, None] * u


def _angle_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise angle between two vectors in degrees (0..180); NaN when one is NaN / zero."""
    return np.degrees(np.arccos(np.clip(_dot(_unit(a), _unit(b)), -1.0, 1.0)))


def trunk_axes(rshoulder: np.ndarray, lshoulder: np.ndarray, rhip: np.ndarray,
               lhip: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(up, lateral, forward)`` unit vectors (N,3) of the trunk: up = mid-hips -> mid-shoulders,
    lateral = LShoulder -> RShoulder made orthogonal to up (the subject's right), forward =
    up x lateral."""
    up = _unit((rshoulder + lshoulder) / 2.0 - (rhip + lhip) / 2.0)
    lateral = _unit(_reject(rshoulder - lshoulder, up))
    forward = np.cross(up, lateral)
    return up, lateral, forward


def continuous_degrees(values: np.ndarray) -> np.ndarray:
    """``values`` (deg) made continuous over time: ``np.unwrap`` (period 360) over the finite
    samples, across NaN gaps, then shifted by a multiple of 360 so the median is in
    [-180, 180). NaN stays NaN."""
    out = np.array(values, dtype=float)
    ok = np.isfinite(out)
    if not ok.any():
        return out
    v = np.unwrap(out[ok], period=360.0)
    v -= 360.0 * np.floor((np.median(v) + 180.0) / 360.0)
    out[ok] = v
    return out


def _arm_angles(shoulder: np.ndarray, elbow: np.ndarray, wrist: np.ndarray, up: np.ndarray,
                forward: np.ndarray, sign: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(shoulder_er, shoulder_abduction, elbow_flexion)`` in degrees of one arm (see the module
    docstring); ``sign`` = +1 right arm, -1 left arm."""
    u = _unit(elbow - shoulder)
    f = wrist - elbow
    flexion = _angle_deg(u, f)
    abduction = _angle_deg(u, -up)
    e1_raw = _reject(forward, u)
    e1 = _unit(e1_raw)
    e2 = sign * np.cross(u, e1)
    fp = _reject(f, u)
    er = np.degrees(np.arctan2(_dot(fp, e2), _dot(fp, e1)))
    undefined = ~(flexion >= MIN_ELBOW_FLEXION_DEG) | ~(_norm(e1_raw) >= MIN_FORWARD_PERP)
    er[undefined] = np.nan
    return continuous_degrees(er), abduction, flexion


def throwing_angles(trc: TrcData | str | Path) -> pd.DataFrame:
    """Throwing angles of a .trc (``TrcData`` or a path), in degrees, indexed by time (s):
    columns ``THROWING_COLUMNS`` (shoulder_er_r/l, shoulder_abduction_r/l,
    elbow_flexion_r/l). Definitions in the module docstring. ``ValueError`` when a trunk marker
    is missing, or the elbow / wrist of both arms."""
    trc = _as_trc(trc)
    src = getattr(trc, "source", None)
    where = f"{Path(src).name}: " if src else ""
    missing = [n for n in TRUNK_MARKERS if _marker(trc, n) is None]
    if missing:
        raise ValueError(f"{where}the .trc has no {', '.join(missing)} keypoint(s); the trunk "
                         f"(both shoulders and hips) is needed for the throwing angles.")
    rs, ls, rh, lh = (_marker(trc, n) for n in TRUNK_MARKERS)
    up, _lateral, forward = trunk_axes(rs, ls, rh, lh)

    n = len(trc.times)
    cols = {c: np.full(n, np.nan) for c in THROWING_COLUMNS}
    missing_arms: list[str] = []
    n_sides = 0
    for side, sign in (("r", 1.0), ("l", -1.0)):
        names = ARM_MARKERS[side]
        absent = [m for m in names[1:] if _marker(trc, m) is None]
        if absent:  # this side stays all-NaN
            missing_arms += absent
            continue
        er, abd, flex = _arm_angles(*(_marker(trc, m) for m in names), up, forward, sign)
        cols[f"shoulder_er_{side}"] = er
        cols[f"shoulder_abduction_{side}"] = abd
        cols[f"elbow_flexion_{side}"] = flex
        n_sides += 1
    if n_sides == 0:
        raise ValueError(f"{where}the .trc has no {', '.join(missing_arms)} keypoint(s); an "
                         f"elbow and a wrist are needed for the throwing angles.")
    index = pd.Index(np.asarray(trc.times, dtype=float), name="time")
    return pd.DataFrame(cols, index=index)


def write_throwing_csv(angles: pd.DataFrame, path: str | Path) -> Path:
    """Write ``throwing_angles`` output: header row (time + columns), units row (s, deg, ...),
    then one row per frame (%.5f, NaN as an empty cell)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["time"] + list(angles.columns))
        w.writerow(["s"] + ["deg"] * len(angles.columns))
        for t, row in zip(angles.index.to_numpy(dtype=float), angles.to_numpy(dtype=float)):
            w.writerow([f"{t:.5f}"] + [f"{x:.5f}" if np.isfinite(x) else "" for x in row])
    return path


def export_throwing_csv(trc: TrcData | str | Path, path: str | Path) -> Path:
    """Compute ``throwing_angles`` of ``trc`` (``TrcData`` or a .trc path) and write them to
    ``path`` (``write_throwing_csv`` layout); return the path."""
    return write_throwing_csv(throwing_angles(trc), path)
