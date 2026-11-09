"""Red / yellow / green rating of an extrinsic calibration, per camera.

Used by the review step: green cameras can be accepted in one click, yellow ones
should be looked at, red ones must be fine-tuned (or re-detected) before the
calibration is accepted.

Three measurements, all from data the calibration already has:
  * own reprojection error - solvePnP of one camera against the flat board;
  * multi-view square-size error - corners triangulated from ALL cameras, mean
    neighbour spacing compared with the configured square size;
  * multi-view board flatness - max distance of those 3D corners to their plane.

The thresholds are first guesses for a 1080p camera ~3 m from the board and are
meant to be tuned on real data from the partner sites.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

# (green_below, red_above)
THRESHOLDS = {
    "own_px": (0.5, 1.5),
    "spacing_pct": (1.0, 3.0),
    "planarity_mm": (2.0, 5.0),
}

LEVELS = ("green", "yellow", "red")


def level_for(metric: str, value: float) -> str:
    g, r = THRESHOLDS[metric]
    if value < g:
        return "green"
    if value > r:
        return "red"
    return "yellow"


def _worst(levels: Sequence[str]) -> str:
    return max(levels, key=LEVELS.index) if levels else "green"


@dataclass
class CameraQuality:
    camera_index: int
    own_px: float
    multiview_px: Optional[float] = None
    level: str = "green"
    reasons: list[str] = field(default_factory=list)


@dataclass
class QualityReport:
    cameras: list[CameraQuality]
    spacing_pct: Optional[float] = None
    planarity_mm: Optional[float] = None
    level: str = "green"

    def summary_lines(self) -> list[str]:
        out = []
        for c in self.cameras:
            mv = f", multi-view {c.multiview_px:.2f}px" if c.multiview_px is not None else ""
            why = f"  ({'; '.join(c.reasons)})" if c.reasons else ""
            out.append(f"cam{c.camera_index:02d}: {c.level.upper():6s} own {c.own_px:.2f}px{mv}{why}")
        if self.spacing_pct is not None:
            out.append(f"square size error {self.spacing_pct:.2f}%  "
                       f"[{level_for('spacing_pct', self.spacing_pct)}]")
        if self.planarity_mm is not None:
            out.append(f"board flatness {self.planarity_mm:.1f} mm  "
                       f"[{level_for('planarity_mm', self.planarity_mm)}]")
        out.append(f"overall: {self.level.upper()}")
        return out


def assess(
    intr_by_cam: dict,
    corners_by_cam: dict,
    extrinsics: Sequence,
    corners_nb: Sequence[int],
    square_m: float,
    invert_z: bool = False,
) -> QualityReport:
    """Rate a solved calibration.

    `intr_by_cam[i]` exposes `.K` and `.dist`; `corners_by_cam[i]` is camera i's
    (N,2) corners in the order the solver used; `extrinsics` are the
    `ExtrinsicResult`s (rvec, T) from `align_extrinsics`.
    """
    from .extrinsic import _solve_pose, _object_points_x5
    from ..calib_check import consistency_report

    objp = _object_points_x5(corners_nb, square_m, invert_z=invert_z)
    cams_q: list[CameraQuality] = []
    cams_dicts, pts_list = [], []
    for res in extrinsics:
        i = res.camera_index
        K = np.asarray(intr_by_cam[i].K, np.float64)
        dist = np.asarray(intr_by_cam[i].dist, np.float64).reshape(1, -1)
        pts = np.asarray(corners_by_cam[i], np.float64).reshape(-1, 2)
        sol = _solve_pose(objp, pts, K, dist)
        own = sol[2] if sol is not None else float("inf")
        q = CameraQuality(i, float(own))
        lv = level_for("own_px", q.own_px)
        if lv != "green":
            q.reasons.append(f"own reprojection {q.own_px:.2f}px")
        q.level = lv
        cams_q.append(q)
        cams_dicts.append({"K": K, "dist": dist,
                           "rvec": np.asarray(res.rvec, np.float64).reshape(3, 1),
                           "tvec": np.asarray(res.T, np.float64).reshape(3, 1)})
        pts_list.append(pts)

    spacing_pct = planarity = None
    if len(cams_dicts) >= 2:
        try:
            rep = consistency_report(cams_dicts, pts_list)
        except Exception:  # noqa: BLE001 - degenerate geometry; report own errors only
            rep = None
        if rep is not None:
            spacing_pct = abs(rep["spacing_mm"] - square_m * 1000.0) / (square_m * 1000.0) * 100.0
            planarity = rep["planarity_mm"]
            for q, mv in zip(cams_q, rep["per_cam_px"]):
                q.multiview_px = float(mv)
                lv = level_for("own_px", q.multiview_px)
                if LEVELS.index(lv) > LEVELS.index(q.level):
                    q.level = lv
                if lv != "green":
                    q.reasons.append(f"multi-view reprojection {q.multiview_px:.2f}px")
    overall = [q.level for q in cams_q]
    if spacing_pct is not None:
        overall.append(level_for("spacing_pct", spacing_pct))
    if planarity is not None:
        overall.append(level_for("planarity_mm", planarity))
    return QualityReport(cams_q, spacing_pct, planarity, _worst(overall))
