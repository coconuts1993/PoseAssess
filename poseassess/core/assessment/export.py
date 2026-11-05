"""Export ALL measurable joint angles to CSV.

Two files, covering every coordinate the OpenSim IK produced (not just the
curated clinical subset):
  • <stem>_angles.csv   full time series: one row per frame, columns = time +
                        every joint coordinate.
  • <stem>_summary.csv  one row per coordinate: min, max, ROM, mean, peak|abs|.

Rotational coordinates are in degrees when the motion's `inDegrees=yes`
(translations like pelvis_tx/ty/tz stay in metres). A units row documents this.
OpenSim converts only *rotational* coordinates to degrees: translations stay in
metres and *coupled* coordinates (Pose2Sim's knee_angle_r_beta / _l_beta, the
patella angle = knee angle) stay in radians even in an inDegrees=yes file
(``coordinate_unit``).
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np

from .mot_io import MotionData
from .metrics import CoordMetric

# Coordinates that are translations (metres), not rotations (degrees): pelvis_tx/ty/tz and
# the Abdomen helper body's Abs_t1 / Abs_t2 (Pose2Sim's muscles_flex model).
_TRANSLATION_RE = re.compile(r"_(t[xyz]|t\d)$")


def _is_translation(coord: str) -> bool:
    return bool(_TRANSLATION_RE.search(coord))


def _coupled_unit(motion: MotionData, coord: str, ang_unit: str) -> str:
    """Unit of a coupled ``*_beta`` coordinate (= its base coordinate, e.g. knee_angle_r).

    OpenSim leaves coupled coordinates in radians in an inDegrees=yes file; decided from the
    data when the base coordinate is there (beta == base * pi/180 -> rad, beta == base -> the
    file's angle unit), else radians."""
    base = coord[: -len("_beta")]
    df = motion.df
    if base in df.columns:
        b = df[coord].to_numpy(dtype=float)
        k = df[base].to_numpy(dtype=float)
        ok = np.isfinite(b) & np.isfinite(k) & (np.abs(k) > 1.0)
        if ok.sum() >= 3:
            err_rad = float(np.median(np.abs(b[ok] - np.radians(k[ok]))))
            err_same = float(np.median(np.abs(b[ok] - k[ok])))
            return ang_unit if err_same < err_rad else "rad"
    return "rad"


def coordinate_unit(motion: MotionData, coord: str) -> str:
    """'m' for translations, else the angle unit of this coordinate ('deg' / 'rad')."""
    if _is_translation(coord):
        return "m"
    ang_unit = "deg" if motion.in_degrees else "rad"
    if ang_unit == "deg" and coord.endswith("_beta"):
        return _coupled_unit(motion, coord, ang_unit)
    return ang_unit


def export_angles_csv(motion: MotionData, path: str | Path) -> Path:
    """Write the full time series of every coordinate to CSV.

    Layout: header row (time, coord1, coord2, ...), a units row, then one row
    per frame. Time in seconds.
    """
    path = Path(path)
    df = motion.df
    coords = list(df.columns)

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time"] + coords)
        w.writerow(["s"] + [coordinate_unit(motion, c) for c in coords])
        t = df.index.to_numpy()
        vals = df.to_numpy()
        for i in range(len(t)):
            w.writerow([f"{t[i]:.5f}"] + [f"{x:.5f}" for x in vals[i]])
    return path


def all_coordinate_stats(motion: MotionData) -> list[CoordMetric]:
    """Per-coordinate min/max/ROM/mean/peak for EVERY coordinate."""
    return [CoordMetric.from_series(c, motion.df[c].to_numpy())
            for c in motion.df.columns]


def export_summary_csv(motion: MotionData, path: str | Path) -> Path:
    """Write a per-coordinate summary (min, max, ROM, mean, peak|abs|)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["coordinate", "unit", "min", "max", "ROM", "mean", "peak_abs"])
        for m in all_coordinate_stats(motion):
            unit = coordinate_unit(motion, m.coord)
            w.writerow([m.coord, unit, f"{m.min:.3f}", f"{m.max:.3f}",
                        f"{m.rom:.3f}", f"{m.mean:.3f}", f"{m.peak_abs:.3f}"])
    return path


def export_all(motion: MotionData, out_dir: str | Path, stem: str) -> dict[str, Path]:
    """Write both the time-series and summary CSVs; return their paths."""
    out_dir = Path(out_dir)
    return {
        "angles": export_angles_csv(motion, out_dir / f"{stem}_angles.csv"),
        "summary": export_summary_csv(motion, out_dir / f"{stem}_summary.csv"),
    }
