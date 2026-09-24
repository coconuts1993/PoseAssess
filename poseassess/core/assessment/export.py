"""Export ALL measurable joint angles to CSV.

Two files, covering every coordinate the OpenSim IK produced (not just the
curated clinical subset):
  • <stem>_angles.csv   full time series: one row per frame, columns = time +
                        every joint coordinate.
  • <stem>_summary.csv  one row per coordinate: min, max, ROM, mean, peak|abs|.

Rotational coordinates are in degrees when the motion's `inDegrees=yes`
(translations like pelvis_tx/ty/tz stay in metres). A units row documents this.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .mot_io import MotionData
from .metrics import CoordMetric

# Coordinates that are translations (metres), not rotations (degrees).
_TRANSLATION_SUFFIXES = ("_tx", "_ty", "_tz")


def _is_translation(coord: str) -> bool:
    return coord.endswith(_TRANSLATION_SUFFIXES)


def export_angles_csv(motion: MotionData, path: str | Path) -> Path:
    """Write the full time series of every coordinate to CSV.

    Layout: header row (time, coord1, coord2, ...), a units row, then one row
    per frame. Time in seconds.
    """
    path = Path(path)
    df = motion.df
    coords = list(df.columns)
    ang_unit = "deg" if motion.in_degrees else "rad"

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time"] + coords)
        w.writerow(["s"] + [("m" if _is_translation(c) else ang_unit) for c in coords])
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
    ang_unit = "deg" if motion.in_degrees else "rad"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["coordinate", "unit", "min", "max", "ROM", "mean", "peak_abs"])
        for m in all_coordinate_stats(motion):
            unit = "m" if _is_translation(m.coord) else ang_unit
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
