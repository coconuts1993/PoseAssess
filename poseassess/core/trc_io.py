"""Read 3D marker trajectories from a .trc file (Pose2Sim triangulation output).

TRC layout: 3 header rows, a marker-name row (row index 3, `Frame# Time <m1> <m2>
...`), a units/axis row, then data rows `frame time x1 y1 z1 x2 y2 z2 ...`.
Returns time-indexed per-marker Nx3 arrays for the 3D skeleton viewer.

Pose2Sim writes the data rows with pandas ``to_csv(sep="\t")`` without ``na_rep``: a missing
(NaN) coordinate is an EMPTY field between two tabs. Data rows are therefore split on every tab
(empty fields kept, -> NaN), so a missing marker never shifts the markers after it.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class TrcData:
    times: np.ndarray                 # (N,)
    markers: dict[str, np.ndarray]    # name -> (N, 3)
    source: Path

    @property
    def n_frames(self) -> int:
        return len(self.times)

    @property
    def marker_names(self) -> list[str]:
        return list(self.markers)

    def frame(self, i: int) -> dict[str, np.ndarray]:
        return {m: xyz[i] for m, xyz in self.markers.items()}

    def all_points(self) -> np.ndarray:
        """(N*M, 3) stacked — handy for computing bounds/centroid."""
        return np.concatenate([xyz for xyz in self.markers.values()], axis=0)


def _cell(v: str) -> float:
    """One data field: empty -> NaN; ValueError when it is not a number."""
    v = v.strip()
    return float(v) if v else float("nan")


def _data_row(line: str, n_vals: int) -> list[float] | None:
    """``[frame, time, x1, y1, z1, ...]`` of one data row (padded / cut to ``n_vals`` marker
    values), or None for a short or non-numeric row (blank line, axis row, text). Fields are
    split on every tab WITHOUT dropping empty ones (an empty field is a missing value)."""
    line = line.rstrip("\r\n")
    parts = line.split("\t") if "\t" in line else line.split()
    if len(parts) < 3:
        return None
    try:
        row = [_cell(p) for p in parts]
    except ValueError:
        return None
    if not np.isfinite(row[1]):  # no Time: not a data row
        return None
    vals = row[2:2 + n_vals]
    return row[:2] + vals + [float("nan")] * (n_vals - len(vals))


def read_trc(path: str | Path) -> TrcData:
    path = Path(path)
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if len(lines) < 6:
        raise ValueError(f"{path} too short to be a TRC file")

    name_row = [n for n in lines[3].split("\t")]
    # markers start after 'Frame#' and 'Time'
    names = [n.strip() for n in name_row if n.strip()]
    if names and names[0].lower().startswith("frame"):
        marker_names = names[2:]
    else:
        marker_names = names

    n_vals = 3 * len(marker_names)
    rows = [r for r in (_data_row(ln, n_vals) for ln in lines[5:]) if r is not None]
    arr = np.asarray(rows, dtype=float)
    if arr.ndim != 2 or arr.shape[0] == 0:
        raise ValueError(f"{path} has no numeric data rows")

    times = arr[:, 1]
    xyz = arr[:, 2:]
    markers: dict[str, np.ndarray] = {}
    for i, m in enumerate(marker_names):
        block = xyz[:, 3 * i:3 * i + 3]
        # skip all-NaN / empty markers
        if np.isfinite(block).any():
            markers[m] = block
    return TrcData(times=times, markers=markers, source=path)
