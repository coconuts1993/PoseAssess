"""Robust .trc reader for the Wii fusion + frame conversions.

``poseassess.core.trc_io.read_trc`` (used by the existing 3D View) drops empty fields, so a row
with a missing (NaN) marker shifts columns, and it drops Frame#. It is left unchanged; the
balance code uses ``read_trc_full`` (port of PoseBoard 3ea6d8a ``pose/external.read_trc`` + the
Frame# column).

Frames
------
* Calib.toml world ("world", Z usually up): checkerboard frame, metres.
* Pose2Sim .trc ("trc", Y-up): ``(X', Y', Z') = (Y, Z, X)`` of the world
  (``Pose2Sim.common.zup2yup``); ``zup_to_yup`` / ``yup_to_zup`` convert points AND directions.
* 3D View display: ``Skeleton3DViewer._tf(zup_to_yup(p_world))`` for points,
  ``zup_to_yup(d_world) @ viewer._M.T`` for directions.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

ZUP_TO_YUP = (1, 2, 0)  # trc[..., k] = world[..., ZUP_TO_YUP[k]]
YUP_TO_ZUP = (2, 0, 1)  # world[..., k] = trc[..., YUP_TO_ZUP[k]]


def zup_to_yup(p: np.ndarray) -> np.ndarray:
    """Calib world (Z-up) -> Pose2Sim .trc frame (Y-up); works on (...,3) points or vectors."""
    return np.asarray(p, np.float64)[..., list(ZUP_TO_YUP)]


def yup_to_zup(p: np.ndarray) -> np.ndarray:
    """Pose2Sim .trc frame (Y-up) -> Calib world (Z-up); inverse of ``zup_to_yup``."""
    return np.asarray(p, np.float64)[..., list(YUP_TO_ZUP)]


def trc_sort_key(p: Path):
    """Same preference as the 3D View (``viz3d_page._trc_sort_key``): filtered first, raw next,
    LSTM-augmented last."""
    name = Path(p).name.lower()
    return ("lstm" in name, "filt" not in name, name)


def find_trc_files(project) -> list[Path]:
    """``pose-3d/*.trc`` of the project in 3D View order (the first is the default)."""
    d = project.pose3d_dir if hasattr(project, "pose3d_dir") else Path(project) / "pose-3d"
    return sorted(d.glob("*.trc"), key=trc_sort_key) if d.is_dir() else []


@dataclass
class TrcTrajectories:
    """A .trc file as arrays. ``coords`` are in metres in the file's own frame (Pose2Sim: Y-up)."""

    path: Path
    frames: np.ndarray          # (N,) int, Frame# column (Pose2Sim: 0-based video frame index)
    times: np.ndarray           # (N,) float, Time column (s)
    names: list[str]            # K marker names (all markers of the header, even all-NaN ones)
    coords: np.ndarray          # (N, K, 3) metres (Units mm/cm are scaled), NaN when missing
    data_rate: float            # DataRate header (Hz), NaN when absent
    header: dict                # raw header key -> value strings

    @property
    def n_frames(self) -> int:
        return len(self.times)

    def world(self) -> np.ndarray:
        """(N,K,3) in the Calib.toml world (Z-up) = ``yup_to_zup(coords)``."""
        return yup_to_zup(self.coords)

    def marker(self, name: str) -> np.ndarray | None:
        """(N,3) trajectory of one marker in the file's frame, or None."""
        try:
            return self.coords[:, self.names.index(name)]
        except ValueError:
            return None


def _num(v: str) -> float:
    v = v.strip()
    if not v or v.lower() in ("nan", "-nan", "none"):
        return np.nan
    try:
        return float(v)
    except ValueError:
        return np.nan


_UNIT_SCALE = {"m": 1.0, "meters": 1.0, "metres": 1.0, "dm": 0.1, "cm": 1e-2, "mm": 1e-3}


def read_trc_full(path: str | Path) -> TrcTrajectories:
    """Read a .trc: tab-split WITHOUT dropping empty fields (missing markers stay in their
    columns; short rows are NaN-padded), Units scaled to metres, Frame# kept (rows without a
    numeric Time are skipped; a missing Frame# becomes the row index). ValueError when the file
    is not a TRC or has no data rows."""
    path = Path(path)
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as e:
        raise ValueError(f"Cannot read {path.name}: {e}") from e
    # Header: "PathFileType ...", keys, values, "Frame#\tTime\t<names>", "\t\tX1\tY1\tZ1...".
    # The names line is located by its "Frame#" (some writers add lines before it).
    i_names = next((i for i, ln in enumerate(lines[:20])
                    if ln.split("\t", 1)[0].strip().lower().startswith("frame")), None)
    if i_names is None or i_names < 2 or not lines[0].strip().lower().startswith("pathfiletype"):
        raise ValueError(f"{path.name} is not a TRC file (no PathFileType / Frame# header)")
    keys = [k.strip() for k in lines[i_names - 2].split("\t")]
    vals = [v.strip() for v in lines[i_names - 1].split("\t")]
    header = {k: v for k, v in zip(keys, vals) if k}
    names = [n.strip() for n in lines[i_names].split("\t")[2:] if n.strip()]
    if not names:
        raise ValueError(f"{path.name}: no marker names in the TRC header")
    scale = _UNIT_SCALE.get(header.get("Units", "m").strip().lower(), 1.0)
    try:
        data_rate = float(header.get("DataRate", "nan"))
    except ValueError:
        data_rate = float("nan")

    n_vals = 3 * len(names)
    frames, times, data = [], [], []
    start = i_names + 1
    if start < len(lines) and not lines[start].split("\t", 1)[0].strip() and \
            any(c.strip()[:1].upper() == "X" for c in lines[start].split("\t")[2:4]):
        start += 1  # the "X1 Y1 Z1 ..." line
    for line in lines[start:]:
        line = line.rstrip("\r\n")
        if not line.strip():
            continue
        parts = line.split("\t") if "\t" in line else line.split()
        if len(parts) < 2:
            continue
        t = _num(parts[1])
        if not np.isfinite(t):
            continue
        f = _num(parts[0])
        frames.append(int(round(f)) if np.isfinite(f) else len(times))
        times.append(t)
        row = [_num(v) for v in parts[2:2 + n_vals]]
        row += [np.nan] * (n_vals - len(row))
        data.append(row)
    if not times:
        raise ValueError(f"{path.name}: the TRC file has no data rows")
    coords = np.asarray(data, np.float64).reshape(len(times), len(names), 3) * scale
    return TrcTrajectories(path, np.asarray(frames, np.int64), np.asarray(times, np.float64),
                           names, coords, data_rate, header)
