"""Read OpenSim motion files (.mot / .sto) into a tidy DataFrame.

Both formats share the same layout: a small text header terminated by a line
`endheader`, then a tab-separated column-name row (first column `time`), then
numeric rows. The header key `inDegrees=yes|no` tells us the rotational unit.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd


@dataclass
class MotionData:
    """A loaded OpenSim motion: time-indexed joint coordinates + metadata."""
    df: pd.DataFrame                    # columns = coordinate names, index = time (s)
    in_degrees: bool
    source: Path
    header: dict = field(default_factory=dict)

    @property
    def coordinates(self) -> list[str]:
        return [c for c in self.df.columns]

    @property
    def duration(self) -> float:
        return float(self.df.index[-1] - self.df.index[0]) if len(self.df) else 0.0

    @property
    def n_frames(self) -> int:
        return len(self.df)


def read_mot(path: str | Path) -> MotionData:
    """Parse a .mot/.sto file into a MotionData."""
    path = Path(path)
    lines = path.read_text().splitlines()

    header: dict = {}
    end_idx = None
    for i, ln in enumerate(lines):
        s = ln.strip()
        if s.lower() == "endheader":
            end_idx = i
            break
        if "=" in s:
            k, _, v = s.partition("=")
            header[k.strip()] = v.strip()
    if end_idx is None:
        raise ValueError(f"{path} has no 'endheader' line — not an OpenSim motion file")

    col_line = lines[end_idx + 1]
    columns = [c.strip() for c in col_line.split("\t") if c.strip() != ""]

    rows = []
    for ln in lines[end_idx + 2:]:
        if not ln.strip():
            continue
        parts = [p for p in ln.replace("\t", " ").split() if p != ""]
        if len(parts) != len(columns):
            continue
        rows.append([float(p) for p in parts])

    df = pd.DataFrame(rows, columns=columns)
    if "time" in df.columns:
        df = df.set_index("time")

    in_degrees = str(header.get("inDegrees", "yes")).lower() == "yes"
    return MotionData(df=df, in_degrees=in_degrees, source=path, header=header)
