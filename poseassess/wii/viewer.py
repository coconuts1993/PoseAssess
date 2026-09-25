"""Wii Balance Board recording viewer: replay the centre-of-pressure trajectory in real time.

Opens a file of the lab's original Wii program (``poseassess.wii.legacy``), a PoseAssess /
PoseBoard ``wii.csv`` or a recording folder, and plays it back at the recorded speed (or
faster / slower): top-down board view with the COP point and its recent trail, plus the load
and COP time curves with a cursor. No Bluetooth / board needed.

    python -m poseassess.wii.viewer [file]        (or run_wii_viewer.bat [file])

Opening a file jumps to 1 s before the subject steps on the board.
Keys: Space = play / pause, Left / Right = -/+ 1 s, Home = start.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MIN_KG = 5.0  # COP is only meaningful with somebody on the board
SPEEDS = ("0.25x", "0.5x", "1x", "2x", "4x")
TRAILS = {"1 s": 1.0, "3 s": 3.0, "5 s": 5.0, "10 s": 10.0, "30 s": 30.0, "All": None}


@dataclass
class WiiTrace:
    """One recording as arrays on a common time axis (s from the first sample)."""

    path: Path
    kind: str  # "original Wii program" / "wii.csv"
    t: np.ndarray  # (N,) s since the first sample
    total_kg: np.ndarray
    cop_x: np.ndarray  # (N,) m, board frame: + = subject's right; NaN when off the board
    cop_y: np.ndarray  # (N,) m, + = front (TL/TR edge)
    sensors: dict[str, np.ndarray]  # TL_kg, TR_kg, BL_kg, BR_kg
    t_unix: np.ndarray  # (N,) computer time or NaN
    clock_note: str = ""

    @property
    def duration(self) -> float:
        return float(self.t[-1] - self.t[0]) if len(self.t) > 1 else 0.0

    def index_at(self, t: float) -> int:
        """Index of the last sample at or before ``t``."""
        return int(np.clip(np.searchsorted(self.t, t, side="right") - 1, 0, len(self.t) - 1))


def load_trace(path: str | Path) -> WiiTrace:
    """Read a recording for viewing; raises ValueError for unsupported files."""
    from poseassess.wii import io as wio
    from poseassess.wii import legacy

    p = Path(path)
    if p.is_file() and legacy.is_legacy_wii_file(p):
        cols, meta = legacy.read_legacy_wii(p)
        kind = "original Wii program"
        note = ("computer time estimated from the file's modification time"
                if meta.get("clock_source") == "file_mtime" else "")
    elif (p.is_dir() and (p / wio.WII_CSV).is_file()) or (p.is_file() and p.suffix.lower() == ".csv"):
        cols = wio.read_wii_csv(p)
        kind = "wii.csv"
        note = ""
    else:
        raise ValueError(f"{p.name}: not a Wii recording (original Wii program file, wii.csv or "
                         "recording folder)")
    t = cols.get("t_rel")
    if t is None or not np.isfinite(t).any():
        t = cols["t"] - np.nanmin(cols["t"])
    ok = np.isfinite(t)
    t = np.asarray(t, float)[ok]
    if len(t) < 2:
        raise ValueError(f"{p.name}: fewer than 2 samples")
    order = np.argsort(t, kind="stable")

    def col(name):
        v = cols.get(name)
        v = np.full(ok.sum(), np.nan) if v is None else np.asarray(v, float)[ok]
        return v[order]

    t = t[order] - t[order][0]
    total = col("total_kg")
    x, y = col("cop_x_board"), col("cop_y_board")
    off = ~(total >= MIN_KG)
    x[off], y[off] = np.nan, np.nan
    sensors = {n: col(n) for n in ("TL_kg", "TR_kg", "BL_kg", "BR_kg")}
    return WiiTrace(p, kind, t, total, x, y, sensors, col("t_unix"), note)


def WiiViewer(path: str | Path | None = None):
    """Stand-alone viewer window; its ``replay`` attribute is the ``WiiReplayWidget``."""
    from PySide6.QtWidgets import QMainWindow

    from poseassess.gui.widgets.wii_replay import WiiReplayWidget

    w = QMainWindow()
    w.setWindowTitle("Wii Balance Board viewer")
    w.resize(1200, 700)
    w.replay = WiiReplayWidget()
    w.setCentralWidget(w.replay)
    if path:
        w.replay.load(path)
    return w


def main(argv=None) -> int:
    from PySide6.QtWidgets import QApplication

    argv = list(sys.argv[1:] if argv is None else argv)
    app = QApplication.instance() or QApplication(sys.argv[:1])
    files = [a for a in argv if not a.startswith("--")]
    w = WiiViewer(files[0] if files else None)
    w.show()
    w.replay.setFocus()
    if argv and w.replay.trace is not None and "--no-play" not in argv:
        w.replay.play()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
