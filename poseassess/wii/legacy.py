"""Reader for the text files of the lab's original Wii Balance Board program.

Format (whitespace separated, no header, one sample per line, ~70 Hz, irregular)::

    <ms> <TL kg> <TR kg> <BL kg> <BR kg> <COP x cm> <COP y cm> <total kg>

* ``ms``: a millisecond counter of that program. It does NOT restart per file (consecutive
  recordings continue the count) and it is not wall-clock time, so the files carry no computer
  time. When the counter is ``Environment.TickCount`` (ms since Windows booted), the boot time
  gives the computer time; otherwise it can only be estimated (see ``clock`` below).
* Sensors: WiimoteLib order TopLeft, TopRight, BottomLeft, BottomRight. The two COP columns
  match exactly ``x = (TR + BR - TL - BL) / total * 25`` and ``y = (BL + BR - TL - TR) / total * 15``
  (cm; x to the right, y to the BACK, i.e. screen coordinates, scaled to half the board size).
  PoseAssess recomputes the COP in its own board frame (x right, y front, metres, sensor
  spacing 433 x 238 mm) from the four sensors; the original values are kept as
  ``legacy_cop_x_cm`` / ``legacy_cop_y_cm`` in the returned columns.

Verify the sensor order once with a person standing on one corner: the corner's column must
carry the load.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np

N_COLS = 8
# Column order of the file -> PoseAssess sensor column names
LEGACY_SENSOR_ORDER = ("TL_kg", "TR_kg", "BL_kg", "BR_kg")
_NUM = re.compile(r"^[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?$")


def _numeric_rows(lines, limit=None):
    rows = []
    for line in lines:
        parts = line.replace(",", " ").split()
        if not parts:
            continue
        rows.append(parts)
        if limit is not None and len(rows) >= limit:
            break
    return rows


def is_legacy_wii_file(path: str | Path) -> bool:
    """True for a file in the original program's format (8 numeric columns, no header)."""
    p = Path(path)
    if not p.is_file():
        return False
    try:
        with open(p, encoding="utf-8-sig", errors="strict") as f:
            head = [next(f, "") for _ in range(20)]
    except (OSError, UnicodeDecodeError):
        return False
    rows = _numeric_rows(head, limit=10)
    return bool(rows) and all(len(r) == N_COLS and all(_NUM.match(v) for v in r) for r in rows)


def read_legacy_wii(path: str | Path, *, clock: str = "file_mtime",
                    counter_zero_unix: float | None = None,
                    sensor_dx_m: float = 0.433, sensor_dy_m: float = 0.238,
                    min_total_kg: float = 5.0) -> tuple[dict[str, np.ndarray], dict]:
    """Read a legacy file into PoseAssess ``wii.csv`` columns and a metadata dict.

    Columns: ``t`` (the file's counter in seconds), ``t_rel`` (s since the first sample),
    ``t_unix`` (estimated computer time, NaN when unknown), the four sensors, ``total_kg``,
    ``cop_x_board`` / ``cop_y_board`` (m, PoseAssess board frame), ``legacy_cop_x_cm`` /
    ``legacy_cop_y_cm``. Rows are kept in file order (the counter is non-decreasing).

    ``clock`` - how ``t_unix`` is obtained:
      * ``"file_mtime"`` (default): the file's modification time is taken as the time of the
        last sample (true when the program writes while recording and closes the file at the
        end, and the file was not modified or re-created since; a copy keeps it on Windows).
        Estimate, typically within about a second.
      * ``"counter_zero"``: ``t_unix = counter_zero_unix + ms / 1000`` - e.g. the Windows boot
        time when the counter is ``Environment.TickCount``.
      * ``"none"``: no computer time (align with sync events).
    """
    p = Path(path)
    with open(p, encoding="utf-8-sig") as f:
        rows = _numeric_rows(f)
    good = [r for r in rows if len(r) == N_COLS]
    if len(good) < 2:
        raise ValueError(f"{p.name}: fewer than 2 samples in the original Wii format")
    a = np.array([[float(v) for v in r] for r in good], np.float64)
    skipped = len(rows) - len(good)

    t = a[:, 0] / 1000.0
    cols: dict[str, np.ndarray] = {"t": t, "t_rel": t - t[0]}
    for j, name in enumerate(LEGACY_SENSOR_ORDER):
        cols[name] = a[:, 1 + j]
    tl, tr, bl, br = (cols[n] for n in LEGACY_SENSOR_ORDER)
    total = tl + tr + bl + br
    cols["total_kg"] = total
    with np.errstate(divide="ignore", invalid="ignore"):
        x = sensor_dx_m / 2 * ((tr + br) - (tl + bl)) / total
        y = sensor_dy_m / 2 * ((tr + tl) - (br + bl)) / total
    low = ~(total >= min_total_kg)
    x[low], y[low] = np.nan, np.nan
    cols["cop_x_board"], cols["cop_y_board"] = x, y
    cols["legacy_cop_x_cm"], cols["legacy_cop_y_cm"] = a[:, 5], a[:, 6]

    meta = {"legacy_format": "original Wii program (ms TL TR BL BR copx_cm copy_cm total)",
            "legacy_counter_first_ms": float(a[0, 0]), "legacy_counter_last_ms": float(a[-1, 0]),
            "legacy_rows_skipped": int(skipped), "clock_source": "none",
            "clock_offset_unix": None, "clock_estimated": True}
    if clock == "file_mtime":
        mtime = os.stat(p).st_mtime
        off = mtime - t[-1]
        meta.update(clock_source="file_mtime", clock_offset_unix=float(off),
                    file_mtime_unix=float(mtime),
                    clock_note="t_unix estimated: file modification time = last sample")
    elif clock == "counter_zero":
        if counter_zero_unix is None:
            raise ValueError("clock='counter_zero' needs counter_zero_unix")
        meta.update(clock_source="counter_zero", clock_offset_unix=float(counter_zero_unix),
                    clock_note="t_unix = counter zero time + counter")
    elif clock != "none":
        raise ValueError(f"clock must be file_mtime, counter_zero or none, not {clock!r}")
    off = meta["clock_offset_unix"]
    cols["t_unix"] = t + off if off is not None else np.full(len(t), np.nan)
    return cols, meta
