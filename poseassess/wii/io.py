"""Recording folder format: file names, CSV columns and readers.

This module is the single source of truth for the files one recording ("take") consists of.
Writers (``poseassess.wii.recorder``, ``poseassess.core.capture``) and readers
(``poseassess.core.balance``, GUI) import the constants from here.

A recording folder (``<project>/wii/recordings/<YYYYmmdd_HHMMSS[_subject]>/``) holds::

    session.json            metadata (clock, devices, streams, export info); SESSION_SCHEMA
    wii.csv                 WII_HEADER, ~100 Hz, only when a board was connected at some point
    events.csv              EVENTS_HEADER, UTF-8 with BOM, only when events were marked
    camNN.mkv               raw camera video (MPEG-4 in Matroska; AVI/MJPG fallback)
    camNN_timestamps.csv    TIMESTAMPS_HEADER, one row per frame written to camNN.mkv
    frames.csv              FRAMES_HEADER_BASE + per camera columns, written by the capture
                            export: one row per frame of the exported videos/camNN.mp4

Clock: every time column uses one clock per recording. ``t`` = ``time.perf_counter()`` of the
recording PC (s), ``t_rel = t - t0`` (s since the recording started), ``t_unix = t +
clock_offset_unix`` (Unix time, s; the offset is measured once at start). Empty fields are NaN.
Imported recordings (another PC / PoseBoard) keep their own ``t``; only ``t_rel`` and
``t_unix`` are meaningful across programs. The last column ``time_local`` repeats ``t_unix`` as the
recording PC's local date and time (``2026-09-25 14:03:12.345+08:00``) so the computer time is
readable in Excel; readers use ``t_unix``.
"""

from __future__ import annotations

import csv
import json
import os
import time
from datetime import datetime
from io import StringIO
from pathlib import Path

import numpy as np

SESSION_JSON = "session.json"
WII_CSV = "wii.csv"
EVENTS_CSV = "events.csv"
FRAMES_CSV = "frames.csv"
TIMESTAMPS_SUFFIX = "_timestamps.csv"

SESSION_SCHEMA = "poseassess.wii.session/1"

TIME_LOCAL = "time_local"  # last column of every time-stamped CSV: t_unix as local date/time
WII_HEADER = ["t", "t_rel", "t_unix", "TR_kg", "BR_kg", "TL_kg", "BL_kg", "total_kg",
              "cop_x_board", "cop_y_board", "cop_x_world", "cop_y_world", "cop_z_world",
              TIME_LOCAL]
SENSOR_COLS = ["TR_kg", "BR_kg", "TL_kg", "BL_kg"]  # same order as ForceSample.kg
EVENTS_HEADER = ["t", "t_rel", "t_unix", "label", TIME_LOCAL]
TIMESTAMPS_HEADER = ["frame", "t", "t_rel", "t_unix", TIME_LOCAL]
# frames.csv: FRAMES_HEADER_BASE, then for each exported camera "<cam>_src" (source frame index
# in <cam>.mkv, -1 when none) and "<cam>_dt_ms" (source frame time minus grid time, ms), then
# TIME_LOCAL.
FRAMES_HEADER_BASE = ["frame", "t", "t_rel", "t_unix"]


def local_time(t_unix: float | None) -> str:
    """``t_unix`` as the local date and time of this PC with milliseconds and UTC offset,
    e.g. ``2026-09-25 14:03:12.345+08:00``; "" when unknown."""
    try:
        if t_unix is None or not np.isfinite(t_unix):
            return ""
        return datetime.fromtimestamp(float(t_unix)).astimezone().isoformat(
            sep=" ", timespec="milliseconds")
    except (OverflowError, OSError, ValueError):
        return ""


def timestamps_csv_name(cam: str) -> str:
    """``cam01`` -> ``cam01_timestamps.csv`` (the name ``CameraStream`` writes next to cam01.mkv)."""
    return f"{cam}{TIMESTAMPS_SUFFIX}"


def is_recording_folder(folder: str | Path) -> bool:
    """True when ``folder`` looks like a recording (has session.json or wii.csv)."""
    f = Path(folder)
    return f.is_dir() and ((f / SESSION_JSON).is_file() or (f / WII_CSV).is_file())


def _num(v: str | None) -> float:
    if v is None:
        return np.nan
    v = v.strip()
    if not v:
        return np.nan
    try:
        return float(v)
    except ValueError:
        return np.nan


def read_csv_columns(path: str | Path, complete_rows_only: bool = False) -> dict[str, np.ndarray]:
    """Read a numeric CSV (header row + rows) into ``{column: float array}``; empty or
    non-numeric fields become NaN. Handles a UTF-8 BOM. Missing trailing fields are NaN.

    ``complete_rows_only``: drop a last row that has no line terminator. Files written row by
    row (wii.csv, camNN_timestamps.csv) end in the middle of a row when the recording process
    was killed; such a row may hold truncated numbers."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        text = f.read()
    rows = list(csv.reader(StringIO(text, newline="")))
    if complete_rows_only and len(rows) > 1 and text and text[-1] not in "\r\n":
        rows = rows[:-1]
    if not rows:
        return {}
    head = [h.strip() for h in rows[0]]
    body = [r for r in rows[1:] if any(c.strip() for c in r)]
    cols: dict[str, np.ndarray] = {}
    for j, h in enumerate(head):
        cols[h] = np.array([_num(r[j]) if j < len(r) else np.nan for r in body], np.float64)
    return cols


def read_wii_csv(path: str | Path) -> dict[str, np.ndarray]:
    """Read wii.csv (a file or a recording folder) as ``{column: array}`` with every
    ``WII_HEADER`` column present (missing ones are all-NaN). Rows are sorted by ``t_rel``
    (or ``t``); a last row cut off by a crash is dropped. Raises FileNotFoundError when there
    is no wii.csv."""
    p = Path(path)
    if p.is_dir():
        p = p / WII_CSV
    cols = read_csv_columns(p, complete_rows_only=True)
    n = len(next(iter(cols.values()))) if cols else 0
    for h in WII_HEADER:
        cols.setdefault(h, np.full(n, np.nan))
    key = "t_rel" if np.isfinite(cols["t_rel"]).any() else "t"
    order = np.argsort(np.where(np.isfinite(cols[key]), cols[key], np.inf), kind="stable")
    return {k: v[order] for k, v in cols.items()}


def read_events_csv(path: str | Path) -> list[dict]:
    """Read events.csv (a file or a recording folder): ``[{t, t_rel, t_unix, label}]`` sorted
    by ``t_rel``; times are floats (NaN when empty). Returns [] when the file does not exist."""
    p = Path(path)
    if p.is_dir():
        p = p / EVENTS_CSV
    if not p.is_file():
        return []
    with open(p, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    out = [{"t": _num(r.get("t")), "t_rel": _num(r.get("t_rel")), "t_unix": _num(r.get("t_unix")),
            "label": (r.get("label") or "").strip()} for r in rows]
    return sorted(out, key=lambda e: e["t_rel"] if np.isfinite(e["t_rel"]) else np.inf)


def read_timestamps_csv(path: str | Path) -> dict[str, np.ndarray]:
    """Read a ``camNN_timestamps.csv`` as ``{frame, t, t_rel, t_unix}`` arrays (a last row cut
    off by a crash is dropped)."""
    cols = read_csv_columns(path, complete_rows_only=True)
    n = len(next(iter(cols.values()))) if cols else 0
    for h in TIMESTAMPS_HEADER:
        cols.setdefault(h, np.full(n, np.nan))
    return cols


def read_frames_csv(path: str | Path) -> dict[str, np.ndarray]:
    """Read frames.csv (a file or a recording folder) as ``{column: array}``."""
    p = Path(path)
    if p.is_dir():
        p = p / FRAMES_CSV
    return read_csv_columns(p)


def read_session_json(folder: str | Path) -> dict:
    """session.json of a recording folder as a dict ({} when missing or unreadable)."""
    p = Path(folder) / SESSION_JSON
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _json_default(o):
    """JSON fallback for values that often end up in metadata: numpy scalars / arrays, paths,
    bytes (HID paths)."""
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, Path):
        return o.as_posix()
    if isinstance(o, (bytes, bytearray)):
        return o.decode("utf-8", "replace")
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


def write_session_json(folder: str | Path, meta: dict) -> Path:
    """Write session.json (UTF-8, indented). Returns its path.

    The file is replaced atomically (temporary file + ``os.replace``), so a reader or a crash
    never sees a half-written file. numpy values, paths and bytes are converted. If another
    program keeps the file open so it cannot be replaced (Windows), it is written in place."""
    p = Path(folder) / SESSION_JSON
    text = json.dumps(meta, indent=2, ensure_ascii=False, default=_json_default)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    for _ in range(10):
        try:
            os.replace(tmp, p)
            return p
        except PermissionError:
            time.sleep(0.02)
    try:
        p.write_text(text, encoding="utf-8")
    finally:
        try:
            tmp.unlink()
        except OSError:
            pass
    return p
