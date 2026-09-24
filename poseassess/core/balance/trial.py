"""The project's trial record ``<root>/wii/trial.json``: which Wii recording belongs to the
trial (a PoseAssess project is one trial) and how TRC time maps to that recording's time.

Time model
----------
Every Wii recording has its own clock; ``t_rel`` = seconds since the recording started.
A .trc row has ``Frame#`` (Pose2Sim: 0-based index of the videos/camNN.* frame) and ``Time``
(= Frame# / Config.frame_rate). The alignment maps a TRC row to ``t_rel``:

* ``recorded``   videos were captured in PoseAssess together with the Wii data: frames.csv of the
                 recording gives ``t_rel`` of every exported video frame; TRC Frame# f ->
                 ``frames.t_rel[f]`` (robust to fps mismatch; exposure times when the camera
                 latency was set for the export, see ``alignment.check_recorded_alignment``).
                 ``offset_s`` = t_rel of frame 0 is kept as a fallback.
* ``sync_event`` / ``xcorr`` / ``manual``  videos recorded elsewhere:
                 ``t_rel = trc_time + offset_s``.
* ``none``       not aligned yet (fusion refuses, the GUI asks the user to align).

trial.json (``TRIAL_SCHEMA``)::

    {"schema": "poseassess.wii.trial/1",
     "recording": "20260924_153000" | null,      # folder name under wii/recordings/
     "source": "capture" | "external",           # capture = videos exported from that take
     "alignment": {"method": "recorded"|"sync_event"|"xcorr"|"manual"|"none",
                   "offset_s": 0.0, "frames_csv": "frames.csv" | null,
                   "trc": "pose-3d/<name>.trc" | null, "details": {...}, "updated": iso},
     "videos_fingerprint": {"cam01.mp4": [size_bytes, mtime_ns], ...},   # at capture export
     "body_mass_kg": null | float,               # user override; default = measured
     "notes": ""}
"""

from __future__ import annotations

import csv
import json
import re
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

from .paths import WiiPaths

TRIAL_SCHEMA = "poseassess.wii.trial/1"
ALIGN_METHODS = ("recorded", "sync_event", "xcorr", "manual", "none")
VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm", ".m4v", ".mpeg", ".mpg")


def now_iso() -> str:
    """Local time, ISO 8601 with timezone, seconds precision."""
    return datetime.now().astimezone().isoformat(timespec="seconds")


@dataclass
class Alignment:
    """How TRC time maps to the Wii recording's ``t_rel`` (see the module docstring)."""

    method: str = "none"
    offset_s: float = 0.0
    frames_csv: str | None = None     # recorded: file name inside the recording folder
    trc: str | None = None            # project-relative .trc the alignment was computed with
    details: dict = field(default_factory=dict)  # detector output (events, score, residual...)
    updated: str = ""

    def to_dict(self) -> dict:
        return {"method": self.method, "offset_s": float(self.offset_s),
                "frames_csv": self.frames_csv, "trc": self.trc, "details": dict(self.details),
                "updated": self.updated}

    @classmethod
    def from_dict(cls, d: dict | None) -> Alignment:
        d = d or {}
        m = d.get("method", "none")
        return cls(m if m in ALIGN_METHODS else "none", float(d.get("offset_s") or 0.0),
                   d.get("frames_csv"), d.get("trc"), dict(d.get("details") or {}),
                   str(d.get("updated") or ""))


@dataclass
class TrialInfo:
    """Content of wii/trial.json (defaults = no Wii data for this trial)."""

    recording: str | None = None
    source: str = "external"          # "capture" | "external"
    alignment: Alignment = field(default_factory=Alignment)
    videos_fingerprint: dict = field(default_factory=dict)
    body_mass_kg: float | None = None
    notes: str = ""

    def to_dict(self) -> dict:
        return {"schema": TRIAL_SCHEMA, "recording": self.recording, "source": self.source,
                "alignment": self.alignment.to_dict(),
                "videos_fingerprint": dict(self.videos_fingerprint),
                "body_mass_kg": self.body_mass_kg, "notes": self.notes}

    @classmethod
    def from_dict(cls, d: dict | None) -> TrialInfo:
        d = d or {}
        bm = d.get("body_mass_kg")
        return cls(d.get("recording"), d.get("source", "external"),
                   Alignment.from_dict(d.get("alignment")), dict(d.get("videos_fingerprint") or {}),
                   None if bm is None else float(bm), str(d.get("notes") or ""))


def load_trial(project) -> TrialInfo:
    """Read wii/trial.json; a default ``TrialInfo()`` when missing or unreadable."""
    p = WiiPaths(project).trial_json
    try:
        return TrialInfo.from_dict(json.loads(p.read_text(encoding="utf-8")))
    except (OSError, ValueError):
        return TrialInfo()


def save_trial(project, info: TrialInfo) -> Path:
    """Write wii/trial.json atomically (tmp file + replace). Returns its path."""
    paths = WiiPaths(project)
    paths.wii_dir.mkdir(parents=True, exist_ok=True)
    p = paths.trial_json
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(info.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(p)
    return p


def videos_fingerprint(project) -> dict:
    """``{file name: [size, mtime_ns]}`` of every video in videos/ (to notice replaced videos)."""
    vdir = project.videos_dir if hasattr(project, "videos_dir") else Path(project) / "videos"
    out = {}
    if vdir.is_dir():
        for f in sorted(vdir.iterdir()):
            if f.is_file() and f.suffix.lower() in VIDEO_EXTS:
                st = f.stat()
                out[f.name] = [st.st_size, st.st_mtime_ns]
    return out


def videos_changed(project, info: TrialInfo) -> bool:
    """True when the trial came from a capture and videos/ no longer matches the export."""
    return bool(info.source == "capture" and info.videos_fingerprint
                and videos_fingerprint(project) != info.videos_fingerprint)


def set_recording(project, rec_id: str | None, source: str = "external",
                  alignment: Alignment | None = None) -> TrialInfo:
    """Make ``rec_id`` the trial's recording (resets the alignment unless given) and save."""
    info = load_trial(project)
    info.recording = rec_id
    info.source = source
    info.alignment = alignment or Alignment(updated=now_iso())
    if source != "capture":
        info.videos_fingerprint = {}
    save_trial(project, info)
    return info


def _safe_id(name: str) -> str:
    """A folder name usable on Windows: keep letters, digits, '-', '_' and '.'."""
    out = re.sub(r"[^\w.-]+", "_", name.strip()).strip("._")
    return out or "recording"


def import_recording(project, src: str | Path, name: str | None = None,
                     make_active: bool = True) -> str:
    """Copy an externally made Wii recording into ``wii/recordings/<id>/`` and return ``id``.

    ``src`` is a recording folder (session.json and/or wii.csv, e.g. from ``python -m
    poseassess.wii`` or PoseBoard) or a bare wii.csv (WII_HEADER columns; at least ``t_rel`` or
    ``t`` plus the four sensor or total_kg columns). Camera videos inside ``src`` are NOT copied.
    session.json gets ``"imported_from": <abs src>`` (created with the minimal keys when
    missing). ``id`` = ``name`` or the source folder name (``_2``... when taken). With
    ``make_active`` the trial is switched to it (``source="external"``, alignment "none").
    Raises ValueError when ``src`` is not a usable recording.

    Details: a bare file named ``wii.csv`` takes the name of its folder, another file its stem.
    A folder that already is ``wii/recordings/<id>`` of this project is not copied (only made
    active). A wii.csv without ``t_rel`` / ``total_kg`` / board COP columns is rewritten in the
    ``WII_HEADER`` format with those columns computed (``t_rel = t - t0``; total = sum of the
    sensors; COP from the sensors with the recording's sensor spacing, default 433 x 238 mm).
    """
    from poseassess.wii import io as wio

    paths = WiiPaths(project)
    src = Path(src).expanduser()
    if src.is_dir():
        folder = src
        wii_file = src / wio.WII_CSV
        default_name = src.resolve().name
    elif src.is_file():
        folder = None
        wii_file = src
        default_name = src.resolve().parent.name if src.stem.lower() == "wii" else src.stem
    else:
        raise ValueError(f"{src} does not exist.")
    if not wii_file.is_file():
        raise ValueError(f"{src} is not a Wii recording: it has no {wio.WII_CSV}.")
    try:
        cols = wio.read_csv_columns(wii_file)
    except (OSError, UnicodeDecodeError, csv.Error) as e:
        raise ValueError(f"Cannot read {wii_file.name}: {e}") from e
    meta = wio.read_session_json(folder) if folder is not None else {}

    def col(name):
        v = cols.get(name)
        return v if v is not None and np.isfinite(v).any() else None

    t, t_rel = col("t"), col("t_rel")
    sensors = [col(c) for c in wio.SENSOR_COLS]
    total = col("total_kg")
    n = len(next(iter(cols.values()))) if cols else 0
    if t_rel is None and t is None:
        raise ValueError(f"{wii_file.name} has no time column (t_rel or t).")
    if total is None and any(v is None for v in sensors):
        raise ValueError(f"{wii_file.name} has no force columns (total_kg or the four sensors "
                         f"{', '.join(wio.SENSOR_COLS)}).")
    if n < 2:
        raise ValueError(f"{wii_file.name} contains fewer than 2 samples.")

    # Already one of this project's recordings: only make it active.
    if folder is not None and folder.resolve().parent == paths.recordings_dir.resolve():
        rec_id = folder.resolve().name
        if make_active:
            set_recording(project, rec_id, source="external")
        return rec_id

    base = _safe_id(name or default_name)
    rec_id, k = base, 2
    while paths.recording_dir(rec_id).exists():
        rec_id, k = f"{base}_{k}", k + 1
    dst = paths.recording_dir(rec_id)
    dst.mkdir(parents=True)
    try:
        t0 = meta.get("t0")
        if not isinstance(t0, (int, float)) and t is not None and t_rel is not None:
            t0 = float(np.nanmedian(t - t_rel))
        if t_rel is None:
            t0 = float(t0) if isinstance(t0, (int, float)) else float(np.nanmin(t))
            t_rel = t - t0
        if total is None:
            total = np.sum(sensors, axis=0)
        cop_given = col("cop_x_board") is not None and col("cop_y_board") is not None
        needs_rewrite = (col("t_rel") is None or col("total_kg") is None
                         or (not cop_given and all(v is not None for v in sensors)))
        if needs_rewrite:
            _write_normalized_wii_csv(dst / wio.WII_CSV, cols, t_rel, total, sensors, meta)
        else:
            shutil.copy2(wii_file, dst / wio.WII_CSV)
        if folder is not None and (folder / wio.EVENTS_CSV).is_file():
            shutil.copy2(folder / wio.EVENTS_CSV, dst / wio.EVENTS_CSV)
        if not meta:
            off = None
            if col("t_unix") is not None:
                ref = t if t is not None else t_rel
                off = float(np.nanmedian(cols["t_unix"] - ref))
            n_ev = len(wio.read_events_csv(dst))
            tr = t_rel[np.isfinite(t_rel)]
            meta = {"schema": wio.SESSION_SCHEMA, "app": "imported", "created": now_iso(),
                    "subject": "", "notes": "", "t0": None if t0 is None else float(t0),
                    "clock_offset_unix": off, "has_wii": True, "has_video": False,
                    "camera_names": [], "streams": [],
                    "duration_s": float(tr.max() - tr.min()) if len(tr) else 0.0,
                    "samples": {"wii": int(n), "events": n_ev}}
        meta = dict(meta)
        meta["imported_from"] = str(src.resolve())
        meta["imported_time"] = now_iso()
        wio.write_session_json(dst, meta)
    except Exception:
        shutil.rmtree(dst, ignore_errors=True)
        raise
    if make_active:
        set_recording(project, rec_id, source="external")
    return rec_id


def _write_normalized_wii_csv(path: Path, cols: dict, t_rel: np.ndarray, total: np.ndarray,
                              sensors: list, meta: dict) -> None:
    """wii.csv in WII_HEADER order with t_rel / total_kg / board COP filled in."""
    from poseassess.wii import io as wio

    n = len(t_rel)
    out = {h: np.asarray(cols.get(h, np.full(n, np.nan)), np.float64) for h in wio.WII_HEADER}
    out["t_rel"] = np.asarray(t_rel, np.float64)
    out["total_kg"] = np.asarray(total, np.float64)
    if all(v is not None for v in sensors) and not (np.isfinite(out["cop_x_board"]).any()
                                                    and np.isfinite(out["cop_y_board"]).any()):
        fs = meta.get("force_source") or {}
        dx = float(fs.get("sensor_dx_m") or 0.433)
        dy = float(fs.get("sensor_dy_m") or 0.238)
        min_kg = float(fs.get("min_total_kg") or 5.0)
        tr_, br, tl, bl = (np.asarray(v, np.float64) for v in sensors)
        tot = tr_ + br + tl + bl
        with np.errstate(divide="ignore", invalid="ignore"):
            x = dx / 2 * ((tr_ + br) - (tl + bl)) / tot
            y = dy / 2 * ((tr_ + tl) - (br + bl)) / tot
        low = ~(tot >= min_kg)
        x[low], y[low] = np.nan, np.nan
        out["cop_x_board"], out["cop_y_board"] = x, y
    order = np.argsort(np.where(np.isfinite(out["t_rel"]), out["t_rel"], np.inf), kind="stable")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(wio.WII_HEADER)
        for i in order:
            w.writerow(["" if not np.isfinite(out[h][i]) else f"{out[h][i]:.6f}"
                        for h in wio.WII_HEADER])
