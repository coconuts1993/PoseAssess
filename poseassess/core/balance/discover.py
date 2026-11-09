"""Wii data the user put somewhere in the project folder, and a computer-clock alignment.

``find_wii_files``: users often copy the Wii files next to the videos (e.g. ``<root>/00/``)
instead of importing them. The 3D View lists what is found so it can be loaded in one click.
Found: files of the original Wii program (``poseassess.wii.legacy``), wii.csv files and
recording folders (session.json / wii.csv). Not searched: ``wii/`` (imported data), the
Pose2Sim folders, ``videos/`` and ``calibration/``.

``clock_offset``: the offset (Wii ``t_rel`` = .trc Time + offset) from computer times: the
Wii recording's ``t_unix`` (recorded, or estimated from the original file's modification time)
and the start of the videos (file modification time - duration, i.e. the file was closed at
the end of the recording). An estimate (typically within 1-2 s): fine-tune it by eye.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

SKIP_DIRS = {"wii", "videos", "calibration", "kinematics", "logs", "__pycache__", ".git",
             "venv", "python311", "models"}
WII_SUFFIXES = {"", ".csv", ".txt", ".dat", ".tsv", ".log"}  # and numbers (".003")
MAX_FILE_MB = 200


def _skip_dir(name: str) -> bool:
    n = name.lower()
    return n in SKIP_DIRS or n.startswith("pose") or n.startswith(".") or "calib" in n


def _is_wii_csv(p: Path) -> bool:
    """A PoseAssess / PoseBoard wii.csv (header with t_rel and the sensor or total columns)."""
    try:
        with open(p, encoding="utf-8-sig", errors="replace") as f:
            head = f.readline()
    except OSError:
        return False
    cols = {c.strip() for c in head.split(",")}
    return "t_rel" in cols and ("total_kg" in cols or "TL_kg" in cols)


def find_wii_files(project, max_depth: int = 3, limit: int = 50) -> list[Path]:
    """Wii recordings inside the project folder (not yet imported), sorted by path."""
    from poseassess.wii import legacy
    from poseassess.wii.io import WII_CSV

    root = Path(getattr(project, "root", project))
    found: list[Path] = []
    if not root.is_dir():
        return found
    for dirpath, dirnames, filenames in os.walk(root):
        d = Path(dirpath)
        depth = len(d.relative_to(root).parts)
        dirnames[:] = sorted(n for n in dirnames if not _skip_dir(n)) if depth < max_depth else []
        if WII_CSV in filenames and depth > 0 and _is_wii_csv(d / WII_CSV):
            found.append(d)  # a recording folder
            continue
        for name in sorted(filenames):
            p = d / name
            suf = p.suffix.lower()
            if suf not in WII_SUFFIXES and not suf[1:].isdigit():
                continue
            try:
                if p.stat().st_size > MAX_FILE_MB * 1e6 or p.stat().st_size == 0:
                    continue
            except OSError:
                continue
            if legacy.is_legacy_wii_file(p) or (p.suffix.lower() == ".csv" and _is_wii_csv(p)):
                found.append(p)
                if len(found) >= limit:
                    return found
    return found


def video_start_unix(project) -> tuple[float, dict]:
    """Estimated computer time of the first video frame: median over the camera videos of
    (modification time - duration). ValueError when no video can be read."""
    import cv2

    vdir = Path(project.videos_dir)
    starts, info = [], {}
    if vdir.is_dir():
        for f in sorted(vdir.iterdir()):
            if not f.is_file() or not f.name.lower().startswith("cam"):
                continue
            cap = cv2.VideoCapture(str(f))
            try:
                n = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                fps = cap.get(cv2.CAP_PROP_FPS)
            finally:
                cap.release()
            if not (n > 0 and fps > 0):
                continue
            dur = float(n / fps)
            mtime = f.stat().st_mtime
            starts.append(mtime - dur)
            info[f.name] = {"mtime_unix": mtime, "duration_s": dur}
    if not starts:
        raise ValueError("No readable camera video in videos/ to estimate the video start time.")
    spread = float(np.ptp(starts)) if len(starts) > 1 else 0.0
    return float(np.median(starts)), {"videos": info, "video_start_spread_s": spread}


def clock_offset(project, rec_id: str | None = None) -> tuple[float, dict]:
    """``(offset_s, details)`` so that Wii ``t_rel`` = .trc Time + offset, from the computer
    times of the Wii recording and of the videos. ValueError when a time is unknown."""
    from poseassess.wii import io as wio

    from .paths import WiiPaths
    from .trial import load_trial

    rec_id = rec_id or load_trial(project).recording
    if not rec_id:
        raise ValueError("No Wii recording for this trial.")
    rec = WiiPaths(project).recording_dir(rec_id)
    cols = wio.read_wii_csv(rec)
    d = np.asarray(cols.get("t_unix"), float) - np.asarray(cols.get("t_rel"), float)
    d = d[np.isfinite(d)]
    if not len(d):
        raise ValueError(f"The Wii recording {rec_id} has no computer time.")
    wii_zero = float(np.median(d))  # computer time of Wii t_rel = 0
    start, details = video_start_unix(project)
    meta = wio.read_session_json(rec)
    details.update(wii_t_rel_zero_unix=wii_zero, video_start_unix=start,
                   wii_clock_source=meta.get("clock_source", "recorded"))
    return start - wii_zero, details
