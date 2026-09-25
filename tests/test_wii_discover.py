"""Wii files the user put in the project folder, and the computer-clock alignment."""

from __future__ import annotations

import os

import numpy as np
import pytest

from tests.test_wii_legacy import write_legacy


def _video(path, n=90, fps=30.0):
    import cv2

    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (64, 48))
    for _ in range(n):
        w.write(np.zeros((48, 64, 3), np.uint8))
    w.release()


def test_find_wii_files(project):
    from poseassess.core.balance.discover import find_wii_files

    root = project.root
    (root / "00").mkdir()
    write_legacy(root / "00" / "814c12f3-002.csv")
    write_legacy(root / "00" / "a8cef4a1-003")          # no extension
    write_legacy(root / "00" / "trial.004")             # numbered extension
    (root / "00" / "notes.txt").write_text("hello\n")
    rec = root / "sessions" / "20260925_120000"
    rec.mkdir(parents=True)
    (rec / "wii.csv").write_text("t,t_rel,t_unix,TL_kg,TR_kg,BL_kg,BR_kg,total_kg\n0,0,1,1,1,1,1,4\n")
    # not searched: imported data, Pose2Sim folders, videos
    (root / "wii" / "recordings" / "x").mkdir(parents=True)
    write_legacy(root / "wii" / "recordings" / "x" / "original_a.csv")
    project.pose3d_dir.mkdir(parents=True, exist_ok=True)
    write_legacy(project.pose3d_dir / "b.csv")
    found = [p.relative_to(root).as_posix() for p in find_wii_files(project)]
    assert found == ["00/814c12f3-002.csv", "00/a8cef4a1-003", "00/trial.004",
                     "sessions/20260925_120000"]


def test_clock_offset(project):
    from poseassess.core.balance.discover import clock_offset
    from poseassess.core.balance.trial import import_recording

    project.videos_dir.mkdir(parents=True, exist_ok=True)
    vid_end = 1_790_000_100.0
    for i in (1, 2):
        f = project.videos_dir / f"cam0{i}.mp4"
        _video(f, n=90, fps=30.0)                      # 3 s
        os.utime(f, (vid_end, vid_end))
    src = project.root.parent / "S01-002.csv"
    raw = write_legacy(src, n=600)
    wii_end = vid_end + 7.0                            # the Wii file was closed 7 s later
    os.utime(src, (wii_end, wii_end))
    import_recording(project, src)
    off, det = clock_offset(project)
    wii_zero = wii_end - (raw[-1, 0] - raw[0, 0]) / 1000   # computer time of Wii t_rel 0
    assert off == pytest.approx((vid_end - 3.0) - wii_zero, abs=1e-3)
    assert det["wii_clock_source"] == "file_mtime" and len(det["videos"]) == 2


def test_clock_offset_needs_videos(project):
    from poseassess.core.balance.discover import clock_offset
    from poseassess.core.balance.trial import import_recording

    src = project.root.parent / "S01-002.csv"
    write_legacy(src)
    import_recording(project, src)
    with pytest.raises(ValueError):
        clock_offset(project)
