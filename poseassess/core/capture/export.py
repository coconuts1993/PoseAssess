"""Export a recorded take into the project's ``videos/`` so the Pose2Sim pipeline can use it.

Why resample: webcams are free-running (different start times, jitter, dropped frames), but
Pose2Sim pairs frame i of every camera. The export builds ONE common time grid
``t_k = t_start + k / fps`` (``t_start`` = latest first frame of all cameras, last grid time <=
earliest last frame) and, for each camera, takes the source frame nearest to ``t_k`` (frames are
duplicated or skipped as needed). Every ``videos/camNN.mp4`` then has the same frame count and
frame k of every camera shows (almost) the same instant (error <= half a source frame period,
reported per camera). ``frames.csv`` (``poseassess.wii.io.FRAMES_HEADER_BASE`` + per camera
``<cam>_src``, ``<cam>_dt_ms``) gives ``t``/``t_rel``/``t_unix`` of every exported frame, so .trc
Frame# f maps to Wii time ``t_rel[f]``.

Camera latency: a frame is stamped when ``cap.read()`` returns, after exposure, transfer and
driver buffering. ``latency_s`` (capture.json ``latency_ms``, global or per camera) is subtracted
from every frame stamp of that camera before the grid is built, so the grid and frames.csv are
exposure times; the value used is recorded in session.json ``export.latency_ms``.

Writing rules (Pose2Sim / Videos page contract):
* write ``videos/camNN.mp4`` (OpenCV ``mp4v``, recorded resolution, fps = the grid fps, int);
* delete every other ``videos/camNN.<video ext>`` of the exported cameras (Pose2Sim would count
  both files as cameras);
* nothing else goes into ``videos/``;
* when the grid fps differs from ``project.config.frame_rate``: set it, ``project.save()`` and
  ``config_gen.generate_config(project)`` (the same thing "Save settings" does);
* trial.json: ``set_recording(project, rec_id, source="capture",
  alignment=Alignment("recorded", ...))`` + ``videos_fingerprint``;
* session.json gets an ``"export"`` entry (fps, n_frames, t_start_rel, videos, frames_csv,
  per-camera max_dt_ms / duplicated / skipped / latency_ms, time);
* raw ``camNN.mkv`` are deleted after a verified export unless ``keep_raw`` (timestamps CSVs are
  always kept).

Safety: every video is first written to ``<recording>/export_tmp/`` and verified (frame count);
only then are the old ``videos/camNN.*`` moved aside and the new files moved in. If that fails
(e.g. a video is open in another program on Windows), the old videos are put back. A cancel or
an error before that point leaves ``videos/`` untouched.
"""

from __future__ import annotations

import csv
import logging
import os
import shutil
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from poseassess.wii import io as wio

log = logging.getLogger(__name__)

EXPORT_TMP = "export_tmp"
RAW_VIDEO_EXTS = (".mkv", ".avi")  # what CameraStream writes (MKV, or the AVI/MJPG fallback)


class ExportCancelled(RuntimeError):
    """The export was cancelled (``cancel()`` returned True); videos/ is unchanged."""


@dataclass
class ExportPlan:
    """Common frame grid of a take (computed from the timestamps CSVs only)."""

    rec_id: str
    fps: int
    t_grid: np.ndarray                  # (M,) perf_counter times of the exported frames
    t_rel: np.ndarray                   # (M,)
    t_unix: np.ndarray                  # (M,) NaN when the take has no clock offset
    src: dict[str, np.ndarray]          # cam -> (M,) source frame index in camNN.mkv
    dt_ms: dict[str, np.ndarray]        # cam -> (M,) source time - grid time (ms)
    sizes: dict[str, tuple[int, int]]   # cam -> (w, h) of the recorded video ((0, 0): missing)
    warnings: list[str] = field(default_factory=list)
    measured_fps: dict[str, float] = field(default_factory=dict)  # cam -> median frame rate
    videos: dict[str, Path | None] = field(default_factory=dict)  # cam -> raw video (None: gone)
    n_source: dict[str, int] = field(default_factory=dict)        # cam -> timestamp rows
    latency_s: dict[str, float] = field(default_factory=dict)     # cam -> latency subtracted

    @property
    def n_frames(self) -> int:
        return int(len(self.t_rel))

    @property
    def cameras(self) -> list[str]:
        return sorted(self.src)

    def duplicated(self, cam: str) -> int:
        """Grid frames that repeat the previous source frame of ``cam``."""
        s = self.src[cam]
        return int(np.sum(s[1:] == s[:-1])) if len(s) > 1 else 0

    def skipped(self, cam: str) -> int:
        """Source frames of ``cam`` inside the exported span that are not used."""
        s = self.src[cam]
        if not len(s):
            return 0
        return int(s[-1] - s[0] + 1 - len(np.unique(s)))

    def max_dt_ms(self, cam: str) -> float:
        d = self.dt_ms[cam]
        return float(np.max(np.abs(d))) if len(d) else 0.0


@dataclass
class ExportResult:
    rec_id: str
    fps: int
    n_frames: int
    t_start_rel: float
    videos: list[Path]
    frames_csv: Path
    max_dt_ms: dict[str, float]
    duplicated: dict[str, int]
    skipped: dict[str, int]
    fps_changed: bool
    warnings: list[str] = field(default_factory=list)


# ------------------------------------------------------------------------------ helpers
def take_cameras(recording_dir: str | Path) -> dict[str, tuple[Path | None, Path]]:
    """``{cam: (raw video or None, timestamps csv)}`` of the cameras a take recorded (from
    session.json ``streams``, else from the ``*_timestamps.csv`` files). Only cameras whose
    timestamps file exists are listed."""
    rec_dir = Path(recording_dir)
    meta = wio.read_session_json(rec_dir)
    out: dict[str, tuple[Path | None, Path]] = {}
    streams = [st for st in (meta.get("streams") or []) if isinstance(st, dict) and st.get("name")]
    if streams:
        for st in streams:
            name = str(st["name"])
            video = st.get("video")
            stem = Path(video).stem if video else name
            ts = rec_dir / (st.get("timestamps") or wio.timestamps_csv_name(stem))
            if not ts.is_file():
                ts = rec_dir / wio.timestamps_csv_name(name)
            if not ts.is_file():
                continue
            out[name] = (_raw_video(rec_dir, name, video), ts)
    else:
        for ts in sorted(rec_dir.glob(f"*{wio.TIMESTAMPS_SUFFIX}")):
            name = ts.name[: -len(wio.TIMESTAMPS_SUFFIX)]
            out[name] = (_raw_video(rec_dir, name, None), ts)
    return out


def _raw_video(rec_dir: Path, name: str, video: str | None) -> Path | None:
    if video and (rec_dir / video).is_file():
        return rec_dir / video
    for ext in RAW_VIDEO_EXTS:
        p = rec_dir / f"{name}{ext}"
        if p.is_file():
            return p
    return None


def _video_size(path: Path | None) -> tuple[int, int]:
    if path is None:
        return (0, 0)
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            return (0, 0)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if not (w and h):
            ok, img = cap.read()
            if ok and img is not None:
                h, w = img.shape[:2]
        return (int(w), int(h))
    finally:
        cap.release()


def _num(v) -> float | None:
    return float(v) if isinstance(v, (int, float)) and np.isfinite(v) else None


def _f6(v: float) -> str:
    return "" if v is None or not np.isfinite(v) else f"{v:.6f}"


# --------------------------------------------------------------------------------- plan
def _latency_map(latency_s, cams) -> dict[str, float]:
    """``latency_s`` (None, a number for every camera, or ``{cam: s}``) -> ``{cam: s}``."""
    if isinstance(latency_s, dict):
        return {c: float(latency_s.get(c) or 0.0) for c in cams}
    return {c: float(latency_s or 0.0) for c in cams}


def plan_export(recording_dir: str | Path, fps: int | None = None,
                cams: list[str] | None = None,
                latency_s: float | dict[str, float] | None = None) -> ExportPlan:
    """Build the common grid. ``fps`` default: round(min over cameras of the measured median
    frame rate). ValueError when the take has no camera timestamps or the cameras do not
    overlap in time. Warnings: a camera's measured fps < 90 % of the grid fps, Calib.toml size
    mismatch is checked by ``export_recording_to_project``.

    ``cams`` restricts the grid to these cameras (ValueError when one is not in the take);
    default: every camera of the take. ``latency_s``: camera capture latency (s, one value or
    ``{cam: s}``) subtracted from the frame stamps (see the module docstring). Other warnings:
    gaps longer than 1.5 output frames
    (several dropped frames), cameras that start or end more than 1 s apart (the export is trimmed to
    the time all cameras recorded), a missing raw video."""
    rec_dir = Path(recording_dir)
    rec_id = rec_dir.name
    meta = wio.read_session_json(rec_dir)
    takes = take_cameras(rec_dir)
    if cams is not None:
        missing = [c for c in cams if c not in takes]
        if missing:
            raise ValueError(f"The take {rec_id} has no video timestamps of "
                             f"{', '.join(missing)}.")
        takes = {c: takes[c] for c in cams}
    if not takes:
        raise ValueError(f"The take {rec_id} has no camera video (a Wii-only take cannot be "
                         "exported to videos/).")
    t0 = _num(meta.get("t0"))
    t0_unix = _num(meta.get("t0_unix"))
    off = _num(meta.get("clock_offset_unix"))
    if t0_unix is None and t0 is not None and off is not None:
        t0_unix = t0 + off

    times: dict[str, np.ndarray] = {}
    frames: dict[str, np.ndarray] = {}
    measured: dict[str, float] = {}
    lat = _latency_map(latency_s, takes)
    for c, (_video, ts_path) in sorted(takes.items()):
        ts = wio.read_timestamps_csv(ts_path)
        tr, t, fr = ts["t_rel"], ts["t"], ts["frame"]
        if not np.isfinite(tr).any() and t0 is not None:
            tr = t - t0
        if t0 is None and np.isfinite(t).any() and np.isfinite(tr).any():
            t0 = float(np.nanmedian(t - tr))
        ok = np.isfinite(tr) & np.isfinite(fr)
        tr, fr = tr[ok] - lat[c], fr[ok].astype(np.int64)  # read() time -> exposure time
        order = np.argsort(tr, kind="stable")
        tr, fr = tr[order], fr[order]
        if len(tr) < 2:
            raise ValueError(f"{c} recorded fewer than 2 frames in the take {rec_id}.")
        d = np.diff(tr)
        d = d[d > 0]
        measured[c] = float(1.0 / np.median(d)) if len(d) else 0.0
        times[c], frames[c] = tr, fr

    if fps is None or not fps:
        good = [m for m in measured.values() if m > 0]
        if not good:
            raise ValueError(f"Cannot measure the frame rate of the take {rec_id}.")
        fps = int(round(min(good)))
    fps = int(round(float(fps)))
    if fps < 1:
        raise ValueError(f"Invalid output frame rate: {fps}")

    start = max(float(v[0]) for v in times.values())
    end = min(float(v[-1]) for v in times.values())
    if end < start:
        raise ValueError(f"The cameras of the take {rec_id} do not overlap in time (one camera "
                         "stopped before another one started).")
    m = int(np.floor((end - start) * fps + 1e-6)) + 1
    t_rel = start + np.arange(m, dtype=np.float64) / fps

    src: dict[str, np.ndarray] = {}
    dt_ms: dict[str, np.ndarray] = {}
    for c, tr in times.items():
        j = np.clip(np.searchsorted(tr, t_rel), 1, len(tr) - 1)
        left = j - 1
        take_left = (t_rel - tr[left]) <= (tr[j] - t_rel)
        k = np.where(take_left, left, j)
        src[c] = frames[c][k]
        dt_ms[c] = (tr[k] - t_rel) * 1000.0

    videos = {c: v for c, (v, _ts) in takes.items()}
    plan = ExportPlan(
        rec_id=rec_id, fps=fps,
        t_grid=t_rel + (t0 if t0 is not None else np.nan),
        t_rel=t_rel,
        t_unix=t_rel + (t0_unix if t0_unix is not None else np.nan),
        src=src, dt_ms=dt_ms,
        sizes={c: _video_size(v) for c, v in videos.items()},
        measured_fps=measured, videos=videos,
        n_source={c: int(len(v)) for c, v in times.items()},
        latency_s={c: lat[c] for c in times},
    )
    period_ms = 1000.0 / fps
    first = min(float(v[0]) for v in times.values())
    last = max(float(v[-1]) for v in times.values())
    for c in plan.cameras:
        mf = measured[c]
        if mf < 0.9 * fps:
            plan.warnings.append(f"{c} delivered only {mf:.1f} fps but the output is {fps} fps: "
                                 f"{plan.duplicated(c)} of {m} frames are duplicates (lower the "
                                 "output fps, or check the light / USB bandwidth).")
        md = plan.max_dt_ms(c)
        if md > 1.5 * period_ms:  # 2+ consecutive frames lost (a single drop is common)
            plan.warnings.append(f"{c}: frames are missing for up to {md:.0f} ms (dropped "
                                 "frames); the nearest frame is used.")
        if float(times[c][0]) - first > 1.0:
            plan.warnings.append(f"{c} started {float(times[c][0]) - first:.1f} s after the "
                                 "first camera: the export starts there.")
        if last - float(times[c][-1]) > 1.0:
            plan.warnings.append(f"{c} stopped {last - float(times[c][-1]):.1f} s before the "
                                 "last camera: the export ends there.")
        if videos[c] is None:
            plan.warnings.append(f"The raw video of {c} is missing (deleted after an earlier "
                                 "export?): this take cannot be exported again.")
    return plan


# ------------------------------------------------------------------------------- export
def _write_camera(cam: str, raw: Path, out: Path, src: np.ndarray, fps: int,
                  size: tuple[int, int], tick: Callable[[str], None]) -> int:
    """Write ``out`` (mp4v) with the frames ``src`` of ``raw``; returns how many grid frames
    had to reuse the last frame because the raw video ended early (crash: its last frames
    were still buffered)."""
    cap = cv2.VideoCapture(str(raw))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot read the raw video {raw.name}.")
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), size)
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot write {out} (mp4v codec not available, or the folder is not "
                           "writable).")
    short = 0
    cur, img, ended = -1, None, False
    try:
        for target in src:
            target = int(target)
            while cur < target and not ended:
                if cur < target - 1:
                    ok = cap.grab()  # skipped source frame: no need to decode it fully
                    if not ok:
                        ended = True
                        break
                    cur += 1
                    continue
                ok, frame = cap.read()
                if not ok or frame is None:
                    ended = True
                    break
                cur, img = cur + 1, frame
            if img is None:
                raise RuntimeError(f"The raw video {raw.name} contains no readable frame.")
            if cur < target:
                short += 1
            if (img.shape[1], img.shape[0]) != tuple(size):
                img = cv2.resize(img, tuple(size))
            writer.write(img)
            tick(cam)
    finally:
        writer.release()
        cap.release()
    return short


def _frame_count(path: Path) -> int:
    cap = cv2.VideoCapture(str(path))
    try:
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if n <= 0:  # container without a frame count: count them
            n = 0
            while cap.grab():
                n += 1
        return n
    finally:
        cap.release()


def _write_frames_csv(path: Path, plan: ExportPlan) -> None:
    head = list(wio.FRAMES_HEADER_BASE)
    for c in plan.cameras:
        head += [f"{c}_src", f"{c}_dt_ms"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(head)
        for k in range(plan.n_frames):
            row = [k, _f6(plan.t_grid[k]), _f6(plan.t_rel[k]), _f6(plan.t_unix[k])]
            for c in plan.cameras:
                row += [int(plan.src[c][k]), _f6(plan.dt_ms[c][k])]
            w.writerow(row)


def apply_export_fps(project, fps: int) -> tuple[bool, list[str]]:
    """Set the project's frame rate to the exported ``fps`` when it differs: ``project.save()``
    + ``config_gen.generate_config`` (what "Save settings" does). Returns ``(changed,
    warnings)``. The GUI calls it on the GUI thread after a worker export with
    ``update_project=False`` (the ``Project`` object is shared by every page)."""
    if int(project.config.frame_rate) == int(fps):
        return False, []
    project.config.frame_rate = int(fps)
    project.save()
    try:
        from poseassess.core.config_gen import generate_config

        generate_config(project)
    except Exception as e:  # noqa: BLE001
        return True, [(f"Config.toml could not be regenerated ({e}): use \"Save settings\" on "
                       "1. Project before running the pipeline.")]
    return True, []


def export_recording_to_project(project, rec_id: str, fps: int | None = None,
                                keep_raw: bool = True, update_project: bool = True,
                                progress: Callable[[int, int, str], None] | None = None,
                                cancel: Callable[[], bool] | None = None,
                                latency_s: float | dict[str, float] | None = None
                                ) -> ExportResult:
    """Execute ``plan_export`` and write videos/, frames.csv, session.json "export", project fps
    / Config.toml and trial.json (see the module docstring). ``progress(done, total, msg)`` and
    ``cancel()`` are called from the calling (worker) thread; on cancel or error the partially
    written files are removed and videos/ is left as before the export (write into
    ``<recording>/export_tmp/camNN.mp4`` and move into videos/ only when every camera is done). Warns (``warnings``) when a recorded
    size differs from the Calib.toml size of that camera. ``update_project=False`` leaves the
    project frame rate / Config.toml to the caller (``apply_export_fps``); ``latency_s``: see
    ``plan_export``.

    The take must contain every project camera (``cam01..camNN``, ``num_cameras``); only those
    are exported. Raises ValueError (not exportable: no such take, a camera or raw video
    missing, no overlap), ``ExportCancelled`` (cancelled; nothing changed) or RuntimeError
    (writing failed; videos/ unchanged)."""
    from poseassess.core.balance.paths import WiiPaths, cam_name
    from poseassess.core.balance.trial import (
        VIDEO_EXTS,
        Alignment,
        now_iso,
        save_trial,
        set_recording,
        videos_fingerprint,
    )

    paths = WiiPaths(project)
    rec_dir = paths.recording_dir(rec_id)
    if not rec_dir.is_dir():
        raise ValueError(f"There is no take {rec_id} in {paths.recordings_dir}.")
    n_cams = int(project.config.num_cameras)
    expected = [cam_name(i) for i in range(1, n_cams + 1)]
    takes = take_cameras(rec_dir)
    missing = [c for c in expected if c not in takes]
    if missing:
        have = ", ".join(sorted(takes)) or "none"
        raise ValueError(f"The take {rec_id} has no video of {', '.join(missing)} (the project "
                         f"has {n_cams} cameras; the take has: {have}).")
    plan = plan_export(rec_dir, fps, cams=expected, latency_s=latency_s)
    gone = [c for c in expected if plan.videos.get(c) is None]
    if gone:
        raise ValueError(f"The raw videos of {', '.join(gone)} are missing in the take {rec_id} "
                         "(deleted after an earlier export without \"Keep raw camera files\"?).")
    bad = [c for c in expected if not all(plan.sizes.get(c, (0, 0)))]
    if bad:
        raise ValueError(f"Cannot read the raw videos of {', '.join(bad)} in the take {rec_id}.")
    warnings = list(plan.warnings)
    extra = sorted(set(takes) - set(expected))
    if extra:
        warnings.append(f"Not exported (not a camera of this project): {', '.join(extra)}.")
    try:
        from poseassess.core.balance.calib import project_cameras

        calib = project_cameras(project)
    except Exception as e:  # noqa: BLE001
        calib = {}
        warnings.append(f"Calib.toml could not be read to check the image sizes: {e}")
    for c in expected:
        cal = calib.get(c)
        if cal is not None and all(cal.image_size) and tuple(cal.image_size) != plan.sizes[c]:
            w, h = plan.sizes[c]
            cw, ch = cal.image_size
            warnings.append(f"{c} was recorded at {w}x{h} but Calib.toml is for {cw}x{ch}: the "
                            "intrinsics are resolution-specific, so the 3D result will be "
                            f"wrong. Record at {cw}x{ch} or calibrate again at {w}x{h}.")

    m = plan.n_frames
    total = m * len(expected) + 1
    done = [0]
    step = max(1, total // 200)

    def check_cancel():
        if cancel is not None and cancel():
            raise ExportCancelled("Export cancelled; videos/ is unchanged.")

    def tick(cam: str):
        done[0] += 1
        check_cancel()
        if progress is not None and (done[0] % step == 0 or done[0] % m == 0):
            progress(done[0], total, f"{cam}: frame {(done[0] - 1) % m + 1}/{m}")

    tmp = rec_dir / EXPORT_TMP
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    short: dict[str, int] = {}
    try:
        for c in expected:
            check_cancel()
            if progress is not None:
                progress(done[0], total, f"{c}: writing {m} frames at {plan.fps} fps")
            out = tmp / f"{c}.mp4"
            short[c] = _write_camera(c, plan.videos[c], out, plan.src[c], plan.fps,
                                     plan.sizes[c], tick)
            n = _frame_count(out)
            if n != m:
                raise RuntimeError(f"{c}: the exported video has {n} frames instead of {m}.")
        check_cancel()
        _write_frames_csv(tmp / wio.FRAMES_CSV, plan)
        for c, k in short.items():
            if k:
                warnings.append(f"{c}: the raw video ended {k} frames before its timestamps (was "
                                "the recording interrupted?); its last frame is repeated.")

        # ---- replace the videos (all or nothing) ----
        if progress is not None:
            progress(done[0], total, "moving the videos into videos/")
        videos_dir = Path(project.videos_dir)
        videos_dir.mkdir(parents=True, exist_ok=True)
        backup = tmp / "previous"
        backup.mkdir()
        moved: list[tuple[Path, Path]] = []
        placed: list[Path] = []
        try:
            for f in sorted(videos_dir.iterdir()):
                if (f.is_file() and f.stem.lower() in expected
                        and f.suffix.lower() in VIDEO_EXTS):
                    b = backup / f.name
                    os.replace(f, b)
                    moved.append((b, f))
            for c in expected:
                dst = videos_dir / f"{c}.mp4"
                os.replace(tmp / f"{c}.mp4", dst)
                placed.append(dst)
        except OSError as e:
            for dst in placed:
                try:
                    dst.unlink()
                except OSError:
                    log.exception("rollback: cannot remove %s", dst)
            for b, f in moved:
                try:
                    os.replace(b, f)
                except OSError:
                    log.exception("rollback: cannot restore %s", f)
            raise RuntimeError(f"Cannot replace the videos in videos/ ({e}). Close every "
                               "program or page that shows these videos and export again; "
                               "videos/ was left as it was.") from e
        os.replace(tmp / wio.FRAMES_CSV, rec_dir / wio.FRAMES_CSV)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise

    # ---- the videos are in place: bookkeeping ----
    videos = [project.videos_dir / f"{c}.mp4" for c in expected]
    max_dt = {c: round(plan.max_dt_ms(c), 3) for c in expected}
    dup = {c: plan.duplicated(c) for c in expected}
    skip = {c: plan.skipped(c) for c in expected}
    t_start = float(plan.t_rel[0])

    fps_changed = False
    if update_project:
        fps_changed, more = apply_export_fps(project, plan.fps)
        warnings += more

    lat_ms = {c: round(1000.0 * plan.latency_s.get(c, 0.0), 3) for c in expected}
    al = Alignment("recorded", t_start, wio.FRAMES_CSV, None,
                   {"source": "capture export", "fps": plan.fps, "n_frames": m,
                    "max_dt_ms": max_dt, "latency_ms": lat_ms}, now_iso())
    info = set_recording(project, rec_id, source="capture", alignment=al)
    info.videos_fingerprint = videos_fingerprint(project)
    save_trial(project, info)

    if not keep_raw:
        for c in expected:
            try:
                plan.videos[c].unlink()
            except OSError as e:
                warnings.append(f"The raw video of {c} could not be deleted: {e}")
    meta = wio.read_session_json(rec_dir)
    meta["export"] = {
        "fps": plan.fps, "n_frames": m, "t_start_rel": t_start,
        "videos": [f"videos/{c}.mp4" for c in expected], "frames_csv": wio.FRAMES_CSV,
        "max_dt_ms": max_dt, "duplicated": dup, "skipped": skip,
        "measured_fps": {c: round(plan.measured_fps[c], 3) for c in expected},
        "keep_raw": bool(keep_raw), "time": now_iso(), "warnings": list(warnings),
        "latency_ms": lat_ms,
    }
    try:
        wio.write_session_json(rec_dir, meta)
    except OSError as e:
        warnings.append(f"session.json could not be updated: {e}")
    shutil.rmtree(tmp, ignore_errors=True)
    if progress is not None:
        progress(total, total, "done")
    return ExportResult(rec_id, plan.fps, m, t_start, videos, rec_dir / wio.FRAMES_CSV, max_dt,
                        dup, skip, fps_changed, warnings)
