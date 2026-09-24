"""Time alignment between the project's .trc and its Wii recording (stored in wii/trial.json).

See ``trial.py`` for the time model. All functions take a ``Project`` (or project root).
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .paths import WiiPaths
from .trial import Alignment, TrialInfo, load_trial, now_iso, save_trial, videos_changed

# auto_align: minimum cross-correlation score accepted as a result
MIN_XCORR_SCORE = 0.5
_METHOD_TEXT = {"recorded": "Recorded with the videos", "sync_event": "Sync event",
                "xcorr": "Cross-correlation", "manual": "Manual offset", "none": "Not aligned"}


@dataclass
class AlignmentResult:
    """Outcome of ``auto_align`` (not saved by itself; ``apply_alignment`` saves it).

    ``signals`` holds curves for the GUI's alignment plot:
    ``{"force_t_rel", "force_total_kg", "trc_t", "trc_foot_up_m", "trc_com_up_m"}``.
    (``trc_foot_up_m`` = height of the lower foot along the world up, ``trc_com_up_m`` = COM
    height, both in metres in the Calib world; plot them against ``trc_t + offset_s``.)
    """

    ok: bool
    alignment: Alignment
    message: str = ""
    force_events: list[dict] = field(default_factory=list)
    trc_events: list[dict] = field(default_factory=list)
    signals: dict[str, np.ndarray] = field(default_factory=dict)


def _json_num(v):
    """float for JSON (None instead of NaN / inf)."""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _json_clean(obj):
    """Recursively convert numpy scalars / tuples / NaN into JSON-friendly values."""
    if isinstance(obj, dict):
        return {str(k): _json_clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_clean(v) for v in obj]
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return _json_num(obj)
    if isinstance(obj, np.ndarray):
        return _json_clean(obj.tolist())
    return obj


def _frames_table(project, rec_id: str, frames_csv: str | None) -> dict[str, np.ndarray] | None:
    from poseassess.wii import io as wio

    f = WiiPaths(project).recording_dir(rec_id) / (frames_csv or wio.FRAMES_CSV)
    if not f.is_file():
        return None
    cols = wio.read_frames_csv(f)
    if "frame" not in cols or "t_rel" not in cols or not len(cols["frame"]):
        return None
    return cols


def trc_rows_to_wii_time(project, frames: np.ndarray, times: np.ndarray,
                         trial: TrialInfo | None = None) -> np.ndarray:
    """Wii recording time ``t_rel`` (s) for each .trc row (``frames`` = Frame#, ``times`` =
    Time). ``recorded``: frames.csv row lookup by Frame# (interpolated; rows beyond the table
    NaN; falls back to ``times + offset_s`` when frames.csv is missing); ``sync_event`` /
    ``xcorr`` / ``manual``: ``times + offset_s``; ``none`` or no recording: all NaN.
    ``trial`` defaults to ``load_trial(project)``."""
    trial = trial if trial is not None else load_trial(project)
    frames = np.asarray(frames, np.float64).ravel()
    times = np.asarray(times, np.float64).ravel()
    al = trial.alignment
    if not trial.recording or al.method == "none":
        return np.full(len(times), np.nan)
    if al.method == "recorded":
        tab = _frames_table(project, trial.recording, al.frames_csv)
        if tab is not None:
            fr, tr = tab["frame"], tab["t_rel"]
            ok = np.isfinite(fr) & np.isfinite(tr)
            fr, tr = fr[ok], tr[ok]
            if len(fr) >= 1:
                order = np.argsort(fr, kind="stable")
                fr, tr = fr[order], tr[order]
                if len(fr) == 1:
                    return np.where(frames == fr[0], tr[0], np.nan)
                return np.interp(frames, fr, tr, left=np.nan, right=np.nan)
    return times + float(al.offset_s)


def wii_time_to_trc_time(project, t_rel: np.ndarray, trial: TrialInfo | None = None,
                         rate_hz: float | None = None) -> np.ndarray:
    """Inverse of ``trc_rows_to_wii_time``: .trc Time for Wii times ``t_rel`` (NaN when not
    aligned). ``recorded`` uses frames.csv (Time = Frame# / ``rate_hz``, default the project's
    frame rate; linear extrapolation outside the table)."""
    trial = trial if trial is not None else load_trial(project)
    t_rel = np.asarray(t_rel, np.float64)
    al = trial.alignment
    if not trial.recording or al.method == "none":
        return np.full(t_rel.shape, np.nan)
    if al.method == "recorded":
        tab = _frames_table(project, trial.recording, al.frames_csv)
        rate = rate_hz or float(getattr(getattr(project, "config", None), "frame_rate", 0) or 0)
        if tab is not None and rate > 0:
            ok = np.isfinite(tab["frame"]) & np.isfinite(tab["t_rel"])
            fr, tr = tab["frame"][ok], tab["t_rel"][ok]
            if len(fr) >= 2:
                order = np.argsort(tr, kind="stable")
                fr, tr = fr[order], tr[order]
                f = np.interp(t_rel, tr, fr)
                lo, hi = t_rel < tr[0], t_rel > tr[-1]
                slope = (fr[-1] - fr[0]) / (tr[-1] - tr[0]) if tr[-1] > tr[0] else rate
                f = np.where(lo, fr[0] + (t_rel - tr[0]) * slope, f)
                f = np.where(hi, fr[-1] + (t_rel - tr[-1]) * slope, f)
                return f / rate
    return t_rel - float(al.offset_s)


def recorded_alignment(project, rec_id: str) -> Alignment:
    """Alignment for a take whose videos were exported by the capture export (method
    ``recorded``; ``offset_s`` = t_rel of exported frame 0 from frames.csv). ValueError when the
    recording has no frames.csv."""
    from poseassess.wii import io as wio

    tab = _frames_table(project, rec_id, wio.FRAMES_CSV)
    if tab is None:
        raise ValueError(f"The recording {rec_id} has no {wio.FRAMES_CSV}: its videos were not "
                         "exported to videos/ by the capture (3b. Capture).")
    fr, tr = tab["frame"], tab["t_rel"]
    ok = np.isfinite(fr) & np.isfinite(tr)
    if not ok.any():
        raise ValueError(f"The {wio.FRAMES_CSV} of the recording {rec_id} has no valid rows.")
    i0 = int(np.argmin(np.where(ok, fr, np.inf)))
    dt = np.diff(tr[ok])
    fps = 1.0 / float(np.median(dt)) if len(dt) and np.median(dt) > 0 else None
    return Alignment("recorded", float(tr[i0]), wio.FRAMES_CSV, None,
                     _json_clean({"n_frames": int(ok.sum()), "fps": fps}), now_iso())


def _trc_rel(project, trc_path) -> str | None:
    return WiiPaths(project).relative(trc_path) if trc_path else None


def world_up(project) -> tuple[np.ndarray, str, str]:
    """World up for marker heights: ``(unit vector, source, note)``.

    ``source`` is "board" (the registered board's normal), "calib" (``world_up_sign`` of the
    Calib.toml cameras x Z) or "default" (+Z, no camera with extrinsics). The board normal is
    used only when the board was located in the current Calib.toml world: not stale (a
    recalibration, e.g. "Flip Z", may have turned the world) and within ~25° of the
    calibration's up. ``note`` says why a registered board was ignored ("" otherwise; it
    completes "Marker heights are measured along the calibration's up axis because ...").
    The alignment plot of the Results page uses the same rule."""
    from .board import load_board
    from .calib import project_cameras, world_up_sign

    try:
        cams = list(project_cameras(project).values())
    except Exception:  # noqa: BLE001
        cams = []
    sign = float(world_up_sign(cams))
    calib_up = np.array([0.0, 0.0, sign])
    source = "calib" if any(c.has_extrinsics for c in cams) else "default"
    reg = load_board(project)
    if reg is None or reg.pose.world_camera is not None:
        return calib_up, source, ""
    if reg.stale:
        return calib_up, source, f"the board position is outdated ({reg.stale.rstrip('.')})"
    up = np.asarray(reg.up_world, np.float64)
    if float(np.dot(up, calib_up)) <= 0.9:
        return calib_up, source, ("the board normal does not point up in the current "
                                  "calibration (recompute the board position on 2b. Wii Board)")
    return up, "board", ""


def _up_vector(project) -> np.ndarray:
    """World up (``world_up(project)[0]``)."""
    return world_up(project)[0]


def auto_align(project, trc_path: str | Path | None = None, rec_id: str | None = None,
               methods: tuple[str, ...] = ("sync_event", "xcorr")) -> AlignmentResult:
    """Estimate the offset for externally recorded videos.

    ``trc_path`` default: ``trc.find_trc_files(project)[0]``; ``rec_id`` default: the trial's
    recording. ``sync_event``: ``detect_force_events`` vs ``detect_trc_events`` +
    ``match_events`` (needs >= 1 matched jump/stomp; >= 2 recommended); ``xcorr``: refine
    around the event offset (or search globally when no events matched). The chosen method is
    the first that succeeds; ``details`` records every candidate (offset, n_matched,
    residual_ms, score). ``ok`` False with a user-readable ``message`` when nothing worked.

    Force and TRC events are always detected (for the plot), whatever ``methods`` holds; an
    xcorr result counts when its score is at least ``MIN_XCORR_SCORE``.
    """
    from poseassess.wii import io as wio

    from . import analysis as an
    from .trc import find_trc_files, read_trc_full

    def fail(msg, **kw) -> AlignmentResult:
        return AlignmentResult(False, Alignment("none", 0.0, None, _trc_rel(project, trc_path),
                                                _json_clean(kw.pop("details", {})), now_iso()),
                               msg, **kw)

    if trc_path is None:
        files = find_trc_files(project)
        if not files:
            return fail("No .trc file in pose-3d/: run the pipeline first (4. Run).")
        trc_path = files[0]
    trial = load_trial(project)
    rec_id = rec_id or trial.recording
    if not rec_id:
        return fail("No Wii recording for this trial: select or import one first.")
    rec = WiiPaths(project).recording_dir(rec_id)
    if not (rec / wio.WII_CSV).is_file():
        return fail(f"The Wii recording {rec_id} has no force data ({wio.WII_CSV}).")
    try:
        wii = wio.read_wii_csv(rec)
        trc = read_trc_full(trc_path)
    except (OSError, ValueError) as e:
        return fail(f"Cannot read the data: {e}")

    t_rel, total = wii["t_rel"], wii["total_kg"]
    if not np.isfinite(t_rel).any():
        t_rel = wii["t"] - np.nanmin(wii["t"])
    up, up_source, up_note = world_up(project)
    world = trc.world()
    heights = an.trc_heights(trc.names, world, up)
    hl, hr, com = heights
    body = an.body_mass_from_force(t_rel, total)
    force_events = an.detect_force_events({"t_rel": t_rel, "t": wii["t"], "t_unix": wii["t_unix"],
                                           "total_kg": total}, body_kg=body)
    trc_events = an.detect_trc_events(trc.times, trc.names, world, up, heights=heights)
    with np.errstate(invalid="ignore"):
        foot = np.fmin(hl, hr)
    signals = {"force_t_rel": np.asarray(t_rel, float), "force_total_kg": np.asarray(total, float),
               "trc_t": np.asarray(trc.times, float), "trc_foot_up_m": foot,
               "trc_com_up_m": com}

    cand: dict[str, dict] = {}
    m = an.match_events(force_events, trc_events)
    ok_event = m["n_matched"] >= 1 and np.isfinite(m["offset_s"])
    cand["sync_event"] = {"ok": bool(ok_event), "offset_s": m["offset_s"],
                          "n_matched": m["n_matched"], "residual_ms": m["residual_ms"],
                          "pairs": [list(p) for p in m["pairs"]]}
    tr_ok = t_rel[np.isfinite(t_rel)]
    search = (float(tr_ok.min() - np.nanmax(trc.times)), float(tr_ok.max() - np.nanmin(trc.times)))
    around = m["offset_s"] if ok_event else None
    if "xcorr" in methods:
        d, score = an.xcorr_offset(t_rel, total, trc.times, com, body_kg=body, search_s=search,
                                   around_s=around)
        cand["xcorr"] = {"ok": bool(np.isfinite(d) and score >= MIN_XCORR_SCORE),
                         "offset_s": d, "score": score,
                         "search_s": [around - 1, around + 1] if around is not None
                         else list(search)}

    chosen = next((k for k in methods if cand.get(k, {}).get("ok")), None)
    details = {"chosen": chosen, "candidates": cand, "body_kg": body,
               "rec_id": rec_id, "up": up.tolist(), "up_source": up_source,
               "force_events": len(force_events), "trc_events": len(trc_events)}
    if up_note:
        details["up_note"] = up_note
    if chosen is None:
        why = []
        if not trc_events:
            why.append("no jump or stomp was found in the markers")
        elif not force_events:
            why.append("no jump or stomp was found in the force")
        elif not ok_event:
            why.append("the jumps/stomps of the force and the markers do not match")
        if "xcorr" in cand:
            sc = cand["xcorr"]["score"]
            why.append("the force and the COM motion do not correlate"
                       + (f" (score {sc:.2f})" if np.isfinite(sc) else ""))
        return fail("Automatic alignment failed: " + "; ".join(why or ["no method succeeded"])
                    + ". Ask the subject to do 2 small jumps or 2 stomps on the board at the "
                    "start of the trial, or set the offset manually."
                    + (f" Marker heights are measured along the calibration's up axis because "
                       f"{up_note}." if up_note else ""),
                    details=details, force_events=force_events, trc_events=trc_events,
                    signals=signals)

    c = cand[chosen]
    al = Alignment(chosen, float(c["offset_s"]), None, _trc_rel(project, trc_path),
                   _json_clean(details), now_iso())
    if chosen == "sync_event":
        msg = (f"Sync event: offset {c['offset_s']:+.3f} s ({c['n_matched']} event(s) matched, "
               f"residual {c['residual_ms']:.0f} ms)")
        if c["n_matched"] < 2:
            msg += ". Only one event matched: check the result (2 jumps or stomps recommended)."
    else:
        msg = f"Cross-correlation: offset {c['offset_s']:+.3f} s (score {c['score']:.2f})"
    xc = cand.get("xcorr")
    if chosen == "sync_event" and xc and xc["ok"] and abs(xc["offset_s"] - c["offset_s"]) > 0.05:
        msg += (f". Note: the cross-correlation suggests {xc['offset_s']:+.3f} s; check the "
                "plot.")
    if up_note:
        msg = (msg.rstrip(".") + ". Marker heights are measured along the calibration's up "
               f"axis because {up_note}.")
    return AlignmentResult(True, al, msg, force_events, trc_events, signals)


def check_recorded_alignment(project, trc_path: str | Path | None = None) -> dict:
    """Timing check of the trial's take recorded WITH the videos (method ``recorded``): the
    frame stamps are taken when ``cap.read()`` returns, so they include the camera latency
    unless it was set before the export (capture.json ``latency_ms``). A jump or stomp in the
    take measures it: ``auto_align`` on the same take is compared with the frame-timestamp
    mapping at the matched events.

    Returns ``{"ok", "latency_ms", "frame_ms", "agrees", "current_latency_ms",
    "suggested_latency_ms", "message", "result"}``: ``latency_ms`` > 0 when the video frames are
    older than their timestamps (force and COP would lead the markers by that much);
    ``agrees`` = within one frame period; ``suggested_latency_ms`` = the capture latency to
    set before exporting the take again (None when not measurable); ``result`` = the
    ``AlignmentResult`` (events for the plot). ``ok`` False (with ``message``) when the trial is
    not a recorded take or no sync event was found."""
    from poseassess.wii import io as wio

    from .trc import find_trc_files, read_trc_full

    out = {"ok": False, "latency_ms": float("nan"), "frame_ms": float("nan"), "agrees": False,
           "current_latency_ms": None, "suggested_latency_ms": None, "message": "",
           "result": None}
    trial = load_trial(project)
    if not trial.recording or trial.alignment.method != "recorded":
        out["message"] = "The trial's Wii recording was not recorded with these videos."
        return out
    if trc_path is None:
        files = find_trc_files(project)
        if not files:
            out["message"] = "No .trc file in pose-3d/: run the pipeline first (4. Run)."
            return out
        trc_path = files[0]
    res = auto_align(project, trc_path, trial.recording)
    out["result"] = res
    if not res.ok:
        out["message"] = res.message
        return out
    trc = read_trc_full(trc_path)
    times = np.asarray(trc.times, float)
    rec_t = trc_rows_to_wii_time(project, trc.frames, times, trial)
    ok = np.isfinite(times) & np.isfinite(rec_t)
    if ok.sum() < 2:
        out["message"] = "The frame timestamps (frames.csv) do not cover this .trc."
        return out
    pairs = (res.alignment.details.get("candidates") or {}).get("sync_event", {}).get("pairs")
    if res.alignment.method == "sync_event" and pairs:
        lat = float(np.median([np.interp(tt, times[ok], rec_t[ok]) - tf for _, tt, tf in pairs]))
    else:
        lat = float(np.median(rec_t[ok] - times[ok])) - float(res.alignment.offset_s)
    dt = np.diff(times[ok])
    frame_s = float(np.median(dt)) if len(dt) else float("nan")
    meta = wio.read_session_json(WiiPaths(project).recording_dir(trial.recording))
    used = (meta.get("export") or {}).get("latency_ms")
    if isinstance(used, dict) and used:
        used = float(np.mean([float(v) for v in used.values()]))
    cur = float(used) if isinstance(used, (int, float)) else 0.0
    out.update(ok=True, latency_ms=1000.0 * lat, frame_ms=1000.0 * frame_s,
               agrees=bool(abs(lat) <= frame_s), current_latency_ms=cur)
    if out["agrees"]:
        how = _METHOD_TEXT.get(res.alignment.method, res.alignment.method).lower()
        out["message"] = (f"Timing check: the {how} agrees with the frame timestamps within "
                          f"{abs(lat) * 1000:.0f} ms (one frame = {frame_s * 1000:.0f} ms).")
        return out
    sugg = cur + 1000.0 * lat
    out["suggested_latency_ms"] = sugg if sugg >= 0 else None
    older = "older" if lat > 0 else "newer"
    msg = (f"Timing check: the video frames are {abs(lat) * 1000:.0f} ms {older} than their "
           f"timestamps ({abs(lat) / frame_s:.1f} frames), so force and COP are shifted against "
           "the markers.")
    if sugg >= 0:
        msg += (f" Set the camera latency to about {sugg:.0f} ms on 3b. Capture (Export to "
                "videos/) and export the take again, or use \"Re-align manually…\".")
    else:
        msg += " Use \"Re-align manually…\" to apply the detected offset."
    out["message"] = msg
    return out


def apply_alignment(project, alignment: Alignment) -> TrialInfo:
    """Store ``alignment`` in trial.json (``updated`` set to now). Returns the new TrialInfo."""
    info = load_trial(project)
    info.alignment = dataclasses.replace(alignment, details=_json_clean(dict(alignment.details)),
                                         offset_s=float(alignment.offset_s), updated=now_iso())
    save_trial(project, info)
    return info


def set_manual_offset(project, offset_s: float, trc_path: str | Path | None = None) -> TrialInfo:
    """Store method ``manual`` with ``offset_s`` (``t_rel = trc_time + offset_s``)."""
    return apply_alignment(project, Alignment("manual", float(offset_s), None,
                                              _trc_rel(project, trc_path), {}, now_iso()))


def alignment_status(project) -> tuple[str, str]:
    """``(level, text)`` for the GUI: level "ok" | "warn" | "none"; text e.g. "Recorded with
    the videos (frame timestamps)", "Sync event: offset +3.215 s (2 events, 4 ms)", "Videos
    changed since the capture: re-align", "No Wii recording for this trial".

    "none" = no Wii recording for this trial; "warn" = a recording exists but the alignment is
    missing or invalid."""
    from poseassess.wii import io as wio

    info = load_trial(project)
    if not info.recording:
        return "none", "No Wii recording for this trial"
    rec = WiiPaths(project).recording_dir(info.recording)
    if not (rec / wio.WII_CSV).is_file():
        return "warn", f"The Wii recording {info.recording} is missing or has no force data"
    al = info.alignment
    if al.method == "none":
        return "warn", "Not aligned yet: detect a sync event or set the offset"
    if info.source == "capture" and videos_changed(project, info):
        return "warn", ("Videos changed since the capture: re-align, or export the take to "
                        "videos/ again")
    if al.method == "recorded":
        if _frames_table(project, info.recording, al.frames_csv) is None:
            return "warn", (f"Recorded with the videos, but frames.csv is missing: using the "
                            f"offset {al.offset_s:+.3f} s")
        return "ok", "Recorded with the videos (frame timestamps)"
    d = al.details or {}
    if al.method == "sync_event":
        c = (d.get("candidates") or {}).get("sync_event") or {}
        extra = ""
        if c.get("n_matched") is not None:
            res = c.get("residual_ms")
            extra = f" ({c['n_matched']} events" + (f", {res:.0f} ms)" if res is not None else ")")
        return "ok", f"Sync event: offset {al.offset_s:+.3f} s{extra}"
    if al.method == "xcorr":
        c = (d.get("candidates") or {}).get("xcorr") or {}
        sc = c.get("score")
        return "ok", (f"Cross-correlation: offset {al.offset_s:+.3f} s"
                      + (f" (score {sc:.2f})" if sc is not None else ""))
    return "ok", f"{_METHOD_TEXT.get(al.method, al.method)}: {al.offset_s:+.3f} s"
