"""Fuse the trial's .trc with its Wii recording: one row per .trc frame.

``fuse_trial`` is what the 3D View and the Results page use. Frames: "world" = Calib.toml world
(metres, see ``trc.py``); "board" = board frame (x = subject's right, y = front, z = up, origin at
the top-surface centre). The Wii measures vertical load only, so the ground reaction force is
``up_world * total_kg * g`` (no shear) applied at the COP on the top surface.

Fused CSV columns (``FUSED_COLUMNS``; empty = NaN): frame, trc_time, t_rel, t_unix, TR_kg,
BR_kg, TL_kg, BL_kg, total_kg, force_N, cop_x_board, cop_y_board, cop_x_world, cop_y_world,
cop_z_world, force_x_world, force_y_world, force_z_world, com_x_world, com_y_world,
com_z_world, com_x_board, com_y_board, com_z_board, com_minus_cop_x, com_minus_cop_y.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .board import BoardRegistration
from .trial import Alignment

G = 9.80665
COP_ARROW_M_PER_KG = 0.005  # 3D View force arrow length: 0.35 m for 70 kg
FUSED_COLUMNS = [
    "frame", "trc_time", "t_rel", "t_unix", "TR_kg", "BR_kg", "TL_kg", "BL_kg", "total_kg",
    "force_N", "cop_x_board", "cop_y_board", "cop_x_world", "cop_y_world", "cop_z_world",
    "force_x_world", "force_y_world", "force_z_world", "com_x_world", "com_y_world",
    "com_z_world", "com_x_board", "com_y_board", "com_z_board", "com_minus_cop_x",
    "com_minus_cop_y"]


@dataclass
class FusedTrial:
    """Per-.trc-row fused data (N rows, same order as the .trc file). Arrays are NaN where a
    value is unknown (outside the Wii recording, nobody on the board, no board registration,
    COM not computable)."""

    trc_path: Path
    recording: str | None
    alignment: Alignment
    frames: np.ndarray          # (N,) int  Frame#
    trc_time: np.ndarray        # (N,)      .trc Time (s)
    t_rel: np.ndarray           # (N,)      Wii recording time of each row (s)
    t_unix: np.ndarray          # (N,)
    kg: np.ndarray              # (N,4)     TR, BR, TL, BL (kg, tared)
    total_kg: np.ndarray        # (N,)
    cop_board: np.ndarray       # (N,2)     m
    cop_world: np.ndarray       # (N,3)     m (NaN without board)
    force_world: np.ndarray     # (N,3)     N (NaN without board)
    com_world: np.ndarray       # (N,3)     m
    com_board: np.ndarray       # (N,3)     m (NaN without board)
    board: BoardRegistration | None = None
    body_mass_kg: float | None = None
    events: list[dict] = field(default_factory=list)   # events.csv rows + "trc_time"
    wii: dict[str, np.ndarray] = field(default_factory=dict)  # full-rate wii.csv + "trc_time"
    warnings: list[str] = field(default_factory=list)

    @property
    def n_frames(self) -> int:
        return len(self.trc_time)

    @property
    def force_n(self) -> np.ndarray:
        """(N,) vertical force magnitude (N) = total_kg * g."""
        return self.total_kg * G

    @property
    def com_minus_cop(self) -> np.ndarray:
        """(N,2) horizontal COM - COP in the board frame (m)."""
        return self.com_board[:, :2] - self.cop_board

    def index_for_time(self, trc_time: float) -> int:
        """Row whose ``trc_time`` is nearest to ``trc_time`` (within half a frame), else -1.
        The 3D View maps its frame i via ``index_for_time(viewer._trc.times[i])``."""
        tt = self.trc_time
        if len(tt) == 0 or not np.isfinite(trc_time):
            return -1
        i = int(np.clip(np.searchsorted(tt, trc_time), 1, len(tt) - 1)) if len(tt) > 1 else 0
        if i > 0 and abs(tt[i - 1] - trc_time) <= abs(tt[i] - trc_time):
            i -= 1
        dt = np.diff(tt)
        half = 0.5 * (float(np.median(dt)) if len(dt) else 1.0)
        return i if abs(tt[i] - trc_time) <= half + 1e-9 else -1

    def table(self) -> dict[str, np.ndarray]:
        """``{column: (N,) array}`` for ``FUSED_COLUMNS``."""
        cmc = self.com_minus_cop
        cols = [self.frames.astype(float), self.trc_time, self.t_rel, self.t_unix,
                *self.kg.T, self.total_kg, self.force_n, *self.cop_board.T, *self.cop_world.T,
                *self.force_world.T, *self.com_world.T, *self.com_board.T, *cmc.T]
        return dict(zip(FUSED_COLUMNS, (np.asarray(c, float) for c in cols), strict=True))


def cop_to_world(board: BoardRegistration, cop_board: np.ndarray) -> np.ndarray:
    """(N,2) or (2,) board-frame COP -> (N,3)/(3,) world points on the top surface."""
    c = np.asarray(cop_board, np.float64)
    pts = np.concatenate([c, np.zeros(c.shape[:-1] + (1,))], axis=-1)
    return board.pose.board_to_world.apply(pts)


def plumb_point(board: BoardRegistration, com_world: np.ndarray) -> np.ndarray:
    """Projection of the COM (N,3)/(3,) along the board normal onto the board's top plane."""
    p = np.asarray(com_world, np.float64)
    up = np.asarray(board.up_world, np.float64)
    t = board.pose.board_to_world.t
    d = (p - t) @ up
    return p - np.asarray(d)[..., None] * up


def _cop_from_sensors(kg: np.ndarray, dx_m: float, dy_m: float, min_kg: float) -> np.ndarray:
    """(N,2) board COP from (N,4) TR, BR, TL, BL loads (``protocol.center_of_pressure``,
    vectorized); NaN below ``min_kg``."""
    tr, br, tl, bl = kg.T
    tot = tr + br + tl + bl
    with np.errstate(divide="ignore", invalid="ignore"):
        x = dx_m / 2 * ((tr + br) - (tl + bl)) / tot
        y = dy_m / 2 * ((tr + tl) - (br + bl)) / tot
    out = np.column_stack([x, y])
    out[~(tot >= min_kg)] = np.nan
    return out


def _force_params(meta: dict, board: BoardRegistration | None) -> tuple[float, float, float]:
    """Sensor spacing (m) and COP threshold (kg): board geometry > recording > defaults."""
    fs = meta.get("force_source") or {}
    if board is not None:
        dx, dy = board.geometry.sensor_dx_mm / 1000, board.geometry.sensor_dy_mm / 1000
    else:
        dx, dy = float(fs.get("sensor_dx_m") or 0.433), float(fs.get("sensor_dy_m") or 0.238)
    return dx, dy, float(fs.get("min_total_kg") or 5.0)


def fuse_trial(project, trc_path: str | Path | None = None) -> FusedTrial:
    """Build the ``FusedTrial`` of the project's trial.

    * .trc: ``trc_path`` or ``trc.find_trc_files(project)[0]``; read with ``read_trc_full``,
      converted to the world with ``yup_to_zup``; COM per row with ``com.center_of_mass``.
    * Wii: the trial's recording (``trial.load_trial``); rows mapped with
      ``alignment.trc_rows_to_wii_time``; force columns interpolated with ``analysis.interp``
      (``max_gap`` 0.1 s). COP world is recomputed from ``cop_board`` with the CURRENT board
      registration (``board.load_board``), never taken from wii.csv.
    * ``warnings``: stale board, no board, Pose2Sim triangulating with another calibration
      file than the board's (``calib.calib_mismatch``), videos changed since capture, alignment
      none, trc not covered by the recording, .trc older than the videos in videos/, ...
    Raises ValueError (user-readable) when there is no .trc or no recording.

    Details: an alignment "none" does NOT raise: the rows get NaN Wii data (the COM is still
    computed) and a warning says to align. The board COP is recomputed from the four sensor
    columns with the board geometry's sensor spacing (falls back to the recorded COP columns);
    it is NaN below the recording's ``min_total_kg`` (default 5 kg). A board located without
    camera extrinsics (``pose.world_camera``) is not in the .trc world: it is ignored (warning).
    ``wii`` holds the full-rate columns (COP as recomputed) plus ``trc_time``; ``events`` the
    marked events plus ``trc_time``.
    """
    from poseassess.wii import io as wio

    from .alignment import trc_rows_to_wii_time, wii_time_to_trc_time
    from .analysis import body_mass_from_force, interp
    from .board import load_board
    from .com import center_of_mass
    from .paths import WiiPaths
    from .trc import find_trc_files, read_trc_full
    from .trial import load_trial, videos_changed

    if trc_path is None:
        files = find_trc_files(project)
        if not files:
            raise ValueError("No .trc file in pose-3d/: run the pipeline first (4. Run).")
        trc_path = files[0]
    trc_path = Path(trc_path)
    trial = load_trial(project)
    if not trial.recording:
        raise ValueError("No Wii recording for this trial: record one (3b. Capture) or import "
                         "one (6. Results > Balance).")
    rec = WiiPaths(project).recording_dir(trial.recording)
    if not (rec / wio.WII_CSV).is_file():
        raise ValueError(f"The Wii recording {trial.recording} is missing or has no force data "
                         f"({wio.WII_CSV}).")
    trc = read_trc_full(trc_path)
    wii = wio.read_wii_csv(rec)
    meta = wio.read_session_json(rec)
    warnings: list[str] = []

    board = load_board(project)
    if board is not None and board.pose.world_camera is not None:
        warnings.append("The board was located without camera extrinsics: its position is not "
                        "in the 3D world of the .trc. Recompute it (2b. Wii Board) after the "
                        "extrinsic calibration.")
        board = None
    elif board is None:
        warnings.append("The board position is not computed (2b. Wii Board): COP and force "
                        "are known in the board frame only.")
    elif board.stale:
        warnings.append(board.stale)
    if board is not None:
        from .calib import calib_mismatch

        mismatch = calib_mismatch(project)
        if mismatch:
            warnings.append(mismatch)
    if trial.source == "capture" and videos_changed(project, trial):
        warnings.append("The videos changed since the capture: the recorded alignment may be "
                        "wrong. Re-align, or export the take to videos/ again (3b. Capture).")
    if trial.alignment.method == "none":
        warnings.append("The Wii recording is not aligned with the videos yet: detect a sync "
                        "event or set the offset (6. Results > Balance).")
    if _trc_older_than_videos(project, trc_path):
        warnings.append(f"{trc_path.name} is older than the videos in videos/: it may belong to "
                        "earlier videos (e.g. before a capture export). Run the pipeline again "
                        "(4. Run).")

    # ---- per-row Wii data
    n = trc.n_frames
    t_rel = trc_rows_to_wii_time(project, trc.frames, trc.times, trial)
    wt = wii["t_rel"]
    if not np.isfinite(wt).any():
        wt = wii["t"] - np.nanmin(wii["t"])
        wii["t_rel"] = wt
    dx, dy, min_kg = _force_params(meta, board)
    kg_full = np.column_stack([wii[c] for c in wio.SENSOR_COLS])
    total_full = wii["total_kg"]
    if not np.isfinite(total_full).any():
        total_full = kg_full.sum(axis=1)
        wii["total_kg"] = total_full
    if np.isfinite(kg_full).all(axis=1).any():
        cop_full = _cop_from_sensors(kg_full, dx, dy, min_kg)
    else:
        cop_full = np.column_stack([wii["cop_x_board"], wii["cop_y_board"]])
        cop_full[~(total_full >= min_kg)] = np.nan
    wii["cop_x_board"], wii["cop_y_board"] = cop_full[:, 0].copy(), cop_full[:, 1].copy()

    kg = interp(wt, kg_full, t_rel)
    total = interp(wt, total_full, t_rel)
    cop = interp(wt, cop_full, t_rel)
    cop[~(total >= min_kg)] = np.nan
    unix_off = np.nanmedian(wii["t_unix"] - wt) if np.isfinite(wii["t_unix"] - wt).any() \
        else np.nan
    t_unix = t_rel + unix_off

    # ---- COM and world quantities
    world = trc.world()
    com_w = np.full((n, 3), np.nan)
    for i in range(n):
        c = center_of_mass(world[i], trc.names)
        if c is not None:
            com_w[i] = c
    if not np.isfinite(com_w).any():
        warnings.append("The centre of mass could not be computed from the markers of this .trc "
                        "(shoulders, hips and knees are needed).")
    if board is not None:
        cop_w = cop_to_world(board, cop)
        with np.errstate(invalid="ignore"):
            force_w = np.asarray(board.up_world)[None, :] * (total * G)[:, None]
        com_b = board.pose.world_to_board.apply(com_w)
    else:
        cop_w = np.full((n, 3), np.nan)
        force_w = np.full((n, 3), np.nan)
        com_b = np.full((n, 3), np.nan)

    # ---- coverage
    if trial.alignment.method != "none":
        covered = np.isfinite(total)
        frac = float(covered.mean()) if n else 0.0
        if frac == 0:
            warnings.append("The .trc does not overlap the Wii recording: check the alignment.")
        elif frac < 0.999:
            warnings.append(f"The Wii recording covers only {100 * frac:.0f}% of the .trc "
                            "frames (no force data for the other frames).")

    # ---- events and full-rate data on the .trc time base
    rate = trc.data_rate if np.isfinite(trc.data_rate) and trc.data_rate > 0 else None
    wii["trc_time"] = wii_time_to_trc_time(project, wt, trial, rate_hz=rate)
    events = []
    for e in wio.read_events_csv(rec):
        e = dict(e)
        e["trc_time"] = float(wii_time_to_trc_time(project, np.array([e["t_rel"]]), trial,
                                                   rate_hz=rate)[0])
        events.append(e)
    body = trial.body_mass_kg if trial.body_mass_kg else body_mass_from_force(wt, total_full)

    return FusedTrial(
        trc_path=trc_path, recording=trial.recording, alignment=trial.alignment,
        frames=np.asarray(trc.frames, int), trc_time=np.asarray(trc.times, float),
        t_rel=t_rel, t_unix=t_unix, kg=kg, total_kg=total, cop_board=cop, cop_world=cop_w,
        force_world=force_w, com_world=com_w, com_board=com_b, board=board,
        body_mass_kg=body, events=events, wii=wii, warnings=warnings)


def _trc_older_than_videos(project, trc_path: Path, slack_s: float = 2.0) -> bool:
    """True when a camera video in videos/ was written after ``trc_path`` (the .trc may then
    belong to other videos, e.g. after "Export to videos/" of a new take)."""
    try:
        trc_m = trc_path.stat().st_mtime
        vids = [p for p in Path(project.videos_dir).iterdir()
                if p.is_file() and p.name.lower().startswith("cam")]
        return any(p.stat().st_mtime > trc_m + slack_s for p in vids)
    except (OSError, AttributeError):
        return False


def _finite(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def _clean(obj):
    """JSON-friendly copy (numpy -> python, NaN -> None)."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return _finite(obj)
    if isinstance(obj, np.ndarray):
        return _clean(obj.tolist())
    if isinstance(obj, Path):
        return str(obj)
    return obj


def balance_summary(fused: FusedTrial, t_from: float | None = None,
                    t_to: float | None = None) -> dict:
    """Metrics over the .trc time window [t_from, t_to] (default: all).

    Keys: trc, recording, alignment (dict), window [t_from, t_to], frames, duration_s,
    body_mass_kg, mean_total_kg, cop (``analysis.cop_sway_metrics`` of the FULL-RATE Wii COP
    inside the window: resampled to 100 Hz and low-pass filtered at 10 Hz before the path
    length / velocity, parameters in ``cop["filter"]``; board frame, mm), com
    (``sway_metrics`` of COM x/y in the board frame),
    com_cop {rms_distance_mm, rms_ml_mm, rms_ap_mm, corr_ml, corr_ap}, force_events (detected,
    with trc_time), events (marked), warnings.

    The COP falls back to the per-frame values when ``fused.wii`` has no full-rate data. The
    result is JSON-ready (NaN -> None); empty dicts when there are too few samples.
    """
    from .analysis import cop_sway_metrics, detect_force_events, sway_metrics

    tt = fused.trc_time
    fin = tt[np.isfinite(tt)]
    lo = float(t_from) if t_from is not None else (float(fin.min()) if len(fin) else 0.0)
    hi = float(t_to) if t_to is not None else (float(fin.max()) if len(fin) else 0.0)
    if hi < lo:
        lo, hi = hi, lo
    rows = (tt >= lo - 1e-9) & (tt <= hi + 1e-9)

    wii = fused.wii or {}
    wt = np.asarray(wii.get("trc_time", []), float)
    if len(wt) and np.isfinite(wt).any():
        wm = (wt >= lo - 1e-9) & (wt <= hi + 1e-9)
        t_w = np.asarray(wii.get("t_rel", wt), float)[wm]
        cop = cop_sway_metrics(t_w, np.asarray(wii["cop_x_board"], float)[wm],
                               np.asarray(wii["cop_y_board"], float)[wm])
        tot = np.asarray(wii.get("total_kg", []), float)[wm]
    else:
        wm = None
        cop = cop_sway_metrics(tt[rows], fused.cop_board[rows, 0], fused.cop_board[rows, 1])
        tot = fused.total_kg[rows]
    com = sway_metrics(tt[rows], fused.com_board[rows, 0], fused.com_board[rows, 1])

    d = fused.com_minus_cop[rows]
    ok = np.isfinite(d).all(axis=1)
    com_cop: dict = {}
    if ok.sum() >= 3:
        dd = d[ok]
        cx, cy = fused.com_board[rows][ok, 0], fused.com_board[rows][ok, 1]
        px, py = fused.cop_board[rows][ok, 0], fused.cop_board[rows][ok, 1]

        def corr(a, b):
            if np.std(a) < 1e-12 or np.std(b) < 1e-12:
                return None
            return float(np.corrcoef(a, b)[0, 1])

        com_cop = {"rms_distance_mm": float(np.sqrt(np.mean(np.sum(dd ** 2, axis=1))) * 1000),
                   "rms_ml_mm": float(np.sqrt(np.mean(dd[:, 0] ** 2)) * 1000),
                   "rms_ap_mm": float(np.sqrt(np.mean(dd[:, 1] ** 2)) * 1000),
                   "mean_ml_mm": float(np.mean(dd[:, 0]) * 1000),
                   "mean_ap_mm": float(np.mean(dd[:, 1]) * 1000),
                   "corr_ml": corr(cx, px), "corr_ap": corr(cy, py), "frames": int(ok.sum())}

    force_events = []
    if len(wt) and "total_kg" in wii and np.isfinite(wt).any():
        try:
            evs = detect_force_events({"t_rel": wii["t_rel"], "t": wii.get("t"),
                                       "t_unix": wii.get("t_unix"),
                                       "total_kg": wii["total_kg"]},
                                      body_kg=fused.body_mass_kg)
        except ValueError:
            evs = []
        order = np.argsort(np.asarray(wii["t_rel"], float))
        for e in evs:
            e = dict(e)
            e["trc_time"] = float(np.interp(e["t_rel"], np.asarray(wii["t_rel"], float)[order],
                                            wt[order]))
            if lo - 1e-9 <= e["trc_time"] <= hi + 1e-9:
                force_events.append(e)
    events = [dict(e) for e in fused.events
              if e.get("trc_time") is not None and np.isfinite(e["trc_time"])
              and lo - 1e-9 <= e["trc_time"] <= hi + 1e-9]
    tot = np.asarray(tot, float)
    out = {"trc": str(fused.trc_path), "recording": fused.recording,
           "alignment": fused.alignment.to_dict(), "window": [lo, hi],
           "frames": int(rows.sum()), "duration_s": hi - lo,
           "body_mass_kg": fused.body_mass_kg,
           "mean_total_kg": float(np.nanmean(tot)) if np.isfinite(tot).any() else None,
           "cop": cop, "com": com, "com_cop": com_cop, "force_events": force_events,
           "events": events, "warnings": list(fused.warnings)}
    return _clean(out)


def export_fused_csv(fused: FusedTrial, path: str | Path) -> Path:
    """Write ``fused.table()`` as CSV (``FUSED_COLUMNS``, %.6f, empty for NaN)."""
    import csv

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tab = fused.table()
    cols = [tab[c] for c in FUSED_COLUMNS]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(FUSED_COLUMNS)
        for i in range(fused.n_frames):
            row = []
            for name, c in zip(FUSED_COLUMNS, cols):
                v = c[i]
                if not np.isfinite(v):
                    row.append("")
                elif name == "frame":
                    row.append(str(int(round(v))))
                else:
                    row.append(f"{v:.6f}")
            w.writerow(row)
    return path


MOT_COLUMNS = ["time", "ground_force_vx", "ground_force_vy", "ground_force_vz",
               "ground_force_px", "ground_force_py", "ground_force_pz",
               "ground_torque_x", "ground_torque_y", "ground_torque_z"]


def export_grf_mot(fused: FusedTrial, path: str | Path) -> Path:
    """OpenSim external-loads .mot (Y-up .trc frame, same time column as the .trc):
    ground_force_vx/vy/vz (N), ground_force_px/py/pz (m), ground_torque_x/y/z (0).
    NaN rows (nobody on the board) are written as zero force at the board centre.
    ValueError when the trial has no usable board registration (``fused.board`` is None)."""
    from .trc import zup_to_yup

    if fused.board is None:
        raise ValueError("The ground reaction force needs the board position (2b. Wii Board).")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    center = zup_to_yup(fused.board.center_world)
    f = zup_to_yup(fused.force_world)
    p = zup_to_yup(fused.cop_world)
    bad = ~(np.isfinite(f).all(axis=1) & np.isfinite(p).all(axis=1))
    f[bad] = 0.0
    p[bad] = center
    t = fused.trc_time
    lines = [path.name, "version=1", f"nRows={fused.n_frames}", f"nColumns={len(MOT_COLUMNS)}",
             "inDegrees=yes", "endheader", "\t".join(MOT_COLUMNS)]
    for i in range(fused.n_frames):
        vals = [t[i], *f[i], *p[i], 0.0, 0.0, 0.0]
        lines.append("\t".join(f"{v:.6f}" for v in vals))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def export_all(project, fused: FusedTrial, out_dir: str | Path | None = None,
               t_from: float | None = None, t_to: float | None = None) -> dict[str, Path]:
    """Write ``<trc stem>_fused.csv``, ``<trc stem>_balance_summary.json`` and
    ``<trc stem>_grf.mot`` to ``out_dir`` (default ``wii/exports/``); returns
    ``{"fused": p, "summary": p, "grf": p}``.

    The fused table holds every .trc row; the summary covers [t_from, t_to]. Without a usable
    board registration the .mot is not written and ``"grf"`` is missing from the result (the
    reason is added to the summary's warnings)."""
    import json

    from .paths import WiiPaths

    out = Path(out_dir) if out_dir is not None else WiiPaths(project).exports_dir
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(fused.trc_path).stem
    paths = {"fused": export_fused_csv(fused, out / f"{stem}_fused.csv")}
    summary = balance_summary(fused, t_from, t_to)
    try:
        paths["grf"] = export_grf_mot(fused, out / f"{stem}_grf.mot")
    except ValueError as e:
        summary["warnings"] = list(summary.get("warnings") or []) + [f"No .mot export: {e}"]
    p = out / f"{stem}_balance_summary.json"
    p.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    paths["summary"] = p
    return {k: paths[k] for k in ("fused", "summary", "grf") if k in paths}
