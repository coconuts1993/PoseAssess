"""Signal analysis for the balance data: interpolation, sway metrics, force events, TRC
events and time-offset estimation.

Ported / adapted from PoseBoard 3ea6d8a ``poseboard/analysis.py`` (``interp``,
``sway_metrics``, ``detect_force_events``, ``estimate_time_offset``, ``write_events_csv``).
New for PoseAssess: ``detect_trc_events``, ``match_events``, ``xcorr_offset``,
``body_mass_from_force`` (plus the helpers ``trc_heights`` and ``foot_marker_indices``).

Time bases: force-side times are the Wii recording's ``t_rel`` (s); TRC-side times are the .trc
``Time`` column (s). An offset ``d`` means ``t_rel = trc_time + d``.
"""

from __future__ import annotations

import csv
import math
import re
from pathlib import Path

import numpy as np

EVENT_TYPES = ("step_on", "step_off", "takeoff", "landing", "stomp")
EVENT_COLS = ["type", "t", "t_rel", "t_unix", "flight_s", "peak_kg"]
G = 9.80665
# Event types seen by both the force and the markers (usable for time alignment).
SYNC_TYPES = ("takeoff", "landing", "stomp")


# ------------------------------------------------------------------ interpolation
def interp(t_src: np.ndarray, v: np.ndarray, t_dst: np.ndarray, max_gap: float = 0.1) -> np.ndarray:
    """Linear interpolation of ``v(t_src)`` (1-D, or (N,k) column-wise) at ``t_dst``; NaN outside
    the finite source range, inside gaps longer than ``max_gap`` s, or with < 2 finite samples."""
    t_src = np.asarray(t_src, np.float64).ravel()
    v = np.asarray(v, np.float64)
    t_dst = np.asarray(t_dst, np.float64)
    if v.ndim == 2:
        cols = [interp(t_src, v[:, j], t_dst, max_gap) for j in range(v.shape[1])]
        return (np.stack(cols, axis=-1) if cols
                else np.zeros(t_dst.shape + (0,), np.float64))
    v = v.ravel()
    out = np.full(t_dst.shape, np.nan)
    ok = np.isfinite(t_src) & np.isfinite(v)
    if ok.sum() < 2:
        return out
    ts, vs = t_src[ok], v[ok]
    if np.any(np.diff(ts) < 0):
        order = np.argsort(ts, kind="stable")
        ts, vs = ts[order], vs[order]
    td = t_dst.ravel()
    res = np.interp(td, ts, vs, left=np.nan, right=np.nan)
    j = np.clip(np.searchsorted(ts, td), 1, len(ts) - 1)
    gap = ts[j] - ts[j - 1]
    with np.errstate(invalid="ignore"):
        near = np.minimum(np.abs(td - ts[j - 1]), np.abs(ts[j] - td))
        res[(gap > max_gap) & (near > 1e-9)] = np.nan
    return res.reshape(t_dst.shape)


# ----------------------------------------------------------------------- metrics
def sway_metrics(t: np.ndarray, x: np.ndarray, y: np.ndarray) -> dict:
    """Postural sway of a planar trajectory in METRES (board frame: x = medio-lateral / subject's
    right, y = antero-posterior / front); results in mm. Keys: duration_s, samples, mean_x_mm,
    mean_y_mm, range_ml_mm, range_ap_mm, rms_ml_mm, rms_ap_mm, path_length_mm,
    mean_velocity_mm_s, ellipse95_area_mm2. {} with fewer than 10 finite samples."""
    t, x, y = (np.asarray(a, np.float64).ravel() for a in (t, x, y))
    ok = np.isfinite(t) & np.isfinite(x) & np.isfinite(y)
    t, x, y = t[ok], x[ok] * 1000, y[ok] * 1000
    if len(t) < 10:
        return {}
    dur = float(t[-1] - t[0])
    path = float(np.sum(np.hypot(np.diff(x), np.diff(y))))
    xc, yc = x - x.mean(), y - y.mean()
    eig = np.linalg.eigvalsh(np.cov(np.vstack([xc, yc])))
    # 95% confidence ellipse area: pi * chi2(0.95, 2) * sqrt(l1 * l2)
    area95 = float(np.pi * 5.991 * np.sqrt(max(eig[0], 0) * max(eig[1], 0)))
    return {
        "duration_s": dur,
        "samples": int(len(t)),
        "mean_x_mm": float(x.mean()), "mean_y_mm": float(y.mean()),
        "range_ml_mm": float(np.ptp(x)), "range_ap_mm": float(np.ptp(y)),
        "rms_ml_mm": float(np.sqrt(np.mean(xc ** 2))),
        "rms_ap_mm": float(np.sqrt(np.mean(yc ** 2))),
        "path_length_mm": path,
        "mean_velocity_mm_s": path / dur if dur > 0 else float("nan"),
        "ellipse95_area_mm2": area95,
    }


# Standard posturography preprocessing of the COP before the sway metrics (path length and
# velocity are otherwise inflated by sensor noise and depend on the irregular Bluetooth rate).
COP_FILTER = {"resample_hz": 100.0, "lowpass_hz": 10.0, "order": 4, "max_gap_s": 0.1}


def filter_cop(t: np.ndarray, x: np.ndarray, y: np.ndarray, resample_hz: float = 100.0,
               lowpass_hz: float = 10.0, order: int = 4,
               max_gap_s: float = 0.1) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """COP ``(t, x, y)`` -> segments ``[(t_u, x_f, y_f), ...]``: split at NaN / gaps longer than
    ``max_gap_s``, resampled to a uniform ``resample_hz`` grid (linear) and low-pass filtered
    (zero-phase Butterworth of ``order`` at ``lowpass_hz``, ``scipy.signal.sosfiltfilt``).
    Segments too short to be filtered (< ~0.2 s at 100 Hz) are dropped."""
    from scipy.signal import butter, sosfiltfilt

    t, x, y = (np.asarray(a, np.float64).ravel() for a in (t, x, y))
    ok = np.isfinite(t) & np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 2:
        return []
    order_idx = np.argsort(t, kind="stable")
    t, x, y, ok = t[order_idx], x[order_idx], y[order_idx], ok[order_idx]
    sos = butter(int(order), float(lowpass_hz), fs=float(resample_hz), output="sos")
    min_len = 3 * (2 * len(sos) + 1) + 1  # sosfiltfilt's default padding + 1
    # a break wherever a sample is missing or the time step is too long
    idx = np.flatnonzero(ok)
    breaks = np.flatnonzero((np.diff(idx) > 1) | (np.diff(t[idx]) > max_gap_s)) + 1
    out = []
    for part in np.split(idx, breaks):
        if len(part) < 2:
            continue
        ts = t[part]
        tu = np.arange(ts[0], ts[-1] + 1e-9, 1.0 / resample_hz)
        if len(tu) < min_len:
            continue
        xu, yu = np.interp(tu, ts, x[part]), np.interp(tu, ts, y[part])
        out.append((tu, sosfiltfilt(sos, xu), sosfiltfilt(sos, yu)))
    return out


def cop_sway_metrics(t: np.ndarray, x: np.ndarray, y: np.ndarray,
                     filter_params: dict | None = None) -> dict:
    """``sway_metrics`` of the COP after ``filter_cop`` (resampled + low-pass filtered).

    The path length (and mean velocity = path / time covered by the segments) is summed within
    the segments, never across a gap. ``samples`` stays the number of raw finite samples;
    extra keys: ``path_length_raw_mm`` (unfiltered, for comparison) and ``filter`` (the
    parameters). {} with fewer than 10 raw samples or nothing left after filtering."""
    fp = {**COP_FILTER, **(filter_params or {})}
    raw = sway_metrics(t, x, y)
    if not raw:
        return {}
    segs = filter_cop(t, x, y, **fp)
    if not segs:
        return {}
    tt = np.concatenate([s[0] for s in segs])
    m = sway_metrics(tt, np.concatenate([s[1] for s in segs]),
                     np.concatenate([s[2] for s in segs]))
    if not m:
        return {}
    path = float(sum(np.sum(np.hypot(np.diff(sx), np.diff(sy))) for _, sx, sy in segs)) * 1000
    covered = float(sum(st[-1] - st[0] for st, _, _ in segs))
    m.update(path_length_mm=path,
             mean_velocity_mm_s=path / covered if covered > 0 else float("nan"),
             samples=raw["samples"], path_length_raw_mm=raw["path_length_mm"],
             filter={"resample_hz": fp["resample_hz"], "lowpass_hz": fp["lowpass_hz"],
                     "order": int(fp["order"]), "max_gap_s": fp["max_gap_s"]})
    return m


def body_mass_from_force(t: np.ndarray, total_kg: np.ndarray, min_kg: float = 20.0,
                         window_s: float = 1.0) -> float | None:
    """Body mass (kg) = median total load over the quietest ``window_s`` windows while someone
    stands on the board (total > ``min_kg``); None when nobody stood on it long enough.

    Windows overlap by half; a window counts when every sample in it is above ``min_kg`` and
    its samples span at least 80 % of ``window_s``. The quietest quarter (lowest standard
    deviation, at least one window) is used, so jumps and steps do not bias the result."""
    t, f = np.asarray(t, np.float64).ravel(), np.asarray(total_kg, np.float64).ravel()
    ok = np.isfinite(t) & np.isfinite(f)
    t, f = t[ok], f[ok]
    if len(t) < 3:
        return None
    order = np.argsort(t, kind="stable")
    t, f = t[order], f[order]
    stats = []
    for s in np.arange(t[0], t[-1] - 0.8 * window_s + 1e-9, window_s / 2):
        a, b = np.searchsorted(t, [s, s + window_s])
        w, tw = f[a:b], t[a:b]
        if len(w) < 5 or tw[-1] - tw[0] < 0.8 * window_s or np.any(w <= min_kg):
            continue
        stats.append((float(np.std(w)), float(np.median(w))))
    if not stats:
        return None
    stats.sort()
    q = max(1, int(math.ceil(0.25 * len(stats))))
    return float(np.median([m for _, m in stats[:q]]))


# ------------------------------------------------------------------ force events
def _load_force(data) -> tuple[np.ndarray, np.ndarray, float, float]:
    """``(t_rel, total_kg, t0, unix_offset)`` from a recording folder, a wii.csv path, a dict of
    columns or a ``(t_rel, total_kg)`` pair. ``t = t_rel + t0``, ``t_unix = t_rel + unix_offset``
    (NaN when unknown)."""
    from poseassess.wii import io as wio

    session_t0 = None
    if isinstance(data, (str, Path)):
        p = Path(data)
        cols = wio.read_wii_csv(p)
        meta = wio.read_session_json(p if p.is_dir() else p.parent)
        if isinstance(meta.get("t0"), (int, float)):
            session_t0 = float(meta["t0"])
    elif isinstance(data, dict):
        cols = data
    else:
        t_rel, total = data
        return (np.asarray(t_rel, np.float64).ravel(), np.asarray(total, np.float64).ravel(),
                float("nan"), float("nan"))

    def col(name):
        v = cols.get(name)
        if v is None:
            return None
        v = np.asarray(v, np.float64).ravel()
        return v if np.isfinite(v).any() else None

    t, t_rel, t_unix = col("t"), col("t_rel"), col("t_unix")
    total = col("total_kg")
    if total is None:
        sensors = [col(c) for c in wio.SENSOR_COLS]
        if any(s is None for s in sensors):
            raise ValueError("No total_kg (or sensor) column in the force data")
        total = np.sum(sensors, axis=0)
    if t_rel is None:
        if t is None:
            raise ValueError("No time column (t_rel or t) in the force data")
        t0 = session_t0 if session_t0 is not None else float(np.nanmin(t))
        t_rel = t - t0
    t0 = float(np.nanmedian(t - t_rel)) if t is not None else float("nan")
    off = float(np.nanmedian(t_unix - t_rel)) if t_unix is not None else float("nan")
    return t_rel, total, t0, off


def detect_force_events(data, *, body_kg: float | None = None, min_load_kg: float = 10.0,
                        on_fraction: float = 0.5, hysteresis: float = 0.1,
                        flight_kg: float = 5.0, min_flight_s: float = 0.05,
                        max_flight_s: float = 1.0, peak_factor: float = 1.5,
                        jump_guard_s: float = 0.5, max_gap_s: float = 0.1) -> list[dict]:
    """Events in the total force (port of PoseBoard's ``detect_force_events``).

    ``data``: a recording folder, a wii.csv path, a dict of wii.csv columns, or a ``(t_rel,
    total_kg)`` pair. Returns events sorted by time: ``{type, t_rel, t, t_unix, [flight_s],
    [peak_kg]}`` with ``type`` in ``EVENT_TYPES`` (step_on/step_off at ``on_fraction`` of body
    weight with hysteresis; takeoff/landing around an unloaded phase < ``max_flight_s``; stomp =
    peak > ``peak_factor`` x body weight that is not a jump). ``t``/``t_unix`` are NaN when the
    input has no such columns.

    Event times: step_on/step_off = crossing of ``on_fraction`` x body weight; takeoff/landing
    = crossing of ``flight_kg`` (last / first foot contact); stomp = time of the force maximum
    (parabolic interpolation). Stomps also carry ``t_rel_onset``: the rise through 10 % of the
    peak above body weight (about the moment of foot contact), used by ``match_events``.

    Data gaps: a jump with a gap of more than ``max_gap_s`` around its unloaded phase is not
    reported (its timing is uncertain), and its push-off / landing peaks are not reported as
    stomps either (they would be matched against real stomps).
    """
    t, f, t0, off = _load_force(data)
    ok = np.isfinite(t) & np.isfinite(f)
    t, f = t[ok], f[ok]
    if len(t) < 3:
        return []
    order = np.argsort(t, kind="stable")
    t, f = t[order], f[order]
    if body_kg is None:
        loaded = f[f > min_load_kg]
        if not len(loaded):
            return []
        body_kg = float(np.median(loaded))
    mid = on_fraction * body_kg
    hi, lo = mid + hysteresis * body_kg, mid - hysteresis * body_kg
    n = len(t)

    def crossing(i: int, level: float, rising: bool) -> float | None:
        """Time where f crosses ``level`` last before sample i (inclusive), interpolated."""
        j = i - 1
        while j >= 0 and ((f[j] >= level) if rising else (f[j] <= level)):
            j -= 1
        if j < 0 or j + 1 >= n or t[j + 1] - t[j] > max_gap_s:
            return None
        den = f[j + 1] - f[j]
        if den == 0:
            return float(t[j + 1])
        return float(t[j] + (level - f[j]) / den * (t[j + 1] - t[j]))

    # Hysteresis state machine: indices where the subject gets on / off the board
    on = f[0] >= mid
    flips: list[tuple[int, bool]] = []
    for i in range(1, n):
        if not on and f[i] >= hi:
            on = True
            flips.append((i, True))
        elif on and f[i] <= lo:
            on = False
            flips.append((i, False))

    events: list[dict] = []

    def add(kind: str, tt: float | None, **extra) -> dict | None:
        if tt is None:
            return None
        ev = {"type": kind, "t_rel": float(tt), "t": float(tt + t0), "t_unix": float(tt + off)}
        ev.update(extra)
        events.append(ev)
        return ev

    landings: list[dict] = []
    takeoffs: list[float] = []
    # Unloaded phases that look like a jump but whose timing cannot be trusted (a data gap,
    # e.g. a Bluetooth dropout or a stalled reader, next to the takeoff or landing): no
    # takeoff/landing is reported, and their push-off / landing peaks are not stomps either.
    unsure: list[tuple[float, float]] = []
    for k, (i, rising) in enumerate(flips):
        if rising:
            if k == 0:  # started empty, then stepped on
                add("step_on", crossing(i, mid, True))
            continue
        # i: the subject left the board (or unloaded it); look at what happens next
        nxt = flips[k + 1][0] if k + 1 < len(flips) else None
        if nxt is None:
            add("step_off", crossing(i, mid, False))
            continue
        t_off, t_on = crossing(i, mid, False), crossing(nxt, mid, True)
        dur = (t_on - t_off) if t_off is not None and t_on is not None else t[nxt] - t[i]
        seg = slice(i, nxt)
        no_gap = np.all(np.diff(t[max(i - 1, 0):nxt + 1]) <= max_gap_s)
        if dur <= max_flight_s and f[seg].min() < flight_kg:
            jumped = False
            if no_gap:
                idx = np.flatnonzero(f[seg] < flight_kg) + i
                t_up = crossing(int(idx[0]), flight_kg, False)
                q = int(idx[-1]) + 1
                t_down = crossing(q, flight_kg, True) if q < n else None
                if t_up is not None and t_down is not None and t_down - t_up >= min_flight_s:
                    add("takeoff", t_up, flight_s=t_down - t_up)
                    takeoffs.append(t_up)
                    landings.append(add("landing", t_down, flight_s=t_down - t_up))
                    jumped = True
            if not jumped:
                unsure.append((float(t[max(i - 1, 0)]), float(t[min(nxt, n - 1)])))
        elif dur > max_flight_s:
            add("step_off", t_off)
            add("step_on", t_on)
        # else: a short partial unloading (e.g. countermovement) - not an event

    # Sharp peaks above peak_factor x body weight
    above = np.concatenate([[False], f > peak_factor * body_kg, [False]])
    edges = np.diff(above.astype(np.int8))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    for a, b in zip(starts, ends):
        p = a + int(np.argmax(f[a:b]))
        tp = float(t[p])
        if 0 < p < n - 1 and t[p + 1] - t[p - 1] <= 2 * max_gap_s:
            den = f[p - 1] - 2 * f[p] + f[p + 1]
            if den < 0:
                tp += (float(np.clip(0.5 * (f[p - 1] - f[p + 1]) / den, -0.5, 0.5))
                       * (t[p + 1] - t[p - 1]) / 2)
        peak = float(f[p])
        land = next((e for e in landings if e["t_rel"] - max_gap_s <= tp
                     <= e["t_rel"] + jump_guard_s), None)
        if land is not None:
            land["peak_kg"] = max(peak, land.get("peak_kg", 0.0))
        elif not any(to - jump_guard_s <= tp <= to for to in takeoffs) and \
                not any(a - jump_guard_s <= tp <= b + jump_guard_s for a, b in unsure):
            ev = add("stomp", tp, peak_kg=peak)
            onset = _rise_onset(t, f, p, body_kg + 0.1 * (peak - body_kg), max_gap_s)
            if onset is not None and tp - onset <= 0.25:
                ev["t_rel_onset"] = onset
    events.sort(key=lambda e: e["t_rel"])
    return events


def _rise_onset(t: np.ndarray, f: np.ndarray, p: int, level: float,
                max_gap_s: float) -> float | None:
    """Time where f last rises through ``level`` before the peak at index ``p``."""
    j = p - 1
    while j >= 0 and f[j] > level:
        j -= 1
    if j < 0 or t[j + 1] - t[j] > max_gap_s or f[j + 1] == f[j]:
        return None
    return float(t[j] + (level - f[j]) / (f[j + 1] - f[j]) * (t[j + 1] - t[j]))


def write_events_csv(path: str | Path, events: list[dict]) -> Path:
    """Write detected events (``EVENT_COLS``) to a CSV. Returns the path."""
    path = Path(path)

    def fmt(v) -> str:
        try:
            v = float(v)
        except (TypeError, ValueError):
            return ""
        return f"{v:.6f}" if np.isfinite(v) else ""

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(EVENT_COLS)
        for e in events:
            w.writerow([e.get("type", "")] + [fmt(e.get(c)) for c in EVENT_COLS[1:]])
    return path


def estimate_time_offset(t_ref, t_other, tolerance: float = 0.05) -> tuple[float, int]:
    """Offset ``d`` with ``t_other + d ≈ t_ref`` from two event-time lists (type-agnostic, PoseBoard
    algorithm: best pairwise difference by number of matches, then median). ``(nan, 0)`` when
    either list is empty."""
    a = np.sort(np.asarray(t_ref, np.float64).ravel())
    b = np.sort(np.asarray(t_other, np.float64).ravel())
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if not len(a) or not len(b):
        return float("nan"), 0
    best = (-1, np.inf, 0.0)
    for d in np.unique((a[:, None] - b[None, :]).ravel()):
        r = np.abs(a[:, None] - (b[None, :] + d)).min(axis=1)
        m = r <= tolerance
        cand = (int(m.sum()), float(r[m].mean()), float(d))
        if cand[0] > best[0] or (cand[0] == best[0] and cand[1] < best[1]):
            best = cand
    d = best[2]
    j = np.abs(a[:, None] - (b[None, :] + d)).argmin(axis=1)
    diff = a - b[j]
    m = np.abs(diff - d) <= tolerance
    return float(np.median(diff[m])), int(m.sum())


# ------------------------------------------------------------------- TRC events
_FOOT_WORDS = re.compile(r"(heel|calc|toe|footindex|foot|ankle|ank|meta)")


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def foot_marker_indices(names: list[str]) -> dict[str, list[int]]:
    """``{"L": [...], "R": [...]}``: indices of the heel / toe / ankle / foot markers per side
    (HALPE ``RHeel``/``RBigToe``, MediaPipe ``right_foot_index``, BVH ``RightFoot``, Pose2Sim
    LSTM ``r_calc_study``...)."""
    out: dict[str, list[int]] = {"L": [], "R": []}
    for i, n in enumerate(names):
        k = _norm(n)
        for side, prefixes in (("L", ("left", "l")), ("R", ("right", "r"))):
            if any(k.startswith(p) and _FOOT_WORDS.search(k[len(p):]) for p in prefixes):
                out[side].append(i)
                break
    return out


def trc_heights(names: list[str], coords_world: np.ndarray, up: np.ndarray | None = None
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(left_foot, right_foot, com)`` heights along ``up`` (N,) in metres (NaN when missing):
    the lowest foot marker per side and the whole-body COM (``com.center_of_mass``)."""
    from .com import center_of_mass

    X = np.asarray(coords_world, np.float64)
    up = np.array([0.0, 0.0, 1.0]) if up is None else np.asarray(up, np.float64).ravel()
    up = up / np.linalg.norm(up)
    h = X @ up  # (N,K)
    feet = foot_marker_indices(names)
    side_h = []
    for side in ("L", "R"):
        idx = feet[side]
        side_h.append(np.fmin.reduce(h[:, idx], axis=1) if idx else np.full(len(X), np.nan))
    com = np.full(len(X), np.nan)
    for i in range(len(X)):
        c = center_of_mass(X[i], names)
        if c is not None:
            com[i] = float(np.asarray(c) @ up)
    return side_h[0], side_h[1], com


def _fill_short_gaps(t: np.ndarray, v: np.ndarray, max_gap: float = 0.2) -> np.ndarray:
    return np.where(np.isfinite(v), v, interp(t, v, t, max_gap=max_gap))


def _rolling_baseline(t: np.ndarray, v: np.ndarray, half_s: float = 1.5, q: float = 20.0,
                      step_s: float = 0.1) -> np.ndarray:
    """Local standing height of a foot: the ``q``-th percentile of ``v`` in a centred
    +-``half_s`` window (evaluated every ``step_s`` and interpolated; ``t`` sorted). Robust to
    jumps and stomps (airborne < 80 % of the window) and follows a subject who steps onto or
    off the board, or stands on one leg. NaN when no window has 5 finite samples."""
    ok = np.isfinite(v)
    if ok.sum() < 5:
        return np.full(len(t), np.nan)
    centers = np.arange(t[0], t[-1] + step_s, step_s)
    lo = np.searchsorted(t, centers - half_s, side="left")
    hi = np.searchsorted(t, centers + half_s, side="right")
    vals = np.full(len(centers), np.nan)
    for i, (a, b) in enumerate(zip(lo, hi)):
        w = v[a:b][ok[a:b]]
        if len(w) >= 5:
            vals[i] = np.percentile(w, q)
    good = np.isfinite(vals)
    if not good.any():
        return np.full(len(t), np.nan)
    return np.interp(t, centers[good], vals[good])


def _level(t: np.ndarray, v: np.ndarray, t_from: float, t_to: float) -> float:
    """Median of the finite ``v`` with ``t_from <= t <= t_to`` (NaN when none)."""
    m = (t >= t_from) & (t <= t_to) & np.isfinite(v)
    return float(np.median(v[m])) if m.any() else float("nan")


def _same_level(t: np.ndarray, v: np.ndarray, t_a: float, t_b: float, tol: float,
                gap_s: float = 0.2, span_s: float = 0.5) -> bool:
    """Whether ``v`` is back at its level of before ``t_a`` after ``t_b`` (within ``tol``):
    False for a step onto / off the board or a leg that stays lifted, which are not jumps or
    stomps. True when either side has no data."""
    pre = _level(t, v, t_a - gap_s - span_s, t_a - gap_s)
    post = _level(t, v, t_b + gap_s, t_b + gap_s + span_s)
    return not (np.isfinite(pre) and np.isfinite(post)) or abs(post - pre) <= tol


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Inclusive (start, end) index pairs of the True runs of ``mask``."""
    m = np.concatenate([[False], np.asarray(mask, bool), [False]])
    d = np.diff(m.astype(np.int8))
    return [(int(a), int(b)) for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1)]


def _crossing(t: np.ndarray, s: np.ndarray, i: int, j: int, level: float) -> float:
    """Time where s crosses ``level`` between frames i and j (linear)."""
    if s[j] == s[i]:
        return float(t[j])
    return float(t[i] + (level - s[i]) / (s[j] - s[i]) * (t[j] - t[i]))


def _extrapolate_contact(t: np.ndarray, s: np.ndarray, k_prev: int, k_next: int) -> float:
    """Time where the height ``s`` reaches 0 between frames ``k_prev`` (in the air) and
    ``k_next`` (on the ground), using the velocity of the airborne frames next to ``k_prev``."""
    lo, hi = sorted((float(t[k_prev]), float(t[k_next])))
    step = 1 if k_next > k_prev else -1
    k2 = k_prev - step  # the airborne frame before k_prev (further from the ground)
    v = np.nan
    if 0 <= k2 < len(s) and np.isfinite(s[k2]) and s[k2] > s[k_prev]:
        v = (s[k2] - s[k_prev]) / abs(t[k2] - t[k_prev])
    if not np.isfinite(v) or v <= 0:
        dv = s[k_prev] - s[k_next]
        v = dv / abs(t[k_next] - t[k_prev]) if dv > 0 else np.nan
    if not np.isfinite(v) or v <= 0:
        return 0.5 * (lo + hi)
    return float(np.clip(t[k_prev] + step * s[k_prev] / v, lo, hi))


def _ballistic_roots(t: np.ndarray, s: np.ndarray) -> tuple[float, float]:
    """Roots of ``s(t) = c0 + c1 (t - tm) - g/2 (t - tm)^2`` fitted to airborne heights (NaN,
    NaN when fewer than 2 finite frames or no real roots)."""
    ok = np.isfinite(s)
    if ok.sum() < 2:
        return float("nan"), float("nan")
    tt, ss = t[ok], s[ok]
    tm = float(tt.mean())
    tau = tt - tm
    y = ss + 0.5 * G * tau ** 2
    A = np.column_stack([np.ones_like(tau), tau])
    (c0, c1), *_ = np.linalg.lstsq(A, y, rcond=None)
    disc = c1 ** 2 + 2 * G * c0
    if disc < 0:
        return float("nan"), float("nan")
    r = math.sqrt(disc)
    return tm + (c1 - r) / G, tm + (c1 + r) / G


def detect_trc_events(times: np.ndarray, names: list[str], coords_world: np.ndarray,
                      up: np.ndarray | None = None, floor_z: float | None = None,
                      lift_m: float = 0.03, min_flight_s: float = 0.05,
                      max_flight_s: float = 1.0, stomp_speed: float = 0.5,
                      heights: tuple | None = None) -> list[dict]:
    """Sync events from marker motion (NEW). ``coords_world`` (N,K,3) in the Calib world
    (``TrcTrajectories.world()``), ``up`` = world up unit vector (default +Z; pass
    ``BoardRegistration.up_world`` when known).

    Uses the height (along ``up``) of the feet (lowest of Heel/BigToe/SmallToe/Ankle per side,
    alias-tolerant like ``com._PART_ALIASES``) and of the COM:
    * ``takeoff`` / ``landing``: both feet leave / return to their standing height
      (baseline + ~3 cm threshold, sub-frame interpolated), flight 0.05..1.0 s;
      ``flight_s`` attached to both.
    * ``stomp``: one foot hits its baseline height after a fast downward motion
      (> ~0.5 m/s) while the other foot stays down.
    A candidate is rejected when a foot is not back at its height of before the event (a step
    onto or off the board, a leg that stays lifted).
    Returns ``[{type, t (TRC time), side?, flight_s?}]`` sorted by ``t``; [] when no feet
    markers are present.

    Details: the standing height of each foot is LOCAL: its 20th-percentile height in a
    centred +-1.5 s window (``_rolling_baseline``), so subjects who start on the floor next to
    the board or stand on one leg for most of the trial work too (or ``floor_z``, an explicit
    standing foot height along ``up``); ``lift_m`` is the threshold above it. Jump
    times are the roots of a ballistic parabola (fixed curvature -g/2) fitted to the airborne
    frames (frame-rate independent), else the threshold crossings extrapolated with the local
    velocity; a jump also needs the COM to rise (when computable). Stomp times are the moment
    of foot contact, extrapolated from the descent velocity. ``heights`` = precomputed
    ``trc_heights(names, coords_world, up)``.
    """
    t = np.asarray(times, np.float64).ravel()
    if len(t) < 5:
        return []
    hl, hr, com = heights if heights is not None else trc_heights(names, coords_world, up)
    sides = {k: _fill_short_gaps(t, np.asarray(v, np.float64)) for k, v in (("L", hl), ("R", hr))
             if np.isfinite(v).sum() >= 5}
    if not sides:
        return []
    rel = {}
    for k, v in sides.items():
        base = float(floor_z) if floor_z is not None else _rolling_baseline(t, v)
        rel[k] = v - base
    com = np.asarray(com, np.float64)
    dt_med = float(np.median(np.diff(t)))
    events: list[dict] = []

    # ---- jumps: both feet above the threshold
    jump_spans: list[tuple[float, float]] = []
    if len(rel) == 2:  # with one foot only a jump cannot be told from a step
        s = np.fmin(rel["L"], rel["R"])
        with np.errstate(invalid="ignore"):
            air = s > lift_m
        for a, b in _runs(air):
            if a == 0 or b >= len(t) - 1 or not (np.isfinite(s[a - 1]) and np.isfinite(s[b + 1])):
                continue
            if _crossing(t, s, b, b + 1, lift_m) - _crossing(t, s, a - 1, a, lift_m) \
                    > max_flight_s + 2 * dt_med:
                continue
            if np.isfinite(com[a:b + 1]).any():
                pre = com[max(0, a - int(round(1.0 / dt_med))):a]
                if np.isfinite(pre).any() and np.nanmax(com[a:b + 1]) < np.nanmedian(pre) + 0.02:
                    continue  # feet up but the body did not rise: marker glitch, not a jump
            if not all(_same_level(t, v, t[a], t[b], lift_m) for v in sides.values()):
                continue  # stepped onto / off the board (the local baseline lags), not a jump
            # fit the parabola to every clearly airborne frame (> 1 cm), not only those above
            # the threshold: short jumps at low frame rates have only 1-2 frames above it
            a_fit, b_fit = a, b
            while a_fit > max(a - 3, 1) and np.isfinite(s[a_fit - 1]) and s[a_fit - 1] > 0.01:
                a_fit -= 1
            while b_fit < min(b + 3, len(t) - 2) and np.isfinite(s[b_fit + 1]) \
                    and s[b_fit + 1] > 0.01:
                b_fit += 1
            t_to, t_ld = _ballistic_roots(t[a_fit:b_fit + 1], s[a_fit:b_fit + 1])
            win = max(2 * dt_med, 0.1)  # the 3 cm threshold is reached well within 0.1 s
            if not (t[a] - win <= t_to <= t[a] and t[b] <= t_ld <= t[b] + win):
                # last / first frame on the ground (within 1 cm) next to the airborne run
                k0 = next((k for k in range(a - 1, max(a - 6, -1), -1)
                           if np.isfinite(s[k]) and s[k] <= 0.01), a - 1)
                k1 = next((k for k in range(b + 1, min(b + 6, len(t)))
                           if np.isfinite(s[k]) and s[k] <= 0.01), b + 1)
                t_to = _extrapolate_contact(t, s, min(k0 + 1, a), k0)
                t_ld = _extrapolate_contact(t, s, max(k1 - 1, b), k1)
            flight = t_ld - t_to
            if not (min_flight_s <= flight <= max_flight_s):
                continue
            events.append({"type": "takeoff", "t": float(t_to), "flight_s": float(flight)})
            events.append({"type": "landing", "t": float(t_ld), "flight_s": float(flight)})
            jump_spans.append((float(t[a - 1]), float(t[b + 1])))

    # ---- stomps: one foot lifted and slammed down while the other stays down
    if len(rel) == 2:
        for side, other in (("L", "R"), ("R", "L")):
            h, o = rel[side], rel[other]
            with np.errstate(invalid="ignore"):
                lifted = h > lift_m
            for a, b in _runs(lifted):
                if a == 0 or b >= len(t) - 1 or t[b] - t[a] > 2.0:
                    continue
                if any(lo <= t[a] <= hi for lo, hi in jump_spans):
                    continue
                win = o[max(0, a - 1):b + 2]
                if np.isfinite(win).sum() < 0.5 * len(win) or np.nanmax(win) > lift_m:
                    continue  # the other foot did not stay down
                # first frame back on the ground (within 1 cm) after the lift
                k1 = next((k for k in range(b + 1, min(b + 4, len(t)))
                           if np.isfinite(h[k]) and h[k] <= 0.01), None)
                if k1 is None or not _same_level(t, sides[side], t[a], t[k1], lift_m):
                    continue  # never came back down (e.g. single-leg stance), not a stomp
                seg = np.arange(max(a, k1 - 5), k1 + 1)
                v = -np.diff(h[seg]) / np.diff(t[seg])
                if not np.isfinite(v).any() or np.nanmax(v) < stomp_speed:
                    continue
                t_c = _extrapolate_contact(t, h, k1 - 1, k1)
                events.append({"type": "stomp", "t": float(t_c), "side": side,
                               "speed_m_s": float(np.nanmax(v))})
    events.sort(key=lambda e: e["t"])
    return events


# ----------------------------------------------------------------- event matching
def _force_sync_time(e: dict) -> float:
    """Force-side time used for matching: the contact onset for stomps, else ``t_rel``."""
    v = e.get("t_rel_onset") if e.get("type") == "stomp" else None
    return float(v if v is not None and np.isfinite(v) else e.get("t_rel", np.nan))


def _pair(F: list, T: list, d: float, tol: float) -> list[tuple[str, float, float]]:
    """Greedy one-to-one same-type pairing of force (type, t_rel) and TRC (type, t) events;
    returns ``[(type, trc_t, t_rel)]`` sorted by ``trc_t``."""
    cand = sorted((abs(tf - (tt + d)), i, j) for i, (a, tf) in enumerate(F)
                  for j, (b, tt) in enumerate(T) if a == b and abs(tf - (tt + d)) <= tol)
    used_f, used_t, out = set(), set(), []
    for _, i, j in cand:
        if i in used_f or j in used_t:
            continue
        used_f.add(i)
        used_t.add(j)
        out.append((T[j][0], T[j][1], F[i][1]))
    return sorted(out, key=lambda p: p[1])


def match_events(force_events: list[dict], trc_events: list[dict], tolerance: float = 0.08,
                 max_abs_offset_s: float | None = None) -> dict:
    """Type-aware matching of force events (``t_rel``) and TRC events (``t``): the offset ``d``
    (``t_rel ≈ trc_t + d``) supported by the most same-type pairs, refined by the median.
    Returns ``{"offset_s": float|nan, "n_matched": int, "residual_ms": float, "pairs":
    [(type, trc_t, t_rel), ...]}``.

    Only ``SYNC_TYPES`` are matched; for force stomps the contact onset (``t_rel_onset``) is
    used when present. Ties are broken by the smaller residual. ``residual_ms`` = RMS of the
    pair differences around the refined offset."""
    F = [(e["type"], _force_sync_time(e)) for e in force_events if e.get("type") in SYNC_TYPES]
    T = [(e["type"], float(e["t"])) for e in trc_events if e.get("type") in SYNC_TYPES]
    F = [x for x in F if np.isfinite(x[1])]
    T = [x for x in T if np.isfinite(x[1])]
    none = {"offset_s": float("nan"), "n_matched": 0, "residual_ms": float("nan"), "pairs": []}
    cands = [tf - tt for a, tf in F for b, tt in T if a == b]
    if max_abs_offset_s is not None:
        cands = [d for d in cands if abs(d) <= max_abs_offset_s]
    if not cands:
        return none
    best = None
    for d in cands:
        pairs = _pair(F, T, d, tolerance)
        res = float(np.mean([abs(tf - tt - d) for _, tt, tf in pairs])) if pairs else np.inf
        key = (len(pairs), -res)
        if best is None or key > best[0]:
            best = (key, d)
    d = best[1]
    pairs = _pair(F, T, d, tolerance)
    d = float(np.median([tf - tt for _, tt, tf in pairs]))
    pairs = _pair(F, T, d, tolerance) or pairs
    diffs = np.array([tf - tt for _, tt, tf in pairs])
    d = float(np.median(diffs))
    return {"offset_s": d, "n_matched": len(pairs),
            "residual_ms": float(np.sqrt(np.mean((diffs - d) ** 2)) * 1000),
            "pairs": [(k, float(tt), float(tf)) for k, tt, tf in pairs]}


# -------------------------------------------------------------- cross-correlation
def _lowpass(x: np.ndarray, rate_hz: float, cutoff_hz: float) -> np.ndarray:
    """Zero-phase low-pass (2nd-order Butterworth run forward and backward; moving average
    when scipy is missing)."""
    if len(x) < 16:
        return x.copy()
    try:
        from scipy.signal import butter, filtfilt
    except ImportError:  # pragma: no cover
        w = max(1, int(round(rate_hz / cutoff_hz / 2)))
        k = np.ones(w) / w
        return np.convolve(np.convolve(x, k, "same")[::-1], k, "same")[::-1]
    b, a = butter(2, min(cutoff_hz / (rate_hz / 2), 0.99))
    return filtfilt(b, a, x, padlen=min(len(x) - 1, 3 * max(len(a), len(b))))


def xcorr_offset(t_force: np.ndarray, total_kg: np.ndarray, t_trc: np.ndarray,
                 com_up_m: np.ndarray, body_kg: float | None = None,
                 search_s: tuple[float, float] = (-60.0, 60.0), rate_hz: float = 100.0,
                 around_s: float | None = None, cutoff_hz: float = 3.0) -> tuple[float, float]:
    """Offset ``d`` (``t_rel = trc_time + d``) maximizing the normalized cross-correlation of
    the measured total force with the force predicted from the COM height,
    ``m * (g + d²z/dt²)`` (low-pass filtered, resampled to ``rate_hz``). ``around_s`` limits
    the search to ``around_s ± 1 s`` (refinement after an event match). Returns
    ``(offset_s, score)``; score = peak correlation (0..1), NaN offset when not computable.

    Offsets whose overlap is shorter than 2 s or than half of the shorter signal are skipped.
    Only movements with vertical COM accelerations (jumps, squats, stomps) give a clear peak:
    quiet standing gives a low score."""
    nan = (float("nan"), float("nan"))
    tf, f = np.asarray(t_force, np.float64).ravel(), np.asarray(total_kg, np.float64).ravel()
    tt, z = np.asarray(t_trc, np.float64).ravel(), np.asarray(com_up_m, np.float64).ravel()
    okf, okt = np.isfinite(tf) & np.isfinite(f), np.isfinite(tt) & np.isfinite(z)
    tf, f, tt, z = tf[okf], f[okf], tt[okt], z[okt]
    if len(tf) < 10 or len(tt) < 10:
        return nan
    of, ot = np.argsort(tf, kind="stable"), np.argsort(tt, kind="stable")
    tf, f, tt, z = tf[of], f[of], tt[ot], z[ot]
    m = body_kg or body_mass_from_force(tf, f)
    if not m:
        loaded = f[f > 10]
        m = float(np.median(loaded)) if len(loaded) else None
    if not m:
        return nan
    dt = 1.0 / rate_hz
    gf = np.arange(tf[0], tf[-1], dt)
    gt = np.arange(tt[0], tt[-1], dt)
    if len(gf) < 2 * rate_hz or len(gt) < rate_hz:
        return nan
    F = _lowpass(np.interp(gf, tf, f), rate_hz, cutoff_hz)
    Z = _lowpass(np.interp(gt, tt, z), rate_hz, cutoff_hz)
    P = m * (G + np.gradient(np.gradient(Z, dt), dt)) / G
    Nf, Np = len(F), len(P)
    lo, hi = (around_s - 1.0, around_s + 1.0) if around_s is not None else search_s
    base = (gt[0] - gf[0]) * rate_hz
    L_lo, L_hi = int(math.ceil(base + lo * rate_hz)), int(math.floor(base + hi * rate_hz))
    min_overlap = max(int(2 * rate_hz), int(0.5 * min(Nf, Np)))
    cf, cf2 = np.concatenate([[0.0], np.cumsum(F)]), np.concatenate([[0.0], np.cumsum(F * F)])
    cp, cp2 = np.concatenate([[0.0], np.cumsum(P)]), np.concatenate([[0.0], np.cumsum(P * P)])
    lags, scores = [], []
    for L in range(L_lo, L_hi + 1):
        j0, j1 = max(0, -L), min(Np, Nf - L)
        n = j1 - j0
        if n < min_overlap:
            continue
        sp, sp2 = cp[j1] - cp[j0], cp2[j1] - cp2[j0]
        sf, sf2 = cf[L + j1] - cf[L + j0], cf2[L + j1] - cf2[L + j0]
        vp, vf = sp2 - sp * sp / n, sf2 - sf * sf / n
        if vp <= 1e-9 * n or vf <= 1e-9 * n:
            continue
        cov = float(np.dot(P[j0:j1], F[L + j0:L + j1])) - sp * sf / n
        lags.append(L)
        scores.append(cov / math.sqrt(vp * vf))
    if not scores:
        return nan
    sc = np.asarray(scores)
    k = int(np.argmax(sc))
    Lb = float(lags[k])
    if 0 < k < len(sc) - 1 and lags[k + 1] - lags[k - 1] == 2:
        den = sc[k - 1] - 2 * sc[k] + sc[k + 1]
        if den < 0:
            Lb += float(np.clip(0.5 * (sc[k - 1] - sc[k + 1]) / den, -0.5, 0.5))
    return float((Lb - base) / rate_hz), float(sc[k])
