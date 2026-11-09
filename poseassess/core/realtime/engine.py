"""``RealtimeEngine``: live frames -> 2D -> 3D -> smoothing -> angles, on a worker thread.

Sources are objects with ``latest() -> Frame | None`` (``core.capture.camera.CameraStream``):
the engine never opens cameras itself, so the Capture page's streams / settings are reused.

Synchronisation is "latest frame of every camera": the engine waits until every camera has a
frame it has not processed yet (at most ``max_wait_s``), so with free-running webcams the views
differ by up to one frame period (~33 ms at 30 fps); ``skew_ms`` in the statistics shows it.

Latency (``latency_ms``) is measured from the moment the OLDEST frame of the set was returned by
the camera driver to the moment the 3D result is ready. The camera's own exposure + USB latency
(tens of ms, see capture settings ``latency_ms``) and the screen refresh come on top.
"""
from __future__ import annotations

import csv
import logging
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from .angles import ANGLE_ORDER, joint_angles
from .filters import OneEuroFilter
from .skeleton import HALPE26
from .triangulate import reprojection_error, triangulate

log = logging.getLogger(__name__)


@dataclass
class EngineConfig:
    min_score: float = 0.3          # 2D keypoints below this confidence are ignored
    max_reproj_px: float = 40.0     # 3D points with a larger mean reprojection error are dropped
    filter_min_cutoff: float = 1.5  # One-Euro: Hz, lower = smoother when still
    filter_beta: float = 0.5        # One-Euro: higher = less lag when moving fast
    max_wait_s: float = 0.1         # wait at most this for a new frame of every camera
    smooth: bool = True


@dataclass
class LiveResult:
    seq: int
    t_frames: list[float]            # perf_counter capture stamps, per camera
    t_done: float
    sizes: list[tuple[int, int]]     # (w, h) per camera
    kpts2d: np.ndarray               # (C, 26, 2) pixels
    scores: np.ndarray               # (C, 26)
    X_raw: np.ndarray | None = None  # (26, 3) metres, None in 2D-only mode
    X: np.ndarray | None = None      # smoothed
    reproj_px: np.ndarray | None = None
    angles: dict[str, float] = field(default_factory=dict)
    timings_ms: dict[str, float] = field(default_factory=dict)

    @property
    def latency_ms(self) -> float:
        return (self.t_done - min(self.t_frames)) * 1000.0

    @property
    def skew_ms(self) -> float:
        return (max(self.t_frames) - min(self.t_frames)) * 1000.0


class Stats:
    """Rolling statistics over the last ``window_s`` seconds of results."""

    def __init__(self, window_s: float = 3.0):
        self.window_s = window_s
        self._q: deque = deque()
        self._lock = threading.Lock()
        self.total = 0

    def add(self, r: LiveResult) -> None:
        row = (r.t_done, r.latency_ms, r.skew_ms, r.timings_ms.get("pose2d", 0.0),
               r.timings_ms.get("triangulate", 0.0),
               float(np.isfinite(r.X).all(axis=1).sum()) if r.X is not None else float("nan"))
        with self._lock:
            self._q.append(row)
            self.total += 1
            while self._q and r.t_done - self._q[0][0] > self.window_s:
                self._q.popleft()

    def summary(self) -> dict[str, float]:
        with self._lock:
            rows = np.array(self._q, dtype=float) if self._q else np.zeros((0, 6))
        if len(rows) < 2:
            return {"fps": 0.0, "latency_ms": float("nan"), "latency_p95_ms": float("nan"),
                    "skew_ms": float("nan"), "pose2d_ms": float("nan"),
                    "triangulate_ms": float("nan"), "joints_3d": float("nan"), "n": len(rows)}
        span = rows[-1, 0] - rows[0, 0]
        return {
            "fps": (len(rows) - 1) / span if span > 0 else 0.0,
            "latency_ms": float(rows[:, 1].mean()),
            "latency_p95_ms": float(np.percentile(rows[:, 1], 95)),
            "skew_ms": float(rows[:, 2].mean()),
            "pose2d_ms": float(rows[:, 3].mean()),
            "triangulate_ms": float(rows[:, 4].mean()),
            "joints_3d": float(np.nanmean(rows[:, 5])) if np.isfinite(rows[:, 5]).any() else float("nan"),
            "n": len(rows),
        }


class RealtimeEngine:
    """``sources``: ``{cam: stream}`` in calibration order; ``calib``: matching
    ``LiveCamera`` list (same length) or None for 2D only; ``estimators``: one
    ``PoseEstimator`` (callable image -> (kpts, scores)) per camera."""

    def __init__(self, sources: dict, estimators: list[Callable], calib: list | None = None,
                 config: EngineConfig | None = None, csv_path: str | Path | None = None,
                 on_result: Callable[[LiveResult], None] | None = None):
        self.cams = list(sources)
        self.sources = [sources[c] for c in self.cams]
        if len(estimators) != len(self.sources):
            raise ValueError("one estimator per camera is needed")
        if calib is not None and len(calib) != len(self.sources):
            raise ValueError(f"{len(self.sources)} cameras but {len(calib)} calibrated ones")
        self.estimators = estimators
        self.calib = calib
        self.cfg = config or EngineConfig()
        self.csv_path = Path(csv_path) if csv_path else None
        self.on_result = on_result
        self.stats = Stats()
        self.error: str | None = None
        self.warnings: list[str] = []
        self._filter = OneEuroFilter(self.cfg.filter_min_cutoff, self.cfg.filter_beta)
        self._latest: LiveResult | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._pool: ThreadPoolExecutor | None = None
        self._checked_sizes: set = set()

    # ------------------------------------------------------------------ control
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self.error = None
        self._filter.reset()
        self._pool = ThreadPoolExecutor(max_workers=len(self.sources), thread_name_prefix="pose2d")
        self._thread = threading.Thread(target=self._run, name="realtime-engine", daemon=True)
        self._thread.start()

    def stop(self, timeout_s: float = 3.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout_s)
        self._thread = None
        if self._pool is not None:
            self._pool.shutdown(wait=False, cancel_futures=True)
            self._pool = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def latest(self) -> LiveResult | None:
        with self._lock:
            return self._latest

    # ------------------------------------------------------------------ worker
    def _run(self) -> None:
        f = None
        try:
            if self.csv_path is not None:
                self.csv_path.parent.mkdir(parents=True, exist_ok=True)
                f = open(self.csv_path, "w", newline="")
                w = csv.writer(f)
                w.writerow(self._csv_header())
            else:
                w = None
            self._loop(w)
        except Exception as e:  # noqa: BLE001 - reported to the GUI / CLI
            log.exception("real-time engine stopped")
            self.error = f"{type(e).__name__}: {e}"
        finally:
            if f is not None:
                f.close()

    def _wait_frames(self, last: list[int]):
        """Latest frame of every camera, once each has a new one (or ``max_wait_s`` passed
        and at least one is new). None when stopping."""
        t_first_new = None
        while not self._stop.is_set():
            frames = [s.latest() for s in self.sources]
            if all(fr is not None for fr in frames):
                new = [fr.index != li for fr, li in zip(frames, last)]
                if all(new):
                    return frames
                if any(new):
                    now = time.perf_counter()
                    t_first_new = t_first_new or now
                    if now - t_first_new >= self.cfg.max_wait_s:
                        return frames
            time.sleep(0.001)
        return None

    def _loop(self, w) -> None:
        last = [-1] * len(self.sources)
        seq = 0
        while not self._stop.is_set():
            t0 = time.perf_counter()
            frames = self._wait_frames(last)
            if frames is None:
                return
            last = [fr.index for fr in frames]
            t1 = time.perf_counter()
            futs = [self._pool.submit(est, fr.image) for est, fr in zip(self.estimators, frames)]
            out = [fu.result() for fu in futs]
            t2 = time.perf_counter()
            kpts = np.stack([o[0] for o in out])
            scores = np.stack([o[1] for o in out])
            sizes = [(fr.image.shape[1], fr.image.shape[0]) for fr in frames]
            r = LiveResult(seq, [fr.t for fr in frames], 0.0, sizes, kpts, scores)
            if self.calib is not None:
                self._reconstruct(r, t_ref=float(np.mean(r.t_frames)))
            t3 = time.perf_counter()
            r.t_done = t3
            r.timings_ms = {"wait": (t1 - t0) * 1e3, "pose2d": (t2 - t1) * 1e3,
                            "triangulate": (t3 - t2) * 1e3, "total": (t3 - t1) * 1e3}
            with self._lock:
                self._latest = r
            self.stats.add(r)
            if w is not None:
                w.writerow(self._csv_row(r))
            if self.on_result is not None:
                self.on_result(r)
            seq += 1

    def _reconstruct(self, r: LiveResult, t_ref: float) -> None:
        for cam, size in zip(self.calib, r.sizes):
            if (cam.name, size) not in self._checked_sizes:
                self._checked_sizes.add((cam.name, size))
                msg = cam.check_size(size)
                if msg:
                    log.warning(msg)
                    self.warnings.append(msg)
        uv = np.stack([cam.normalize(k, size) for cam, k, size in
                       zip(self.calib, r.kpts2d, r.sizes)])            # (C, 26, 2)
        wts = np.where(r.scores >= self.cfg.min_score, r.scores, 0.0)
        P = np.stack([cam.P_norm for cam in self.calib])
        X = triangulate(P, uv, wts)
        err = reprojection_error(P, uv, wts, X)
        fx = np.array([cam.K[0, 0] for cam in self.calib]).mean()
        err_px = err * fx
        X[~(err_px <= self.cfg.max_reproj_px)] = np.nan
        r.X_raw = X
        r.reproj_px = err_px
        r.X = self._filter(X, t_ref) if self.cfg.smooth else X
        r.angles = joint_angles(r.X)

    # --------------------------------------------------------------------- csv
    def _csv_header(self) -> list[str]:
        h = ["seq", "t_done", "latency_ms", "skew_ms", "pose2d_ms", "triangulate_ms"]
        h += [f"t_{c}" for c in self.cams]
        if self.calib is not None:
            h += [f"angle_{a}" for a in ANGLE_ORDER]
            h += [f"{n}_{ax}" for n in HALPE26 for ax in "xyz"]
        return h

    def _csv_row(self, r: LiveResult) -> list:
        row = [r.seq, f"{r.t_done:.6f}", f"{r.latency_ms:.2f}", f"{r.skew_ms:.2f}",
               f"{r.timings_ms['pose2d']:.2f}", f"{r.timings_ms['triangulate']:.2f}"]
        row += [f"{t:.6f}" for t in r.t_frames]
        if self.calib is not None:
            row += [f"{r.angles.get(a, float('nan')):.2f}" for a in ANGLE_ORDER]
            row += [f"{v:.5f}" for v in r.X.reshape(-1)]
        return row
