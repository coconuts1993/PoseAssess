"""Automatic checkerboard corner detection for the extrinsic step.

Replaces hand-clicking every corner with a detector tuned for the hard case in
this project: a small board, seen at an angle by a 1080p camera ~3 m away
(squares of only ~10-20 px).

Pipeline for one image (`detect_checkerboard`):
  1. `findChessboardCornersSB` (sector-based, far more robust than the classic
     detector on blur, low resolution and steep angles) on the full image.
  2. If that fails, the same on a 2x up-sampled image (small boards).
  3. If that fails, the classic `findChessboardCorners` + `cornerSubPix`.
  4. If the caller gives a region of interest (e.g. where the board was found in
     another frame, or a rough box the operator drew), steps 1-3 run again on that
     crop up-sampled to ~1200 px wide.

For a static board, `detect_in_frames` / `detect_in_video` run the detector on
many frames, bring every detection into the same corner order, and take the
per-corner median - cheaper noise reduction than any model, and a board missed
in most frames is still found in a few.

When everything fails, `corners_from_outer4` turns 4 clicks on the outermost
inner corners into the whole grid (homography prediction + sub-pixel snap), so
the operator never clicks 40 points again.

Corner order: row-major over `corners_nb = [rows, cols]` (cols run fastest),
the same as `intrinsic._object_points`. Which physical corner is index 0 is NOT
decided here - see `origin.py`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np

# The 4 in-plane re-readings of an R x C grid that keep its shape.
GRID_SYMMETRIES = ("id", "flipR", "flipC", "rot180")


@dataclass
class Detection:
    """Result of a detection attempt. `points` is (rows*cols, 2) or None."""
    points: Optional[np.ndarray]
    method: str = "none"
    frames_used: int = 0
    frames_tried: int = 0
    spread_px: float = 0.0          # per-corner scatter across fused frames
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.points is not None


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _to_gray(img: np.ndarray) -> np.ndarray:
    import cv2
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def reorder_grid(points: np.ndarray, corners_nb: Sequence[int], sym: str) -> np.ndarray:
    """Re-read an R x C row-major grid with one of `GRID_SYMMETRIES`."""
    r, c = int(corners_nb[0]), int(corners_nb[1])
    g = np.asarray(points, np.float64).reshape(r, c, 2)
    if sym == "flipR":
        g = g[::-1]
    elif sym == "flipC":
        g = g[:, ::-1]
    elif sym == "rot180":
        g = g[::-1, ::-1]
    elif sym != "id":
        raise ValueError(f"unknown grid symmetry {sym!r}")
    return g.reshape(-1, 2).copy()


def _min_spacing(points: np.ndarray, corners_nb: Sequence[int]) -> float:
    r, c = int(corners_nb[0]), int(corners_nb[1])
    g = np.asarray(points, np.float64).reshape(r, c, 2)
    d = []
    if c > 1:
        d.append(np.linalg.norm(np.diff(g, axis=1), axis=2).ravel())
    if r > 1:
        d.append(np.linalg.norm(np.diff(g, axis=0), axis=2).ravel())
    return float(np.concatenate(d).min()) if d else 0.0


def _subpix(gray: np.ndarray, pts: np.ndarray, corners_nb: Sequence[int]) -> np.ndarray:
    """cornerSubPix with a window sized to the board's square size in pixels, so
    it never jumps to a neighbouring corner on a small board."""
    import cv2
    sp = _min_spacing(pts, corners_nb)
    win = int(max(2, min(11, round(sp * 0.35))))
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)
    out = cv2.cornerSubPix(gray, np.asarray(pts, np.float32).reshape(-1, 1, 2),
                           (win, win), (-1, -1), crit)
    return out.reshape(-1, 2).astype(np.float64)


def _try_once(gray: np.ndarray, corners_nb: Sequence[int]) -> tuple[Optional[np.ndarray], str]:
    import cv2
    rows, cols = int(corners_nb[0]), int(corners_nb[1])
    pattern = (cols, rows)
    sb = getattr(cv2, "findChessboardCornersSB", None)
    if sb is not None:
        flags = (cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
                 | cv2.CALIB_CB_NORMALIZE_IMAGE)
        ok, corners = sb(gray, pattern, flags=flags)
        if ok and corners is not None and len(corners) == rows * cols:
            return corners.reshape(-1, 2).astype(np.float64), "SB"
    # The classic detector can take ~25 s to give up on a 1080p frame with no
    # board; FAST_CHECK rejects those in milliseconds. Small crops can afford
    # the full search.
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    if gray.shape[0] * gray.shape[1] > 900_000:
        flags |= cv2.CALIB_CB_FAST_CHECK
    ok, corners = cv2.findChessboardCorners(gray, pattern, flags=flags)
    if ok and corners is not None and len(corners) == rows * cols:
        pts = corners.reshape(-1, 2).astype(np.float64)
        return _subpix(gray, pts, corners_nb), "classic"
    return None, ""


def _try_scaled(gray: np.ndarray, corners_nb: Sequence[int], scale: float):
    import cv2
    if scale == 1.0:
        return _try_once(gray, corners_nb)
    big = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    pts, how = _try_once(big, corners_nb)
    if pts is None:
        return None, ""
    # map back: pixel centres scale as (p + 0.5) / s - 0.5
    pts = (pts + 0.5) / scale - 0.5
    return _subpix(gray, pts, corners_nb), f"{how}@x{scale:g}"


def _sharpen(img: np.ndarray, sigma: float) -> np.ndarray:
    import cv2
    return cv2.addWeighted(img, 1.8, cv2.GaussianBlur(img, (0, 0), sigma), -0.8, 0)


def _try_crop(gray: np.ndarray, corners_nb: Sequence[int], roi: Sequence[float]):
    """Detect inside one region: crop with margin, up-sample, optionally sharpen.
    Returns (points in full-image pixels, method) or (None, "")."""
    import cv2
    h, w = gray.shape[:2]
    x0, y0, x1, y1 = [float(v) for v in roi]
    mx, my = 0.2 * (x1 - x0) + 8, 0.2 * (y1 - y0) + 8
    xa, ya = int(max(0, x0 - mx)), int(max(0, y0 - my))
    xb, yb = int(min(w, x1 + mx)), int(min(h, y1 + my))
    if xb - xa < 16 or yb - ya < 16:
        return None, ""
    crop = gray[ya:yb, xa:xb]
    base = float(np.clip(600.0 / max(xb - xa, yb - ya), 1.0, 4.0))
    for scale in sorted({1.0, 2.0, round(base), 4.0}):
        big = crop if scale == 1.0 else cv2.resize(crop, None, fx=scale, fy=scale,
                                                    interpolation=cv2.INTER_CUBIC)
        for pre in ("plain", "sharp"):
            im = big if pre == "plain" else _sharpen(big, 2.0 * scale)
            pts, how = _try_once(im, corners_nb)
            if pts is None:
                continue
            pts = (pts + 0.5) / scale - 0.5 + np.array([xa, ya], np.float64)
            return _subpix(gray, pts, corners_nb), f"roi:{how}@x{scale:g}{'+sharp' if pre == 'sharp' else ''}"
    return None, ""


def locate_board_candidates(gray: np.ndarray, max_regions: int = 3,
                            min_saddles: int = 12) -> list[tuple[float, float, float, float]]:
    """Rough regions that look like a checkerboard: places where many saddle
    points (the X-junctions between squares) cluster. Cheap (~0.1 s at 1080p)
    and only used to pick crops for the real detector."""
    import cv2
    g = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 1.0)
    gxx = cv2.Sobel(g, cv2.CV_32F, 2, 0, ksize=3)
    gyy = cv2.Sobel(g, cv2.CV_32F, 0, 2, ksize=3)
    gxy = cv2.Sobel(g, cv2.CV_32F, 1, 1, ksize=3)
    sad = np.maximum(-(gxx * gyy - gxy * gxy), 0)
    sad /= float(sad.max()) + 1e-9
    peaks = (sad == cv2.dilate(sad, np.ones((5, 5), np.uint8))) & (sad > 0.05)
    ys, xs = np.nonzero(peaks)
    if len(xs) < min_saddles:
        return []
    dens = cv2.boxFilter(peaks.astype(np.float32), -1, (61, 61), normalize=False)
    out = []
    for _ in range(max_regions):
        y, x = np.unravel_index(int(np.argmax(dens)), dens.shape)
        if dens[y, x] < min_saddles:
            break
        near = np.hypot(xs - x, ys - y) < 160
        if near.sum() >= min_saddles:
            out.append((float(xs[near].min()), float(ys[near].min()),
                        float(xs[near].max()), float(ys[near].max())))
        cv2.circle(dens, (int(x), int(y)), 200, 0, -1)
    return out


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #
def detect_checkerboard(
    img: np.ndarray,
    corners_nb: Sequence[int],
    roi: Optional[Sequence[float]] = None,
    allow_upscale: bool = True,
    search_rois: bool = True,
) -> Detection:
    """Find the full inner-corner grid in one image (BGR or gray).

    `roi` = (x0, y0, x1, y1) in pixels narrows the search to a crop that is
    up-sampled (and sharpened) before detection - the main lever for very small
    boards. With `search_rois`, candidate board regions are also located
    automatically from the density of checkerboard saddle points.
    """
    gray = _to_gray(img)
    h, w = gray.shape[:2]
    notes: list[str] = []

    # A known region first (cheap, and the most likely to succeed), then the
    # full image, then automatically located regions, then the whole image
    # up-sampled (slowest).
    if roi is not None:
        cpts, chow = _try_crop(gray, corners_nb, roi)
        if cpts is not None:
            return Detection(cpts, chow, 1, 1, 0.0, notes)
    pts, how = _try_once(gray, corners_nb)
    if pts is not None:
        return Detection(pts, how, 1, 1, 0.0, notes)
    rois = [r for r in locate_board_candidates(gray)] if search_rois else []
    for rr in rois:
        cpts, chow = _try_crop(gray, corners_nb, rr)
        if cpts is not None:
            return Detection(cpts, chow, 1, 1, 0.0, notes)
    if allow_upscale and max(h, w) <= 2400:
        pts, how = _try_scaled(gray, corners_nb, 2.0)
        if pts is not None:
            return Detection(pts, how, 1, 1, 0.0, notes)
    notes.append(f"not found (searched full image and {len(rois)} candidate region(s))")
    return Detection(None, "none", 0, 1, 0.0, notes)


def align_to_reference(points: np.ndarray, ref: np.ndarray,
                       corners_nb: Sequence[int]) -> tuple[np.ndarray, float]:
    """Re-read `points` with the grid symmetry that best matches `ref`.
    Returns (reordered, mean distance px). Detectors may number a symmetric
    board from a different end in different frames."""
    best, best_d = None, np.inf
    for sym in GRID_SYMMETRIES:
        cand = reorder_grid(points, corners_nb, sym)
        d = float(np.linalg.norm(cand - ref, axis=1).mean())
        if d < best_d:
            best, best_d = cand, d
    return best, best_d


def fuse_detections(dets: Sequence[np.ndarray], corners_nb: Sequence[int],
                    outlier_px: float = 2.0) -> tuple[Optional[np.ndarray], int, float]:
    """Median-fuse several detections of a STATIC board from a fixed camera.
    Returns (points, frames kept, spread px)."""
    dets = [np.asarray(d, np.float64) for d in dets if d is not None]
    if not dets:
        return None, 0, 0.0
    ref = dets[0]
    aligned = [align_to_reference(d, ref, corners_nb)[0] for d in dets]
    stack = np.stack(aligned)                       # (F, N, 2)
    med = np.median(stack, axis=0)
    # drop frames that disagree with the median (board/camera moved, bad detect)
    dist = np.linalg.norm(stack - med, axis=2).mean(axis=1)
    keep = dist <= max(outlier_px, 3.0 * float(np.median(dist)) + 1e-9)
    if keep.sum() >= 1:
        stack = stack[keep]
        med = np.median(stack, axis=0)
    spread = float(np.median(np.linalg.norm(stack - med, axis=2))) if len(stack) > 1 else 0.0
    return med, int(len(stack)), spread


def detect_in_frames(frames: Iterable[np.ndarray], corners_nb: Sequence[int],
                     roi: Optional[Sequence[float]] = None,
                     max_frames: int = 60) -> Detection:
    """Detect in each frame, then median-fuse. Once the board is found, later
    frames reuse its bounding box as the ROI (helps the frames where the
    full-image search alone misses it)."""
    found: list[np.ndarray] = []
    methods: list[str] = []
    tried = 0
    gray_last = None
    for img in frames:
        if tried >= max_frames:
            break
        tried += 1
        gray = _to_gray(img)
        gray_last = gray
        hint = roi
        if hint is None and found:
            p = found[-1]
            hint = (p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max())
        det = detect_checkerboard(gray, corners_nb, roi=hint)
        if det.ok:
            found.append(det.points)
            methods.append(det.method)
    if not found:
        return Detection(None, "none", 0, tried, 0.0, ["board not found in any frame"])
    pts, kept, spread = fuse_detections(found, corners_nb)
    if gray_last is not None and pts is not None:
        pts = _subpix(gray_last, pts, corners_nb) if kept == 1 else pts
    method = max(set(methods), key=methods.count)
    return Detection(pts, method, kept, tried, spread)


def iter_video_frames(video_path: Path, n_samples: int = 40):
    """Yield up to `n_samples` frames spread evenly over a video."""
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        if total <= 0:
            k = 0
            while k < n_samples:
                ok, fr = cap.read()
                if not ok:
                    break
                yield fr
                k += 1
            return
        for idx in np.linspace(0, max(0, total - 1), num=min(n_samples, total)).astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, fr = cap.read()
            if ok:
                yield fr
    finally:
        cap.release()


def detect_in_video(video_path: Path, corners_nb: Sequence[int],
                    n_samples: int = 40, roi: Optional[Sequence[float]] = None) -> Detection:
    return detect_in_frames(iter_video_frames(video_path, n_samples), corners_nb,
                            roi=roi, max_frames=n_samples)


def detect_in_images(paths: Sequence[Path], corners_nb: Sequence[int],
                     roi: Optional[Sequence[float]] = None) -> Detection:
    import cv2

    def gen():
        for p in paths:
            im = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if im is not None:
                yield im
    return detect_in_frames(gen(), corners_nb, roi=roi, max_frames=len(paths))


def corners_from_outer4(img: np.ndarray, corners_nb: Sequence[int],
                        outer4: Sequence[Sequence[float]]) -> np.ndarray:
    """Predict the full grid from 4 clicks on the outermost INNER corners.

    The clicks may come in any order. Their convex-hull order is matched to the
    grid's 4 extreme corners under every rotation / reflection; the reading whose
    predicted corners move least under sub-pixel refinement wins (a wrong
    rows/cols assignment puts predictions between real corners). Corner 0 of the
    result is arbitrary - run `origin.canonicalize` afterwards.
    """
    import cv2
    gray = _to_gray(img)
    r, c = int(corners_nb[0]), int(corners_nb[1])
    q = np.asarray(outer4, np.float64).reshape(4, 2)
    # convex-hull (angular) order around the centroid
    ctr = q.mean(0)
    q = q[np.argsort(np.arctan2(q[:, 1] - ctr[1], q[:, 0] - ctr[0]))]
    grid_corners = np.array([[0, 0], [c - 1, 0], [c - 1, r - 1], [0, r - 1]], np.float64)
    gx, gy = np.meshgrid(np.arange(c, dtype=np.float64), np.arange(r, dtype=np.float64))
    grid = np.stack([gx.ravel(), gy.ravel()], 1)       # row-major, cols fastest
    best, best_score = None, np.inf
    for flip in (False, True):
        qq = q[::-1] if flip else q
        for k in range(4):
            src = grid_corners
            dst = np.roll(qq, k, axis=0)
            H, _ = cv2.findHomography(src, dst)
            if H is None:
                continue
            pred = cv2.perspectiveTransform(grid.reshape(-1, 1, 2), H).reshape(-1, 2)
            if (pred < 0).any() or (pred[:, 0] >= gray.shape[1]).any() or (pred[:, 1] >= gray.shape[0]).any():
                continue
            ref = _subpix(gray, pred, corners_nb)
            # A correct reading snaps every prediction onto its own corner and
            # the snapped points still form a clean projective grid. A wrong
            # rows/cols assignment snaps several predictions onto the same corner
            # or off the grid, so its homography refit residual is large.
            H2, _ = cv2.findHomography(grid, ref)
            if H2 is None:
                continue
            refit = cv2.perspectiveTransform(grid.reshape(-1, 1, 2), H2).reshape(-1, 2)
            resid = float(np.linalg.norm(refit - ref, axis=1).mean())
            sp = _min_spacing(ref, corners_nb)
            collapsed = sp < 0.3 * _min_spacing(pred, corners_nb)
            score = resid + (1e6 if collapsed else 0.0)
            if score < best_score:
                best, best_score = ref, score
    if best is None:
        raise ValueError("could not build a grid from the 4 clicked corners")
    return best
