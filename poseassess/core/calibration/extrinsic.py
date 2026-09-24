"""Extrinsic camera calibration from manually-clicked checkerboard corners.

For the scene/extrinsic frame, automatic checkerboard detection is unreliable
(board seen at a steep angle, partial occlusion), so the user clicks the corners
by hand in the GUI. Given those 2D image points, the known 3D board geometry,
and the camera intrinsics (from the intrinsic step), `cv2.solvePnP` recovers the
camera pose (R, T) that places every camera in one shared world frame.

Clicked points are stored per camera as JSON (see `load_clicked_points`) so the
calibration is reproducible and the manual work is never lost.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from .intrinsic import _object_points


@dataclass
class ExtrinsicResult:
    camera_index: int
    rvec: np.ndarray   # 3x1 Rodrigues rotation
    R: np.ndarray      # 3x3 rotation matrix
    T: np.ndarray      # 3x1 translation
    reproj_error: float


def load_clicked_points(points_file: Path) -> np.ndarray:
    """Load hand-clicked corner pixels from JSON.

    Accepts either our own `{"points": [[x,y], ...]}` schema or a LabelMe
    `{"shapes": [{"points": [[x,y],...]}]}` export (backward compatible with the
    old add-on's LabelMe workflow).
    """
    with open(points_file) as f:
        data = json.load(f)
    if "points" in data:
        pts = data["points"]
    elif "shapes" in data and data["shapes"]:
        pts = data["shapes"][0]["points"]
    elif "keypoints2d" in data:
        pts = [[p[0], p[1]] for p in data["keypoints2d"]]
    else:
        raise ValueError(f"unrecognized clicked-points file: {points_file}")
    return np.asarray(pts, dtype=np.float64)


def save_clicked_points(points_file: Path, points: Sequence[Sequence[float]]) -> None:
    points_file.parent.mkdir(parents=True, exist_ok=True)
    with open(points_file, "w") as f:
        json.dump({"points": [[float(x), float(y)] for x, y in points]}, f, indent=2)


def calibrate_extrinsic_camera(
    image_points: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    corners_nb: Sequence[int],
    square_size: float,
    camera_index: int = 1,
) -> ExtrinsicResult:
    """Recover one camera's pose from clicked corners via solvePnP.

    `image_points` must be ordered the same way as the board object points
    (row-major over the inner-corner grid), i.e. the user clicks corners in the
    documented order. Length must equal rows*cols.
    """
    import cv2

    objp = _object_points(corners_nb, square_size)  # (N,3)
    image_points = np.asarray(image_points, dtype=np.float64).reshape(-1, 2)
    if image_points.shape[0] != objp.shape[0]:
        raise ValueError(
            f"cam{camera_index:02d}: got {image_points.shape[0]} clicked points "
            f"but board has {objp.shape[0]} corners ({corners_nb[0]}x{corners_nb[1]})"
        )

    K = np.asarray(K, dtype=np.float64)
    dist = np.asarray(dist, dtype=np.float64).reshape(1, -1)

    ok, rvec, tvec = cv2.solvePnP(objp, image_points, K, dist,
                                  flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        raise RuntimeError(f"cam{camera_index:02d}: solvePnP failed")

    # Reprojection error, for quality feedback in the UI.
    proj, _ = cv2.projectPoints(objp, rvec, tvec, K, dist)
    err = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - image_points) ** 2, axis=1))))

    R, _ = cv2.Rodrigues(rvec)
    return ExtrinsicResult(
        camera_index=camera_index,
        rvec=np.asarray(rvec, dtype=np.float64).reshape(3, 1),
        R=np.asarray(R, dtype=np.float64),
        T=np.asarray(tvec, dtype=np.float64).reshape(3, 1),
        reproj_error=err,
    )


# --------------------------------------------------------------------------- #
# Automatic corner-ordering alignment across cameras.
#
# Whether corners come from `findChessboardCorners` or hand-clicking, each board
# is numbered starting from whichever physical corner it happens to face in that
# view, and — for a rectangular board seen rotated ~90 degrees — the detector can
# even return the grid with its rows and columns SWAPPED (an 8x5 traversal of a
# board configured as 5x8). So camera A's corner ordering and camera B's need not
# agree at all: A's #0 and B's #0 can be different physical corners, and their
# grids can be transposed. solvePnP then places the cameras in inconsistent world
# frames and multi-view triangulation explodes (the 82315px / 48-million-px
# failure the user hit).
#
# Fixing this by hand (six Reverse buttons + hoping the dims match) is hopeless,
# so we canonicalise automatically in two stages:
#   1. Per camera, try every grid re-ordering — both dimension readings (RxC and
#      CxR) times the 4 in-plane symmetries (identity, flip-rows, flip-cols,
#      180deg) — and keep the ones whose OWN solvePnP fits a flat grid well
#      (< own_thresh px). This resolves a transposed/mis-dimensioned detection.
#      A symmetric checkerboard can't distinguish the 4 flips from a single view
#      (they fit equally well), so several survive: that residual gauge freedom
#      is a global choice, resolved in stage 2.
#   2. Brute-force the surviving combinations across cameras and keep the one
#      whose triangulated corners reproject with the lowest error — i.e. the one
#      where every camera agrees on the same physical origin and axes.
# The winner reprojects at sub-pixel error with a flat board at the true square
# size; a camera that can't be made consistent is flagged for re-marking.
# --------------------------------------------------------------------------- #
def _solve_pose(objp: np.ndarray, image_points: np.ndarray,
                K: np.ndarray, dist: np.ndarray):
    """solvePnP -> (rvec, tvec, own_reproj_px) or None if it fails."""
    import cv2
    ip = np.asarray(image_points, np.float64).reshape(-1, 2)
    K = np.asarray(K, np.float64)
    dist = np.asarray(dist, np.float64).reshape(1, -1)
    ok, rvec, tvec = cv2.solvePnP(objp, ip, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None
    rvec = np.asarray(rvec, np.float64).reshape(3, 1)
    tvec = np.asarray(tvec, np.float64).reshape(3, 1)
    proj, _ = cv2.projectPoints(objp, rvec, tvec, K, dist)
    err = float(np.linalg.norm(proj.reshape(-1, 2) - ip, axis=1).mean())
    return rvec, tvec, err


def _object_points_x5(corners_nb: Sequence[int], square_size: float,
                      invert_z: bool = False) -> np.ndarray:
    """Board 3D points with a FIXED world convention: origin at corner index 0,
    X axis along the corners_nb[0]-direction (the '5-corner' side), Y along
    corners_nb[1] (the '8-corner' side), Z up. Row-major in (X, Y).

    `invert_z` negates the X object coordinate (like the old add-on's
    `invert_extri_cal`), which flips the world Z direction — use it when the
    calibrated world comes out Z-down and the subject reconstructs upside-down."""
    r, c = int(corners_nb[0]), int(corners_nb[1])
    sx = -1.0 if invert_z else 1.0
    objp = np.zeros((r * c, 3), np.float64)
    idx = 0
    for x in range(r):        # X = corners_nb[0] direction (5)
        for y in range(c):    # Y = corners_nb[1] direction (8)
            objp[idx] = (sx * x * square_size, y * square_size, 0.0)
            idx += 1
    return objp


def _anchored_reorderings(pts: np.ndarray, corners_nb: Sequence[int],
                          square_size: float, invert_z: bool = False):
    """Yield (tag, reordered_pts, objp) orderings that KEEP the marked corner 0
    (index 0) at the world origin, with X = corners_nb[0]-dir, Y = corners_nb[1].

    Only two are possible without moving corner 0:
      A: the clicked points are already row-major (r x c) from corner 0
      B: the board was seen rotated ~90 deg, so they are row-major (c x r)
         — a transpose brings them back, and the transpose fixes index 0.
    (In-plane flips are deliberately NOT offered: they would move the origin off
    the user's marked corner 0.)"""
    r, c = int(corners_nb[0]), int(corners_nb[1])
    pts = np.asarray(pts, np.float64).reshape(-1, 2)
    objp = _object_points_x5(corners_nb, square_size, invert_z=invert_z)
    out = []
    if pts.shape[0] == r * c:
        out.append(("A", pts.copy(), objp))
        g = pts.reshape(c, r, 2)                       # rotated-board reading
        out.append(("B", g.transpose(1, 0, 2).reshape(-1, 2).copy(), objp))
    return out


def _grid_reorderings(pts: np.ndarray, corners_nb: Sequence[int],
                      square_size: float):
    """Yield (tag, reordered_pts (N,2), objp (N,3)) for every plausible reading
    of the clicked corners: both dimension orders x the 4 in-plane symmetries."""
    r, c = int(corners_nb[0]), int(corners_nb[1])
    pts = np.asarray(pts, np.float64).reshape(-1, 2)
    out = []
    for (R, C) in ((r, c), (c, r)):
        if R * C != pts.shape[0]:
            continue
        objp = _object_points((R, C), square_size)
        g = pts.reshape(R, C, 2)
        for tag, arr in (("id", g), ("flipR", g[::-1]),
                         ("flipC", g[:, ::-1]), ("rot180", g[::-1, ::-1])):
            out.append((f"{R}x{C}_{tag}", arr.reshape(-1, 2).copy(), objp))
    return out


def _multiview_reproj(cams: list[dict], corners_list: list[np.ndarray]):
    """Triangulate the shared corners and return (mean_px, per_cam_px, pts3d)."""
    import cv2
    Ps, und = [], []
    for c, pts in zip(cams, corners_list):
        R, _ = cv2.Rodrigues(c["rvec"])
        Ps.append(c["K"] @ np.hstack([R, c["tvec"]]))
        p = np.asarray(pts, np.float64).reshape(-1, 1, 2)
        und.append(cv2.undistortPoints(p, c["K"], c["dist"], P=c["K"]).reshape(-1, 2))
    N = und[0].shape[0]
    pts3d = np.zeros((N, 3))
    for k in range(N):
        A = []
        for uc, P in zip(und, Ps):
            u, v = uc[k]
            A.append(u * P[2] - P[0])
            A.append(v * P[2] - P[1])
        _, _, Vt = np.linalg.svd(np.asarray(A))
        X = Vt[-1]
        pts3d[k] = X[:3] / X[3]
    per_cam = []
    for c, pts in zip(cams, corners_list):
        rp, _ = cv2.projectPoints(pts3d, c["rvec"], c["tvec"], c["K"], c["dist"])
        e = np.linalg.norm(rp.reshape(-1, 2) - np.asarray(pts, np.float64), axis=1)
        per_cam.append(float(e.mean()))
    return float(np.mean(per_cam)), per_cam, pts3d


def align_extrinsics(
    intr_by_cam: dict,
    corners_by_cam: dict,
    corners_nb: Sequence[int],
    square_size: float,
    own_thresh_px: float = 2.0,
    reproj_warn_px: float = 3.0,
    max_combos: int = 200_000,
    anchor_origin: bool = True,
    invert_z: bool = False,
    progress=None,
) -> tuple[list[ExtrinsicResult], dict[int, np.ndarray]]:
    """Solve every camera's pose with a globally-consistent board corner order.

    `intr_by_cam[i]` must expose `.K` and `.dist`; `corners_by_cam[i]` is that
    camera's ordered (N,2) clicked corners. `square_size` is the board square in
    the desired WORLD units (pass metres so the calibration is metric for
    Pose2Sim/OpenSim).

    With `anchor_origin=True` (default) the world frame is FIXED to the user's
    convention: origin at each camera's marked corner 0, X along the
    corners_nb[0]-direction (5-corner side), Y along corners_nb[1] (8 side). Only
    the transpose ambiguity (board seen rotated 90 deg) is resolved automatically;
    corner 0 is never moved. This requires the user to have marked corner 0 as the
    SAME physical corner in every camera (Reverse/Rotate tools help) — if not, the
    cameras won't agree and the worst one is flagged. With `anchor_origin=False`
    the origin/axes are chosen freely to minimise error (may not match the mark).

    Returns `(results, aligned_by_cam)`: one ExtrinsicResult per camera in index
    order, and the corner arrays re-ordered into the shared canonical order.
    """
    import itertools

    def log(msg: str) -> None:
        if progress:
            progress(msg)

    idxs = sorted(corners_by_cam)
    if anchor_origin:
        def reorder(pts, cn, sq):
            return _anchored_reorderings(pts, cn, sq, invert_z=invert_z)
    else:
        reorder = _grid_reorderings

    # ---- stage 1: per-camera structurally-valid candidate orderings ---------
    cand: dict[int, list] = {}
    for i in idxs:
        K, dist = intr_by_cam[i].K, intr_by_cam[i].dist
        opts = []
        for tag, pts, objp in reorder(corners_by_cam[i], corners_nb, square_size):
            sol = _solve_pose(objp, pts, K, dist)
            if sol is None:
                continue
            rvec, tvec, err = sol
            opts.append({"tag": tag, "pts": pts, "objp": objp,
                         "rvec": rvec, "tvec": tvec, "err": err})
        if not opts:
            raise RuntimeError(f"cam{i:02d}: solvePnP failed for every corner ordering")
        opts.sort(key=lambda o: o["err"])
        keep = [o for o in opts if o["err"] < own_thresh_px] or [opts[0]]
        cand[i] = keep
        log(f"[extrinsic] cam{i:02d}: {len(keep)} candidate ordering(s), "
            f"best own-fit {keep[0]['err']:.2f}px")

    # keep the combinatorics bounded (the 4 flips of the correct structure tie,
    # so a small cap per camera loses nothing)
    total = 1
    for i in idxs:
        total *= len(cand[i])
    while total > max_combos:
        big = max(idxs, key=lambda i: len(cand[i]))
        cand[big] = cand[big][:max(1, len(cand[big]) - 1)]
        total = 1
        for i in idxs:
            total *= len(cand[i])

    # ---- stage 2: global consistency over the surviving combinations --------
    best = None  # (mean_px, per_cam, chosen[list of opt dicts])
    for combo in itertools.product(*[cand[i] for i in idxs]):
        cams = [{"K": np.asarray(intr_by_cam[i].K, float),
                 "dist": np.asarray(intr_by_cam[i].dist, float).reshape(1, -1),
                 "rvec": o["rvec"], "tvec": o["tvec"]}
                for i, o in zip(idxs, combo)]
        clist = [o["pts"] for o in combo]
        mean_px, per_cam, _ = _multiview_reproj(cams, clist)
        if best is None or mean_px < best[0]:
            best = (mean_px, per_cam, combo)

    mean_px, per_cam, chosen = best
    log(f"[extrinsic] origin auto-align: global mean {mean_px:.2f}px")

    results: list[ExtrinsicResult] = []
    aligned_by_cam: dict[int, np.ndarray] = {}
    for i, o, pe in zip(idxs, chosen, per_cam):
        R, _ = _rodrigues(o["rvec"])
        results.append(ExtrinsicResult(
            camera_index=i, rvec=o["rvec"], R=R, T=o["tvec"], reproj_error=o["err"],
        ))
        aligned_by_cam[i] = o["pts"].copy()
        warn = "  <-- BAD: re-capture / re-mark this camera" if pe > reproj_warn_px else ""
        log(f"[extrinsic] cam{i:02d}: order {o['tag']}, multiview {pe:.2f}px{warn}")
    return results, aligned_by_cam


def _rodrigues(rvec: np.ndarray) -> tuple[np.ndarray, None]:
    import cv2
    R, _ = cv2.Rodrigues(np.asarray(rvec, np.float64).reshape(3, 1))
    return np.asarray(R, np.float64), None
