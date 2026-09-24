"""Calibration QC: project a known 3D cube + world axes back into each camera.

If intrinsics (K, distortion) and extrinsics (R, T) are correct, the cube sits on
the checkerboard origin, looks like a real cube from every angle, and is
consistent across cameras; because `cv2.projectPoints` applies distortion, the
overlay also confirms the distortion model. This mirrors the old add-on's
`CheckCalibExtriCube`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib


def read_calib_for_projection(calib_toml: str | Path) -> list[dict]:
    """Per-camera {name, K, dist, rvec, tvec, size} for cv2.projectPoints."""
    import cv2
    with open(calib_toml, "rb") as f:
        data = tomllib.load(f)
    out = []
    for key, cam in data.items():
        if key.lower() == "metadata" or not isinstance(cam, dict):
            continue
        if "rotation" not in cam:
            continue
        K = np.asarray(cam["matrix"], dtype=float)
        dist = np.asarray(cam.get("distortions", [0, 0, 0, 0]), dtype=float).ravel()
        rvec = np.asarray(cam["rotation"], dtype=float).reshape(3, 1)
        tvec = np.asarray(cam["translation"], dtype=float).reshape(3, 1)
        out.append({"name": str(cam.get("name", key)), "K": K, "dist": dist,
                    "rvec": rvec, "tvec": tvec,
                    "size": cam.get("size", [1920, 1080])})
    return out


def world_unit_scale(cams: list[dict]) -> str:
    """Guess whether the calibration world is in metres or millimetres from the
    translation magnitudes (cameras are a few metres away)."""
    mags = [float(np.linalg.norm(c["tvec"])) for c in cams]
    med = float(np.median(mags)) if mags else 1.0
    return "m" if med < 50 else "mm"


def draw_overlay(img_bgr: np.ndarray, cam: dict, square_size_world: float,
                 n_squares: int = 3, draw_cube: bool = True) -> np.ndarray:
    """Project world axes (+optionally a cube) at the origin onto the image."""
    import cv2

    K, dist, rvec, tvec = cam["K"], cam["dist"], cam["rvec"], cam["tvec"]
    s = square_size_world * n_squares
    out = img_bgr.copy()

    def proj(pts3d):
        p, _ = cv2.projectPoints(np.asarray(pts3d, dtype=float), rvec, tvec, K, dist)
        return p.reshape(-1, 2)

    # world axes: X red, Y green, Z blue
    axes = proj([[0, 0, 0], [s, 0, 0], [0, s, 0], [0, 0, s]])
    o = tuple(np.round(axes[0]).astype(int))
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # BGR: X,Y,Z
    labels = ["X", "Y", "Z"]
    for i in range(3):
        p = tuple(np.round(axes[i + 1]).astype(int))
        cv2.line(out, o, p, colors[i], 3, cv2.LINE_AA)
        cv2.putText(out, labels[i], p, cv2.FONT_HERSHEY_SIMPLEX, 1.0, colors[i], 2, cv2.LINE_AA)
    cv2.circle(out, o, 5, (0, 255, 255), -1)

    if draw_cube:
        c = proj([
            [0, 0, 0], [s, 0, 0], [s, s, 0], [0, s, 0],          # base
            [0, 0, s], [s, 0, s], [s, s, s], [0, s, s],          # top
        ]).astype(int)
        edges = [(0, 1), (1, 2), (2, 3), (3, 0),
                 (4, 5), (5, 6), (6, 7), (7, 4),
                 (0, 4), (1, 5), (2, 6), (3, 7)]
        for a, b in edges:
            cv2.line(out, tuple(c[a]), tuple(c[b]), (60, 220, 255), 2, cv2.LINE_AA)
    return out


def undistort(img_bgr: np.ndarray, cam: dict) -> np.ndarray:
    """Return the distortion-corrected image (straight lines become straight)."""
    import cv2
    return cv2.undistort(img_bgr, cam["K"], cam["dist"])


# --------------------------------------------------------------------------- #
# Convention-free consistency check: triangulate the annotated board corners
# across all cameras and measure reprojection error + board flatness/regularity.
# This validates the calibration without assuming where the world origin is or
# how big the squares are.
# --------------------------------------------------------------------------- #
def _proj_matrix(cam: dict) -> np.ndarray:
    import cv2
    R, _ = cv2.Rodrigues(cam["rvec"])
    return cam["K"] @ np.hstack([R, cam["tvec"]])


def triangulate_multiview(cams: list[dict], pts_per_cam: list[np.ndarray]) -> np.ndarray:
    """DLT-triangulate N points seen (undistorted) by all cameras -> (N,3).

    `pts_per_cam[c]` is that camera's (N,2) annotated corners, in matching order.
    Points are undistorted first so the linear projection matrices apply.
    """
    import cv2
    Ps = [_proj_matrix(c) for c in cams]
    # undistort the 2D observations to normalized->pixel (P already includes K)
    und = []
    for c, pts in zip(cams, pts_per_cam):
        p = np.asarray(pts, float).reshape(-1, 1, 2)
        u = cv2.undistortPoints(p, c["K"], c["dist"], P=c["K"]).reshape(-1, 2)
        und.append(u)
    n = und[0].shape[0]
    out = np.zeros((n, 3))
    for k in range(n):
        A = []
        for uc, P in zip(und, Ps):
            u, v = uc[k]
            A.append(u * P[2] - P[0])
            A.append(v * P[2] - P[1])
        _, _, Vt = np.linalg.svd(np.asarray(A))
        X = Vt[-1]
        out[k] = X[:3] / X[3]
    return out


def reproject(cam: dict, pts3d: np.ndarray) -> np.ndarray:
    import cv2
    p, _ = cv2.projectPoints(np.asarray(pts3d, float), cam["rvec"], cam["tvec"],
                             cam["K"], cam["dist"])
    return p.reshape(-1, 2)


def draw_cube_from_corners(img_bgr: np.ndarray, cam: dict, corners_3d: np.ndarray,
                           n_squares: int = 2) -> np.ndarray:
    """Draw a cube + axes anchored to the USER-MARKED corners (not the world
    origin). Origin = the first marked corner; X along corner[0]->corner[1];
    Z = board normal; size = the marked corner spacing. So the cube sits on the
    board exactly where you labelled it, at the size you labelled."""
    import cv2

    P = np.asarray(corners_3d, float)
    o = P[0].copy()
    x = P[1] - P[0]
    step = float(np.linalg.norm(x))
    if step < 1e-9:
        return img_bgr
    x = x / step
    # board normal from best-fit plane
    ctr = P.mean(0)
    _, _, Vt = np.linalg.svd(P - ctr)
    z = Vt[-1] / (np.linalg.norm(Vt[-1]) + 1e-9)
    y = np.cross(z, x); y /= (np.linalg.norm(y) + 1e-9)
    z = np.cross(x, y)  # re-orthogonalize, right-handed
    s = step * n_squares
    out = img_bgr.copy()

    def P3(a, b, c):
        return o + a * s * x + b * s * y + c * s * z

    def pj(pts):
        p, _ = cv2.projectPoints(np.asarray(pts, float), cam["rvec"], cam["tvec"],
                                 cam["K"], cam["dist"])
        return p.reshape(-1, 2)

    # axes at the marked origin
    ax = pj([o, o + s * x, o + s * y, o + s * z])
    org = tuple(np.round(ax[0]).astype(int))
    for i, col in enumerate([(0, 0, 255), (0, 255, 0), (255, 0, 0)]):
        cv2.line(out, org, tuple(np.round(ax[i + 1]).astype(int)), col, 3, cv2.LINE_AA)
    cv2.circle(out, org, 6, (0, 255, 255), -1)
    # cube
    corners = pj([P3(a, b, c) for a in (0, 1) for b in (0, 1) for c in (0, 1)]).astype(int)
    idx = {(a, b, c): i for i, (a, b, c) in enumerate(
        [(a, b, c) for a in (0, 1) for b in (0, 1) for c in (0, 1)])}
    edges = [((0, 0, 0), (1, 0, 0)), ((0, 0, 0), (0, 1, 0)), ((0, 0, 0), (0, 0, 1)),
             ((1, 1, 1), (0, 1, 1)), ((1, 1, 1), (1, 0, 1)), ((1, 1, 1), (1, 1, 0)),
             ((1, 0, 0), (1, 1, 0)), ((1, 0, 0), (1, 0, 1)), ((0, 1, 0), (1, 1, 0)),
             ((0, 1, 0), (0, 1, 1)), ((0, 0, 1), (1, 0, 1)), ((0, 0, 1), (0, 1, 1))]
    for a, b in edges:
        cv2.line(out, tuple(corners[idx[a]]), tuple(corners[idx[b]]), (60, 220, 255), 2, cv2.LINE_AA)
    return out


def consistency_report(cams: list[dict], pts_per_cam: list[np.ndarray]) -> dict:
    """Triangulate the annotated corners and quantify calibration quality."""
    pts3d = triangulate_multiview(cams, pts_per_cam)
    per_cam_err = []
    all_err = []
    for c, pts in zip(cams, pts_per_cam):
        rp = reproject(c, pts3d)
        e = np.linalg.norm(rp - np.asarray(pts, float), axis=1)
        per_cam_err.append(float(e.mean()))
        all_err.append(e)
    all_err = np.concatenate(all_err)
    # board flatness: distance to best-fit plane
    ctr = pts3d.mean(0)
    _, _, Vt = np.linalg.svd(pts3d - ctr)
    planar_mm = float(np.abs((pts3d - ctr) @ Vt[-1]).max() * 1000)
    # grid regularity: nearest-neighbour spacing spread
    from scipy.spatial import cKDTree
    d, _ = cKDTree(pts3d).query(pts3d, k=2)
    sp = d[:, 1]
    mean_err = float(all_err.mean())
    if mean_err > 30.0 or planar_mm > 200.0:
        verdict = "FAILED"   # triangulation blew up -> corners don't correspond
    elif mean_err < 3.0 and planar_mm < 20.0:
        verdict = "GOOD"
    else:
        verdict = "CHECK"
    return {
        "points_3d": pts3d,
        "mean_px": mean_err,
        "median_px": float(np.median(all_err)),
        "max_px": float(all_err.max()),
        "per_cam_px": per_cam_err,
        "planarity_mm": planar_mm,
        "spacing_mm": float(sp.mean() * 1000),
        "spacing_std_mm": float(sp.std() * 1000),
        "verdict": verdict,
    }
