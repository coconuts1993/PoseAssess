"""Draw the board registration on camera images (OpenCV, BGR in place), for QC views on the
"2b. Wii Board" page and (optionally) the capture preview. Port of PoseBoard 3ea6d8a
``poseboard/overlay.py`` without the live-pose part.
"""

from __future__ import annotations

import cv2
import numpy as np

from .board import BoardRegistration
from .calib import CameraCalibration
from .geometry import LANDMARK_NAMES, SENSOR_NAMES

CLICK_COLORS = [(0, 0, 255), (0, 200, 255), (0, 255, 0), (255, 128, 0), (255, 0, 255)]  # BGR
BOARD_COLOR = (255, 255, 0)     # outline (cyan)
FRONT_COLOR = (0, 140, 255)     # front edge TL-TR (orange)
COP_COLOR = (255, 80, 0)        # COP + force (blue)


def project_world(cam: CameraCalibration, pts_w: np.ndarray) -> np.ndarray:
    """World points -> pixels (N,2); NaN rows for points behind the camera."""
    pts_w = np.asarray(pts_w, np.float64).reshape(-1, 3)
    out = np.full((len(pts_w), 2), np.nan)
    ok = np.all(np.isfinite(pts_w), axis=1)
    if not ok.any():
        return out
    rvec = np.asarray(cam.rvec if cam.has_extrinsics else np.zeros(3), np.float64).ravel()
    tvec = np.asarray(cam.tvec if cam.has_extrinsics else np.zeros(3), np.float64).ravel()
    pc = pts_w[ok] @ cv2.Rodrigues(rvec)[0].T + tvec
    img, _ = cv2.projectPoints(pts_w[ok], rvec, tvec, cam.K, cam.dist)
    img = img.reshape(-1, 2)
    img[pc[:, 2] <= 1e-3] = np.nan  # behind (or at) the camera
    out[ok] = img
    return out


def _p(pt) -> tuple[int, int] | None:
    if pt is None or not np.all(np.isfinite(pt)) or np.any(np.abs(pt) > 1e5):
        return None
    return int(round(float(pt[0]))), int(round(float(pt[1])))


def _line(img, a, b, color, th=2):
    a, b = _p(a), _p(b)
    if a and b:
        cv2.line(img, a, b, color, th, cv2.LINE_AA)


def _scale(img) -> float:
    """Line / font scale for the image size (1.0 at 1280 px wide)."""
    return max(0.5, min(3.0, img.shape[1] / 1280.0))


def draw_clicks(img: np.ndarray, clicks: list, next_hint: str | None = None) -> None:
    """Clicked landmarks with labels ``i:TL``...; outline once 4 corners exist; optional hint."""
    k = _scale(img)
    th = max(1, int(round(2 * k)))
    for i, c in enumerate(list(clicks)[:len(LANDMARK_NAMES)]):
        p = _p(c)
        if p:
            col = CLICK_COLORS[i % len(CLICK_COLORS)]
            cv2.circle(img, p, int(6 * k), col, th, cv2.LINE_AA)
            cv2.drawMarker(img, p, col, cv2.MARKER_CROSS, int(18 * k), max(1, th // 2), cv2.LINE_AA)
            cv2.putText(img, f"{i + 1}:{LANDMARK_NAMES[i]}", (p[0] + int(8 * k), p[1] - int(8 * k)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6 * k, col, th, cv2.LINE_AA)
    if len(clicks) >= 4:
        for i in range(4):
            _line(img, clicks[i], clicks[(i + 1) % 4], (200, 200, 200), max(1, th // 2))
    if next_hint:
        cv2.putText(img, next_hint, (int(12 * k), int(30 * k)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8 * k, (0, 255, 255), th, cv2.LINE_AA)


def draw_board(img: np.ndarray, cam: CameraCalibration, reg: BoardRegistration,
               label: bool = True) -> None:
    """Board outline, the front edge highlighted (TL-TR), sensors, and board axes (x red,
    y green, z blue) projected with ``cam``. ``label``: write the corner and sensor names (the
    "front" label is always written)."""
    k = _scale(img)
    th = max(1, int(round(2 * k)))
    T, geo = reg.pose.board_to_world, reg.geometry
    lm = project_world(cam, T.apply(geo.landmarks()))
    for i in range(4):
        _line(img, lm[i], lm[(i + 1) % 4], BOARD_COLOR, th)
    _line(img, lm[0], lm[1], FRONT_COLOR, 2 * th)  # front edge (TL-TR, opposite the power button)
    if label:
        for i in range(4):
            p = _p(lm[i])
            if p:
                cv2.putText(img, LANDMARK_NAMES[i], (p[0] + int(6 * k), p[1] - int(6 * k)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55 * k, BOARD_COLOR, max(1, th // 2),
                            cv2.LINE_AA)
    mid = _p((lm[0] + lm[1]) / 2) if np.isfinite(lm[:2]).all() else None
    if mid:
        cv2.putText(img, "front", (mid[0] + int(6 * k), mid[1] - int(6 * k)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55 * k, FRONT_COLOR, max(1, th // 2),
                    cv2.LINE_AA)
    sensors = project_world(cam, T.apply(geo.sensors()))
    for name, s in zip(SENSOR_NAMES, sensors):
        p = _p(s)
        if p:
            cv2.circle(img, p, int(5 * k), (180, 180, 180), -1, cv2.LINE_AA)
            if label:
                cv2.putText(img, name, (p[0] + int(6 * k), p[1] + int(4 * k)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45 * k, (230, 230, 230), 1, cv2.LINE_AA)
    axes = project_world(cam, T.apply(np.array([[0, 0, 0], [0.1, 0, 0], [0, 0.1, 0],
                                                [0, 0, 0.1]], float)))
    for i, col in zip((1, 2, 3), ((0, 0, 255), (0, 255, 0), (255, 0, 0))):
        _line(img, axes[0], axes[i], col, th)


def draw_cop(img: np.ndarray, cam: CameraCalibration, reg: BoardRegistration,
             cop_board: np.ndarray, total_kg: float, m_per_kg: float = 0.005) -> None:
    """COP dot + vertical force line (``m_per_kg`` metres per kg) projected with ``cam``."""
    cop = np.asarray(cop_board, np.float64).ravel()[:2]
    if not np.all(np.isfinite(cop)) or not np.isfinite(total_kg):
        return
    k = _scale(img)
    T = reg.pose.board_to_world
    base = T.apply([cop[0], cop[1], 0.0])
    tip = base + reg.up_world * float(total_kg) * m_per_kg
    pts = project_world(cam, np.vstack([base, tip]))
    _line(img, pts[0], pts[1], COP_COLOR, max(1, int(round(3 * k))))
    p = _p(pts[0])
    if p:
        cv2.circle(img, p, int(8 * k), COP_COLOR, -1, cv2.LINE_AA)
        cv2.putText(img, f"COP {total_kg:.1f}kg", (p[0] + int(10 * k), p[1] + int(20 * k)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6 * k, COP_COLOR, max(1, int(round(2 * k))),
                    cv2.LINE_AA)


def draw_press_here(img: np.ndarray, cam: CameraCalibration, reg: BoardRegistration,
                    corner: str = "TL") -> None:
    """Two rings + "PRESS" at the registered board's ``corner`` (the corner check: the operator
    presses the corner that was clicked as TL, found from the camera image). The label is
    drawn towards the board centre, so a view zoomed to the board keeps it."""
    i = LANDMARK_NAMES.index(corner)
    lm = project_world(cam, reg.pose.board_to_world.apply(reg.geometry.landmarks()))
    p, c = _p(lm[i]), _p(lm[4])
    if not p:
        return
    k = _scale(img)
    th = max(2, int(round(3 * k)))
    col = (0, 255, 255)  # yellow
    r = int(22 * k)
    cv2.circle(img, p, int(14 * k), col, th, cv2.LINE_AA)
    cv2.circle(img, p, r, col, th, cv2.LINE_AA)
    text = "PRESS"
    (tw, tht), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7 * k, th)
    d = np.subtract(c, p, dtype=float) if c else np.array([1.0, 0.0])
    d = d / (np.linalg.norm(d) or 1.0)
    cx, cy = p[0] + d[0] * (r + 8 * k + tw / 2), p[1] + d[1] * (r + 8 * k + tht / 2)
    org = (int(cx - tw / 2), int(cy + tht / 2))
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.7 * k, (0, 0, 0), th + 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.7 * k, col, th, cv2.LINE_AA)


def board_check_image(img: np.ndarray, cam: CameraCalibration, reg: BoardRegistration,
                      clicks: np.ndarray | None = None,
                      highlight: str | None = None) -> np.ndarray:
    """Copy of ``img`` with the registered board drawn and (optionally) the clicks, plus the
    reprojection error text: the per-camera cell of the QC grid. ``highlight`` (e.g. "TL"):
    mark that corner with ``draw_press_here`` (running corner check)."""
    out = np.ascontiguousarray(img).copy()
    if out.ndim == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    if clicks is not None and len(clicks):
        draw_clicks(out, [tuple(map(float, c)) for c in np.asarray(clicks, float).reshape(-1, 2)])
    draw_board(out, cam, reg, label=clicks is None or not len(clicks))
    if highlight:
        draw_press_here(out, cam, reg, highlight)
    k = _scale(out)
    err = reg.pose.reproj_error_px.get(cam.name)
    if err is not None:
        text, col = f"{cam.name}: reprojection {err:.1f} px", ((0, 200, 0) if err <= 5 else
                                                              (0, 165, 255) if err <= 15 else
                                                              (0, 0, 255))
    else:
        text, col = f"{cam.name}: not used for the board position", (200, 200, 200)
    org = (int(12 * k), out.shape[0] - int(14 * k))
    cv2.putText(out, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.7 * k, (0, 0, 0),
                max(2, int(round(5 * k))), cv2.LINE_AA)
    cv2.putText(out, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.7 * k, col,
                max(1, int(round(2 * k))), cv2.LINE_AA)
    if reg.stale:
        cv2.putText(out, "STALE: recompute the board position", (int(12 * k), int(30 * k)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7 * k, (0, 0, 255), max(1, int(round(2 * k))),
                    cv2.LINE_AA)
    return out
