"""Render synthetic views of a floor checkerboard with known ground truth.

World frame matches the calibration convention: board on the floor (Z = 0),
origin at inner corner 0, X along corners_nb[0] (rows), Y along corners_nb[1]
(cols), Z up. Cameras sit around the board looking down at it.
"""
from __future__ import annotations

import numpy as np
import cv2


def make_K(width=1920, height=1080, hfov_deg=87.0):
    f = (width / 2) / np.tan(np.radians(hfov_deg) / 2)
    return np.array([[f, 0, width / 2 - 0.5], [0, f, height / 2 - 0.5], [0, 0, 1]], np.float64)


def look_at(cam_pos, target, up=(0, 0, 1)):
    """rvec, tvec of a camera at `cam_pos` (world) looking at `target`."""
    cam_pos = np.asarray(cam_pos, float)
    z = np.asarray(target, float) - cam_pos
    z /= np.linalg.norm(z)
    x = np.cross(z, up)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])           # rows = camera axes in world
    t = -R @ cam_pos
    rvec, _ = cv2.Rodrigues(R)
    return rvec.reshape(3, 1), t.reshape(3, 1)


def board_points(corners_nb, square_m):
    r, c = corners_nb
    pts = np.zeros((r * c, 3))
    k = 0
    for x in range(r):
        for y in range(c):
            pts[k] = (x * square_m, y * square_m, 0.0)
            k += 1
    return pts


def render(corners_nb, square_m, rvec, tvec, K, size=(1920, 1080),
           blur=0.8, noise=3.0, seed=0, dist=None):
    """Return (bgr image, ground-truth (N,2) corners in the world-convention order)."""
    r, c = corners_nb
    W, H = size
    ppm = 2000.0                                     # texture pixels per metre
    n_sx, n_sy = r + 1, c + 1                        # squares along X, Y
    margin = 1.0 * square_m
    tex_w = int(round((n_sy * square_m + 2 * margin) * ppm))   # texture u ~ world Y
    tex_h = int(round((n_sx * square_m + 2 * margin) * ppm))   # texture v ~ world X
    tex = np.full((tex_h, tex_w), 235, np.uint8)
    sq = square_m * ppm
    m = margin * ppm
    for i in range(n_sx):
        for j in range(n_sy):
            if (i + j) % 2 == 0:
                y0, x0 = int(round(m + i * sq)), int(round(m + j * sq))
                tex[y0:int(round(m + (i + 1) * sq)), x0:int(round(m + (j + 1) * sq))] = 25
    # texture pixel (u, v) -> world (X, Y): inner corner (0,0) is at the corner of
    # square (0,0)/(1,1), i.e. texture (m + sq, m + sq).
    def tex_to_world(u, v):
        return ((v + 0.5 - m - sq) / ppm, (u + 0.5 - m - sq) / ppm)

    src = np.array([[0, 0], [tex_w - 1, 0], [tex_w - 1, tex_h - 1], [0, tex_h - 1]], np.float64)
    world = np.array([[*tex_to_world(u, v), 0.0] for u, v in src])
    img_pts, _ = cv2.projectPoints(world, rvec, tvec, K, np.zeros(5) if dist is None else dist)
    Hm = cv2.getPerspectiveTransform(src.astype(np.float32), img_pts.reshape(-1, 2).astype(np.float32))
    bg = np.random.default_rng(seed).integers(90, 160, (H, W)).astype(np.uint8)
    bg = cv2.GaussianBlur(bg, (0, 0), 6)
    warped = cv2.warpPerspective(tex, Hm, (W, H), flags=cv2.INTER_AREA, borderValue=0)
    mask = cv2.warpPerspective(np.full_like(tex, 255), Hm, (W, H), flags=cv2.INTER_NEAREST)
    img = np.where(mask > 0, warped, bg).astype(np.float64)
    if blur > 0:
        img = cv2.GaussianBlur(img, (0, 0), blur)
    img += np.random.default_rng(seed + 1).normal(0, noise, img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)
    gt, _ = cv2.projectPoints(board_points(corners_nb, square_m), rvec, tvec, K,
                              np.zeros(5) if dist is None else dist)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), gt.reshape(-1, 2)


def ring_cameras(n, radius=3.0, height=2.0, center=(0.1, 0.2, 0.0)):
    out = []
    for k in range(n):
        a = 2 * np.pi * k / n + 0.3
        pos = (center[0] + radius * np.cos(a), center[1] + radius * np.sin(a), height)
        out.append(look_at(pos, center))
    return out
