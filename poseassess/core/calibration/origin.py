"""Decide which physical board corner is the world origin (corner 0).

World convention (same as `extrinsic._object_points_x5`): origin at corner 0,
X along `corners_nb[0]` (row index), Y along `corners_nb[1]` (column index),
Z = X x Y. Any of the board's 4 outer corners can be corner 0; for a given
corner the reading of the grid is then fixed.

Two cues place the origin without clicks:

* Square colour. The square diagonally inside each outer corner is dark or
  light. With one odd and one even corner count (e.g. 5 x 8) the four corners
  split 2 + 2 by colour, so knowing the origin square's colour narrows the
  origin to two corners. Colour is sampled INSIDE the board (that square and its
  neighbour), so background clutter never matters.
* Handedness. For cameras looking down at a floor board, two of the four
  readings give Z up and two give Z down (`invert_z` flips which). Colour +
  handedness together pin a single corner for odd x even boards.

The operator always confirms; if the guess is wrong, one click on the right
corner (`reorder_to_corner`) renumbers the grid, and the colour of that corner is
remembered (`origin_square`) so the next cameras come out right by themselves.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from .board_detect import reorder_grid, GRID_SYMMETRIES


@dataclass
class OriginGuess:
    points: np.ndarray            # reordered so index 0 is the chosen origin
    sym: str                      # grid symmetry applied to the input
    origin_xy: tuple[float, float]
    square_colour: str            # colour of the origin square: "dark"/"light"/""
    z_up: bool                    # handedness of the chosen reading
    confident: bool               # True when colour + handedness agree on one corner
    reason: str


def _grid(points, corners_nb):
    r, c = int(corners_nb[0]), int(corners_nb[1])
    return np.asarray(points, np.float64).reshape(r, c, 2)


def z_up(points: np.ndarray, corners_nb: Sequence[int], invert_z: bool = False) -> bool:
    """Handedness test for a camera looking DOWN at a floor board.

    In image coordinates (y down) a right-handed board frame whose Z points
    toward the camera has cross(dX, dY) < 0, with dX the step to the next row
    and dY the step to the next column. `invert_z` (which negates board X)
    flips the test.
    """
    g = _grid(points, corners_nb)
    dX = g[-1, 0] - g[0, 0]
    dY = g[0, -1] - g[0, 0]
    cross = dX[0] * dY[1] - dX[1] * dY[0]
    up = cross < 0
    return bool((not up) if invert_z else up)


def _sample_square(gray: np.ndarray, quad: np.ndarray) -> float:
    """Mean intensity inside the central part of one board square given its 4
    corner pixels (any order around the square)."""
    import cv2
    ctr = quad.mean(0)
    inner = ctr + 0.45 * (quad - ctr)               # stay clear of the edges
    mask = np.zeros(gray.shape[:2], np.uint8)
    cv2.fillConvexPoly(mask, np.round(inner).astype(np.int32), 1)
    vals = gray[mask.astype(bool)]
    if vals.size == 0:                              # tiny square: bilinear sample at centre
        x, y = float(ctr[0]), float(ctr[1])
        x0, y0 = int(np.clip(np.floor(x), 0, gray.shape[1] - 2)), int(np.clip(np.floor(y), 0, gray.shape[0] - 2))
        fx, fy = x - x0, y - y0
        p = gray[y0:y0 + 2, x0:x0 + 2].astype(np.float64)
        return float(p[0, 0] * (1 - fx) * (1 - fy) + p[0, 1] * fx * (1 - fy)
                     + p[1, 0] * (1 - fx) * fy + p[1, 1] * fx * fy)
    return float(vals.mean())


def corner_square_colour(gray: np.ndarray, points: np.ndarray,
                         corners_nb: Sequence[int]) -> tuple[str, float]:
    """Colour of the square diagonally inside corner 0 of this reading, with a
    contrast score (0..1) from comparing it to its neighbouring square."""
    g = _grid(points, corners_nb)
    r, c = g.shape[:2]
    if r < 2 or c < 2:
        return "", 0.0
    sq = np.array([g[0, 0], g[0, 1], g[1, 1], g[1, 0]])
    a = _sample_square(gray, sq)
    if c >= 3:
        nb = np.array([g[0, 1], g[0, 2], g[1, 2], g[1, 1]])
    else:
        nb = np.array([g[1, 0], g[1, 1], g[2, 1], g[2, 0]])
    b = _sample_square(gray, nb)
    contrast = abs(a - b) / 255.0
    return ("dark" if a < b else "light"), float(contrast)


def candidate_readings(points: np.ndarray, corners_nb: Sequence[int]):
    """The 4 readings that keep the grid shape, one per outer corner."""
    for sym in GRID_SYMMETRIES:
        yield sym, reorder_grid(points, corners_nb, sym)


def canonicalize(
    img: np.ndarray,
    points: np.ndarray,
    corners_nb: Sequence[int],
    origin_square: str = "",
    invert_z: bool = False,
    min_contrast: float = 0.08,
) -> OriginGuess:
    """Choose corner 0 automatically from colour + handedness.

    `origin_square` is the remembered colour of the origin square ("" if not
    known yet). Returns the best reading and whether the choice is unique.
    """
    from .board_detect import _to_gray
    gray = _to_gray(img)
    cands = []
    for sym, pts in candidate_readings(points, corners_nb):
        colour, contrast = corner_square_colour(gray, pts, corners_nb)
        cands.append({"sym": sym, "pts": pts, "colour": colour,
                      "contrast": contrast, "z_up": z_up(pts, corners_nb, invert_z)})
    good = [k for k in cands if k["z_up"]]
    reason = "handedness only"
    pool = good or cands
    if origin_square in ("dark", "light"):
        by_colour = [k for k in pool if k["colour"] == origin_square
                     and k["contrast"] >= min_contrast]
        if by_colour:
            pool = by_colour
            reason = f"origin square is {origin_square} + handedness"
        else:
            reason = "origin colour not visible clearly; handedness only"
    # Prefer, among remaining ties, the corner closest to the image top-left so
    # repeated runs on the same view give the same answer.
    pool = sorted(pool, key=lambda k: (k["pts"][0, 1] + k["pts"][0, 0]))
    best = pool[0]
    confident = bool(len(pool) == 1 and best["z_up"])
    return OriginGuess(
        points=best["pts"], sym=best["sym"],
        origin_xy=(float(best["pts"][0, 0]), float(best["pts"][0, 1])),
        square_colour=best["colour"], z_up=best["z_up"],
        confident=confident, reason=reason,
    )


def reorder_to_corner(points: np.ndarray, corners_nb: Sequence[int],
                      click_xy: Sequence[float]) -> tuple[np.ndarray, str]:
    """Renumber the grid so the outer corner nearest `click_xy` becomes corner 0.
    Returns (points, symmetry applied)."""
    click = np.asarray(click_xy, np.float64).reshape(2)
    best, best_sym, best_d = None, "id", np.inf
    for sym, pts in candidate_readings(points, corners_nb):
        d = float(np.linalg.norm(pts[0] - click))
        if d < best_d:
            best, best_sym, best_d = pts, sym, d
    return best, best_sym
