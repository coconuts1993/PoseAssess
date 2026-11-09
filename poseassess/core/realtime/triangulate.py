"""Confidence-weighted multi-view DLT triangulation (vectorised over keypoints)."""
from __future__ import annotations

import numpy as np


def triangulate(P: np.ndarray, uv: np.ndarray, w: np.ndarray, min_views: int = 2) -> np.ndarray:
    """Triangulate J points seen by C cameras.

    P:  (C, 3, 4) projection matrices (normalised or pixel, matching ``uv``)
    uv: (C, J, 2) image coordinates
    w:  (C, J) weights (keypoint confidence; 0 = not seen)
    Returns (J, 3); NaN where fewer than ``min_views`` cameras have weight > 0.
    """
    P = np.asarray(P, dtype=np.float64)
    uv = np.asarray(uv, dtype=np.float64)
    w = np.where(np.isfinite(w), np.asarray(w, dtype=np.float64), 0.0)
    w = np.where(np.isfinite(uv).all(axis=-1), w, 0.0)
    uv = np.nan_to_num(uv)
    C, J = w.shape
    # rows: w * (u * P3 - P1), w * (v * P3 - P2)  -> A: (J, 2C, 4)
    p1, p2, p3 = P[:, 0, :], P[:, 1, :], P[:, 2, :]          # (C, 4)
    u = uv[..., 0][..., None]                                 # (C, J, 1)
    v = uv[..., 1][..., None]
    ww = w[..., None]
    rows_u = ww * (u * p3[:, None, :] - p1[:, None, :])       # (C, J, 4)
    rows_v = ww * (v * p3[:, None, :] - p2[:, None, :])
    A = np.concatenate([rows_u, rows_v], axis=0).transpose(1, 0, 2)  # (J, 2C, 4)
    _, _, vt = np.linalg.svd(A)
    X = vt[:, -1, :]                                          # (J, 4)
    with np.errstate(divide="ignore", invalid="ignore"):
        X = X[:, :3] / X[:, 3:4]
    ok = (w > 0).sum(axis=0) >= min_views
    X[~ok] = np.nan
    return X


def reprojection_error(P: np.ndarray, uv: np.ndarray, w: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Mean reprojection error per point (same units as ``uv``) over the cameras with w > 0."""
    Xh = np.hstack([X, np.ones((len(X), 1))])                # (J, 4)
    proj = np.einsum("cij,kj->cki", P, Xh)                    # (C, J, 3)
    with np.errstate(divide="ignore", invalid="ignore"):
        px = proj[..., :2] / proj[..., 2:3]
        d = np.linalg.norm(px - uv, axis=-1)                  # (C, J)
        seen = (w > 0) & np.isfinite(d)
        err = np.where(seen, d, 0.0).sum(0) / seen.sum(0)
    return err
