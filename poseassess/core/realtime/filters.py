"""One-Euro filter (Casiez et al. 2012), vectorised over any array shape.

Low jitter when still (``min_cutoff``), low lag when moving fast (``beta``). Timestamps are the
capture times of the frames, so a dropped frame does not distort the filter.
"""
from __future__ import annotations

import math

import numpy as np


def _alpha(cutoff, dt):
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class OneEuroFilter:
    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.3, d_cutoff: float = 1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.reset()

    def reset(self) -> None:
        self._x = None
        self._dx = None
        self._t = None

    def __call__(self, x: np.ndarray, t: float) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if self._x is None or self._x.shape != x.shape:
            self._x = x.copy()
            self._dx = np.zeros_like(x)
            self._t = t
            return x.copy()
        dt = t - self._t
        if dt <= 0:
            dt = 1e-3
        self._t = t
        prev = self._x
        # a point that (re)appears starts from its raw value; a missing point stays missing
        fresh = np.isfinite(x) & ~np.isfinite(prev)
        prev = np.where(fresh, x, prev)
        dx = np.where(fresh, 0.0, (x - prev) / dt)
        a_d = _alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1 - a_d) * np.nan_to_num(self._dx)
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        tau = 1.0 / (2.0 * math.pi * cutoff)
        a = 1.0 / (1.0 + tau / dt)
        x_hat = a * x + (1 - a) * prev
        x_hat = np.where(np.isfinite(x), x_hat, np.nan)
        # keep the last estimate of a point that is momentarily lost, for when it returns
        self._x = np.where(np.isfinite(x_hat), x_hat, prev)
        self._dx = np.where(np.isfinite(dx_hat), dx_hat, 0.0)
        return x_hat
