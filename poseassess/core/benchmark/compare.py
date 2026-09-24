"""Compare two OpenSim motions coordinate-by-coordinate.

This is the scientific core of the benchmark: given two joint-angle time series
of the *same* trial (one vs a marker-based reference, or two 2D backends),
quantify how much they differ per joint. Because "SOTA on COCO" does not imply
"accurate joint angles", these biomechanical error metrics — not CV keypoint AP
— are what we report.

Honest error decomposition (added after the first OpenCap validation, where raw
RMSE was dominated by confounds rather than tracking error):

  raw RMSE      difference with no alignment.
  time lag      a single global sync offset between the two series, found by
                cross-correlation on the strongest-motion joint and applied to
                ALL joints (one physical time offset for the whole trial).
  aligned RMSE  RMSE after removing the time lag.
  bias          the constant per-joint offset that remains — this is mostly the
                difference in how two OpenSim MODELS define a joint's zero pose,
                NOT a tracking error.
  residual RMSE aligned RMSE with the constant bias removed = the real,
                waveform-tracking error. This is the headline accuracy number.

`compare_motions` stays backward-compatible: `.rmse` and `.mean_rmse` keep their
original (raw) meaning; the new fields are additive.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Optional

import numpy as np

from ..assessment.mot_io import MotionData
from ..assessment.metrics import CLINICAL_JOINTS


@dataclass
class CoordComparison:
    coord: str
    mae: float
    rmse: float                      # raw RMSE (no alignment) — backward compatible
    max_err: float
    rom_ref: float
    rom_other: float
    rom_diff: float                  # other - ref
    pearson_r: float
    n: int
    # --- honest decomposition (defaults keep old constructors working) ---
    rmse_aligned: float = float("nan")   # after removing global time lag
    bias: float = float("nan")           # constant offset (cross-model, not tracking)
    rmse_residual: float = float("nan")  # after removing lag AND bias = tracking error
    lr_swapped: bool = False             # this coord was matched to the contralateral GT side


@dataclass
class MotionComparison:
    reference: str
    other: str
    coords: list[CoordComparison]
    mean_rmse: float                 # raw, over compared clinical coords
    mean_mae: float
    lag_samples: int = 0
    lag_ms: float = 0.0
    mean_rmse_residual: float = float("nan")
    n_lr_swapped: int = 0            # bilateral joints matched to the contralateral side

    def to_dict(self) -> dict:
        return {
            "reference": self.reference,
            "other": self.other,
            "mean_rmse": self.mean_rmse,
            "mean_mae": self.mean_mae,
            "mean_rmse_residual": self.mean_rmse_residual,
            "lag_samples": self.lag_samples,
            "lag_ms": self.lag_ms,
            "n_lr_swapped": self.n_lr_swapped,
            "coords": [asdict(c) for c in self.coords],
        }


def _common_time_grid(a: MotionData, b: MotionData, n: Optional[int] = None) -> np.ndarray:
    ta, tb = a.df.index.to_numpy(), b.df.index.to_numpy()
    t0 = max(ta.min(), tb.min())
    t1 = min(ta.max(), tb.max())
    if not (t1 > t0):
        raise ValueError("motions do not overlap in time")
    if n is None:
        n = min(len(ta), len(tb))
    return np.linspace(t0, t1, int(n))


def _resample(times: np.ndarray, values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    return np.interp(grid, times, values)


def _estimate_global_lag(
    ref_series: dict[str, np.ndarray],
    oth_series: dict[str, np.ndarray],
    max_lag: int,
) -> int:
    """Best integer shift `s` maximizing ROM-weighted agreement across joints.

    One physical time offset governs the whole trial, so we estimate a single
    lag — but aggregate the cross-correlation over all shared joints, weighted by
    reference ROM, so a single noisy high-ROM joint (e.g. arms in a sit-to-stand)
    can't dominate the estimate.
    """
    coords = [c for c in ref_series
              if ref_series[c].std() > 1e-9 and oth_series[c].std() > 1e-9]
    if not coords:
        return 0
    weights = {c: float(np.ptp(ref_series[c])) for c in coords}
    wsum = sum(weights.values()) or 1.0

    best_s, best_score = 0, -np.inf
    for s in range(-max_lag, max_lag + 1):
        score = 0.0
        for c in coords:
            corr = np.corrcoef(ref_series[c], np.roll(oth_series[c], s))[0, 1]
            if not np.isnan(corr):
                score += weights[c] * corr
        score /= wsum
        if score > best_score:
            best_score, best_s = score, s
    return best_s


# Minimum decisive margin ((swap-identity)/(|swap|+|identity|)) to accept a
# whole-body L/R swap. Genuine gait flips score ~1.0; symmetric-task ties ~0.01.
_LR_SWAP_MARGIN = 0.15


def _lr_match(shared: list[str], ref_series: dict, oth_series: dict) -> dict[str, str]:
    """Map each reference coord to the OTHER coord it should be compared against.

    Decides ONE whole-body left/right assignment for the trial: either identity
    (r<->r, l<->l) or a full swap (r<->l, l<->r), chosen by the ROM-weighted sum
    of correlations over ALL bilateral joints. A per-joint decision would flip
    individual joints on noise (producing physically impossible mixed skeletons),
    so the swap is all-or-nothing. This isolates limb-motion tracking accuracy
    from left/right *labeling* — a separable, reportable failure mode (in gait
    the sides are anti-phase, so a mislabel flips the sign of the correlation).
    """
    def corr(a, b):
        a, b = ref_series[a], oth_series[b]
        return np.corrcoef(a, b)[0, 1] if a.std() > 1e-9 and b.std() > 1e-9 else 0.0

    identity = swapped = wsum = 0.0
    bases = {c[:-2] for c in shared if c.endswith(("_r", "_l"))}
    for base in bases:
        r, l = base + "_r", base + "_l"
        if not all(k in ref_series and k in oth_series for k in (r, l)):
            continue
        w = float(np.ptp(ref_series[r]) + np.ptp(ref_series[l]))  # ROM weight
        identity += w * (corr(r, r) + corr(l, l))
        swapped += w * (corr(r, l) + corr(l, r))
        wsum += 2 * w

    # Only swap when the same-side match is genuinely POOR (a real L/R error makes
    # it anti-correlated) AND the swap wins DECISIVELY. A real whole-body flip in
    # gait drives identity anti-correlated, so the margin is large (~100%); a
    # symmetric task (STS/squats) ties within noise (margin ~1%) and must NOT be
    # counted as a swap. Both guards together reject symmetric-task false positives.
    identity_mean = identity / wsum if wsum else 0.0
    margin = (swapped - identity) / (abs(identity) + abs(swapped) + 1e-9)
    if swapped > identity and identity_mean < 0.5 and margin > _LR_SWAP_MARGIN:
        return {c: (c[:-2] + "_r" if c.endswith("_l") else c[:-2] + "_l")
                if c.endswith(("_r", "_l")) else c
                for c in shared}
    return {c: c for c in shared}


def compare_motions(
    reference: MotionData,
    other: MotionData,
    ref_name: str = "reference",
    other_name: str = "other",
    coordinates: Optional[list[str]] = None,
    align_time: bool = True,
    max_lag_fraction: float = 0.2,
    max_lag_ms: float = 200.0,
    lr_match: bool = False,
) -> MotionComparison:
    """Compare `other` against `reference` on their shared clinical coordinates.

    Both series are resampled onto a shared time grid. When `align_time`, one
    global cross-correlation lag (capped at both `max_lag_fraction` of the signal
    and an absolute `max_lag_ms`) is removed before the aligned/residual metrics
    — the absolute cap stops the lag from jumping a whole cycle on short cyclic
    signals (e.g. a ~1 s gait clip). When `lr_match`, each bilateral joint is
    matched to whichever GT side correlates better, and the swap is recorded;
    this measures limb-tracking accuracy independent of left/right labeling.
    The raw `.rmse` is always kept unaligned, same-side, for transparency.
    """
    grid = _common_time_grid(reference, other)
    n = len(grid)
    dt = (grid[-1] - grid[0]) / max(n - 1, 1)
    t_ref = reference.df.index.to_numpy()
    t_oth = other.df.index.to_numpy()

    if coordinates is None:
        coordinates = [base + suf for base in CLINICAL_JOINTS for suf in ("_r", "_l")]

    ref_cols = set(reference.df.columns)
    oth_cols = set(other.df.columns)
    shared = [c for c in coordinates if c in ref_cols and c in oth_cols]

    ref_series = {c: _resample(t_ref, reference.df[c].to_numpy(), grid) for c in shared}
    oth_series = {c: _resample(t_oth, other.df[c].to_numpy(), grid) for c in shared}

    match = _lr_match(shared, ref_series, oth_series) if lr_match else {c: c for c in shared}

    # matched other series, keyed by the reference coord
    matched_oth = {c: oth_series[match[c]] for c in shared}
    max_lag = max(1, min(int(n * max_lag_fraction), int((max_lag_ms / 1000.0) / dt)))
    lag = _estimate_global_lag(ref_series, matched_oth, max_lag) if align_time else 0
    trim = abs(lag)

    results: list[CoordComparison] = []
    for c in shared:
        r = ref_series[c]
        o = matched_oth[c]
        err = o - r
        mae = float(np.mean(np.abs(err)))
        rmse = float(np.sqrt(np.mean(err ** 2)))
        max_err = float(np.max(np.abs(err)))
        rom_ref = float(r.max() - r.min())
        rom_oth = float(o.max() - o.min())

        o_shift = np.roll(o, lag)
        if trim > 0:
            ra, oa = r[trim:n - trim], o_shift[trim:n - trim]
        else:
            ra, oa = r, o_shift
        if len(ra):
            aerr = oa - ra
            rmse_aligned = float(np.sqrt(np.mean(aerr ** 2)))
            bias = float(np.mean(aerr))
            rmse_residual = float(np.sqrt(np.mean((aerr - bias) ** 2)))
            pr = (float(np.corrcoef(ra, oa)[0, 1])
                  if ra.std() > 1e-9 and oa.std() > 1e-9 else float("nan"))
        else:
            rmse_aligned = bias = rmse_residual = pr = float("nan")

        results.append(CoordComparison(
            coord=c, mae=mae, rmse=rmse, max_err=max_err,
            rom_ref=rom_ref, rom_other=rom_oth, rom_diff=rom_oth - rom_ref,
            pearson_r=pr, n=len(grid),
            rmse_aligned=rmse_aligned, bias=bias, rmse_residual=rmse_residual,
            lr_swapped=(match[c] != c),
        ))

    if results:
        mean_rmse = float(np.mean([c.rmse for c in results]))
        mean_mae = float(np.mean([c.mae for c in results]))
        resid = [c.rmse_residual for c in results if not np.isnan(c.rmse_residual)]
        mean_resid = float(np.mean(resid)) if resid else float("nan")
    else:
        mean_rmse = mean_mae = mean_resid = float("nan")

    return MotionComparison(
        reference=ref_name, other=other_name, coords=results,
        mean_rmse=mean_rmse, mean_mae=mean_mae,
        lag_samples=lag, lag_ms=float(lag * dt * 1000.0),
        mean_rmse_residual=mean_resid,
        n_lr_swapped=sum(1 for c in results if c.lr_swapped),
    )
