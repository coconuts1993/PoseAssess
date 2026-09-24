"""Clinical rehabilitation metrics from OpenSim joint angles.

This is the layer that turns raw kinematics into numbers a clinician reads:
range of motion (ROM), peak angles, and left/right symmetry — the gap that
Pose2Sim / caliscope / OpenCap leave open. Everything is derived from a
`MotionData` (see mot_io); nothing here depends on how the angles were produced.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np

from .mot_io import MotionData


# Curated, clinically meaningful coordinates with human-readable names.
# Translations (tx/ty/tz), coupling coords (*_beta) and per-vertebra spine
# coordinates are intentionally excluded from the default report.
CLINICAL_JOINTS: dict[str, str] = {
    # lower limb (bilateral base names, without _r/_l)
    "hip_flexion": "Hip flexion/extension",
    "hip_adduction": "Hip ab/adduction",
    "hip_rotation": "Hip rotation",
    "knee_angle": "Knee flexion",
    "ankle_angle": "Ankle dorsi/plantarflexion",
    "subtalar_angle": "Subtalar inv/eversion",
    # upper limb (bilateral)
    "arm_flex": "Shoulder flexion",
    "arm_add": "Shoulder ab/adduction",
    "arm_rot": "Shoulder rotation",
    "elbow_flex": "Elbow flexion",
    "pro_sup": "Forearm pro/supination",
    "wrist_flex": "Wrist flexion",
    "wrist_dev": "Wrist deviation",
}

# Unilateral (midline) coordinates worth reporting.
AXIAL_JOINTS: dict[str, str] = {
    "pelvis_tilt": "Pelvic tilt",
    "pelvis_list": "Pelvic obliquity",
    "pelvis_rotation": "Pelvic rotation",
    "neck_flexion": "Neck flexion",
    "neck_bending": "Neck lateral bending",
    "neck_rotation": "Neck rotation",
}


@dataclass
class CoordMetric:
    coord: str
    min: float
    max: float
    rom: float
    mean: float
    peak_abs: float

    @classmethod
    def from_series(cls, coord: str, values: np.ndarray) -> "CoordMetric":
        v = np.asarray(values, dtype=float)
        v = v[~np.isnan(v)]
        if v.size == 0:
            return cls(coord, 0, 0, 0, 0, 0)
        vmin, vmax = float(v.min()), float(v.max())
        return cls(coord, vmin, vmax, vmax - vmin, float(v.mean()),
                   float(np.abs(v).max()))


@dataclass
class BilateralMetric:
    joint: str            # base name, e.g. "knee_angle"
    label: str            # human-readable
    right: Optional[CoordMetric]
    left: Optional[CoordMetric]
    rom_symmetry_index: Optional[float]   # % — 0 is perfectly symmetric
    rom_diff: Optional[float]             # deg, right - left

    @property
    def flagged(self) -> bool:
        """True if asymmetry exceeds a common clinical concern threshold (10%)."""
        return self.rom_symmetry_index is not None and self.rom_symmetry_index > 10.0


def _symmetry_index(rom_r: float, rom_l: float) -> Optional[float]:
    """ROM symmetry index (%). |R-L| / (0.5(R+L)) * 100. 0 = symmetric.

    Robust choice: ROM is always non-negative, so no sign cancellation.
    """
    denom = 0.5 * (abs(rom_r) + abs(rom_l))
    if denom < 1e-6:
        return None
    return abs(rom_r - rom_l) / denom * 100.0


@dataclass
class AssessmentReport:
    source: str
    duration_s: float
    n_frames: int
    in_degrees: bool
    bilateral: list[BilateralMetric]
    axial: list[CoordMetric]

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "duration_s": self.duration_s,
            "n_frames": self.n_frames,
            "in_degrees": self.in_degrees,
            "bilateral": [
                {
                    "joint": b.joint, "label": b.label,
                    "right": asdict(b.right) if b.right else None,
                    "left": asdict(b.left) if b.left else None,
                    "rom_symmetry_index": b.rom_symmetry_index,
                    "rom_diff": b.rom_diff,
                    "flagged": b.flagged,
                }
                for b in self.bilateral
            ],
            "axial": [asdict(a) for a in self.axial],
        }


def analyze(motion: MotionData) -> AssessmentReport:
    """Compute the clinical assessment report from a loaded motion."""
    df = motion.df
    cols = set(df.columns)

    bilateral: list[BilateralMetric] = []
    for base, label in CLINICAL_JOINTS.items():
        r_col, l_col = f"{base}_r", f"{base}_l"
        r = CoordMetric.from_series(r_col, df[r_col].to_numpy()) if r_col in cols else None
        l = CoordMetric.from_series(l_col, df[l_col].to_numpy()) if l_col in cols else None
        if r is None and l is None:
            continue
        si = _symmetry_index(r.rom, l.rom) if (r and l) else None
        diff = (r.rom - l.rom) if (r and l) else None
        bilateral.append(BilateralMetric(base, label, r, l, si, diff))

    axial: list[CoordMetric] = []
    for coord, label in AXIAL_JOINTS.items():
        if coord in cols:
            m = CoordMetric.from_series(coord, df[coord].to_numpy())
            m.coord = label  # display label
            axial.append(m)

    return AssessmentReport(
        source=str(motion.source),
        duration_s=round(motion.duration, 3),
        n_frames=motion.n_frames,
        in_degrees=motion.in_degrees,
        bilateral=bilateral,
        axial=axial,
    )
