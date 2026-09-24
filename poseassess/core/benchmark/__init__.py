"""Backend benchmark + ground-truth validation.

- compare.py   : biomechanical error metrics between two motions (with honest
                 lag/bias/residual decomposition).
- runner.py    : run one trial through multiple 2D backends and compare.
- opencap_dataset.py / validation.py : validate our pipeline against the OpenCap
                 marker ground truth across many subjects/tasks.
"""
from .compare import compare_motions, MotionComparison, CoordComparison
from .runner import run_backends, compare_runs, BenchmarkResult, BackendRun
from .opencap_dataset import OpenCapZip, assemble_trial_project, TrialSpec
from .validation import (
    validate_trial, run_validation_batch, TrialResult,
    aggregate_per_joint, format_summary,
)

__all__ = [
    "compare_motions", "MotionComparison", "CoordComparison",
    "run_backends", "compare_runs", "BenchmarkResult", "BackendRun",
    "OpenCapZip", "assemble_trial_project", "TrialSpec",
    "validate_trial", "run_validation_batch", "TrialResult",
    "aggregate_per_joint", "format_summary",
]
