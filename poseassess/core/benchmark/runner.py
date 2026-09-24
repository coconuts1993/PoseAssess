"""Run one project through several 2D backends and collect the joint angles.

The whole comparison rests on a *fair* setup: identical calibration, identical
videos, identical downstream stages (triangulation -> filtering -> marker
augmentation -> OpenSim IK). Only the 2D backend changes. For each backend we
clear the intermediate folders, run pose->kinematics, and archive the resulting
.mot under `benchmark/<backend>/`.
"""
from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from ..pipeline import Pipeline

ProgressFn = Callable[[str], None]

# Stages to (re)run per backend. Calibration is done once and reused.
BENCH_STAGES = ["pose", "synchronization", "personAssociation",
                "triangulation", "filtering", "markerAugmentation", "kinematics"]


@dataclass
class BackendRun:
    backend: str
    ok: bool
    mot_file: Optional[Path]
    message: str = ""


@dataclass
class BenchmarkResult:
    runs: list[BackendRun] = field(default_factory=list)

    def successful(self) -> list[BackendRun]:
        return [r for r in self.runs if r.ok and r.mot_file]


def _clear_intermediates(project) -> None:
    # Delete files individually (not rmtree): OpenSim keeps opensim_logs.txt open
    # on Windows, so rmtree of the kinematics dir raises PermissionError.
    for d in (project.pose2d_dir, project.pose3d_dir, project.kinematics_dir):
        d.mkdir(parents=True, exist_ok=True)
        for p in d.rglob("*"):
            if p.is_file():
                try:
                    p.unlink()
                except (PermissionError, OSError):
                    pass


def run_backends(
    project,
    backends: list[str],
    progress: Optional[ProgressFn] = None,
    backend_options: Optional[dict[str, dict]] = None,
) -> BenchmarkResult:
    """Run `project` through each backend; archive each .mot to benchmark/<backend>/.

    Requires calibration to already be present (Calib.toml) and videos in place.
    The project's configured backend is restored afterwards.
    """
    from ..config_gen import generate_config

    def log(m: str) -> None:
        if progress:
            progress(m)

    backend_options = backend_options or {}
    bench_dir = project.root / "benchmark"
    bench_dir.mkdir(parents=True, exist_ok=True)

    original_backend = project.config.pose2d_backend
    result = BenchmarkResult()

    try:
        for backend in backends:
            log(f"===== backend: {backend} =====")
            _clear_intermediates(project)
            project.config.pose2d_backend = backend
            # apply any per-backend extra options (e.g. custom model paths)
            for k, v in backend_options.get(backend, {}).items():
                setattr(project.config, k, v)
            project.save()
            generate_config(project)

            pipe = Pipeline(project, progress=log)
            results = pipe.run_all(stages=BENCH_STAGES, stop_on_error=True)
            failed = [r for r in results if not r.ok]

            mots = sorted(project.kinematics_dir.glob("*.mot"))
            if failed or not mots:
                msg = failed[0].message if failed else "no .mot produced"
                log(f"[{backend}] FAILED: {msg}")
                result.runs.append(BackendRun(backend, False, None, msg))
                continue

            dst_dir = bench_dir / backend
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst = dst_dir / "result.mot"
            shutil.copy(mots[-1], dst)
            log(f"[{backend}] OK -> {dst}")
            result.runs.append(BackendRun(backend, True, dst, "ok"))
    finally:
        project.config.pose2d_backend = original_backend
        project.save()
        generate_config(project)

    return result


def compare_runs(result: BenchmarkResult, reference: Optional[str] = None):
    """Compare every successful backend against a reference backend's motion.

    Returns (reference_backend, list[MotionComparison]). If `reference` is None,
    the first successful backend is used as reference.
    """
    from ..assessment.mot_io import read_mot
    from .compare import compare_motions

    ok = result.successful()
    if len(ok) < 2:
        raise ValueError("need at least two successful backend runs to compare")

    ref_run = next((r for r in ok if r.backend == reference), ok[0])
    ref_motion = read_mot(ref_run.mot_file)

    comparisons = []
    for r in ok:
        if r.backend == ref_run.backend:
            continue
        m = read_mot(r.mot_file)
        comparisons.append(
            compare_motions(ref_motion, m, ref_name=ref_run.backend, other_name=r.backend)
        )
    return ref_run.backend, comparisons
