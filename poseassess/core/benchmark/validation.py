"""Batch validation against the OpenCap marker ground truth.

For each (subject, task): assemble a project from the zip, run our pipeline
(pose -> ... -> kinematics), and compare the resulting joint angles to the
marker-based OpenSim IK ground truth with the honest error decomposition from
`compare.py`. Results are written per-trial as JSON (so the batch is resumable)
and aggregated into per-joint mean +/- std tables.
"""
from __future__ import annotations

import json
import time
import traceback
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Iterable, Optional

import numpy as np

from ..project import Project
from ..pipeline import Pipeline
from ..assessment.mot_io import read_mot
from .compare import compare_motions, MotionComparison
from .opencap_dataset import OpenCapZip, assemble_trial_project

# Pipeline stages for a pre-calibrated, pre-synced trial.
RUN_STAGES = ["pose", "personAssociation", "triangulation",
              "filtering", "markerAugmentation", "kinematics"]

ProgressFn = Callable[[str], None]


@dataclass
class TrialResult:
    subject: str
    task: str
    ok: bool
    message: str
    seconds: float
    lag_ms: float
    mean_rmse: float           # raw
    mean_rmse_residual: float  # tracking error
    n_lr_swapped: int = 0      # bilateral joints matched to contralateral GT side
    comparison: Optional[dict] = None  # full MotionComparison.to_dict()

    def to_dict(self) -> dict:
        return asdict(self)


def _latest_mot(project: Project) -> Optional[Path]:
    mots = sorted(project.kinematics_dir.glob("*.mot"))
    return mots[-1] if mots else None


def validate_trial(
    project: Project,
    gt_mot: Path,
    subject: str,
    task: str,
    progress: Optional[ProgressFn] = None,
    lr_match: bool = True,
    motions_dir: Optional[Path] = None,
) -> TrialResult:
    """Run the pipeline on an assembled project and compare vs ground truth.

    `lr_match` enables left/right-aware joint matching (records `n_lr_swapped`);
    harmless for symmetric tasks, essential for gait. If `motions_dir` is given,
    the output .mot and the GT .mot are archived there so the comparison can be
    re-run later with a different method without re-running the pipeline.
    """
    import shutil

    log = progress or (lambda m: None)
    t0 = time.time()
    pipe = Pipeline(project, progress=log)
    for stage in RUN_STAGES:
        res = getattr(pipe, stage)()
        if not res.ok:
            return TrialResult(subject, task, False, f"stage {stage} failed: {res.message}",
                               time.time() - t0, 0.0, float("nan"), float("nan"))

    ours_path = _latest_mot(project)
    if ours_path is None:
        return TrialResult(subject, task, False, "no output .mot", time.time() - t0,
                           0.0, float("nan"), float("nan"))

    if motions_dir is not None:
        motions_dir = Path(motions_dir); motions_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(ours_path, motions_dir / f"{subject}_{task}_ours.mot")
        shutil.copy(gt_mot, motions_dir / f"{subject}_{task}_GT.mot")

    gt = read_mot(gt_mot)
    ours = read_mot(ours_path)
    cmp = compare_motions(gt, ours, ref_name=f"{subject}_{task}_GT", other_name="ours",
                          lr_match=lr_match)
    return TrialResult(
        subject=subject, task=task, ok=True, message="ok",
        seconds=time.time() - t0, lag_ms=cmp.lag_ms,
        mean_rmse=cmp.mean_rmse, mean_rmse_residual=cmp.mean_rmse_residual,
        n_lr_swapped=cmp.n_lr_swapped, comparison=cmp.to_dict(),
    )


def run_validation_batch(
    zip_path: str | Path,
    specs: Iterable[tuple[str, str]],
    out_dir: str | Path,
    work_root: str | Path,
    resume: bool = True,
    keep_projects: bool = False,
    lr_match: bool = True,
    progress: Optional[ProgressFn] = None,
) -> list[TrialResult]:
    """Validate many (subject, task) trials. Resumable via per-trial JSON.

    Args:
        specs: iterable of (subject, task).
        out_dir: where per-trial result JSONs are written.
        work_root: scratch dir where each trial's project is assembled.
        resume: skip trials whose result JSON already exists.
        keep_projects: keep the assembled project folders (else delete after).
    """
    import shutil

    log = progress or print
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    work_root = Path(work_root)
    zip_obj = OpenCapZip(zip_path)

    results: list[TrialResult] = []
    specs = list(specs)
    for i, (subject, task) in enumerate(specs, 1):
        rid = f"{subject}_{task}"
        rj = out_dir / f"{rid}.json"
        if resume and rj.exists():
            log(f"[{i}/{len(specs)}] {rid}: cached, skipping")
            results.append(TrialResult(**{k: v for k, v in json.loads(rj.read_text()).items()}))
            continue

        log(f"[{i}/{len(specs)}] {rid}: assembling…")
        proj_root = work_root / rid
        try:
            proj, gt = assemble_trial_project(zip_obj, subject, task, proj_root)
            log(f"[{i}/{len(specs)}] {rid}: running pipeline…")
            tr = validate_trial(proj, gt, subject, task, progress=lambda m: None,
                                lr_match=lr_match, motions_dir=out_dir / "motions")
        except Exception as e:  # noqa: BLE001
            tr = TrialResult(subject, task, False, f"{type(e).__name__}: {e}",
                             0.0, 0.0, float("nan"), float("nan"))
            log(f"[{i}/{len(specs)}] {rid}: ERROR {tr.message}\n{traceback.format_exc()}")

        rj.write_text(json.dumps(tr.to_dict(), indent=2))
        results.append(tr)
        status = "ok" if tr.ok else "FAIL"
        log(f"[{i}/{len(specs)}] {rid}: {status} "
            f"resid={tr.mean_rmse_residual:.2f}° ({tr.seconds:.0f}s)")

        if not keep_projects and proj_root.exists():
            shutil.rmtree(proj_root, ignore_errors=True)

    return results


def run_backend_benchmark_batch(
    zip_path: str | Path,
    trials: Iterable[tuple[str, str]],
    backends: list[str],
    out_dir: str | Path,
    work_root: str | Path,
    resume: bool = True,
    lr_match: bool = True,
    progress: Optional[ProgressFn] = None,
) -> list[dict]:
    """Run each (subject, task) through several 2D backends, compare each vs GT.

    Fair design: one assembled project per trial (identical calibration, videos,
    downstream); only the 2D backend changes. Resumable — each (subject, task,
    backend) writes `<out>/<subject>_<task>__<backend>.json`. Projects are reused
    across backends and deleted once all backends for a trial are done.
    """
    import shutil
    from ..config_gen import generate_config

    log = progress or print
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    work_root = Path(work_root)
    zip_obj = OpenCapZip(zip_path)
    trials = list(trials)

    def clear_intermediates(proj):
        for d in (proj.pose2d_dir, proj.pose3d_dir, proj.kinematics_dir):
            d.mkdir(parents=True, exist_ok=True)
            for p in d.rglob("*"):
                if p.is_file():
                    try:
                        p.unlink()
                    except (PermissionError, OSError):
                        pass

    all_rows: list[dict] = []
    for ti, (subject, task) in enumerate(trials, 1):
        rid = f"{subject}_{task}"
        # which backends still need running?
        todo = [b for b in backends
                if not (resume and (out_dir / f"{rid}__{b}.json").exists())]
        for b in backends:
            jf = out_dir / f"{rid}__{b}.json"
            if jf.exists():
                all_rows.append(json.loads(jf.read_text()))
        if not todo:
            log(f"[{ti}/{len(trials)}] {rid}: all backends cached")
            continue

        proj_root = work_root / rid
        try:
            proj, gt = assemble_trial_project(zip_obj, subject, task, proj_root)
        except Exception as e:  # noqa: BLE001
            log(f"[{ti}/{len(trials)}] {rid}: assemble ERROR {e}")
            continue

        for b in todo:
            log(f"[{ti}/{len(trials)}] {rid} / {b}: running…")
            clear_intermediates(proj)
            proj.config.pose2d_backend = b
            proj.save(); generate_config(proj)
            try:
                tr = validate_trial(proj, gt, subject, task, progress=lambda m: None,
                                    lr_match=lr_match)
            except Exception as e:  # noqa: BLE001
                tr = TrialResult(subject, task, False, f"{type(e).__name__}: {e}",
                                 0.0, 0.0, float("nan"), float("nan"))
            row = tr.to_dict(); row["backend"] = b
            (out_dir / f"{rid}__{b}.json").write_text(json.dumps(row, indent=2))
            all_rows.append(row)
            log(f"[{ti}/{len(trials)}] {rid} / {b}: "
                f"{'ok' if tr.ok else 'FAIL'} resid={tr.mean_rmse_residual:.2f}° "
                f"({tr.seconds:.0f}s)")

        shutil.rmtree(proj_root, ignore_errors=True)

    return all_rows


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def aggregate_per_joint(results: list[TrialResult]) -> dict[str, dict]:
    """Per-joint mean +/- std of residual RMSE across all successful trials."""
    per_joint: dict[str, list[float]] = {}
    for r in results:
        if not r.ok or not r.comparison:
            continue
        for c in r.comparison["coords"]:
            v = c.get("rmse_residual")
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                per_joint.setdefault(c["coord"], []).append(v)
    agg = {}
    for joint, vals in per_joint.items():
        a = np.array(vals, float)
        agg[joint] = {"mean": float(a.mean()), "std": float(a.std()),
                      "n": len(vals), "min": float(a.min()), "max": float(a.max())}
    return dict(sorted(agg.items(), key=lambda kv: kv[1]["mean"]))


def format_summary(results: list[TrialResult]) -> str:
    ok = [r for r in results if r.ok]
    lines = [f"trials: {len(ok)}/{len(results)} ok"]
    if ok:
        resid = np.array([r.mean_rmse_residual for r in ok], float)
        raw = np.array([r.mean_rmse for r in ok], float)
        lines.append(f"overall residual RMSE: {np.nanmean(resid):.2f} +/- {np.nanstd(resid):.2f} deg")
        lines.append(f"overall raw RMSE:      {np.nanmean(raw):.2f} +/- {np.nanstd(raw):.2f} deg")
        swapped = [r for r in ok if r.n_lr_swapped > 0]
        if swapped:
            lines.append(f"L/R-swapped trials:    {len(swapped)}/{len(ok)} "
                         f"(whole-body left/right labeling flipped — a real limitation to report)")
    lines.append("")
    lines.append(f"{'joint':20s} {'resid mean':>10s} {'std':>7s} {'n':>4s}")
    lines.append("-" * 44)
    for joint, s in aggregate_per_joint(ok).items():
        lines.append(f"{joint:20s} {s['mean']:10.2f} {s['std']:7.2f} {s['n']:4d}")
    return "\n".join(lines)
