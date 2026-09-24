"""OpenSim inverse kinematics: 3D marker trajectories (.trc) -> joint angles (.mot).

Port of the add-on's `OpenSimConfigFile` + `RunOpenSim`: fill the IK setup XML
template (marker file, time range read from the .trc header, model + results
paths) and run `opensim-cmd run-tool` on it. OpenSim itself stays an external
executable, so no Python binding is required.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

ProgressFn = Callable[[str], None]

# Placeholders in the IK setup template (kept identical to the original add-on).
PH_RESULTS = "####RESULTS_DIR####"
PH_TRC = "####TRC_FILE####"
PH_TIME_START = "####TIME_START####"
PH_TIME_END = "####TIME_END####"
PH_OSIM = "####OSIM_MODEL####"


def read_trc_time_range(trc_file: Path) -> tuple[str, str]:
    """Read first/last timestamps from a .trc file's data rows.

    TRC layout: 3 header rows, then a column header, then a blank/units row,
    then data rows where column 1 (index 1) is the time. This mirrors the
    original code's `Lines[5]` first-sample / last-line last-sample logic.
    """
    lines = Path(trc_file).read_text().splitlines()
    # Data starts at line index 5 in the standard Pose2Sim TRC output.
    data_lines = [ln for ln in lines[5:] if ln.strip() and "\t" in ln]
    if not data_lines:
        raise RuntimeError(f"no data rows found in {trc_file}")
    time_start = data_lines[0].split("\t")[1].strip()
    time_end = data_lines[-1].split("\t")[1].strip()
    return time_start, time_end


def build_ik_setup(
    template_file: Path,
    out_file: Path,
    results_dir: Path,
    trc_file: Path,
    osim_model: Path,
    time_start: str,
    time_end: str,
) -> Path:
    """Render the IK setup XML from the template by substituting placeholders."""
    text = Path(template_file).read_text()
    text = (
        text.replace(PH_RESULTS, str(results_dir))
        .replace(PH_TRC, str(trc_file))
        .replace(PH_TIME_START, time_start)
        .replace(PH_TIME_END, time_end)
        .replace(PH_OSIM, str(osim_model))
    )
    out_file = Path(out_file)
    out_file.write_text(text)
    return out_file


@dataclass
class IKResult:
    setup_file: Path
    motion_file: Path
    returncode: int


def run_ik(
    opensim_cmd: Path,
    setup_file: Path,
    results_dir: Path,
    motion_name: str = "result.mot",
    progress: Optional[ProgressFn] = None,
) -> IKResult:
    """Run `opensim-cmd run-tool <setup_file>`."""
    opensim_cmd = Path(opensim_cmd)
    if not opensim_cmd.exists():
        raise FileNotFoundError(f"opensim-cmd not found: {opensim_cmd}")

    cmd = f'"{opensim_cmd}" run-tool "{setup_file}"'
    if progress:
        progress(f"[opensim] {cmd}")
    proc = subprocess.run(cmd, shell=True, check=True)
    return IKResult(
        setup_file=Path(setup_file),
        motion_file=Path(results_dir) / motion_name,
        returncode=proc.returncode,
    )


def run_opensim_ik(project, template_file: Path, progress: Optional[ProgressFn] = None) -> IKResult:
    """End-to-end IK for a project: pick the filtered .trc, template, run.

    Requires `project.config.opensim_cmd` and `project.config.osim_model` set.
    """
    import glob

    cfg = project.config
    if not cfg.opensim_cmd:
        raise RuntimeError("project.config.opensim_cmd is not set")
    if not cfg.osim_model:
        raise RuntimeError("project.config.osim_model is not set")

    trc_candidates = sorted(glob.glob(str(project.pose3d_dir / "*.trc")))
    if not trc_candidates:
        raise FileNotFoundError(f"no .trc file in {project.pose3d_dir}")
    trc_file = Path(trc_candidates[-1])  # filtered output sorts last

    t0, t1 = read_trc_time_range(trc_file)
    setup = build_ik_setup(
        template_file=template_file,
        out_file=project.root / "IK_MARKER.xml",
        results_dir=project.opensim_dir,
        trc_file=trc_file,
        osim_model=Path(cfg.osim_model),
        time_start=t0,
        time_end=t1,
    )
    if progress:
        progress(f"[opensim] setup written: {setup} (t={t0}..{t1})")
    project.opensim_dir.mkdir(parents=True, exist_ok=True)
    return run_ik(Path(cfg.opensim_cmd), setup, project.opensim_dir, progress=progress)
