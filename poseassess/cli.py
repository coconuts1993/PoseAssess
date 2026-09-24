"""PoseAssess command-line interface (headless core).

Usage:
    python -m poseassess.cli init   <workspace> [--cameras N] [--fps R] [--backend NAME]
    python -m poseassess.cli info    <workspace>
    python -m poseassess.cli config  <workspace>          # (re)generate Config.toml
    python -m poseassess.cli calib   <workspace>          # run calibration stage
    python -m poseassess.cli run     <workspace> [--from STAGE] [--only STAGE ...] [--continue]
    python -m poseassess.cli backends                     # list pluggable 2D backends
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from poseassess.core.project import Project, ProjectConfig
from poseassess.plugins import base


def _log(msg: str) -> None:
    print(msg, flush=True)


def cmd_init(args) -> int:
    cfg = ProjectConfig(
        name=args.name or Path(args.workspace).name,
        num_cameras=args.cameras,
        frame_rate=args.fps,
        pose2d_backend=args.backend,
    )
    proj = Project(args.workspace, cfg).create()
    from poseassess.core.config_gen import generate_config
    generate_config(proj)
    _log(f"created project: {proj}")
    _log(f"  drop trial videos as {proj.videos_dir}\\cam01.mp4 ... cam{cfg.num_cameras:02d}.mp4")
    _log(f"  intrinsic frames  -> {proj.intrinsic_frames_dir}\\camNN\\")
    _log(f"  extrinsic frames  -> {proj.extrinsic_frames_dir}\\camNN\\")
    return 0


def cmd_info(args) -> int:
    proj = Project.load(args.workspace)
    c = proj.config
    _log(f"Project: {c.name}")
    _log(f"  root:      {proj.root}")
    _log(f"  cameras:   {c.num_cameras} @ {c.frame_rate} Hz")
    _log(f"  2D backend:{c.pose2d_backend}  3D backend:{c.pose3d_backend}")
    _log(f"  model:     {c.pose_model}")
    _log(f"  board:     {c.checkerboard.corners_nb} inner corners, {c.checkerboard.square_size} mm")
    # readiness
    n_vid = len(list(proj.videos_dir.glob("*.mp4")))
    calib = "yes" if list(proj.calibration_dir.glob("Calib*.toml")) else "no"
    _log(f"  videos:    {n_vid} found | calibration: {calib}")
    return 0


def cmd_config(args) -> int:
    proj = Project.load(args.workspace)
    from poseassess.core.config_gen import generate_config
    out = generate_config(proj)
    _log(f"wrote {out}")
    return 0


def cmd_calib(args) -> int:
    proj = Project.load(args.workspace)
    from poseassess.core.pipeline import Pipeline
    res = Pipeline(proj, progress=_log).calibration()
    _log(f"calibration: {'OK' if res.ok else 'FAILED'} — {res.message}")
    return 0 if res.ok else 1


def cmd_run(args) -> int:
    proj = Project.load(args.workspace)
    from poseassess.core.pipeline import Pipeline
    pipe = Pipeline(proj, progress=_log)
    if args.only:
        stages = args.only
    elif args.start:
        i = Pipeline.STAGES.index(args.start)
        stages = Pipeline.STAGES[i:]
    else:
        stages = None
    results = pipe.run_all(stages=stages, stop_on_error=not args.keep_going)
    _log("\n=== summary ===")
    ok = True
    for r in results:
        _log(f"  {r.name:18s} {'ok' if r.ok else 'FAIL'}  {r.message}")
        ok = ok and r.ok
    return 0 if ok else 1


def cmd_backends(args) -> int:
    base.load_builtin_backends()
    _log("Registered 2D backends:")
    for name in base.available_pose2d_backends():
        be = base.get_pose2d_backend(name, openpose_dir="")
        avail, why = be.is_available()
        status = "available" if avail else f"unavailable ({why})"
        gpu = " [GPU]" if be.requires_gpu else ""
        _log(f"  {name:12s} {be.display_name}{gpu} — {status}")
    _log("  rtmpose      RTMPose via Pose2Sim built-in (default)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="poseassess", description="3D pose assessment (headless core)")
    sub = p.add_subparsers(dest="command", required=True)

    pi = sub.add_parser("init", help="create a new project workspace")
    pi.add_argument("workspace")
    pi.add_argument("--name", default="")
    pi.add_argument("--cameras", type=int, default=6)
    pi.add_argument("--fps", type=int, default=60)
    pi.add_argument("--backend", default="rtmpose")
    pi.set_defaults(func=cmd_init)

    pinfo = sub.add_parser("info", help="show project status")
    pinfo.add_argument("workspace")
    pinfo.set_defaults(func=cmd_info)

    pc = sub.add_parser("config", help="(re)generate Config.toml")
    pc.add_argument("workspace")
    pc.set_defaults(func=cmd_config)

    pcal = sub.add_parser("calib", help="run calibration stage")
    pcal.add_argument("workspace")
    pcal.set_defaults(func=cmd_calib)

    pr = sub.add_parser("run", help="run the pipeline")
    pr.add_argument("workspace")
    pr.add_argument("--from", dest="start", default=None, help="start from this stage")
    pr.add_argument("--only", nargs="+", default=None, help="run only these stages")
    pr.add_argument("--keep-going", action="store_true", help="don't stop on first failure")
    pr.set_defaults(func=cmd_run)

    pb = sub.add_parser("backends", help="list pluggable 2D backends")
    pb.set_defaults(func=cmd_backends)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
