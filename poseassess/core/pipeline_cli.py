"""CLI entry point that runs pipeline stages in a child process.

Run as: ``python -m poseassess.core.pipeline_cli <project_root> <stage,stage,…>``

The GUI launches this as a subprocess so a long-running stage (2D pose, OpenSim
IK) can be interrupted: the GUI's Cancel button terminates the whole process,
which actually stops the native Pose2Sim work mid-stage — something a Python
cancel-flag checked between stages can't do.

Progress is streamed on stdout. Two machine-readable markers frame each stage so
the parent can report per-stage results:
  ``@@STAGE_START <name>`` and ``@@STAGE_RESULT <name> <OK|FAIL> <message>``,
then ``@@DONE`` at the end. Everything else on stdout is human log text.
"""
from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    if not argv:
        print("usage: pipeline_cli <project_root> [stage,stage,...] [--continue]")
        return 2
    project_root = argv[0]
    stages = argv[1].split(",") if len(argv) > 1 and not argv[1].startswith("-") else None
    stop_on_error = "--continue" not in argv

    # headless matplotlib so filtering/sync figures never try to open a window
    try:
        import os
        os.environ["MPLBACKEND"] = "Agg"
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        plt.show = lambda *a, **k: None
        plt.pause = lambda *a, **k: None
    except Exception:  # noqa: BLE001
        pass

    from poseassess.core.project import Project
    from poseassess.core.pipeline import Pipeline, StageResult

    project = Project.load(project_root)
    pipe = Pipeline(project, progress=lambda m: print(m, flush=True))
    stages = stages or list(Pipeline.STAGES)

    for name in stages:
        print(f"@@STAGE_START {name}", flush=True)
        try:
            res = getattr(pipe, name)()
        except Exception as e:  # noqa: BLE001
            print(f"@@STAGE_RESULT {name} FAIL {type(e).__name__}: {e}", flush=True)
            if stop_on_error:
                break
            continue
        status = "OK" if res.ok else "FAIL"
        print(f"@@STAGE_RESULT {name} {status} {res.message}", flush=True)
        if not res.ok and stop_on_error:
            break
    print("@@DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
