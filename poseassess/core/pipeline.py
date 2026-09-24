"""Pipeline orchestrator: drive the Pose2Sim 0.10 stages for a project.

Stages, in order:
    calibration -> pose (2D) -> synchronization -> personAssociation
    -> triangulation -> filtering -> markerAugmentation -> kinematics

The 2D *pose* stage is where the pluggable backends plug in:
  - backend 'rtmpose'  -> Pose2Sim's built-in `poseEstimation()` (RTMPose/rtmlib)
  - any other backend  -> run the registered plugin on each video to emit
                          OpenPose-format JSON into pose/camNN_json/, which the
                          downstream Pose2Sim stages then consume unchanged.

Each stage can be run individually (GUI "run this step" buttons) or via
`run_all`. Stages are thin wrappers so the heavy lifting stays in Pose2Sim.
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

ProgressFn = Callable[[str], None]


@contextmanager
def _in_dir(path: Path):
    """Temporarily chdir into `path` (Pose2Sim resolves relative paths off CWD)."""
    prev = os.getcwd()
    os.chdir(str(path))
    try:
        yield
    finally:
        os.chdir(prev)


@dataclass
class StageResult:
    name: str
    ok: bool
    message: str = ""


class Pipeline:
    """Runs Pose2Sim stages for a `Project`, honoring the chosen 2D backend."""

    #: full stage order
    STAGES = [
        "calibration", "pose", "synchronization", "personAssociation",
        "triangulation", "filtering", "markerAugmentation", "kinematics",
    ]

    def __init__(self, project, progress: Optional[ProgressFn] = None):
        self.project = project
        self.progress = progress or (lambda m: None)

    def _log(self, msg: str) -> None:
        self.progress(msg)

    def _config_path(self) -> str:
        return str(self.project.config_file)

    # ---- individual stages ------------------------------------------------ #
    def calibration(self) -> StageResult:
        """Produce calibration/Calib.toml.

        If our own OpenCV calibration already wrote intri/extri.yml, Pose2Sim's
        'convert' (from easymocap) turns them into Calib.toml. If Calib.toml is
        already present and valid we skip.
        """
        from Pose2Sim import Pose2Sim as ps
        self._log("[calibration] running")
        with _in_dir(self.project.root):
            ps.calibration(self._config_path())
        ok = self.project.calib_toml.exists() or any(self.project.calibration_dir.glob("Calib*.toml"))
        return StageResult("calibration", ok, "Calib.toml ready" if ok else "no Calib.toml produced")

    def pose(self) -> StageResult:
        """2D keypoint estimation, dispatched to the chosen backend."""
        from poseassess.core.pose_backends import get_backend
        backend = self.project.config.pose2d_backend
        spec = get_backend(backend)
        if spec.native:
            from Pose2Sim import Pose2Sim as ps
            self._log(f"[pose] Pose2Sim rtmlib: {spec.display}")
            with _in_dir(self.project.root):
                ps.poseEstimation(self._config_path())
        else:
            self._run_plugin_pose(spec.name)
        produced = list(self.project.pose2d_dir.glob("*_json"))
        return StageResult("pose", bool(produced), f"{len(produced)} camera json folders")

    def _run_plugin_pose(self, backend: str) -> None:
        """Run an external 2D backend (mediapipe/openpose/custom) per camera."""
        from poseassess.plugins import base
        base.load_builtin_backends()
        opts = {
            "openpose_dir": self.project.config.openpose_dir,
        }
        be = base.get_pose2d_backend(backend, **opts)
        avail, why = be.is_available()
        if not avail:
            raise RuntimeError(f"backend {backend!r} unavailable: {why}")

        videos = sorted(self.project.videos_dir.glob("*.mp4")) + \
            sorted(self.project.videos_dir.glob("*.avi")) + \
            sorted(self.project.videos_dir.glob("*.mov"))
        if not videos:
            raise FileNotFoundError(f"no videos in {self.project.videos_dir}")

        for i, vid in enumerate(videos, start=1):
            out_dir = self.project.pose2d_cam_dir(i)
            self._log(f"[pose:{backend}] cam{i:02d} <- {vid.name}")
            be.process_video(
                vid, out_dir, camera_index=i,
                progress=lambda f, t, m: self._log(f"  {m}") if f % 50 == 0 else None,
            )

    def _pose2sim_stage(self, name: str) -> StageResult:
        from Pose2Sim import Pose2Sim as ps
        fn = getattr(ps, name)
        self._log(f"[{name}] running")
        with _in_dir(self.project.root):
            fn(self._config_path())
        return StageResult(name, True)

    def synchronization(self) -> StageResult:
        return self._pose2sim_stage("synchronization")

    def personAssociation(self) -> StageResult:
        return self._pose2sim_stage("personAssociation")

    def triangulation(self) -> StageResult:
        res = self._pose2sim_stage("triangulation")
        res.ok = any(self.project.pose3d_dir.glob("*.trc"))
        return res

    def filtering(self) -> StageResult:
        return self._pose2sim_stage("filtering")

    def markerAugmentation(self) -> StageResult:
        return self._pose2sim_stage("markerAugmentation")

    def kinematics(self) -> StageResult:
        res = self._pose2sim_stage("kinematics")
        res.ok = any(self.project.kinematics_dir.glob("*.mot"))
        return res

    # ---- full run --------------------------------------------------------- #
    def run_all(self, stages: Optional[list[str]] = None,
                stop_on_error: bool = True) -> list[StageResult]:
        stages = stages or self.STAGES
        results: list[StageResult] = []
        for name in stages:
            try:
                res = getattr(self, name)()
            except Exception as e:  # noqa: BLE001 - report, optionally continue
                res = StageResult(name, False, f"{type(e).__name__}: {e}")
                self._log(f"[{name}] FAILED: {res.message}")
                results.append(res)
                if stop_on_error:
                    break
                continue
            results.append(res)
            self._log(f"[{name}] {'ok' if res.ok else 'incomplete'} — {res.message}")
        return results
