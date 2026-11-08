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

# where openpose_setup.resolve_openpose_dir found the install
_PROJECT_SETTING = "project setting"
_REMEMBERED = "remembered on this computer"


def _remove_json(folder: Path) -> None:
    """Delete the *.json files (2D results) directly in `folder`, if it exists."""
    if not folder.is_dir():
        return
    for f in folder.glob("*.json"):
        try:
            f.unlink()
        except FileNotFoundError:
            pass
        except OSError as e:
            raise RuntimeError(f"cannot remove the earlier 2D result {f}: {e}") from e


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
            "openpose_dir": (self._openpose_dir() if backend == "openpose"
                             else self.project.config.openpose_dir),
        }
        if backend == "openpose":
            opts["net_resolution"] = self._openpose_net_resolution()
        be = base.get_pose2d_backend(backend, **opts)
        avail, why = be.is_available()
        if not avail:
            raise RuntimeError(f"backend {backend!r} unavailable: {why}")
        # every message of a throttled backend is worth logging; others: every 50th frame
        log_all = getattr(be, "progress_throttled", False)

        videos = sorted(self.project.videos_dir.glob("*.mp4")) + \
            sorted(self.project.videos_dir.glob("*.avi")) + \
            sorted(self.project.videos_dir.glob("*.mov"))
        if not videos:
            raise FileNotFoundError(f"no videos in {self.project.videos_dir}")

        # Pose2Sim reads every *.json of every camNN_json folder: clear all cameras
        # first, so a camera after a failed or cancelled one keeps no older results
        for i in range(1, len(videos) + 1):
            _remove_json(self.project.pose2d_cam_dir(i))

        for i, vid in enumerate(videos, start=1):
            out_dir = self.project.pose2d_cam_dir(i)
            self._log(f"[pose:{backend}] cam{i:02d} <- {vid.name}")
            be.process_video(
                vid, out_dir, camera_index=i,
                progress=lambda f, t, m: self._log(f"  {m}") if log_all or f % 50 == 0 else None,
            )

    def _openpose_net_resolution(self) -> str:
        """The project's OpenPose --net_resolution (1. Project), else the default."""
        from poseassess.core import openpose_setup as ops
        raw = getattr(self.project.config, "openpose_net_resolution", "") or ""
        value = ops.normalize_net_resolution(raw)
        if value is None:
            if str(raw).strip():
                self._log(f"[pose:openpose] network resolution {raw!r} is not usable "
                          f"(-1xH, H a multiple of 16): using {ops.NET_RESOLUTION}")
            value = ops.NET_RESOLUTION
        return value

    def _openpose_dir(self) -> str:
        """The local OpenPose install to run: the project setting, the folder remembered
        on this computer or a usual place (logged); else the project setting, and the
        backend's availability check says where it looked."""
        from poseassess.core.openpose_setup import check_install, resolve_openpose_dir
        setting = self.project.config.openpose_dir or ""
        root, how = resolve_openpose_dir(setting)
        if root is None:
            return setting
        if setting.strip().strip('"').strip() and how != _PROJECT_SETTING:
            errors = check_install(setting).errors
            self._log(f"[pose:openpose] warning: the OpenPose folder set on 1. Project "
                      f"({setting}) is not usable: {errors[0] if errors else 'incomplete'}")
        if how not in (_PROJECT_SETTING, _REMEMBERED):
            how = "found in a usual place"
        self._log(f"[pose:openpose] OpenPose at {root} ({how})")
        for w in check_install(root).warnings:
            self._log(f"[pose:openpose] warning: {w}")
        return str(root)

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
