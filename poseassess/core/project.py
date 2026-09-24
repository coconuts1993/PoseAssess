"""Project / workspace model for PoseAssess.

A *project* is a single capture session for one subject: a workspace folder
holding calibration data, the multi-camera trial videos, and every
intermediate + final artifact of the pipeline
(2D keypoints -> 3D points -> OpenSim joint angles).

This module has NO heavy dependencies (only the standard library + tomli_w),
so it can be imported and tested without OpenCV / MediaPipe / Pose2Sim
installed. All pipeline stages read their paths from a `Project` instance.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

# tomllib is stdlib on 3.11+; fall back to the `tomli` backport otherwise.
if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib

try:
    import tomli_w
except ImportError:  # pragma: no cover - writing needs the dep, reading does not
    tomli_w = None


PROJECT_FILE = "project.toml"
CONFIG_FILE = "Config.toml"  # Pose2Sim config, kept in the workspace root


# --------------------------------------------------------------------------- #
# Configuration dataclasses
# --------------------------------------------------------------------------- #
@dataclass
class CheckerboardConfig:
    """Chessboard used for camera calibration.

    `corners_nb` is [rows, cols] of *inner* corners (Pose2Sim / OpenCV convention:
    for an 8x6 squares board that is [7, 5]). `square_size` in millimetres.
    """
    corners_nb: list[int] = field(default_factory=lambda: [4, 3])
    square_size: float = 135.0  # mm
    # Flip the world Z axis (negates the board-X object coordinate, like the old
    # add-on's `invert_extri_cal`). Use when the calibrated world comes out with Z
    # pointing DOWN, so the reconstructed subject ends up upside-down.
    invert_z: bool = False


@dataclass
class ProjectConfig:
    """Everything the app needs to know about a capture session.

    Serialized to `project.toml` in the workspace root. Kept deliberately flat
    and human-editable.
    """
    name: str = "untitled"
    num_cameras: int = 6
    frame_rate: int = 60           # Hz
    resolution: int = 1080         # vertical px, informational

    # Pluggable backends, resolved against the plugin registry by name.
    pose2d_backend: str = "mediapipe"
    pose3d_backend: str = "pose2sim"

    # Skeleton / model the whole chain is wired for.
    pose_model: str = "BODY_25B"

    checkerboard: CheckerboardConfig = field(default_factory=CheckerboardConfig)

    # Optional external tool paths (filled in by the user in settings).
    openpose_dir: str = ""         # folder containing OpenPoseDemo.exe
    opensim_cmd: str = ""          # path to opensim-cmd.exe
    osim_model: str = ""           # path to the .osim model file

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ProjectConfig":
        d = dict(d)
        cb = d.pop("checkerboard", None)
        cfg = cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
        if isinstance(cb, dict):
            cfg.checkerboard = CheckerboardConfig(
                corners_nb=cb.get("corners_nb", [4, 3]),
                square_size=cb.get("square_size", 135.0),
                invert_z=bool(cb.get("invert_z", False)),
            )
        return cfg


# --------------------------------------------------------------------------- #
# Project (workspace layout + lifecycle)
# --------------------------------------------------------------------------- #
class Project:
    """Owns a workspace folder and exposes every path the pipeline needs.

    Layout follows the Pose2Sim 0.10 convention so its pipeline functions work
    when run with the project root as CWD::

        <root>/
        ├── project.toml                    app config (this module)
        ├── Config.toml                     Pose2Sim config (generated)
        ├── calibration/
        │   ├── intrinsics/camNN/           frames for intrinsic calibration
        │   ├── extrinsics/camNN/           extrinsic frame + clicked points.json
        │   ├── intri.yml / extri.yml       our OpenCV calibration (easymocap fmt)
        │   └── Calib.toml                  Pose2Sim calibration (converted)
        ├── videos/                         trial videos: cam01.mp4 ... camNN.mp4
        ├── pose/camNN_json/                2D keypoints, OpenPose json format
        ├── pose-3d/                        triangulated + filtered 3D (.trc/.c3d)
        ├── kinematics/                     OpenSim scaled model + IK (.mot)
        └── logs/
    """

    def __init__(self, root: str | Path, config: Optional[ProjectConfig] = None):
        self.root = Path(root).resolve()
        self.config = config or ProjectConfig()

    # ---- path properties ------------------------------------------------- #
    @property
    def project_file(self) -> Path:
        return self.root / PROJECT_FILE

    @property
    def config_file(self) -> Path:
        return self.root / CONFIG_FILE

    @property
    def calibration_dir(self) -> Path:
        return self.root / "calibration"

    @property
    def intrinsic_frames_dir(self) -> Path:
        """Pose2Sim reads intrinsic checkerboard frames from here (per camera)."""
        return self.calibration_dir / "intrinsics"

    @property
    def extrinsic_frames_dir(self) -> Path:
        """Extrinsic scene frame + our clicked points.json per camera."""
        return self.calibration_dir / "extrinsics"

    @property
    def intri_yml(self) -> Path:
        return self.calibration_dir / "intri.yml"

    @property
    def extri_yml(self) -> Path:
        return self.calibration_dir / "extri.yml"

    @property
    def calib_toml(self) -> Path:
        return self.calibration_dir / "Calib.toml"

    @property
    def videos_dir(self) -> Path:
        return self.root / "videos"

    @property
    def pose2d_dir(self) -> Path:
        return self.root / "pose"

    @property
    def pose3d_dir(self) -> Path:
        return self.root / "pose-3d"

    @property
    def kinematics_dir(self) -> Path:
        return self.root / "kinematics"

    @property
    def logs_dir(self) -> Path:
        return self.root / "logs"

    # ---- per-camera helpers ---------------------------------------------- #
    def cam_name(self, index: int) -> str:
        """1-based camera folder name, e.g. index 1 -> 'cam01'."""
        return f"cam{index:02d}"

    def video_file(self, index: int, ext: str = "mp4") -> Path:
        return self.videos_dir / f"{self.cam_name(index)}.{ext}"

    def intrinsic_cam_dir(self, index: int) -> Path:
        return self.intrinsic_frames_dir / self.cam_name(index)

    def extrinsic_cam_dir(self, index: int) -> Path:
        return self.extrinsic_frames_dir / self.cam_name(index)

    def extrinsic_points_file(self, index: int) -> Path:
        return self.extrinsic_cam_dir(index) / "points.json"

    def pose2d_cam_dir(self, index: int) -> Path:
        """OpenPose-style json output dir for camera `index` (Pose2Sim: pose/camNN_json/)."""
        return self.pose2d_dir / f"{self.cam_name(index)}_json"

    def all_dirs(self) -> list[Path]:
        return [
            self.calibration_dir,
            self.intrinsic_frames_dir,
            self.extrinsic_frames_dir,
            self.videos_dir,
            self.pose2d_dir,
            self.pose3d_dir,
            self.kinematics_dir,
            self.logs_dir,
        ]

    # ---- lifecycle ------------------------------------------------------- #
    def create(self, exist_ok: bool = True) -> "Project":
        """Create the workspace folder tree and write project.toml."""
        self.root.mkdir(parents=True, exist_ok=exist_ok)
        for d in self.all_dirs():
            d.mkdir(parents=True, exist_ok=True)
        # Per-camera calibration folders.
        for i in range(1, self.config.num_cameras + 1):
            self.intrinsic_cam_dir(i).mkdir(parents=True, exist_ok=True)
            self.extrinsic_cam_dir(i).mkdir(parents=True, exist_ok=True)
        self.save()
        return self

    def save(self) -> None:
        """Persist project.toml."""
        if tomli_w is None:
            raise RuntimeError("tomli-w is required to save projects (pip install tomli-w)")
        data = self.config.to_dict()
        with open(self.project_file, "wb") as f:
            tomli_w.dump(data, f)

    @classmethod
    def load(cls, root: str | Path) -> "Project":
        """Load an existing workspace from its project.toml."""
        root = Path(root)
        pf = root / PROJECT_FILE
        if not pf.exists():
            raise FileNotFoundError(f"No {PROJECT_FILE} found in {root}")
        with open(pf, "rb") as f:
            data = tomllib.load(f)
        return cls(root, ProjectConfig.from_dict(data))

    @classmethod
    def is_project(cls, root: str | Path) -> bool:
        return (Path(root) / PROJECT_FILE).exists()

    def __repr__(self) -> str:
        return f"<Project {self.config.name!r} cams={self.config.num_cameras} root={self.root}>"
