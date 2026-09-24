"""Pluggable pose-estimation backends.

The whole point of this module is the promise you asked for: *swap the model
without touching the rest of the pipeline*. Everything downstream
(triangulation, OpenSim) consumes **OpenPose-format JSON**, so a 2D backend's
only contract is: given a video, write one JSON file per frame in that format.

To add a backend, subclass `Pose2DBackend`, implement `process_video`, and
register it with `@register_pose2d("your_name")`. It then shows up in the app's
backend dropdown automatically.
"""
from __future__ import annotations

import abc
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional


# --------------------------------------------------------------------------- #
# OpenPose JSON helpers (the interchange format between 2D and 3D stages)
# --------------------------------------------------------------------------- #
def write_openpose_json(out_file: Path, keypoints_2d: list[float], version: float = 1.3) -> None:
    """Write a single-person OpenPose-format keypoints file.

    `keypoints_2d` is a flat [x0, y0, c0, x1, y1, c1, ...] list already ordered
    for the target skeleton (e.g. BODY_25B). Confidence c in [0, 1].
    """
    doc = {
        "version": version,
        "people": [
            {
                "person_id": [-1],
                "pose_keypoints_2d": keypoints_2d,
                "face_keypoints_2d": [],
                "hand_left_keypoints_2d": [],
                "hand_right_keypoints_2d": [],
                "pose_keypoints_3d": [],
                "face_keypoints_3d": [],
                "hand_left_keypoints_3d": [],
                "hand_right_keypoints_3d": [],
            }
        ],
    }
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(doc, f)


# Progress callback: (current_frame, total_frames_or_None, message)
ProgressFn = Callable[[int, Optional[int], str], None]


@dataclass
class Pose2DResult:
    """Summary returned by a backend after processing one video."""
    camera_index: int
    frames_written: int
    output_dir: Path


# --------------------------------------------------------------------------- #
# Backend interface
# --------------------------------------------------------------------------- #
class Pose2DBackend(abc.ABC):
    """Base class for a 2D keypoint estimator.

    Subclasses estimate 2D keypoints from a video and emit OpenPose-format JSON,
    one file per frame, named `frame_00000.json`, `frame_00001.json`, ...
    """

    #: Human-readable name shown in the UI.
    display_name: str = "unnamed"
    #: Skeleton this backend natively produces (e.g. "BODY_25B").
    skeleton: str = "BODY_25B"
    #: True if the backend needs a GPU to be practical.
    requires_gpu: bool = False

    def __init__(self, **options):
        self.options = options

    def is_available(self) -> tuple[bool, str]:
        """Return (available, reason). Override to check for exes / imports.

        Default assumes available. GUI uses this to grey out backends whose
        dependencies are missing.
        """
        return True, ""

    @abc.abstractmethod
    def process_video(
        self,
        video_path: Path,
        output_dir: Path,
        camera_index: int,
        progress: Optional[ProgressFn] = None,
    ) -> Pose2DResult:
        """Estimate 2D keypoints for one camera's video.

        Args:
            video_path: input video file.
            output_dir: where to write `frame_NNNNN.json` (created if needed).
            camera_index: 1-based camera number (for logging/results).
            progress: optional callback (frame_idx, total, message).
        """
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
_POSE2D_REGISTRY: dict[str, type[Pose2DBackend]] = {}


def register_pose2d(name: str) -> Callable[[type[Pose2DBackend]], type[Pose2DBackend]]:
    """Class decorator registering a 2D backend under `name`."""
    def deco(cls: type[Pose2DBackend]) -> type[Pose2DBackend]:
        key = name.lower()
        if key in _POSE2D_REGISTRY:
            raise ValueError(f"pose2d backend {name!r} already registered")
        _POSE2D_REGISTRY[key] = cls
        return cls
    return deco


def get_pose2d_backend(name: str, **options) -> Pose2DBackend:
    key = name.lower()
    if key not in _POSE2D_REGISTRY:
        raise KeyError(
            f"unknown pose2d backend {name!r}; available: {sorted(_POSE2D_REGISTRY)}"
        )
    return _POSE2D_REGISTRY[key](**options)


def available_pose2d_backends() -> list[str]:
    return sorted(_POSE2D_REGISTRY)


def load_builtin_backends() -> None:
    """Import built-in backend modules so their @register runs.

    Imported lazily and defensively: a backend whose heavy dependency
    (mediapipe, etc.) is missing must not break the whole registry.
    """
    import importlib

    for mod in ("poseassess.plugins.mediapipe_backend",
                "poseassess.plugins.openpose_backend"):
        try:
            importlib.import_module(mod)
        except Exception as e:  # noqa: BLE001 - backend deps are optional
            import logging
            logging.getLogger(__name__).warning("backend %s unavailable: %s", mod, e)
