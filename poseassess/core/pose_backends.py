"""Catalog of selectable 2D pose backends.

Single source of truth shared by `config_gen` (how to configure Pose2Sim's pose
stage) and `pipeline` (whether to run Pose2Sim's built-in rtmlib estimator or an
external plugin that writes OpenPose JSON).

Two families:
  • native  — run through Pose2Sim's embedded rtmlib (`poseEstimation`); we just
              set `pose_model` + `mode`. RTMPose size sweep (t/m/x) on the SAME
              HALPE_26 skeleton is the fair, same-downstream benchmark; Whole_body
              (DWPose) is a different-skeleton option.
  • plugin  — an external estimator (MediaPipe, OpenPose) run by our plugin
              registry, emitting pose/camNN_json before the Pose2Sim stages.

`mode` may be a string ('lightweight'|'balanced'|'performance') or a dict of
explicit rtmlib model URLs (kept as a triple-quoted string in the TOML).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Union


@dataclass(frozen=True)
class PoseBackendSpec:
    name: str
    display: str
    native: bool                 # True -> Pose2Sim poseEstimation (rtmlib)
    pose_model: str = ""         # Pose2Sim [pose] pose_model (native only)
    mode: Union[str, None] = None  # Pose2Sim [pose] mode (native only)
    skeleton: str = "HALPE_26"   # skeleton the backend produces (for the record)
    requires_gpu: bool = False


# NOTE: RTMPose t/m/x all produce HALPE_26 -> identical downstream, so differences
# in joint angles are attributable purely to the 2D model. That is the clean
# benchmark axis.
POSE_BACKENDS: dict[str, PoseBackendSpec] = {
    "rtmpose_lite": PoseBackendSpec(
        "rtmpose_lite", "RTMPose-t · HALPE_26 (lightweight)", True,
        "Body_with_feet", "lightweight", "HALPE_26"),
    "rtmpose": PoseBackendSpec(
        "rtmpose", "RTMPose-m · HALPE_26 (balanced)", True,
        "Body_with_feet", "balanced", "HALPE_26"),
    "rtmpose_perf": PoseBackendSpec(
        "rtmpose_perf", "RTMPose-x · HALPE_26 (performance)", True,
        "Body_with_feet", "performance", "HALPE_26"),
    "wholebody": PoseBackendSpec(
        "wholebody", "DWPose · COCO_133 wholebody (balanced)", True,
        "Whole_body", "balanced", "COCO_133"),
    # --- external plugins (produce OpenPose JSON) ---
    "mediapipe": PoseBackendSpec(
        "mediapipe", "MediaPipe · BLAZEPOSE (CPU)", False, skeleton="BLAZEPOSE"),
    "openpose": PoseBackendSpec(
        "openpose", "OpenPose · BODY_25B (GPU)", False, skeleton="BODY_25B",
        requires_gpu=True),
}

DEFAULT_BACKEND = "rtmpose"


def get_backend(name: str) -> PoseBackendSpec:
    key = (name or "").lower()
    if key not in POSE_BACKENDS:
        raise KeyError(f"unknown pose backend {name!r}; "
                       f"available: {sorted(POSE_BACKENDS)}")
    return POSE_BACKENDS[key]


def is_native(name: str) -> bool:
    return get_backend(name).native


def native_backends() -> list[str]:
    return [n for n, s in POSE_BACKENDS.items() if s.native]


def all_backends() -> list[str]:
    return list(POSE_BACKENDS)
