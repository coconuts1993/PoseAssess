"""Generate a Pose2Sim Config.toml from the app's ProjectConfig.

Rather than hand-write the (large, fast-moving) Pose2Sim config, we copy the
template that ships with the installed Pose2Sim version and override only the
fields the app manages. This keeps us forward-compatible: new Pose2Sim options
keep their sensible defaults, we just steer the ones that matter.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib

import tomli_w

from .pose_backends import get_backend


# Map our friendly backend + skeleton choices to Pose2Sim's pose_model values.
# Only RTMPose is embedded; others require the JSON to be produced externally
# (our plugin backends do that) and Pose2Sim then consumes pose/camNN_json/.
POSE_MODEL_MAP = {
    ("rtmpose", "HALPE_26"): "Body_with_feet",
    ("rtmpose", "COCO_133"): "Whole_body",
    ("rtmpose", "COCO_17"): "Body",
    ("mediapipe", "BLAZEPOSE"): "BLAZEPOSE",
    ("openpose", "BODY_25B"): "BODY_25B",
    ("openpose", "BODY_25"): "BODY_25",
}


def pose2sim_template_path() -> Path:
    """Locate the Config.toml shipped inside the installed Pose2Sim package."""
    import Pose2Sim

    root = Path(Pose2Sim.__file__).parent
    cand = root / "Demo_SinglePerson" / "Config.toml"
    if not cand.exists():
        raise FileNotFoundError(f"Pose2Sim template not found at {cand}")
    return cand


def _load(path: Path) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


def generate_config(project, out_file: Path | None = None) -> Path:
    """Write a Config.toml into the project root, tuned to the project settings.

    Returns the written path. Backends that are *not* RTMPose set the pose model
    to the external skeleton name; the pipeline runner is responsible for having
    produced the JSON beforehand (Pose2Sim reads it from pose/).
    """
    cfg = project.config
    template = pose2sim_template_path()
    data = _load(template)

    # ---- [project] ----
    proj = data.setdefault("project", {})
    proj["project_dir"] = str(project.root)
    proj["frame_rate"] = cfg.frame_rate
    proj["multi_person"] = False
    # A generic, permissive participant profile; refined later in the GUI.
    proj.setdefault("participant_height", 1.72)
    proj.setdefault("participant_mass", 70.0)

    # ---- [pose] ----
    pose = data.setdefault("pose", {})
    spec = get_backend(cfg.pose2d_backend)
    if spec.native:
        # Pose2Sim's built-in rtmlib estimator: pick the model + quality mode.
        pose["pose_model"] = spec.pose_model
        if spec.mode is not None:
            pose["mode"] = spec.mode
    else:
        # External backend: the app produces the JSON; tell Pose2Sim which
        # skeleton so downstream stages interpret keypoints correctly.
        pose["pose_model"] = POSE_MODEL_MAP.get(
            (spec.name, spec.skeleton), spec.skeleton)
    pose["overwrite_pose"] = False
    pose["display_detection"] = False  # headless / parallel-friendly

    # ---- [synchronization] ----
    # Default off: many datasets (and our own capture) are pre-synced. The GUI
    # exposes a toggle. Leaving it on pops blocking plots and re-times frames.
    sync = data.setdefault("synchronization", {})
    sync["display_sync_plots"] = False

    # ---- [filtering] ----
    # display_figures=True renders a blocking matplotlib figure per coordinate,
    # which stalled a headless run for ~40 min. Force it off.
    filt = data.setdefault("filtering", {})
    filt["display_figures"] = False

    # ---- [calibration] ----
    # We calibrate ourselves and hand Pose2Sim a ready Calib.toml, so default to
    # 'convert' from our easymocap-format yml. (GUI can switch to 'calculate'.)
    calib = data.setdefault("calibration", {})
    calib["calibration_type"] = "convert"
    conv = calib.setdefault("convert", {})
    conv["convert_from"] = "easymocap"

    # Also propagate the board geometry into the 'calculate' branch so the
    # built-in calibration path works if the user chooses it.
    calc = calib.setdefault("calculate", {})
    intr = calc.setdefault("intrinsics", {})
    intr["intrinsics_corners_nb"] = list(cfg.checkerboard.corners_nb)
    intr["intrinsics_square_size"] = cfg.checkerboard.square_size

    out_file = Path(out_file) if out_file else project.config_file
    with open(out_file, "wb") as f:
        tomli_w.dump(data, f)
    return out_file
