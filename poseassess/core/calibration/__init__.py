"""Camera calibration: intrinsic (auto) + extrinsic (manual clicks) -> Calib.toml."""
from .intrinsic import calibrate_intrinsic_camera, IntrinsicResult
from .extrinsic import (
    calibrate_extrinsic_camera,
    ExtrinsicResult,
    load_clicked_points,
    save_clicked_points,
)
from .write_yml import write_intri_yml, write_extri_yml, convert_yml_to_toml
from .orchestrator import (
    calibrate_project,
    extract_frames_from_video,
    auto_extract_intrinsic_frames,
)

__all__ = [
    "calibrate_intrinsic_camera", "IntrinsicResult",
    "calibrate_extrinsic_camera", "ExtrinsicResult",
    "load_clicked_points", "save_clicked_points",
    "write_intri_yml", "write_extri_yml", "convert_yml_to_toml",
    "calibrate_project", "extract_frames_from_video",
    "auto_extract_intrinsic_frames",
]
