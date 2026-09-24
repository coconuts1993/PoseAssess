"""Write intri.yml / extri.yml in the OpenCV YAML format Pose2Sim expects.

We use `cv2.FileStorage`, which emits the exact `!!opencv-matrix` blocks and
`names` sequence the downstream `Pose2Sim.Utilities.calib_yml_to_toml`
converter reads (byte-for-byte compatible with the original add-on's output).
Camera keys are two-digit strings: "01", "02", ...
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from .intrinsic import IntrinsicResult
from .extrinsic import ExtrinsicResult


def _cam_key(index: int) -> str:
    return f"{index:02d}"


def write_intri_yml(path: Path, results: Sequence[IntrinsicResult]) -> None:
    import cv2

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_WRITE)
    try:
        fs.startWriteStruct("names", cv2.FileNode_SEQ)
        for r in results:
            fs.write("", _cam_key(r.camera_index))
        fs.endWriteStruct()
        for r in results:
            key = _cam_key(r.camera_index)
            # EasyMocap/Pose2Sim also store per-cam image size under these keys.
            w, h = r.image_size
            fs.write(f"K_{key}", np.asarray(r.K, dtype=np.float64))
            fs.write(f"dist_{key}", np.asarray(r.dist, dtype=np.float64).reshape(1, -1))
    finally:
        fs.release()


def write_extri_yml(path: Path, results: Sequence[ExtrinsicResult]) -> None:
    import cv2

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_WRITE)
    try:
        fs.startWriteStruct("names", cv2.FileNode_SEQ)
        for r in results:
            fs.write("", _cam_key(r.camera_index))
        fs.endWriteStruct()
        for r in results:
            key = _cam_key(r.camera_index)
            fs.write(f"R_{key}", np.asarray(r.rvec, dtype=np.float64).reshape(3, 1))
            fs.write(f"Rot_{key}", np.asarray(r.R, dtype=np.float64))
            fs.write(f"T_{key}", np.asarray(r.T, dtype=np.float64).reshape(3, 1))
    finally:
        fs.release()


def convert_yml_to_toml(
    intri_yml: Path,
    extri_yml: Path,
    sizes: Sequence[tuple[int, int]] | None = None,
) -> Path:
    """Convert intri/extri yml -> Pose2Sim Calib.toml via Pose2Sim's converter.

    Pose2Sim's easymocap converter estimates image size as twice the optical
    centre, which is often wrong; pass `sizes` [(w, h), ...] in camera order to
    overwrite the size lines with the true resolution.

    Returns the produced Calib.toml path.
    """
    from Pose2Sim.Utilities import calib_easymocap_to_toml

    intri_yml, extri_yml = Path(intri_yml), Path(extri_yml)
    calib_easymocap_to_toml.calib_easymocap_to_toml_func(str(intri_yml), str(extri_yml))
    out = intri_yml.parent / "Calib.toml"

    if sizes and out.exists():
        _patch_calib_sizes(out, sizes)
    return out


def _patch_calib_sizes(calib_toml: Path, sizes: Sequence[tuple[int, int]]) -> None:
    """Rewrite each camera's `size = [...]` line with the true (w, h)."""
    lines = calib_toml.read_text().splitlines()
    cam_i = -1
    for idx, ln in enumerate(lines):
        stripped = ln.strip()
        if stripped.startswith("[") and not stripped.startswith("[["):
            # new camera table (ignore a [metadata] trailer)
            if stripped.lower() != "[metadata]":
                cam_i += 1
        elif stripped.startswith("size") and 0 <= cam_i < len(sizes):
            w, h = sizes[cam_i]
            lines[idx] = f"size = [ {float(w)}, {float(h)},]"
    calib_toml.write_text("\n".join(lines) + "\n")
