"""High-level calibration driver for a whole project.

Ties the per-camera intrinsic + extrinsic steps together and produces the
Pose2Sim `Calib.toml`. Also provides `extract_frames_from_video`, the headless
replacement for the Blender add-on's "extract frames from the calibration clip"
step (used both for intrinsic frame selection and to grab the single extrinsic
frame per camera).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np

from .intrinsic import calibrate_intrinsic_camera, IntrinsicResult
from .extrinsic import (
    calibrate_extrinsic_camera, ExtrinsicResult, load_clicked_points,
    save_clicked_points, align_extrinsics,
)
from .write_yml import write_intri_yml, write_extri_yml, convert_yml_to_toml

ProgressFn = Callable[[str], None]


def extract_frames_from_video(
    video_path: Path,
    out_dir: Path,
    frames: Optional[Sequence[int]] = None,
    every_n: Optional[int] = None,
    prefix: str = "",
    ext: str = "jpg",
) -> list[Path]:
    """Save selected frames of a video as images.

    Provide either explicit `frames` (0-based indices) or `every_n` to sample
    one frame every N. With neither, saves only the first frame (index 0) —
    handy for grabbing the single extrinsic reference frame.
    """
    import cv2

    video_path = Path(video_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if frames is not None:
        wanted = sorted(set(int(f) for f in frames))
    elif every_n is not None and every_n > 0:
        wanted = list(range(0, total, every_n))
    else:
        wanted = [0]

    saved: list[Path] = []
    for idx in wanted:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, img = cap.read()
        if not ok:
            continue
        name = f"{prefix}{idx:06d}.{ext}"
        out_file = out_dir / name
        cv2.imwrite(str(out_file), img)
        saved.append(out_file)
    cap.release()
    return saved


def auto_extract_intrinsic_frames(
    video_path: Path,
    out_dir: Path,
    corners_nb,
    max_frames: int = 20,
    step: int = 15,
    min_center_dist_frac: float = 0.08,
    progress: Optional[ProgressFn] = None,
) -> list[Path]:
    """Scan a video and save well-spread frames where the checkerboard is found.

    Keeps a frame only if its board centre is at least `min_center_dist_frac` of
    the image diagonal away from every already-kept frame's board centre, so the
    intrinsic set covers varied board positions instead of near-duplicates.

    Returns the list of saved image paths.
    """
    import cv2
    import numpy as np

    video_path, out_dir = Path(video_path), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, cols = int(corners_nb[0]), int(corners_nb[1])
    pattern = (cols, rows)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    diag = (w ** 2 + h ** 2) ** 0.5
    min_dist = min_center_dist_frac * diag

    kept_centers: list[np.ndarray] = []
    saved: list[Path] = []
    idx = 0
    while idx < total and len(saved) < max_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, img = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, pattern, flags=flags)
        if found:
            center = corners.reshape(-1, 2).mean(axis=0)
            if all(np.linalg.norm(center - c) >= min_dist for c in kept_centers):
                out = out_dir / f"{idx:06d}.jpg"
                cv2.imwrite(str(out), img)
                saved.append(out)
                kept_centers.append(center)
                if progress:
                    progress(f"kept frame {idx} ({len(saved)}/{max_frames})")
        idx += step
    cap.release()
    if progress:
        progress(f"done: {len(saved)} intrinsic frames from {video_path.name}")
    return saved


@dataclass
class CalibrationOutcome:
    intrinsics: list[IntrinsicResult]
    extrinsics: list[ExtrinsicResult]
    calib_toml: Path


def calibrate_project(project, progress: Optional[ProgressFn] = None) -> CalibrationOutcome:
    """Run full calibration for a `Project`.

    Expects, per camera N (1-based):
      - intrinsic frames in `<intrinsic_frames>/camNN/`
      - the clicked extrinsic corners in `<extrinsic_points>/camNN/points.json`
        (produced by the GUI corner-picker; ordered row-major over the board)

    Writes intri.yml, extri.yml and Calib.toml into the calibration folder.
    """
    def log(msg: str) -> None:
        if progress:
            progress(msg)

    cfg = project.config
    cb = cfg.checkerboard
    n = cfg.num_cameras

    # ---- intrinsic ----
    intr: list[IntrinsicResult] = []
    for i in range(1, n + 1):
        frames_dir = project.intrinsic_cam_dir(i)
        log(f"[intrinsic] cam{i:02d}: calibrating from {frames_dir}")
        res = calibrate_intrinsic_camera(
            frames_dir, cb.corners_nb, cb.square_size, camera_index=i, progress=progress
        )
        log(f"[intrinsic] cam{i:02d}: RMS={res.rms_error:.3f}px "
            f"({res.frames_used}/{res.frames_total} frames)")
        intr.append(res)

    # ---- extrinsic ----
    # Board squares are configured in mm; solve the extrinsics in METRES so the
    # world frame is metric (what Pose2Sim triangulation + OpenSim scaling
    # expect). solvePnP translations then come out in metres, not millimetres.
    #
    # The clicked corner PIXELS in points.json are never moved or reordered on
    # disk. But the 2D<->3D correspondence (which clicked corner is grid cell
    # (0,0), (1,0), ...) must be resolved so every camera shares one physical
    # origin/orientation — auto-detect returns different grid traversals when the
    # board is seen rotated. `align_extrinsics` figures this out (the EasyMocap
    # equivalent is storing keypoints3d per clicked point); it only affects the
    # internal correspondence, not the user's marks.
    square_m = cb.square_size / 1000.0
    intr_by_cam = {r.camera_index: r for r in intr}
    corners_by_cam: dict[int, np.ndarray] = {}
    for i in range(1, n + 1):
        points_file = project.extrinsic_points_file(i)
        if not points_file.exists():
            raise FileNotFoundError(
                f"cam{i:02d}: missing clicked corners {points_file}. "
                f"Click the extrinsic board corners in the GUI first."
            )
        corners_by_cam[i] = load_clicked_points(points_file)
    extr, _aligned = align_extrinsics(
        intr_by_cam, corners_by_cam, cb.corners_nb, square_m,
        invert_z=getattr(cb, "invert_z", False), progress=log,
    )  # note: does NOT write points.json back — the user's marks are untouched

    # ---- write yml + convert to Calib.toml ----
    write_intri_yml(project.intri_yml, intr)
    write_extri_yml(project.extri_yml, extr)
    log(f"[calib] wrote {project.intri_yml.name}, {project.extri_yml.name}")
    sizes = [r.image_size for r in intr]
    calib_toml = convert_yml_to_toml(project.intri_yml, project.extri_yml, sizes=sizes)
    log(f"[calib] wrote {calib_toml}")

    return CalibrationOutcome(intrinsics=intr, extrinsics=extr, calib_toml=calib_toml)


def resolve_extrinsics_only(project, progress: Optional[ProgressFn] = None) -> Path:
    """Re-solve ONLY the extrinsics from the saved corners, reusing the existing
    Calib.toml intrinsics, and patch the camera rotation/translation in place.

    Fast (no intrinsic re-calibration) — used by the 'Flip Z axis' toggle, which
    just changes the world convention (`checkerboard.invert_z`). Returns the
    Calib.toml path. Raises if there is no Calib.toml to reuse.
    """
    from types import SimpleNamespace
    from ..calib_check import read_calib_for_projection

    def log(msg: str) -> None:
        if progress:
            progress(msg)

    cb = project.config.checkerboard
    n = project.config.num_cameras
    calib_toml = next(iter(sorted(project.calibration_dir.glob("Calib*.toml"))), None)
    if calib_toml is None:
        raise FileNotFoundError("no Calib.toml yet — run a full calibration first")
    cams = read_calib_for_projection(calib_toml)
    if len(cams) < n:
        raise ValueError("Calib.toml has fewer cameras than the project — re-run "
                         "the full calibration")
    intr_by_cam = {i + 1: SimpleNamespace(K=cams[i]["K"], dist=cams[i]["dist"])
                   for i in range(n)}
    corners_by_cam = {i: load_clicked_points(project.extrinsic_points_file(i))
                      for i in range(1, n + 1)}
    extr, _ = align_extrinsics(
        intr_by_cam, corners_by_cam, cb.corners_nb, cb.square_size / 1000.0,
        invert_z=getattr(cb, "invert_z", False), progress=log,
    )
    _patch_calib_extrinsics(calib_toml, {r.camera_index: r for r in extr})
    log(f"[calib] re-solved extrinsics (invert_z={getattr(cb, 'invert_z', False)})")
    return calib_toml


def _patch_calib_extrinsics(calib_toml: Path, results_by_cam: dict) -> None:
    """Overwrite each camera's `rotation`/`translation` line in Calib.toml with a
    freshly-solved ExtrinsicResult, leaving intrinsics/size untouched. Camera
    tables appear in index order (a `[metadata]` trailer is ignored)."""
    lines = Path(calib_toml).read_text().splitlines()
    cam_i = 0
    for k, ln in enumerate(lines):
        s = ln.strip()
        if s.startswith("[") and not s.startswith("[[") and s.lower() != "[metadata]":
            cam_i += 1
        elif s.startswith("rotation") and cam_i in results_by_cam:
            rv = np.asarray(results_by_cam[cam_i].rvec, float).ravel()
            lines[k] = f"rotation = [ {rv[0]}, {rv[1]}, {rv[2]},]"
        elif s.startswith("translation") and cam_i in results_by_cam:
            tv = np.asarray(results_by_cam[cam_i].T, float).ravel()
            lines[k] = f"translation = [ {tv[0]}, {tv[1]}, {tv[2]},]"
    Path(calib_toml).write_text("\n".join(lines) + "\n")
