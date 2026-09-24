"""Load trials from the OpenCap LabValidation dataset (zip) into PoseAssess projects.

The dataset ships as one big zip:
    LabValidation_withVideos/<subject>/
        VideoData/<Session>/Cam<k>/cameraIntrinsicsExtrinsics.pickle   # per-session calib
        VideoData/<Session>/Cam<k>/<task>/<task>_syncdWithMocap.avi     # synced videos
        OpenSimData/Mocap/IK/<task>.mot                                 # marker GT joint angles

`assemble_trial_project` extracts just one (subject, task) — 5 synced videos +
5 calibration pickles — builds a PoseAssess project with a Calib.toml, and
returns the ground-truth .mot path so a validation run can compare against it.
Nothing is fully unzipped; only the needed members are extracted.
"""
from __future__ import annotations

import re
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ..project import Project, ProjectConfig
from ..config_gen import generate_config
from ..calibration.opencap_import import opencap_to_calib_toml

ROOT = "LabValidation_withVideos"
_VID_RE = re.compile(
    rf"{ROOT}/(subject\d+)/VideoData/(Session\d+)/Cam(\d+)/([^/]+)/\4_syncdWithMocap\.avi")


@dataclass
class TrialSpec:
    subject: str
    task: str
    session: str
    cams: int
    has_gt: bool


class OpenCapZip:
    """Read-only accessor over the LabValidation zip."""

    def __init__(self, zip_path: str | Path):
        self.zip_path = Path(zip_path)
        self._zip = zipfile.ZipFile(self.zip_path)
        self._names = self._zip.namelist()
        self._nameset = set(self._names)

    def subjects(self) -> list[str]:
        subs = set(m.group(1) for n in self._names
                   for m in [re.search(r"(subject\d+)", n)] if m)
        return sorted(subs, key=lambda s: int(s[7:]))

    def trials(self, subject: str) -> list[TrialSpec]:
        """All tasks for a subject that have >=5 synced-video cameras."""
        cams: dict[str, set[int]] = {}
        sess: dict[str, str] = {}
        for n in self._names:
            m = _VID_RE.match(n)
            if m and m.group(1) == subject:
                _, se, cam, task = m.groups()
                cams.setdefault(task, set()).add(int(cam))
                sess[task] = se
        out = []
        for task in sorted(cams):
            out.append(TrialSpec(
                subject=subject, task=task, session=sess[task],
                cams=len(cams[task]), has_gt=self._gt_member(subject, task) is not None))
        return out

    def _gt_member(self, subject: str, task: str) -> Optional[str]:
        cand = f"{ROOT}/{subject}/OpenSimData/Mocap/IK/{task}.mot"
        return cand if cand in self._nameset else None

    def _video_member(self, subject: str, session: str, cam: int, task: str) -> str:
        return f"{ROOT}/{subject}/VideoData/{session}/Cam{cam}/{task}/{task}_syncdWithMocap.avi"

    def _calib_member(self, subject: str, session: str, cam: int) -> str:
        return f"{ROOT}/{subject}/VideoData/{session}/Cam{cam}/cameraIntrinsicsExtrinsics.pickle"

    def _extract(self, member: str, dest: Path) -> Path:
        dest.parent.mkdir(parents=True, exist_ok=True)
        with self._zip.open(member) as src, open(dest, "wb") as out:
            shutil.copyfileobj(src, out)
        return dest


def assemble_trial_project(
    zip_obj: OpenCapZip,
    subject: str,
    task: str,
    dest_root: Path,
    n_cams: int = 5,
    frame_rate: int = 60,
    backend: str = "rtmpose",
) -> tuple[Project, Path]:
    """Extract one (subject, task) into a ready-to-run PoseAssess project.

    Returns (project, ground_truth_mot_path).
    """
    import cv2

    trials = {t.task: t for t in zip_obj.trials(subject)}
    if task not in trials:
        raise KeyError(f"{subject} has no task {task!r}")
    spec = trials[task]
    session = spec.session
    gt_member = zip_obj._gt_member(subject, task)
    if gt_member is None:
        raise FileNotFoundError(f"no ground-truth IK for {subject}/{task}")

    dest_root = Path(dest_root)
    if dest_root.exists():
        shutil.rmtree(dest_root)

    cfg = ProjectConfig(name=f"{subject}_{task}", num_cameras=n_cams,
                        frame_rate=frame_rate, pose2d_backend=backend,
                        pose_model="HALPE_26")
    proj = Project(dest_root, cfg).create()

    # videos -> videos/cam01.avi .. camNN.avi ; calib pickles -> temp
    sizes: list[tuple[int, int]] = []
    pickles: list[Path] = []
    tmp = Path(tempfile.mkdtemp(prefix="opencap_calib_"))
    for cam in range(n_cams):
        vdst = proj.videos_dir / f"cam{cam+1:02d}.avi"
        zip_obj._extract(zip_obj._video_member(subject, session, cam, task), vdst)
        c = cv2.VideoCapture(str(vdst)); w = int(c.get(3)); h = int(c.get(4)); c.release()
        sizes.append((w, h))
        pk = zip_obj._extract(zip_obj._calib_member(subject, session, cam), tmp / f"cam{cam}.pickle")
        pickles.append(pk)

    opencap_to_calib_toml(pickles, sizes, proj.calib_toml)
    shutil.rmtree(tmp, ignore_errors=True)

    generate_config(proj)

    gt_dst = proj.root / f"GT_{task}.mot"
    zip_obj._extract(gt_member, gt_dst)
    return proj, gt_dst
