"""OpenPose 2D backend (external executable).

OpenPose is not pip-installable; it ships as `OpenPoseDemo.exe` plus model
folders and needs an NVIDIA GPU in practice. This backend shells out to it and
lets OpenPose write its native BODY_25B JSON directly into the output folder,
which is already the format the rest of the pipeline consumes.

Configure the OpenPose install folder via `openpose_dir` (the folder that
contains `bin/OpenPoseDemo.exe` and the `models/` directory).
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional

from .base import Pose2DBackend, Pose2DResult, ProgressFn, register_pose2d


@register_pose2d("openpose")
class OpenPoseBackend(Pose2DBackend):
    display_name = "OpenPose (GPU)"
    skeleton = "BODY_25B"
    requires_gpu = True

    def _demo_exe(self) -> Optional[Path]:
        root = self.options.get("openpose_dir", "")
        if not root:
            return None
        root = Path(root)
        for cand in (root / "bin" / "OpenPoseDemo.exe", root / "OpenPoseDemo.exe"):
            if cand.exists():
                return cand
        return None

    def is_available(self) -> tuple[bool, str]:
        exe = self._demo_exe()
        if exe is None:
            return False, "OpenPoseDemo.exe not found (set openpose_dir)"
        return True, ""

    def process_video(
        self,
        video_path: Path,
        output_dir: Path,
        camera_index: int,
        progress: Optional[ProgressFn] = None,
    ) -> Pose2DResult:
        exe = self._demo_exe()
        if exe is None:
            raise RuntimeError("OpenPose not configured: set openpose_dir")

        video_path = Path(video_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        model_pose = self.options.get("model_pose", "BODY_25B")
        # OpenPose must run with its own folder as cwd so it finds models/.
        op_root = exe.parent.parent if exe.parent.name == "bin" else exe.parent

        cmd = [
            str(exe),
            "--video", str(video_path),
            "--write_json", str(output_dir),
            "--model_pose", model_pose,
            "--display", "0",
            "--render_pose", "0",
        ]
        extra = self.options.get("extra_args")
        if extra:
            cmd += list(extra)

        if progress:
            progress(0, None, f"cam{camera_index:02d}: launching OpenPose")
        subprocess.run(cmd, cwd=str(op_root), check=True)

        # OpenPose names files <videostem>_000000000000_keypoints.json; normalize
        # them to frame_NNNNN.json for a uniform downstream contract.
        written = self._normalize_output(output_dir)
        if progress:
            progress(written, written, f"cam{camera_index:02d}: {written} frames")
        return Pose2DResult(camera_index=camera_index, frames_written=written, output_dir=output_dir)

    @staticmethod
    def _normalize_output(output_dir: Path) -> int:
        files = sorted(output_dir.glob("*_keypoints.json"))
        for i, f in enumerate(files):
            f.rename(output_dir / f"frame_{i:05d}.json")
        return len(files)
