"""Where Wii Balance Board data lives inside a PoseAssess project.

Everything goes below ONE new top-level folder ``<root>/wii/`` so that Pose2Sim and the existing
pages never see it::

    <root>/wii/
        board/camNN/*.jpg          reference frames for clicking the board (any name)
        board/camNN/points.json    clicked landmarks TL, TR, BR, BL, C (see board.CameraClicks)
        board/board.json           board registration result (see board.BoardRegistration)
        recordings/<id>/           one folder per take (see poseassess.wii.io)
        trial.json                 the project's trial: active recording + alignment (trial.py)
        capture.json               Capture page settings (core.capture.settings)
        exports/                   fused tables / summaries exported by the Results page

Rules (from the Pose2Sim / PoseAssess code, do not break them):
1. No top-level folder with "calib" in its name (Pose2Sim picks the first one).
2. Never write *.toml / *.yml into ``calibration/`` (Pose2Sim reads the newest *.toml there).
3. Nothing but ``camNN.<video>`` in ``videos/`` (every video file there is a camera).
4. Nothing new in ``pose*/``, ``pose-3d/``, ``kinematics/`` (globbed by Pose2Sim, wiped by
   the benchmark).
5. No ``Config.toml`` in any sub-folder (Pose2Sim batch mode).
6. Wii settings are NOT stored in project.toml (``ProjectConfig`` drops unknown keys).
"""

from __future__ import annotations

from pathlib import Path

from poseassess.wii.io import is_recording_folder

WII_DIR = "wii"
BOARD_DIR = "board"
RECORDINGS_DIR = "recordings"
EXPORTS_DIR = "exports"
BOARD_JSON = "board.json"
POINTS_JSON = "points.json"
TRIAL_JSON = "trial.json"
CAPTURE_JSON = "capture.json"


def cam_name(index: int) -> str:
    """1-based camera name, identical to ``Project.cam_name``: 1 -> ``cam01``."""
    return f"cam{index:02d}"


def cam_index(name: str) -> int:
    """``cam01`` -> 1 (inverse of ``cam_name``); raises ValueError for other names."""
    if not name.startswith("cam") or not name[3:].isdigit():
        raise ValueError(f"not a camera name: {name!r}")
    return int(name[3:])


class WiiPaths:
    """Paths of the Wii data of one project. Accepts a ``Project`` or a project root path.
    Creates nothing unless ``ensure()`` is called."""

    def __init__(self, project_or_root):
        root = getattr(project_or_root, "root", project_or_root)
        self.root = Path(root).resolve()

    @property
    def wii_dir(self) -> Path:
        return self.root / WII_DIR

    @property
    def board_dir(self) -> Path:
        return self.wii_dir / BOARD_DIR

    def board_cam_dir(self, index: int) -> Path:
        """``wii/board/camNN`` (1-based index)."""
        return self.board_dir / cam_name(index)

    def board_points_file(self, index: int) -> Path:
        """``wii/board/camNN/points.json``."""
        return self.board_cam_dir(index) / POINTS_JSON

    @property
    def board_json(self) -> Path:
        return self.board_dir / BOARD_JSON

    @property
    def recordings_dir(self) -> Path:
        return self.wii_dir / RECORDINGS_DIR

    def recording_dir(self, rec_id: str) -> Path:
        """``wii/recordings/<rec_id>`` (not created)."""
        return self.recordings_dir / rec_id

    @property
    def trial_json(self) -> Path:
        return self.wii_dir / TRIAL_JSON

    @property
    def capture_json(self) -> Path:
        return self.wii_dir / CAPTURE_JSON

    @property
    def exports_dir(self) -> Path:
        return self.wii_dir / EXPORTS_DIR

    def relative(self, path: str | Path) -> str:
        """``path`` relative to the project root with forward slashes (absolute if outside)."""
        p = Path(path).resolve()
        try:
            return p.relative_to(self.root).as_posix()
        except ValueError:
            return p.as_posix()

    def resolve(self, rel: str | Path) -> Path:
        """Inverse of ``relative``: project-relative (or absolute) path -> absolute Path."""
        p = Path(rel)
        return p if p.is_absolute() else self.root / p

    def ensure(self) -> WiiPaths:
        """Create wii/, wii/board/, wii/recordings/ and wii/exports/. Returns self."""
        for d in (self.wii_dir, self.board_dir, self.recordings_dir, self.exports_dir):
            d.mkdir(parents=True, exist_ok=True)
        return self

    def list_recordings(self) -> list[str]:
        """Recording ids (folder names under wii/recordings/ that contain session.json or
        wii.csv), oldest first (names start with a timestamp)."""
        d = self.recordings_dir
        if not d.is_dir():
            return []
        return sorted(p.name for p in d.iterdir() if is_recording_folder(p))
