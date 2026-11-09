"""Capture page settings, stored per project in ``wii/capture.json``::

    {"cameras": {"cam01": {"source": 0, "width": 1920, "height": 1080, "fps": 30.0,
                           "enabled": true}, ...},
     "output_fps": 30,            # null: use the project's frame_rate
     "keep_raw": true,            # keep camNN.mkv in the recording folder after export
     "subject": "",
     "latency_ms": 0.0}           # camera capture latency, see below

``source`` is a device index (int), or a string (URL / video file; a file loops, for tests).

``latency_ms``: a frame is stamped when ``cap.read()`` returns, i.e. after exposure, USB
transfer and driver buffering (USB webcams: typically tens of ms, up to ~100 ms); the Wii
samples are stamped when the HID report arrives. The export subtracts this latency from the
frame timestamps (frames.csv = exposure times). A camera entry may override it with its own
``"latency_ms"`` (only written when set). Measure it with "Check timing with a sync event" on
6. Results > Balance (Wii) (a jump recorded with the videos).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from poseassess.core.balance.paths import WiiPaths, cam_name


@dataclass
class CameraSetting:
    source: int | str = 0
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    enabled: bool = True
    latency_ms: float | None = None  # None: the global CaptureSettings.latency_ms

    def to_dict(self) -> dict:
        d = {"source": self.source, "width": self.width, "height": self.height,
             "fps": self.fps, "enabled": self.enabled}
        if self.latency_ms is not None:
            d["latency_ms"] = self.latency_ms
        return d

    @classmethod
    def from_dict(cls, d: dict) -> CameraSetting:
        src = d.get("source", 0)
        if isinstance(src, str) and src.strip().isdigit():
            src = int(src)

        def _i(v):
            return None if v in (None, "", 0) else int(v)

        fps = d.get("fps")
        return cls(src, _i(d.get("width")), _i(d.get("height")),
                   None if fps in (None, "", 0) else float(fps), bool(d.get("enabled", True)),
                   _ms(d.get("latency_ms")))


@dataclass
class CaptureSettings:
    cameras: dict[str, CameraSetting] = field(default_factory=dict)
    output_fps: int | None = None
    keep_raw: bool = True
    subject: str = ""
    latency_ms: float = 0.0

    def to_dict(self) -> dict:
        return {"cameras": {k: v.to_dict() for k, v in sorted(self.cameras.items())},
                "output_fps": self.output_fps, "keep_raw": self.keep_raw,
                "subject": self.subject, "latency_ms": self.latency_ms}

    @classmethod
    def from_dict(cls, d: dict) -> CaptureSettings:
        fps = d.get("output_fps")
        return cls({k: CameraSetting.from_dict(v) for k, v in (d.get("cameras") or {}).items()},
                   None if fps in (None, "", 0) else int(fps), bool(d.get("keep_raw", True)),
                   str(d.get("subject") or ""), _ms(d.get("latency_ms")) or 0.0)

    def latency_s(self, cams) -> dict[str, float]:
        """``{cam: capture latency in s}`` for the export (camera override, else global)."""
        out = {}
        for c in cams:
            cs = self.cameras.get(c)
            ms = cs.latency_ms if cs is not None and cs.latency_ms is not None else self.latency_ms
            out[c] = float(ms or 0.0) / 1000.0
        return out


def _ms(v) -> float | None:
    """A latency (ms) from JSON: float in [0, 1000], None when missing or invalid."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if 0.0 <= f <= 1000.0 else None


def default_settings(num_cameras: int) -> CaptureSettings:
    """camNN -> device index NN-1 for every project camera."""
    return CaptureSettings({cam_name(i): CameraSetting(source=i - 1)
                            for i in range(1, num_cameras + 1)})


def load_capture_settings(project) -> CaptureSettings:
    """wii/capture.json, completed with defaults for project cameras it does not list and
    without cameras beyond ``project.config.num_cameras``."""
    n = int(getattr(getattr(project, "config", None), "num_cameras", 0) or 0)
    s = default_settings(n)
    p = WiiPaths(project).capture_json
    try:
        loaded = CaptureSettings.from_dict(json.loads(p.read_text(encoding="utf-8")))
    except (OSError, ValueError):
        return s
    for k in list(s.cameras):
        if k in loaded.cameras:
            s.cameras[k] = loaded.cameras[k]
    s.output_fps, s.keep_raw, s.subject = loaded.output_fps, loaded.keep_raw, loaded.subject
    s.latency_ms = loaded.latency_ms
    return s


def save_capture_settings(project, settings: CaptureSettings) -> Path:
    """Write wii/capture.json. Returns its path."""
    paths = WiiPaths(project)
    paths.wii_dir.mkdir(parents=True, exist_ok=True)
    p = paths.capture_json
    p.write_text(json.dumps(settings.to_dict(), indent=2), encoding="utf-8")
    return p
