"""MediaPipe 2D pose backend.

Faithful port of the original Blender add-on's `ConvertVideoMP` operator:
run MediaPipe Pose on each frame and emit BODY_25B OpenPose-format JSON so the
downstream Pose2Sim triangulation is unchanged. No GPU required.

The mapping below reproduces exactly the keypoint ordering the original code
used (confidence = MediaPipe landmark `visibility`).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from .base import (
    Pose2DBackend,
    Pose2DResult,
    ProgressFn,
    register_pose2d,
    write_openpose_json,
)

# BODY_25B slot -> MediaPipe Pose landmark index (None => absent, written as 0,0,0).
# Index i of this list is the i-th BODY_25B keypoint; value is the MP landmark.
BODY_25B_FROM_MEDIAPIPE: list[Optional[int]] = [
    0,     # 0  Nose
    None,  # 1  (Neck)          - not provided by MediaPipe
    None,  # 2  (RShoulder)     - filled below via L/R shoulder? kept as original: 0
    None,  # 3
    None,  # 4
    11,    # 5  LShoulder
    12,    # 6  RShoulder
    13,    # 7  LElbow
    14,    # 8  RElbow
    15,    # 9  LWrist
    16,    # 10 RWrist
    23,    # 11 LHip
    24,    # 12 RHip
    25,    # 13 LKnee
    26,    # 14 RKnee
    27,    # 15 LAnkle
    28,    # 16 RAnkle
    None,  # 17
    None,  # 18
    31,    # 19 LBigToe (left_foot_index)
    None,  # 20 LSmallToe
    29,    # 21 LHeel
    32,    # 22 RBigToe (right_foot_index)
    None,  # 23 RSmallToe
    30,    # 24 RHeel
]


@register_pose2d("mediapipe")
class MediaPipeBackend(Pose2DBackend):
    display_name = "MediaPipe Pose (CPU)"
    skeleton = "BODY_25B"
    requires_gpu = False

    def is_available(self) -> tuple[bool, str]:
        try:
            import mediapipe  # noqa: F401
            import cv2  # noqa: F401
        except ImportError as e:
            return False, f"missing dependency: {e.name}"
        return True, ""

    def process_video(
        self,
        video_path: Path,
        output_dir: Path,
        camera_index: int,
        progress: Optional[ProgressFn] = None,
    ) -> Pose2DResult:
        import cv2
        import mediapipe as mp

        video_path = Path(video_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        mp_pose = mp.solutions.pose

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"could not open video: {video_path}")
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None

        model_complexity = int(self.options.get("model_complexity", 2))
        min_conf = float(self.options.get("min_detection_confidence", 0.5))

        frame = 0
        written = 0
        with mp_pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            enable_segmentation=False,
            min_detection_confidence=min_conf,
        ) as pose:
            while True:
                ok, image = cap.read()
                if not ok:
                    break
                h, w, _ = image.shape
                results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

                out_file = output_dir / f"frame_{frame:05d}.json"
                if not results.pose_landmarks:
                    # Emit an all-zero frame so frame indices stay aligned across cameras.
                    write_openpose_json(out_file, [0.0] * (25 * 3))
                else:
                    lm = results.pose_landmarks.landmark
                    flat: list[float] = []
                    for mp_idx in BODY_25B_FROM_MEDIAPIPE:
                        if mp_idx is None:
                            flat += [0.0, 0.0, 0.0]
                        else:
                            p = lm[mp_idx]
                            flat += [
                                round(p.x * w, 3),
                                round(p.y * h, 3),
                                round(p.visibility, 3),
                            ]
                    write_openpose_json(out_file, flat)
                written += 1

                if progress:
                    progress(frame, total, f"cam{camera_index:02d} frame {frame}")
                frame += 1

        cap.release()
        return Pose2DResult(camera_index=camera_index, frames_written=written, output_dir=output_dir)
