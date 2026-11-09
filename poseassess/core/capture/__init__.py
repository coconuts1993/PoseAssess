"""In-app trial capture: live multi-camera streams + synchronized recording with the Wii
Balance Board, and export of a take into ``videos/camNN.mp4`` for the Pose2Sim pipeline.

* ``camera``    ``CameraStream`` capture thread + crash-safe MKV writer with per-frame
                timestamps (port of PoseBoard 3ea6d8a ``poseboard/camera.py``).
* ``settings``  ``CaptureSettings`` stored in ``wii/capture.json``.
* ``session``   ``CaptureSession``: opens the project's cameras, records takes into
                ``wii/recordings/<id>/`` through ``poseassess.wii.recorder.WiiRecorder``;
                ``probe_cameras``.
* ``export``    resample the raw per-camera MKVs of a take onto one common frame grid and write
                ``videos/camNN.mp4`` + ``frames.csv``; updates the project fps and trial.json.
"""

from __future__ import annotations
