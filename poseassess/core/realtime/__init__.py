"""Real-time 3D skeleton from live webcams (prototype).

Pipeline per frame set: latest frame of every camera (``core.capture``) -> RTMPose 2D
keypoints per camera (rtmlib, onnxruntime CUDA/CPU) -> confidence-weighted DLT
triangulation with the project's ``Calib.toml`` -> One-Euro smoothing -> joint angles.

* ``skeleton``     HALPE_26 keypoint names and bones.
* ``calib``        ``Calib.toml`` -> projection matrices (rescaled to the live frame size).
* ``triangulate``  multi-view weighted DLT + reprojection error.
* ``filters``      vectorised One-Euro filter.
* ``angles``       simple joint angles (knee, hip, elbow, shoulder, trunk lean).
* ``pose2d``       rtmlib wrapper (one estimator per camera).
* ``engine``       ``RealtimeEngine``: worker thread + FPS / latency statistics.
* ``cli``          ``python -m poseassess.core.realtime`` headless benchmark.

The offline pipeline (Pose2Sim + OpenSim IK) is unchanged; nothing here writes into the project.
"""
