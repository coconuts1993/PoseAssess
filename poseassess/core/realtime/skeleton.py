"""HALPE_26 keypoints (RTMPose ``Body_with_feet``), as used by Pose2Sim."""
from __future__ import annotations

HALPE26 = [
    "Nose", "LEye", "REye", "LEar", "REar", "LShoulder", "RShoulder", "LElbow", "RElbow",
    "LWrist", "RWrist", "LHip", "RHip", "LKnee", "RKnee", "LAnkle", "RAnkle", "Head", "Neck",
    "Hip", "LBigToe", "RBigToe", "LSmallToe", "RSmallToe", "LHeel", "RHeel",
]
IDX = {n: i for i, n in enumerate(HALPE26)}

# (a, b, side) with side "R" / "L" / "C"
BONES = [
    ("Hip", "Neck", "C"), ("Neck", "Head", "C"), ("Head", "Nose", "C"),
    ("Hip", "RHip", "R"), ("RHip", "RKnee", "R"), ("RKnee", "RAnkle", "R"),
    ("RAnkle", "RHeel", "R"), ("RAnkle", "RBigToe", "R"), ("RBigToe", "RSmallToe", "R"),
    ("Neck", "RShoulder", "R"), ("RShoulder", "RElbow", "R"), ("RElbow", "RWrist", "R"),
    ("Hip", "LHip", "L"), ("LHip", "LKnee", "L"), ("LKnee", "LAnkle", "L"),
    ("LAnkle", "LHeel", "L"), ("LAnkle", "LBigToe", "L"), ("LBigToe", "LSmallToe", "L"),
    ("Neck", "LShoulder", "L"), ("LShoulder", "LElbow", "L"), ("LElbow", "LWrist", "L"),
]
BONE_IDX = [(IDX[a], IDX[b], s) for a, b, s in BONES]


def side(name: str) -> str:
    return "R" if name.startswith("R") else "L" if name.startswith("L") else "C"
