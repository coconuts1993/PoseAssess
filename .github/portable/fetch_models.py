"""Download the rtmlib 2D-pose models Pose2Sim uses (HALPE_26 'Body_with_feet') into the bundle.

Usage: python fetch_models.py <bundle>/models/rtmlib/hub/checkpoints [mode ...]
Default modes: lightweight balanced (the 'performance' models, ~600 MB, are downloaded by
rtmlib on first use when a PC has internet).
"""

import sys
from pathlib import Path

import rtmlib
from rtmlib.tools.file import download_checkpoint

dst = Path(sys.argv[1])
dst.mkdir(parents=True, exist_ok=True)
modes = sys.argv[2:] or ["lightweight", "balanced"]
for mode in modes:
    m = rtmlib.BodyWithFeet.MODE[mode]
    for key in ("det", "pose"):
        print(mode, key, download_checkpoint(m[key], dst_dir=str(dst)))
print(sorted(p.name for p in dst.iterdir()))
