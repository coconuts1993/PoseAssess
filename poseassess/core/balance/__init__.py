"""Wii Balance Board ("balance") processing for PoseAssess projects — no Qt, no hidapi.

Sub-modules
-----------
* ``paths``      where Wii data lives inside a project (``<root>/wii/``).
* ``trial``      the project's trial record ``wii/trial.json`` (active recording + time alignment).
* ``calib``      Calib.toml -> ``CameraCalibration`` objects (camera i = i-th table = camNN).
* ``geometry``   board model, rigid transforms, ``register_board`` (triangulation / PnP, floor fit).
* ``board``      project-level board registration: clicks per camera, ``board.json``, staleness.
* ``trc``        robust .trc reader (keeps Frame#, NaN-safe) + Y-up <-> Z-up conversion.
* ``com``        whole-body centre of mass (Winter segment model).
* ``analysis``   interpolation, sway metrics, force events, TRC events, offset estimation.
* ``alignment``  TRC time -> Wii recording time (recorded / sync event / xcorr / manual).
* ``fusion``     per-TRC-frame fused table (force, COP, COM), summary metrics, CSV/.mot export.
* ``overlay``    draw the board / COP / clicks on camera images (OpenCV), for QC views.

Import sub-modules explicitly (``from poseassess.core.balance.board import load_board``);
this package imports nothing heavy.
"""

from __future__ import annotations
