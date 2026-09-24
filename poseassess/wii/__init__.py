"""Wii Balance Board support for PoseAssess (optional feature).

Sub-modules
-----------
* ``protocol``   HID report encoding/decoding, sensor calibration, COP (pure functions).
* ``device``     ``ForceSource`` readers: ``BalanceBoardHID``, plug-and-play ``WiiAutoConnect``,
                 ``SimulatedBoard``; ``hidapi_status()``.
* ``recorder``   ``WiiRecorder``: writes a recording folder (session.json, wii.csv, events.csv)
                 on one perf_counter clock; cameras (duck-typed) can record into the same folder.
* ``io``         Readers for the recording files + file-name / column constants (the format
                 contract shared by every other module).
* ``record_cli`` Wii-only command-line recorder: ``python -m poseassess.wii``.

Importing this package (or any sub-module) never imports ``hid``: hidapi is loaded lazily by
``device._import_hid()`` only when a real board is searched for, so PoseAssess starts and works
without hidapi or without a board.
"""

from __future__ import annotations
