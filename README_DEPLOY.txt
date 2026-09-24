PoseAssess — portable (offline) deployment
==========================================

This folder is a SELF-CONTAINED copy of PoseAssess. It runs on any 64-bit
Windows PC with NO Python installation and NO internet connection.

TO USE ON ANOTHER COMPUTER
--------------------------
1. Copy this ENTIRE folder (PoseAssess_Portable) to the target PC
   (USB drive, network share, etc.). Keep all sub-folders together.
2. Double-click  run.bat
That's it. The first launch copies the pose models into your user cache
(one-time, a few seconds), then the app opens.

WHAT'S INSIDE
-------------
  python311\   a private copy of Python 3.11 (with its standard library)
  venv\        all Python dependencies, INCLUDING OpenSim 4.6 (no separate
               OpenSim install is needed — it is a bundled package)
  poseassess\  the application source
  models\      the RTMPose / YOLOX 2D-pose models
  run.bat      the launcher (fixes internal paths for wherever you copied it)

REQUIREMENTS ON THE TARGET
--------------------------
  * 64-bit Windows 10/11.
  * ~2 GB free disk space.
  * NOTHING else — no Python, no internet, no admin rights.

REBUILDING THIS BUNDLE (on the original dev machine)
----------------------------------------------------
  powershell -ExecutionPolicy Bypass -File deploy\build_portable.ps1
(Run a 2D-pose stage at least once first so the pose models are present in
 %USERPROFILE%\.cache\rtmlib before building.)

NOTES
-----
  * Your project/session data is separate from this folder — the app writes
    into whatever project folder you open, not here.
  * If Windows SmartScreen warns about run.bat, choose "More info -> Run
    anyway" (it is a plain batch file, not signed).
