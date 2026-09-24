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

WII BALANCE BOARD (OPTIONAL)
----------------------------
PoseAssess can record a Nintendo Wii Balance Board together with the
trial (pages "2b. Wii Board" and "3b. Capture", results on "5. 3D View"
and "6. Results" > "Balance (Wii)"). Without the hidapi package the
status bar shows "Wii: off (hidapi not installed ...)"; with hidapi but no
board switched on it shows "Wii: searching for a paired board...".
Everything else works exactly as before.

1. New dependency: hidapi (the only one; see requirements-wii.txt).
   Add it to the portable venv once, on the DEV/BUILD PC with internet,
   after building the bundle (from the bundle folder; the Windows wheel
   contains the HID DLL, nothing else has to be installed):
       venv\Scripts\python.exe -m pip install hidapi
   (same as: venv\Scripts\python.exe -m pip install -r requirements-wii.txt)
   Then copy the bundle to the target PCs as usual.
   For a bundle that is already on an offline PC: on a PC with internet
   run   pip download hidapi --only-binary=:all: --python-version 3.11
   --platform win_amd64 -d wheels   and copy the "wheels" folder over,
   then (run run.bat once first, it repairs the venv paths):
       venv\Scripts\python.exe -m pip install --no-index --find-links wheels hidapi
   Check: venv\Scripts\python.exe -c "import hid; print('hidapi OK')"
   (pip_freeze.txt lists the bundle as built; hidapi is an optional
    addition and is not in it.)

2. Pair the board once per PC (Windows 10/11): Settings > Bluetooth &
   devices > Add device > Bluetooth, then press the red SYNC button in the
   board's battery compartment; choose "Nintendo RVL-WBC-01" and leave the
   PIN empty (or choose "Connect without a PIN"). With the Microsoft
   Bluetooth stack the pairing often has to be repeated after the board
   was switched off: remove the device and pair it again.

3. Plug and play: while PoseAssess runs it looks for a paired board every
   2 s, connects as soon as it is switched on (power button, blue LED on)
   and reconnects after a Bluetooth drop. Tare (zero) the EMPTY board on
   "2b. Wii Board"; the tare is remembered per board while PoseAssess is
   running (also across reconnects), but not after a restart: tare again
   at the start of every session.
   Environment variable POSEASSESS_WII: 0 = do not connect at start,
   sim = start the built-in simulator (demo without a board).

4. Wii-only recording (videos recorded by another program):
       run_wii.bat                        records into .\recordings\
       run_wii.bat --project "D:\Projects\Trial01" --subject S01
                                          records into the project's
                                          wii\recordings\ folder
       run_wii.bat --list                 lists the paired boards
       run_wii.bat --simulate --seconds 10   test without a board
       run_wii.bat --help                 all options
   While recording, type a label + Enter to mark an event (e.g. "sync"),
   press Enter on an empty line (or Ctrl+C) to stop. For videos from
   another program ask the subject to do 2 small jumps (or 2 stomps) at
   the start: PoseAssess uses them to align the Wii data with the videos
   ("6. Results" > "Balance (Wii)"). Exit codes: 0 recorded, 2 no board /
   hidapi missing, 3 error before recording, 4 no samples, 130 interrupted.

5. Where the Wii data goes: everything is inside the PROJECT folder, in
   a "wii" sub-folder (nothing is written into calibration\, pose\ or
   pose-3d\, so the Pose2Sim pipeline is not affected):
       wii\board\camNN\          frames + your clicks of the board corners
                                 (points.json) for camera NN
       wii\board\board.json      the board position in the Calib.toml world
                                 (recompute it after a new calibration)
       wii\recordings\<date_time>_<subject>\   one take: wii.csv (force,
                                 ~100 Hz, every sample with t / t_rel /
                                 t_unix), events.csv, session.json and, for
                                 camera takes, camNN.mkv + camNN_timestamps.csv
                                 (+ frames.csv once exported to videos\)
       wii\trial.json            which take belongs to the trial + its
                                 alignment with the videos
       wii\capture.json          the camera settings of "3b. Capture"
       wii\exports\              "Export fused table" (per-frame CSV, balance
                                 summary JSON, ground reaction force .mot)
   A camera take recorded on "3b. Capture" is exported into videos\ as
   cam01.mp4 ... camNN.mp4 (like "3. Videos" does), so "4. Run" works
   unchanged; its frame timestamps give the Wii alignment automatically.
   A frame is timestamped when the camera delivers it (after exposure,
   USB transfer and driver buffering: typically 30-100 ms for webcams).
   Measure this once per camera setup: record 2 small jumps, export, run
   the pipeline, then "Check timing with a sync event" on "6. Results" >
   "Balance (Wii)"; enter the value as "Camera latency" ("3b. Capture" >
   "Export to videos/") and export the take again.

6. Typical session: 2. Calibration -> 2b. Wii Board (click TL, TR, BR, BL,
   C of the board in the camera frames, "Compute board position", then
   the corner check with the board connected) -> 3b. Capture (record the
   cameras and the board together) -> 4. Run -> 5. 3D View (board, centre
   of pressure, force arrow, centre of mass) -> 6. Results > Balance (Wii)
   (sway metrics, plots, export).
