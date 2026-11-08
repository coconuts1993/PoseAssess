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

WII BOARD DOES NOT CONNECT (WINDOWS)
------------------------------------
  * Close every other Wii program first (BrainBLoX, Wii Fit tools...): they
    reconfigure the board and the two programs disturb each other.
  * Pair the board: Windows Settings > Bluetooth & devices > Add device >
    Bluetooth, press the red SYNC button in the board's battery compartment,
    choose "Nintendo RVL-WBC-01" (it may also be listed as
    "Nintendo RVL-CNT-01") and leave the PIN empty (Next / Skip).
  * A board paired this way often cannot reconnect after it was switched off:
    remove it under Bluetooth & devices and pair it again with SYNC.
  * Status line at the bottom of the window:
      "Wii: connected ..."                 working
      "Wii: searching for a paired board"  Windows lists no board: pair it
      "Wii: board found but not responding" listed but silent: switch it on,
                                           close other Wii programs, or
                                           remove + pair it again
    Page 2b. Wii Board shows the last error in full.
  * Diagnosis: in the PoseAssess folder, type cmd in the Explorer address
    bar and press Enter, then run:   run_wii.bat --diagnose
    It lists every Nintendo Bluetooth device and tries three ways of talking
    to each one (Windows Bluetooth stacks differ); send its output when
    asking for help.

WII DATA: COMPUTER TIME AND FILES OF THE ORIGINAL WII PROGRAM
------------------------------------------------------------
  * Every time-stamped file PoseAssess writes (wii.csv, events.csv,
    camNN_timestamps.csv, frames.csv) ends with a column "time_local": the
    recording PC's date and time of that row, e.g.
    "2026-09-25 14:03:12.345+08:00" (t_unix holds the same instant as Unix
    seconds). Recordings made on different programs / PCs can be matched by it
    (as far as the PCs' clocks agree).
  * Files of the original Wii program (text, no header, 8 columns:
    ms TL TR BL BR copx_cm copy_cm total_kg) can be imported:
    6. Results > Balance (Wii) > Import Wii recording... > "wii.csv or
    original Wii program file...". They are converted to wii.csv; the
    original is kept as original_<name> in the recording folder.
  * Those files contain no computer time (the first column is a millisecond
    counter of that program). PoseAssess ESTIMATES it from the file's
    modification time (= time of the last sample), so import the original file
    from the PC it was recorded on (copying keeps the time; e-mail / upload
    does not). session.json then says clock_source = file_mtime and
    clock_estimated = true. For an exact alignment use a sync event (2 small
    jumps or stomps) as described above.

WII RECORDING REPLAY (NO BOARD NEEDED)
--------------------------------------
  * In the app (run.bat): 2b. Wii Board > "Replay file" tab > Open recording...
    (or drop a file on it). Stand-alone: double-click run_wii_viewer.bat, or
    drag a file onto it.
  * Opens files of the original Wii program, a wii.csv or a recording folder
    and replays the centre-of-pressure trajectory in real time on the board
    (trail 1-30 s or all), with the load / COP curves, the four sensors and
    COP sway figures. Speed 0.25x-4x; Space = play/pause, Left/Right = -/+1 s,
    Home = start. Opening a file jumps to 1 s before the subject steps on.

WII DATA IN THE 3D VIEW (5. 3D View)
------------------------------------
  * Wii files copied anywhere into the project folder (e.g. F:\dongyang\00\)
    are found: the line under the 3D view names them, and
    "Wii data" > "Wii files found in this project" loads one in one click.
    If no jump / stomp is found, the offset is estimated from the computer
    clock (modification times of the Wii file and of the videos, ~1-2 s):
    fine-tune it with "Wii offset". Copy the files with Explorer (keeps the
    modification time), not by e-mail / download.
  * "Wii data" > "Load Wii file for this trial..." takes a file of the
    original Wii program or a wii.csv. The app tries to align it
    automatically (the subject does 2 small jumps or stomps on the board).
    If that fails, the offset is 0: change "Wii offset" (Wii time = .trc
    time + offset) while playing, until the red COP follows the feet.
    Each change is saved as the trial's alignment.
  * Drawn with the skeleton: the board, the COP (red point + 2 s trail), and
    the force as a green arrow that starts at the COP and points up. The
    arrow is 0.5 cm per kg (35 cm for 70 kg); its label gives kg and N.
  * "Axes" shows two coordinate frames (x red, y lime green, z blue):
      - thick, 0.5 m, X / Y / Z, "O world": the 3D-pose world frame (origin
        of the calibration checkerboard, Z up);
      - thin, 0.25 m, x / y / z Wii: the board frame (board centre, x =
        subject's right, y = front edge, z = up). COP ML / AP are in this
        frame; the line under the view also gives the COP in world X / Y / Z.
  * If the board is not under the subject's feet, the board position is
    wrong (redo the clicks on 2b. Wii Board with the calibration used for
    the .trc).

OPENPOSE BODY_25B (OPTIONAL, NVIDIA GPU)
----------------------------------------
The 2D backend "OpenPose · BODY_25B (GPU)" (1. Project) runs CMU's OpenPose
with its BODY_25B model (25 body keypoints incl. the feet) instead of the
built-in RTMPose / MediaPipe models. It uses YOUR OWN OpenPose install.

License: OpenPose (Carnegie Mellon University) may be used for academic or
non-profit NONCOMMERCIAL RESEARCH only and must not be redistributed, so it
is NOT included in this package. Never add it to a copy of PoseAssess that
leaves your organization, and never upload it (lab PCs: see INTERNAL LAB
PACKAGE WITH OPENPOSE below).
License text:
https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/LICENSE

Requirements:
  * an NVIDIA graphics card with driver 456.38 or newer. RTX 50 series
    (Blackwell) cards do NOT work with OpenPose 1.7.0, nor do very old cards
    (compute capability below 5.2);
  * OpenPose 1.7.0, GPU build for Windows:
    openpose-1.7.0-binaries-win64-gpu-python3.7-flir-3d_recommended.zip
    (https://github.com/CMU-Perceptual-Computing-Lab/openpose/releases/tag/v1.7.0).
    The CPU build cannot run BODY_25B (the GPU build has
    bin\cudart64_110.dll next to bin\OpenPoseDemo.exe);
  * the BODY_25B files (step 2). CMU no longer offers the weights
    (.caffemodel) for download: you supply them (e.g. from your lab).

Setup:
1. Put OpenPose in C:\openpose (the folder with bin\OpenPoseDemo.exe and
   models\): PoseAssess finds it there by itself. It also looks in
   <PoseAssess folder>\openpose (next to run.bat: the lab's internal
   package puts it there), D:\openpose, %USERPROFILE%\openpose and the
   last folder used on this PC. Anywhere else: enter the folder under
   "OpenPose folder" on 1. Project, or use Browse... next to it. Avoid
   non-ASCII characters in the path.
   Or: 1. Project > "Get OpenPose..." > tick the license box > "Download
   official OpenPose 1.7.0 (418 MB)...": it downloads CMU's official release
   from GitHub onto this PC and unpacks it into <chosen folder>\openpose
   (choose C:\ to get C:\openpose; about 1.5 GB of free disk space needed).
   The bar shows the download; then the line under it says "Unpacking...".
   Cancel (or closing the window) stops it and removes the partial files
   (if you quit PoseAssess right after, it finishes that before it ends).
2. The BODY_25B files must have EXACTLY these names:
       models\pose\body_25b\pose_deploy.prototxt
       models\pose\body_25b\pose_iter_XXXXXX.caffemodel
   ("XXXXXX" literally: rename e.g. pose_iter_464000.caffemodel to
   pose_iter_XXXXXX.caffemodel). Other pose_iter_*.caffemodel files in that
   folder (e.g. copies of the COCO or BODY_25 weights) are ignored.
   The prototxt must belong to the same network as the weights:
   "1_25BBkg" (Option 2, recommended by CMU) or "1_25BSuperModel11FullVGG"
   (Option 1, maximum accuracy, slower), see
   https://github.com/CMU-Perceptual-Computing-Lab/openpose_train/tree/master/experimental_models
   Easier: "Get OpenPose..." > choose the variant (that of an installed
   official prototxt is preselected) > "Import BODY_25B model file
   (.caffemodel)...": the file is copied under the right name and the
   matching official prototxt is downloaded.
3. Choose "OpenPose · BODY_25B (GPU)" as 2D backend and press Save settings.
   The line under "OpenPose folder" should say "OpenPose ready: ..." in
   green. If no usable OpenPose is found, the field is framed in amber and
   the line says "Enter the OpenPose folder (e.g. C:\openpose) or use Get
   OpenPose...", with what is missing and how to fix it. If the folder you
   entered is incomplete but another OpenPose (e.g. C:\openpose) is ready,
   the line says so: "... is not ready (...); using C:\openpose instead".
   The line also warns about an unsupported GPU / driver.
   A folder that checks OK is remembered on this PC for every project.
4. "Network resolution" (under the OpenPose folder): 368 is OpenPose's own
   default (also what OpenPoseDemo.exe uses when run by hand) and the usual
   choice. 480 (CMU's suggestion for BODY_25B) sees small or distant people
   a little better but needs a GPU with MORE than 11 GB of memory: on an
   11 GB GTX 1080 Ti it runs at about 1 fps instead of 16 fps. Use the same
   setting for all trials you compare. The Run page shows the speed (fps)
   and the time left for each camera.

Lab PCs: use the INTERNAL package made with pack_with_openpose.bat (see
INTERNAL LAB PACKAGE WITH OPENPOSE below): PoseAssess together with the
lab's own OpenPose, kept on the lab's shared drive. Never upload that
package or pass it on outside the lab. The license allows such copies only
inside your organization: "If Licensee is an organization, it may make that
number of copies of the Software necessary for internal noncommercial use at
a single site within its organization provided that all information
appearing in or on the original labels, including the copyright and
trademark notices are copied onto the labels of the copies." Never pass
OpenPose (or its model files) on to anyone outside your organization / site.

INTERNAL LAB PACKAGE WITH OPENPOSE (pack_with_openpose.bat)
-----------------------------------------------------------
Makes a copy of this PoseAssess folder WITH the lab's OpenPose inside, for
the lab's other PCs. It is for internal noncommercial use at your site only
(OpenPose license, see above): keep it on the lab's shared drive, NEVER
upload or publish it.

Starting from nothing (PoseAssess not downloaded yet): on the PC with
OpenPose, double-click scripts\get_poseassess_with_openpose.bat from the
PoseAssess repository (keep the .ps1 next to it). It opens the download
page (GitHub login), waits for PoseAssess_Portable_win64.zip, unzips both
zip levels to D:\PoseAssess_Portable (or C:\PoseAssess\...) and runs the
steps below for you.

1. On a PC where OpenPose works (1. Project says "OpenPose ready"),
   double-click  pack_with_openpose.bat  in the PoseAssess folder. It finds
   that OpenPose like PoseAssess does (e.g. C:\openpose), checks it, shows
   the license note and asks before it copies anything.
2. It copies only what BODY_25B needs into <PoseAssess folder>\openpose:
   bin\ (all of it), models\pose\body_25b\pose_deploy.prototxt,
   models\pose\body_25b\pose_iter_XXXXXX.caffemodel and CMU's license /
   readme files (no other models). An existing openpose folder there is
   replaced only after you confirm.
3. It zips PoseAssess's own files (python311\, venv\, poseassess\,
   models\, openpose\, docs\, the .bat files, readme and requirement
   lists) to  ..\<this folder's name>_with_OpenPose.zip  (next to this
   folder). Anything else in the folder, e.g. recordings\ of run_wii.bat
   (participant data), is left out and listed. Put the zip on the lab's
   shared drive.
4. On a lab PC: unzip it to the local disk and double-click run.bat. The
   openpose folder next to run.bat is found automatically; choose
   "OpenPose · BODY_25B (GPU)" on 1. Project. The PC still needs an NVIDIA
   GPU and driver as listed under Requirements above.
Options (from a console in the PoseAssess folder):
    pack_with_openpose.bat --src "D:\tools\openpose"   (another OpenPose)
    pack_with_openpose.bat --no-zip                     (only the copy)
    pack_with_openpose.bat --zip-path "S:\Lab\PoseAssess"   (zip elsewhere)
    pack_with_openpose.bat --help
"Get OpenPose..." never downloads into the PoseAssess folder (a copy of
PoseAssess must not carry OpenPose by accident); only this tool puts it
there, on purpose, for the internal package.
