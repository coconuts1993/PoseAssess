@echo off
REM ====================================================================
REM  PoseAssess - make the lab's INTERNAL package with OpenPose included.
REM  Copies the BODY_25B parts of this PC's OpenPose (e.g. C:\openpose:
REM  bin\ + models\pose\body_25b\ + CMU's license files) into .\openpose
REM  of this folder, then zips this whole folder to
REM  ..\[this folder]_with_OpenPose.zip for the lab's shared drive.
REM  OpenPose license: internal non-commercial use at one site of the
REM  licensee's organization only - NEVER upload or publish the zip.
REM  Only PoseAssess's own files are zipped (not e.g. .\recordings).
REM  Options, from a console:
REM     pack_with_openpose.bat --src "D:\tools\openpose"
REM     pack_with_openpose.bat --no-zip
REM     pack_with_openpose.bat --zip-path "S:\Lab\PoseAssess"
REM     pack_with_openpose.bat --help
REM ====================================================================
setlocal EnableExtensions
set "ROOT=%~dp0"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"

REM --- Point the bundled venv at the bundled Python (same as run.bat). --
set "PYHOME=%ROOT%\python311"
if not exist "%PYHOME%\python.exe" (
    echo [ERROR] Missing "%PYHOME%\python.exe". The bundle is incomplete.
    pause
    exit /b 1
)
> "%ROOT%\venv\pyvenv.cfg" echo home = %PYHOME%
>> "%ROOT%\venv\pyvenv.cfg" echo include-system-site-packages = false
>> "%ROOT%\venv\pyvenv.cfg" echo version = 3.11.3
>> "%ROOT%\venv\pyvenv.cfg" echo executable = %PYHOME%\python.exe

REM --- Copy OpenPose and write the zip (the current folder is kept, so a
REM     relative --src / --zip-path works from a console). ---------------
REM     PYTHONSAFEPATH=1 (Python 3.11+): never import a "poseassess" folder
REM     from the console's current folder instead of this bundle's.
set "PYTHONPATH=%ROOT%"
set "PYTHONSAFEPATH=1"
set "PYTHONUTF8=1"
"%ROOT%\venv\Scripts\python.exe" -m poseassess.tools.pack_openpose %*
set "RC=%ERRORLEVEL%"
echo.
if not "%RC%"=="0" echo [Packing exited with code %RC% - see the messages above.]
pause
endlocal & exit /b %RC%
