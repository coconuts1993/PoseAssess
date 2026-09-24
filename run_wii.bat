@echo off
REM ====================================================================
REM  PoseAssess - Wii Balance Board recorder (no cameras, no GUI).
REM  Double-click to record Wii data into .\recordings\<date_time>\ of
REM  this folder, or run it from a console with options, e.g.
REM     run_wii.bat --project "D:\Projects\Trial01" --subject S01
REM     run_wii.bat --list
REM     run_wii.bat --help
REM  Needs hidapi in the bundle (see README_DEPLOY.txt, "Wii Balance
REM  Board"); --simulate works without a board.
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

REM --- Run the recorder (the current folder is kept, so relative --out /
REM     --project paths work from a console). -----------------------------
set "PYTHONPATH=%ROOT%"
"%ROOT%\venv\Scripts\python.exe" -m poseassess.wii %*
set "RC=%ERRORLEVEL%"
echo.
if not "%RC%"=="0" echo [Wii recorder exited with code %RC% - see the messages above.]
pause
endlocal & exit /b %RC%
