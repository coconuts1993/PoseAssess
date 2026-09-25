@echo off
REM ====================================================================
REM  PoseAssess - Wii Balance Board viewer: replays a recording and shows
REM  the centre-of-pressure trajectory in real time (no board needed).
REM  Double-click and use "Open...", or drag a file onto this .bat:
REM  a file of the original Wii program, a wii.csv or a recording folder.
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

REM --- Start the viewer. -------------------------------------------------
set "PYTHONPATH=%ROOT%"
"%ROOT%\venv\Scripts\python.exe" -m poseassess.wii.viewer %*
if errorlevel 1 (
    echo.
    echo [Wii viewer exited with an error - see the message above.]
    pause
)
endlocal
