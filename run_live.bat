@echo off
REM ====================================================================
REM  PoseAssess - live 3D skeleton benchmark (no GUI). Measures FPS and
REM  latency of webcams -> RTMPose 2D -> 3D on this PC. Examples:
REM     run_live.bat --project "D:\Projects\Trial01" --show
REM     run_live.bat --project "D:\Projects\Trial01" --size 1280x720 --mode lightweight
REM     run_live.bat --sources 0 1 --size 1280x720 --no-3d --show
REM     run_live.bat --help
REM  See README_LIVE.md (GPU setup).
REM ====================================================================
setlocal EnableExtensions
set "ROOT=%~dp0"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"

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

if not exist "%USERPROFILE%\.cache\rtmlib\hub\checkpoints" (
    if exist "%ROOT%\models\rtmlib" (
        echo First run: copying pose models to the user cache...
        xcopy /E /I /Q /Y "%ROOT%\models\rtmlib" "%USERPROFILE%\.cache\rtmlib" >nul
    )
)

set "PYTHONPATH=%ROOT%"
"%ROOT%\venv\Scripts\python.exe" -m poseassess.core.realtime %*
set "RC=%ERRORLEVEL%"
echo.
if not "%RC%"=="0" echo [Live benchmark exited with code %RC% - see the messages above.]
pause
endlocal & exit /b %RC%
