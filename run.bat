@echo off
REM ====================================================================
REM  PoseAssess portable launcher.
REM  Runs the app from this self-contained folder on ANY Windows x64 PC
REM  with NO Python install and NO internet. Double-click to start.
REM ====================================================================
setlocal EnableExtensions
set "ROOT=%~dp0"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"

REM --- 1. Point the bundled venv at the bundled Python (fixes the stdlib
REM        path no matter where this folder was copied to). --------------
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

REM --- 2. Install the 2D pose models into the user cache on first run
REM        (rtmlib looks in %USERPROFILE%\.cache\rtmlib). ----------------
if not exist "%USERPROFILE%\.cache\rtmlib\hub\checkpoints" (
    if exist "%ROOT%\models\rtmlib" (
        echo First run: copying pose models to the user cache...
        xcopy /E /I /Q /Y "%ROOT%\models\rtmlib" "%USERPROFILE%\.cache\rtmlib" >nul
    )
)

REM --- 3. Launch the GUI (app source lives in this folder). ------------
cd /d "%ROOT%"
set "PYTHONPATH=%ROOT%"
set "MPLBACKEND=Agg"
echo.
echo   Starting PoseAssess...
echo   (First launch loads ~1.5 GB of libraries. From a USB / external
echo    drive this can take 1-2 minutes - please wait. Copy this folder
echo    to the local disk for a much faster start.)
echo.
"%ROOT%\venv\Scripts\python.exe" -m poseassess.gui
if errorlevel 1 (
    echo.
    echo [PoseAssess exited with an error - see the message above.]
    pause
)
endlocal
