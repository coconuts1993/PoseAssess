@echo off
REM ====================================================================
REM  PoseAssess: download -> unzip -> the lab's INTERNAL package with
REM  OpenPose BODY_25B. Double-click this file. Keep it in the same folder
REM  as get_poseassess_with_openpose.ps1 (it runs that script).
REM  1. finds PoseAssess_Portable_win64.zip in Downloads, or opens the
REM     GitHub download page and waits for the download;
REM  2. unzips it to D:\PoseAssess_Portable (or C:\PoseAssess\...);
REM  3. runs pack_with_openpose.bat: OpenPose (e.g. C:\openpose) is copied
REM     in and ..\PoseAssess_Portable_with_OpenPose.zip is written.
REM  OpenPose license: internal non-commercial use at your site only -
REM  NEVER upload or publish the _with_OpenPose zip.
REM  Options, from a console:
REM     get_poseassess_with_openpose.bat -Dest E:\ -OpenPose D:\tools\openpose
REM     get_poseassess_with_openpose.bat -Zip C:\path\PoseAssess_Portable_win64.zip
REM     get_poseassess_with_openpose.bat -OutZip S:\Lab\PoseAssess
REM     get_poseassess_with_openpose.bat -NoPack
REM ====================================================================
setlocal EnableExtensions
set "PS1=%~dp0get_poseassess_with_openpose.ps1"
if exist "%PS1%" goto run
echo [ERROR] get_poseassess_with_openpose.ps1 must be in the same folder as this file:
echo         "%~dp0"
pause
exit /b 1

:run
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%PS1%" %*
set "RC=%ERRORLEVEL%"
echo.
if not "%RC%"=="0" echo [The script stopped with code %RC% - see the messages above.]
pause
endlocal & exit /b %RC%
