@echo off
REM Wrapper used by the LiveTrackingFlameWeb scheduled task so we capture
REM stderr/stdout and run unprivileged (so deploys can restart it without UAC,
REM unlike the old NSSM LocalSystem service).
REM Repo root = parent of this script's directory (no hardcoded user paths).
set REPO=%~dp0..
for %%I in ("%REPO%") do set REPO=%%~fI
cd /d "%REPO%"
set LOGDIR=%REPO%\runtime\service-logs
if not exist "%LOGDIR%" mkdir "%LOGDIR%"
"%REPO%\.venv\Scripts\python.exe" -u -m livetracking.daemon.flame_web >> "%LOGDIR%\flame_web.stdout.log" 2>> "%LOGDIR%\flame_web.stderr.log"
