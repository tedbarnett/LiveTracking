@echo off
REM Wrapper used by the LiveTrackingPerception scheduled task so we capture stderr/stdout.
REM Repo root = parent of this script's directory (no hardcoded user paths).
set REPO=%~dp0..
for %%I in ("%REPO%") do set REPO=%%~fI
cd /d "%REPO%"
set LOGDIR=%REPO%\runtime\service-logs
if not exist "%LOGDIR%" mkdir "%LOGDIR%"
"%REPO%\.venv\Scripts\python.exe" -u -m livetracking.daemon.perception >> "%LOGDIR%\perception.stdout.log" 2>> "%LOGDIR%\perception.stderr.log"
