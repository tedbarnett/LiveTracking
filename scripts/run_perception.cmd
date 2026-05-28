@echo off
REM Wrapper used by the LiveTrackingPerception scheduled task so we capture stderr/stdout.
cd /d C:\Users\timew\Github\LiveTracking
set LOGDIR=C:\Users\timew\Github\LiveTracking\runtime\service-logs
if not exist "%LOGDIR%" mkdir "%LOGDIR%"
"C:\Users\timew\Github\LiveTracking\.venv\Scripts\python.exe" -u -m livetracking.daemon.perception >> "%LOGDIR%\perception.stdout.log" 2>> "%LOGDIR%\perception.stderr.log"
