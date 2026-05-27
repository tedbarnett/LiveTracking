@echo off
REM Install the LiveTracking flame web UI as a Windows service via NSSM.
REM Idempotent: re-running updates the parameters.

set SVC=LiveTrackingFlameWeb
set REPO=C:\Users\timew\Github\LiveTracking
set PY=%REPO%\.venv\Scripts\python.exe
set SCRIPT=%REPO%\src\livetracking\daemon\flame_web.py
set LOGDIR=%REPO%\runtime\service-logs

if not exist "%LOGDIR%" mkdir "%LOGDIR%"

nssm stop %SVC% >nul 2>&1
nssm remove %SVC% confirm >nul 2>&1

nssm install %SVC% "%PY%" "%SCRIPT%"
if errorlevel 1 (
  echo nssm install failed
  exit /b 1
)

nssm set %SVC% AppDirectory "%REPO%"
nssm set %SVC% AppStdout "%LOGDIR%\flame_web.stdout.log"
nssm set %SVC% AppStderr "%LOGDIR%\flame_web.stderr.log"
nssm set %SVC% AppRotateFiles 1
nssm set %SVC% AppRotateBytes 1048576
nssm set %SVC% Start SERVICE_AUTO_START
nssm set %SVC% DisplayName "LiveTracking Flame Web UI"
nssm set %SVC% Description "Flask single-button UI for projecting blue flame on the guitar."

nssm start %SVC%

echo ---
sc query %SVC%
