# LiveTracking — Windows service / scheduled-task installer
#
# Run as Administrator from any shell. Idempotent.
#
# Installs / reconfigures three pieces:
#   - LiveTrackingFlameWeb       NSSM service, LocalSystem, AUTO_START
#                                Flask UI on :5070 (tunneled to barnettlabs.tech).
#   - LiveTrackingPerception     Scheduled task, runs at user logon, runs as user.
#                                Holds the RealSense D455 + GPU pipeline.
#   - LiveTrackingProjector      Scheduled task, runs at user logon, runs as user.
#                                Holds pygame fullscreen on the JMGO (display 1).
#
# Perception + Projector are user-session tasks (NOT LocalSystem services) because:
#   * Projector needs to grab a fullscreen pygame surface on display 1 — only
#     possible inside an active user desktop, not session 0.
#   * Perception's CUDA + USB camera access also benefits from a user session.
#
# To uninstall everything: -Uninstall
param(
    [switch]$Uninstall = $false
)

$ErrorActionPreference = 'Stop'

$RepoRoot = "C:\Users\timew\Github\LiveTracking"
$Venv     = "$RepoRoot\.venv\Scripts\python.exe"
$LogDir   = "$RepoRoot\runtime\service-logs"
$User     = "$env:USERDOMAIN\$env:USERNAME"
$Nssm     = (Get-Command nssm.exe -ErrorAction Stop).Source

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

function Remove-IfPresent {
    param([string]$Name, [string]$Kind)
    if ($Kind -eq 'service') {
        $svc = Get-Service -Name $Name -ErrorAction SilentlyContinue
        if ($svc) {
            Write-Host "stopping + removing service $Name"
            & sc.exe stop $Name | Out-Null
            Start-Sleep -Seconds 2
            & $Nssm remove $Name confirm | Out-Null
        }
    } elseif ($Kind -eq 'task') {
        $task = Get-ScheduledTask -TaskName $Name -ErrorAction SilentlyContinue
        if ($task) {
            Write-Host "removing scheduled task $Name"
            Unregister-ScheduledTask -TaskName $Name -Confirm:$false
        }
    }
}

if ($Uninstall) {
    Remove-IfPresent -Name "LiveTrackingFlameWeb"   -Kind 'service'
    Remove-IfPresent -Name "LiveTrackingPerception" -Kind 'task'
    Remove-IfPresent -Name "LiveTrackingProjector"  -Kind 'task'
    Remove-IfPresent -Name "LiveTrackingCalibrate"  -Kind 'task'
    Write-Host "[done] uninstalled"
    exit 0
}

# ---- 1) LiveTrackingFlameWeb (NSSM service, LocalSystem) -------------------
Remove-IfPresent -Name "LiveTrackingFlameWeb" -Kind 'service'
Write-Host "installing LiveTrackingFlameWeb (NSSM, LocalSystem, AUTO_START)"
& $Nssm install LiveTrackingFlameWeb $Venv | Out-Null
& $Nssm set LiveTrackingFlameWeb AppParameters "-u -m livetracking.daemon.flame_web" | Out-Null
& $Nssm set LiveTrackingFlameWeb AppDirectory $RepoRoot | Out-Null
& $Nssm set LiveTrackingFlameWeb AppExit Default Restart | Out-Null
& $Nssm set LiveTrackingFlameWeb AppStdout "$LogDir\flame_web.stdout.log" | Out-Null
& $Nssm set LiveTrackingFlameWeb AppStderr "$LogDir\flame_web.stderr.log" | Out-Null
& $Nssm set LiveTrackingFlameWeb AppRotateFiles 1 | Out-Null
& $Nssm set LiveTrackingFlameWeb AppRotateBytes 1048576 | Out-Null
& $Nssm set LiveTrackingFlameWeb DisplayName "LiveTracking Flame Web UI" | Out-Null
& $Nssm set LiveTrackingFlameWeb Description "Flask web UI for the LiveTracking perception pipeline. Subscribes to perception daemon over ZMQ and serves the hover-to-illuminate interface at livetracking.barnettlabs.tech." | Out-Null
& $Nssm set LiveTrackingFlameWeb Start SERVICE_AUTO_START | Out-Null

# ---- 2) LiveTrackingPerception (Scheduled task at user logon) --------------
Remove-IfPresent -Name "LiveTrackingPerception" -Kind 'task'
Write-Host "installing LiveTrackingPerception (scheduled task, at logon, as $User)"
$pAction = New-ScheduledTaskAction `
    -Execute $Venv `
    -Argument "-u -m livetracking.daemon.perception" `
    -WorkingDirectory $RepoRoot
$pTrigger = New-ScheduledTaskTrigger -AtLogOn -User $User
$pSettings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -ExecutionTimeLimit ([TimeSpan]::Zero) `
    -RestartCount 99 -RestartInterval (New-TimeSpan -Minutes 1)
$pPrincipal = New-ScheduledTaskPrincipal -UserId $User -LogonType Interactive -RunLevel Highest
$pTask = New-ScheduledTask `
    -Action $pAction -Trigger $pTrigger -Settings $pSettings -Principal $pPrincipal `
    -Description "LiveTracking perception daemon: D455 capture + Stage 1 depth + DINO + SAM 2 on RTX 5090."
Register-ScheduledTask -TaskName "LiveTrackingPerception" -InputObject $pTask -Force | Out-Null

# ---- 3) LiveTrackingProjector (Scheduled task at user logon) ---------------
Remove-IfPresent -Name "LiveTrackingProjector" -Kind 'task'
Write-Host "installing LiveTrackingProjector (scheduled task, at logon, as $User)"
$jAction = New-ScheduledTaskAction `
    -Execute $Venv `
    -Argument "-u -m livetracking.daemon.projector" `
    -WorkingDirectory $RepoRoot
$jTrigger = New-ScheduledTaskTrigger -AtLogOn -User $User
$jSettings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -ExecutionTimeLimit ([TimeSpan]::Zero) `
    -RestartCount 99 -RestartInterval (New-TimeSpan -Minutes 1)
$jPrincipal = New-ScheduledTaskPrincipal -UserId $User -LogonType Interactive -RunLevel Highest
$jTask = New-ScheduledTask `
    -Action $jAction -Trigger $jTrigger -Settings $jSettings -Principal $jPrincipal `
    -Description "LiveTracking projector daemon: pygame fullscreen on JMGO display 1, ZMQ PULL hover commands."
Register-ScheduledTask -TaskName "LiveTrackingProjector" -InputObject $jTask -Force | Out-Null

# ---- 4) LiveTrackingCalibrate (Scheduled task, on-demand) ------------------
# Manual-trigger only (no logon trigger). The Flask UI invokes
#   schtasks /run /tn LiveTrackingCalibrate
# which fires this task in the user's desktop session so the orchestrator
# can grab the JMGO + RealSense.
Remove-IfPresent -Name "LiveTrackingCalibrate" -Kind 'task'
Write-Host "installing LiveTrackingCalibrate (scheduled task, on-demand, as $User)"
$cAction = New-ScheduledTaskAction `
    -Execute $Venv `
    -Argument "-u $RepoRoot\scripts\run_calibration.py" `
    -WorkingDirectory $RepoRoot
$cSettings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 5)
$cPrincipal = New-ScheduledTaskPrincipal -UserId $User -LogonType Interactive -RunLevel Highest
$cTask = New-ScheduledTask `
    -Action $cAction -Settings $cSettings -Principal $cPrincipal `
    -Description "LiveTracking on-demand calibration: stops perception+projector, re-runs camera->projector homography calibration, restarts them."
Register-ScheduledTask -TaskName "LiveTrackingCalibrate" -InputObject $cTask -Force | Out-Null

# ---- 5) LiveTrackingParallaxCalibrate (Scheduled task, on-demand) ----------
# Manual two-plane parallax alignment. Flask UI invokes
#   schtasks /run /tn LiveTrackingParallaxCalibrate
# which fires this task in the user's desktop session. Operator drives the
# alignment with arrow keys / + - / [ ] on the laptop keyboard while the
# projector shows the live camera feed. ExecutionTimeLimit is generous
# (30 min) because this is a human-paced UI.
Remove-IfPresent -Name "LiveTrackingParallaxCalibrate" -Kind 'task'
Write-Host "installing LiveTrackingParallaxCalibrate (scheduled task, on-demand, as $User)"
$pAction = New-ScheduledTaskAction `
    -Execute $Venv `
    -Argument "-u $RepoRoot\scripts\run_parallax_calibration.py" `
    -WorkingDirectory $RepoRoot
$pSettings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 30)
$pPrincipal = New-ScheduledTaskPrincipal -UserId $User -LogonType Interactive -RunLevel Highest
$pTask = New-ScheduledTask `
    -Action $pAction -Settings $pSettings -Principal $pPrincipal `
    -Description "LiveTracking parallax calibration: stops perception+projector, runs manual two-plane alignment, restarts them."
Register-ScheduledTask -TaskName "LiveTrackingParallaxCalibrate" -InputObject $pTask -Force | Out-Null

Write-Host ""
Write-Host "[done] installed:"
Write-Host "  * LiveTrackingFlameWeb       (service, auto-start on boot)"
Write-Host "  * LiveTrackingPerception     (task, auto-start at user logon)"
Write-Host "  * LiveTrackingProjector      (task, auto-start at user logon)"
Write-Host ""
Write-Host "To start everything now without rebooting:"
Write-Host "  sc.exe start LiveTrackingFlameWeb"
Write-Host "  Start-ScheduledTask -TaskName LiveTrackingPerception"
Write-Host "  Start-ScheduledTask -TaskName LiveTrackingProjector"
Write-Host ""
Write-Host "Logs:"
Write-Host "  $LogDir\flame_web.{stdout,stderr}.log"
Write-Host "  Task Scheduler -> LiveTrackingPerception / LiveTrackingProjector -> History"
