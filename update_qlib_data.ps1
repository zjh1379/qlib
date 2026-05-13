# Daily CSI300 data refresh via baostock + dump_bin (incremental).
# Replaces the old full-fetch path; only the missing days are fetched and dump_update appended.
# Schedule with Task Scheduler at ~17:30 weekdays.
# Manual test: powershell -ExecutionPolicy Bypass -File update_qlib_data.ps1

$ErrorActionPreference = 'Continue'
$py       = 'F:\Tools\Anaconda\envs\qlib\python.exe'
$repo     = 'E:\Projects\qlib\.claude\worktrees\eager-morse-4be0a4'
$logdir   = "$repo\logs"

if (-not (Test-Path $logdir)) { New-Item -ItemType Directory $logdir | Out-Null }
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$log = "$logdir\daily_update_bs_$stamp.log"

function LogStep($msg) {
    $line = "[{0}] {1}" -f (Get-Date -Format 'HH:mm:ss'), $msg
    Write-Output $line
    Add-Content -Path $log -Value $line
}

LogStep "Daily incremental update started (baostock + dump_update)"

# Single incremental pass: per-stock incremental fetch, dump_update, benchmark, csi300.txt.
& $py "$repo\production\incremental_refresh.py" *>&1 | Tee-Object -FilePath $log -Append

LogStep "Done. Calendar end:"
$cal_file = "$env:USERPROFILE\.qlib\qlib_data\cn_data_bs\calendars\day.txt"
if (Test-Path $cal_file) {
    $cal = Get-Content $cal_file -Tail 1
    LogStep "  $cal"
} else {
    LogStep "  (calendar file missing)"
}
