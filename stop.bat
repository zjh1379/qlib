@echo off
REM ============================================================
REM  qlib companion app - one-click STOP (backend + frontend)
REM  Kills the windows started by start.bat (and anything still
REM  listening on the app ports, as a fallback).
REM ============================================================
echo Stopping backend / frontend ...

REM 1) Close the titled windows + their child servers (process tree).
taskkill /FI "WINDOWTITLE eq qlib-backend*"  /T /F >nul 2>&1
taskkill /FI "WINDOWTITLE eq qlib-frontend*" /T /F >nul 2>&1

REM 2) Fallback: kill whatever is still listening on :8000 / :5173.
powershell -NoProfile -Command "Get-NetTCPConnection -LocalPort 8000,5173 -State Listen -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess -Unique | ForEach-Object { Stop-Process -Id $_ -Force -ErrorAction SilentlyContinue }"

echo Done. Backend (:8000) and frontend (:5173) stopped.
timeout /t 2 /nobreak >nul
