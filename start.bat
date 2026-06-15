@echo off
REM ============================================================
REM  qlib companion app - one-click launcher (backend + frontend)
REM  Double-click to start both servers and open the browser.
REM  Requires: the qlib conda env python + npm on PATH.
REM  Backend reads backend\.env (provider key etc.) automatically.
REM ============================================================
setlocal
set "ROOT=%~dp0"
set "PYENV=F:\Tools\Anaconda\envs\qlib\python.exe"

if not exist "%PYENV%" (
  echo [ERROR] python not found at %PYENV%
  echo Edit the PYENV line in this script to point to your qlib env python.
  pause
  exit /b 1
)

echo Starting backend  -^> http://127.0.0.1:8000  (window "qlib-backend")
start "qlib-backend" cmd /k "cd /d %ROOT%backend && %PYENV% -m uvicorn app.main:app --host 127.0.0.1 --port 8000"

echo Starting frontend -^> http://localhost:5173   (window "qlib-frontend")
start "qlib-frontend" cmd /k "cd /d %ROOT%frontend && npm run dev"

REM wait a few seconds for the servers to boot, then open the app
timeout /t 6 /nobreak >nul
start "" http://localhost:5173

echo.
echo   App opening at http://localhost:5173
echo   To STOP: close the two windows titled "qlib-backend" and "qlib-frontend".
echo.
endlocal
