@echo off
title X-Ray Sentry Launcher
echo ===================================================
echo     X-Ray Sentry: Dual-Model Security Scanner
echo ===================================================
echo.

echo [1/3] Booting FastAPI Backend (PyTorch + OpenCV)...
:: Activates the root venv first, THEN moves into backend and starts uvicorn
start "X-Ray Sentry Backend" cmd /k "call venv\Scripts\activate && cd backend && python -m uvicorn main:app --reload"

echo [2/3] Booting Vite/React Frontend...
start "X-Ray Sentry Frontend" cmd /k "cd frontend && npm run dev"

echo [3/3] Waiting for servers to spin up...
timeout /t 4 /nobreak > nul

echo Launching application in default browser...
start http://localhost:5173

echo.
echo System is live! You can close this window.
pause