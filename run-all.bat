@echo off
cd /d "%~dp0"
echo ====================================================
echo Starting Breast Cancer Predictor (Backend + Frontend)
echo ====================================================

start "Breast Cancer API Backend" cmd /k "call breast-cancer-predictor\start-backend.bat"

timeout /t 3 /nobreak >nul

start "Breast Cancer Frontend" cmd /k "call breast-cancer-predictor\start-frontend.bat"

echo.
echo Both servers have been launched in separate windows!
echo Backend:  http://127.0.0.1:8000 (Docs: http://127.0.0.1:8000/docs)
echo Frontend: http://127.0.0.1:5173
echo.
pause
