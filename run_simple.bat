@echo off
chcp 1251 >nul
echo Launching MMAssistant (simplified version)...
echo.

REM Check Python availability
python --version >nul 2>&1
if errorlevel 1 (
    echo Python not found. Install Python 3.8+ and add to PATH.
    pause
    exit /b 1
)

REM Check minimal dependencies
echo Checking minimal dependencies...
python -c "import tkinter; import requests; import keyring; print('Minimal dependencies available')" >nul 2>&1
if errorlevel 1 (
    echo Installing minimal dependencies...
    pip install requests keyring --quiet
)

REM Launch application
echo Launching application...
python gui.py

echo.
echo Application finished.
pause