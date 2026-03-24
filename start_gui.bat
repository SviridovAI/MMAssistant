@echo off
chcp 1251 >nul

if exist "venv\Scripts\activate.bat" (
    echo Using virtual environment...
    call venv\Scripts\activate
    start /B pythonw gui.py
) else (
    echo Using system Python...
    start /B pythonw gui.py
)

echo GUI started in background (no console window).
echo Application window should appear shortly.