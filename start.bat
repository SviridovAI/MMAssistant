@echo off
chcp 1251 >nul
echo Launching MMAssistant...
echo.

REM Check if virtual environment exists
if not exist "venv\Scripts\activate.bat" (
    echo Virtual environment not found.
    echo Run setup.bat for installation.
    pause
    exit /b 1
)

REM Activate virtual environment
call venv\Scripts\activate

REM Launch application
python gui.py

REM Keep window open after application closes
echo.
echo Application finished.
pause