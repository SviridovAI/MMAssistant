@echo off
chcp 1251 >nul
echo Installing MMAssistant...
echo.

echo 1. Creating virtual environment...
python -m venv venv
if errorlevel 1 (
    echo Error: Failed to create virtual environment
    echo Make sure Python is installed and added to PATH
    pause
    exit /b 1
)

echo 2. Activating virtual environment...
call venv\Scripts\activate
if errorlevel 1 (
    echo Error: Failed to activate virtual environment
    pause
    exit /b 1
)

echo 3. Updating pip...
python -m pip install --upgrade pip
if errorlevel 1 (
    echo Warning: Failed to update pip, continuing...
)

echo 4. Installing dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo Error: Failed to install dependencies
    echo Check requirements.txt file and internet connection
    pause
    exit /b 1
)

echo 5. Checking keyring...
python -c "import keyring; print('Keyring installed and working')"
if errorlevel 1 (
    echo Warning: Keyring not working correctly
    echo API keys cannot be stored securely
)

echo.
echo ============================================
echo Installation completed successfully!
echo.
echo IMPORTANT: API keys will be stored in Windows Credential Manager
echo.
echo For first launch:
echo 1. Run start.bat
echo 2. Click 'Settings' button
echo 3. Go to 'LLM' tab
echo 4. Enter your API key
echo 5. Click 'Save'
echo.
echo To launch application use: start.bat
echo ============================================
pause