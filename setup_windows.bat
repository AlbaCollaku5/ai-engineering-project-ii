@echo off
echo Setting up AI Knowledge Assistant on Windows...

echo.
echo Step 1: Creating virtual environment...
python -m venv venv
if errorlevel 1 (
    echo Failed to create virtual environment. Make sure Python is installed.
    pause
    exit /b 1
)

echo.
echo Step 2: Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo Step 3: Upgrading pip...
python -m pip install --upgrade pip

echo.
echo Step 4: Installing dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo Failed to install with full requirements. Trying minimal...
    pip install -r requirements_minimal.txt
    if errorlevel 1 (
        echo Failed to install dependencies. Please check your Python installation.
        pause
        exit /b 1
    )
)

echo.
echo Step 5: Testing imports...
python test_imports_simple.py
if errorlevel 1 (
    echo Import test failed. Some dependencies may be missing.
    echo You can still try running the application.
)

echo.
echo Setup complete! You can now run:
echo   run_windows.bat
echo.
pause
